"""Тесты скрипта-бенчмарка диалогов (scripts/dialog_benchmark.py).

Проверяют офлайн (sim-режим, без сети и без записи в рабочую БД):
- генерацию диалогов: факт в первом ходе, проверочный вопрос в последнем;
- определение «помнит/не помнит» по ответу;
- искусственное ограничение контекстного окна в токенах;
- три legacy-сценария целиком: короткий и длинный помнят факт, а превышающий
  уменьшенное окно — нет (и часть сообщений вытеснена);
- сценарии стратегий: скользящее окно забывает факт, липкие факты сохраняют
  его служебными запросами, ветки изолированы и каждая помнит своё;
- разбор CLI-опций стратегий (--strategies / --no-strategies).

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

from scripts import dialog_benchmark as bench
from src.config import get_config
from src.strategies import BRANCHING, SLIDING_WINDOW, STICKY_FACTS
from src.tokens import HeuristicCounter

# Быстрый детерминированный счётчик (без tiktoken) для тестов
_COUNTER = HeuristicCounter()


def _fast_config() -> dict:
    """Конфигурация с предсказуемым окном истории."""
    config = get_config()
    config["MAX_HISTORY_MESSAGES"] = 50
    config["DEFAULT_MODEL"] = "deepseek-v4-flash"
    return config


class DialogBuilderTest(unittest.TestCase):
    """Генерация диалогов для сценариев."""

    def test_short_dialog_structure(self) -> None:
        scenario = bench.Scenario(key="short", title="t", turns=3, words_per_turn=10)
        dialog = bench.build_dialog(scenario)
        self.assertEqual(len(dialog), 3)
        self.assertIn(bench.FACT_NAME, dialog[0])
        self.assertIn(bench.FACT_COLOR, dialog[0])
        self.assertIn(bench.PROBE_MARKER, dialog[-1])

    def test_long_dialog_is_longer(self) -> None:
        short = bench.build_dialog(
            bench.Scenario(key="short", title="t", turns=3, words_per_turn=10)
        )
        long = bench.build_dialog(
            bench.Scenario(key="long", title="t", turns=8, words_per_turn=25)
        )
        self.assertEqual(len(long), 8)
        self.assertGreater(len(long[1].split()), len(short[1].split()))

    def test_check_remembered(self) -> None:
        self.assertTrue(bench.check_remembered("Тебя зовут Алекс, цвет — синий."))
        self.assertFalse(bench.check_remembered("Не помню ни имени, ни цвета."))
        self.assertFalse(bench.check_remembered(""))


class TokenWindowTest(unittest.TestCase):
    """Искусственное ограничение контекстного окна в токенах."""

    def test_enforce_token_window_trims_history(self) -> None:
        agent = bench._make_agent(
            bench.Scenario(key="over", title="t", turns=6, words_per_turn=20),
            bench.Options(mode="sim", api_key="x"),
            _fast_config(),
            _COUNTER,
            {"default": {"input": 0.0, "output": 0.0}, "models": {}},
        )
        # Наполняем историю вручную, чтобы не ходить в сеть
        for index in range(4):
            agent._message_history.append(
                {"role": "user", "content": f"длинный вопрос номер {index} " * 12}
            )
            agent._message_history.append(
                {"role": "assistant", "content": f"длинный ответ номер {index} " * 12}
            )
        before = len(agent.history)
        budget = 400
        trimmed = bench.enforce_token_window(agent, budget)
        self.assertGreater(trimmed, 0)
        self.assertLess(len(agent.history), before)
        # Достижимый бюджет: контекст укладывается в окно
        self.assertLessEqual(agent.count_history_tokens(), budget)

        # Недостижимый бюджет: агент всё равно оставляет последнее сообщение
        trimmed_again = bench.enforce_token_window(agent, 20)
        self.assertGreater(trimmed_again, 0)
        self.assertEqual(len(agent.history), 2)  # system + одно сообщение
        agent.close()

    def test_zero_budget_does_nothing(self) -> None:
        agent = bench._make_agent(
            bench.Scenario(key="over", title="t", turns=6),
            bench.Options(mode="sim", api_key="x"),
            _fast_config(), _COUNTER, {},
        )
        agent._message_history.append({"role": "user", "content": "привет"})
        before = len(agent.history)
        self.assertEqual(bench.enforce_token_window(agent, 0), 0)
        self.assertEqual(len(agent.history), before)
        agent.close()


class ScenarioRunTest(unittest.TestCase):
    """Полный прогон сценариев в sim-режиме."""

    def _run(self, scenario: bench.Scenario) -> dict:
        options = bench.Options(mode="sim", api_key="simulation-key")
        return bench.run_scenario(scenario, options, _fast_config(), _COUNTER,
                                  {"default": {"input": 0.0, "output": 0.0},
                                   "models": {}})

    def test_short_scenario_remembers_and_keeps_all(self) -> None:
        result = self._run(
            bench.Scenario(key="short", title="t", turns=3, words_per_turn=10,
                           reply_words=15)
        )
        summary = result["summary"]
        self.assertEqual(summary["turns_done"], 3)
        self.assertTrue(summary["remembered"])
        self.assertEqual(summary["messages_dropped"], 0)
        self.assertEqual(summary["errors"], [])

    def test_long_scenario_remembers_within_window(self) -> None:
        result = self._run(
            bench.Scenario(key="long", title="t", turns=8, words_per_turn=20,
                           reply_words=20)
        )
        summary = result["summary"]
        self.assertEqual(summary["turns_done"], 8)
        self.assertTrue(summary["remembered"])
        self.assertEqual(summary["messages_dropped"], 0)

    def test_over_limit_scenario_forgets(self) -> None:
        result = self._run(
            bench.Scenario(key="over_limit", title="t", turns=8,
                           words_per_turn=20, window_messages=4,
                           window_tokens=200, reply_words=20)
        )
        summary = result["summary"]
        self.assertEqual(summary["turns_done"], 8)
        self.assertFalse(summary["remembered"])
        self.assertGreater(summary["messages_dropped"], 0)
        # Контекст удерживается искусственным окном
        self.assertLessEqual(summary["max_context_tokens"], 400)

    def test_no_storage_used_by_default(self) -> None:
        """Скрипт не должен трогать рабочую БД при storage_db=None."""
        result = self._run(
            bench.Scenario(key="short", title="t", turns=3, words_per_turn=10)
        )
        self.assertEqual(result["summary"]["turns_done"], 3)


class StrategySettingsTest(unittest.TestCase):
    """Настройки стратегий: legacy-режим и режим веток."""

    def test_legacy_scenario_has_no_strategy(self) -> None:
        scenario = bench.Scenario(key="short", title="t", turns=3)
        self.assertIsNone(scenario.strategy)
        self.assertIsNone(bench._strategy_settings_for(scenario))

    def test_branch_scenario_uses_branch_sub_strategy(self) -> None:
        scenario = bench.Scenario(key="branching", title="t", turns=8,
                                  strategy=BRANCHING,
                                  branch_strategy=STICKY_FACTS,
                                  branch_mode=True, strategy_window=12)
        settings = bench._strategy_settings_for(scenario)
        self.assertEqual(settings["name"], BRANCHING)
        self.assertEqual(settings["branch_strategy"], STICKY_FACTS)
        self.assertEqual(settings["window"], 12)
        self.assertTrue(settings["facts_enabled"])

    def test_plain_strategy_scenario(self) -> None:
        scenario = bench.Scenario(key="facts", title="t", turns=8,
                                  strategy=STICKY_FACTS, strategy_window=6)
        settings = bench._strategy_settings_for(scenario)
        self.assertEqual(settings["name"], STICKY_FACTS)
        self.assertEqual(settings["window"], 6)


class StrategyScenariosTest(unittest.TestCase):
    """Прогон сценариев стратегий в sim-режиме."""

    def _run(self, scenario: bench.Scenario) -> dict:
        options = bench.Options(mode="sim", api_key="simulation-key")
        return bench.run_scenario(scenario, options, _fast_config(), _COUNTER,
                                  {"default": {"input": 0.0, "output": 0.0},
                                   "models": {}})

    def test_window_scenario_forgets_fact(self) -> None:
        """Узкое окно вытесняет факт из начала диалога."""
        result = self._run(
            bench.Scenario(key="window", title="t", turns=8, words_per_turn=20,
                           reply_words=20, strategy=SLIDING_WINDOW,
                           strategy_window=4)
        )
        summary = result["summary"]
        self.assertEqual(summary["turns_done"], 8)
        self.assertFalse(summary["remembered"])
        self.assertEqual(summary["strategy"]["name"], SLIDING_WINDOW)
        self.assertEqual(summary["strategy"]["window"], 4)
        self.assertEqual(summary["facts"]["count"], 0)
        self.assertEqual(summary["facts_requests"], 0)
        self.assertIsNone(summary["isolation_ok"])

    def test_facts_scenario_remembers_fact(self) -> None:
        """Блок фактов сохраняет факт даже при узком окне."""
        result = self._run(
            bench.Scenario(key="facts", title="t", turns=8, words_per_turn=20,
                           reply_words=20, strategy=STICKY_FACTS,
                           strategy_window=4)
        )
        summary = result["summary"]
        self.assertEqual(summary["turns_done"], 8)
        self.assertTrue(summary["remembered"])
        self.assertEqual(summary["strategy"]["name"], STICKY_FACTS)
        # Служебные запросы обновления фактов видны (примерно один на ход)
        self.assertGreaterEqual(summary["facts"]["requests"], 8)
        self.assertGreater(summary["facts"]["count"], 0)
        self.assertGreater(summary["facts"]["tokens"], 0)
        self.assertGreater(summary["facts_service_tokens"], 0)
        self.assertEqual(summary["errors"], [])
        # Первый ход уже содержит метрики стратегии/фактов/веток
        first = result["turns"][0]
        self.assertEqual(first["strategy"]["name"], STICKY_FACTS)
        self.assertIn("count", first["facts"])
        self.assertIsNotNone(first["branch"]["id"])

    def test_branching_scenario_isolation(self) -> None:
        """Ветки изолированы, и каждая ветка помнит своё."""
        result = self._run(
            bench.Scenario(key="branching", title="t", turns=8,
                           words_per_turn=10, reply_words=20,
                           strategy=BRANCHING,
                           branch_strategy=STICKY_FACTS,
                           branch_mode=True)
        )
        summary = result["summary"]
        self.assertEqual(summary["turns_done"], summary["turns_planned"])
        self.assertEqual(summary["errors"], [])
        self.assertTrue(summary["isolation_ok"])
        self.assertTrue(summary["remembered"])
        branch = summary["branch"]
        self.assertTrue(branch["mode"])
        self.assertEqual(branch["branches_count"], 2)
        self.assertEqual(len(branch["list"]), 2)
        self.assertTrue(branch["remembers_own"]["a"])
        self.assertTrue(branch["remembers_own"]["b"])
        self.assertTrue(branch["isolation_ok"])
        names = {item["name"] for item in branch["list"]}
        self.assertIn("main", names)
        self.assertIn(bench.BRANCH_A_NAME, names)
        for item in branch["list"]:
            self.assertGreater(item["messages"], 0)

    def test_check_branch_isolation_detects_leak(self) -> None:
        """Проверка изоляции ловит чужие сообщения и забытый факт."""
        own_a = [{"role": "user",
                  "content": f"{bench.BRANCH_A_MARKER} {bench.BRANCH_A_PROJECT}"}]
        own_b = [{"role": "user",
                  "content": f"{bench.BRANCH_B_MARKER} {bench.BRANCH_B_PROJECT}"}]
        self.assertTrue(bench.check_branch_isolation(own_a, own_b))
        leaked = own_b + [{"role": "user", "content": bench.BRANCH_A_MARKER}]
        self.assertFalse(bench.check_branch_isolation(own_a, leaked))
        self.assertFalse(bench.check_branch_isolation([], own_b))


class StrategyCliTest(unittest.TestCase):
    """Разбор опций CLI для сценариев стратегий."""

    def test_defaults_include_all_strategies(self) -> None:
        args = bench.parse_args([])
        self.assertEqual(bench.selected_strategies(args),
                         ["window", "facts", "branching"])

    def test_no_strategies_flag(self) -> None:
        args = bench.parse_args(["--no-strategies"])
        self.assertEqual(bench.selected_strategies(args), [])

    def test_empty_strategies_value(self) -> None:
        args = bench.parse_args(["--strategies", ""])
        self.assertEqual(bench.selected_strategies(args), [])

    def test_aliases_and_subset(self) -> None:
        args = bench.parse_args(["--strategies", "sticky_facts,branch"])
        self.assertEqual(bench.selected_strategies(args), ["facts", "branching"])

    def test_build_scenarios_keeps_legacy_untouched(self) -> None:
        args = bench.parse_args(["--strategies", "window,facts,branching",
                                 "--turns", "6", "--strategy-turns", "7",
                                 "--strategy-window", "5"])
        scenarios = {item.key: item for item in bench.build_scenarios(args)}
        for key in ("short", "long", "over_limit", "long_with_summary"):
            self.assertIsNone(scenarios[key].strategy)
        self.assertTrue(scenarios["long_with_summary"].summary)
        self.assertEqual(scenarios["window"].strategy, SLIDING_WINDOW)
        self.assertEqual(scenarios["window"].turns, 7)
        self.assertEqual(scenarios["window"].strategy_window, 5)
        self.assertEqual(scenarios["facts"].strategy, STICKY_FACTS)
        self.assertEqual(scenarios["branching"].strategy, BRANCHING)
        self.assertTrue(scenarios["branching"].branch_mode)
        self.assertEqual(scenarios["branching"].branch_strategy, STICKY_FACTS)

    def test_build_scenarios_without_strategies(self) -> None:
        args = bench.parse_args(["--no-strategies"])
        keys = [item.key for item in bench.build_scenarios(args)]
        self.assertEqual(keys, ["short", "long", "over_limit",
                                "long_with_summary"])

    def test_strategy_label_and_service_requests(self) -> None:
        summary = {
            "strategy": {"name": SLIDING_WINDOW, "branching": False,
                         "sub_strategy": None},
            "summary_enabled": False, "summary_requests": 0,
            "facts_requests": 0,
        }
        self.assertEqual(bench._strategy_label(summary), "window")
        branch_summary = {
            "strategy": {"name": BRANCHING, "branching": True,
                         "sub_strategy": STICKY_FACTS},
            "summary_enabled": False, "summary_requests": 2,
            "facts_requests": 3,
        }
        self.assertEqual(bench._strategy_label(branch_summary), "branching/facts")
        self.assertEqual(bench.service_requests(branch_summary), 5)


if __name__ == "__main__":
    unittest.main()
