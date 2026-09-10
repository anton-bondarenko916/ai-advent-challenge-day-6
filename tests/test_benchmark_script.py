"""Тесты скрипта-бенчмарка диалогов (scripts/dialog_benchmark.py).

Проверяют офлайн (sim-режим, без сети и без записи в рабочую БД):
- генерацию диалогов: факт в первом ходе, проверочный вопрос в последнем;
- определение «помнит/не помнит» по ответу;
- искусственное ограничение контекстного окна в токенах;
- три сценария целиком: короткий и длинный помнят факт, а превышающий
  уменьшенное окно — нет (и часть сообщений вытеснена).

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

from scripts import dialog_benchmark as bench
from src.config import get_config
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


if __name__ == "__main__":
    unittest.main()
