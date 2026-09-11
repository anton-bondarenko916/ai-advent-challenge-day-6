"""Тесты стратегий управления контекстом (без сети, офлайн).

Проверяют:
- реестр и фабрику стратегий (неизвестное имя → понятная ошибка, псевдонимы);
- ``SlidingWindowStrategy``: system первым, ровно N последних сообщений;
- ``StickyFactsStrategy``: порядок «system → факты → окно»;
- переключатель в агенте: смена стратегии, сохранение выбора между запусками,
  начальное извлечение фактов при переходе на sticky_facts;
- sliding window отбрасывает сообщения только в запросе — история остаётся
  в памяти и в БД, а смена стратегии возвращает полный контекст;
- счётчики ``facts_*`` ведутся отдельно от статистики диалога.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from src.agent import DeepSeekAgent
from src.config import get_strategy_settings
from src.facts import FactsStore
from src.storage import ContextStore
from src.strategies import (
    AVAILABLE_STRATEGIES,
    BRANCHING,
    SLIDING_WINDOW,
    STICKY_FACTS,
    SlidingWindowStrategy,
    StickyFactsStrategy,
    available_strategies,
    create_strategy,
    normalize_name,
)
from src.tokens import HeuristicCounter
from src.utils import AgentConfigError
from tests import gui_helpers

_TESTS_TMP = Path(__file__).resolve().parent / "tmp_strategies"
_COUNTER = HeuristicCounter()

_SETTINGS = {
    "name": SLIDING_WINDOW,
    "window": 4,
    "branch_strategy": SLIDING_WINDOW,
    "branch_max": 20,
    "facts_enabled": True,
    "facts_model": "",
    "facts_temperature": 0.0,
    "facts_max_tokens": 300,
    "facts_max_items": 50,
    "facts_prompt": "",
}


def _dialog(count: int) -> list:
    """Список сообщений диалога (без system) длиной count."""
    messages = []
    for index in range(count):
        role = "user" if index % 2 == 0 else "assistant"
        messages.append({"role": role, "content": f"сообщение {index}"})
    return messages


def _chat_payload(text: str = "Ответ") -> dict:
    return {
        "content": text,
        "model": "deepseek-v4-flash",
        "tokens": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
        "finish_reason": "stop",
    }


def _facts_payload(text: str = "{}") -> dict:
    return {
        "content": text,
        "model": "deepseek-v4-flash",
        "tokens": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
        "finish_reason": "stop",
    }


class RegistryTest(unittest.TestCase):
    """Реестр и фабрика стратегий."""

    def test_available_strategies(self) -> None:
        self.assertEqual(list(AVAILABLE_STRATEGIES),
                         ["sliding_window", "sticky_facts", "branching"])
        self.assertEqual(available_strategies(),
                         ["sliding_window", "sticky_facts", "branching"])

    def test_create_sliding_window(self) -> None:
        strategy = create_strategy(SLIDING_WINDOW, {"window": 7})
        self.assertIsInstance(strategy, SlidingWindowStrategy)
        self.assertEqual(strategy.describe()["window"], 7)
        self.assertFalse(strategy.describe()["facts"])

    def test_create_sticky_facts(self) -> None:
        strategy = create_strategy(STICKY_FACTS, {"window": 3})
        self.assertIsInstance(strategy, StickyFactsStrategy)
        self.assertTrue(strategy.describe()["facts"])

    def test_aliases(self) -> None:
        self.assertEqual(normalize_name("window"), SLIDING_WINDOW)
        self.assertEqual(normalize_name("facts"), STICKY_FACTS)
        self.assertEqual(normalize_name("branch"), BRANCHING)
        self.assertIsInstance(create_strategy("facts", {}), StickyFactsStrategy)

    def test_branching_uses_branch_sub_strategy(self) -> None:
        strategy = create_strategy(
            BRANCHING, {"branch_strategy": STICKY_FACTS, "window": 5}
        )
        self.assertIsInstance(strategy, StickyFactsStrategy)
        self.assertEqual(strategy.describe()["window"], 5)

    def test_unknown_strategy_raises(self) -> None:
        with self.assertRaises(AgentConfigError) as ctx:
            create_strategy("магическая")
        message = str(ctx.exception)
        self.assertIn("sliding_window", message)
        self.assertIn("sticky_facts", message)


class SlidingWindowBuildTest(unittest.TestCase):
    """Сборка контекста скользящим окном."""

    def test_keeps_exactly_last_n(self) -> None:
        strategy = SlidingWindowStrategy({"window": 4})
        context = strategy.build_context("Система", _dialog(10))
        self.assertEqual(context[0], {"role": "system", "content": "Система"})
        self.assertEqual(len(context), 5)
        self.assertEqual(context[1]["content"], "сообщение 6")
        self.assertEqual(context[-1]["content"], "сообщение 9")

    def test_short_dialog_is_kept_whole(self) -> None:
        strategy = SlidingWindowStrategy({"window": 20})
        context = strategy.build_context("Система", _dialog(3))
        self.assertEqual(len(context), 4)

    def test_empty_dialog(self) -> None:
        strategy = SlidingWindowStrategy({"window": 5})
        self.assertEqual(strategy.build_context("Система", []),
                         [{"role": "system", "content": "Система"}])

    def test_after_turn_does_nothing(self) -> None:
        strategy = SlidingWindowStrategy({"window": 5})
        result = strategy.after_turn("в", "о", _dialog(2), {}, None)
        self.assertFalse(result["updated"])
        self.assertIsNone(result["error"])


class StickyFactsBuildTest(unittest.TestCase):
    """Сборка контекста стратегией липких фактов."""

    def test_order_system_facts_window(self) -> None:
        store = FactsStore(counter=_COUNTER)
        store.apply({"preferences": {"имя": "Алекс"}})
        strategy = StickyFactsStrategy({"window": 2})
        context = strategy.build_context(
            "Система", _dialog(6), {"facts": store, "counter": _COUNTER}
        )
        self.assertEqual([m["role"] for m in context],
                         ["system", "system", "user", "assistant"])
        self.assertEqual(context[0]["content"], "Система")
        self.assertIn("Известные факты о диалоге:", context[1]["content"])
        self.assertIn("Алекс", context[1]["content"])
        self.assertEqual(context[2]["content"], "сообщение 4")

    def test_empty_facts_no_extra_message(self) -> None:
        strategy = StickyFactsStrategy({"window": 2})
        store = FactsStore(counter=_COUNTER)
        context = strategy.build_context("Система", _dialog(2),
                                         {"facts": store})
        self.assertEqual(len(context), 3)

    def test_after_turn_without_service_uses_local_only(self) -> None:
        strategy = StickyFactsStrategy({"window": 2})
        store = FactsStore(counter=_COUNTER)
        result = strategy.after_turn(
            "Запомни: дедлайн — пятница", "Хорошо", _dialog(2),
            {"facts": store, "counter": _COUNTER}, None,
        )
        self.assertTrue(result["updated"])
        self.assertEqual(store.data["decisions"]["дедлайн"], "пятница")
        self.assertIsNone(result["error"])


class AgentStrategyTest(unittest.TestCase):
    """Переключатель стратегий внутри агента."""

    def setUp(self) -> None:
        self.db_path = _TESTS_TMP / f"strategy_{uuid.uuid4().hex}.db"

    def tearDown(self) -> None:
        shutil.rmtree(_TESTS_TMP, ignore_errors=True)

    def _make_agent(self, settings=None, name=None,
                    storage=None) -> DeepSeekAgent:
        merged = dict(_SETTINGS)
        merged.update(settings or {})
        strategy_name = name or merged["name"]
        return DeepSeekAgent(
            api_key="k",
            model="deepseek-v4-flash",
            system_prompt="Система.",
            token_counter=_COUNTER,
            storage=storage,
            strategy=strategy_name,
            strategy_settings=merged,
        )

    def _patch(self, agent: DeepSeekAgent, facts: str = "{}") -> None:
        agent._call_api = lambda: _chat_payload()
        agent._call_service_api = lambda request: _facts_payload(facts)

    def test_available_strategies_on_agent(self) -> None:
        agent = self._make_agent()
        self.assertEqual(agent.available_strategies(),
                         ["sliding_window", "sticky_facts", "branching"])
        agent.close()

    def test_sliding_window_sends_only_window(self) -> None:
        agent = self._make_agent()
        self._patch(agent)
        for index in range(10):
            agent.send_message(f"Вопрос {index}")
        # История в памяти целиком (окно отбрасывает только в запросе)
        self.assertEqual(len(agent.history), 21)
        context = agent._build_context_messages()
        self.assertEqual(len(context), 5)  # system + 4
        stats = agent.get_token_stats()
        self.assertEqual(stats["context"]["messages_raw"], 20)
        self.assertEqual(stats["context"]["messages_sent"], 5)
        self.assertEqual(stats["strategy"]["name"], SLIDING_WINDOW)
        agent.close()

    def test_history_kept_in_db_and_returned_after_switch(self) -> None:
        store = ContextStore(db_path=self.db_path, enabled=True)
        agent = self._make_agent(storage=store)
        self._patch(agent)
        for index in range(10):
            agent.send_message(f"Вопрос {index}")
        self.assertEqual(len(agent.history), 21)
        # В БД сохранена полная история активной ветки
        agent.close()
        state = ContextStore(db_path=self.db_path,
                             enabled=True).load_state()
        self.assertEqual(len(state["messages"]), 21)

        # Переключение на sticky_facts возвращает полный контекст в память
        agent2 = self._make_agent(
            settings={"name": STICKY_FACTS},
            storage=ContextStore(db_path=self.db_path, enabled=True),
        )
        self._patch(agent2)
        self.assertEqual(len(agent2.history), 21)
        agent2.set_strategy(SLIDING_WINDOW)
        self.assertEqual(len(agent2.history), 21)
        agent2.close()

    def test_set_strategy_changes_context(self) -> None:
        agent = self._make_agent()
        self._patch(agent, '{"preferences": {"имя": "Алекс"}}')
        for index in range(4):
            agent.send_message(f"Вопрос {index}")
        before = agent._build_context_messages()
        self.assertEqual(sum(1 for m in before if m["role"] == "system"), 1)

        agent.set_strategy(STICKY_FACTS)
        after = agent._build_context_messages()
        system_messages = [m for m in after if m["role"] == "system"]
        self.assertEqual(len(system_messages), 2)
        self.assertIn("Известные факты", system_messages[1]["content"])
        self.assertEqual(agent.get_strategy()["name"], STICKY_FACTS)
        self.assertEqual(agent.get_strategy()["sub_strategy"], STICKY_FACTS)
        agent.close()

    def test_set_strategy_branching_keeps_sub_strategy(self) -> None:
        agent = self._make_agent(settings={"name": STICKY_FACTS},
                                 name=STICKY_FACTS)
        self._patch(agent)
        agent.send_message("Привет")
        agent.set_strategy(BRANCHING)
        state = agent.get_strategy()
        self.assertEqual(state["name"], BRANCHING)
        self.assertTrue(state["branching"])
        self.assertEqual(state["sub_strategy"], STICKY_FACTS)
        # Контекст по-прежнему собирает подстратегия ветки
        self.assertTrue(agent._uses_facts())
        agent.close()

    def test_set_strategy_unknown_raises(self) -> None:
        agent = self._make_agent()
        with self.assertRaises(AgentConfigError):
            agent.set_strategy("нет-такой")
        agent.close()

    def test_initial_facts_extraction_on_switch(self) -> None:
        agent = self._make_agent()
        self._patch(agent, '{"preferences": {"имя": "Алекс"}}')
        for index in range(3):
            agent.send_message(f"Вопрос {index}")
        self.assertEqual(agent.get_facts()["count"], 0)

        agent.set_strategy(STICKY_FACTS)
        # Начальное извлечение фактов по текущему диалогу выполнено
        self.assertGreaterEqual(agent.get_facts()["count"], 1)
        agent.close()

    def test_strategy_choice_survives_restart(self) -> None:
        store = ContextStore(db_path=self.db_path, enabled=True,
                             default_strategy=SLIDING_WINDOW)
        agent = self._make_agent(storage=store)
        self._patch(agent)
        agent.send_message("Привет")
        agent.set_strategy(STICKY_FACTS)
        agent.close()

        restored = self._make_agent(
            storage=ContextStore(db_path=self.db_path, enabled=True,
                                 default_strategy=SLIDING_WINDOW),
        )
        self.assertEqual(restored.get_strategy()["sub_strategy"], STICKY_FACTS)
        self.assertEqual(restored.get_strategy()["name"], STICKY_FACTS)
        restored.close()

    def test_legacy_mode_without_strategy_settings(self) -> None:
        """Без стратегий агент работает как раньше (summary/история)."""
        agent = DeepSeekAgent(
            api_key="k", model="deepseek-v4-flash", token_counter=_COUNTER,
            max_history_messages=4,
        )
        agent._call_api = lambda: _chat_payload()
        for index in range(10):
            agent.send_message(f"Вопрос {index}")
        # Прежнее поведение: «сырая» история ограничена MAX_HISTORY_MESSAGES
        self.assertLessEqual(len(agent.history), 5)
        self.assertIsNone(agent.get_strategy()["name"])
        agent.close()

    def test_facts_stats_separate_from_dialog(self) -> None:
        agent = self._make_agent(settings={"name": STICKY_FACTS},
                                 name=STICKY_FACTS)
        self._patch(agent, '{"preferences": {"имя": "Алекс"}}')
        agent.send_message("Первый вопрос")
        agent.send_message("Второй вопрос")
        session = agent.get_token_stats()["session"]
        self.assertEqual(session["requests"], 2)
        self.assertEqual(session["facts_requests"], 2)
        self.assertEqual(session["summary_requests"], 0)
        self.assertGreater(session["facts_completion_tokens"], 0)
        agent.close()

    def test_window_setting_is_applied(self) -> None:
        agent = self._make_agent(settings={"window": 2})
        self._patch(agent)
        for index in range(5):
            agent.send_message(f"Вопрос {index}")
        self.assertEqual(len(agent._build_context_messages()), 3)
        agent.close()

    def test_mock_mode_supports_strategies_without_network(self) -> None:
        settings = get_strategy_settings({"STRATEGY": STICKY_FACTS})
        agent = DeepSeekAgent(
            api_key="", mock_mode=True, token_counter=_COUNTER,
            strategy=STICKY_FACTS, strategy_settings=settings,
        )
        result = agent.send_message("Запомни: дедлайн — пятница")
        self.assertIsNone(result["error"])
        self.assertEqual(agent.get_facts()["facts"]["decisions"]["дедлайн"],
                         "пятница")
        agent.close()


class ConfigStrategySettingsTest(unittest.TestCase):
    """Настройки стратегий из конфигурации."""

    def test_defaults(self) -> None:
        settings = get_strategy_settings({})
        self.assertEqual(settings["name"], SLIDING_WINDOW)
        self.assertEqual(settings["window"], 20)
        self.assertTrue(settings["facts_enabled"])
        self.assertEqual(settings["branch_max"], 20)

    def test_branching_default_sub_strategy(self) -> None:
        settings = get_strategy_settings({"STRATEGY": BRANCHING})
        self.assertEqual(settings["name"], BRANCHING)
        self.assertEqual(settings["branch_strategy"], SLIDING_WINDOW)

    def test_custom_values(self) -> None:
        settings = get_strategy_settings({
            "STRATEGY": "sticky_facts",
            "WINDOW_KEEP_LAST_N": 3,
            "FACTS_MAX_ITEMS": 5,
            "FACTS_MODEL": "deepseek-v4-pro",
        })
        self.assertEqual(settings["name"], STICKY_FACTS)
        self.assertEqual(settings["window"], 3)
        self.assertEqual(settings["facts_max_items"], 5)
        self.assertEqual(settings["facts_model"], "deepseek-v4-pro")


class StrategyComboGuiTest(unittest.TestCase):
    """Переключатель стратегий в интерфейсе (пропускается без дисплея)."""

    root = None
    _ChatApplication = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = gui_helpers.get_root()
        if cls.root is None:
            raise unittest.SkipTest("графическая среда недоступна (tkinter/Tk)")
        from src.ui import ChatApplication

        cls._ChatApplication = ChatApplication

    def setUp(self) -> None:
        gui_helpers.reset_children(self.root)
        settings = get_strategy_settings({
            "STRATEGY": SLIDING_WINDOW, "WINDOW_KEEP_LAST_N": 5,
        })
        self.agent = DeepSeekAgent(
            api_key="", mock_mode=True, token_counter=HeuristicCounter(),
            strategy=SLIDING_WINDOW, strategy_settings=settings,
        )
        self.app = self._ChatApplication(root=self.root, agent=self.agent)
        self.root.update()

    def tearDown(self) -> None:
        for closer in ("_close_context_dialog", "_close_branches_dialog"):
            method = getattr(self.app, closer, None)
            if callable(method):
                try:
                    method()
                except Exception:  # закрытие окна не должно ронять тест
                    pass
        self.agent.close()

    def test_combobox_lists_all_strategies(self) -> None:
        values = tuple(str(value) for value in self.app.strategy_combo["values"])
        self.assertEqual(values,
                         ("sliding_window", "sticky_facts", "branching"))

    def test_selecting_strategy_changes_agent(self) -> None:
        self.app.strategy_combo.set("sticky_facts")
        self.app._on_strategy_change()
        self.root.update()
        self.assertEqual(self.agent.get_strategy()["name"], "sticky_facts")
        self.assertEqual(self.agent.get_strategy()["window"], 5)

    def test_invalid_selection_does_not_break_ui(self) -> None:
        try:
            self.app.strategy_combo.set("нет-такой")
            self.app._on_strategy_change()
            self.root.update()
        except Exception as exc:  # UI обязан сообщить об ошибке, а не упасть
            self.fail(f"интерфейс не должен падать: {exc}")
        self.assertEqual(self.agent.get_strategy()["name"], SLIDING_WINDOW)


if __name__ == "__main__":
    unittest.main()
