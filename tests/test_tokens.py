"""Тесты подсчёта токенов, статистики сессии и стоимости.

Проверяют:
- эвристический счётчик и (при наличии пакета) tiktoken-счётчик;
- fallback на эвристику, если tiktoken недоступен;
- точный учёт из usage API и оценочный — при отсутствии usage;
- накопление статистики сессии, отсутствие изменений при ошибке API;
- расчёт стоимости по ценам (общий и per-model приоритет);
- сохранение и восстановление статистики из SQLite;
- обратную совместимость БД без ключа stats.

Сеть не используется: HTTP-вызов подменяется через unittest.mock.
Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path
from unittest import mock

from src import tokens as tokens_module
from src.agent import DeepSeekAgent
from src.storage import ContextStore
from src.tokens import (
    CHAT_MESSAGE_OVERHEAD,
    CHAT_PRIMING_OVERHEAD,
    HeuristicCounter,
    create_counter,
)
from src.utils import ApiError

_TESTS_TMP = Path(__file__).resolve().parent / "tmp_tokens"


def _make_db() -> Path:
    """Уникальный путь к БД внутри tests/tmp_tokens (внутри репозитория)."""
    return _TESTS_TMP / f"tokens_{uuid.uuid4().hex}.db"


def _payload(content: str = "Ответ", prompt: int = 10, completion: int = 5,
             usage: bool = True) -> dict:
    """Формирует ответ _call_api (как _handle_response)."""
    if usage:
        tokens = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "estimated": False,
            "cache_hit_tokens": None,
            "cache_miss_tokens": None,
        }
    else:
        tokens = None
    return {
        "content": content,
        "model": "deepseek-v4-flash",
        "tokens": tokens,
        "finish_reason": "stop",
    }


class HeuristicCounterTest(unittest.TestCase):
    """Базовая проверка эвристического счётчика."""

    def setUp(self) -> None:
        self.counter = HeuristicCounter()

    def test_empty_text_is_zero(self) -> None:
        self.assertEqual(self.counter.count_text(""), 0)
        self.assertEqual(self.counter.count_text("   "), 0)

    def test_positive_and_monotonic(self) -> None:
        short = self.counter.count_text("привет")
        long = self.counter.count_text("привет, как дела? расскажи подробнее")
        self.assertGreater(short, 0)
        self.assertGreater(long, short)

    def test_cyrillic_is_counted(self) -> None:
        self.assertGreater(self.counter.count_text("Здравствуйте"), 0)

    def test_messages_include_chat_overhead(self) -> None:
        messages = [
            {"role": "system", "content": "система"},
            {"role": "user", "content": "вопрос"},
        ]
        expected = (
            CHAT_PRIMING_OVERHEAD
            + len(messages) * CHAT_MESSAGE_OVERHEAD
            + sum(self.counter.count_text(m["content"]) for m in messages)
        )
        self.assertEqual(self.counter.count_messages(messages), expected)
        # Оценка сообщений строго больше простой суммы текстов
        self.assertGreater(
            self.counter.count_messages(messages),
            sum(self.counter.count_text(m["content"]) for m in messages),
        )


class CounterFactoryTest(unittest.TestCase):
    """Выбор движка счётчика и корректный fallback."""

    def test_disabled_uses_heuristic(self) -> None:
        counter = create_counter(model="deepseek-v4-flash", enabled=False)
        self.assertIsInstance(counter, HeuristicCounter)
        self.assertFalse(counter.is_exact)

    def test_fallback_when_tiktoken_unavailable(self) -> None:
        with mock.patch.object(
            tokens_module, "TiktokenCounter", side_effect=ImportError("нет tiktoken")
        ):
            counter = create_counter(model="deepseek-v4-flash", enabled=True)
        self.assertIsInstance(counter, HeuristicCounter)

    def test_tiktoken_counter_if_available(self) -> None:
        try:
            counter = create_counter(model="deepseek-v4-flash", enabled=True)
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"create_counter не должен падать: {exc}")
        if isinstance(counter, HeuristicCounter):
            self.skipTest("tiktoken недоступен в этой среде — проверка пропущена")
        self.assertGreater(counter.count_text("hello world"), 0)
        self.assertTrue(counter.name.startswith("tiktoken:"))


class AgentTokenAccountingTest(unittest.TestCase):
    """Учёт токенов агентом: точный, оценочный, стоимость."""

    def setUp(self) -> None:
        self.db_path = _make_db()

    def tearDown(self) -> None:
        shutil.rmtree(_TESTS_TMP, ignore_errors=True)

    def _make_agent(self, prices=None, counter=None) -> DeepSeekAgent:
        return DeepSeekAgent(
            api_key="k",
            model="deepseek-v4-flash",
            storage=ContextStore(db_path=self.db_path, enabled=True),
            prices=prices,
            token_counter=counter,
        )

    def test_exact_usage_counted(self) -> None:
        agent = self._make_agent(counter=HeuristicCounter())
        with mock.patch.object(agent, "_call_api", return_value=_payload()):
            result = agent.send_message("Вопрос")
        tokens = result["tokens"]
        self.assertFalse(tokens["estimated"])
        self.assertEqual(tokens["prompt_tokens"], 10)
        self.assertEqual(tokens["completion_tokens"], 5)
        self.assertEqual(tokens["total_tokens"], 15)
        self.assertGreater(tokens["context_tokens"], 0)
        self.assertEqual(tokens["session_total_tokens"], 15)
        self.assertIsNone(tokens["cost_usd"])  # цены не заданы
        stats = agent.get_token_stats()
        self.assertEqual(stats["session"]["total_tokens"], 15)
        self.assertEqual(stats["session"]["requests"], 1)
        self.assertEqual(stats["session"]["exact_requests"], 1)
        self.assertFalse(stats["session"]["prices_set"])
        agent.close()

    def test_missing_usage_falls_back_to_estimate(self) -> None:
        counter = HeuristicCounter()
        agent = self._make_agent(counter=counter)
        with mock.patch.object(
            agent, "_call_api", return_value=_payload(usage=False)
        ):
            result = agent.send_message("Вопрос про оценку")
        tokens = result["tokens"]
        self.assertTrue(tokens["estimated"])
        # Оценка ≥ оценки одного текста ответа
        self.assertGreaterEqual(
            tokens["completion_tokens"], counter.count_text("Ответ")
        )
        stats = agent.get_token_stats()
        self.assertEqual(stats["session"]["total_tokens"], 0)
        self.assertGreater(stats["session"]["estimated_total_tokens"], 0)
        self.assertEqual(stats["session"]["exact_requests"], 0)
        self.assertTrue(stats["request"]["estimated"])
        agent.close()

    def test_error_does_not_change_counters(self) -> None:
        agent = self._make_agent(counter=HeuristicCounter())
        with mock.patch.object(
            agent, "_call_api", side_effect=ApiError("Сетевая ошибка")
        ):
            result = agent.send_message("Вопрос")
        self.assertIsNotNone(result["error"])
        self.assertIsNone(result["tokens"])
        stats = agent.get_token_stats()
        self.assertEqual(stats["session"]["total_with_estimates"], 0)
        self.assertEqual(stats["session"]["requests"], 0)
        self.assertIsNone(stats["request"]["tokens"])
        agent.close()

    def test_cost_calculation(self) -> None:
        prices = {"default": {"input": 2.0, "output": 8.0}, "models": {}}
        agent = self._make_agent(prices=prices, counter=HeuristicCounter())
        with mock.patch.object(agent, "_call_api", return_value=_payload()):
            result = agent.send_message("Вопрос")
        # 10/1e6*2 + 5/1e6*8 = 0.00002 + 0.00004 = 0.00006
        self.assertAlmostEqual(result["tokens"]["cost_usd"], 0.00006, places=9)
        self.assertAlmostEqual(
            result["tokens"]["session_cost_usd"], 0.00006, places=9
        )
        self.assertTrue(agent.get_token_stats()["session"]["prices_set"])
        agent.close()

    def test_per_model_price_has_priority(self) -> None:
        prices = {
            "default": {"input": 1.0, "output": 1.0},
            "models": {"deepseek-v4-flash": {"input": 10.0, "output": 20.0}},
        }
        agent = self._make_agent(prices=prices, counter=HeuristicCounter())
        with mock.patch.object(agent, "_call_api", return_value=_payload()):
            result = agent.send_message("Вопрос")
        # 10/1e6*10 + 5/1e6*20 = 0.0001 + 0.0001 = 0.0002
        self.assertAlmostEqual(result["tokens"]["cost_usd"], 0.0002, places=9)
        agent.close()

    def test_mock_tokens_are_marked_estimated(self) -> None:
        agent = DeepSeekAgent(api_key="", mock_mode=True,
                              token_counter=HeuristicCounter())
        result = agent.send_message("Привет")
        self.assertTrue(result["tokens"]["estimated"])
        self.assertGreater(result["tokens"]["total_tokens"], 0)
        agent.close()

    def test_context_tokens_grow_and_reset(self) -> None:
        agent = self._make_agent(counter=HeuristicCounter())
        with mock.patch.object(agent, "_call_api", return_value=_payload()):
            first = agent.send_message("Один")
            after_first = agent.get_token_stats()["context"]["tokens"]
            agent.send_message("Два")
            after_second = agent.get_token_stats()["context"]["tokens"]
        self.assertGreater(after_second, after_first)
        agent.clear_history()
        self.assertLess(
            agent.get_token_stats()["context"]["tokens"], after_first
        )
        self.assertEqual(
            agent.get_token_stats()["session"]["total_with_estimates"], 0
        )
        agent.close()

    def test_count_tokens_public_api(self) -> None:
        agent = self._make_agent(counter=HeuristicCounter())
        self.assertEqual(agent.count_tokens(""), 0)
        self.assertGreater(agent.count_tokens("некоторый текст"), 0)
        self.assertGreater(agent.count_history_tokens(), 0)  # system prompt
        agent.close()


class TokenStatsPersistenceTest(unittest.TestCase):
    """Статистика сессии сохраняется и восстанавливается из SQLite."""

    def setUp(self) -> None:
        self.db_path = _make_db()

    def tearDown(self) -> None:
        shutil.rmtree(_TESTS_TMP, ignore_errors=True)

    def test_stats_restored_after_restart(self) -> None:
        prices = {"default": {"input": 1.0, "output": 1.0}, "models": {}}
        agent1 = DeepSeekAgent(
            api_key="k1",
            model="deepseek-v4-flash",
            prices=prices,
            token_counter=HeuristicCounter(),
            storage=ContextStore(db_path=self.db_path, enabled=True),
        )
        with mock.patch.object(agent1, "_call_api", return_value=_payload()):
            agent1.send_message("Вопрос 1")
            agent1.send_message("Вопрос 2")
        agent1.close()

        agent2 = DeepSeekAgent(
            api_key="k2",
            model="deepseek-v4-flash",
            prices=prices,
            token_counter=HeuristicCounter(),
            storage=ContextStore(db_path=self.db_path, enabled=True),
        )
        stats = agent2.get_token_stats()
        self.assertEqual(stats["session"]["total_tokens"], 30)  # 15 + 15
        self.assertEqual(stats["session"]["requests"], 2)
        self.assertEqual(stats["session"]["exact_requests"], 2)
        self.assertAlmostEqual(
            stats["session"]["cost_usd"],
            (10 + 10) / 1e6 * 1.0 + (5 + 5) / 1e6 * 1.0,
            places=9,
        )
        # История тоже восстановлена
        self.assertEqual(len(agent2.history), 5)
        agent2.close()

    def test_clear_history_resets_persisted_stats(self) -> None:
        agent1 = DeepSeekAgent(
            api_key="k",
            storage=ContextStore(db_path=self.db_path, enabled=True),
            token_counter=HeuristicCounter(),
        )
        with mock.patch.object(agent1, "_call_api", return_value=_payload()):
            agent1.send_message("Вопрос")
        agent1.clear_history()
        agent1.close()

        agent2 = DeepSeekAgent(
            api_key="k2",
            storage=ContextStore(db_path=self.db_path, enabled=True),
            token_counter=HeuristicCounter(),
        )
        self.assertEqual(
            agent2.get_token_stats()["session"]["total_with_estimates"], 0
        )
        self.assertEqual(len(agent2.history), 1)
        agent2.close()

    def test_old_db_without_stats_is_compatible(self) -> None:
        """БД предыдущей версии (без ключа stats) читается корректно."""
        store = ContextStore(db_path=self.db_path, enabled=True)
        store.save_state(  # stats не передан — как в старой версии
            messages=[
                {"role": "system", "content": "система"},
                {"role": "user", "content": "вопрос"},
                {"role": "assistant", "content": "ответ"},
            ],
            params={"model": "deepseek-v4-flash",
                    "system_prompt": "система",
                    "temperature": 0.7,
                    "max_tokens": 100},
        )
        store.close()

        agent = DeepSeekAgent(
            api_key="k",
            storage=ContextStore(db_path=self.db_path, enabled=True),
            token_counter=HeuristicCounter(),
        )
        self.assertEqual(len(agent.history), 3)
        stats = agent.get_token_stats()
        self.assertEqual(stats["session"]["total_with_estimates"], 0)
        self.assertEqual(stats["session"]["requests"], 0)
        agent.close()

    def test_corrupt_stats_do_not_break_startup(self) -> None:
        store = ContextStore(db_path=self.db_path, enabled=True)
        store.save_state(
            messages=[{"role": "system", "content": "система"}],
            params={"model": "deepseek-v4-flash", "system_prompt": "система"},
            stats={"total_tokens": "не число", "unknown_key": 42},
        )
        store.close()

        agent = DeepSeekAgent(
            api_key="k",
            storage=ContextStore(db_path=self.db_path, enabled=True),
            token_counter=HeuristicCounter(),
        )
        stats = agent.get_token_stats()
        self.assertEqual(stats["session"]["total_tokens"], 0)  # мусор отброшен
        self.assertNotIn("unknown_key", stats["session"])
        agent.close()


if __name__ == "__main__":
    unittest.main()
