"""Тесты блока фактов (Sticky Facts / Key-Value Memory) без сети.

Проверяют:
- слияние фактов: перезапись, удаление через null, отбрасывание мусора;
- лимиты по числу пар и по токенам (вытесняются самые старые);
- устойчивость к битому JSON (прежние факты сохраняются);
- локальный разбор явных команд («запомни…», «меня зовут…», «мой … — …»);
- обновление фактов после каждого хода через подменённый служебный вызов;
- публичное API агента (remember/forget/clear_facts/update_facts_now);
- персистентность фактов в БД (переживают «перезапуск»).

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from src.agent import DeepSeekAgent
from src.facts import (
    DEFAULT_FACTS_PROMPT,
    FACTS_MARKER,
    FactsStore,
    build_request,
    extract_json_object,
    parse_response,
)
from src.storage import ContextStore
from src.tokens import HeuristicCounter

# Временная папка тестов ВНУТРИ репозитория (песочница не любит системный TEMP)
_TESTS_TMP = Path(__file__).resolve().parent / "tmp_facts"

_COUNTER = HeuristicCounter()

_STRATEGY_SETTINGS = {
    "name": "sticky_facts",
    "window": 3,
    "branch_strategy": "sticky_facts",
    "branch_max": 20,
    "facts_enabled": True,
    "facts_model": "",
    "facts_temperature": 0.0,
    "facts_max_tokens": 300,
    "facts_max_items": 50,
    "facts_prompt": "",
}


def _make_db() -> Path:
    """Уникальный путь к файлу БД внутри tests/tmp_facts."""
    return _TESTS_TMP / f"facts_{uuid.uuid4().hex}.db"


def _chat_payload(text: str = "Ответ") -> dict:
    return {
        "content": text,
        "model": "deepseek-v4-flash",
        "tokens": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
        "finish_reason": "stop",
    }


def _facts_payload(text: str) -> dict:
    return {
        "content": text,
        "model": "deepseek-v4-flash",
        "tokens": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        "finish_reason": "stop",
    }


class MergeTest(unittest.TestCase):
    """Правила слияния фактов (чистая функция)."""

    def test_new_value_overwrites_old(self) -> None:
        merged = FactsStore.merge(
            {"preferences": {"цвет": "синий"}},
            {"preferences": {"цвет": "зелёный"}},
        )
        self.assertEqual(merged["preferences"]["цвет"], "зелёный")

    def test_null_and_empty_remove_key(self) -> None:
        merged = FactsStore.merge(
            {"preferences": {"цвет": "синий", "имя": "Алекс"}},
            {"preferences": {"цвет": None, "имя": ""}},
        )
        self.assertNotIn("preferences", merged)

    def test_unknown_category_is_dropped(self) -> None:
        merged = FactsStore.merge({}, {"hobbies": {"спорт": "бег"}})
        self.assertEqual(merged, {})

    def test_non_string_value_is_dropped(self) -> None:
        merged = FactsStore.merge(
            {}, {"decisions": {"число": 42, "список": ["a"], "флаг": True,
                               "ок": "строка"}}
        )
        self.assertEqual(merged["decisions"], {"ок": "строка"})

    def test_missing_keys_are_preserved(self) -> None:
        merged = FactsStore.merge(
            {"goal": {"цель": "отчёт"}, "preferences": {"имя": "Алекс"}},
            {"goal": {"цель": "письмо"}},
        )
        self.assertEqual(merged["preferences"]["имя"], "Алекс")
        self.assertEqual(merged["goal"]["цель"], "письмо")

    def test_garbage_incoming_does_not_break(self) -> None:
        self.assertEqual(FactsStore.merge({"goal": {"цель": "X"}}, "мусор"),
                         {"goal": {"цель": "X"}})
        self.assertEqual(FactsStore.merge({"goal": {"цель": "X"}}, None),
                         {"goal": {"цель": "X"}})
        self.assertEqual(FactsStore.merge(None, {"goal": {"a": "b"}}),
                         {"goal": {"a": "b"}})


class StoreTest(unittest.TestCase):
    """Хранилище фактов: доступ, лимиты, сериализация."""

    def test_apply_and_text(self) -> None:
        store = FactsStore(counter=_COUNTER)
        stats = store.apply({"goal": {"цель": "написать отчёт"},
                             "preferences": {"имя": "Алекс"}})
        self.assertEqual(stats["added"], 2)
        self.assertEqual(store.count(), 2)
        text = store.to_text()
        self.assertIn("Известные факты о диалоге:", text)
        self.assertIn("[goal] цель: написать отчёт", text)
        self.assertIn("[preferences] имя: Алекс", text)
        self.assertEqual(len(store.to_messages()), 1)
        self.assertGreater(store.token_count(), 0)

    def test_to_messages_empty(self) -> None:
        store = FactsStore(counter=_COUNTER)
        self.assertEqual(store.to_messages(), [])
        self.assertEqual(store.to_text(), "")

    def test_remember_and_forget(self) -> None:
        store = FactsStore(counter=_COUNTER)
        self.assertTrue(store.remember("цель", "отчёт", "goal"))
        self.assertFalse(store.remember("", "x"))
        self.assertFalse(store.remember("k", "v", "нет_такой"))
        self.assertEqual(store.get("цель"), "отчёт")
        self.assertTrue(store.forget("цель"))
        self.assertFalse(store.forget("цель"))
        self.assertTrue(store.is_empty())

    def test_limit_by_items_keeps_newest(self) -> None:
        store = FactsStore(max_items=2, counter=_COUNTER)
        store.apply({"decisions": {"a": "1"}})
        store.apply({"decisions": {"b": "2"}})
        store.apply({"decisions": {"c": "3"}})
        removed = store.limit(counter=_COUNTER, max_items=2, max_tokens=0)
        self.assertEqual(removed, 1)
        self.assertEqual(store.count(), 2)
        self.assertNotIn("a", store.data.get("decisions", {}))
        self.assertIn("c", store.data["decisions"])

    def test_limit_by_tokens(self) -> None:
        store = FactsStore(counter=_COUNTER)
        for index in range(40):
            store.apply({"decisions": {f"ключ{index}": "значение " * 4}})
        before = store.count()
        removed = store.limit(counter=_COUNTER, max_items=1000, max_tokens=40)
        self.assertGreater(removed, 0)
        self.assertLess(store.count(), before)
        self.assertLessEqual(store.token_count(_COUNTER), 40)

    def test_serialization_round_trip(self) -> None:
        store = FactsStore(counter=_COUNTER)
        store.apply({"preferences": {"имя": "Алекс", "цвет": "синий"}})
        raw = store.as_dict()
        restored = FactsStore(counter=_COUNTER)
        restored.from_dict(raw)
        self.assertEqual(restored.as_dict(), raw)
        # Мусор в данных не роняет загрузку
        broken = FactsStore(counter=_COUNTER)
        broken.from_dict({"data": {"unknown": {"k": "v"},
                                   "preferences": {"имя": 5, "ок": "да"}},
                          "updated_at": "мусор"})
        self.assertEqual(broken.data, {"preferences": {"ок": "да"}})


class ExtractLocalTest(unittest.TestCase):
    """Локальный разбор явных команд пользователя."""

    def test_remember_with_colon(self) -> None:
        found = FactsStore.extract_local("Запомни: дедлайн — пятница")
        self.assertEqual(found["decisions"]["дедлайн"], "пятница")

    def test_remember_what(self) -> None:
        found = FactsStore.extract_local("Запомни, что бюджет 10000 рублей")
        self.assertEqual(found["decisions"]["заметка"], "бюджет 10000 рублей")

    def test_polite_remember_is_ignored(self) -> None:
        self.assertEqual(FactsStore.extract_local("Запомни это, пожалуйста."), {})

    def test_name_and_my_patterns(self) -> None:
        found = FactsStore.extract_local(
            "Меня зовут Алекс, мой любимый цвет — синий. Я живу в Казани."
        )
        self.assertEqual(found["preferences"]["имя"], "Алекс")
        self.assertEqual(found["preferences"]["любимый_цвет"], "синий")
        self.assertEqual(found["preferences"]["город"], "Казани")

    def test_empty_text(self) -> None:
        self.assertEqual(FactsStore.extract_local(""), {})
        self.assertEqual(FactsStore.extract_local(None), {})


class ResponseParsingTest(unittest.TestCase):
    """Разбор ответа модели с фактами."""

    def test_plain_json(self) -> None:
        parsed, error = parse_response('{"goal": {"цель": "отчёт"}}')
        self.assertIsNone(error)
        self.assertEqual(parsed["goal"]["цель"], "отчёт")

    def test_json_in_code_fence_with_prose(self) -> None:
        text = ('Вот результат:\n```json\n{"preferences": {"имя": "Алекс"}}\n```\n'
                'Готово.')
        parsed, error = parse_response(text)
        self.assertIsNone(error)
        self.assertEqual(parsed["preferences"]["имя"], "Алекс")

    def test_braces_inside_strings(self) -> None:
        parsed = extract_json_object('{"goal": {"текст": "фигурная {скобка}"}}')
        self.assertEqual(parsed["goal"]["текст"], "фигурная {скобка}")

    def test_broken_json_returns_error(self) -> None:
        parsed, error = parse_response("не json вовсе")
        self.assertIsNone(parsed)
        self.assertIsNotNone(error)
        self.assertIsNotNone(parse_response("")[1])

    def test_request_contains_marker_and_current_facts(self) -> None:
        store = FactsStore(counter=_COUNTER)
        store.apply({"preferences": {"имя": "Алекс"}})
        request = build_request(store, "Привет", "Здравствуйте",
                                model="deepseek-v4-flash")
        joined = "\n".join(m["content"] for m in request["messages"])
        self.assertIn(FACTS_MARKER, joined)
        self.assertIn("Алекс", joined)
        self.assertIn("Привет", joined)
        self.assertIn("goal", DEFAULT_FACTS_PROMPT)


class AgentFactsTest(unittest.TestCase):
    """Обновление фактов внутри агента (служебный вызов подменяется)."""

    def setUp(self) -> None:
        self.service_requests: list = []
        self.replies: list = []

    def tearDown(self) -> None:
        shutil.rmtree(_TESTS_TMP, ignore_errors=True)

    def _make_agent(self, storage=None, settings=None) -> DeepSeekAgent:
        return DeepSeekAgent(
            api_key="k",
            model="deepseek-v4-flash",
            system_prompt="Системный промпт.",
            token_counter=_COUNTER,
            prices={"default": {"input": 1.0, "output": 1.0}, "models": {}},
            storage=storage,
            strategy="sticky_facts",
            strategy_settings=dict(settings or _STRATEGY_SETTINGS),
        )

    def _patch(self, agent: DeepSeekAgent, chat: str = "Ответ") -> None:
        agent._call_api = lambda: _chat_payload(chat)

        def fake_service(request):
            self.service_requests.append(request)
            reply = self.replies.pop(0) if self.replies else "{}"
            return _facts_payload(reply)

        agent._call_service_api = fake_service

    def test_facts_updated_after_each_turn(self) -> None:
        agent = self._make_agent()
        self._patch(agent)
        self.replies.append('{"preferences": {"имя": "Алекс"}}')
        agent.send_message("Меня зовут Алекс")
        self.assertEqual(len(self.service_requests), 1)
        self.assertEqual(agent.get_facts()["count"], 1)

        self.replies.append(
            '{"preferences": {"имя": "Алекс", "цвет": "синий"}}'
        )
        agent.send_message("Любимый цвет — синий")
        self.assertEqual(len(self.service_requests), 2)
        self.assertEqual(agent.get_facts()["count"], 2)

        # Служебный запрос содержит маркер и уходит отдельным вызовом
        joined = "\n".join(
            m["content"] for m in self.service_requests[0]["messages"]
        )
        self.assertIn(FACTS_MARKER, joined)
        agent.close()

    def test_facts_sent_as_system_message(self) -> None:
        agent = self._make_agent()
        self._patch(agent)
        self.replies.append('{"preferences": {"имя": "Алекс"}}')
        agent.send_message("Меня зовут Алекс")
        context = agent._build_context_messages()
        self.assertEqual(context[0]["role"], "system")
        self.assertEqual(context[1]["role"], "system")
        self.assertIn("Известные факты о диалоге:", context[1]["content"])
        self.assertIn("Алекс", context[1]["content"])
        agent.close()

    def test_broken_json_keeps_previous_facts(self) -> None:
        agent = self._make_agent()
        self._patch(agent)
        self.replies.append('{"preferences": {"имя": "Алекс"}}')
        agent.send_message("Меня зовут Алекс")
        self.replies.append("это не JSON")
        result = agent.send_message("Просто реплика")
        self.assertIsNone(result["error"])          # ход не сломан
        self.assertIsNotNone(agent.get_facts()["error"])
        self.assertEqual(agent.get_facts()["count"], 1)  # факты сохранены
        agent.close()

    def test_service_error_keeps_turn_alive(self) -> None:
        agent = self._make_agent()
        agent._call_api = lambda: _chat_payload("Ответ")

        def failing(request):
            self.service_requests.append(request)
            raise RuntimeError("служебный запрос не прошёл")

        agent._call_service_api = failing
        result = agent.send_message("Любимый цвет — синий")
        self.assertIsNone(result["error"])
        self.assertIn("не прошёл", agent.get_facts()["error"])
        agent.close()

    def test_local_command_works_without_llm(self) -> None:
        """Явное «запомни» работает даже при сбое служебного запроса."""
        agent = self._make_agent()
        agent._call_api = lambda: _chat_payload("Ответ")

        def failing(request):
            raise RuntimeError("нет сети")

        agent._call_service_api = failing
        agent.send_message("Запомни: дедлайн — пятница")
        facts = agent.get_facts()["facts"]
        self.assertEqual(facts["decisions"]["дедлайн"], "пятница")
        agent.close()

    def test_facts_disabled_skips_service_call(self) -> None:
        settings = dict(_STRATEGY_SETTINGS)
        settings["facts_enabled"] = False
        agent = self._make_agent(settings=settings)
        self._patch(agent)
        agent.send_message("Меня зовут Алекс")
        self.assertEqual(self.service_requests, [])
        # Локальный разбор всё равно сработал
        self.assertEqual(agent.get_facts()["count"], 1)
        agent.close()

    def test_public_api_remember_forget_clear(self) -> None:
        agent = self._make_agent()
        self._patch(agent)
        result = agent.remember("цель", "написать отчёт", "goal")
        self.assertTrue(result["ok"])
        self.assertEqual(result["count"], 1)
        self.assertTrue(agent.forget("цель")["ok"])
        self.assertEqual(agent.get_facts()["count"], 0)
        agent.remember("имя", "Алекс", "preferences")
        self.assertEqual(agent.clear_facts()["count"], 0)
        agent.close()

    def test_facts_counters_are_separate(self) -> None:
        agent = self._make_agent()
        self._patch(agent)
        self.replies.append('{"preferences": {"имя": "Алекс"}}')
        agent.send_message("Меня зовут Алекс")
        stats = agent.get_token_stats()
        self.assertEqual(stats["session"]["facts_requests"], 1)
        self.assertGreater(
            int(stats["session"]["facts_prompt_tokens"])
            + int(stats["session"]["facts_completion_tokens"]), 0
        )
        # Статистика диалога не смешивается со служебной
        self.assertEqual(stats["session"]["requests"], 1)
        self.assertEqual(stats["session"]["summary_requests"], 0)
        agent.close()

    def test_update_facts_now_extracts_from_dialog(self) -> None:
        agent = self._make_agent()
        self._patch(agent)
        agent.send_message("Меня зовут Алекс, любимый цвет — синий.")
        self.service_requests.clear()
        self.replies.append(
            '{"preferences": {"имя": "Алекс", "цвет": "синий"}}'
        )
        result = agent.update_facts_now(force=True)
        self.assertTrue(result["updated"])
        self.assertEqual(result["count"], 2)
        self.assertEqual(len(self.service_requests), 1)
        agent.close()

    def test_update_facts_now_without_facts_strategy(self) -> None:
        agent = DeepSeekAgent(
            api_key="k", model="deepseek-v4-flash", token_counter=_COUNTER,
            strategy="sliding_window",
            strategy_settings={"name": "sliding_window", "window": 5},
        )
        result = agent.update_facts_now()
        self.assertFalse(result["updated"])
        self.assertIsNotNone(result["error"])
        agent.close()

    def test_clear_history_clears_facts(self) -> None:
        agent = self._make_agent()
        self._patch(agent)
        self.replies.append('{"preferences": {"имя": "Алекс"}}')
        agent.send_message("Меня зовут Алекс")
        agent.clear_history()
        self.assertEqual(agent.get_facts()["count"], 0)
        self.assertEqual(len(agent.history), 1)
        agent.close()


class FactsPersistenceTest(unittest.TestCase):
    """Факты переживают «перезапуск» приложения."""

    def setUp(self) -> None:
        self.db_path = _make_db()

    def tearDown(self) -> None:
        shutil.rmtree(_TESTS_TMP, ignore_errors=True)

    def test_facts_restored_after_restart(self) -> None:
        store1 = ContextStore(db_path=self.db_path, enabled=True,
                              default_strategy="sticky_facts")
        agent1 = DeepSeekAgent(
            api_key="k", model="deepseek-v4-flash", token_counter=_COUNTER,
            storage=store1, strategy="sticky_facts",
            strategy_settings=dict(_STRATEGY_SETTINGS),
        )
        agent1._call_api = lambda: _chat_payload("Ответ")
        agent1._call_service_api = lambda request: _facts_payload(
            '{"preferences": {"имя": "Алекс", "цвет": "синий"}, '
            '"goal": {"цель": "написать отчёт"}}'
        )
        agent1.send_message("Меня зовут Алекс, цвет — синий")
        self.assertEqual(agent1.get_facts()["count"], 3)
        agent1.close()

        store2 = ContextStore(db_path=self.db_path, enabled=True,
                              default_strategy="sticky_facts")
        agent2 = DeepSeekAgent(
            api_key="k", model="deepseek-v4-flash", token_counter=_COUNTER,
            storage=store2, strategy="sticky_facts",
            strategy_settings=dict(_STRATEGY_SETTINGS),
        )
        facts = agent2.get_facts()
        self.assertEqual(facts["count"], 3)
        self.assertEqual(facts["facts"]["preferences"]["имя"], "Алекс")
        # Стратегия ветки восстановлена
        self.assertEqual(agent2.get_strategy()["sub_strategy"], "sticky_facts")
        agent2.close()

    def test_facts_stored_in_meta(self) -> None:
        store = ContextStore(db_path=self.db_path, enabled=True,
                             default_strategy="sticky_facts")
        agent = DeepSeekAgent(
            api_key="k", model="deepseek-v4-flash", token_counter=_COUNTER,
            storage=store, strategy="sticky_facts",
            strategy_settings=dict(_STRATEGY_SETTINGS),
        )
        agent.remember("цель", "отчёт", "goal")
        agent.close()

        state = ContextStore(db_path=self.db_path, enabled=True).load_state()
        self.assertIsNotNone(state)
        raw = state["facts"]
        self.assertEqual(raw["data"]["goal"]["цель"], "отчёт")
        self.assertIn("updated_at", raw)


class FactsLimitsInAgentTest(unittest.TestCase):
    """Лимиты фактов применяются и внутри агента."""

    def tearDown(self) -> None:
        shutil.rmtree(_TESTS_TMP, ignore_errors=True)

    def test_items_limit_applied_after_turn(self) -> None:
        settings = dict(_STRATEGY_SETTINGS)
        settings["facts_max_items"] = 2
        agent = DeepSeekAgent(
            api_key="k", model="deepseek-v4-flash", token_counter=_COUNTER,
            strategy="sticky_facts", strategy_settings=settings,
        )
        agent._call_api = lambda: _chat_payload("Ответ")
        payloads = [
            '{"decisions": {"a": "1"}}',
            '{"decisions": {"a": "1", "b": "2"}}',
            '{"decisions": {"a": "1", "b": "2", "c": "3"}}',
        ]
        agent._call_service_api = lambda request: _facts_payload(payloads.pop(0))
        for index in range(3):
            agent.send_message(f"Вопрос {index}")
        self.assertEqual(agent.get_facts()["count"], 2)
        agent.close()


if __name__ == "__main__":
    unittest.main()
