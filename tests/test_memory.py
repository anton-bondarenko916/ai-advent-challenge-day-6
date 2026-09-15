"""Тесты явной модели памяти агента (три слоя) — без GUI и без сети.

Проверяют:
- единый роутер ``MemoryManager.route``: неизвестный слой — ошибка, русские
  подписи принимаются, источник записывается;
- краткосрочную память (pin): лимит, присутствие закреплённого сообщения в
  запросе за пределами окна, отсутствие откреплённого;
- рабочую память: поля задачи, элементы, «Новая задача» с переносом решений,
  привязка к ветке, восстановление после «перезапуска»;
- долговременную память: отдельный файл, независимость от очистки истории и
  удаления БД чата, forget/clear_section, экспорт в Markdown;
- изоляцию слоёв (запись в один не меняет остальные);
- предложения LLM (pending): кандидаты не попадают в слои до подтверждения,
  accept/reject, битый JSON, ошибка служебного запроса не ломает ход;
- сборку контекста: порядок блоков, отсутствие пустых блоков и дублей
  (sticky_facts), усечение по бюджету (goal/status сохраняются);
- MEMORY_ENABLED=false → контекст идентичен текущему поведению;
- mock-режим: ни chat_history.db, ни long_memory.db не создаются;
- миграцию схемы БД чата v2 → v3.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import unittest
import uuid
from pathlib import Path
from unittest import mock

from src.agent import DeepSeekAgent
from src.config import DEFAULTS, get_memory_settings, get_strategy_settings
from src.facts import FACTS_MARKER
from src.memory import (
    DEFAULT_MEMORY_PROMPT,
    LAYERS,
    LAYER_TITLES,
    LONG_TERM,
    MEMORY_MARKER,
    SHORT_TERM,
    WORKING,
    LongTermStore,
    MemoryClassifier,
    MemoryError,
    MemoryManager,
    PendingMemoryStore,
    WorkingMemory,
    normalize_layer,
    normalize_section,
)
from src.storage import SCHEMA_VERSION, ContextStore
from src.tokens import HeuristicCounter

_TESTS_TMP = Path(__file__).resolve().parent / "tmp_memory"
_COUNTER = HeuristicCounter()

_PAYLOAD = {
    "content": "Тестовый ответ",
    "model": "deepseek-v4-flash",
    "tokens": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
    "finish_reason": "stop",
}


def _path(prefix: str) -> Path:
    _TESTS_TMP.mkdir(parents=True, exist_ok=True)
    return _TESTS_TMP / f"{prefix}_{uuid.uuid4().hex}.db"


def _memory_settings(long_path: Path | None = None, **overrides) -> dict:
    """Настройки памяти с временным путём долговременной БД по умолчанию."""
    settings = get_memory_settings(dict(DEFAULTS))
    if long_path is None and "long_path" not in overrides:
        long_path = _path("long")
    if long_path is not None:
        settings["long_path"] = str(long_path)
        settings["long_enabled"] = True
    settings.update(overrides)
    return settings


def _strategy_settings(**overrides) -> dict:
    settings = get_strategy_settings({})
    settings.update(overrides)
    return settings


def _make_agent(chat_db: Path | None = None, long_db: Path | None = None,
                *, strategy: str = "sliding_window", window: int = 20,
                memory: bool = True, **memory_overrides) -> DeepSeekAgent:
    storage = ContextStore(db_path=chat_db, enabled=True) if chat_db else None
    memory_settings = (_memory_settings(long_db, **memory_overrides)
                       if memory else None)
    return DeepSeekAgent(
        api_key="k", model="deepseek-v4-flash", token_counter=_COUNTER,
        storage=storage, strategy=strategy,
        strategy_settings=_strategy_settings(window=window),
        memory_settings=memory_settings,
    )


class _MemoryTestBase(unittest.TestCase):
    """Общая обвязка: временная папка внутри репозитория."""

    def tearDown(self) -> None:
        shutil.rmtree(_TESTS_TMP, ignore_errors=True)


# ---------------------------------------------------------------------------
# Роутер
# ---------------------------------------------------------------------------

class RouterTest(_MemoryTestBase):
    """Единая точка записи: MemoryManager.route."""

    def test_unknown_layer_returns_error_and_writes_nothing(self) -> None:
        long_db = _path("long")
        manager = MemoryManager(
            settings=_memory_settings(long_db),
            long_store=LongTermStore(long_db, enabled=True),
            counter=_COUNTER,
        )
        result = manager.route("нет-такого-слоя", "значение")
        self.assertFalse(result["ok"])
        self.assertIn("неизвестн", result["error"].lower())
        self.assertEqual(manager.working.count(), 0)
        self.assertEqual(manager.long.list_items(), [])

    def test_russian_layer_labels_accepted(self) -> None:
        self.assertEqual(normalize_layer("В диалог"), SHORT_TERM)
        self.assertEqual(normalize_layer("в рабочую"), WORKING)
        self.assertEqual(normalize_layer("В долговременную"), LONG_TERM)
        self.assertEqual(normalize_layer("рабочая"), WORKING)

    def test_normalize_layer_rejects_unknown(self) -> None:
        with self.assertRaises(MemoryError):
            normalize_layer("куда-нибудь")

    def test_normalize_section_rejects_unknown(self) -> None:
        with self.assertRaises(MemoryError):
            normalize_section("нет-раздела")

    def test_route_records_source(self) -> None:
        long_db = _path("long")
        manager = MemoryManager(
            settings=_memory_settings(long_db),
            long_store=LongTermStore(long_db, enabled=True),
            counter=_COUNTER,
        )
        result = manager.route(LONG_TERM, "Алекс", key="имя",
                               section="profile", source="confirmed")
        self.assertTrue(result["ok"])
        items = manager.long.list_items("profile")
        self.assertEqual(items[0]["source"], "confirmed")

    def test_route_short_term_requires_message_index(self) -> None:
        manager = MemoryManager(settings=_memory_settings(), counter=_COUNTER)
        result = manager.route(SHORT_TERM, "текст")
        self.assertFalse(result["ok"])
        self.assertIn("индекс", result["error"].lower())

    def test_route_autogenerates_key(self) -> None:
        long_db = _path("long")
        manager = MemoryManager(
            settings=_memory_settings(long_db),
            long_store=LongTermStore(long_db, enabled=True),
            counter=_COUNTER,
        )
        result = manager.route(LONG_TERM, "любимый цвет синий",
                               section="profile")
        self.assertTrue(result["ok"])
        self.assertTrue(result["key"])


# ---------------------------------------------------------------------------
# Краткосрочная память (pin)
# ---------------------------------------------------------------------------

class ShortTermMemoryTest(_MemoryTestBase):
    """Закрепление сообщений (краткосрочная память)."""

    def test_pin_and_unpin(self) -> None:
        agent = _make_agent()
        try:
            agent._append_message({"role": "user", "content": "привет"})
            agent._append_message({"role": "assistant", "content": "ответ"})
            result = agent.pin_message(0)
            self.assertTrue(result["ok"])
            self.assertEqual([p["index"] for p in agent.list_pinned()], [0])
            result = agent.unpin_message(0)
            self.assertTrue(result["ok"])
            self.assertEqual(agent.list_pinned(), [])
        finally:
            agent.close()

    def test_pin_limit_evicts_oldest(self) -> None:
        agent = _make_agent(pin_max=2)
        try:
            for i in range(4):
                agent._append_message({"role": "user", "content": f"вопрос {i}"})
            agent.pin_message(0)
            agent.pin_message(1)
            agent.pin_message(2)
            pinned = [p["index"] for p in agent.list_pinned()]
            self.assertEqual(pinned, [1, 2])
        finally:
            agent.close()

    def test_pinned_message_in_preview_beyond_window(self) -> None:
        """Закреплённое сообщение уходит в запрос, даже выйдя за окно."""
        agent = _make_agent(window=2)
        try:
            for i in range(12):
                agent._append_message({"role": "user", "content": f"уникальное-{i}"})
                agent._append_message({"role": "assistant", "content": f"ответ-{i}"})
            agent.pin_message(0)  # «уникальное-0» выходит за окно 2
            preview = agent.preview_context()
            pinned_blocks = [p for p in preview if p["layer"] == "pinned"]
            self.assertEqual(len(pinned_blocks), 1)
            self.assertIn("уникальное-0", pinned_blocks[0]["content"])
            # Без pin это сообщение не попало бы в запрос
            agent.unpin_message(0)
            preview = agent.preview_context()
            self.assertFalse([p for p in preview if p["layer"] == "pinned"])
        finally:
            agent.close()

    def test_unpinned_message_not_in_preview(self) -> None:
        agent = _make_agent(window=2)
        try:
            for i in range(12):
                agent._append_message({"role": "user", "content": f"деталь-{i}"})
            agent.pin_message(1)
            joined = "\n".join(p["content"] for p in agent.preview_context())
            self.assertNotIn("деталь-0", joined)
            self.assertIn("деталь-1", joined)
        finally:
            agent.close()


# ---------------------------------------------------------------------------
# Рабочая память (задача)
# ---------------------------------------------------------------------------

class WorkingMemoryTest(_MemoryTestBase):
    """Рабочая память: поля задачи, элементы, привязка к ветке."""

    def test_task_fields_and_items(self) -> None:
        agent = _make_agent()
        try:
            agent.set_task_field("goal", "написать отчёт")
            agent.set_task_field("plan", ["собрать данные", "написать текст"])
            agent.add_working_item("дедлайн", "20 мая", "note")
            working = agent.get_working_memory()
            self.assertEqual(working["task"]["goal"], "написать отчёт")
            self.assertEqual(working["task"]["plan"],
                             ["собрать данные", "написать текст"])
            self.assertIn("дедлайн", working["items"])
            self.assertEqual(working["items"]["дедлайн"]["value"], "20 мая")
        finally:
            agent.close()

    def test_start_new_task_carries_decisions(self) -> None:
        long_db = _path("long")
        agent = _make_agent(long_db=long_db)
        try:
            agent.add_working_item("архитектура", "выбрали SQLite", "decision")
            agent.add_working_item("файл", "отчёт.md", "artifact")
            result = agent.start_new_task(carry_to_long=["архитектура"])
            self.assertTrue(result["ok"])
            self.assertEqual(result["carried"], 1)
            self.assertEqual(agent.get_working_memory()["items"], {})
            decisions = agent.get_long_memory("decisions")
            self.assertTrue(any(i["key"] == "архитектура" for i in decisions))
        finally:
            agent.close()

    def test_working_memory_restored_after_restart(self) -> None:
        chat_db = _path("chat")
        long_db = _path("long")
        agent1 = _make_agent(chat_db, long_db)
        agent1.set_task_field("goal", "задача-1")
        agent1.add_working_item("ключ", "значение", "decision")
        agent1._persist()
        agent1.close()

        agent2 = _make_agent(chat_db, long_db)
        try:
            working = agent2.get_working_memory()
            self.assertEqual(working["task"]["goal"], "задача-1")
            self.assertEqual(working["items"]["ключ"]["value"], "значение")
        finally:
            agent2.close()

    def test_working_memory_per_branch(self) -> None:
        chat_db = _path("chat")
        long_db = _path("long")
        agent = _make_agent(chat_db, long_db)
        try:
            for i in range(3):
                agent._append_message({"role": "user", "content": f"вопрос {i}"})
                agent._append_message({"role": "assistant", "content": f"ответ {i}"})
            agent.add_working_item("проект", "ветка-A", "note")
            created = agent.create_branch(name="ветка-A")
            self.assertIsNone(created["error"])
            branch_a = created["branch"]["id"]
            # Новая ветка скопировала рабочую память префикса
            self.assertIn("проект", agent.get_working_memory()["items"])
            agent.clear_working_memory()
            self.assertEqual(agent.get_working_memory()["items"], {})

            # Возврат в корневую ветку: у неё своя рабочая память
            agent.switch_branch(1)
            self.assertIn("проект", agent.get_working_memory()["items"])
            # Переключение обратно в A: там пусто
            agent.switch_branch(branch_a)
            self.assertEqual(agent.get_working_memory()["items"], {})
        finally:
            agent.close()


# ---------------------------------------------------------------------------
# Долговременная память (отдельный файл)
# ---------------------------------------------------------------------------

class LongTermMemoryTest(_MemoryTestBase):
    """Долговременная память: отдельный файл и независимость от истории."""

    def test_file_created_lazily(self) -> None:
        long_db = _path("long")
        store = LongTermStore(long_db, enabled=True)
        self.assertFalse(long_db.exists())
        store.add("profile", "имя", "Алекс")
        self.assertTrue(long_db.exists())
        store.close()

    def test_clear_history_keeps_long_memory(self) -> None:
        chat_db = _path("chat")
        long_db = _path("long")
        agent = _make_agent(chat_db, long_db)
        try:
            agent.remember_forever("profile", "имя", "Алекс")
            agent._append_message({"role": "user", "content": "привет"})
            agent._persist()
            agent.clear_history()
            self.assertEqual(agent.history[1:], [])
            items = agent.get_long_memory("profile")
            self.assertTrue(any(i["key"] == "имя" for i in items))
        finally:
            agent.close()

    def test_deleting_chat_db_keeps_long_memory(self) -> None:
        chat_db = _path("chat")
        long_db = _path("long")
        agent = _make_agent(chat_db, long_db)
        agent.remember_forever("profile", "имя", "Алекс")
        agent._append_message({"role": "user", "content": "привет"})
        agent._persist()
        agent.close()

        # Удаляем БД чата — долговременная память остаётся
        chat_db.unlink()
        agent2 = _make_agent(chat_db, long_db)
        try:
            items = agent2.get_long_memory("profile")
            self.assertTrue(any(i["key"] == "имя" for i in items))
        finally:
            agent2.close()

    def test_forget_and_clear_section(self) -> None:
        long_db = _path("long")
        agent = _make_agent(long_db=long_db)
        try:
            agent.remember_forever("profile", "имя", "Алекс")
            agent.remember_forever("profile", "город", "Казань")
            agent.remember_forever("decisions", "архитектура", "SQLite")
            self.assertTrue(agent.forget_forever("profile", "имя")["ok"])
            self.assertFalse(any(i["key"] == "имя"
                                 for i in agent.get_long_memory("profile")))
            agent.clear_long_memory("decisions")
            self.assertEqual(agent.get_long_memory("decisions"), [])
            self.assertTrue(agent.get_long_memory("profile"))
        finally:
            agent.close()

    def test_storage_clear_does_not_touch_long_memory(self) -> None:
        """ContextStore.clear() (БД чата) долговременную память не трогает."""
        chat_db = _path("chat")
        long_db = _path("long")
        store = ContextStore(db_path=chat_db, enabled=True)
        long_store = LongTermStore(long_db, enabled=True)
        long_store.add("profile", "имя", "Алекс")
        store.save_state(
            messages=[{"role": "system", "content": "с"},
                      {"role": "user", "content": "вопрос"}],
            params={"model": "deepseek-v4-flash"},
        )
        self.assertTrue(store.clear())
        store.close()

        # Долговременная память цела и в своём отдельном файле
        self.assertTrue(long_db.exists())
        items = LongTermStore(long_db, enabled=True).list_items("profile")
        self.assertEqual(items[0]["value"], "Алекс")

    def test_export_markdown(self) -> None:
        long_db = _path("long")
        export_path = _TESTS_TMP / f"export_{uuid.uuid4().hex}.md"
        agent = _make_agent(long_db=long_db)
        try:
            agent.remember_forever("profile", "имя", "Алекс")
            agent.remember_forever("knowledge", "правило", "без сети")
            result = agent.export_long_memory(str(export_path))
            self.assertTrue(result["ok"])
            text = export_path.read_text(encoding="utf-8")
            self.assertIn("profile", text)
            self.assertIn("имя", text)
            self.assertIn("knowledge", text)
        finally:
            agent.close()


# ---------------------------------------------------------------------------
# Изоляция слоёв
# ---------------------------------------------------------------------------

class LayerIsolationTest(_MemoryTestBase):
    """Запись в один слой не меняет остальные."""

    def test_layers_isolated(self) -> None:
        long_db = _path("long")
        agent = _make_agent(long_db=long_db)
        try:
            agent._append_message({"role": "user", "content": "сообщение"})
            before = agent.get_memory_report()
            agent.save_to_memory(LONG_TERM, "знание", key="k",
                                 section="knowledge")
            after = agent.get_memory_report()
            self.assertEqual(before["working"]["count"],
                             after["working"]["count"])
            self.assertEqual(before["short_term"]["pinned_count"],
                             after["short_term"]["pinned_count"])
            self.assertEqual(after["long_term"]["count"],
                             before["long_term"]["count"] + 1)
        finally:
            agent.close()


# ---------------------------------------------------------------------------
# Предложения LLM (pending)
# ---------------------------------------------------------------------------

def _candidates_json(**overrides) -> str:
    payload = {
        "short_term": [],
        "working": [
            {"key": "дедлайн", "value": "20 мая", "kind": "note",
             "confidence": 0.9},
        ],
        "long_term": [
            {"section": "profile", "key": "любимый_цвет", "value": "синий",
             "confidence": 0.95},
        ],
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


class PendingMemoryTest(_MemoryTestBase):
    """LLM только предлагает (pending); пользователь решает."""

    def test_candidates_go_to_pending_not_layers(self) -> None:
        long_db = _path("long")
        agent = _make_agent(long_db=long_db, propose_llm=True)
        try:
            with mock.patch.object(agent, "_call_api", return_value=dict(_PAYLOAD)), \
                    mock.patch.object(agent, "_call_service_api",
                                      return_value={"content": _candidates_json(),
                                                    "model": "m",
                                                    "tokens": None}):
                agent.send_message("привет")
            pending = agent.get_pending_memory()
            self.assertGreaterEqual(len(pending), 2)
            # В слои кандидаты не попали
            self.assertEqual(agent.get_long_memory(), [])
            self.assertEqual(agent.get_working_memory()["items"], {})
        finally:
            agent.close()

    def test_accept_pending_writes_layer(self) -> None:
        long_db = _path("long")
        agent = _make_agent(long_db=long_db)
        try:
            with mock.patch.object(agent, "_call_api", return_value=dict(_PAYLOAD)), \
                    mock.patch.object(agent, "_call_service_api",
                                      return_value={"content": _candidates_json(),
                                                    "model": "m",
                                                    "tokens": None}):
                agent.send_message("привет")
            pending = agent.get_pending_memory()
            long_item = next(p for p in pending if p["layer"] == LONG_TERM)
            result = agent.accept_pending(long_item["id"])
            self.assertTrue(result["ok"])
            self.assertTrue(any(i["key"] == "любимый_цвет"
                                for i in agent.get_long_memory("profile")))
        finally:
            agent.close()

    def test_reject_pending_writes_nothing(self) -> None:
        long_db = _path("long")
        agent = _make_agent(long_db=long_db)
        try:
            with mock.patch.object(agent, "_call_api", return_value=dict(_PAYLOAD)), \
                    mock.patch.object(agent, "_call_service_api",
                                      return_value={"content": _candidates_json(),
                                                    "model": "m",
                                                    "tokens": None}):
                agent.send_message("привет")
            pending = agent.get_pending_memory()
            agent.reject_pending(pending[0]["id"])
            self.assertLess(len(agent.get_pending_memory()), len(pending))
            self.assertEqual(agent.get_long_memory(), [])
            self.assertEqual(agent.get_working_memory()["items"], {})
        finally:
            agent.close()

    def test_broken_json_pending_empty_layers_intact(self) -> None:
        long_db = _path("long")
        agent = _make_agent(long_db=long_db)
        try:
            agent.remember_forever("profile", "имя", "Алекс")
            with mock.patch.object(agent, "_call_api", return_value=dict(_PAYLOAD)), \
                    mock.patch.object(agent, "_call_service_api",
                                      return_value={"content": "не JSON",
                                                    "model": "m", "tokens": None}):
                result = agent.send_message("привет")
            self.assertIsNone(result["error"])
            self.assertEqual(agent.get_pending_memory(), [])
            self.assertTrue(any(i["key"] == "имя"
                                for i in agent.get_long_memory("profile")))
        finally:
            agent.close()

    def test_service_error_does_not_break_turn(self) -> None:
        agent = _make_agent()
        try:
            with mock.patch.object(agent, "_call_api", return_value=dict(_PAYLOAD)), \
                    mock.patch.object(agent, "_call_service_api",
                                      side_effect=RuntimeError("нет сети")):
                result = agent.send_message("привет")
            self.assertIsNone(result["error"])
            self.assertEqual(agent.get_pending_memory(), [])
        finally:
            agent.close()

    def test_memory_counters_grow_not_dialog(self) -> None:
        agent = _make_agent()
        try:
            with mock.patch.object(agent, "_call_api", return_value=dict(_PAYLOAD)), \
                    mock.patch.object(agent, "_call_service_api",
                                      return_value={"content": _candidates_json(),
                                                    "model": "m",
                                                    "tokens": {"prompt_tokens": 7,
                                                               "completion_tokens": 3,
                                                               "total_tokens": 10,
                                                               "estimated": False}}):
                agent.send_message("привет")
            session = agent.get_token_stats()["session"]
            self.assertEqual(session["memory_requests"], 1)
            self.assertEqual(session["memory_prompt_tokens"], 7)
            self.assertEqual(session["memory_completion_tokens"], 3)
            # Статистика диалога не изменилась служебным запросом
            self.assertEqual(session["requests"], 1)
        finally:
            agent.close()

    def test_parse_prompt_requires_strict_json(self) -> None:
        self.assertIn("JSON", DEFAULT_MEMORY_PROMPT)
        self.assertIn(MEMORY_MARKER, DEFAULT_MEMORY_PROMPT)
        self.assertIn("не выдумывай", DEFAULT_MEMORY_PROMPT.lower())
        parsed = MemoryClassifier.parse("мусор без json")
        self.assertIn("error", parsed)
        self.assertEqual(parsed[SHORT_TERM], [])


# ---------------------------------------------------------------------------
# Сборка контекста
# ---------------------------------------------------------------------------

class ContextAssemblyTest(_MemoryTestBase):
    """Порядок блоков памяти, отсутствие пустых и дублей, бюджет."""

    def _seeded_agent(self, *, strategy="sliding_window", long_db=None):
        agent = _make_agent(long_db=long_db, strategy=strategy)
        agent.remember_forever("profile", "имя", "Алекс")
        agent.add_working_item("дедлайн", "20 мая", "note")
        agent._append_message({"role": "user", "content": "старое сообщение"})
        agent._append_message({"role": "assistant", "content": "старый ответ"})
        agent.pin_message(0)
        for i in range(6):
            agent._append_message({"role": "user", "content": f"новое {i}"})
            agent._append_message({"role": "assistant", "content": f"ответ {i}"})
        return agent

    def test_block_order(self) -> None:
        long_db = _path("long")
        agent = self._seeded_agent(long_db=long_db)
        try:
            preview = agent.preview_context()
            layers = [p["layer"] for p in preview]
            # system-промпт → long_term → working → pinned → диалог
            self.assertEqual(layers[0], "system")
            self.assertEqual(layers[1], LONG_TERM)
            self.assertEqual(layers[2], WORKING)
            self.assertEqual(layers[3], "pinned")
            self.assertIn("short_term", layers)
            self.assertLess(layers.index(LONG_TERM), layers.index(WORKING))
            self.assertLess(layers.index(WORKING), layers.index("pinned"))
        finally:
            agent.close()

    def test_no_empty_blocks(self) -> None:
        agent = _make_agent()
        try:
            agent._append_message({"role": "user", "content": "привет"})
            preview = agent.preview_context()
            for item in preview:
                self.assertNotEqual(item["content"].strip(), "")
            # Ни одного блока памяти — все слои пусты
            self.assertFalse([p for p in preview
                              if p["layer"] in (LONG_TERM, WORKING, "pinned")])
        finally:
            agent.close()

    def test_no_duplicates_with_sticky_facts(self) -> None:
        """Рабочая память не повторяет пары, ушедшие блоком фактов."""
        long_db = _path("long")
        agent = _make_agent(long_db=long_db, strategy="sticky_facts")
        try:
            agent.add_working_item("дедлайн", "20 мая", "decision")
            agent._append_message({"role": "user", "content": "привет"})
            preview = agent.preview_context()
            facts_text = "\n".join(
                p["content"] for p in preview if p["layer"] == "system"
            )
            working_text = "\n".join(
                p["content"] for p in preview if p["layer"] == WORKING
            )
            # Пара есть в блоке фактов и НЕ дублируется в рабочем блоке
            self.assertIn("дедлайн", facts_text)
            self.assertNotIn("дедлайн", working_text)
        finally:
            agent.close()

    def test_budget_truncation_keeps_goal_status(self) -> None:
        long_db = _path("long")
        agent = _make_agent(long_db=long_db, working_max_tokens=40,
                            max_tokens=200)
        try:
            agent.set_task_field("goal", "ГЛАВНАЯ-ЦЕЛЬ")
            agent.set_task_field("status", "В-РАБОТЕ")
            agent.set_task_field("plan", ["шаг-" + "очень длинный " * 10] * 8)
            agent.set_task_field("open_questions", ["вопрос " * 20] * 5)
            agent._append_message({"role": "user", "content": "привет"})
            working_text = "\n".join(
                p["content"] for p in agent.preview_context()
                if p["layer"] == WORKING
            )
            self.assertIn("ГЛАВНАЯ-ЦЕЛЬ", working_text)
            self.assertIn("В-РАБОТЕ", working_text)
            self.assertNotIn("шаг-", working_text)  # plan вытесняется первым
        finally:
            agent.close()


# ---------------------------------------------------------------------------
# Отключение памяти и mock-режим
# ---------------------------------------------------------------------------

class DisabledAndMockTest(_MemoryTestBase):
    """MEMORY_ENABLED=false и изоляция mock-режима."""

    def test_memory_disabled_context_unchanged(self) -> None:
        """MEMORY_ENABLED=false даёт контекст, идентичный текущему поведению."""
        off_no_settings = _make_agent(memory=False)  # memory_settings=None
        off_flag = _make_agent()                     # память включена...
        off_flag._memory_settings = dict(off_flag._memory_settings, enabled=False)
        off_flag._memory_enabled = False             # ...но выключена флагом
        try:
            for agent in (off_no_settings, off_flag):
                agent.remember_forever("profile", "имя", "Алекс")
                agent.add_working_item("дедлайн", "20 мая", "note")
                for i in range(12):
                    agent._append_message({"role": "user", "content": f"вопрос {i}"})
                    agent._append_message({"role": "assistant", "content": f"ответ {i}"})
                agent.pin_message(0)
            self.assertEqual(
                off_no_settings._build_context_messages(),
                off_flag._build_context_messages(),
            )
            # Блоков памяти в контексте нет
            self.assertFalse([
                p for p in off_flag.preview_context()
                if p["layer"] in (LONG_TERM, WORKING, "pinned")
            ])
        finally:
            off_no_settings.close()
            off_flag.close()

    def test_mock_mode_creates_no_db_files(self) -> None:
        chat_db = _path("chat")
        long_db = _path("long")
        agent = DeepSeekAgent(
            api_key="", mock_mode=True, token_counter=_COUNTER,
            storage=ContextStore(db_path=chat_db, enabled=True),
            memory_settings=_memory_settings(long_db),
        )
        try:
            agent.remember_forever("profile", "имя", "Алекс")
            agent.add_working_item("дедлайн", "20 мая", "note")
            agent._append_message({"role": "user", "content": "привет"})
            agent.pin_message(0)
            result = agent.send_message("привет из mock")
            self.assertIsNone(result["error"])
            # Память живёт только в процессе; файлы НЕ создаются
            self.assertFalse(chat_db.exists())
            self.assertFalse(long_db.exists())
            # В процессе данные есть
            self.assertEqual(agent.get_long_memory(), [])  # long-слой отключён
            self.assertIn("дедлайн", agent.get_working_memory()["items"])
        finally:
            agent.close()
        self.assertFalse(chat_db.exists())
        self.assertFalse(long_db.exists())


# ---------------------------------------------------------------------------
# Миграция схемы v2 → v3
# ---------------------------------------------------------------------------

def _make_v2_db(path: Path) -> None:
    """Создаёт БД схемы v2 (сообщения + ветки, без pinned)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            """
            CREATE TABLE messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL,
                branch_id  INTEGER NOT NULL DEFAULT 1,
                seq        INTEGER
            );
            CREATE TABLE branches (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT NOT NULL,
                parent_id  INTEGER NULL,
                forked_at  INTEGER NOT NULL DEFAULT 0,
                strategy   TEXT NOT NULL DEFAULT 'sliding_window',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        now = "2026-01-01T00:00:00"
        conn.execute(
            "INSERT INTO branches(id, name, parent_id, forked_at, strategy, "
            "created_at, updated_at) VALUES (1, 'main', NULL, 0, "
            "'sliding_window', ?, ?)", (now, now),
        )
        conn.executemany(
            "INSERT INTO messages(role, content, created_at, branch_id, seq) "
            "VALUES (?, ?, ?, 1, ?)",
            [("system", "Старая система.", now, 0),
             ("user", "Вопрос", now, 1),
             ("assistant", "Ответ", now, 2)],
        )
        conn.execute("INSERT INTO meta(key, value) VALUES ('schema_version', '2')")
        conn.commit()
    finally:
        conn.close()


class MigrationTest(_MemoryTestBase):
    """Миграция v2 → v3 без потери данных."""

    def test_v2_db_migrates_to_v3(self) -> None:
        db = _path("v2")
        _make_v2_db(db)
        store = ContextStore(db_path=db, enabled=True)
        state = store.load_state()
        self.assertIsNotNone(state)
        self.assertEqual(len(state["messages"]), 3)
        self.assertEqual(state["pinned"], [])
        store.close()

        conn = sqlite3.connect(str(db))
        try:
            version = conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()[0]
            columns = {row[1] for row in
                       conn.execute("PRAGMA table_info(messages)")}
            pinned = [row[0] for row in
                      conn.execute("SELECT pinned FROM messages")]
        finally:
            conn.close()
        self.assertEqual(version, SCHEMA_VERSION)
        self.assertEqual(version, "3")
        self.assertIn("pinned", columns)
        self.assertIn("pin_note", columns)
        self.assertEqual(pinned, [0, 0, 0])

    def test_v3_pins_round_trip(self) -> None:
        db = _path("v3")
        store = ContextStore(db_path=db, enabled=True)
        store.save_state(
            messages=[{"role": "system", "content": "с"},
                      {"role": "user", "content": "вопрос"},
                      {"role": "assistant", "content": "ответ"}],
            params={"model": "deepseek-v4-flash"},
            pinned={0: "важное"},
            working={"task": {"goal": "цель"}, "items": {}},
        )
        state = ContextStore(db_path=db, enabled=True).load_state()
        self.assertEqual(state["pinned"], [{"index": 0, "note": "важное"}])
        self.assertEqual(state["working_memory"]["task"]["goal"], "цель")


if __name__ == "__main__":
    unittest.main()
