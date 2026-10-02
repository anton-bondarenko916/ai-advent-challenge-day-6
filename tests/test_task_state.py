"""Тесты памяти задачи (ТЗ ai_1_24, модули 1-2): TaskState и репозиторий.

Проверяются: структура task_state (цель/уточнения/ограничения/термины),
to_prompt_block (формат системного промпта), дедупликация при мердже
обновлений LLM, персистентность в SQLite (get_or_create/update/reset),
миграция messages.session_id + индекс и маркировка сообщений сессией.

Запуск из корня проекта:
    python -m unittest tests.test_task_state -v
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.task_state import TaskState, TaskStateRepository


def make_repo() -> tuple:
    """Репозиторий над временной БД (+ путь для проверки персистентности)."""
    db_path = Path(tempfile.mkdtemp()) / "chat_history.db"
    return TaskStateRepository(db_path), db_path


class TaskStateTest(unittest.TestCase):
    """Dataclass TaskState: словарь, промпт-блок, мердж обновлений."""

    def test_defaults_empty(self) -> None:
        state = TaskState(session_id="s", profile_id=1)
        self.assertTrue(state.is_empty())
        self.assertEqual(state.to_prompt_block(), "Состояние диалога пусто.")
        self.assertEqual(
            state.to_dict(),
            {"session_id": "s", "profile_id": 1, "goal": "",
             "clarifications": [], "constraints": [], "terms": {}},
        )

    def test_to_prompt_block_format(self) -> None:
        state = TaskState(
            session_id="s", profile_id=1,
            goal="Настроить RAG с Ollama",
            clarifications=["документы на русском"],
            constraints=["без Docker"],
            terms={"RAG": "Retrieval-Augmented Generation"},
        )
        block = state.to_prompt_block()
        self.assertIn("🎯 Цель диалога: Настроить RAG с Ollama", block)
        self.assertIn("📝 Уточнения:", block)
        self.assertIn("  • документы на русском", block)
        self.assertIn("⚠️ Ограничения:", block)
        self.assertIn("  • без Docker", block)
        self.assertIn("📚 Термины:", block)
        self.assertIn("  • RAG: Retrieval-Augmented Generation", block)

    def test_to_prompt_block_partial(self) -> None:
        state = TaskState(session_id="s", profile_id=1, goal="Цель")
        block = state.to_prompt_block()
        self.assertIn("🎯 Цель", block)
        self.assertNotIn("Уточнения", block)
        self.assertNotIn("Термины", block)

    def test_apply_update_merges_without_duplicates(self) -> None:
        state = TaskState(
            session_id="s", profile_id=1,
            clarifications=["используется Ollama"],
        )
        changed = state.apply_update({
            "goal": "",
            "clarifications": ["используется Ollama",
                               "Используется Ollama",
                               "документы на русском"],
            "constraints": ["без Docker"],
            "terms": {"RAG": "Retrieval-Augmented Generation"},
        })
        self.assertTrue(changed)
        self.assertEqual(state.clarifications,
                         ["используется Ollama", "документы на русском"])
        self.assertEqual(state.constraints, ["без Docker"])
        self.assertEqual(state.terms,
                         {"RAG": "Retrieval-Augmented Generation"})

    def test_apply_update_replaces_goal_only_non_empty(self) -> None:
        state = TaskState(session_id="s", profile_id=1, goal="Старая цель")
        state.apply_update({"goal": "Новая цель"})
        self.assertEqual(state.goal, "Новая цель")
        # Пустая цель от модели цель не стирает
        changed = state.apply_update({"goal": ""})
        self.assertEqual(state.goal, "Новая цель")
        self.assertFalse(changed)

    def test_apply_update_terms_upsert(self) -> None:
        state = TaskState(session_id="s", profile_id=1,
                          terms={"RAG": "старое"})
        state.apply_update({"terms": {"RAG": "новое", "chunk": "фрагмент"}})
        self.assertEqual(state.terms,
                         {"RAG": "новое", "chunk": "фрагмент"})

    def test_apply_update_garbage_safe(self) -> None:
        state = TaskState(session_id="s", profile_id=1)
        self.assertFalse(state.apply_update(None))
        self.assertFalse(state.apply_update({"clarifications": ["", "  "]}))
        self.assertFalse(state.apply_update({"terms": {"": "определение"}}))


class TaskStateRepositoryTest(unittest.TestCase):
    """SQLite-репозиторий: схема, CRUD, сброс, маркировка messages."""

    def setUp(self) -> None:
        self.repo, self.db_path = make_repo()
        self.session_id = "11111111-2222-3333-4444-555555555555"

    def test_schema_task_state_table(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        columns = {row[1] for row in
                   conn.execute("PRAGMA table_info(task_state)")}
        indexes = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name NOT LIKE 'sqlite_%'")}
        conn.close()
        self.assertIn("task_state", tables)
        self.assertLessEqual(
            {"session_id", "profile_id", "goal", "clarifications",
             "constraints", "terms"}, columns)
        self.assertIn("idx_task_state_session", indexes)

    def test_get_or_create_then_update_roundtrip(self) -> None:
        state = self.repo.get_or_create(self.session_id, 7)
        self.assertTrue(state.is_empty())
        state.goal = "Настроить RAG"
        state.clarifications = ["документы на русском"]
        state.constraints = ["без Docker"]
        state.terms = {"chunk": "фрагмент документа"}
        self.assertTrue(self.repo.update(state))

        # Новый экземпляр репозитория — те же данные (только SQLite)
        second = TaskStateRepository(self.db_path)
        loaded = second.get_or_create(self.session_id, 7)
        self.assertEqual(loaded.goal, "Настроить RAG")
        self.assertEqual(loaded.clarifications, ["документы на русском"])
        self.assertEqual(loaded.constraints, ["без Docker"])
        self.assertEqual(loaded.terms, {"chunk": "фрагмент документа"})

    def test_reset_deletes_row(self) -> None:
        state = self.repo.get_or_create(self.session_id, 1)
        state.goal = "Цель"
        self.repo.update(state)
        self.assertTrue(self.repo.reset(self.session_id))
        self.assertTrue(self.repo.get_or_create(self.session_id, 1).is_empty())

    def test_update_upserts_missing_row(self) -> None:
        # UPDATE без строки (сессию удалили параллельно) — INSERT fallback
        state = TaskState(session_id="fresh-session", profile_id=2,
                          goal="Цель")
        self.assertTrue(self.repo.update(state))
        self.assertEqual(
            self.repo.get_or_create("fresh-session", 2).goal, "Цель")

    def test_unicode_roundtrip(self) -> None:
        state = self.repo.get_or_create(self.session_id, 1)
        state.clarifications = ["документация на русском языке",
                                "эмбеддер nomic-embed-text"]
        state.terms = {"эмбеддинг": "векторное представление текста"}
        self.repo.update(state)
        loaded = TaskStateRepository(self.db_path).get_or_create(
            self.session_id, 1)
        self.assertEqual(loaded.clarifications, state.clarifications)
        self.assertEqual(loaded.terms, state.terms)

    def test_messages_session_column_and_tagging(self) -> None:
        # Миграция ТЗ: колонка session_id в messages + индекс
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "role TEXT NOT NULL, content TEXT NOT NULL, "
            "created_at TEXT NOT NULL)")
        conn.execute("INSERT INTO messages(role, content, created_at) "
                     "VALUES ('user', 'Вопрос', '2026-10-02 10:00:00')")
        conn.commit()
        conn.close()

        self.assertTrue(self.repo.tag_session_messages(self.session_id))
        conn = sqlite3.connect(str(self.db_path))
        columns = {row[1] for row in
                   conn.execute("PRAGMA table_info(messages)")}
        indexes = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name NOT LIKE 'sqlite_%'")}
        row = conn.execute(
            "SELECT session_id FROM messages").fetchone()
        conn.close()
        self.assertIn("session_id", columns)
        self.assertIn("idx_messages_session", indexes)
        self.assertEqual(row[0], self.session_id)

    def test_tag_messages_only_stamps_null(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "role TEXT NOT NULL, content TEXT NOT NULL, "
            "created_at TEXT NOT NULL, session_id TEXT)")
        conn.execute("INSERT INTO messages(role, content, created_at, "
                     "session_id) VALUES ('user', 'x', '2026', 'other')")
        conn.commit()
        conn.close()
        self.assertTrue(self.repo.tag_session_messages(self.session_id))
        conn = sqlite3.connect(str(self.db_path))
        values = [row[0] for row in
                  conn.execute("SELECT session_id FROM messages")]
        conn.close()
        self.assertEqual(values, ["other"])

    def test_tag_messages_without_table_no_crash(self) -> None:
        # messages может ещё не существовать (файл БД создан репозиторием)
        self.assertFalse(self.repo.tag_session_messages(self.session_id))

    def test_corrupted_json_degrades_to_empty(self) -> None:
        state = self.repo.get_or_create(self.session_id, 1)
        state.goal = "Цель"
        self.repo.update(state)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("UPDATE task_state SET clarifications = '{битый json', "
                     "terms = '[не объект]' WHERE session_id = ?",
                     (self.session_id,))
        conn.commit()
        conn.close()
        loaded = self.repo.get_or_create(self.session_id, 1)
        self.assertEqual(loaded.goal, "Цель")
        self.assertEqual(loaded.clarifications, [])
        self.assertEqual(loaded.terms, {})


if __name__ == "__main__":
    unittest.main()
