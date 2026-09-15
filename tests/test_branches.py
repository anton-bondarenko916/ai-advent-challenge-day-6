"""Тесты веток диалога (branching): checkpoint, развилки, изоляция.

Проверяют:
- миграцию схемы БД v1 → v2 без потери истории (вся история → ветка main);
- создание ветки от конца диалога (checkpoint) и от выбранного сообщения;
- изоляцию сообщений веток и независимое продолжение диалога;
- переключение веток с восстановлением стратегии и фактов;
- переименование, удаление (лист / с детьми / последняя ветка), лимит веток;
- восстановление активной ветки после «перезапуска» приложения.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

from src.agent import DeepSeekAgent
from src.config import get_strategy_settings
from src.storage import ROOT_BRANCH_ID, ContextStore
from src.tokens import HeuristicCounter
from src.utils import AgentConfigError
from tests import gui_helpers

_TESTS_TMP = Path(__file__).resolve().parent / "tmp_branches"
_COUNTER = HeuristicCounter()

_SETTINGS = {
    "name": "sliding_window",
    "window": 20,
    "branch_strategy": "sliding_window",
    "branch_max": 20,
    "facts_enabled": True,
    "facts_model": "",
    "facts_temperature": 0.0,
    "facts_max_tokens": 300,
    "facts_max_items": 50,
    "facts_prompt": "",
}


def _db_path() -> Path:
    return _TESTS_TMP / f"branch_{uuid.uuid4().hex}.db"


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


def _make_v1_db(path: Path, messages: list) -> None:
    """Создаёт БД СТАРОЙ схемы (версия 1, без веток) для теста миграции."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            """
            CREATE TABLE messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        now = "2026-01-01T00:00:00"
        conn.executemany(
            "INSERT INTO messages(role, content, created_at) VALUES (?, ?, ?)",
            [(role, content, now) for role, content in messages],
        )
        conn.execute(
            "INSERT INTO meta(key, value) VALUES ('schema_version', '1')"
        )
        conn.execute(
            "INSERT INTO meta(key, value) VALUES ('params', ?)",
            (json.dumps({"model": "deepseek-v4-flash",
                         "system_prompt": "Старая система.",
                         "temperature": 0.5, "max_tokens": 300},
                        ensure_ascii=False),),
        )
        conn.commit()
    finally:
        conn.close()


class MigrationTest(unittest.TestCase):
    """Миграция v1 → v2 (ветки) без потери истории."""

    def setUp(self) -> None:
        self.db_path = _db_path()
        self.messages = [
            ("system", "Старая система."),
            ("user", "Вопрос 1"),
            ("assistant", "Ответ 1"),
            ("user", "Вопрос 2"),
            ("assistant", "Ответ 2"),
        ]

    def tearDown(self) -> None:
        shutil.rmtree(_TESTS_TMP, ignore_errors=True)

    def test_migration_creates_main_branch(self) -> None:
        _make_v1_db(self.db_path, self.messages)
        store = ContextStore(db_path=self.db_path, enabled=True,
                             default_strategy="sticky_facts")
        state = store.load_state()
        self.assertIsNotNone(state)
        # История не потеряна
        self.assertEqual(len(state["messages"]), 5)
        self.assertEqual(state["messages"][1]["content"], "Вопрос 1")
        # Появилась корневая ветка, и вся история принадлежит ей
        branches = store.load_branches()
        self.assertEqual(len(branches), 1)
        self.assertEqual(branches[0]["id"], ROOT_BRANCH_ID)
        self.assertEqual(branches[0]["name"], "main")
        self.assertEqual(branches[0]["strategy"], "sticky_facts")
        self.assertEqual(branches[0]["messages"], 5)
        self.assertEqual(store.get_active_branch(), ROOT_BRANCH_ID)
        store.close()

        # Схема обновилась до версии 3, сообщения привязаны к ветке
        conn = sqlite3.connect(str(self.db_path))
        try:
            version = conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()[0]
            branch_ids = [row[0] for row in
                          conn.execute("SELECT DISTINCT branch_id FROM messages")]
            seqs = [row[0] for row in conn.execute("SELECT seq FROM messages")]
        finally:
            conn.close()
        self.assertEqual(version, "3")
        self.assertEqual(branch_ids, [ROOT_BRANCH_ID])
        self.assertEqual(seqs, [0, 1, 2, 3, 4])

    def test_migration_then_agent_restores_history(self) -> None:
        _make_v1_db(self.db_path, self.messages)
        agent = DeepSeekAgent(
            api_key="k", model="deepseek-v4-flash", token_counter=_COUNTER,
            storage=ContextStore(db_path=self.db_path, enabled=True),
            strategy="sliding_window", strategy_settings=dict(_SETTINGS),
        )
        self.assertEqual(len(agent.history), 5)
        self.assertEqual(agent.get_active_branch()["name"], "main")
        agent.close()

    def test_migration_failure_keeps_single_history_mode(self) -> None:
        """Если ветки недоступны (v1-схема), хранилище работает как раньше."""
        _make_v1_db(self.db_path, self.messages)
        store = ContextStore(db_path=self.db_path, enabled=True)
        # Ломаем ALTER: колонка branch_id уже есть, но таблицы branches нет —
        # миграция должна достроить её и не упасть
        with store._session() as conn:
            try:
                conn.execute(
                    "ALTER TABLE messages ADD COLUMN branch_id INTEGER "
                    "NOT NULL DEFAULT 1"
                )
            except sqlite3.Error:
                pass
        state = store.load_state()
        self.assertIsNotNone(state)
        self.assertEqual(len(state["messages"]), 5)
        store.close()


class BranchManagerTest(unittest.TestCase):
    """Ветки в памяти (без хранилища) и в БД."""

    def setUp(self) -> None:
        self.db_path = _db_path()

    def tearDown(self) -> None:
        shutil.rmtree(_TESTS_TMP, ignore_errors=True)

    def _agent(self, storage=None, strategy: str = "sliding_window",
               window: int = 20, branch_max: int = 20) -> DeepSeekAgent:
        settings = dict(_SETTINGS)
        settings.update({"name": strategy, "branch_strategy": strategy,
                         "window": window, "branch_max": branch_max})
        agent = DeepSeekAgent(
            api_key="k", model="deepseek-v4-flash",
            system_prompt="Система.", token_counter=_COUNTER,
            storage=storage, strategy=strategy, strategy_settings=settings,
        )
        agent._call_api = lambda: _chat_payload()
        agent._call_service_api = lambda request: _facts_payload("{}")
        return agent

    def test_checkpoint_creates_branch_copy(self) -> None:
        agent = self._agent()
        agent.send_message("Вопрос 1")
        result = agent.checkpoint(name="ветка-A")
        self.assertIsNone(result["error"])
        branches = agent.list_branches()
        self.assertEqual(len(branches), 2)
        self.assertEqual(result["branch"]["name"], "ветка-A")
        self.assertEqual(len(agent.history), 3)
        self.assertEqual(result["branch"]["forked_at"], 2)
        agent.close()

    def test_fork_from_selected_message(self) -> None:
        agent = self._agent()
        agent.send_message("Вопрос 1")
        agent.send_message("Вопрос 2")
        self.assertEqual(len(agent.history), 5)
        # Ответвляемся от первого сообщения: в новой ветке только оно
        result = agent.create_branch(from_index=1, name="ветка-от-первого")
        self.assertIsNone(result["error"])
        self.assertEqual(len(agent.history), 2)
        self.assertEqual(agent.history[0]["role"], "system")
        self.assertEqual(agent.history[1]["content"], "Вопрос 1")
        agent.close()

    def test_branches_are_isolated(self) -> None:
        agent = self._agent()
        agent.send_message("Общий вопрос")
        agent.checkpoint(name="ветка-A")
        agent.send_message("Только в ветке A")
        agent.switch_branch(1)              # корневая ветка main
        self.assertEqual(len(agent.history), 3)
        agent.send_message("Только в main")
        contents_main = [m["content"] for m in agent.history]

        branch_a = [b for b in agent.list_branches() if b["name"] == "ветка-A"][0]
        agent.switch_branch(branch_a["id"])
        contents_a = [m["content"] for m in agent.history]

        self.assertIn("Только в ветке A", contents_a)
        self.assertNotIn("Только в main", contents_a)
        self.assertIn("Только в main", contents_main)
        self.assertNotIn("Только в ветке A", contents_main)
        agent.close()

    def test_switch_restores_strategy_and_facts(self) -> None:
        agent = self._agent(strategy="sticky_facts")
        agent._call_service_api = lambda request: _facts_payload(
            '{"preferences": {"имя": "Алекс"}}'
        )
        agent.send_message("Меня зовут Алекс")
        self.assertEqual(agent.get_facts()["count"], 1)
        agent.checkpoint(name="ветка-A")
        branch_a = [b for b in agent.list_branches() if b["name"] == "ветка-A"][0]

        # В корневой ветке — свои факты
        agent.switch_branch(1)
        self.assertEqual(agent.get_facts()["count"], 1)
        agent.remember("имя", "Борис", "preferences")

        # Переключение возвращает артефакты и стратегию ветки
        agent.switch_branch(branch_a["id"])
        self.assertEqual(agent.get_facts()["count"], 1)
        self.assertEqual(agent.get_facts()["facts"]["preferences"]["имя"], "Алекс")
        self.assertEqual(agent.get_strategy()["sub_strategy"], "sticky_facts")

        agent.switch_branch(1)
        self.assertEqual(agent.get_facts()["facts"]["preferences"]["имя"], "Борис")
        agent.close()

    def test_facts_are_per_branch_in_db(self) -> None:
        store = ContextStore(db_path=self.db_path, enabled=True)
        agent = self._agent(storage=store)
        agent.remember("имя", "Алекс", "preferences")
        agent.checkpoint(name="ветка-A")
        branch_a = [b for b in agent.list_branches() if b["name"] == "ветка-A"][0]

        agent.switch_branch(1)
        agent.clear_facts()
        agent.remember("имя", "Борис", "preferences")

        branches = {b["name"]: b for b in agent.list_branches()}
        self.assertEqual(branches["ветка-A"]["facts_count"], 1)
        self.assertEqual(branches["main"]["facts_count"], 1)

        agent.switch_branch(branch_a["id"])
        self.assertEqual(agent.get_facts()["facts"]["preferences"]["имя"], "Алекс")
        agent.switch_branch(1)
        self.assertEqual(agent.get_facts()["facts"]["preferences"]["имя"], "Борис")
        agent.close()

    def test_rename_branch(self) -> None:
        agent = self._agent()
        agent.send_message("Вопрос")
        agent.checkpoint(name="ветка-A")
        branch_id = agent.get_active_branch()["id"]
        result = agent.rename_branch(branch_id, "Переименованная")
        self.assertIsNone(result["error"])
        self.assertEqual(result["branch"]["name"], "Переименованная")
        self.assertIn("Переименованная",
                      [b["name"] for b in agent.list_branches()])
        self.assertIsNotNone(agent.rename_branch(branch_id, "")["error"])
        agent.close()

    def test_delete_leaf_branch(self) -> None:
        agent = self._agent()
        agent.send_message("Общий вопрос")
        agent.checkpoint(name="ветка-A")
        branch_a = [b for b in agent.list_branches() if b["name"] == "ветка-A"][0]
        result = agent.delete_branch(branch_a["id"])
        self.assertIsNone(result["error"])
        self.assertEqual(len(agent.list_branches()), 1)
        # Удаление активной ветки переключает на родителя
        self.assertEqual(agent.get_active_branch()["name"], "main")
        agent.close()

    def test_delete_parent_without_cascade_fails(self) -> None:
        agent = self._agent()
        agent.send_message("Общий вопрос")
        agent.checkpoint(name="ветка-A")
        agent.send_message("Продолжение A")
        agent.checkpoint(name="ветка-B")
        branch_a = [b for b in agent.list_branches() if b["name"] == "ветка-A"][0]
        result = agent.delete_branch(branch_a["id"])
        self.assertIsNotNone(result["error"])
        self.assertIn("дочерние", result["error"])
        # С каскадом — удаляется вместе с потомком
        result = agent.delete_branch(branch_a["id"], cascade=True)
        self.assertIsNone(result["error"])
        self.assertEqual(len(agent.list_branches()), 1)
        agent.close()

    def test_cannot_delete_last_branch(self) -> None:
        agent = self._agent()
        agent.send_message("Вопрос")
        result = agent.delete_branch(ROOT_BRANCH_ID)
        self.assertIsNotNone(result["error"])
        self.assertIn("последнюю", result["error"])
        agent.close()

    def test_branch_limit(self) -> None:
        agent = self._agent(branch_max=2)
        agent.send_message("Вопрос")
        self.assertIsNone(agent.checkpoint()["error"])
        result = agent.checkpoint()
        self.assertIsNotNone(result["error"])
        self.assertIn("лимит", result["error"].lower())
        agent.close()

    def test_unknown_branch_switch(self) -> None:
        agent = self._agent()
        self.assertIsNotNone(agent.switch_branch(999)["error"])
        agent.close()

    def test_active_branch_restored_after_restart(self) -> None:
        store = ContextStore(db_path=self.db_path, enabled=True)
        agent = self._agent(storage=store)
        agent.send_message("Вопрос 1")
        agent.checkpoint(name="ветка-A")
        agent.send_message("Только в ветке A")
        agent.close()

        restored = self._agent(storage=ContextStore(db_path=self.db_path,
                                                    enabled=True))
        self.assertEqual(restored.get_active_branch()["name"], "ветка-A")
        contents = [m["content"] for m in restored.history]
        self.assertIn("Только в ветке A", contents)
        self.assertEqual(len(restored.list_branches()), 2)
        restored.close()

    def test_branch_strategy_is_remembered(self) -> None:
        agent = self._agent()
        agent.send_message("Вопрос")
        agent.checkpoint(name="ветка-A")
        branch_a = [b for b in agent.list_branches() if b["name"] == "ветка-A"][0]
        # В корневой ветке — своя стратегия, в ветке-A свой набор фактов
        agent.switch_branch(1)
        agent.set_strategy("sticky_facts")
        agent.switch_branch(branch_a["id"])
        self.assertEqual(agent.get_strategy()["sub_strategy"], "sliding_window")
        agent.switch_branch(1)
        self.assertEqual(agent.get_strategy()["sub_strategy"], "sticky_facts")
        agent.close()


class ManagerUnitTest(unittest.TestCase):
    """Прямые тесты BranchManager (без агента)."""

    def tearDown(self) -> None:
        shutil.rmtree(_TESTS_TMP, ignore_errors=True)

    def test_in_memory_manager_without_store(self) -> None:
        from src.branches import BranchManager

        manager = BranchManager(store=None, max_branches=3)
        manager.load()
        self.assertEqual(manager.count(), 1)
        self.assertEqual(manager.get_active()["name"], "main")

        history = [{"role": "system", "content": "S"},
                   {"role": "user", "content": "U"},
                   {"role": "assistant", "content": "A"}]
        created = manager.create(history, {}, from_index=None, name="копия")
        self.assertEqual(created["branch"]["name"], "копия")
        self.assertEqual(len(created["messages"]), 3)
        self.assertEqual(manager.count(), 2)

        manager.create(history, {}, name="копия-2")
        self.assertEqual(manager.count(), 3)
        with self.assertRaises(AgentConfigError):
            manager.create(history, {}, name="копия-3")  # лимит веток

    def test_no_store_last_branch_protection(self) -> None:
        from src.branches import BranchManager

        manager = BranchManager(store=None)
        manager.load()
        with self.assertRaises(AgentConfigError):
            manager.delete(manager.active_id())
        with self.assertRaises(AgentConfigError):
            manager.switch(999)

    def test_manager_over_store(self) -> None:
        from src.branches import BranchManager

        store = ContextStore(db_path=_db_path(), enabled=True,
                             default_strategy="sliding_window")
        manager = BranchManager(store=store, max_branches=5)
        manager.load()
        self.assertEqual(len(manager.list_branches()), 1)
        self.assertEqual(manager.list_branches()[0]["name"], "main")

        manager.create(
            [{"role": "system", "content": "S"},
             {"role": "user", "content": "U"}],
            {}, name="ветка-2", strategy="sticky_facts",
        )
        branches = manager.list_branches()
        self.assertEqual(len(branches), 2)
        names = {b["name"]: b for b in branches}
        self.assertEqual(names["ветка-2"]["parent_id"], ROOT_BRANCH_ID)
        self.assertEqual(names["ветка-2"]["strategy"], "sticky_facts")
        self.assertEqual(names["ветка-2"]["messages"], 2)
        self.assertTrue(names["ветка-2"]["active"])

        # Переключение и чтение сообщений из БД
        manager.switch(ROOT_BRANCH_ID)
        self.assertEqual(manager.load_messages(ROOT_BRANCH_ID), [])
        self.assertEqual(len(manager.load_messages(names["ветка-2"]["id"])), 2)


class BranchGuiTest(unittest.TestCase):
    """Окно «Ветки» и ветка от сообщения (пропускается без дисплея)."""

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
        settings = get_strategy_settings({"STRATEGY": "branching",
                                          "BRANCH_MAX": 5})
        self.agent = DeepSeekAgent(
            api_key="", mock_mode=True, token_counter=HeuristicCounter(),
            strategy="branching", strategy_settings=settings,
        )
        self.app = self._ChatApplication(root=self.root, agent=self.agent)
        self.root.update()

    def tearDown(self) -> None:
        for closer in ("_close_branches_dialog", "_close_context_dialog"):
            method = getattr(self.app, closer, None)
            if callable(method):
                try:
                    method()
                except Exception:  # закрытие окна не должно ронять тест
                    pass
        self.agent.close()

    def _pump(self, timeout: float = 5.0) -> None:
        """Дожидается завершения фоновой операции интерфейса."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.root.update()
            try:
                self.app._poll_action_queue()
            except Exception:
                pass
            if not self.app._action_busy:
                break
            time.sleep(0.02)
        self.root.update()

    def _branch_names(self) -> list:
        return [
            str(self.app.branches_tree.item(row, "values")[0])
            for row in self.app.branches_tree.get_children()
        ]

    def test_branching_mode_selected_in_combobox(self) -> None:
        self.app.strategy_combo.set("branching")
        self.app._on_strategy_change()
        self.root.update()
        state = self.agent.get_strategy()
        self.assertEqual(state["name"], "branching")
        self.assertTrue(state["branching"])

    def test_branches_window_checkpoint_and_switch(self) -> None:
        self.agent.send_message("Первый вопрос")
        self.app._refresh_chat()
        self.root.update()

        self.app._open_branches_dialog()
        self.root.update()
        self.assertIsNotNone(self.app._branches_window)
        self.assertEqual(len(self.app.branches_tree.get_children()), 1)
        self.assertEqual(self.agent.get_active_branch()["name"], "main")

        with mock.patch("tkinter.simpledialog.askstring",
                        return_value="ветка-A"):
            self.app._on_branches_checkpoint()
            self._pump()

        names = self._branch_names()
        self.assertEqual(len(names), 2)
        self.assertIn("ветка-A", names)
        self.assertEqual(self.agent.get_active_branch()["name"], "ветка-A")
        # Активная ветка выделена в таблице
        active_rows = [row for row in self.app.branches_tree.get_children()
                       if "active" in self.app.branches_tree.item(row, "tags")]
        self.assertEqual(len(active_rows), 1)

        # Переключение обратно на корневую ветку через таблицу
        main_row = [row for row in self.app.branches_tree.get_children()
                    if str(self.app.branches_tree.item(row, "values")[0])
                    == "main"][0]
        self.app.branches_tree.selection_set(main_row)
        self.app._on_branches_switch()
        self._pump()
        self.assertEqual(self.agent.get_active_branch()["id"], ROOT_BRANCH_ID)

    def test_branch_from_selected_message(self) -> None:
        self.agent.send_message("Первый вопрос")
        self.agent.send_message("Второй вопрос")
        self.app._refresh_chat()
        self.root.update()

        # Теги msg_<i> привязаны к сообщениям диалога (без system)
        ranges = self.app.chat.tag_ranges("msg_0")
        self.assertTrue(ranges)
        self.assertEqual(self.app._message_index_at(str(ranges[0])), 0)
        self.assertTrue(self.app.chat.tag_ranges("msg_2"))

        # Правый клик по сообщению делает его выбранным
        self.app._chat_selected_index = 2
        self.assertEqual(self.app._selected_chat_index(), 2)

        # В контекстном меню чата есть пункт создания ветки
        labels = [
            str(self.app.chat_menu.entrycget(index, "label"))
            for index in range(self.app.chat_menu.index("end") + 1)
        ]
        self.assertTrue(any("Создать ветку отсюда" in label for label in labels))

        with mock.patch("tkinter.simpledialog.askstring",
                        return_value="от-второго"):
            self.app._on_create_branch_from_chat()
            self._pump()

        self.assertEqual(self.agent.get_active_branch()["name"], "от-второго")
        contents = [str(message.get("content") or "")
                    for message in self.agent.history]
        # Ветка получила префикс диалога до выбранного сообщения
        self.assertTrue(any("Первый вопрос" in text for text in contents))
        self.assertFalse(any("Второй вопрос" in text for text in contents))


if __name__ == "__main__":
    unittest.main()
