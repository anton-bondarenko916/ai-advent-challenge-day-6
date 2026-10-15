"""Тесты UI команды /support (ТЗ ai_1_34).

Проверяются: автодополнение при вводе «/» предлагает /support и /help,
выпадающий список пользователей CRM (заполнение из list_users, выбор
меняет user_id для /support, недоступный CRM — список пуст), парсер
списка пользователей, панель индексации FAQ в Настройках.

Запуск из корня проекта:
    python -m unittest tests.test_support_ui -v
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.rag.support_rag import SUPPORT_FAQ_COLLECTION, SupportRAG
from app.ui.chat_view import ChatView, parse_crm_users
from app.ui.support_docs_view import SupportDocsPanel
from tests.test_app_agent import make_agent
from tests.test_support_rag import KeywordEmbedder
from app.rag.vector_store import VectorStore


def run(coroutine):
    return asyncio.run(coroutine)


def make_app(agent, page=None, **extra) -> SimpleNamespace:
    """Заглушка владельца вкладок (как ChatViewTest в test_flet_views)."""
    base = {"agent": agent, "page": page, "file_picker": None,
            "notify": lambda message: None}
    base.update(extra)
    return SimpleNamespace(**base)


class ParseCrmUsersTest(unittest.TestCase):
    """Список пользователей из ответа MCP (list_users CRM)."""

    def test_unwraps_structured_result(self) -> None:
        raw = json.dumps({"result": [
            {"id": 42, "name": "Иван Петров"},
            {"id": 43, "name": "Мария Сидорова"},
        ]}, ensure_ascii=False)
        users = parse_crm_users(raw)
        self.assertEqual([u["id"] for u in users], [42, 43])

    def test_plain_list(self) -> None:
        raw = json.dumps([{"id": 1, "name": "A"}])
        self.assertEqual(len(parse_crm_users(raw)), 1)

    def test_garbage_is_empty(self) -> None:
        self.assertEqual(parse_crm_users("не json"), [])
        self.assertEqual(parse_crm_users(""), [])
        self.assertEqual(parse_crm_users(json.dumps({"a": 1})), [])

    def test_entries_without_id_dropped(self) -> None:
        raw = json.dumps([{"id": 1, "name": "A"}, {"name": "без id"}])
        self.assertEqual(len(parse_crm_users(raw)), 1)


class SupportAutocompleteTest(unittest.TestCase):
    """Автодополнение при вводе «/»: /support и /help (ТЗ)."""

    def setUp(self) -> None:
        self.app = make_app(make_agent())
        self.view = ChatView(self.app)

    def test_suggestions_include_support_and_help(self) -> None:
        labels = [
            str(control.content.value)
            for control in self.view._suggestion_controls()
        ]
        self.assertTrue(any("/support" in label for label in labels))
        self.assertTrue(any("/help" in label for label in labels))

    def test_slash_input_shows_suggestions(self) -> None:
        self.view.input_field.value = "/sup"
        self.view._on_input_change()
        self.assertTrue(self.view.suggestions.visible)

    def test_plain_input_hides_suggestions(self) -> None:
        self.view.input_field.value = "обычный вопрос"
        self.view._on_input_change()
        self.assertFalse(self.view.suggestions.visible)

    def test_fill_support_suggestion(self) -> None:
        self.view._fill_suggestion("/support ")
        self.assertEqual(self.view.input_field.value, "/support ")
        self.assertFalse(self.view.suggestions.visible)


class SupportUserDropdownTest(unittest.TestCase):
    """Выпадающий список пользователей CRM (user_id НЕ хардкодится)."""

    def setUp(self) -> None:
        self.app = make_app(make_agent(), support_user_id=0,
                            support_client=None)
        self.view = ChatView(self.app)

    def test_dropdown_present_in_header(self) -> None:
        self.assertIn(self.view.support_user_dropdown,
                      self.view.chat_actions_row.controls)

    def test_selection_sets_app_user_id(self) -> None:
        """ТЗ: смена пользователя в списке меняет контекст /support."""
        self.view.support_user_dropdown.value = "43"
        self.view._on_support_user_change()
        self.assertEqual(self.app.support_user_id, 43)

    def test_selection_invalid_value_is_zero(self) -> None:
        self.view.support_user_dropdown.value = None
        self.view._on_support_user_change()
        self.assertEqual(self.app.support_user_id, 0)

    def test_refresh_fills_options_from_crm(self) -> None:
        class FakeCRM:
            async def call_tool(self, tool_name, arguments=None):
                return json.dumps({"result": [
                    {"id": 42, "name": "Иван Петров"},
                    {"id": 43, "name": "Мария Сидорова"},
                ]}, ensure_ascii=False)

        self.app.support_client = FakeCRM()
        run(self.view.refresh_support_users_async())
        self.assertEqual(len(self.view.support_user_dropdown.options), 2)
        # Первый пользователь выбирается по умолчанию
        self.assertEqual(self.app.support_user_id, 42)
        self.assertEqual(self.view.support_user_dropdown.value, "42")

    def test_refresh_without_client_is_noop(self) -> None:
        self.app.support_client = None
        run(self.view.refresh_support_users_async())
        self.assertEqual(self.view.support_user_dropdown.options, [])
        self.assertEqual(self.app.support_user_id, 0)

    def test_refresh_with_unavailable_crm_disables_dropdown(self) -> None:
        class DeadCRM:
            async def call_tool(self, tool_name, arguments=None):
                raise ConnectionError("CRM-сервер не запущен")

        self.app.support_client = DeadCRM()
        run(self.view.refresh_support_users_async())
        self.assertEqual(self.view.support_user_dropdown.options, [])
        self.assertTrue(self.view.support_user_dropdown.disabled)
        # Ход /support при этом не ломается (фолбэк в пайплайне)
        self.assertEqual(self.app.support_user_id, 0)


class SupportDocsPanelTest(unittest.TestCase):
    """Панель индексации FAQ (Настройки): индексация в support_faq."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.docs = self.tmp / "support_docs"
        self.docs.mkdir()
        (self.docs / "faq_authorization.md").write_text(
            "# FAQ: Авторизация\n\n## Не работает авторизация\n\n"
            "Вход блокируется для статуса pending_verification.\n",
            encoding="utf-8")
        self.store = VectorStore(persist_dir=self.tmp / "db")
        self.rag = SupportRAG(self.store, KeywordEmbedder())
        self.app = make_app(make_agent(), support_rag=self.rag)
        self.panel = SupportDocsPanel(self.app)
        self.panel.path_field.value = str(self.docs)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_index_async_indexes_collection(self) -> None:
        run(self.panel._index_async())
        self.assertGreater(self.rag.count(), 0)
        self.assertIn("Проиндексировано", self.panel.status_text.value)

    def test_index_async_missing_path_error(self) -> None:
        self.panel.path_field.value = str(self.tmp / "nope")
        run(self.panel._index_async())
        self.assertIn("не найден", self.panel.status_text.value)
        self.assertEqual(self.rag.count(), 0)

    def test_idle_status_mentions_collection(self) -> None:
        status = self.panel._idle_status()
        self.assertIn(SUPPORT_FAQ_COLLECTION, status)


if __name__ == "__main__":
    unittest.main()
