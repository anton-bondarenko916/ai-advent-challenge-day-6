"""Тесты UI команды /help: автодополнение и панель документации.

Офлайн, без запуска Flet: контролы — обычные объекты, async-обработчики
зовутся напрямую через asyncio.run (как в test_flet_views).

Проверяются: подсказка при вводе «/» (появляется/прячется, клик
подставляет команду, отправка прячет), компоновка вкладки чата,
панель «Документация проекта» (валидация пути, индексация, сохранение
пути в app_state, применение к HelpPipeline, ошибки), наличие панели
на вкладке «Настройки».

Запуск из корня проекта:
    python -m unittest tests.test_help_ui -v
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import flet as ft

from app.rag.embedder import EmbedderError
from app.rag.project_rag import PROJECT_DOCS_PATH_KEY
from app.ui.chat_view import ChatView
from app.ui.project_docs_view import ProjectDocsPanel
from app.ui.settings_view import SettingsView
from src.app_state import AppStateStore
from tests.test_app_agent import make_agent


def run(coroutine):
    return asyncio.run(coroutine)


def make_app(agent, page=None, **extra) -> SimpleNamespace:
    """Заглушка владельца вкладок (минимум для панелей /help)."""
    base = {"agent": agent, "page": page,
            "notify": lambda message: None}
    base.update(extra)
    return SimpleNamespace(**base)


class FakeProjectRAG:
    """ProjectRAG-заглушка: считает вызовы индексации."""

    collection_name = "project_docs"

    def __init__(self) -> None:
        self.calls: list = []
        self.chunks = 0
        self.error: Exception | None = None

    def index_project(self, path, chunk_size=500, overlap=50) -> dict:
        self.calls.append((path, chunk_size, overlap))
        if self.error is not None:
            raise self.error
        self.chunks += 12
        return {"files": 3, "chunks": 12, "errors": [], "duration": 0.5}

    def count(self) -> int:
        return self.chunks


class FakeHelpPipeline:
    """HelpPipeline-заглушка: путь проекта."""

    def __init__(self, project_path: str = "") -> None:
        self.project_path = project_path


# ----------------------------------------------------------------------
# Автодополнение при вводе «/»
# ----------------------------------------------------------------------
class SuggestionTest(unittest.TestCase):
    """Подсказка команды /help в поле чата."""

    def setUp(self) -> None:
        self.view = ChatView(make_app(make_agent()))

    def test_slash_shows_suggestion(self) -> None:
        self.view.input_field.value = "/"
        self.view._on_input_change()
        self.assertTrue(self.view.suggestions.visible)
        self.assertTrue(self.view.suggestions.controls)

    def test_partial_command_shows_suggestion(self) -> None:
        self.view.input_field.value = "/he"
        self.view._on_input_change()
        self.assertTrue(self.view.suggestions.visible)

    def test_long_text_hides_suggestion(self) -> None:
        self.view.input_field.value = "/help вопрос целиком"
        self.view._on_input_change()
        self.assertFalse(self.view.suggestions.visible)

    def test_plain_text_hides_suggestion(self) -> None:
        self.view.input_field.value = "привет"
        self.view._on_input_change()
        self.assertFalse(self.view.suggestions.visible)

    def test_fill_suggestion_inserts_command(self) -> None:
        self.view.input_field.value = "/"
        self.view._on_input_change()
        self.view._fill_suggestion("/help ")
        self.assertEqual(self.view.input_field.value, "/help ")
        self.assertFalse(self.view.suggestions.visible)

    def test_send_hides_suggestion(self) -> None:
        self.view.input_field.value = "/help как устроена архитектура?"
        self.view.suggestions.visible = True
        run(self.view._send_async())
        self.assertFalse(self.view.suggestions.visible)

    def test_build_contains_suggestions_between_input_and_status(self) -> None:
        controls = self.view.build().controls
        names = [type(control).__name__ for control in controls]
        self.assertIn("Column", names)  # панель подсказок (Column)
        index = controls.index(self.view.suggestions)
        self.assertEqual(type(controls[index - 1]).__name__, "Row")
        self.assertIs(controls[index + 1], self.view.status_text)

    def test_suggestion_text_mentions_help(self) -> None:
        controls = self.view._suggestion_controls()
        self.assertEqual(len(controls), 1)
        self.assertIn("/help", controls[0].content.value)


# ----------------------------------------------------------------------
# Панель «Документация проекта»
# ----------------------------------------------------------------------
class ProjectDocsPanelTest(unittest.TestCase):
    """Панель пути и индексации документации проекта."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.project = self.tmp / "project"
        self.project.mkdir()
        (self.project / "README.md").write_text("# T\n", encoding="utf-8")
        self.rag = FakeProjectRAG()
        self.pipeline = FakeHelpPipeline(str(self.project))
        self.app = make_app(
            make_agent(), project_rag=self.rag, help_pipeline=self.pipeline)
        self.panel = ProjectDocsPanel(self.app)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_build_returns_container(self) -> None:
        self.assertEqual(type(self.panel.build()).__name__, "Container")

    def test_initial_path_from_pipeline(self) -> None:
        self.assertEqual(self.panel.path_field.value, str(self.project))

    def test_initial_path_from_app_state(self) -> None:
        store = AppStateStore(self.tmp / "state.db")
        store.set(PROJECT_DOCS_PATH_KEY, str(self.project))
        app = make_app(make_agent(), app_state=store,
                       project_rag=self.rag, help_pipeline=None)
        panel = ProjectDocsPanel(app)
        self.assertEqual(panel.path_field.value, str(self.project))

    def test_index_success(self) -> None:
        self.panel.path_field.value = str(self.project)
        run(self.panel._index_async())
        self.assertEqual(self.rag.calls, [(str(self.project), 500, 50)])
        self.assertIn("Проиндексировано: 3 файлов, 12 чанков",
                      self.panel.status_text.value)
        self.assertNotEqual(self.panel.status_text.color, ft.Colors.RED_600)
        # Путь применён к /help
        self.assertEqual(self.pipeline.project_path, str(self.project))

    def test_index_saves_path_to_app_state(self) -> None:
        store = AppStateStore(self.tmp / "state.db")
        self.app.app_state = store
        self.panel.path_field.value = str(self.project)
        run(self.panel._index_async())
        self.assertEqual(store.get(PROJECT_DOCS_PATH_KEY), str(self.project))

    def test_index_missing_path_shows_error(self) -> None:
        self.panel.path_field.value = str(self.tmp / "no_such_dir")
        run(self.panel._index_async())
        self.assertIn("не найден", self.panel.status_text.value)
        self.assertEqual(self.rag.calls, [])
        self.assertEqual(self.pipeline.project_path, str(self.project))

    def test_index_embedder_error(self) -> None:
        self.rag.error = EmbedderError("Ollama недоступна")
        self.panel.path_field.value = str(self.project)
        run(self.panel._index_async())
        self.assertIn("Ошибка индексации", self.panel.status_text.value)
        self.assertEqual(self.pipeline.project_path, str(self.project))

    def test_index_empty_path_shows_error(self) -> None:
        self.panel.path_field.value = "   "
        run(self.panel._index_async())
        self.assertIn("не найден", self.panel.status_text.value)
        self.assertEqual(self.rag.calls, [])

    def test_idle_status_mentions_collection(self) -> None:
        self.rag.chunks = 12
        status = self.panel._idle_status()
        self.assertIn("project_docs", status)
        self.assertIn("12", status)


# ----------------------------------------------------------------------
# Интеграция со вкладкой «Настройки»
# ----------------------------------------------------------------------
class SettingsIntegrationTest(unittest.TestCase):
    """Панель документации проекта присутствует в настройках."""

    def test_settings_view_builds_with_panel(self) -> None:
        view = SettingsView(make_app(make_agent()))
        column = view.build()
        self.assertTrue(hasattr(view, "project_docs_panel"))
        # Панель в компоновке и вызывает on_show без ошибок
        view.on_show()
        self.assertIsNotNone(view.project_docs_panel.build())

    def test_settings_view_with_app_state(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        try:
            store = AppStateStore(tmp / "state.db")
            view = SettingsView(make_app(
                make_agent(), app_state=store))
            view.on_show()
            self.assertIsNotNone(view.project_docs_panel)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
