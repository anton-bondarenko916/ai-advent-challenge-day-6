"""Тесты вкладки «🎓 Обучение» (app/ui/learning_view.py) — офлайн.

Контролы Flet 1.0 — обычные объекты: вкладка собирается с заглушкой
app (без page); сервисы — настоящие (LearningRepository + PlanService/
ReviewService/PDFService на tmp-каталоге, LLM — MockJsonLLM, fs —
фейковый клиент, PDF — фейковый renderer).

Проверяются: экран входа (успех/ошибка), дашборд (5 тем, streak),
ленивое открытие темы, экран подтемы, проверка задания (фейковый fs +
MockJsonLLM), отметка «выполнено», PDF-кнопка, навигация «назад»,
передача КОРУТИННОЙ ФУНКЦИИ в page.run_task (регрессия ТЗ ai_1_21).

Запуск из корня проекта:
    python -m unittest tests.test_learning_ui -v
"""

from __future__ import annotations

import asyncio
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import flet as ft

from app.learning.db import LearningRepository
from app.learning.llm import MockJsonLLM
from app.learning.pdf import PDFService
from app.learning.plan import PlanService, TOPIC_TITLES
from app.learning.review import ReviewService
from app.ui.learning_view import LEARNING_PROJECT_PATH_KEY, LearningView, \
    topic_badge

TREE = {"app": {"src": {"MainActivity.kt": "file"}},
        "README.md": "file"}


def screen_texts(screen) -> list:
    """Все тексты дерева контролов: Text.value, Markdown.value и
    подписи кнопок (у кнопок Flet 1.0 label — строка в content)."""
    def walk(control) -> list:
        if isinstance(control, str):
            return [control] if control.strip() else []
        texts = []
        value = getattr(control, "value", None)
        if isinstance(value, str) and value.strip():
            texts.append(value)
        for child in getattr(control, "controls", None) or []:
            texts.extend(walk(child))
        content = getattr(control, "content", None)
        if content is not None:
            texts.extend(walk(content))
        return texts
    return walk(screen)


def run(coroutine):
    return asyncio.run(coroutine)


class FakeFsClient:
    """fs-клиент-заглушка: дерево + содержимое проекта."""

    def __init__(self) -> None:
        self.calls: list = []
        self.closed = False

    async def call_tool(self, tool_name, arguments=None):
        self.calls.append((tool_name, dict(arguments or {})))
        if tool_name == "tree":
            return json.dumps({"result": TREE})
        if tool_name == "read_multiple_files":
            payload = {path: f"код {path}"
                       for path in arguments["paths"]}
            return json.dumps({"result": payload})
        raise AssertionError(tool_name)

    async def close(self) -> None:
        self.closed = True


class FakeRenderer:
    """PDF-renderer-заглушка: создаёт файл."""

    def __call__(self, html: str, target: Path) -> None:
        target.write_bytes(b"%PDF-fake")


class FakePage:
    """Страница-заглушка: запоминает задачи page.run_task."""

    def __init__(self) -> None:
        self.tasks: list = []
        self.updates = 0

    def run_task(self, handler, *args) -> None:
        self.tasks.append((handler, args))

    def update(self) -> None:
        self.updates += 1


def make_app(tmp: Path, page=None, notes: list | None = None):
    """Заглушка ChatApp с НАСТОЯЩИМИ сервисами обучения (tmp-БД)."""
    notes = notes if notes is not None else []
    repo = LearningRepository(tmp / "learning.db")
    llm = MockJsonLLM()
    fs = FakeFsClient()
    return SimpleNamespace(
        page=page,
        notify=notes.append,
        learning_repo=repo,
        learning_plan=PlanService(repo, llm),
        learning_review=ReviewService(repo, llm,
                                      fs_provider=lambda path: fs),
        learning_pdf=PDFService(output_dir=tmp / "pdf",
                                renderer=FakeRenderer()),
        app_state=None,
        file_picker=None,
    ), notes, fs


class LoginFlowTest(unittest.TestCase):
    """Экран входа: успех, ошибка, выход."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.app, self.notes, _ = make_app(self.tmp)
        self.view = LearningView(self.app)
        self.view.build()

    def _login(self, username: str, password: str) -> None:
        self.view.username_field.value = username
        self.view.password_field.value = password
        run(self.view._login_async())

    def test_login_success_shows_dashboard(self) -> None:
        self._login("ivan", "ivan123")
        self.assertIsNotNone(self.view.user)
        self.assertEqual(self.view.user["level"], "beginner")
        texts = self._screen_texts()
        self.assertTrue(any("Привет, Иван" in text for text in texts))
        self.assertTrue(any("Streak" in text for text in texts))
        # 5 тем курса на дашборде (кнопки с заголовками тем)
        topic_buttons = sum(
            1 for text in texts
            if any(title in text for title in TOPIC_TITLES))
        self.assertEqual(topic_buttons, len(TOPIC_TITLES))

    def test_login_wrong_password_shows_error(self) -> None:
        self._login("ivan", "неверный")
        self.assertIsNone(self.view.user)
        self.assertIn("Неверный логин или пароль",
                      self.view.login_error.value)

    def test_logout_returns_to_login(self) -> None:
        self._login("maria", "maria123")
        self.view._logout()
        self.assertIsNone(self.view.user)
        texts = self._screen_texts()
        self.assertTrue(any("Android Learning Assistant" in text
                            for text in texts))

    def test_empty_repo_shows_unavailable(self) -> None:
        app, _, _ = make_app(self.tmp)
        app.learning_repo = None
        view = LearningView(app)
        view.build()
        view.username_field.value = "ivan"
        view.password_field.value = "ivan123"
        run(view._login_async())
        self.assertIn("недоступен", view.login_error.value)

    def _screen_texts(self) -> list:
        return screen_texts(self.view.screen)


class RunTaskCoroutineFunctionTest(unittest.TestCase):
    """Регрессия ТЗ ai_1_21: в page.run_task — корутинная ФУНКЦИЯ."""

    def test_handlers_pass_coroutine_function(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        page = FakePage()
        app, _, _ = make_app(tmp, page=page)
        view = LearningView(app)
        view.build()
        cases = [view._on_login, view._on_review, view._on_pdf,
                 view._on_refresh, view._on_toggle_completed,
                 view._on_pick_folder]
        for handler in cases:
            page.tasks.clear()
            handler()
            self.assertEqual(len(page.tasks), 1)
            task_handler, args = page.tasks[0]
            self.assertTrue(
                inspect.iscoroutinefunction(task_handler),
                f"{task_handler} — не корутинная функция")
            self.assertFalse(inspect.iscoroutine(task_handler))

    def test_no_page_is_noop(self) -> None:
        """Без page (офлайн-тесты) обработчики не создают задач."""
        tmp = Path(tempfile.mkdtemp())
        app, _, _ = make_app(tmp, page=None)
        view = LearningView(app)
        view.build()
        view._on_login()  # не бросает
        view._on_pdf()


class TopicAndSubtopicFlowTest(unittest.TestCase):
    """Навигация: дашборд → тема (ленивая генерация) → подтема."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "project").mkdir()  # папка «проекта» для проверки
        self.app, self.notes, self.fs = make_app(self.tmp)
        self.view = LearningView(self.app)
        self.view.build()
        self.view.username_field.value = "alex"
        self.view.password_field.value = "alex123"
        run(self.view._login_async())

    def test_open_topic_generates_lazily(self) -> None:
        topics = self.app.learning_plan.ensure_topics(self.view.user["id"])
        topic_id = topics[0]["id"]
        run(self.view._open_topic_async(topic_id))
        texts = self._texts()
        self.assertTrue(any("К списку тем" in text for text in texts))
        self.assertTrue(any("PDF-конспект" in text for text in texts))
        # Теория подтем — на экране
        self.assertTrue(any("Основы" in text for text in texts))
        # Повторное открытие — из БД (MockJsonLLM — детерминированный)
        run(self.view._open_topic_async(topic_id))
        self.assertTrue(any("закрепление" in text
                            for text in self._texts()))

    def test_open_unknown_topic_notifies(self) -> None:
        run(self.view._open_topic_async(99999))
        self.assertTrue(any("Не удалось открыть тему" in note
                            for note in self.notes))

    def test_subtopic_review_flow(self) -> None:
        topics = self.app.learning_plan.ensure_topics(self.view.user["id"])
        run(self.view._open_topic_async(topics[0]["id"]))
        subtopic_id = self.app.learning_repo.get_topic(
            self.view.user["id"], topics[0]["id"])["subtopics"][0]["id"]
        run(self.view._open_subtopic_async(subtopic_id))
        texts = self._texts()
        self.assertTrue(any("Теория" in text for text in texts))
        self.assertTrue(any("Задание" in text for text in texts))
        self.assertTrue(any("Проверка задания" in text for text in texts))

        # Пустой путь — сообщение об ошибке, fs не вызывается
        self.view._path_field.value = ""
        run(self.view._review_async())
        self.assertIn("Укажите путь", self.view.status_text.value)
        self.assertEqual(self.fs.calls, [])

        # Проверка: fs MCP (дерево + чтение) → фидбек на экране
        self.view._path_field.value = str(self.tmp / "project")
        run(self.view._review_async())
        tools = [name for name, _ in self.fs.calls]
        self.assertEqual(tools, ["tree", "read_multiple_files"])
        self.assertTrue(self.fs.closed)
        self.assertTrue(any("Фидбек AI" in text
                            for text in self._texts()))
        self.assertTrue(any("Оценка: 5/5" in note for note in self.notes))
        # Оценка ≥ 4 — подтема завершена, прогресс сохранён
        subtopic = self.app.learning_repo.get_subtopic_for_user(
            self.view.user["id"], subtopic_id)
        self.assertTrue(subtopic["is_completed"])

    def test_mark_completed_toggle(self) -> None:
        topics = self.app.learning_plan.ensure_topics(self.view.user["id"])
        run(self.view._open_topic_async(topics[0]["id"]))
        subtopic_id = self.app.learning_repo.get_topic(
            self.view.user["id"], topics[0]["id"])["subtopics"][0]["id"]
        run(self.view._open_subtopic_async(subtopic_id))
        run(self.view._toggle_completed_async())
        subtopic = self.app.learning_repo.get_subtopic_for_user(
            self.view.user["id"], subtopic_id)
        self.assertTrue(subtopic["is_completed"])
        # Возврат на экран темы после отметки
        self.assertTrue(any("К списку тем" in text
                            for text in self._texts()))

    def test_pdf_button(self) -> None:
        topics = self.app.learning_plan.ensure_topics(self.view.user["id"])
        run(self.view._open_topic_async(topics[0]["id"]))
        run(self.view._pdf_async())
        self.assertIn("PDF", self.view.status_text.value)
        self.assertTrue(any("PDF создан" in note for note in self.notes))

    def _texts(self) -> list:
        return screen_texts(self.view.screen)


class SubtopicContrastTest(unittest.TestCase):
    """Контраст блоков подтемы (регрессия).

    Контейнеры задания и фидбека имеют ФИКСИРОВАННО светлый фон
    (ORANGE_50/GREEN_50): если текст внутри берёт цвет темы, в тёмной
    теме он светлый и нечитаемый («светлое на светлом», урок ai_1_21).
    Заголовки, тело Markdown и рубрика обязаны иметь явный тёмный цвет.
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "project").mkdir()
        self.app, self.notes, self.fs = make_app(self.tmp)
        self.view = LearningView(self.app)
        self.view.build()
        self.view.username_field.value = "ivan"
        self.view.password_field.value = "ivan123"
        run(self.view._login_async())
        topics = self.app.learning_plan.ensure_topics(
            self.view.user["id"])
        run(self.view._open_topic_async(topics[0]["id"]))
        subtopic_id = self.app.learning_repo.get_topic(
            self.view.user["id"], topics[0]["id"])["subtopics"][0]["id"]
        run(self.view._open_subtopic_async(subtopic_id))
        self.view._path_field.value = str(self.tmp / "project")
        run(self.view._review_async())  # фидбек AI появляется на экране

    def _containers(self, bgcolor) -> list:
        def walk(control) -> list:
            found = []
            if getattr(control, "bgcolor", None) == bgcolor:
                found.append(control)
            for child in getattr(control, "controls", None) or []:
                found.extend(walk(child))
            content = getattr(control, "content", None)
            if content is not None:
                found.extend(walk(content))
            return found
        return walk(self.view.screen)

    def test_task_block_has_explicit_dark_text(self) -> None:
        (container,) = self._containers(ft.Colors.ORANGE_50)
        header, markdown, rubric = container.content.controls
        self.assertEqual(header.value, "📋 Задание")
        self.assertIsNotNone(header.color)
        self.assertEqual(markdown.md_style_sheet.p_text_style.color,
                         ft.Colors.GREY_900)
        self.assertIsNotNone(rubric.color)

    def test_feedback_block_has_explicit_dark_text(self) -> None:
        (container,) = self._containers(ft.Colors.GREEN_50)
        header, markdown = container.content.controls
        self.assertIn("Фидбек AI", header.value)
        self.assertIsNotNone(header.color)
        self.assertEqual(markdown.md_style_sheet.p_text_style.color,
                         ft.Colors.GREY_900)


class TopicBadgeTest(unittest.TestCase):
    """Строки статуса тем для карточек дашборда."""

    def test_badges(self) -> None:
        self.assertIn("не сгенерирована", topic_badge({
            "display_status": "not_generated",
            "subtopics_total": 0, "subtopics_completed": 0}))
        self.assertIn("(2/4)", topic_badge({
            "display_status": "in_progress",
            "subtopics_total": 4, "subtopics_completed": 2}))
        self.assertIn("завершена", topic_badge({
            "display_status": "completed",
            "subtopics_total": 3, "subtopics_completed": 3}))

    def test_saved_project_path_key(self) -> None:
        self.assertEqual(LEARNING_PROJECT_PATH_KEY, "learning_project_path")


if __name__ == "__main__":
    unittest.main()
