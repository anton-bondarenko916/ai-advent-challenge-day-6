"""Тесты платформы обучения (ТЗ ai_1_36).

Проверяются (нумерация — из ТЗ «ТЕСТЫ ДЛЯ ПРОВЕРКИ»):

1. авторизация: ivan/ivan123 входит, неверный пароль — ошибка;
   3 мок-пользователя созданы, пароли в БД только хешами;
2. план: тема #1 при первом входе генерируется LLM и сохраняется в БД;
3. повторный вход: тема #1 загружается из БД, LLM НЕ вызывается;
4. проверка задания: папка → fs MCP (дерево → выбор файлов → чтение)
   → LLM-фидбек с рубрикой, оценка ≥ 4 завершает подтему;
5. PDF: конспект генерируется, содержит теорию и задания;
6. прогресс: завершение/снятие, сводка, переключение тем;
7. напоминание: после генерации плана add_reminder через MCP
   (интервал 3 дня, без повтора), недоступный сервер не ломает ход;
8. уровни: beginner/advanced попадают в промпт генерации.

Плюс: streak (чистая функция), normalize_plan/normalize_selected_files,
JSON-клиент (response_format + ретрай с удвоением), MockJsonLLM,
настройки get_learning_settings и ЖИВОЙ интеграционный тест проверки
задания через поднятый fs_server.py (офлайн: LLM — мок).

Запуск из корня проекта:
    python -m unittest tests.test_learning_platform -v
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from app.learning.db import (
    AUTO_COMPLETE_SCORE,
    MOCK_USERS,
    LearningRepository,
    compute_streak,
    hash_password,
)
from app.learning.llm import (
    JSON_MAX_TOKENS,
    PLAN_MARKER,
    REVIEW_MARKER,
    SELECT_FILES_MARKER,
    DeepSeekJsonLLM,
    LLMError,
    MockJsonLLM,
    build_json_llm,
    parse_json_object,
)
from app.learning.pdf import (
    GTK_MARKER_DLL,
    PANGO_TOO_OLD_HINT,
    PDFError,
    PDFService,
    WEASYPRINT_HINT,
    _gtk_candidate_dirs,
    ensure_gtk_on_path,
    render_topic_html,
    weasyprint_renderer,
)
from app.learning.plan import (
    LEVEL_GUIDANCE,
    REMINDER_DELAY_MINUTES,
    TOPIC_TITLES,
    PlanGenerationError,
    PlanService,
    build_plan_prompt,
    normalize_plan,
)
from app.learning.review import (
    MAX_SELECTED_FILES,
    ReviewError,
    ReviewService,
    EphemeralFsClient,
    normalize_selected_files,
    parse_tool_json,
)
from src.config import get_learning_settings

try:  # SDK mcp нужен только живому интеграционному тесту
    from src.mcp_client import MCP_SDK_AVAILABLE
    from src.reminder_client import HTTP_AVAILABLE

    MCP_SDK = MCP_SDK_AVAILABLE and HTTP_AVAILABLE
except ImportError:  # pragma: no cover
    MCP_SDK = False


def run(coroutine):
    return asyncio.run(coroutine)


def make_repo(tmp: Path) -> LearningRepository:
    return LearningRepository(tmp / "learning.db")


# ----------------------------------------------------------------------
# 1. Мок-авторизация
# ----------------------------------------------------------------------
class AuthTest(unittest.TestCase):
    """Вход по логину/паролю (SHA-256), 3 предзахардкоженных юзера."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = make_repo(self.tmp)

    def test_three_mock_users_seeded(self) -> None:
        for user in MOCK_USERS:
            found = self.repo.authenticate(user["username"],
                                           user["password"])
            self.assertIsNotNone(found, user["username"])
            self.assertEqual(found["display_name"], user["display_name"])
            self.assertEqual(found["level"], user["level"])

    def test_login_success_ivan(self) -> None:
        user = self.repo.authenticate("ivan", "ivan123")
        self.assertIsNotNone(user)
        self.assertEqual(user["level"], "beginner")

    def test_login_wrong_password(self) -> None:
        self.assertIsNone(self.repo.authenticate("ivan", "nope"))

    def test_login_unknown_user(self) -> None:
        self.assertIsNone(self.repo.authenticate("ghost", "ivan123"))

    def test_password_stored_hashed(self) -> None:
        import sqlite3

        conn = sqlite3.connect(self.repo.db_path)
        try:
            row = conn.execute(
                "SELECT password_hash FROM users WHERE username = 'ivan'"
            ).fetchone()
        finally:
            conn.close()
        self.assertNotIn("ivan123", row[0])
        self.assertEqual(row[0], hash_password("ivan123"))

    def test_get_user(self) -> None:
        user = self.repo.authenticate("maria", "maria123")
        loaded = self.repo.get_user(user["id"])
        self.assertEqual(loaded["username"], "maria")
        self.assertIsNone(self.repo.get_user(99999))


# ----------------------------------------------------------------------
# Streak (чистая функция) + прогресс/переключение тем (ТЗ тест 6)
# ----------------------------------------------------------------------
class StreakAndProgressTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = make_repo(self.tmp)
        self.user = self.repo.authenticate("ivan", "ivan123")
        self.uid = self.user["id"]
        self.today = date(2026, 10, 17)

    def test_compute_streak_pure(self) -> None:
        self.assertEqual(compute_streak([], self.today), 0)
        self.assertEqual(compute_streak([self.today], self.today), 1)
        self.assertEqual(
            compute_streak(
                [self.today - timedelta(days=1), self.today], self.today),
            2)
        # Серия жива, если последний день — вчера (сегодня ещё впереди)
        self.assertEqual(
            compute_streak([self.today - timedelta(days=1)], self.today), 1)
        # Разрыв серии
        self.assertEqual(
            compute_streak(
                [self.today - timedelta(days=2), self.today - timedelta(
                    days=3)], self.today),
            0)
        # Будущие даты не учитываются
        self.assertEqual(
            compute_streak([self.today + timedelta(days=1)], self.today), 0)

    def _topic_with_subtopics(self, count: int = 2) -> int:
        topics = self.repo.ensure_topics(self.uid, TOPIC_TITLES)
        topic_id = topics[0]["id"]
        self.repo.save_subtopics(topic_id, [
            {"title": f"Подтема {i}", "theory": "t", "task": "z",
             "rubric": "r"} for i in range(count)])
        self.repo.mark_topic_generated(
            topic_id, "Тема", "Описание")
        return topic_id

    def test_review_score_completes_subtopic(self) -> None:
        self._topic_with_subtopics()
        overview = self.repo.topics_overview(self.uid)
        topic = self.repo.get_topic(self.uid, overview[0]["id"])
        first = topic["subtopics"][0]
        completed = self.repo.save_review(
            self.uid, first["id"], "Хорошо", AUTO_COMPLETE_SCORE)
        self.assertTrue(completed)
        updated = self.repo.get_topic(self.uid, overview[0]["id"])
        self.assertTrue(updated["subtopics"][0]["is_completed"])
        self.assertEqual(updated["display_status"], "in_progress")

    def test_low_score_not_completed(self) -> None:
        self._topic_with_subtopics()
        overview = self.repo.topics_overview(self.uid)
        topic = self.repo.get_topic(self.uid, overview[0]["id"])
        first = topic["subtopics"][0]
        completed = self.repo.save_review(self.uid, first["id"], "Так себе",
                                          AUTO_COMPLETE_SCORE - 1)
        self.assertFalse(completed)
        self.assertFalse(
            self.repo.get_topic(self.uid, overview[0]["id"])
            ["subtopics"][0]["is_completed"])

    def test_all_completed_topic_completed(self) -> None:
        self._topic_with_subtopics(count=2)
        overview = self.repo.topics_overview(self.uid)
        topic = self.repo.get_topic(self.uid, overview[0]["id"])
        for sub in topic["subtopics"]:
            self.repo.mark_completed(self.uid, sub["id"])
        refreshed = self.repo.get_topic(self.uid, overview[0]["id"])
        self.assertEqual(refreshed["display_status"], "completed")

    def test_mark_completed_toggle_and_streak(self) -> None:
        self._topic_with_subtopics()
        overview = self.repo.topics_overview(self.uid)
        sub = self.repo.get_topic(self.uid, overview[0]["id"])["subtopics"][0]
        self.repo.mark_completed(self.uid, sub["id"])
        self.assertEqual(self.repo.streak_days(self.uid, self.today), 1)
        self.repo.mark_completed(self.uid, sub["id"], completed=False)
        self.assertEqual(self.repo.streak_days(self.uid, self.today), 0)

    def test_progress_summary_and_topic_switch(self) -> None:
        # Прогресс сохраняется при переключении между темами (ТЗ тест 6)
        self._topic_with_subtopics()
        topics = self.repo.ensure_topics(self.uid, TOPIC_TITLES)
        second = topics[1]
        self.repo.save_subtopics(second["id"], [
            {"title": "Другая", "theory": "t", "task": "z", "rubric": "r"}])
        self.repo.mark_topic_generated(second["id"], "Тема 2", "")
        first = self.repo.get_topic(
            self.uid, self.repo.topics_overview(self.uid)[0]["id"])
        self.repo.mark_completed(self.uid, first["subtopics"][0]["id"])
        summary = self.repo.progress_summary(self.uid)
        self.assertEqual(summary, {"total": 3, "completed": 1})

    def test_subtopic_belongs_to_user(self) -> None:
        self._topic_with_subtopics()
        overview = self.repo.topics_overview(self.uid)
        sub = self.repo.get_topic(self.uid, overview[0]["id"])["subtopics"][0]
        other = self.repo.authenticate("maria", "maria123")
        self.assertIsNone(
            self.repo.get_subtopic_for_user(other["id"], sub["id"]))
        self.assertIsNotNone(
            self.repo.get_subtopic_for_user(self.uid, sub["id"]))


# ----------------------------------------------------------------------
# 2-3, 7-8. План: ленивая генерация, повторный вход, напоминание, уровни
# ----------------------------------------------------------------------
class RecordingLLM:
    """Фейковый JSON-LLM: записывает промпты, отвечает по маркеру."""

    def __init__(self, plan: dict | None = None) -> None:
        self.plan = plan or {
            "title": "Сгенерированная тема",
            "description": "Описание от LLM",
            "subtopics": [
                {"title": "A", "theory": "ta", "task": "za", "rubric": "ra"},
                {"title": "B", "theory": "tb", "task": "zb", "rubric": "rb"},
                {"title": "C", "theory": "tc", "task": "zc", "rubric": "rc"},
            ],
        }
        self.prompts: list = []
        self.calls = 0

    async def generate_json_async(self, prompt: str) -> dict:
        self.calls += 1
        self.prompts.append(prompt)
        return dict(self.plan)


class RecordingReminderClient:
    """Фейковый MCP-клиент reminder-сервера: запоминает add_reminder."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list = []

    async def call_tool(self, tool_name, arguments=None):
        self.calls.append((tool_name, dict(arguments or {})))
        if self.error is not None:
            raise self.error
        return '{"result": "ok"}'


class PlanServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = make_repo(self.tmp)
        self.uid = self.repo.authenticate("ivan", "ivan123")["id"]
        self.llm = RecordingLLM()
        self.reminder = RecordingReminderClient()
        self.plan = PlanService(self.repo, self.llm, self.reminder)

    def test_ensure_topics_creates_five(self) -> None:
        topics = self.plan.ensure_topics(self.uid)
        self.assertEqual(len(topics), 5)
        self.assertEqual([t["title"] for t in topics],
                         [t for t in TOPIC_TITLES])
        self.assertFalse(any(t["is_generated"] for t in topics))
        # Повтор — без дублей
        self.assertEqual(len(self.plan.ensure_topics(self.uid)), 5)

    def test_lazy_generation_saves_to_db(self) -> None:
        topics = self.plan.ensure_topics(self.uid)
        topic_id = topics[0]["id"]
        topic = run(self.plan.get_or_generate_topic(self.uid, topic_id))
        self.assertTrue(topic["is_generated"])
        self.assertEqual(len(topic["subtopics"]), 3)
        self.assertEqual(topic["subtopics"][0]["theory"], "ta")
        self.assertEqual(self.llm.calls, 1)

    def test_second_visit_uses_db_no_llm(self) -> None:
        topics = self.plan.ensure_topics(self.uid)
        topic_id = topics[0]["id"]
        run(self.plan.get_or_generate_topic(self.uid, topic_id))
        self.llm.calls = 0
        again = run(self.plan.get_or_generate_topic(self.uid, topic_id))
        self.assertEqual(self.llm.calls, 0)  # из БД, без генерации
        self.assertTrue(again["is_generated"])
        self.assertEqual(again["subtopics"][1]["title"], "B")

    def test_level_in_prompt(self) -> None:
        # ivan — beginner: подсказка уровня попадает в промпт (ТЗ 8)
        topics = self.plan.ensure_topics(self.uid)
        run(self.plan.get_or_generate_topic(self.uid, topics[0]["id"]))
        prompt = self.llm.prompts[0]
        self.assertIn("Уровень пользователя: beginner", prompt)
        self.assertIn(LEVEL_GUIDANCE["beginner"], prompt)
        self.assertIn(PLAN_MARKER, prompt)

    def test_advanced_level_prompt(self) -> None:
        prompt = build_plan_prompt("Любая тема", "advanced")
        self.assertIn(LEVEL_GUIDANCE["advanced"], prompt)
        self.assertIn("advanced", prompt)

    def test_reminder_created_after_generation(self) -> None:
        topics = self.plan.ensure_topics(self.uid)
        topic = run(self.plan.get_or_generate_topic(
            self.uid, topics[0]["id"]))
        self.assertEqual(len(self.reminder.calls), 1)
        name, args = self.reminder.calls[0]
        self.assertEqual(name, "add_reminder")
        self.assertEqual(args["interval_minutes"], REMINDER_DELAY_MINUTES)
        self.assertEqual(args["interval_minutes"], 3 * 24 * 60)
        self.assertFalse(args["repeat"])
        self.assertIn(topic["title"], args["text"])
        self.assertIn("создано", topic["reminder_status"])

    def test_reminder_failure_does_not_break_generation(self) -> None:
        plan = PlanService(
            self.repo, self.llm,
            RecordingReminderClient(error=RuntimeError("сервер упал")))
        topics = self.plan.ensure_topics(self.uid)
        topic = run(plan.get_or_generate_topic(self.uid, topics[0]["id"]))
        self.assertTrue(topic["is_generated"])
        self.assertIn("не создано", topic["reminder_status"])

    def test_no_reminder_client(self) -> None:
        plan = PlanService(self.repo, self.llm, None)
        topics = self.plan.ensure_topics(self.uid)
        topic = run(plan.get_or_generate_topic(self.uid, topics[0]["id"]))
        self.assertIsNone(topic["reminder_status"])

    def test_unknown_topic(self) -> None:
        with self.assertRaises(PlanGenerationError):
            run(self.plan.get_or_generate_topic(self.uid, 99999))


class NormalizePlanTest(unittest.TestCase):
    def test_clips_to_four_subtopics(self) -> None:
        data = {"title": "T", "description": "D",
                "subtopics": [{"title": f"S{i}", "theory": "x",
                               "task": "y", "rubric": "z"}
                              for i in range(6)]}
        subtopics = normalize_plan(data)
        self.assertEqual(len(subtopics), 4)

    def test_missing_fields_filled(self) -> None:
        subtopics = normalize_plan({"subtopics": [{"title": "A"}]})
        self.assertEqual(subtopics[0]["theory"],
                         "(теория не сгенерирована)")
        self.assertEqual(subtopics[0]["rubric"],
                         "(критерии оценки не сгенерированы)")

    def test_empty_subtopics_raises(self) -> None:
        with self.assertRaises(PlanGenerationError):
            normalize_plan({"title": "T", "subtopics": []})
        with self.assertRaises(PlanGenerationError):
            normalize_plan("не словарь")


# ----------------------------------------------------------------------
# 4. Проверка задания через fs MCP
# ----------------------------------------------------------------------
TREE = {
    "app": {"src": {"main": {"MainActivity.kt": "file"}}},
    "README.md": "file",
    "build.gradle.kts": "file",
}


class FakeFsClient:
    """fs-клиент-заглушка: дерево + содержимое (с обёрткой {"result"})."""

    def __init__(self) -> None:
        self.calls: list = []
        self.closed = False

    async def call_tool(self, tool_name, arguments=None):
        self.calls.append((tool_name, dict(arguments or {})))
        if tool_name == "tree":
            return json.dumps({"result": TREE})
        if tool_name == "read_multiple_files":
            paths = arguments["paths"]
            payload = {path: f"// код {path}" for path in paths}
            return json.dumps({"result": payload})
        raise AssertionError(f"неожиданный инструмент {tool_name}")

    async def close(self) -> None:
        self.closed = True


class ReviewServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = make_repo(self.tmp)
        self.uid = self.repo.authenticate("ivan", "ivan123")["id"]
        (self.tmp / "project").mkdir()  # папка «пользователя» существует
        topics = self.repo.ensure_topics(self.uid, TOPIC_TITLES)
        self.repo.save_subtopics(topics[0]["id"], [
            {"title": "A", "theory": "ta", "task": "Сделай MainActivity",
             "rubric": "Есть MainActivity; код компилируется"},
            {"title": "B", "theory": "tb", "task": "Без рубрики",
             "rubric": ""},
        ])
        self.repo.mark_topic_generated(topics[0]["id"], "Тема", "Описание")
        self.topic = self.repo.get_topic(self.uid, topics[0]["id"])
        self.fs = FakeFsClient()
        self.review_llm = RecordingReviewLLM()
        self.service = ReviewService(
            self.repo, self.review_llm, fs_provider=lambda path: self.fs)

    def _first_subtopic(self) -> dict:
        return self.topic["subtopics"][0]

    def test_full_review_flow(self) -> None:
        result = run(self.service.review_task(
            self.uid, self._first_subtopic()["id"],
            str(self.tmp / "project")))
        # Цепочка ТЗ: tree → (выбор LLM) → read_multiple_files → ревью
        tools = [name for name, _ in self.fs.calls]
        self.assertEqual(tools[0], "tree")
        self.assertEqual(tools[1], "read_multiple_files")
        tree_args = self.fs.calls[0][1]
        self.assertEqual(tree_args["path"], ".")
        read_args = self.fs.calls[1][1]
        self.assertEqual(
            read_args["paths"],
            ["app/src/main/MainActivity.kt", "README.md"])
        self.assertTrue(self.fs.closed)  # эфемерный сервер остановлен
        # LLM: промпт оценки содержит задание и РУБРИКУ (без неё нельзя)
        self.assertIn("Сделай MainActivity", self.review_llm.review_prompt)
        self.assertIn("Есть MainActivity", self.review_llm.review_prompt)
        self.assertIn("// код app/src/main/MainActivity.kt",
                      self.review_llm.review_prompt)
        # Результат и прогресс
        self.assertEqual(result["score"], 4)
        self.assertTrue(result["completed"])
        self.assertTrue(self.fs.closed)
        updated = self.repo.get_subtopic_for_user(
            self.uid, self._first_subtopic()["id"])
        self.assertTrue(updated["is_completed"])
        self.assertEqual(updated["ai_feedback"], "Неплохо, но добавь тесты")
        self.assertIn("MainActivity.kt", updated["task_answer"])

    def test_rubric_required(self) -> None:
        second = self.topic["subtopics"][1]
        with self.assertRaises(ReviewError):
            run(self.service.review_task(self.uid, second["id"],
                                         str(self.tmp / "project")))
        self.assertEqual(self.fs.calls, [])  # fs даже не вызывался

    def test_bad_path(self) -> None:
        with self.assertRaises(ReviewError):
            run(self.service.review_task(
                self.uid, self._first_subtopic()["id"],
                str(self.tmp / "нет_такой")))

    def test_unknown_subtopic(self) -> None:
        with self.assertRaises(ReviewError):
            run(self.service.review_task(self.uid, 99999,
                                         str(self.tmp / "project")))

    def test_empty_selection_raises(self) -> None:
        self.review_llm.select = {"files": []}
        with self.assertRaises(ReviewError):
            run(self.service.review_task(
                self.uid, self._first_subtopic()["id"],
                str(self.tmp / "project")))

    def test_fs_error_payload(self) -> None:
        class ErrorFs(FakeFsClient):
            async def call_tool(self, tool_name, arguments=None):
                if tool_name == "tree":
                    return json.dumps(
                        {"result": {"error": "Доступ запрещён"}})
                return await super().call_tool(tool_name, arguments)

        self.fs = ErrorFs()
        service = ReviewService(
            self.repo, self.review_llm, fs_provider=lambda path: self.fs)
        with self.assertRaises(ReviewError):
            run(service.review_task(
                self.uid, self._first_subtopic()["id"],
                str(self.tmp / "project")))


class RecordingReviewLLM:
    """LLM-заглушка проверки: выбор файлов — из дерева, оценка — своя."""

    def __init__(self) -> None:
        self.select = {"files": ["app/src/main/MainActivity.kt",
                                 "README.md"]}
        self.review = {"score": 4, "feedback": "Неплохо, но добавь тесты"}
        self.prompts: list = []
        self.review_prompt = ""

    async def generate_json_async(self, prompt: str) -> dict:
        self.prompts.append(prompt)
        if SELECT_FILES_MARKER in prompt:
            return self.select
        if REVIEW_MARKER in prompt:
            self.review_prompt = prompt
            return dict(self.review)
        raise AssertionError("неизвестный маркер в промпте")


class NormalizeSelectedFilesTest(unittest.TestCase):
    def test_dedupe_cap_and_backslashes(self) -> None:
        files = normalize_selected_files({
            "files": ["a\\b.kt", "a/b.kt", " c ", "", "d", "e", "f", "g"]})
        self.assertEqual(files, ["a/b.kt", "c", "d", "e", "f"])
        self.assertLessEqual(len(files), MAX_SELECTED_FILES)

    def test_bare_list_accepted(self) -> None:
        self.assertEqual(normalize_selected_files(["x.kt"]), ["x.kt"])
        self.assertEqual(normalize_selected_files("мусор"), [])
        self.assertEqual(normalize_selected_files({"files": "мусор"}), [])


class ParseToolJsonTest(unittest.TestCase):
    def test_unwrap_result(self) -> None:
        self.assertEqual(parse_tool_json('{"result": {"a": 1}}'), {"a": 1})
        self.assertEqual(parse_tool_json('{"a": 1}'), {"a": 1})
        self.assertIsNone(parse_tool_json("не JSON"))


class EnsureGtkOnPathTest(unittest.TestCase):
    """ensure_gtk_on_path: автоподключение каталога GTK к PATH процесса.

    Рантайм, установленный «вживую», не виден работающему приложению
    (PATH процесса застывает на старте, error 0x7e у пользователя) —
    функция чинит PATH на месте перед импортом WeasyPrint.
    """

    def _fake_gtk_dir(self) -> Path:
        bin_dir = Path(tempfile.mkdtemp()) / "bin"
        bin_dir.mkdir()
        (bin_dir / GTK_MARKER_DLL).write_bytes(b"dll")
        return bin_dir

    def test_adds_dir_with_dll_to_path_front(self) -> None:
        bin_dir = self._fake_gtk_dir()
        with mock.patch.dict(os.environ):
            os.environ["PATH"] = "C:\\somewhere"
            self.assertEqual(ensure_gtk_on_path([bin_dir]), bin_dir)
            self.assertEqual(
                os.environ["PATH"].split(os.pathsep)[0], str(bin_dir))

    def test_skips_dir_without_marker_dll(self) -> None:
        empty = Path(tempfile.mkdtemp())
        with mock.patch.dict(os.environ):
            before = os.environ.get("PATH")
            self.assertIsNone(ensure_gtk_on_path([empty]))
            self.assertEqual(os.environ.get("PATH"), before)

    def test_no_duplicate_path_entry(self) -> None:
        bin_dir = self._fake_gtk_dir()
        with mock.patch.dict(os.environ):
            ensure_gtk_on_path([bin_dir])
            ensure_gtk_on_path([bin_dir])
            entries = [entry.casefold() for entry
                       in os.environ["PATH"].split(os.pathsep)]
            self.assertEqual(entries.count(str(bin_dir).casefold()), 1)

    def test_gtk_runtime_bin_override_is_first_candidate(self) -> None:
        bin_dir = self._fake_gtk_dir()
        with mock.patch.dict(os.environ,
                             {"GTK_RUNTIME_BIN": str(bin_dir)}):
            self.assertEqual(_gtk_candidate_dirs()[0], bin_dir)

    def test_non_windows_noop(self) -> None:
        bin_dir = self._fake_gtk_dir()
        with mock.patch.object(sys, "platform", "linux"), \
                mock.patch.dict(os.environ):
            before = os.environ.get("PATH")
            self.assertIsNone(ensure_gtk_on_path([bin_dir]))
            self.assertEqual(os.environ.get("PATH"), before)


# ----------------------------------------------------------------------
# 5. PDF-конспект (WeasyPrint)
# ----------------------------------------------------------------------
class CapturingRenderer:
    """Фейковый renderer: пишет файл, запоминает HTML."""

    def __init__(self) -> None:
        self.html = ""
        self.fail = False

    def __call__(self, html: str, target: Path) -> None:
        if self.fail:
            raise RuntimeError("диск переполнен")
        self.html = html
        target.write_bytes(b"%PDF-fake")


def _weasyprint_available() -> bool:
    """Импортируется ли WeasyPrint (после подключения каталогов GTK)."""
    try:
        ensure_gtk_on_path()
        import weasyprint  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 — ImportError/DLL
        return False


class PDFServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.renderer = CapturingRenderer()
        self.service = PDFService(output_dir=self.tmp / "pdf",
                                  renderer=self.renderer)
        self.topic = {"id": 3, "title": "Жизненный <цикл>",
                      "description": "Про Activity \nи Fragment",
                      "generated_at": "2026-10-17 10:00:00"}
        self.subtopics = [
            {"title": "Первая", "theory": "# Теория\nсо ```код```",
             "task": "Сделай <экран>", "rubric": "Работает",
             "is_completed": True, "score": 5,
             "ai_feedback": "Молодец!"},
            {"title": "Вторая", "theory": "Ещё теория",
             "task": "Ещё задание", "rubric": "Тоже", "is_completed": False,
             "score": None, "ai_feedback": ""},
        ]

    def test_pdf_created_with_html_content(self) -> None:
        path = self.service.generate_summary(self.topic, self.subtopics,
                                             user_name="Иван")
        file = Path(path)
        self.assertTrue(file.is_file())
        self.assertEqual(file.parent, self.tmp / "pdf")  # parent.mkdir
        self.assertTrue(file.name.startswith("summary_3_"))  # формат ТЗ
        html = self.renderer.html
        # Содержимое: теория И задания (ТЗ тест 5), фидбек, экранирование
        self.assertIn("Жизненный &lt;цикл&gt;", html)
        self.assertIn("# Теория", html)
        self.assertIn("Сделай &lt;экран&gt;", html)
        self.assertIn("Критерии оценки", html)
        self.assertIn("Молодец!", html)
        self.assertIn("Иван", html)
        self.assertIn("1. Первая ✅", html)

    def test_empty_subtopics_raises(self) -> None:
        with self.assertRaises(PDFError):
            self.service.generate_summary(self.topic, [])

    def test_renderer_failure_wrapped(self) -> None:
        self.renderer.fail = True
        with self.assertRaises(PDFError):
            self.service.generate_summary(self.topic, self.subtopics)

    def test_renderer_error_without_gtk(self) -> None:
        """Без GTK-DLL — PDFError с подсказкой про установку.

        Проверка идёт в ПОДПРОЦЕССЕ: PATH очищается от gtk-каталогов и
        WEASYPRINT_DLL_DIRECTORIES обнуляется (WeasyPrint сам
        регистрирует стандартный каталог рантайма через
        os.add_dll_directory — иначе на машинах с установленным GTK
        «без-GTK» сценарий невоспроизводим). gtk_candidates=[] — без
        нашего ensure_gtk_on_path.
        """
        if sys.platform != "win32":
            self.skipTest("GTK-зависимость проверяется только в Windows")
        output_pdf = self.tmp / "no_gtk.pdf"
        code = (
            "import sys\n"
            f"sys.path.insert(0, r'{PROJECT_ROOT}')\n"
            "from app.learning.pdf import weasyprint_renderer, PDFError\n"
            "try:\n"
            "    weasyprint_renderer('<html/>', "
            f"r'{output_pdf}', gtk_candidates=[])\n"
            "except PDFError as exc:\n"
            "    print('PDFError:', 'GTK' in str(exc))\n"
            "    raise SystemExit(0)\n"
            "print('RENDER_OK')\n"
            "raise SystemExit(1)\n"
        )
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join(
            entry for entry in env.get("PATH", "").split(os.pathsep)
            if entry and "gtk" not in entry.lower())
        env["WEASYPRINT_DLL_DIRECTORIES"] = ""
        proc = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True,
            text=True, timeout=180, cwd=str(PROJECT_ROOT))
        self.assertEqual(
            proc.returncode, 0,
            f"stdout: {proc.stdout}\nstderr: {proc.stderr[-500:]}")
        self.assertIn("PDFError: True", proc.stdout)

    def test_pango_too_old_hint_in_module(self) -> None:
        """Подсказка про старый Pango — отдельный текст (не как «нет GTK»)."""
        self.assertIn("1.44", PANGO_TOO_OLD_HINT)
        self.assertNotEqual(PANGO_TOO_OLD_HINT, WEASYPRINT_HINT)

    @unittest.skipUnless(_weasyprint_available(),
                         "WeasyPrint + GTK недоступны в этом окружении")
    def test_real_weasyprint_render(self) -> None:
        """Живой рендер: WeasyPrint + GTK (где установлен рантайм).

        Выполняется автоматически на машинах с GTK3-Runtime Win64;
        без рантайма пропускается (оффлайн-прогон не ломается).
        """
        service = PDFService(output_dir=self.tmp / "pdf_real")
        path = service.generate_summary(self.topic, self.subtopics,
                                        user_name="Иван")
        data = Path(path).read_bytes()
        self.assertTrue(data.startswith(b"%PDF-"))
        self.assertGreater(len(data), 500)


# ----------------------------------------------------------------------
# JSON-клиенты
# ----------------------------------------------------------------------
class JsonLLMTest(unittest.TestCase):
    def test_parse_json_object_fenced(self) -> None:
        self.assertEqual(parse_json_object('```json\n{"a": 1}\n```'),
                         {"a": 1})
        self.assertEqual(parse_json_object('проза {"a": 1} конец'),
                         {"a": 1})
        with self.assertRaises(LLMError):
            parse_json_object("совсем не JSON")

    def test_response_format_and_retry(self) -> None:
        calls: list = []

        def make_response(content: str, finish: str):
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    finish_reason=finish)])

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    return make_response("", "length")  # обрезан
                return make_response('{"score": 5}', "stop")

        llm = DeepSeekJsonLLM(
            "key", client=SimpleNamespace(
                chat=SimpleNamespace(completions=FakeCompletions())))
        result = llm.generate_json(f"{REVIEW_MARKER}\nпроверь")
        self.assertEqual(result, {"score": 5})
        # JSON-режим включён; ретрай — с удвоенным max_tokens
        self.assertEqual(calls[0]["response_format"],
                         {"type": "json_object"})
        self.assertEqual(calls[1]["max_tokens"], JSON_MAX_TOKENS * 2)

    def test_build_from_config_mock(self) -> None:
        llm = build_json_llm({"DEEPSEEK_API_KEY": "k", "MOCK_MODE": True})
        self.assertTrue(llm.mock_mode)
        plan = llm.generate_json(
            f"{PLAN_MARKER}\nТема: Тест\nУровень пользователя: beginner.")
        self.assertEqual(plan["title"], "Тест")
        self.assertEqual(len(plan["subtopics"]), 3)
        self.assertIn("мок-генерация", plan["description"])


class MockJsonLLMTest(unittest.TestCase):
    def setUp(self) -> None:
        self.llm = MockJsonLLM()

    def test_plan_by_level(self) -> None:
        prompt = build_plan_prompt("Compose", "intermediate")
        plan = self.llm.generate_json(prompt)
        self.assertEqual(plan["title"], "Compose")
        self.assertIn("3-4", prompt)
        self.assertEqual(len(plan["subtopics"]), 3)

    def test_select_files_from_tree(self) -> None:
        prompt = (f"{SELECT_FILES_MARKER}\nЗадание:\nчто-то\n"
                  "Дерево проекта (JSON):\n"
                  + json.dumps(TREE, ensure_ascii=False))
        data = self.llm.generate_json(prompt)
        self.assertIn("app/src/main/MainActivity.kt", data["files"])
        self.assertIn("README.md", data["files"])
        self.assertLessEqual(len(data["files"]), 5)

    def test_review(self) -> None:
        data = self.llm.generate_json(f"{REVIEW_MARKER}\nзадание")
        self.assertEqual(data["score"], 5)
        self.assertTrue(data["feedback"])

    def test_unknown_prompt(self) -> None:
        with self.assertRaises(LLMError):
            self.llm.generate_json("просто текст")


# ----------------------------------------------------------------------
# Настройки
# ----------------------------------------------------------------------
class LearningSettingsTest(unittest.TestCase):
    def test_defaults(self) -> None:
        settings = get_learning_settings({})
        self.assertTrue(Path(settings["db_path"]).is_absolute())
        self.assertEqual(Path(settings["pdf_dir"]),
                         PROJECT_ROOT / "output" / "pdf")

    def test_env_override(self) -> None:
        with mock.patch.dict(os.environ, {
                "LEARNING_DB_PATH": str(PROJECT_ROOT / "x" / "l.db"),
                "LEARNING_PDF_DIR": "custom_pdf"}):
            settings = get_learning_settings()
            self.assertEqual(settings["db_path"],
                             str(PROJECT_ROOT / "x" / "l.db"))
            self.assertEqual(
                settings["pdf_dir"],
                str(PROJECT_ROOT / "custom_pdf"))


# ----------------------------------------------------------------------
# Живая интеграция: fs_server.py как подпроцесс + MockJsonLLM (офлайн)
# ----------------------------------------------------------------------
@unittest.skipUnless(MCP_SDK, "SDK mcp недоступен")
class EphemeralFsReviewIntegrationTest(unittest.TestCase):
    """Полный цикл проверки на ЖИВОМ fs_server.py (LLM — мок).

    Эфемерный сервер с --root в tmp-проекте; tree/read_multiple_files
    идут по HTTP MCP (как из UI), но сеть наружу не трогается.
    """

    def test_review_via_live_fs_server(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        project = tmp / "AndroidDemo"
        src = project / "app" / "src" / "main"
        src.mkdir(parents=True)
        (src / "MainActivity.kt").write_text(
            "class MainActivity {\n    // TODO: задание\n}\n",
            encoding="utf-8")
        (project / "README.md").write_text(
            "# Демо-проект\nВыполненное задание.\n", encoding="utf-8")
        (project / "build.gradle.kts").write_text(
            "plugins { id(\"com.android.application\") }\n",
            encoding="utf-8")

        repo = make_repo(tmp)
        uid = repo.authenticate("ivan", "ivan123")["id"]
        topics = repo.ensure_topics(uid, TOPIC_TITLES)
        repo.save_subtopics(topics[0]["id"], [
            {"title": "A", "theory": "t",
             "task": "Создай MainActivity", "rubric": "Есть класс"}])
        repo.mark_topic_generated(topics[0]["id"], "Тема", "")
        subtopic = repo.get_topic(uid, topics[0]["id"])["subtopics"][0]

        client = EphemeralFsClient(str(project))
        try:
            client.start()
            service = ReviewService(repo, MockJsonLLM(),
                                    fs_provider=lambda path: client)
            result = run(service.review_task(uid, subtopic["id"],
                                             str(project)))
        finally:
            run(client.close())

        self.assertEqual(result["score"], 5)
        self.assertTrue(result["completed"])
        self.assertTrue(any("MainActivity.kt" in f for f in result["files"]))
        updated = repo.get_subtopic_for_user(uid, subtopic["id"])
        self.assertTrue(updated["is_completed"])


if __name__ == "__main__":
    unittest.main()
