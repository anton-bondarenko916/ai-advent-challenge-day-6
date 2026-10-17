"""БД платформы обучения (ТЗ ai_1_36): мок-авторизация, план, прогресс.

Схема из ТЗ: ``users`` (мок-пользователи с SHA-256 хешем пароля),
``topics`` (5 тем курса, ленивая генерация), ``subtopics`` (3–4 подтемы
с теорией/заданием/рубрикой), ``progress`` (статус, фидбек, оценка,
дата завершения — UNIQUE(user_id, subtopic_id)).

Соединения SQLite открываются и закрываются внутри одного вызова
(contextmanager ``_db()``): «висящее» соединение держит файл БД, и
на Windows временная папка тестов не удаляется (WinError 32).

Streak — число дней подряд (заканчивая сегодня или вчера), в которые
пользователь завершал хотя бы одну подтему; чистая функция
:func:`compute_streak` покрывается офлайн-тестами.
"""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from src.utils import get_logger

logger = get_logger("deepseek-chat.learning.db")

# Мок-пользователи из ТЗ (создаются при инициализации БД, INSERT OR IGNORE)
MOCK_USERS = [
    {"username": "ivan", "password": "ivan123",
     "display_name": "Иван", "level": "beginner"},
    {"username": "maria", "password": "maria123",
     "display_name": "Мария", "level": "intermediate"},
    {"username": "alex", "password": "alex123",
     "display_name": "Алекс", "level": "advanced"},
]

# Оценка ReviewService, начиная с которой подтема завершается автоматически
AUTO_COMPLETE_SCORE = 4

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    display_name TEXT,
    level TEXT DEFAULT 'beginner',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    order_index INTEGER,
    status TEXT DEFAULT 'not_started',
    generated_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS subtopics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    theory TEXT,
    task TEXT,
    rubric TEXT,
    order_index INTEGER,
    status TEXT DEFAULT 'not_started',
    FOREIGN KEY (topic_id) REFERENCES topics(id)
);

CREATE TABLE IF NOT EXISTS progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    subtopic_id INTEGER NOT NULL,
    status TEXT DEFAULT 'not_started',
    task_answer TEXT,
    ai_feedback TEXT,
    score INTEGER,
    completed_at TIMESTAMP,
    UNIQUE(user_id, subtopic_id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (subtopic_id) REFERENCES subtopics(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_topics_user_order
    ON topics(user_id, order_index);
"""


def hash_password(password: str) -> str:
    """SHA-256 хеш пароля (ТЗ: в открытом виде пароли не хранятся)."""
    return hashlib.sha256(str(password or "").encode("utf-8")).hexdigest()


def compute_streak(days: Sequence[date], today: date) -> int:
    """Длина серии дней подряд, заканчивающейся сегодня или вчера.

    Args:
        days: даты завершений (дубликаты не важны).
        today: «сегодня» (параметр — для офлайн-тестов).

    Returns:
        Число дней в серии; пустой список или разрыв серии — 0.
    """
    unique = {day for day in days if day <= today}
    if not unique:
        return 0
    # Серия жива, если последний завершённый день — сегодня или вчера
    cursor = today if today in unique else today - timedelta(days=1)
    if cursor not in unique:
        return 0
    streak = 0
    while cursor in unique:
        streak += 1
        cursor -= timedelta(days=1)
    return streak


def _parse_date(raw: Any) -> Optional[date]:
    """Дата из TIMESTAMP SQLite («YYYY-MM-DD HH:MM:SS»); None при сбое."""
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.split(".")[0]).date()
    except ValueError:
        return None


class LearningRepository:
    """Репозиторий платформы обучения (SQLite, схема из ТЗ).

    Attributes:
        db_path: путь к файлу БД (data/learning.db).
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as conn:
            conn.executescript(_SCHEMA)
        self._seed_mock_users()

    # ------------------------------------------------------------------
    # Инфраструктура
    # ------------------------------------------------------------------
    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        """Соединение с гарантией закрытия (см. докстринг модуля)."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _seed_mock_users(self) -> None:
        """Три предзахардкоженных пользователя из ТЗ (идемпотентно)."""
        with self._db() as conn:
            for user in MOCK_USERS:
                conn.execute(
                    "INSERT OR IGNORE INTO users "
                    "(username, password_hash, display_name, level) "
                    "VALUES (?, ?, ?, ?)",
                    (user["username"], hash_password(user["password"]),
                     user["display_name"], user["level"]),
                )

    @staticmethod
    def _user_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        return dict(row) if row is not None else None

    # ------------------------------------------------------------------
    # Мок-авторизация
    # ------------------------------------------------------------------
    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Вход по логину/паролю: словарь пользователя или None.

        Пароль сравнивается как SHA-256 хеш (в БД хранится только хеш).
        """
        with self._db() as conn:
            row = conn.execute(
                "SELECT id, username, display_name, level FROM users "
                "WHERE username = ? AND password_hash = ?",
                (str(username or "").strip(),
                 hash_password(password)),
            ).fetchone()
        return self._user_dict(row)

    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Пользователь по id (уровень нужен промпту генерации плана)."""
        with self._db() as conn:
            row = conn.execute(
                "SELECT id, username, display_name, level FROM users "
                "WHERE id = ?", (int(user_id),)
            ).fetchone()
        return self._user_dict(row)

    # ------------------------------------------------------------------
    # Темы и подтемы
    # ------------------------------------------------------------------
    def ensure_topics(self, user_id: int,
                      titles: Sequence[str]) -> List[Dict[str, Any]]:
        """Создаёт 5 тем курса, если их ещё нет, и возвращает список.

        Темы создаются «пустыми» (generated_at NULL): контент конкретной
        темы генерируется лениво при первом входе (PlanService).
        """
        with self._db() as conn:
            for index, title in enumerate(titles):
                conn.execute(
                    "INSERT OR IGNORE INTO topics (user_id, title, "
                    "order_index) VALUES (?, ?, ?)",
                    (int(user_id), str(title), int(index)),
                )
        return self.topics_overview(user_id)

    def topics_overview(self, user_id: int) -> List[Dict[str, Any]]:
        """Темы пользователя по порядку + счётчики завершённых подтем."""
        with self._db() as conn:
            topics = [dict(row) for row in conn.execute(
                "SELECT id, title, description, order_index, status, "
                "generated_at FROM topics WHERE user_id = ? "
                "ORDER BY order_index", (int(user_id),))]
            for topic in topics:
                total, completed = conn.execute(
                    "SELECT COUNT(*), "
                    "COALESCE(SUM(CASE WHEN p.status = 'completed' "
                    "THEN 1 ELSE 0 END), 0) "
                    "FROM subtopics s LEFT JOIN progress p "
                    "ON p.subtopic_id = s.id AND p.user_id = ? "
                    "WHERE s.topic_id = ?",
                    (int(user_id), int(topic["id"]))
                ).fetchone()
                topic["subtopics_total"] = int(total or 0)
                topic["subtopics_completed"] = int(completed or 0)
                topic["is_generated"] = topic["generated_at"] is not None
        for topic in topics:
            topic["display_status"] = self._topic_display_status(topic)
        return topics

    @staticmethod
    def _topic_display_status(topic: Dict[str, Any]) -> str:
        """Статус для UI: not_generated / not_started / in_progress /
        completed (колонка topics.status хранит упрощённо, вычисляем
        по факту — прогрессу и generated_at)."""
        if not topic.get("is_generated"):
            return "not_generated"
        total = int(topic.get("subtopics_total") or 0)
        completed = int(topic.get("subtopics_completed") or 0)
        if total and completed >= total:
            return "completed"
        if completed:
            return "in_progress"
        return "not_started"

    def get_topic(self, user_id: int, topic_id: int) -> Optional[Dict[str, Any]]:
        """Тема с подтемами и прогрессом по каждой (или None — чужая тема)."""
        with self._db() as conn:
            row = conn.execute(
                "SELECT id, user_id, title, description, order_index, "
                "status, generated_at FROM topics "
                "WHERE id = ? AND user_id = ?",
                (int(topic_id), int(user_id)),
            ).fetchone()
            if row is None:
                return None
            topic = dict(row)
            topic["subtopics"] = []
            for sub in conn.execute(
                "SELECT s.id, s.topic_id, s.title, s.theory, s.task, "
                "s.rubric, s.order_index, s.status, "
                "p.status AS progress_status, p.score, p.ai_feedback, "
                "p.completed_at "
                "FROM subtopics s LEFT JOIN progress p "
                "ON p.subtopic_id = s.id AND p.user_id = ? "
                "WHERE s.topic_id = ? ORDER BY s.order_index, s.id",
                (int(user_id), int(topic_id)),
            ):
                subtopic = dict(sub)
                subtopic["is_completed"] = \
                    subtopic["progress_status"] == "completed"
                topic["subtopics"].append(subtopic)
        topic["is_generated"] = topic["generated_at"] is not None
        topic["subtopics_completed"] = sum(
            1 for sub in topic["subtopics"] if sub["is_completed"])
        topic["subtopics_total"] = len(topic["subtopics"])
        topic["display_status"] = self._topic_display_status(topic)
        return topic

    def mark_topic_generated(self, topic_id: int, title: str,
                             description: str) -> None:
        """Фиксация ленивой генерации: заголовок/описание + generated_at."""
        with self._db() as conn:
            conn.execute(
                "UPDATE topics SET title = ?, description = ?, "
                "generated_at = CURRENT_TIMESTAMP, "
                "status = 'in_progress' WHERE id = ?",
                (str(title), str(description or ""), int(topic_id)),
            )

    def save_subtopics(self, topic_id: int,
                       subtopics: List[Dict[str, Any]]) -> None:
        """Сохраняет подтемы сгенерированной темы (старые затираются —
        генерация одноразовая, повторный вызов get_or_generate_topic
        в БД не заходит)."""
        with self._db() as conn:
            conn.execute("DELETE FROM subtopics WHERE topic_id = ?",
                         (int(topic_id),))
            for index, sub in enumerate(subtopics):
                conn.execute(
                    "INSERT INTO subtopics (topic_id, title, theory, "
                    "task, rubric, order_index) VALUES (?, ?, ?, ?, ?, ?)",
                    (int(topic_id), str(sub["title"]),
                     str(sub.get("theory") or ""),
                     str(sub.get("task") or ""),
                     str(sub.get("rubric") or ""), int(index)),
                )

    def get_subtopic_for_user(self, user_id: int,
                              subtopic_id: int) -> Optional[Dict[str, Any]]:
        """Подтема с заголовком темы; None — не найдена или чужая."""
        with self._db() as conn:
            row = conn.execute(
                "SELECT s.id, s.topic_id, s.title, s.theory, s.task, "
                "s.rubric, s.order_index, t.title AS topic_title, "
                "t.user_id, p.status AS progress_status, p.score, "
                "p.ai_feedback, p.task_answer, p.completed_at "
                "FROM subtopics s JOIN topics t ON t.id = s.topic_id "
                "LEFT JOIN progress p "
                "ON p.subtopic_id = s.id AND p.user_id = ? "
                "WHERE s.id = ? AND t.user_id = ?",
                (int(user_id), int(subtopic_id), int(user_id)),
            ).fetchone()
        if row is None:
            return None
        subtopic = dict(row)
        subtopic["is_completed"] = \
            subtopic["progress_status"] == "completed"
        return subtopic

    # ------------------------------------------------------------------
    # Прогресс
    # ------------------------------------------------------------------
    def save_review(self, user_id: int, subtopic_id: int, feedback: str,
                    score: int, task_answer: str = "") -> bool:
        """Сохраняет результат проверки; оценка ≥ AUTO_COMPLETE_SCORE
        завершает подтему. Returns: True — подтема теперь завершена."""
        score = max(1, min(5, int(score or 0)))
        completed = score >= AUTO_COMPLETE_SCORE
        with self._db() as conn:
            conn.execute(
                "INSERT INTO progress (user_id, subtopic_id, status, "
                "task_answer, ai_feedback, score, completed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, "
                "CASE WHEN ? THEN datetime('now', 'localtime') END) "
                "ON CONFLICT(user_id, subtopic_id) DO UPDATE SET "
                "status = excluded.status, "
                "task_answer = excluded.task_answer, "
                "ai_feedback = excluded.ai_feedback, "
                "score = excluded.score, "
                "completed_at = excluded.completed_at",
                (int(user_id), int(subtopic_id),
                 "completed" if completed else "in_progress",
                 str(task_answer or ""), str(feedback or ""), score,
                 completed),
            )
            self._refresh_topic_status(conn, user_id, subtopic_id)
        return completed

    def mark_completed(self, user_id: int, subtopic_id: int,
                       completed: bool = True) -> None:
        """Ручная отметка «выполнено» (кнопка в UI подтемы)."""
        with self._db() as conn:
            conn.execute(
                "INSERT INTO progress (user_id, subtopic_id, status, "
                "completed_at) VALUES (?, ?, ?, "
                "CASE WHEN ? THEN datetime('now', 'localtime') END) "
                "ON CONFLICT(user_id, subtopic_id) DO UPDATE SET "
                "status = excluded.status, "
                "completed_at = excluded.completed_at",
                (int(user_id), int(subtopic_id),
                 "completed" if completed else "in_progress", completed),
            )
            self._refresh_topic_status(conn, user_id, subtopic_id)

    def _refresh_topic_status(self, conn: sqlite3.Connection,
                              user_id: int, subtopic_id: int) -> None:
        """topics.status — упрощённое зеркало прогресса (completed, когда
        завершены все подтемы)."""
        conn.execute(
            "UPDATE topics SET status = CASE WHEN EXISTS ("
            "SELECT 1 FROM subtopics s LEFT JOIN progress p "
            "ON p.subtopic_id = s.id AND p.user_id = ? "
            "WHERE s.topic_id = topics.id "
            "AND (p.status IS NULL OR p.status != 'completed')"
            ") THEN 'in_progress' ELSE 'completed' END "
            "WHERE id = (SELECT topic_id FROM subtopics WHERE id = ?)",
            (int(user_id), int(subtopic_id)),
        )

    def progress_summary(self, user_id: int) -> Dict[str, int]:
        """Сводка дашборда: total/completed подтем по всем темам."""
        with self._db() as conn:
            total, completed = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(CASE WHEN p.status = "
                "'completed' THEN 1 ELSE 0 END), 0) "
                "FROM subtopics s JOIN topics t ON t.id = s.topic_id "
                "LEFT JOIN progress p "
                "ON p.subtopic_id = s.id AND p.user_id = ? "
                "WHERE t.user_id = ?", (int(user_id), int(user_id)),
            ).fetchone()
        return {"total": int(total or 0), "completed": int(completed or 0)}

    def completion_days(self, user_id: int) -> List[date]:
        """Дни, в которые пользователь завершал подтемы (для streak)."""
        with self._db() as conn:
            rows = conn.execute(
                "SELECT completed_at FROM progress "
                "WHERE user_id = ? AND status = 'completed' "
                "AND completed_at IS NOT NULL", (int(user_id),))
            days = [day for day in
                    (_parse_date(row["completed_at"]) for row in rows)
                    if day is not None]
        return days

    def streak_days(self, user_id: int, today: Optional[date] = None) -> int:
        """Серия дней подряд с завершениями (сегодня/вчера — серия жива)."""
        return compute_streak(self.completion_days(user_id),
                              today or date.today())
