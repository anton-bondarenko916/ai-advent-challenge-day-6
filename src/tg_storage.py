"""История диалогов Telegram-бота: SQLite на пользователя (ТЗ ai_1_31).

Две таблицы в одной БД (по образцу reminders.db):

* ``messages`` — ходы диалогов всех пользователей (role: user/assistant,
  content, created_at); в модель уходит только хвост из
  TELEGRAM_HISTORY_LIMIT сообщений, но на диске хранится больше
  (подрезка до последних 200 — защита от бесконечного роста);
* ``user_state`` — выбранная пользователем модель (/model).

Ловушки проекта соблюдены: короткоживущие соединения sqlite3 с
обязательным ``close()`` (Windows держит файл, пока соединение открыто),
таблицы создаются при каждом подключении (``CREATE TABLE IF NOT
EXISTS`` — файл могли удалить на ходу), ошибки БД НЕ роняют бота —
методы деградируют (лог + пустой результат), ход продолжается без
истории.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List

from .utils import get_logger

logger = get_logger("deepseek-chat.tg-storage")

# Сколько последних сообщений хранить на пользователя (подрезка при append)
KEEP_PER_USER = 200


def _now_iso() -> str:
    """Метка времени ISO 8601 (UTC)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TelegramHistoryStore:
    """История диалогов и выбор модели на пользователя (SQLite).

    Attributes:
        db_path: путь к файлу БД (каталог создаётся при необходимости).
    """

    def __init__(self, db_path: Any) -> None:
        self.db_path = Path(db_path)
        if str(self.db_path.parent):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # История диалогов
    # ------------------------------------------------------------------
    def append(self, user_id: int, role: str, content: str) -> bool:
        """Добавляет сообщение в историю (ошибка БД → False, не исключение).

        Заодно подрезает историю пользователя до последних KEEP_PER_USER
        сообщений.
        """
        try:
            with self._db() as conn:
                conn.execute(
                    "INSERT INTO messages (user_id, role, content,"
                    " created_at) VALUES (?, ?, ?, ?)",
                    (int(user_id), str(role), str(content), _now_iso()),
                )
                conn.execute(
                    "DELETE FROM messages WHERE user_id = ? AND id NOT IN ("
                    "  SELECT id FROM messages WHERE user_id = ?"
                    "  ORDER BY id DESC LIMIT ?)",
                    (int(user_id), int(user_id), KEEP_PER_USER),
                )
            return True
        except sqlite3.Error as exc:
            logger.warning("История: append не удался: %s", exc)
            return False

    def load(self, user_id: int, limit: int = 20) -> List[Dict[str, str]]:
        """Последние ``limit`` сообщений в формате Ollama
        ([{"role", "content"}]), хронологический порядок.

        Пустой список — истории нет или БД недоступна (бот отвечает
        без контекста, ход не ломается).
        """
        try:
            with self._db() as conn:
                rows = conn.execute(
                    "SELECT role, content FROM messages WHERE user_id = ?"
                    " ORDER BY id DESC LIMIT ?",
                    (int(user_id), max(1, int(limit))),
                ).fetchall()
        except sqlite3.Error as exc:
            logger.warning("История: load не удался: %s", exc)
            return []
        return [{"role": str(role), "content": str(content)}
                for role, content in reversed(rows)]

    def clear(self, user_id: int) -> int:
        """Удаляет историю пользователя (/clear); возвращает число строк."""
        try:
            with self._db() as conn:
                cursor = conn.execute(
                    "DELETE FROM messages WHERE user_id = ?", (int(user_id),))
                return cursor.rowcount if cursor.rowcount > 0 else 0
        except sqlite3.Error as exc:
            logger.warning("История: clear не удался: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Выбранная модель (/model)
    # ------------------------------------------------------------------
    def get_model(self, user_id: int) -> str:
        """Выбранная пользователем модель (``""`` — не выбирал)."""
        try:
            with self._db() as conn:
                row = conn.execute(
                    "SELECT model FROM user_state WHERE user_id = ?",
                    (int(user_id),),
                ).fetchone()
        except sqlite3.Error as exc:
            logger.warning("user_state: get_model не удался: %s", exc)
            return ""
        return str(row[0]) if row else ""

    def set_model(self, user_id: int, model: str) -> bool:
        """Запоминает выбор модели (upsert)."""
        try:
            with self._db() as conn:
                conn.execute(
                    "INSERT INTO user_state (user_id, model, updated_at)"
                    " VALUES (?, ?, ?)"
                    " ON CONFLICT(user_id) DO UPDATE SET"
                    " model = excluded.model, updated_at = excluded.updated_at",
                    (int(user_id), str(model), _now_iso()),
                )
            return True
        except sqlite3.Error as exc:
            logger.warning("user_state: set_model не удался: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Внутреннее
    # ------------------------------------------------------------------
    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        """Короткоживущее соединение с гарантией close().

        ВАЖНО (ловушка Windows): ``with conn:`` у sqlite3 — это
        ТРАНЗАКЦИЯ, соединение при выходе не закрывается, и файл
        остаётся залочен. Поэтому closing + явный close в finally;
        ``with conn`` внутри — только для commit/rollback.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS messages ("
                    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    " user_id INTEGER NOT NULL,"
                    " role TEXT NOT NULL,"
                    " content TEXT NOT NULL,"
                    " created_at TEXT NOT NULL)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_user"
                    " ON messages(user_id, id)"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS user_state ("
                    " user_id INTEGER PRIMARY KEY,"
                    " model TEXT NOT NULL DEFAULT '',"
                    " updated_at TEXT NOT NULL)"
                )
                yield conn
        finally:
            conn.close()
