"""Мелкое key-value хранилище состояния UI (SQLite, таблица app_state).

Сюда складываются настройки интерфейса, которые должны переживать
перезапуск приложения, но не заслуживают отдельной сущности в домене:
пока это состояние сворачивания MCP-панели.

Хранилище сознательно «неубиваемое»: любая ошибка SQLite (нет прав,
занятый файл, кривой путь) логируется и превращается в None/False —
потеря состояния панели никогда не должна ломать запуск приложения.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from . import utils

logger = utils.get_logger("deepseek-chat.appstate")


class AppStateStore:
    """Доступ к таблице ``app_state`` (ключ → строковое значение).

    Attributes:
        db_path: путь к файлу БД SQLite.
    """

    def __init__(self, db_path: str | Path) -> None:
        """Запоминает путь и создаёт таблицу, если БД доступна.

        Args:
            db_path: путь к файлу БД (каталог создаётся при необходимости).
        """
        self.db_path = Path(db_path)
        self._ensure_table()

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[str]:
        """Возвращает значение по ключу или None.

        Args:
            key: имя параметра (например, ``mcp_panel_collapsed``).

        Returns:
            Строковое значение; None — ключа нет или БД недоступна.
        """
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT value FROM app_state WHERE key = ?", (key,)
                ).fetchone()
                return row[0] if row else None
            finally:
                # ВАЖНО: соединение закрывается всегда — иначе на Windows
                # файл БД остаётся заблокированным (контекстный менеджер
                # sqlite3.Connection закрывает только транзакцию).
                conn.close()
        except sqlite3.Error as exc:
            logger.warning("app_state: чтение %s не удалось: %s", key, exc)
            return None

    def set(self, key: str, value: str) -> bool:
        """Сохраняет значение (upsert).

        Args:
            key: имя параметра.
            value: строковое значение.

        Returns:
            True — сохранено; False — БД недоступна (не фатально).
        """
        try:
            conn = self._connect()
            try:
                with conn:  # commit/rollback транзакции
                    conn.execute(
                        "INSERT INTO app_state (key, value) VALUES (?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (key, value),
                    )
                return True
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.warning("app_state: запись %s не удалась: %s", key, exc)
            return False

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Читает логический параметр («1»/«0»).

        Args:
            key: имя параметра.
            default: значение при отсутствии ключа или ошибке БД.
        """
        raw = self.get(key)
        if raw is None:
            return default
        return raw.strip() == "1"

    def set_bool(self, key: str, value: bool) -> bool:
        """Пишет логический параметр («1»/«0»)."""
        return self.set(key, "1" if value else "0")

    # ------------------------------------------------------------------
    # Внутреннее
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Открывает соединение (короткоживущее, на одну операцию)."""
        conn = sqlite3.connect(str(self.db_path), timeout=2.0)
        conn.execute("PRAGMA busy_timeout = 2000")
        return conn

    def _ensure_table(self) -> None:
        """Создаёт таблицу app_state (миграция не нужна: схема тривиальна)."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS app_state ("
                        "key TEXT PRIMARY KEY, "
                        "value TEXT NOT NULL)"
                    )
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.warning(
                "app_state: таблица не создана (%s) — состояние UI не "
                "будет сохраняться", exc,
            )
