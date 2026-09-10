"""Хранилище контекста диалога на SQLite (стандартная библиотека).

Класс ContextStore отвечает за персистентность диалога DeepSeek Chat:
- таблица messages хранит сообщения {role, content} в порядке добавления;
- таблица meta хранит параметры агента (JSON) и версию схемы;
- файл БД создаётся лениво — при первом реальном сохранении;
- операции используют короткоживущие соединения sqlite3, что безопасно
  при вызове из разных потоков (приложение выполняет запросы к API
  в отдельном потоке на каждый запрос);
- ошибки БД не пробрасываются наружу: логируются через utils.get_logger,
  вызывающий код продолжает работать с состоянием в памяти.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from . import utils

# Тип сообщения истории: {"role": ..., "content": str}
Message = Dict[str, str]

# Версия схемы БД. При изменении структуры таблиц — увеличить и добавить
# миграцию; при несовпадении версии хранилище не читается.
SCHEMA_VERSION = "1"

# Ключи в таблице meta
_KEY_PARAMS = "params"
_KEY_SCHEMA = "schema_version"
_KEY_STATS = "stats"


class ContextStore:
    """SQLite-хранилище истории диалога и параметров агента.

    Attributes:
        db_path: путь к файлу БД (Path).
        enabled: если False, все операции становятся no-op.

    Приватные атрибуты:
        _log: логгер модуля.
        _closed: флаг: close() уже вызывался.
    """

    def __init__(self, db_path: str | Path, enabled: bool = True) -> None:
        """Инициализирует хранилище (диск не трогается до первой записи).

        Args:
            db_path: путь к файлу БД (.db).
            enabled: False — хранилище отключено (все операции no-op).
        """
        self.db_path = Path(db_path)
        self.enabled = bool(enabled)
        self._log = utils.get_logger("deepseek-chat.storage")
        self._closed = False

    # ------------------------------------------------------------------
    # Публичное API
    # ------------------------------------------------------------------

    @property
    def exists(self) -> bool:
        """True, если файл БД уже создан на диске."""
        return self.db_path.exists()

    def load_state(self) -> Optional[Dict[str, Any]]:
        """Загружает сохранённый диалог, параметры и статистику токенов.

        Returns:
            Словарь {"messages": [...], "params": {...}, "stats": {...}}
            либо None, если: файла нет, хранилище отключено, схема не
            совпадает или диалог пуст.

        Raises:
            Не выбрасывает: ошибки БД логируются, возвращается None.
        """
        if not self.enabled or self._closed or not self.exists:
            return None

        try:
            with self._session() as conn:
                schema = self._read_meta(conn, _KEY_SCHEMA)
                if schema is not None and schema != SCHEMA_VERSION:
                    self._log.warning(
                        "Версия схемы БД %r не совпадает с ожидаемой %r — "
                        "сохранённый контекст не загружен.",
                        schema, SCHEMA_VERSION,
                    )
                    return None

                rows = conn.execute(
                    "SELECT role, content FROM messages ORDER BY id"
                ).fetchall()
                if not rows:
                    return None

                params: Dict[str, Any] = {}
                params_raw = self._read_meta(conn, _KEY_PARAMS)
                if params_raw:
                    try:
                        parsed = json.loads(params_raw)
                        if isinstance(parsed, dict):
                            params = parsed
                    except (ValueError, TypeError):
                        self._log.warning("Не удалось разобрать параметры из БД.")

                stats: Dict[str, Any] = {}
                stats_raw = self._read_meta(conn, _KEY_STATS)
                if stats_raw:
                    try:
                        parsed_stats = json.loads(stats_raw)
                        if isinstance(parsed_stats, dict):
                            stats = parsed_stats
                    except (ValueError, TypeError):
                        self._log.warning(
                            "Не удалось разобрать статистику токенов из БД."
                        )

                messages = [
                    {"role": role, "content": content} for role, content in rows
                ]
                return {
                    "messages": messages,
                    "params": params,
                    "stats": stats,
                }
        except sqlite3.Error as exc:
            self._log.error("Ошибка загрузки контекста из БД: %s", exc)
            return None

    def save_state(self, messages: List[Message], params: Dict[str, Any],
                   stats: Optional[Dict[str, Any]] = None) -> bool:
        """Полностью перезаписывает диалог и параметры в одной транзакции.

        Полная перезапись таблицы messages (≤ 50–60 строк) надёжнее
        точечных UPDATE: исключает рассинхрон с урезанием истории агента.

        Args:
            messages: список сообщений {role, content} (включая system).
            params: словарь параметров агента для сохранения.
            stats: накопленная статистика токенов; None — не изменять
                уже сохранённое значение (обратная совместимость).

        Returns:
            True при успехе, False при ошибке БД (ошибка залогирована).
        """
        if not self.enabled or self._closed:
            return False
        try:
            self._prepare_disk()
            with self._session() as conn:
                self._ensure_schema(conn)
                now = datetime.now().isoformat(timespec="seconds")
                conn.execute("DELETE FROM messages")
                conn.executemany(
                    "INSERT INTO messages(role, content, created_at) "
                    "VALUES (?, ?, ?)",
                    [(m.get("role", ""), m.get("content", ""), now)
                     for m in messages],
                )
                self._write_meta(
                    conn, _KEY_PARAMS, json.dumps(params, ensure_ascii=False)
                )
                if stats is not None:
                    self._write_meta(
                        conn, _KEY_STATS, json.dumps(stats, ensure_ascii=False)
                    )
            return True
        except (sqlite3.Error, OSError, TypeError, ValueError) as exc:
            self._log.error("Ошибка сохранения контекста в БД: %s", exc)
            return False

    def save_params(self, params: Dict[str, Any]) -> bool:
        """Обновляет только параметры агента (история не трогается).

        Args:
            params: словарь параметров агента.

        Returns:
            True при успехе, False при ошибке БД.
        """
        if not self.enabled or self._closed:
            return False
        try:
            self._prepare_disk()
            with self._session() as conn:
                self._ensure_schema(conn)
                self._write_meta(
                    conn, _KEY_PARAMS, json.dumps(params, ensure_ascii=False)
                )
            return True
        except (sqlite3.Error, OSError) as exc:
            self._log.error("Ошибка сохранения параметров в БД: %s", exc)
            return False

    def clear(self) -> bool:
        """Удаляет диалог, параметры и статистику из БД (схема остаётся).

        Returns:
            True при успехе, False при ошибке БД или отсутствии файла.
        """
        if not self.enabled or self._closed or not self.exists:
            return False
        try:
            with self._session() as conn:
                self._ensure_schema(conn)
                conn.execute("DELETE FROM messages")
                conn.execute(
                    "DELETE FROM meta WHERE key IN (?, ?)",
                    (_KEY_PARAMS, _KEY_STATS),
                )
            return True
        except sqlite3.Error as exc:
            self._log.error("Ошибка очистки контекста в БД: %s", exc)
            return False

    def close(self) -> None:
        """Закрывает хранилище (идемпотентно).

        Соединения в этом классе короткоживущие, поэтому close только
        фиксирует состояние; повторные вызовы безопасны.
        """
        self._closed = True

    # ------------------------------------------------------------------
    # Приватные методы
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Открывает новое соединение с БД.

        Короткоживущее соединение на операцию безопасно использовать из
        любого потока: sqlite3 сам сериализует записи, а конфликтов
        «соединение создано в потоке A, используется в потоке B» не
        возникает (check_same_thread=True по умолчанию).

        Returns:
            sqlite3.Connection.
        """
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        """Контекст соединения: транзакция + гарантированное закрытие.

        Внутренний `with conn` открывает транзакцию при первой записи,
        фиксирует её при выходе и откатывает при исключении.
        """
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _prepare_disk(self) -> None:
        """Создаёт каталог БД, если его нет (ленивое создание файла)."""
        parent = self.db_path.parent
        if str(parent) and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        """Создаёт таблицы и записывает версию схемы (если их нет)."""
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
            (_KEY_SCHEMA, SCHEMA_VERSION),
        )

    def _read_meta(self, conn: sqlite3.Connection, key: str) -> Optional[str]:
        """Читает значение из таблицы meta (None, если ключа нет)."""
        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return str(row[0]) if row else None

    def _write_meta(self, conn: sqlite3.Connection, key: str, value: str) -> None:
        """Записывает/обновляет значение в таблице meta."""
        conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
