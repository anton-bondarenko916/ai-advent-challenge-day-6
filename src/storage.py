"""Хранилище контекста диалога на SQLite (стандартная библиотека).

Класс ContextStore отвечает за персистентность диалога DeepSeek Chat:

- таблица ``messages`` хранит сообщения ``{role, content}`` в порядке
  добавления и их принадлежность ветке (``branch_id`` + ``seq``);
- таблица ``branches`` хранит дерево веток диалога (checkpoint и развилки);
- таблица ``meta`` хранит параметры агента (JSON), статистику, конспект
  (summary), факты (facts), активную ветку и версию схемы;
- файл БД создаётся лениво — при первом реальном сохранении;
- операции используют короткоживущие соединения sqlite3, что безопасно
  при вызове из разных потоков (приложение выполняет запросы к API
  в отдельном потоке на каждый запрос);
- ошибки БД не пробрасываются наружу: логируются через utils.get_logger,
  вызывающий код продолжает работать с состоянием в памяти.

Схема v2 (ветки) добавляется **миграцией на месте**: старая БД версии 1
получает таблицу ``branches`` и колонки ``branch_id``/``seq``, вся прежняя
история попадает в корневую ветку ``main`` (id=1) — без потери данных.
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
# миграцию.
SCHEMA_VERSION = "2"

#: Идентификатор корневой ветки (создаётся всегда)
ROOT_BRANCH_ID = 1
#: Имя корневой ветки
ROOT_BRANCH_NAME = "main"

# Ключи в таблице meta
_KEY_PARAMS = "params"
_KEY_SCHEMA = "schema_version"
_KEY_STATS = "stats"
_KEY_SUMMARY = "summary"
_KEY_FACTS = "facts"
_KEY_ACTIVE_BRANCH = "active_branch"


def _facts_key(branch_id: int) -> str:
    """Ключ meta с фактами конкретной ветки (артефакты принадлежат ветке)."""
    return f"facts:{int(branch_id)}"


class ContextStore:
    """SQLite-хранилище истории диалога, веток и параметров агента.

    Attributes:
        db_path: путь к файлу БД (Path).
        enabled: если False, все операции становятся no-op.
        default_strategy: стратегия корневой ветки при первой миграции.

    Приватные атрибуты:
        _log: логгер модуля.
        _closed: флаг: close() уже вызывался.
    """

    def __init__(self, db_path: str | Path, enabled: bool = True,
                 default_strategy: str = "sliding_window") -> None:
        """Инициализирует хранилище (диск не трогается до первой записи).

        Args:
            db_path: путь к файлу БД (.db).
            enabled: False — хранилище отключено (все операции no-op).
            default_strategy: стратегия для корневой ветки (нужна при
                миграции старой БД и создании первой ветки).
        """
        self.db_path = Path(db_path)
        self.enabled = bool(enabled)
        self.default_strategy = str(default_strategy or "sliding_window")
        self._log = utils.get_logger("deepseek-chat.storage")
        self._closed = False

    # ------------------------------------------------------------------
    # Публичное API: состояние диалога
    # ------------------------------------------------------------------

    @property
    def exists(self) -> bool:
        """True, если файл БД уже создан на диске."""
        return self.db_path.exists()

    def load_state(self) -> Optional[Dict[str, Any]]:
        """Загружает диалог активной ветки, параметры, статистику и артефакты.

        Returns:
            Словарь {"messages", "params", "stats", "summary", "facts",
            "branches", "active_branch"} либо None, если: файла нет,
            хранилище отключено, схема несовместима или диалог пуст.

        Raises:
            Не выбрасывает: ошибки БД логируются, возвращается None.
        """
        if not self.enabled or self._closed or not self.exists:
            return None

        try:
            with self._session() as conn:
                has_v2 = self._ensure_schema(conn) == SCHEMA_VERSION
                schema = self._read_meta(conn, _KEY_SCHEMA)
                if schema is not None and schema not in ("1", SCHEMA_VERSION):
                    self._log.warning(
                        "Версия схемы БД %r не совпадает с ожидаемой %r — "
                        "сохранённый контекст не загружен.",
                        schema, SCHEMA_VERSION,
                    )
                    return None

                active_branch = self._read_active_branch(conn)
                if active_branch is None and has_v2:
                    active_branch = ROOT_BRANCH_ID

                if has_v2:
                    rows = conn.execute(
                        "SELECT role, content FROM messages "
                        "WHERE branch_id = ? "
                        "ORDER BY COALESCE(seq, id), id",
                        (active_branch if active_branch is not None
                         else ROOT_BRANCH_ID,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT role, content FROM messages ORDER BY id"
                    ).fetchall()
                if not rows:
                    return None

                params = self._read_json_meta(conn, _KEY_PARAMS, "параметры")
                stats = self._read_json_meta(conn, _KEY_STATS, "статистику токенов")
                summary = self._read_json_meta(conn, _KEY_SUMMARY, "конспект диалога")
                facts = self._read_json_meta(conn, _KEY_FACTS, "факты диалога")

                messages = [
                    {"role": role, "content": content} for role, content in rows
                ]
                return {
                    "messages": messages,
                    "params": params if isinstance(params, dict) else {},
                    "stats": stats if isinstance(stats, dict) else {},
                    "summary": summary if isinstance(summary, dict) else None,
                    "facts": facts if isinstance(facts, dict) else None,
                    "branches": self._load_branches(conn) if has_v2 else [],
                    "active_branch": active_branch,
                }
        except sqlite3.Error as exc:
            self._log.error("Ошибка загрузки контекста из БД: %s", exc)
            return None

    def save_state(self, messages: List[Message], params: Dict[str, Any],
                   stats: Optional[Dict[str, Any]] = None,
                   summary: Optional[Dict[str, Any]] = None,
                   branch_id: Optional[int] = None,
                   facts: Optional[Dict[str, Any]] = None) -> bool:
        """Перезаписывает сообщения ветки и параметры в одной транзакции.

        Полная перезапись сообщений ветки (≤ 50–60 строк) надёжнее точечных
        UPDATE: исключает рассинхрон с урезанием истории агента.

        Args:
            messages: список сообщений {role, content} (включая system).
            params: словарь параметров агента для сохранения.
            stats: накопленная статистика токенов; None — не изменять
                уже сохранённое значение (обратная совместимость).
            summary: конспект диалога; None — не изменять сохранённый,
                пустой словарь {} — явно стереть конспект.
            branch_id: ветка, в которую пишутся сообщения; None — активная.
            facts: факты диалога; None — не изменять сохранённые,
                пустой словарь {} — явно стереть факты.

        Returns:
            True при успехе, False при ошибке БД (ошибка залогирована).
        """
        if not self.enabled or self._closed:
            return False
        try:
            self._prepare_disk()
            with self._session() as conn:
                has_v2 = self._ensure_schema(conn) == SCHEMA_VERSION
                now = datetime.now().isoformat(timespec="seconds")
                target = branch_id
                if target is None:
                    target = self._read_active_branch(conn)
                if target is None:
                    target = ROOT_BRANCH_ID
                target = int(target)

                if has_v2:
                    conn.execute(
                        "DELETE FROM messages WHERE branch_id = ?", (target,)
                    )
                    conn.executemany(
                        "INSERT INTO messages(role, content, created_at, "
                        "branch_id, seq) VALUES (?, ?, ?, ?, ?)",
                        [(m.get("role", ""), m.get("content", ""), now, target, index)
                         for index, m in enumerate(messages)],
                    )
                    conn.execute(
                        "UPDATE branches SET updated_at = ? WHERE id = ?",
                        (now, target),
                    )
                else:
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
                if summary is not None:
                    self._write_meta(
                        conn, _KEY_SUMMARY, json.dumps(summary, ensure_ascii=False)
                    )
                if facts is not None:
                    self._write_meta(
                        conn, _KEY_FACTS, json.dumps(facts, ensure_ascii=False)
                    )
                    # Артефакты принадлежат ветке: держим копию по её id,
                    # чтобы переключение веток возвращало свои факты.
                    if has_v2:
                        self._write_meta(
                            conn, _facts_key(target),
                            json.dumps(facts, ensure_ascii=False),
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
        """Удаляет диалог, ветки, артефакты и параметры (схема остаётся).

        Returns:
            True при успехе, False при ошибке БД или отсутствии файла.
        """
        if not self.enabled or self._closed or not self.exists:
            return False
        try:
            with self._session() as conn:
                has_v2 = self._ensure_schema(conn) == SCHEMA_VERSION
                conn.execute("DELETE FROM messages")
                if has_v2:
                    conn.execute("DELETE FROM branches")
                conn.execute(
                    "DELETE FROM meta WHERE key IN (?, ?, ?, ?, ?)",
                    (_KEY_PARAMS, _KEY_STATS, _KEY_SUMMARY, _KEY_FACTS,
                     _KEY_ACTIVE_BRANCH),
                )
                conn.execute("DELETE FROM meta WHERE key LIKE 'facts:%'")
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
    # Публичное API: ветки диалога
    # ------------------------------------------------------------------

    def load_branches(self) -> List[Dict[str, Any]]:
        """Возвращает список веток (с числом сообщений в каждой).

        Returns:
            Список словарей {"id", "name", "parent_id", "forked_at",
            "strategy", "created_at", "updated_at", "messages"}.
            Пустой список, если веток нет/хранилище недоступно.
        """
        if not self.enabled or self._closed or not self.exists:
            return []
        try:
            with self._session() as conn:
                if self._ensure_schema(conn) != SCHEMA_VERSION:
                    return []
                return self._load_branches(conn)
        except sqlite3.Error as exc:
            self._log.error("Ошибка чтения веток из БД: %s", exc)
            return []

    def get_branch(self, branch_id: int) -> Optional[Dict[str, Any]]:
        """Возвращает одну ветку по идентификатору (или None)."""
        target = int(branch_id)
        for branch in self.load_branches():
            if int(branch.get("id", -1)) == target:
                return branch
        return None

    def save_branch(self, name: str, parent_id: Optional[int] = None,
                    forked_at: int = 0, strategy: str = "sliding_window",
                    branch_id: Optional[int] = None,
                    messages: Optional[List[Message]] = None) -> Optional[int]:
        """Создаёт новую ветку или обновляет существующую.

        Args:
            name: имя ветки.
            parent_id: родительская ветка (None — корневая).
            forked_at: индекс сообщения в родителе, от которого ответвление.
            strategy: подстратегия ветки (sliding_window | sticky_facts).
            branch_id: id существующей ветки для обновления (None — создать).
            messages: если передан, сообщения ветки полностью заменяются
                (используется при создании ветки-копии префикса).

        Returns:
            Идентификатор ветки либо None при ошибке.
        """
        if not self.enabled or self._closed:
            return None
        try:
            self._prepare_disk()
            with self._session() as conn:
                if self._ensure_schema(conn) != SCHEMA_VERSION:
                    return None
                now = datetime.now().isoformat(timespec="seconds")
                if branch_id is None:
                    cursor = conn.execute(
                        "INSERT INTO branches(name, parent_id, forked_at, "
                        "strategy, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (str(name), parent_id, int(forked_at),
                         str(strategy), now, now),
                    )
                    target = int(cursor.lastrowid or 0)
                else:
                    target = int(branch_id)
                    conn.execute(
                        "UPDATE branches SET name = ?, parent_id = ?, "
                        "forked_at = ?, strategy = ?, updated_at = ? "
                        "WHERE id = ?",
                        (str(name), parent_id, int(forked_at), str(strategy),
                         now, target),
                    )
                if messages is not None:
                    conn.execute(
                        "DELETE FROM messages WHERE branch_id = ?", (target,)
                    )
                    conn.executemany(
                        "INSERT INTO messages(role, content, created_at, "
                        "branch_id, seq) VALUES (?, ?, ?, ?, ?)",
                        [(m.get("role", ""), m.get("content", ""), now, target, index)
                         for index, m in enumerate(messages)],
                    )
            return target if target else None
        except (sqlite3.Error, OSError, TypeError, ValueError) as exc:
            self._log.error("Ошибка сохранения ветки в БД: %s", exc)
            return None

    def delete_branch(self, branch_id: int, cascade: bool = False) -> bool:
        """Удаляет ветку (и, при cascade, все вложенные ветки) с сообщениями.

        Args:
            branch_id: идентификатор удаляемой ветки.
            cascade: True — удалить также дочерние ветки рекурсивно.

        Returns:
            True при успехе.
        """
        if not self.enabled or self._closed or not self.exists:
            return False
        try:
            with self._session() as conn:
                if self._ensure_schema(conn) != SCHEMA_VERSION:
                    return False
                targets = [int(branch_id)]
                if cascade:
                    targets = self._branch_subtree(conn, int(branch_id))
                placeholders = ",".join("?" for _ in targets)
                conn.execute(
                    f"DELETE FROM messages WHERE branch_id IN ({placeholders})",
                    tuple(targets),
                )
                conn.execute(
                    f"DELETE FROM branches WHERE id IN ({placeholders})",
                    tuple(targets),
                )
                active = self._read_active_branch(conn)
                if active is not None and int(active) in targets:
                    conn.execute(
                        "DELETE FROM meta WHERE key = ?", (_KEY_ACTIVE_BRANCH,)
                    )
            return True
        except sqlite3.Error as exc:
            self._log.error("Ошибка удаления ветки из БД: %s", exc)
            return False

    def set_active_branch(self, branch_id: int) -> bool:
        """Запоминает активную ветку в meta.

        Returns:
            True при успехе, False при ошибке БД.
        """
        if not self.enabled or self._closed:
            return False
        try:
            self._prepare_disk()
            with self._session() as conn:
                self._ensure_schema(conn)
                self._write_meta(conn, _KEY_ACTIVE_BRANCH, str(int(branch_id)))
            return True
        except (sqlite3.Error, OSError, TypeError, ValueError) as exc:
            self._log.error("Ошибка сохранения активной ветки: %s", exc)
            return False

    def get_active_branch(self) -> Optional[int]:
        """Возвращает идентификатор активной ветки.

        Если ключ в meta ещё не записан (свежая или только что
        смигрированная БД), активной считается корневая ветка.
        """
        if not self.enabled or self._closed or not self.exists:
            return None
        try:
            with self._session() as conn:
                if self._ensure_schema(conn) != SCHEMA_VERSION:
                    return None
                return self._read_active_branch(conn) or ROOT_BRANCH_ID
        except sqlite3.Error as exc:
            self._log.error("Ошибка чтения активной ветки: %s", exc)
            return None

    def count_branch_messages(self, branch_id: int) -> int:
        """Число сообщений в ветке (0 при ошибке/отсутствии)."""
        if not self.enabled or self._closed or not self.exists:
            return 0
        try:
            with self._session() as conn:
                if self._ensure_schema(conn) != SCHEMA_VERSION:
                    return 0
                row = conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE branch_id = ?",
                    (int(branch_id),),
                ).fetchone()
            return int(row[0]) if row else 0
        except sqlite3.Error as exc:
            self._log.error("Ошибка подсчёта сообщений ветки: %s", exc)
            return 0

    def load_branch_messages(self, branch_id: int) -> List[Message]:
        """Читает сообщения конкретной ветки в порядке следования."""
        if not self.enabled or self._closed or not self.exists:
            return []
        try:
            with self._session() as conn:
                if self._ensure_schema(conn) != SCHEMA_VERSION:
                    return []
                rows = conn.execute(
                    "SELECT role, content FROM messages WHERE branch_id = ? "
                    "ORDER BY COALESCE(seq, id), id",
                    (int(branch_id),),
                ).fetchall()
            return [{"role": role, "content": content} for role, content in rows]
        except sqlite3.Error as exc:
            self._log.error("Ошибка чтения сообщений ветки: %s", exc)
            return []

    def save_branch_facts(self, branch_id: int,
                          facts: Optional[Dict[str, Any]]) -> bool:
        """Сохраняет факты ветки (артефакты ветки хранятся по её id).

        Args:
            branch_id: идентификатор ветки.
            facts: сериализованные факты; {} — явно стереть.

        Returns:
            True при успехе.
        """
        if not self.enabled or self._closed:
            return False
        try:
            self._prepare_disk()
            with self._session() as conn:
                if self._ensure_schema(conn) != SCHEMA_VERSION:
                    return False
                self._write_meta(
                    conn, _facts_key(branch_id),
                    json.dumps(facts or {}, ensure_ascii=False),
                )
            return True
        except (sqlite3.Error, OSError, TypeError, ValueError) as exc:
            self._log.error("Ошибка сохранения фактов ветки: %s", exc)
            return False

    def load_branch_facts(self, branch_id: int) -> Optional[Dict[str, Any]]:
        """Читает факты ветки (None, если их нет)."""
        if not self.enabled or self._closed or not self.exists:
            return None
        try:
            with self._session() as conn:
                if self._ensure_schema(conn) != SCHEMA_VERSION:
                    return None
                raw = self._read_meta(conn, _facts_key(branch_id))
                if raw is None and int(branch_id) == ROOT_BRANCH_ID:
                    raw = self._read_meta(conn, _KEY_FACTS)
                if not raw:
                    return None
                try:
                    parsed = json.loads(raw)
                except (ValueError, TypeError):
                    self._log.warning("Не удалось разобрать факты ветки.")
                    return None
                return parsed if isinstance(parsed, dict) else None
        except sqlite3.Error as exc:
            self._log.error("Ошибка чтения фактов ветки: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Приватные методы: соединение и схема
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

    @staticmethod
    def _table_names(conn: sqlite3.Connection) -> set:
        """Множество имён таблиц в БД."""
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        return {str(row[0]) for row in rows}

    @staticmethod
    def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
        """True, если у таблицы есть указанная колонка."""
        try:
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        except sqlite3.Error:
            return False
        return any(str(row[1]) == column for row in rows)

    def _ensure_schema(self, conn: sqlite3.Connection) -> str:
        """Создаёт или мигрирует схему БД.

        Свежая БД создаётся сразу в версии 2. Старая БД версии 1 получает
        таблицу ``branches`` и колонки ``branch_id``/``seq``; вся история
        проставляется в корневую ветку ``main``.

        Returns:
            Фактическая версия схемы: "2" при успехе, "1" — если ветки
            недоступны (тогда хранилище работает как раньше, одной историей).
        """
        tables = self._table_names(conn)
        fresh = "messages" not in tables or "meta" not in tables
        if fresh:
            self._create_tables(conn)
        else:
            version = self._read_meta(conn, _KEY_SCHEMA)
            if version not in (None, "1", SCHEMA_VERSION):
                self._log.warning(
                    "Версия схемы БД %r неизвестна — работаю в режиме "
                    "одной истории.", version,
                )
                return "1"
            self._upgrade_columns(conn)

        if not self._has_v2_schema(conn):
            self._log.warning(
                "Не удалось подготовить схему веток (v2) — работаю в режиме "
                "одной истории."
            )
            return "1"

        self._ensure_root_branch(conn)
        if self._read_meta(conn, _KEY_SCHEMA) != SCHEMA_VERSION:
            self._write_meta(conn, _KEY_SCHEMA, SCHEMA_VERSION)
        return SCHEMA_VERSION

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """Создаёт таблицы схемы v2 (идемпотентно)."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL,
                branch_id  INTEGER NOT NULL DEFAULT 1,
                seq        INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS branches (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT NOT NULL,
                parent_id  INTEGER NULL,
                forked_at  INTEGER NOT NULL DEFAULT 0,
                strategy   TEXT NOT NULL DEFAULT 'sliding_window',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS meta ("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_branch "
            "ON messages(branch_id, seq)"
        )

    def _upgrade_columns(self, conn: sqlite3.Connection) -> None:
        """Добавляет колонки веток и таблицу branches (миграция v1 → v2).

        Каждый шаг идемпотентен и не роняет приложение: при сбое отдельного
        DDL-оператора схема просто остаётся прежней, а хранилище продолжает
        работать в режиме одной истории.
        """
        for statement in (
            "ALTER TABLE messages ADD COLUMN branch_id INTEGER NOT NULL DEFAULT 1",
            "ALTER TABLE messages ADD COLUMN seq INTEGER",
        ):
            try:
                conn.execute(statement)
            except sqlite3.Error:
                pass  # колонка уже есть
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS branches (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    name       TEXT NOT NULL,
                    parent_id  INTEGER NULL,
                    forked_at  INTEGER NOT NULL DEFAULT 0,
                    strategy   TEXT NOT NULL DEFAULT 'sliding_window',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
        except sqlite3.Error as exc:
            self._log.error("Не удалось создать таблицу веток: %s", exc)
            return
        # Вся прежняя история принадлежит корневой ветке
        try:
            conn.execute(
                "UPDATE messages SET branch_id = ? "
                "WHERE branch_id IS NULL OR branch_id = 0",
                (ROOT_BRANCH_ID,),
            )
            # seq задаёт порядок внутри ветки; в старой БД все сообщения шли
            # подряд по id, поэтому id-1 сохраняет порядок и начинает с нуля.
            conn.execute("UPDATE messages SET seq = id - 1 WHERE seq IS NULL")
        except sqlite3.Error as exc:
            self._log.warning(
                "Не удалось проставить принадлежность старых сообщений: %s", exc
            )

    def _has_v2_schema(self, conn: sqlite3.Connection) -> bool:
        """True, если доступны и таблица веток, и колонки принадлежности."""
        return (
            "branches" in self._table_names(conn)
            and self._has_column(conn, "messages", "branch_id")
        )

    def _ensure_root_branch(self, conn: sqlite3.Connection) -> None:
        """Гарантирует существование корневой ветки main (id=1)."""
        row = conn.execute(
            "SELECT id FROM branches WHERE id = ?", (ROOT_BRANCH_ID,)
        ).fetchone()
        if row:
            return
        now = datetime.now().isoformat(timespec="seconds")
        try:
            conn.execute(
                "INSERT INTO branches(id, name, parent_id, forked_at, strategy, "
                "created_at, updated_at) VALUES (?, ?, NULL, 0, ?, ?, ?)",
                (ROOT_BRANCH_ID, ROOT_BRANCH_NAME, self.default_strategy,
                 now, now),
            )
        except sqlite3.Error as exc:
            self._log.error("Не удалось создать корневую ветку: %s", exc)

    # ------------------------------------------------------------------
    # Приватные методы: чтение/запись
    # ------------------------------------------------------------------

    def _read_json_meta(self, conn: sqlite3.Connection, key: str,
                        human: str) -> Any:
        """Читает JSON-значение из meta (None при отсутствии/ошибке)."""
        raw = self._read_meta(conn, key)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            self._log.warning("Не удалось разобрать %s из БД.", human)
            return None

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

    def _read_active_branch(self, conn: sqlite3.Connection) -> Optional[int]:
        """Читает активную ветку из meta (None, если неизвестна)."""
        raw = self._read_meta(conn, _KEY_ACTIVE_BRANCH)
        if not raw:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            self._log.warning("Некорректная активная ветка в meta: %r.", raw)
            return None

    def _load_branches(self, conn: sqlite3.Connection) -> List[Dict[str, Any]]:
        """Читает ветки с числом сообщений (для UI и менеджера веток)."""
        counts: Dict[int, int] = {}
        for branch_id, total in conn.execute(
            "SELECT branch_id, COUNT(*) FROM messages GROUP BY branch_id"
        ).fetchall():
            try:
                counts[int(branch_id)] = int(total)
            except (TypeError, ValueError):
                continue

        rows = conn.execute(
            "SELECT id, name, parent_id, forked_at, strategy, created_at, "
            "updated_at FROM branches ORDER BY id"
        ).fetchall()
        branches: List[Dict[str, Any]] = []
        for row in rows:
            branch_id = int(row[0])
            parent = row[2]
            branches.append({
                "id": branch_id,
                "name": str(row[1]),
                "parent_id": int(parent) if parent is not None else None,
                "forked_at": int(row[3] or 0),
                "strategy": str(row[4] or "sliding_window"),
                "created_at": row[5],
                "updated_at": row[6],
                "messages": counts.get(branch_id, 0),
            })
        return branches

    def _branch_subtree(self, conn: sqlite3.Connection,
                        branch_id: int) -> List[int]:
        """Возвращает ветку и всех её потомков (для cascade-удаления)."""
        edges: Dict[int, List[int]] = {}
        for row in conn.execute("SELECT id, parent_id FROM branches").fetchall():
            child = int(row[0])
            parent = row[1]
            if parent is None:
                continue
            edges.setdefault(int(parent), []).append(child)

        result: List[int] = []
        stack = [int(branch_id)]
        seen = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            result.append(current)
            stack.extend(edges.get(current, []))
        return result
