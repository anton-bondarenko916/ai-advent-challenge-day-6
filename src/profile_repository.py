"""Слой доступа к данным профилей пользователя (SQLite).

Класс :class:`ProfileRepository` — единственное место, где живут SQL-запросы
к таблицам персонализации:

- ``profiles`` — профили пользователей (стиль, формат, язык, ограничения);
- ``profile_agent_settings`` — настройки агента, привязанные к профилю 1:1;
- ``messages.profile_id`` — колонка связи истории чата с профилем
  (добавляется миграцией к существующей таблице сообщений, сама таблица
  не пересоздаётся).

Миграция (:meth:`ProfileRepository.migrate`) идемпотентна и запускается
при создании репозитория:

1. создаёт таблицы ``profiles`` и ``profile_agent_settings``, если их нет;
2. добавляет колонку ``profile_id`` в существующую таблицу ``messages``
   (если её ещё нет) и индекс ``idx_messages_profile``;
3. при первом запуске (нет ни одного профиля) создаёт профиль ``Default``
   с настройками из .env (DEFAULT_MODEL, DEFAULT_TEMPERATURE,
   DEFAULT_MAX_TOKENS, DEFAULT_SYSTEM_PROMPT), помечает его активным и
   привязывает к нему все существующие сообщения.

Все ошибки миграции пишутся в отдельный файл ``migration.log`` (в корне
проекта) и не прерывают запуск приложения.

Как и :class:`~src.storage.ContextStore`, репозиторий использует
короткоживущие соединения sqlite3 — это безопасно при вызовах из разных
потоков (запросы к API выполняются в отдельном потоке).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from . import utils
from .models import (
    DEFAULT_PROFILE_MODEL,
    DEFAULT_PROFILE_PROMPT,
    ProfileAgentSettings,
    UserProfile,
    parse_constraints,
    parse_context,
    serialize_constraints,
    serialize_context,
)

# Имя профиля, создаваемого миграцией и используемого как «безопасная» цель
# при удалении активного профиля.
DEFAULT_PROFILE_NAME = "Default"

# Стиль профиля Default (по ТЗ — «нейтральный»)
DEFAULT_PROFILE_STYLE = "нейтральный"

_MIGRATION_LOG_NAME = "migration.log"


def _migration_logger() -> Any:
    """Логгер миграции, пишущий в отдельный файл ``migration.log``.

    Файл лежит в корне проекта. Обработчик добавляется один раз
    (повторные вызовы безопасны); в консоль лог не дублируется.
    """
    import logging

    logger = logging.getLogger("deepseek-chat.profile_migration")
    if not logger.handlers:
        try:
            from .config import PROJECT_ROOT

            log_path = PROJECT_ROOT / _MIGRATION_LOG_NAME
        except Exception:  # pragma: no cover — конфиг всегда доступен
            log_path = Path(_MIGRATION_LOG_NAME)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


class ProfileRepository:
    """Репозиторий профилей: CRUD, настройки агента, миграция схемы.

    Attributes:
        db_path: путь к файлу БД (тот же файл, что у ContextStore —
            чтобы внешний ключ messages.profile_id работал).
        enabled: False переводит все операции в no-op (как ContextStore).

    Все SQL-запросы к таблицам персонализации инкапсулированы здесь:
    агент и UI работают только с :class:`UserProfile` /
    :class:`ProfileAgentSettings`.
    """

    def __init__(self, db_path: str | Path,
                 config: Optional[Dict[str, Any]] = None,
                 enabled: bool = True) -> None:
        """Создаёт репозиторий и выполняет миграцию схемы.

        Args:
            db_path: путь к файлу БД (рекомендуется тот же, что у
                хранилища контекста — ``data/chat_history.db``).
            config: конфигурация приложения (результат ``load_config``);
                используется для настроек профиля Default при первой
                миграции. None — значения по умолчанию.
            enabled: False — репозиторий отключён (все операции no-op).
        """
        self.db_path = Path(db_path)
        self.enabled = bool(enabled)
        self._log = utils.get_logger("deepseek-chat.profile_repo")
        self._migration_log = _migration_logger()
        self._config: Dict[str, Any] = dict(config or {})
        if self.enabled:
            self.migrate()

    # ------------------------------------------------------------------
    # Соединения
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Открывает короткоживущее соединение с включёнными FK."""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        """Контекст соединения: транзакция + гарантированное закрытие."""
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Миграция
    # ------------------------------------------------------------------

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

    def migrate(self) -> bool:
        """Создаёт схему персонализации и профиль Default (идемпотентно).

        Ошибки логируются в ``migration.log`` и не пробрасываются:
        приложение продолжает работать без персонализации.

        Returns:
            True при успехе, False при ошибке.
        """
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._session() as conn:
                self._create_tables(conn)
                self._add_profile_column(conn)
                self._create_default_profile(conn)
                self._attach_orphan_messages(conn)
            return True
        except (sqlite3.Error, OSError) as exc:
            self._migration_log.error("Ошибка миграции профилей: %s", exc)
            self._log.error("Ошибка миграции профилей: %s", exc)
            return False

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """Создаёт таблицы profiles и profile_agent_settings (идемпотентно)."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                name         TEXT NOT NULL UNIQUE,
                display_name TEXT,
                style        TEXT,
                format       TEXT,
                language     TEXT DEFAULT 'русский',
                constraints  TEXT,
                context      TEXT,
                is_active    INTEGER DEFAULT 0,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS profile_agent_settings (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id         INTEGER NOT NULL UNIQUE,
                model              TEXT DEFAULT 'deepseek-v4-flash',
                temperature        REAL DEFAULT 0.7,
                max_tokens         INTEGER DEFAULT 1000,
                base_system_prompt TEXT,
                FOREIGN KEY (profile_id)
                    REFERENCES profiles(id) ON DELETE CASCADE
            )
            """
        )

    def _add_profile_column(self, conn: sqlite3.Connection) -> None:
        """Добавляет ``messages.profile_id`` и индекс (если таблица есть).

        Существующая таблица сообщений не пересоздаётся. Внешний ключ
        ``REFERENCES profiles(id) ON DELETE CASCADE`` добавляется, если
        SQLite позволяет (значение по умолчанию NULL); при неудаче
        колонка добавляется без FK, а каскад выполняет сам репозиторий
        в :meth:`delete`.
        """
        if "messages" not in self._table_names(conn):
            # Таблицы сообщений ещё нет (хранилище не создавалось).
            # Создаём её сразу в схеме v3 ContextStore + колонка profile_id:
            # тогда изоляция истории по профилям работает с первого запуска
            # (ContextStore создаёт свои таблицы через IF NOT EXISTS).
            self._migration_log.info(
                "Таблица messages отсутствует — создаю схему v3 "
                "с колонкой profile_id."
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    branch_id   INTEGER NOT NULL DEFAULT 1,
                    seq         INTEGER,
                    pinned      INTEGER NOT NULL DEFAULT 0,
                    pin_note    TEXT,
                    profile_id  INTEGER REFERENCES profiles(id)
                                ON DELETE CASCADE
                )
                """
            )
        elif not self._has_column(conn, "messages", "profile_id"):
            for statement in (
                "ALTER TABLE messages ADD COLUMN profile_id "
                "INTEGER REFERENCES profiles(id) ON DELETE CASCADE",
                "ALTER TABLE messages ADD COLUMN profile_id INTEGER",
            ):
                try:
                    conn.execute(statement)
                    break
                except sqlite3.Error as exc:
                    self._migration_log.warning(
                        "ALTER TABLE messages (profile_id) не выполнен: %s", exc
                    )
        try:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_profile "
                "ON messages(profile_id, created_at)"
            )
        except sqlite3.Error as exc:
            self._migration_log.warning(
                "Не удалось создать индекс idx_messages_profile: %s", exc
            )

    def _default_settings_from_config(self) -> Dict[str, Any]:
        """Настройки профиля Default из конфигурации (.env) с fallback."""
        config = self._config

        def _str(key: str, default: str) -> str:
            value = config.get(key)
            return str(value) if value not in (None, "") else default

        def _float(key: str, default: float) -> float:
            try:
                return float(config.get(key, default))
            except (TypeError, ValueError):
                return default

        def _int(key: str, default: int) -> int:
            try:
                return int(config.get(key, default))
            except (TypeError, ValueError):
                return default

        model = _str("DEFAULT_MODEL", DEFAULT_PROFILE_MODEL)
        if model not in utils.AVAILABLE_MODELS:
            self._migration_log.warning(
                "Модель %r из конфигурации недоступна — беру %s.",
                model, DEFAULT_PROFILE_MODEL,
            )
            model = DEFAULT_PROFILE_MODEL
        return {
            "model": model,
            "temperature": _float("DEFAULT_TEMPERATURE", 0.7),
            "max_tokens": _int("DEFAULT_MAX_TOKENS", 1000),
            "base_system_prompt": _str(
                "DEFAULT_SYSTEM_PROMPT", DEFAULT_PROFILE_PROMPT
            ),
        }

    def _create_default_profile(self, conn: sqlite3.Connection) -> None:
        """Создаёт профиль Default (is_active=1), если профилей ещё нет."""
        row = conn.execute("SELECT COUNT(*) FROM profiles").fetchone()
        if row and int(row[0]) > 0:
            # Гарантируем наличие активного профиля даже в повреждённой БД
            active = conn.execute(
                "SELECT COUNT(*) FROM profiles WHERE is_active = 1"
            ).fetchone()
            if active and int(active[0]) == 0:
                conn.execute(
                    "UPDATE profiles SET is_active = 1 "
                    "WHERE id = (SELECT MIN(id) FROM profiles)"
                )
            return

        settings = self._default_settings_from_config()
        cursor = conn.execute(
            "INSERT INTO profiles(name, display_name, style, format, "
            "language, constraints, context, is_active) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
            (
                DEFAULT_PROFILE_NAME,
                DEFAULT_PROFILE_NAME,
                DEFAULT_PROFILE_STYLE,
                "",
                "русский",
                "[]",
                "{}",
            ),
        )
        profile_id = int(cursor.lastrowid or 0)
        conn.execute(
            "INSERT INTO profile_agent_settings(profile_id, model, "
            "temperature, max_tokens, base_system_prompt) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                profile_id,
                settings["model"],
                settings["temperature"],
                settings["max_tokens"],
                settings["base_system_prompt"],
            ),
        )
        self._migration_log.info(
            "Создан профиль %r (id=%d, модель=%s, temperature=%.2f, "
            "max_tokens=%d) с настройками из .env.",
            DEFAULT_PROFILE_NAME, profile_id, settings["model"],
            settings["temperature"], settings["max_tokens"],
        )

    def _attach_orphan_messages(self, conn: sqlite3.Connection) -> None:
        """Привязывает сообщения без профиля к активному (Default при миграции).

        Вызывается после создания профиля Default: все существующие
        сообщения наследуются активным профилем.
        """
        if "messages" not in self._table_names(conn):
            return
        if not self._has_column(conn, "messages", "profile_id"):
            return
        cursor = conn.execute(
            "UPDATE messages SET profile_id = "
            "(SELECT id FROM profiles WHERE is_active = 1 ORDER BY id LIMIT 1) "
            "WHERE profile_id IS NULL"
        )
        if cursor.rowcount:
            self._migration_log.info(
                "Привязано %d существующих сообщений к активному профилю.",
                cursor.rowcount,
            )

    # ------------------------------------------------------------------
    # Чтение профилей
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_profile(row: sqlite3.Row) -> UserProfile:
        """Собирает UserProfile из строки таблицы profiles."""
        return UserProfile(
            id=int(row["id"]),
            name=str(row["name"] or ""),
            display_name=str(row["display_name"] or ""),
            style=str(row["style"] or ""),
            format=str(row["format"] or ""),
            language=str(row["language"] or "русский"),
            constraints=parse_constraints(row["constraints"]),
            context=parse_context(row["context"]),
            is_active=bool(row["is_active"]),
        )

    @staticmethod
    def _row_to_settings(row: sqlite3.Row) -> ProfileAgentSettings:
        """Собирает ProfileAgentSettings из строки таблицы настроек."""
        def _float(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _int(value: Any, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        return ProfileAgentSettings(
            profile_id=int(row["profile_id"]),
            model=str(row["model"] or DEFAULT_PROFILE_MODEL),
            temperature=_float(row["temperature"], 0.7),
            max_tokens=_int(row["max_tokens"], 1000),
            base_system_prompt=str(row["base_system_prompt"] or ""),
        )

    def get_all(self) -> List[UserProfile]:
        """Список всех профилей (в порядке создания)."""
        if not self.enabled:
            return []
        try:
            with self._session() as conn:
                rows = conn.execute(
                    "SELECT * FROM profiles ORDER BY id"
                ).fetchall()
            return [self._row_to_profile(row) for row in rows]
        except sqlite3.Error as exc:
            self._log.error("Ошибка чтения профилей: %s", exc)
            return []

    def get_by_id(self, profile_id: int) -> Optional[UserProfile]:
        """Профиль по идентификатору (None, если не найден)."""
        if not self.enabled:
            return None
        try:
            with self._session() as conn:
                row = conn.execute(
                    "SELECT * FROM profiles WHERE id = ?", (int(profile_id),)
                ).fetchone()
            return self._row_to_profile(row) if row else None
        except sqlite3.Error as exc:
            self._log.error("Ошибка чтения профиля id=%s: %s", profile_id, exc)
            return None

    def get_active(self) -> Optional[UserProfile]:
        """Активный профиль (is_active=1) или None, если профилей нет."""
        if not self.enabled:
            return None
        try:
            with self._session() as conn:
                row = conn.execute(
                    "SELECT * FROM profiles WHERE is_active = 1 "
                    "ORDER BY id LIMIT 1"
                ).fetchone()
            return self._row_to_profile(row) if row else None
        except sqlite3.Error as exc:
            self._log.error("Ошибка чтения активного профиля: %s", exc)
            return None

    def get_by_name(self, name: str) -> Optional[UserProfile]:
        """Профиль по уникальному имени (None, если не найден)."""
        if not self.enabled:
            return None
        try:
            with self._session() as conn:
                row = conn.execute(
                    "SELECT * FROM profiles WHERE name = ?", (str(name),)
                ).fetchone()
            return self._row_to_profile(row) if row else None
        except sqlite3.Error as exc:
            self._log.error("Ошибка чтения профиля %r: %s", name, exc)
            return None

    # ------------------------------------------------------------------
    # Изменение профилей
    # ------------------------------------------------------------------

    def create(self, profile: UserProfile,
               settings: Optional[ProfileAgentSettings] = None
               ) -> UserProfile:
        """Создаёт профиль (и, при переданных, его настройки агента).

        Args:
            profile: профиль; ``id`` игнорируется, ``name`` обязан быть
                непустым и уникальным.
            settings: настройки агента; None — создать со значениями
                по умолчанию.

        Returns:
            Профиль с заполненным ``id``.

        Raises:
            ValueError: если имя пустое или уже занято.
            sqlite3.IntegrityError: при нарушении ограничений БД.
        """
        name = str(profile.name or "").strip()
        if not name:
            raise ValueError("Имя профиля не может быть пустым.")
        if self.get_by_name(name) is not None:
            raise ValueError(f"Профиль с именем {name!r} уже существует.")
        with self._session() as conn:
            cursor = conn.execute(
                "INSERT INTO profiles(name, display_name, style, format, "
                "language, constraints, context, is_active) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                (
                    name,
                    str(profile.display_name or ""),
                    str(profile.style or ""),
                    str(profile.format or ""),
                    str(profile.language or "русский"),
                    serialize_constraints(profile.constraints),
                    serialize_context(profile.context),
                ),
            )
            profile_id = int(cursor.lastrowid or 0)
            effective = settings or ProfileAgentSettings(profile_id=profile_id)
            conn.execute(
                "INSERT INTO profile_agent_settings(profile_id, model, "
                "temperature, max_tokens, base_system_prompt) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    profile_id,
                    str(effective.model or DEFAULT_PROFILE_MODEL),
                    float(effective.temperature),
                    int(effective.max_tokens),
                    str(effective.base_system_prompt or ""),
                ),
            )
        profile.id = profile_id
        return profile

    def update(self, profile: UserProfile) -> UserProfile:
        """Обновляет поля профиля и ``updated_at``.

        Args:
            profile: профиль с заполненным ``id``.

        Returns:
            Переданный профиль (без изменений).

        Raises:
            ValueError: если id не задан или имя занято другим профилем.
        """
        if not profile.id:
            raise ValueError("Нельзя обновить профиль без id.")
        existing = self.get_by_name(profile.name)
        if existing is not None and existing.id != profile.id:
            raise ValueError(
                f"Профиль с именем {profile.name!r} уже существует."
            )
        with self._session() as conn:
            conn.execute(
                "UPDATE profiles SET name = ?, display_name = ?, style = ?, "
                "format = ?, language = ?, constraints = ?, context = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (
                    str(profile.name or "").strip(),
                    str(profile.display_name or ""),
                    str(profile.style or ""),
                    str(profile.format or ""),
                    str(profile.language or "русский"),
                    serialize_constraints(profile.constraints),
                    serialize_context(profile.context),
                    int(profile.id),
                ),
            )
        return profile

    def delete(self, profile_id: int) -> bool:
        """Удаляет профиль каскадно (настройки + история чата).

        Явно удаляет связанные строки настроек и сообщений — каскад
        гарантируется даже если внешний ключ в ``messages`` не удалось
        добавить миграцией. Последний оставшийся профиль удалить нельзя.

        Args:
            profile_id: идентификатор профиля.

        Returns:
            True при успехе, False при ошибке/отсутствии профиля.
        """
        if not self.enabled:
            return False
        try:
            with self._session() as conn:
                tables = self._table_names(conn)
                count = conn.execute(
                    "SELECT COUNT(*) FROM profiles"
                ).fetchone()
                if not count or int(count[0]) <= 1:
                    self._log.warning(
                        "Нельзя удалить последний профиль (id=%s).",
                        profile_id,
                    )
                    return False
                if not conn.execute(
                    "SELECT 1 FROM profiles WHERE id = ?", (int(profile_id),)
                ).fetchone():
                    return False
                conn.execute(
                    "DELETE FROM profile_agent_settings WHERE profile_id = ?",
                    (int(profile_id),),
                )
                if "messages" in tables and self._has_column(
                        conn, "messages", "profile_id"):
                    conn.execute(
                        "DELETE FROM messages WHERE profile_id = ?",
                        (int(profile_id),),
                    )
                conn.execute(
                    "DELETE FROM profiles WHERE id = ?", (int(profile_id),)
                )
                # Если удалили активный — активируем первый оставшийся
                active = conn.execute(
                    "SELECT COUNT(*) FROM profiles WHERE is_active = 1"
                ).fetchone()
                if int(active[0]) == 0:
                    conn.execute(
                        "UPDATE profiles SET is_active = 1 "
                        "WHERE id = (SELECT MIN(id) FROM profiles)"
                    )
            return True
        except sqlite3.Error as exc:
            self._log.error("Ошибка удаления профиля id=%s: %s", profile_id, exc)
            return False

    def set_active(self, profile_id: int) -> bool:
        """Делает профиль активным (остальные деактивируются).

        Args:
            profile_id: идентификатор профиля.

        Returns:
            True при успехе, False если профиль не найден.
        """
        if not self.enabled:
            return False
        try:
            with self._session() as conn:
                exists = conn.execute(
                    "SELECT 1 FROM profiles WHERE id = ?", (int(profile_id),)
                ).fetchone()
                if not exists:
                    return False
                conn.execute("UPDATE profiles SET is_active = 0")
                conn.execute(
                    "UPDATE profiles SET is_active = 1, "
                    "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (int(profile_id),),
                )
            return True
        except sqlite3.Error as exc:
            self._log.error(
                "Ошибка активации профиля id=%s: %s", profile_id, exc
            )
            return False

    # ------------------------------------------------------------------
    # Настройки агента, привязанные к профилю
    # ------------------------------------------------------------------

    def get_settings(self, profile_id: int) -> ProfileAgentSettings:
        """Настройки агента профиля (создаёт запись со значениями по умолчанию).

        Args:
            profile_id: идентификатор профиля.

        Returns:
            ProfileAgentSettings; если записи нет, она создаётся со
            значениями по умолчанию (и привязкой к .env-конфигурации).
        """
        defaults = ProfileAgentSettings(
            profile_id=int(profile_id),
            **{
                key: value
                for key, value in self._default_settings_from_config().items()
            },
        )
        if not self.enabled:
            return defaults
        try:
            with self._session() as conn:
                row = conn.execute(
                    "SELECT * FROM profile_agent_settings WHERE profile_id = ?",
                    (int(profile_id),),
                ).fetchone()
                if row is None:
                    conn.execute(
                        "INSERT INTO profile_agent_settings(profile_id, model, "
                        "temperature, max_tokens, base_system_prompt) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            int(profile_id),
                            defaults.model,
                            defaults.temperature,
                            defaults.max_tokens,
                            defaults.base_system_prompt,
                        ),
                    )
                    return defaults
            return self._row_to_settings(row)
        except sqlite3.Error as exc:
            self._log.error(
                "Ошибка чтения настроек профиля id=%s: %s", profile_id, exc
            )
            return defaults

    def update_settings(self, settings: ProfileAgentSettings) -> bool:
        """Сохраняет настройки агента профиля (INSERT или UPDATE).

        Args:
            settings: настройки с заполненным ``profile_id``.

        Returns:
            True при успехе.
        """
        if not self.enabled or not settings.profile_id:
            return False
        try:
            with self._session() as conn:
                conn.execute(
                    "INSERT INTO profile_agent_settings(profile_id, model, "
                    "temperature, max_tokens, base_system_prompt) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(profile_id) DO UPDATE SET "
                    "model = excluded.model, "
                    "temperature = excluded.temperature, "
                    "max_tokens = excluded.max_tokens, "
                    "base_system_prompt = excluded.base_system_prompt",
                    (
                        int(settings.profile_id),
                        str(settings.model or DEFAULT_PROFILE_MODEL),
                        float(settings.temperature),
                        int(settings.max_tokens),
                        str(settings.base_system_prompt or ""),
                    ),
                )
            return True
        except (sqlite3.Error, TypeError, ValueError) as exc:
            self._log.error(
                "Ошибка сохранения настроек профиля id=%s: %s",
                settings.profile_id, exc,
            )
            return False

    # ------------------------------------------------------------------
    # Служебные запросы (для тестов и диагностики)
    # ------------------------------------------------------------------

    def count_profile_messages(self, profile_id: int) -> int:
        """Число сообщений истории, привязанных к профилю (0 при ошибке)."""
        if not self.enabled:
            return 0
        try:
            with self._session() as conn:
                if "messages" not in self._table_names(conn):
                    return 0
                if not self._has_column(conn, "messages", "profile_id"):
                    return 0
                row = conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE profile_id = ?",
                    (int(profile_id),),
                ).fetchone()
            return int(row[0]) if row else 0
        except sqlite3.Error as exc:
            self._log.error("Ошибка подсчёта сообщений профиля: %s", exc)
            return 0
