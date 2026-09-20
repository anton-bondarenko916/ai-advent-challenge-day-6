"""Слой доступа к данным инвариантов (SQLite).

Класс :class:`InvariantRepository` — единственное место, где живут
SQL-запросы к таблицам подсистемы инвариантов:

- ``invariant_categories`` — категории (группировка и приоритеты);
- ``invariants`` — сами правила (scope: global | profile);
- ``invariant_violations`` — лог нарушений (аналитика и отладка).

Миграция (:meth:`InvariantRepository.migrate`) идемпотентна и запускается
при создании репозитория:

1. создаёт три таблицы и индексы (см. схему из ТЗ), если их нет;
2. при первом запуске добавляет 7 предустановленных категорий
   (security/architecture/stack помечены ``is_system = 1`` — их удалить
   нельзя);
3. если у существующих профилей ещё нет ни одного инварианта — создаёт
   1–2 базовых глобальных инварианта для примера (их можно удалить).

Репозиторий хранит данные в том же файле БД, что и профили
(``data/chat_history.db``): на ``profiles(id)`` есть внешние ключи.
Как и :class:`~src.profile_repository.ProfileRepository`, использует
короткоживущие соединения sqlite3 — это безопасно при вызовах из разных
потоков. Ошибки миграции пишутся в ``migration.log`` и не прерывают запуск.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from . import utils
from .invariant_models import (
    SCOPE_GLOBAL,
    SCOPE_PROFILE,
    VALID_SCOPES,
    Invariant,
    InvariantCategory,
    InvariantViolation,
)

_MIGRATION_LOG_NAME = "migration.log"

#: Предустановленные категории (name, display_name, description, priority,
#: is_system) — добавляются при первой миграции.
PRESET_CATEGORIES = (
    ("security", "Безопасность",
     "Секреты, ключи, деструктивные операции", 10, 1),
    ("architecture", "Архитектура",
     "Слои, модули, точки расширения", 20, 1),
    ("stack", "Стек и технологии",
     "Языки, фреймворки, библиотеки", 30, 1),
    ("data", "Данные и хранение",
     "Форматы, схемы, места хранения", 40, 0),
    ("business", "Бизнес-правила",
     "Доменные правила проекта", 50, 0),
    ("style", "Стиль кода",
     "Оформление и соглашения", 60, 0),
    ("custom", "Пользовательские",
     "Всё, что не попало в другие категории", 100, 0),
)

#: Примеры инвариантов для первого запуска (создаются, только если
#: инвариантов ещё нет; их можно удалить в UI).
SAMPLE_INVARIANTS = (
    ("security", "Секреты только в .env",
     "Пример инварианта — можно отредактировать или удалить.",
     "API-ключи и пароли хранятся только в .env. "
     "Никогда не выводить и не логировать секреты.",
     SCOPE_GLOBAL),
    ("stack", "Вся работа с LLM — через DeepSeekAgent",
     "Пример инварианта — можно отредактировать или удалить.",
     "Любые вызовы LLM выполняются только через класс DeepSeekAgent; "
     "прямые вызовы API из UI и скриптов запрещены.",
     SCOPE_GLOBAL),
)


def _migration_logger() -> Any:
    """Логгер миграции, пишущий в отдельный файл ``migration.log``.

    Единый обработчик с репозиторием профилей: файл один, обработчик
    добавляется один раз (повторные вызовы безопасны).
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


class InvariantRepository:
    """Репозиторий инвариантов: CRUD, лог нарушений, миграция схемы.

    Attributes:
        db_path: путь к файлу БД (тот же файл, что у хранилища контекста
            и репозитория профилей).
        enabled: False переводит все операции в no-op.

    Все SQL-запросы к таблицам инвариантов инкапсулированы здесь: агент,
    валидатор и UI работают только с dataclass-моделями из
    :mod:`src.invariant_models`.
    """

    def __init__(self, db_path: str | Path,
                 enabled: bool = True) -> None:
        """Создаёт репозиторий и выполняет миграцию схемы.

        Args:
            db_path: путь к файлу БД (рекомендуется тот же, что у
                хранилища контекста — ``data/chat_history.db``).
            enabled: False — репозиторий отключён (все операции no-op).
        """
        self.db_path = Path(db_path)
        self.enabled = bool(enabled)
        self._log = utils.get_logger("deepseek-chat.invariant_repo")
        self._migration_log = _migration_logger()
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

    def migrate(self) -> bool:
        """Создаёт схему инвариантов и базовые данные (идемпотентно).

        Ошибки логируются в ``migration.log`` и не пробрасываются:
        приложение продолжает работать без инвариантов.

        Returns:
            True при успехе, False при ошибке.
        """
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._session() as conn:
                self._create_tables(conn)
                self._create_preset_categories(conn)
                self._create_sample_invariants(conn)
            return True
        except (sqlite3.Error, OSError) as exc:
            self._migration_log.error("Ошибка миграции инвариантов: %s", exc)
            self._log.error("Ошибка миграции инвариантов: %s", exc)
            return False

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """Создаёт таблицы и индексы (идемпотентно, по схеме из ТЗ)."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS invariant_categories (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                name         TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                description  TEXT,
                priority     INTEGER DEFAULT 100,
                is_system    INTEGER DEFAULT 0,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS invariants (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                category_id INTEGER NOT NULL,
                title       TEXT NOT NULL,
                description TEXT,
                rule        TEXT NOT NULL,
                scope       TEXT NOT NULL DEFAULT 'profile',
                profile_id  INTEGER,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (category_id)
                    REFERENCES invariant_categories(id) ON DELETE CASCADE,
                FOREIGN KEY (profile_id)
                    REFERENCES profiles(id) ON DELETE CASCADE,
                CHECK (
                    (scope = 'global' AND profile_id IS NULL) OR
                    (scope = 'profile' AND profile_id IS NOT NULL)
                )
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS invariant_violations (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                invariant_id   INTEGER NOT NULL,
                profile_id     INTEGER,
                user_query     TEXT,
                agent_response TEXT,
                regenerated    INTEGER DEFAULT 0,
                resolved       INTEGER DEFAULT 0,
                detected_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (invariant_id)
                    REFERENCES invariants(id) ON DELETE CASCADE,
                FOREIGN KEY (profile_id)
                    REFERENCES profiles(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_invariants_category "
            "ON invariants(category_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_invariants_scope "
            "ON invariants(scope, profile_id)"
        )

    def _create_preset_categories(self, conn: sqlite3.Connection) -> None:
        """Добавляет предустановленные категории (только отсутствующие).

        INSERT OR IGNORE: повторные миграции и частично заполненная БД
        не приводят к дублям (``name`` уникален).
        """
        added = 0
        for name, display_name, description, priority, is_system \
                in PRESET_CATEGORIES:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO invariant_categories"
                "(name, display_name, description, priority, is_system) "
                "VALUES (?, ?, ?, ?, ?)",
                (name, display_name, description, int(priority),
                 int(is_system)),
            )
            added += cursor.rowcount
        if added:
            self._migration_log.info(
                "Добавлено предустановленных категорий инвариантов: %d.",
                added,
            )

    def _create_sample_invariants(self, conn: sqlite3.Connection) -> None:
        """Создаёт 1–2 базовых инварианта-примера при первом запуске.

        Условие из ТЗ: только если инвариантов ещё нет вообще (их можно
        удалить в UI без вреда).
        """
        row = conn.execute("SELECT COUNT(*) FROM invariants").fetchone()
        if row and int(row[0]) > 0:
            return
        added = 0
        for category_name, title, description, rule, scope \
                in SAMPLE_INVARIANTS:
            category = conn.execute(
                "SELECT id FROM invariant_categories WHERE name = ?",
                (category_name,),
            ).fetchone()
            if category is None:
                continue
            conn.execute(
                "INSERT INTO invariants(category_id, title, description, "
                "rule, scope, profile_id) VALUES (?, ?, ?, ?, ?, NULL)",
                (int(category["id"]), title, description, rule, scope),
            )
            added += 1
        if added:
            self._migration_log.info(
                "Создано базовых инвариантов-примеров: %d (можно удалить).",
                added,
            )

    # ------------------------------------------------------------------
    # Категории: чтение
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_category(row: sqlite3.Row) -> InvariantCategory:
        """Собирает InvariantCategory из строки таблицы."""
        return InvariantCategory(
            id=int(row["id"]),
            name=str(row["name"] or ""),
            display_name=str(row["display_name"] or ""),
            description=str(row["description"] or ""),
            priority=int(row["priority"] if row["priority"] is not None
                         else 100),
            is_system=bool(row["is_system"]),
        )

    @staticmethod
    def _row_to_invariant(row: sqlite3.Row) -> Invariant:
        """Собирает Invariant из строки таблицы."""
        profile_id = row["profile_id"]
        return Invariant(
            id=int(row["id"]),
            category_id=int(row["category_id"]),
            title=str(row["title"] or ""),
            description=str(row["description"] or ""),
            rule=str(row["rule"] or ""),
            scope=str(row["scope"] or SCOPE_PROFILE),
            profile_id=int(profile_id) if profile_id is not None else None,
        )

    def get_all_categories(self) -> List[InvariantCategory]:
        """Список категорий по приоритету (меньше = важнее = выше)."""
        if not self.enabled:
            return []
        try:
            with self._session() as conn:
                rows = conn.execute(
                    "SELECT * FROM invariant_categories "
                    "ORDER BY priority, id"
                ).fetchall()
            return [self._row_to_category(row) for row in rows]
        except sqlite3.Error as exc:
            self._log.error("Ошибка чтения категорий инвариантов: %s", exc)
            return []

    def get_category_by_id(self, category_id: int
                           ) -> Optional[InvariantCategory]:
        """Категория по идентификатору (None, если не найдена)."""
        if not self.enabled:
            return None
        try:
            with self._session() as conn:
                row = conn.execute(
                    "SELECT * FROM invariant_categories WHERE id = ?",
                    (int(category_id),),
                ).fetchone()
            return self._row_to_category(row) if row else None
        except sqlite3.Error as exc:
            self._log.error(
                "Ошибка чтения категории id=%s: %s", category_id, exc
            )
            return None

    # ------------------------------------------------------------------
    # Категории: изменение
    # ------------------------------------------------------------------

    def create_category(self, category: InvariantCategory
                        ) -> InvariantCategory:
        """Создаёт категорию.

        Args:
            category: категория; ``id`` игнорируется.

        Returns:
            Категория с заполненным ``id``.

        Raises:
            ValueError: если имя/display_name пустые или name занято.
        """
        name = str(category.name or "").strip()
        display_name = str(category.display_name or "").strip()
        if not name:
            raise ValueError("Имя категории не может быть пустым.")
        if not display_name:
            raise ValueError("Отображаемое имя категории не может быть пустым.")
        if not self.enabled:
            raise ValueError("Репозиторий инвариантов отключён.")
        with self._session() as conn:
            exists = conn.execute(
                "SELECT 1 FROM invariant_categories WHERE name = ?",
                (name,),
            ).fetchone()
            if exists:
                raise ValueError(
                    f"Категория с именем {name!r} уже существует."
                )
            cursor = conn.execute(
                "INSERT INTO invariant_categories"
                "(name, display_name, description, priority, is_system) "
                "VALUES (?, ?, ?, ?, 0)",
                (
                    name,
                    display_name,
                    str(category.description or ""),
                    self._validate_priority(category.priority),
                ),
            )
            category.id = int(cursor.lastrowid or 0)
        return category

    def update_category(self, category: InvariantCategory) -> None:
        """Обновляет категорию (name, display_name, description, priority).

        Raises:
            ValueError: если нет id, поля пустые или name занято другой
                категорией.
        """
        if not category.id:
            raise ValueError("Нельзя обновить категорию без id.")
        name = str(category.name or "").strip()
        display_name = str(category.display_name or "").strip()
        if not name:
            raise ValueError("Имя категории не может быть пустым.")
        if not display_name:
            raise ValueError("Отображаемое имя категории не может быть пустым.")
        with self._session() as conn:
            exists = conn.execute(
                "SELECT 1 FROM invariant_categories WHERE name = ? AND id != ?",
                (name, int(category.id)),
            ).fetchone()
            if exists:
                raise ValueError(
                    f"Категория с именем {name!r} уже существует."
                )
            conn.execute(
                "UPDATE invariant_categories SET name = ?, display_name = ?, "
                "description = ?, priority = ? WHERE id = ?",
                (
                    name,
                    display_name,
                    str(category.description or ""),
                    self._validate_priority(category.priority),
                    int(category.id),
                ),
            )

    def delete_category(self, category_id: int) -> bool:
        """Удаляет категорию вместе с её инвариантами (CASCADE).

        Системные категории (``is_system = 1``) удалить нельзя.

        Returns:
            True при успехе; False — если категория не найдена, отключён
            репозиторий или категория системная.
        """
        if not self.enabled:
            return False
        try:
            with self._session() as conn:
                row = conn.execute(
                    "SELECT is_system FROM invariant_categories WHERE id = ?",
                    (int(category_id),),
                ).fetchone()
                if row is None:
                    return False
                if bool(row["is_system"]):
                    self._log.warning(
                        "Нельзя удалить системную категорию id=%s.",
                        category_id,
                    )
                    return False
                conn.execute(
                    "DELETE FROM invariants WHERE category_id = ?",
                    (int(category_id),),
                )
                conn.execute(
                    "DELETE FROM invariant_categories WHERE id = ?",
                    (int(category_id),),
                )
            return True
        except sqlite3.Error as exc:
            self._log.error(
                "Ошибка удаления категории id=%s: %s", category_id, exc
            )
            return False

    @staticmethod
    def _validate_priority(priority: Any) -> int:
        """Приводит приоритет к целому в диапазоне [1, 1000]."""
        try:
            value = int(priority)
        except (TypeError, ValueError):
            return 100
        return max(1, min(1000, value))

    # ------------------------------------------------------------------
    # Инварианты: чтение
    # ------------------------------------------------------------------

    def get_invariants_for_profile(
        self, profile_id: Optional[int]
    ) -> List[Invariant]:
        """Инварианты, действующие для профиля: global + profile.

        Args:
            profile_id: идентификатор профиля; None — только глобальные
                (агент без персонализации).

        Returns:
            Глобальные инварианты, затем профильные (внутри групп — по id).
        """
        if not self.enabled:
            return []
        try:
            with self._session() as conn:
                if profile_id is None:
                    rows = conn.execute(
                        "SELECT * FROM invariants WHERE scope = 'global' "
                        "ORDER BY id"
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM invariants WHERE scope = 'global' "
                        "OR profile_id = ? ORDER BY scope DESC, id",
                        (int(profile_id),),
                    ).fetchall()
            return [self._row_to_invariant(row) for row in rows]
        except sqlite3.Error as exc:
            self._log.error(
                "Ошибка чтения инвариантов профиля %s: %s", profile_id, exc
            )
            return []

    def get_invariants_by_category(
        self, profile_id: Optional[int], category_id: int
    ) -> List[Invariant]:
        """Инварианты категории, действующие для профиля (global + profile)."""
        return [
            invariant
            for invariant in self.get_invariants_for_profile(profile_id)
            if invariant.category_id == int(category_id)
        ]

    def get_all_invariants(self) -> List[Invariant]:
        """Все инварианты всех областей (для проверки конфликтов в UI)."""
        if not self.enabled:
            return []
        try:
            with self._session() as conn:
                rows = conn.execute(
                    "SELECT * FROM invariants ORDER BY id"
                ).fetchall()
            return [self._row_to_invariant(row) for row in rows]
        except sqlite3.Error as exc:
            self._log.error("Ошибка чтения инвариантов: %s", exc)
            return []

    def get_invariant_by_id(self, invariant_id: int) -> Optional[Invariant]:
        """Инвариант по идентификатору (None, если не найден)."""
        if not self.enabled:
            return None
        try:
            with self._session() as conn:
                row = conn.execute(
                    "SELECT * FROM invariants WHERE id = ?",
                    (int(invariant_id),),
                ).fetchone()
            return self._row_to_invariant(row) if row else None
        except sqlite3.Error as exc:
            self._log.error(
                "Ошибка чтения инварианта id=%s: %s", invariant_id, exc
            )
            return None

    def count_invariants(self, profile_id: Optional[int]) -> Dict[str, int]:
        """Счётчики для статус-бара: всего, глобальных, профильных.

        Args:
            profile_id: профиль; None — считаются только глобальные.

        Returns:
            ``{"total": N, "global": X, "profile": Y}`` (0 при отключённом
            репозитории/ошибке).
        """
        invariants = self.get_invariants_for_profile(profile_id)
        global_count = sum(
            1 for item in invariants if item.scope == SCOPE_GLOBAL
        )
        return {
            "total": len(invariants),
            "global": global_count,
            "profile": len(invariants) - global_count,
        }

    # ------------------------------------------------------------------
    # Инварианты: изменение
    # ------------------------------------------------------------------

    def create_invariant(self, invariant: Invariant) -> Invariant:
        """Создаёт инвариант.

        Args:
            invariant: инвариант; ``id`` игнорируется.

        Returns:
            Инвариант с заполненным ``id``.

        Raises:
            ValueError: если не заполнены title/rule, категория не найдена,
                scope некорректен или не согласован с profile_id.
        """
        title = str(invariant.title or "").strip()
        rule = str(invariant.rule or "").strip()
        if not title:
            raise ValueError("Название инварианта не может быть пустым.")
        if not rule:
            raise ValueError("Правило инварианта не может быть пустым.")
        scope = str(invariant.scope or SCOPE_PROFILE)
        if scope not in VALID_SCOPES:
            raise ValueError(f"Некорректная область действия: {scope!r}.")
        profile_id: Optional[int] = None
        if scope == SCOPE_GLOBAL:
            if invariant.profile_id is not None:
                raise ValueError(
                    "Глобальный инвариант не может быть привязан к профилю."
                )
        else:
            if invariant.profile_id is None:
                raise ValueError(
                    "Профильный инвариант требует указания профиля."
                )
            profile_id = int(invariant.profile_id)
        if not self.enabled:
            raise ValueError("Репозиторий инвариантов отключён.")
        with self._session() as conn:
            if not conn.execute(
                "SELECT 1 FROM invariant_categories WHERE id = ?",
                (int(invariant.category_id),),
            ).fetchone():
                raise ValueError(
                    f"Категория id={invariant.category_id} не найдена."
                )
            cursor = conn.execute(
                "INSERT INTO invariants(category_id, title, description, "
                "rule, scope, profile_id) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    int(invariant.category_id),
                    title,
                    str(invariant.description or ""),
                    rule,
                    scope,
                    profile_id,
                ),
            )
            invariant.id = int(cursor.lastrowid or 0)
        return invariant

    def update_invariant(self, invariant: Invariant) -> None:
        """Обновляет поля инварианта и ``updated_at``.

        Raises:
            ValueError: если нет id или поля некорректны (см.
                :meth:`create_invariant`).
        """
        if not invariant.id:
            raise ValueError("Нельзя обновить инвариант без id.")
        title = str(invariant.title or "").strip()
        rule = str(invariant.rule or "").strip()
        if not title:
            raise ValueError("Название инварианта не может быть пустым.")
        if not rule:
            raise ValueError("Правило инварианта не может быть пустым.")
        scope = str(invariant.scope or SCOPE_PROFILE)
        if scope not in VALID_SCOPES:
            raise ValueError(f"Некорректная область действия: {scope!r}.")
        profile_id: Optional[int] = None
        if scope == SCOPE_GLOBAL:
            if invariant.profile_id is not None:
                raise ValueError(
                    "Глобальный инвариант не может быть привязан к профилю."
                )
        else:
            if invariant.profile_id is None:
                raise ValueError(
                    "Профильный инвариант требует указания профиля."
                )
            profile_id = int(invariant.profile_id)
        with self._session() as conn:
            if not conn.execute(
                "SELECT 1 FROM invariant_categories WHERE id = ?",
                (int(invariant.category_id),),
            ).fetchone():
                raise ValueError(
                    f"Категория id={invariant.category_id} не найдена."
                )
            conn.execute(
                "UPDATE invariants SET category_id = ?, title = ?, "
                "description = ?, rule = ?, scope = ?, profile_id = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (
                    int(invariant.category_id),
                    title,
                    str(invariant.description or ""),
                    rule,
                    scope,
                    profile_id,
                    int(invariant.id),
                ),
            )

    def delete_invariant(self, invariant_id: int) -> bool:
        """Удаляет инвариант (True при успехе)."""
        if not self.enabled:
            return False
        try:
            with self._session() as conn:
                cursor = conn.execute(
                    "DELETE FROM invariants WHERE id = ?",
                    (int(invariant_id),),
                )
            return bool(cursor.rowcount)
        except sqlite3.Error as exc:
            self._log.error(
                "Ошибка удаления инварианта id=%s: %s", invariant_id, exc
            )
            return False

    # ------------------------------------------------------------------
    # Лог нарушений
    # ------------------------------------------------------------------

    def log_violation(
        self,
        invariant_id: int,
        profile_id: Optional[int],
        user_query: str,
        agent_response: str,
        regenerated: bool = False,
        resolved: bool = False,
    ) -> int:
        """Записывает нарушение/отказ в ``invariant_violations``.

        Args:
            invariant_id: затронутый инвариант.
            profile_id: активный профиль (может быть None).
            user_query: запрос пользователя.
            agent_response: финальный ответ агента.
            regenerated: True, если была перегенерация.
            resolved: True, если нарушение устранено.

        Returns:
            Идентификатор записи; 0 при ошибке/отключённом репозитории.
        """
        if not self.enabled:
            return 0
        try:
            with self._session() as conn:
                cursor = conn.execute(
                    "INSERT INTO invariant_violations(invariant_id, "
                    "profile_id, user_query, agent_response, regenerated, "
                    "resolved) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        int(invariant_id),
                        int(profile_id) if profile_id is not None else None,
                        str(user_query or ""),
                        str(agent_response or ""),
                        1 if regenerated else 0,
                        1 if resolved else 0,
                    ),
                )
                return int(cursor.lastrowid or 0)
        except sqlite3.Error as exc:
            self._log.error("Ошибка записи нарушения: %s", exc)
            return 0

    def mark_resolved(self, violation_id: int) -> bool:
        """Помечает нарушение устранённым (resolved = 1)."""
        if not self.enabled:
            return False
        try:
            with self._session() as conn:
                cursor = conn.execute(
                    "UPDATE invariant_violations SET resolved = 1 "
                    "WHERE id = ?",
                    (int(violation_id),),
                )
            return bool(cursor.rowcount)
        except sqlite3.Error as exc:
            self._log.error(
                "Ошибка обновления нарушения id=%s: %s", violation_id, exc
            )
            return False

    def get_violations(self, invariant_id: Optional[int] = None
                       ) -> List[InvariantViolation]:
        """Записи лога нарушений (все или по инварианту) — для аналитики."""
        if not self.enabled:
            return []
        try:
            with self._session() as conn:
                if invariant_id is None:
                    rows = conn.execute(
                        "SELECT * FROM invariant_violations "
                        "ORDER BY detected_at DESC, id DESC"
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM invariant_violations "
                        "WHERE invariant_id = ? "
                        "ORDER BY detected_at DESC, id DESC",
                        (int(invariant_id),),
                    ).fetchall()
            violations: List[InvariantViolation] = []
            for row in rows:
                profile_id = row["profile_id"]
                violations.append(InvariantViolation(
                    id=int(row["id"]),
                    invariant_id=int(row["invariant_id"]),
                    profile_id=(int(profile_id)
                                if profile_id is not None else None),
                    user_query=str(row["user_query"] or ""),
                    agent_response=str(row["agent_response"] or ""),
                    regenerated=bool(row["regenerated"]),
                    resolved=bool(row["resolved"]),
                ))
            return violations
        except sqlite3.Error as exc:
            self._log.error("Ошибка чтения лога нарушений: %s", exc)
            return []
