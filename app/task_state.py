"""Память задачи диалога (task state) — ТЗ ai_1_24, модули 1 и 2.

Память задачи — это «что происходит в текущем диалоге»: цель,
уточнения, ограничения и термины. Она ПРИВЯЗАНА К СЕССИИ диалога
(``session_id`` — UUID) и НЕ смешивается с другими видами памяти
приложения: FSM задачи (src/task_fsm.py), профилем пользователя,
долговременной памятью (src/memory.py) и рабочей памятью веток
(src/storage.py) — у каждого своё хранилище и свой жизненный цикл.

Хранение — только SQLite (таблица ``task_state`` в той же БД, что и
история чата, ``data/chat_history.db``); держать память в полях Python
нельзя — «Новый диалог» и перезапуск приложения обязаны видеть одно и
то же состояние. Дополнительно к таблице ``messages`` добавляется
колонка ``session_id`` (миграция на месте, идемпотентно): сообщения
активного диалога помечаются идентификатором сессии после каждого хода.

Как и всё хранилище приложения, репозиторий использует короткоживущие
соединения sqlite3 (безопасно из любых потоков) и не пробрасывает
ошибки БД наружу: сбой логируется, чат продолжается с состоянием в
памяти. Ограничение FOREIGN KEY из ТЗ объявлено в DDL, но PRAGMA
foreign_keys намеренно не включается — профиль может ещё не существовать
(свежая БД), а потеря хода из-за ссылочной целостности недопустима.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils import get_logger

logger = get_logger("deepseek-chat.task_state")

# JSON-поля памяти задачи (структура из ТЗ)
_STATE_GOAL = "goal"
_STATE_CLARIFICATIONS = "clarifications"
_STATE_CONSTRAINTS = "constraints"
_STATE_TERMS = "terms"


@dataclass
class TaskState:
    """Память задачи текущей сессии диалога (структура из ТЗ).

    Attributes:
        session_id: UUID сессии диалога.
        profile_id: профиль-владелец (0 — профиль не выбран).
        goal: цель диалога (например, «Настроить RAG с Ollama»).
        clarifications: уточнения контекста («документы на русском»).
        constraints: ограничения («без Docker», «только локальные»).
        terms: словарь терминов диалога ({"RAG": "Retrieval-…"}).
    """

    session_id: str
    profile_id: int
    goal: str = ""
    clarifications: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    terms: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Словарь для UI и результата хода (структура из ТЗ)."""
        return asdict(self)

    def is_empty(self) -> bool:
        """True, если ни одно из полей памяти не заполнено."""
        return not (self.goal or self.clarifications
                    or self.constraints or self.terms)

    def to_prompt_block(self) -> str:
        """Форматирует память задачи для системного промпта (формат ТЗ)."""
        parts = []
        if self.goal:
            parts.append(f"🎯 Цель диалога: {self.goal}")
        if self.clarifications:
            parts.append("📝 Уточнения:\n"
                         + "\n".join(f"  • {c}" for c in self.clarifications))
        if self.constraints:
            parts.append("⚠️ Ограничения:\n"
                         + "\n".join(f"  • {c}" for c in self.constraints))
        if self.terms:
            parts.append("📚 Термины:\n"
                         + "\n".join(f"  • {k}: {v}"
                                     for k, v in self.terms.items()))
        return "\n\n".join(parts) if parts else "Состояние диалога пусто."

    def apply_update(self, update: Dict[str, Any]) -> bool:
        """Мерджит обновление от LLM в память; True — что-то изменилось.

        Списки пополняются БЕЗ дубликатов (сравнение без учёта регистра
        — модель может прислать то же уточнение в другом регистре),
        terms обновляются по ключу, цель заменяется только непустым
        новым значением.

        Args:
            update: словарь вида task_state из JSON-ответа LLM.

        Returns:
            True, если память изменилась (есть что сохранять в БД).
        """
        update = update if isinstance(update, dict) else {}
        changed = False

        goal = str(update.get(_STATE_GOAL) or "").strip()
        if goal and goal != self.goal:
            self.goal = goal
            changed = True

        for key, target in ((_STATE_CLARIFICATIONS, self.clarifications),
                            (_STATE_CONSTRAINTS, self.constraints)):
            existing = {item.strip().lower() for item in target}
            for item in update.get(key) or []:
                text = str(item or "").strip()
                if text and text.lower() not in existing:
                    target.append(text)
                    existing.add(text.lower())
                    changed = True

        terms = update.get(_STATE_TERMS)
        if isinstance(terms, dict):
            for term, definition in terms.items():
                name = str(term or "").strip()
                value = str(definition or "").strip()
                if name and value and self.terms.get(name) != value:
                    self.terms[name] = value
                    changed = True
        return changed


def normalize_task_update(update: Any) -> Dict[str, Any]:
    """Приводит task_state из JSON-ответа модели к безопасным типам."""
    update = update if isinstance(update, dict) else {}
    clarifications = [str(item or "").strip()
                      for item in update.get(_STATE_CLARIFICATIONS) or []]
    constraints = [str(item or "").strip()
                   for item in update.get(_STATE_CONSTRAINTS) or []]
    terms_raw = update.get(_STATE_TERMS)
    terms = ({str(k): str(v) for k, v in terms_raw.items()}
             if isinstance(terms_raw, dict) else {})
    return {
        _STATE_GOAL: str(update.get(_STATE_GOAL) or "").strip(),
        _STATE_CLARIFICATIONS: [c for c in clarifications if c],
        _STATE_CONSTRAINTS: [c for c in constraints if c],
        _STATE_TERMS: terms,
    }


class TaskStateRepository:
    """Репозиторий памяти задачи в SQLite (таблица ``task_state``).

    Attributes:
        db_path: путь к файлу БД (та же БД, что у истории чата).
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.ensure_schema()

    # ------------------------------------------------------------------
    # Схема (Модуль 1 ТЗ)
    # ------------------------------------------------------------------
    def ensure_schema(self) -> None:
        """Создаёт таблицу task_state и мигрирует messages.session_id.

        Идемпотентно: каждая DDL-операция в try/except — сбой миграции
        не ломает приложение (память просто не сохранится, причина в
        логе), а get_or_create продолжит работать в рамках хода.
        """
        self._prepare_disk()
        try:
            with self._session() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS task_state (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL UNIQUE,
                        profile_id INTEGER NOT NULL,
                        goal TEXT,
                        clarifications TEXT,
                        constraints TEXT,
                        terms TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (profile_id) REFERENCES profiles(id)
                            ON DELETE CASCADE
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_task_state_session "
                    "ON task_state(session_id)"
                )
                self._ensure_messages_session_column(conn)
        except sqlite3.Error as exc:
            logger.error("Не удалось подготовить схему task_state: %s", exc)

    def _ensure_messages_session_column(self, conn: sqlite3.Connection
                                        ) -> None:
        """Колонка messages.session_id + индекс (миграция из ТЗ).

        Вызывается и из ensure_schema, и лениво из tag_session_messages:
        таблица messages создаётся ядром позже репозитория (файл БД
        появляется при первой записи истории).
        """
        tables = {
            str(row[0]) for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        if "messages" not in tables:
            return
        columns = {str(row[1]) for row in
                   conn.execute("PRAGMA table_info(messages)")}
        if "session_id" not in columns:
            try:
                conn.execute("ALTER TABLE messages ADD COLUMN session_id TEXT")
            except sqlite3.Error as exc:
                logger.warning(
                    "Не удалось добавить messages.session_id: %s", exc)
                return
        try:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session "
                "ON messages(session_id, created_at)"
            )
        except sqlite3.Error as exc:
            logger.warning("Не удалось создать idx_messages_session: %s", exc)

    # ------------------------------------------------------------------
    # Публичное API (Модуль 2 ТЗ)
    # ------------------------------------------------------------------
    def get_or_create(self, session_id: str, profile_id: int) -> TaskState:
        """Загружает память задачи сессии или создаёт новую.

        Ошибка БД не ломает ход: возвращается состояние в памяти
        (причина — в логе).

        Args:
            session_id: UUID сессии диалога.
            profile_id: профиль-владелец (для новой записи).

        Returns:
            TaskState (из БД либо пустой).
        """
        state = TaskState(session_id=str(session_id), profile_id=int(profile_id))
        try:
            self._prepare_disk()
            with self._session() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO task_state(session_id, profile_id) "
                    "VALUES (?, ?)",
                    (state.session_id, state.profile_id),
                )
                row = conn.execute(
                    "SELECT goal, clarifications, constraints, terms "
                    "FROM task_state WHERE session_id = ?",
                    (state.session_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            logger.error("Ошибка чтения памяти задачи: %s", exc)
            return state
        if row is not None:
            state.goal = row[0] or ""
            state.clarifications = _loads_list(row[1])
            state.constraints = _loads_list(row[2])
            state.terms = _loads_dict(row[3])
        return state

    def update(self, state: TaskState) -> bool:
        """Сохраняет память задачи (goal/clarifications/constraints/terms).

        Args:
            state: обновлённое состояние (ключ — session_id).

        Returns:
            True при успехе, False при ошибке БД.
        """
        try:
            self._prepare_disk()
            with self._session() as conn:
                cursor = conn.execute(
                    """
                    UPDATE task_state
                    SET goal = ?, clarifications = ?, constraints = ?,
                        terms = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                    """,
                    (
                        state.goal,
                        json.dumps(state.clarifications, ensure_ascii=False),
                        json.dumps(state.constraints, ensure_ascii=False),
                        json.dumps(state.terms, ensure_ascii=False),
                        state.session_id,
                    ),
                )
                if cursor.rowcount == 0:
                    conn.execute(
                        "INSERT INTO task_state(session_id, profile_id, goal,"
                        " clarifications, constraints, terms)"
                        " VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            state.session_id, int(state.profile_id), state.goal,
                            json.dumps(state.clarifications,
                                       ensure_ascii=False),
                            json.dumps(state.constraints,
                                       ensure_ascii=False),
                            json.dumps(state.terms, ensure_ascii=False),
                        ),
                    )
            return True
        except sqlite3.Error as exc:
            logger.error("Ошибка сохранения памяти задачи: %s", exc)
            return False

    def reset(self, session_id: str) -> bool:
        """Сбрасывает память задачи сессии (кнопки «Новый диалог» и
        «Сбросить состояние»).

        Args:
            session_id: UUID сессии, чью память удалить.

        Returns:
            True при успехе, False при ошибке БД.
        """
        try:
            self._prepare_disk()
            with self._session() as conn:
                conn.execute(
                    "DELETE FROM task_state WHERE session_id = ?",
                    (str(session_id),),
                )
            return True
        except sqlite3.Error as exc:
            logger.error("Ошибка сброса памяти задачи: %s", exc)
            return False

    def tag_session_messages(self, session_id: str) -> bool:
        """Помечает сообщения активного диалога session_id сессии.

        Ядро перезаписывает таблицу messages после каждого хода (без
        колонки session_id), поэтому после хода все «свежие» строки с
        NULL помечаются текущей сессией. Миграция колонки выполняется
        лениво — таблица messages может появиться позже репозитория.

        Args:
            session_id: UUID текущей сессии.

        Returns:
            True при успехе, False при ошибке/отсутствии таблицы.
        """
        try:
            self._prepare_disk()
            with self._session() as conn:
                self._ensure_messages_session_column(conn)
                columns = {str(row[1]) for row in
                           conn.execute("PRAGMA table_info(messages)")}
                if "session_id" not in columns:
                    return False
                conn.execute(
                    "UPDATE messages SET session_id = ? "
                    "WHERE session_id IS NULL",
                    (str(session_id),),
                )
            return True
        except sqlite3.Error as exc:
            logger.warning("Не удалось пометить сообщения сессией: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Служебное
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        """Короткоживущее соединение (без PRAGMA foreign_keys — см. модуль)."""
        return sqlite3.connect(str(self.db_path), timeout=10.0)

    def _session(self):
        """Контекст соединения с транзакцией (как в src/storage.py)."""
        return _session_scope(self._connect())

    def _prepare_disk(self) -> None:
        """Создаёт каталог БД, если его нет (ленивое создание файла)."""
        parent = self.db_path.parent
        if str(parent) and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)


class _session_scope:
    """Контекст: транзакция + гарантированное закрытие соединения."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def __enter__(self) -> sqlite3.Connection:
        return self._conn

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()
        finally:
            self._conn.close()


def _loads_list(raw: Optional[str]) -> List[str]:
    """JSON-массив строк из БД (устойчиво к мусору)."""
    try:
        parsed = json.loads(raw) if raw else []
    except (ValueError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item or "").strip()]


def _loads_dict(raw: Optional[str]) -> Dict[str, str]:
    """JSON-объект терминов из БД (устойчиво к мусору)."""
    try:
        parsed = json.loads(raw) if raw else {}
    except (ValueError, TypeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): str(v) for k, v in parsed.items()}
