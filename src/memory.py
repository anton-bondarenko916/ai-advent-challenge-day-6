"""Явная модель памяти агента из трёх слоёв.

Модуль реализует три слоя памяти, каждый из которых хранится отдельно и
живёт по своим правилам:

* **short_term** (краткосрочная) — сообщения активной ветки плюс механизм
  закрепления (``pin``): закреплённое сообщение всегда уходит в запрос,
  даже если вышло за окно ``WINDOW_KEEP_LAST_N``;
* **working** (рабочая) — структурированное состояние текущей задачи
  (``goal``/``status``/``plan``/``constraints``/``open_questions``) и элементы
  ``key → value`` с видом (``decision``/``artifact``/``note``). Привязана к
  ветке: у каждой ветки своя рабочая память;
* **long_term** (долговременная) — глобальный профиль/решения/знания, живёт
  в отдельном файле SQLite (``data/long_memory.db``) и переживает очистку
  истории, удаление ветки и удаление БД чата.

Запись в слои происходит **только явно** и **только через единый роутер**
:class:`MemoryManager.route`. Модель лишь предлагает кандидатов
(:class:`MemoryClassifier`), которые кладутся в ``pending`` и не попадают в
слои до явного принятия/отклонения пользователем.

Модуль не импортирует tkinter и не выполняет сетевых вызовов: служебный
запрос классификации выполняет переданный снаружи callable.

Правило-мнемоника: *«диалог забывается окном, задача — её закрытием,
долговременная память — только явным "Забыть"»*.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from . import utils
from .facts import extract_json_object

logger = utils.get_logger("deepseek-chat.memory")

#: Тип сообщения чата
Message = Dict[str, str]

# ---------------------------------------------------------------------------
# Константы слоёв и разделов
# ---------------------------------------------------------------------------

SHORT_TERM = "short_term"
WORKING = "working"
LONG_TERM = "long_term"

#: Канонический порядок слоёв (для блоков запроса и отчётов)
LAYERS: Tuple[str, ...] = (SHORT_TERM, WORKING, LONG_TERM)

#: Русские подписи слоёв для UI и блоков запроса
LAYER_TITLES: Dict[str, str] = {
    SHORT_TERM: "Краткосрочная (диалог)",
    WORKING: "Рабочая (задача)",
    LONG_TERM: "Долговременная (профиль, решения, знания)",
}

#: Разделы долговременной памяти
SECTIONS: Tuple[str, ...] = ("profile", "decisions", "knowledge")

#: Заголовки блоков долговременной/рабочей памяти в запросе
LONG_HEADER = "ДОЛГОВРЕМЕННАЯ ПАМЯТЬ (профиль, решения, знания):"
WORKING_HEADER = "РАБОЧАЯ ПАМЯТЬ (текущая задача):"
PINNED_HEADER = "ЗАКРЕПЛЁННЫЕ СООБЩЕНИЯ ДИАЛОГА:"

#: Виды элементов рабочей памяти
WORKING_KINDS: Tuple[str, ...] = ("decision", "artifact", "note")

#: Вид → категория FactsStore (для стратегии sticky_facts)
_KIND_TO_FACTS_CATEGORY: Dict[str, str] = {
    "decision": "decisions",
    "artifact": "agreements",
}

#: Маркер служебного запроса классификации памяти
MEMORY_MARKER = "[[КЛАССИФИКАЦИЯ_ПАМЯТИ]]"

#: Значения по умолчанию
DEFAULT_MEMORY_MAX_TOKENS = 600
DEFAULT_LONG_MAX_TOKENS = 300
DEFAULT_WORKING_MAX_TOKENS = 250
DEFAULT_PIN_MAX = 5
DEFAULT_LONG_MAX_ITEMS = 200
DEFAULT_MIN_CONFIDENCE = 0.6
DEFAULT_MEMORY_REQUEST_MAX_TOKENS = 600

#: Промпт-инструкция классификации по умолчанию
DEFAULT_MEMORY_PROMPT = (
    "Ты — модуль классификации памяти диалога. По приведённому фрагменту "
    "диалога предложи, что стоит сохранить в память, и в какой слой.\n"
    "Требования:\n"
    "— ответь СТРОГО одним JSON-объектом, без пояснений и без markdown;\n"
    "— структура ответа:\n"
    '{"short_term": [{"text": "...", "confidence": 0.0..1.0}],\n'
    ' "working": [{"key": "...", "value": "...", '
    '"kind": "decision|artifact|note", "confidence": ...}],\n'
    ' "long_term": [{"section": "profile|decisions|knowledge", "key": "...", '
    '"value": "...", "confidence": ...}]}\n'
    "— ключи и значения — короткие строки на языке диалога;\n"
    "— в short_term — данные, важные только для текущего диалога;\n"
    "— в working — цель, статус, план, ограничения, решения и артефакты "
    "текущей задачи;\n"
    "— в long_term — устойчивый профиль пользователя, принятые решения, знания;\n"
    "— не выдумывай факты, которых нет в тексте;\n"
    "— если сохранять нечего, верни пустые списки.\n"
    f"Служебная метка запроса: {MEMORY_MARKER}"
)


class MemoryError(ValueError):
    """Ошибка модели памяти: неизвестный слой/раздел/некорректные данные."""


def _now() -> str:
    """Текущее локальное время в формате ISO (секунды)."""
    return datetime.now().isoformat(timespec="seconds")


def _clean(value: Any) -> str:
    """Строка без краевых пробелов (пустая для не-строк)."""
    if not isinstance(value, str):
        return ""
    return value.strip()


def _as_list(value: Any) -> List[str]:
    """Список строк из произвольного значения (устойчиво к мусору)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value if _clean(item)]
    return []


# ---------------------------------------------------------------------------
# Нормализация слоя
# ---------------------------------------------------------------------------

#: Русские подписи кнопок и их вариации → каноническое имя слоя
_LAYER_ALIASES: Dict[str, str] = {
    "short_term": SHORT_TERM,
    "short": SHORT_TERM,
    "short-term": SHORT_TERM,
    "краткосрочная": SHORT_TERM,
    "краткосрочная (диалог)": SHORT_TERM,
    "в диалог": SHORT_TERM,
    "диалог": SHORT_TERM,
    "закрепить": SHORT_TERM,
    "working": WORKING,
    "рабочая": WORKING,
    "рабочая (задача)": WORKING,
    "в рабочую": WORKING,
    "задача": WORKING,
    "long_term": LONG_TERM,
    "long": LONG_TERM,
    "long-term": LONG_TERM,
    "долговременная": LONG_TERM,
    "долговременная (профиль, решения, знания)": LONG_TERM,
    "в долговременную": LONG_TERM,
    "профиль": LONG_TERM,
    "навсегда": LONG_TERM,
}


def normalize_layer(name: Any) -> str:
    """Приводит имя слоя к каноническому виду.

    Понимает русские подписи кнопок («В диалог», «В рабочую»,
    «В долговременную»). Неизвестный слой — ошибка, а не тихий fallback.

    Raises:
        MemoryError: если имя слоя неизвестно.
    """
    raw = str(name or "").strip().lower()
    if raw in LAYERS:
        return raw
    if raw in _LAYER_ALIASES:
        return _LAYER_ALIASES[raw]
    raise MemoryError(
        f"Неизвестный слой памяти: {name!r}. Допустимые: {', '.join(LAYERS)}"
    )


def normalize_section(name: Any) -> str:
    """Приводит имя раздела долговременной памяти к каноническому виду.

    Raises:
        MemoryError: если раздел неизвестен.
    """
    raw = str(name or "").strip().lower()
    aliases = {
        "profile": "profile", "профиль": "profile", "о пользователе": "profile",
        "decisions": "decisions", "решения": "decisions",
        "договорённости": "decisions",
        "knowledge": "knowledge", "знания": "knowledge", "правила": "knowledge",
        "ссылки": "knowledge",
    }
    if raw in SECTIONS:
        return raw
    if raw in aliases:
        return aliases[raw]
    raise MemoryError(
        f"Неизвестный раздел долговременной памяти: {name!r}. "
        f"Допустимые: {', '.join(SECTIONS)}"
    )


def _count_text(counter: Any, text: str) -> int:
    """Оценка токенов текста (0 при отсутствии счётчика или ошибке)."""
    if counter is None or not text:
        return 0
    try:
        return int(counter.count_text(text))
    except Exception as exc:
        logger.warning("Ошибка счётчика токенов в memory: %s", exc)
        return 0


def _trim_lines(text: str, counter: Any, max_tokens: int) -> str:
    """Урезает текст до бюджета, отбрасывая строки с конца (заголовок остаётся).

    Args:
        text: исходный многострочный текст.
        counter: счётчик токенов.
        max_tokens: бюджет в токенах.

    Returns:
        Урезанный текст (не длиннее бюджета, если это возможно).
    """
    if max_tokens <= 0 or not text:
        return "" if max_tokens <= 0 else text
    if counter is None:
        return text
    try:
        if counter.count_text(text) <= max_tokens:
            return text
    except Exception:
        return text
    lines = text.split("\n")
    while len(lines) > 1:
        lines = lines[:-1]
        candidate = "\n".join(lines)
        try:
            if counter.count_text(candidate) <= max_tokens:
                return candidate
        except Exception:
            return candidate
    return lines[0] if lines else ""


# ---------------------------------------------------------------------------
# Рабочая память (задача, привязана к ветке)
# ---------------------------------------------------------------------------

class WorkingMemory:
    """Структурированное состояние текущей задачи (рабочая память).

    Attributes:
        task: поля задачи ``{goal, status, plan, constraints, open_questions}``.
        items: элементы ``{key: {value, kind, source, source_message,
            updated_at}}``. Вид — ``decision``/``artifact``/``note``.
        counter: счётчик токенов (может быть задан позже).
    """

    def __init__(self, counter: Any = None) -> None:
        self.counter = counter
        self.task: Dict[str, Any] = {
            "goal": "",
            "status": "",
            "plan": [],
            "constraints": [],
            "open_questions": [],
        }
        self.items: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Доступ к полям задачи
    # ------------------------------------------------------------------

    def set_field(self, name: str, value: Any) -> None:
        """Устанавливает поле задачи (``goal``/``status``/``plan``/``constraints``/
        ``open_questions``). Для ``plan``/``constraints``/``open_questions``
        значение — список строк (или строка с переносами строк)."""
        name = str(name or "").strip().lower()
        if name in ("plan", "constraints", "open_questions"):
            if isinstance(value, str):
                value = [line.strip() for line in value.split("\n") if line.strip()]
            self.task[name] = _as_list(value)
        elif name in ("goal", "status"):
            self.task[name] = _clean(value)
        else:
            raise MemoryError(f"Неизвестное поле рабочей памяти: {name!r}")

    def drop_field(self, name: str) -> None:
        """Сбрасывает поле задачи в пустое значение."""
        name = str(name or "").strip().lower()
        if name in ("plan", "constraints", "open_questions"):
            self.task[name] = []
        elif name in ("goal", "status"):
            self.task[name] = ""
        else:
            raise MemoryError(f"Неизвестное поле рабочей памяти: {name!r}")

    def remember(self, key: str, value: str, kind: str = "note",
                 source: str = "manual", message_index: Optional[int] = None) -> None:
        """Добавляет/обновляет элемент ``key → value`` с видом ``kind``."""
        key = " ".join(str(key or "").split()).strip()
        value = _clean(value)
        if not key or not value:
            raise MemoryError("Ключ и значение рабочей памяти не могут быть пустыми.")
        if kind not in WORKING_KINDS:
            raise MemoryError(
                f"Неизвестный вид элемента рабочей памяти: {kind!r}. "
                f"Допустимые: {', '.join(WORKING_KINDS)}"
            )
        self.items[key] = {
            "value": value,
            "kind": kind,
            "source": str(source or "manual"),
            "source_message": message_index,
            "updated_at": _now(),
        }

    def forget(self, key: str) -> bool:
        """Удаляет элемент по ключу (True, если он был)."""
        key = str(key or "").strip()
        if key in self.items:
            del self.items[key]
            return True
        return False

    def clear(self) -> None:
        """Полностью очищает рабочую память (и поля, и элементы)."""
        self.task = {
            "goal": "", "status": "", "plan": [],
            "constraints": [], "open_questions": [],
        }
        self.items.clear()

    # ------------------------------------------------------------------
    # Сериализация
    # ------------------------------------------------------------------

    def as_dict(self) -> Dict[str, Any]:
        """Сериализуемое представление рабочей памяти (копия)."""
        return {
            "task": {
                "goal": self.task.get("goal", ""),
                "status": self.task.get("status", ""),
                "plan": list(self.task.get("plan", [])),
                "constraints": list(self.task.get("constraints", [])),
                "open_questions": list(self.task.get("open_questions", [])),
            },
            "items": {key: dict(item) for key, item in self.items.items()},
        }

    def from_dict(self, raw: Any) -> "WorkingMemory":
        """Загружает рабочую память из словаря (устойчиво к мусору).

        Битый JSON/не-словарь → пустая память + WARNING.
        """
        if not isinstance(raw, dict):
            logger.warning("Рабочая память не является словарём — очищаю.")
            self.clear()
            return self
        task = raw.get("task") if isinstance(raw.get("task"), dict) else {}
        self.task = {
            "goal": _clean(task.get("goal")),
            "status": _clean(task.get("status")),
            "plan": _as_list(task.get("plan")),
            "constraints": _as_list(task.get("constraints")),
            "open_questions": _as_list(task.get("open_questions")),
        }
        self.items = {}
        items = raw.get("items")
        if isinstance(items, dict):
            for key, item in items.items():
                k = str(key).strip()
                if not k or not isinstance(item, dict):
                    continue
                value = _clean(item.get("value"))
                if not value:
                    continue
                kind = str(item.get("kind") or "note").strip().lower()
                if kind not in WORKING_KINDS:
                    kind = "note"
                self.items[k] = {
                    "value": value,
                    "kind": kind,
                    "source": str(item.get("source") or "manual"),
                    "source_message": item.get("source_message"),
                    "updated_at": str(item.get("updated_at") or _now()),
                }
        return self

    def is_empty(self) -> bool:
        """True, если в рабочей памяти нет ни полей, ни элементов."""
        task = self.task
        has_task = bool(
            task.get("goal") or task.get("status") or task.get("plan")
            or task.get("constraints") or task.get("open_questions")
        )
        return not has_task and not self.items

    def count(self) -> int:
        """Число элементов (поля задачи не считаются)."""
        return len(self.items)

    # ------------------------------------------------------------------
    # Вывод в запрос
    # ------------------------------------------------------------------

    def _section_blocks(self, skip_keys: Optional[set] = None) -> List[Tuple[str, List[str]]]:
        """Секции текста блока в порядке важности (для приоритетного усечения)."""
        skip = skip_keys or set()
        task = self.task
        blocks: List[Tuple[str, List[str]]] = []
        goal = _clean(task.get("goal"))
        if goal:
            blocks.append(("goal", ["Цель: " + goal]))
        status = _clean(task.get("status"))
        if status:
            blocks.append(("status", ["Статус: " + status]))
        constraints = _as_list(task.get("constraints"))
        if constraints:
            blocks.append(("constraints", ["Ограничения:"] + ["- " + c for c in constraints]))
        item_lines: List[str] = []
        for key in sorted(self.items):
            item = self.items[key]
            if key in skip:
                continue
            kind = item.get("kind", "note")
            item_lines.append(f"- [{kind}] {key}: {item.get('value', '')}")
        if item_lines:
            blocks.append(("items", ["Элементы:"] + item_lines))
        open_q = _as_list(task.get("open_questions"))
        if open_q:
            blocks.append(("open_questions", ["Открытые вопросы:"] + ["- " + q for q in open_q]))
        plan = _as_list(task.get("plan"))
        if plan:
            blocks.append(("plan", ["План:"] + ["- " + p for p in plan]))
        return blocks

    def to_text(self, counter: Any = None, max_tokens: Optional[int] = None,
                skip_keys: Optional[set] = None) -> str:
        """Текст блока рабочей памяти с приоритетным усечением.

        Порядок вытеснения при превышении бюджета: ``plan`` → ``open_questions``
        → ``items`` → ``constraints``. ``goal``/``status`` никогда не вытесняются.

        Args:
            counter: счётчик токенов (по умолчанию — self.counter).
            max_tokens: бюджет; None — без усечения.
            skip_keys: ключи элементов, которые не включать (уже ушли блоком
                фактов sticky_facts — чтобы не дублировать).
        """
        engine = counter if counter is not None else self.counter
        blocks = self._section_blocks(skip_keys)
        lines: List[str] = [WORKING_HEADER]
        for _, block_lines in blocks:
            lines.extend(block_lines)
        text = "\n".join(lines)

        if max_tokens is None or engine is None:
            return text
        try:
            if engine.count_text(text) <= max_tokens:
                return text
        except Exception:
            return text

        # Приоритетное вытеснение (кумулятивное): plan → open_questions →
        # items → constraints. goal/status никогда не вытесняются.
        kept = list(blocks)
        for name in ("plan", "open_questions", "items", "constraints"):
            kept = [b for b in kept if b[0] != name]
            candidate = WORKING_HEADER
            if kept:
                candidate += "\n" + "\n".join(ln for _, bl in kept for ln in bl)
            try:
                if engine.count_text(candidate) <= max_tokens:
                    return candidate
            except Exception:
                return candidate
        # Бюджет исчерпан, но goal/status остаются всегда
        base = [WORKING_HEADER]
        for name, block_lines in blocks:
            if name in ("goal", "status"):
                base.extend(block_lines)
        return "\n".join(base)

    def token_count(self, counter: Any = None,
                    skip_keys: Optional[set] = None) -> int:
        """Оценка токенов блока рабочей памяти."""
        engine = counter if counter is not None else self.counter
        return _count_text(engine, self.to_text(engine, None, skip_keys))


# ---------------------------------------------------------------------------
# Долговременная память (отдельный файл SQLite)
# ---------------------------------------------------------------------------

class LongTermStore:
    """Долговременная память в отдельном файле SQLite.

    Файл создаётся **лениво** (при первой записи) и **не создаётся в
    mock-режиме** (если ``enabled=False``). Короткоживущее соединение на
    операцию — тот же паттерн, что у ``ContextStore._session()``.

    Схема (собственная ``SCHEMA_VERSION = "1"`` в своей таблице ``meta``):

    * ``long_memory`` — записи ``UNIQUE(section, key)``; новое значение
      перезаписывает старое (``updated_at``, ``source``), но не меняет раздел;
    * ``pending_memory`` — предложения классификации от LLM.
    """

    SCHEMA_VERSION = "1"

    def __init__(self, db_path: str | Path, enabled: bool = True,
                 max_items: int = DEFAULT_LONG_MAX_ITEMS) -> None:
        self.db_path = Path(db_path)
        self.enabled = bool(enabled)
        self.max_items = max(1, int(max_items))
        self._log = utils.get_logger("deepseek-chat.memory.long")
        self._closed = False

    # ------------------------------------------------------------------
    # Соединение и схема
    # ------------------------------------------------------------------

    @property
    def exists(self) -> bool:
        """True, если файл БД уже создан на диске."""
        return self.db_path.exists()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _prepare_disk(self) -> None:
        parent = self.db_path.parent
        if str(parent) and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS meta ("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS long_memory (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                section        TEXT NOT NULL,
                key            TEXT NOT NULL,
                value          TEXT NOT NULL,
                source         TEXT NOT NULL DEFAULT 'manual',
                source_message INTEGER,
                created_at     TEXT NOT NULL,
                updated_at     TEXT NOT NULL,
                used_at        TEXT,
                UNIQUE(section, key)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_memory (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                layer          TEXT NOT NULL,
                section        TEXT,
                key            TEXT,
                value          TEXT NOT NULL,
                kind           TEXT,
                confidence     REAL,
                source_message INTEGER,
                created_at     TEXT NOT NULL
            )
            """
        )
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        if not row:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
                (self.SCHEMA_VERSION,),
            )

    def _read_meta(self, conn: sqlite3.Connection, key: str) -> Optional[str]:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return str(row[0]) if row else None

    def _write_meta(self, conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    # ------------------------------------------------------------------
    # Чтение
    # ------------------------------------------------------------------

    def load(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Загружает все записи: ``{section: {key: {value, source, ...}}}``."""
        if not self.enabled or self._closed or not self.exists:
            return {}
        try:
            with self._session() as conn:
                self._ensure_schema(conn)
                rows = conn.execute(
                    "SELECT section, key, value, source, source_message, "
                    "created_at, updated_at, used_at FROM long_memory "
                    "ORDER BY section, key"
                ).fetchall()
            result: Dict[str, Dict[str, Dict[str, Any]]] = {}
            for section, key, value, source, sm, created, updated, used in rows:
                result.setdefault(str(section), {})[str(key)] = {
                    "value": str(value),
                    "source": str(source or "manual"),
                    "source_message": sm,
                    "created_at": created,
                    "updated_at": updated,
                    "used_at": used,
                }
            return result
        except sqlite3.Error as exc:
            self._log.error("Ошибка загрузки долговременной памяти: %s", exc)
            return {}

    def list_items(self, section: Optional[str] = None) -> List[Dict[str, Any]]:
        """Список записей (``section`` None — все разделы)."""
        data = self.load()
        rows: List[Dict[str, Any]] = []
        for sec in sorted(data):
            if section is not None and sec != str(section):
                continue
            for key in sorted(data[sec]):
                item = data[sec][key]
                rows.append({
                    "section": sec, "key": key,
                    "value": item.get("value", ""),
                    "source": item.get("source", "manual"),
                    "source_message": item.get("source_message"),
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                    "used_at": item.get("used_at"),
                })
        return rows

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Поиск по ключам и значениям (без учёта регистра)."""
        needle = str(query or "").strip().lower()
        if not needle:
            return self.list_items()
        return [
            item for item in self.list_items()
            if needle in str(item.get("key", "")).lower()
            or needle in str(item.get("value", "")).lower()
        ]

    def as_dict(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Сериализуемое представление (копия)."""
        return {
            section: {key: dict(item) for key, item in items.items()}
            for section, items in self.load().items()
        }

    # ------------------------------------------------------------------
    # Запись
    # ------------------------------------------------------------------

    def add(self, section: str, key: str, value: str,
            source: str = "manual", source_message: Optional[int] = None) -> bool:
        """Добавляет/обновляет запись (upsert по ``UNIQUE(section, key)``).

        Новое значение перезаписывает старое (``updated_at``, ``source``),
        но **не** меняет раздел.
        """
        section = normalize_section(section)
        key = " ".join(str(key or "").split()).strip()
        value = _clean(value)
        if not key or not value:
            self._log.warning("Пустой ключ/значение долговременной памяти — не сохраняю.")
            return False
        if not self.enabled or self._closed:
            self._log.warning("Долговременная память отключена — запись игнорирована.")
            return False
        try:
            self._prepare_disk()
            with self._session() as conn:
                self._ensure_schema(conn)
                now = _now()
                conn.execute(
                    "INSERT INTO long_memory(section, key, value, source, "
                    "source_message, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(section, key) DO UPDATE SET "
                    "value = excluded.value, source = excluded.source, "
                    "source_message = excluded.source_message, "
                    "updated_at = excluded.updated_at",
                    (section, key, value, str(source or "manual"),
                     source_message, now, now),
                )
            self._enforce_max_items()
            return True
        except (sqlite3.Error, OSError, MemoryError) as exc:
            self._log.error("Ошибка записи в долговременную память: %s", exc)
            return False

    def update(self, section: str, key: str, value: str,
               source: str = "manual") -> bool:
        """Синоним :meth:`add` (перезапись значения в том же разделе)."""
        return self.add(section, key, value, source=source)

    def forget(self, section: str, key: str) -> bool:
        """Удаляет запись по разделу и ключу (True, если была)."""
        section = normalize_section(section)
        if not self.enabled or self._closed or not self.exists:
            return False
        try:
            with self._session() as conn:
                self._ensure_schema(conn)
                cursor = conn.execute(
                    "DELETE FROM long_memory WHERE section = ? AND key = ?",
                    (section, str(key).strip()),
                )
            return cursor.rowcount > 0
        except (sqlite3.Error, MemoryError) as exc:
            self._log.error("Ошибка удаления из долговременной памяти: %s", exc)
            return False

    def clear_section(self, section: str) -> bool:
        """Очищает раздел целиком."""
        section = normalize_section(section)
        if not self.enabled or self._closed or not self.exists:
            return False
        try:
            with self._session() as conn:
                self._ensure_schema(conn)
                conn.execute(
                    "DELETE FROM long_memory WHERE section = ?", (section,)
                )
            return True
        except (sqlite3.Error, MemoryError) as exc:
            self._log.error("Ошибка очистки раздела: %s", exc)
            return False

    def clear_all(self) -> bool:
        """Полностью очищает долговременную память."""
        if not self.enabled or self._closed or not self.exists:
            return False
        try:
            with self._session() as conn:
                self._ensure_schema(conn)
                conn.execute("DELETE FROM long_memory")
            return True
        except sqlite3.Error as exc:
            self._log.error("Ошибка очистки долговременной памяти: %s", exc)
            return False

    def mark_used(self, section: str, key: str) -> None:
        """Отмечает запись использованной (``used_at``)."""
        section = normalize_section(section)
        if not self.enabled or self._closed or not self.exists:
            return
        try:
            with self._session() as conn:
                self._ensure_schema(conn)
                conn.execute(
                    "UPDATE long_memory SET used_at = ? "
                    "WHERE section = ? AND key = ?",
                    (_now(), section, str(key).strip()),
                )
        except (sqlite3.Error, MemoryError) as exc:
            self._log.warning("Не удалось отметить использование записи: %s", exc)

    def _enforce_max_items(self) -> None:
        """Вытесняет самые старые записи при превышении лимита."""
        if not self.enabled or self._closed or not self.exists:
            return
        try:
            with self._session() as conn:
                self._ensure_schema(conn)
                count = conn.execute(
                    "SELECT COUNT(*) FROM long_memory"
                ).fetchone()
                total = int(count[0]) if count else 0
                excess = total - self.max_items
                if excess <= 0:
                    return
                rows = conn.execute(
                    "SELECT id FROM long_memory "
                    "ORDER BY COALESCE(used_at, updated_at), id LIMIT ?",
                    (excess,),
                ).fetchall()
                for (row_id,) in rows:
                    conn.execute(
                        "DELETE FROM long_memory WHERE id = ?", (int(row_id),)
                    )
                self._log.warning(
                    "Долговременная память превысила лимит (%d): вытеснено %d.",
                    self.max_items, excess,
                )
        except sqlite3.Error as exc:
            self._log.error("Ошибка применения лимита долговременной памяти: %s", exc)

    # ------------------------------------------------------------------
    # Вывод в запрос
    # ------------------------------------------------------------------

    def to_text(self, counter: Any = None, max_tokens: Optional[int] = None) -> str:
        """Текст блока долговременной памяти (с приоритетным усечением по бюджету).

        Разделы выводятся в порядке ``profile`` → ``decisions`` → ``knowledge``.
        """
        data = self.load()
        lines = [LONG_HEADER]
        for section in SECTIONS:
            if section not in data:
                continue
            for key in sorted(data[section]):
                item = data[section][key]
                lines.append(
                    f"[{section}] {key}: {item.get('value', '')}"
                )
        text = "\n".join(lines)
        if max_tokens is not None and counter is not None:
            return _trim_lines(text, counter, max_tokens)
        return text

    def token_count(self, counter: Any = None) -> int:
        """Оценка токенов блока долговременной памяти."""
        return _count_text(counter, self.to_text(counter))

    # ------------------------------------------------------------------
    # Экспорт / импорт Markdown
    # ------------------------------------------------------------------

    def export_markdown(self, path: str | Path) -> bool:
        """Экспортирует память в Markdown (разделы → заголовки, записи → списки)."""
        data = self.load()
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            parts = ["# Долговременная память", ""]
            for section in SECTIONS:
                parts.append(f"## {section}")
                if section in data:
                    for key in sorted(data[section]):
                        item = data[section][key]
                        parts.append(f"- **{key}**: {item.get('value', '')}")
                parts.append("")
            target.write_text("\n".join(parts), encoding="utf-8")
            return True
        except OSError as exc:
            self._log.error("Ошибка экспорта долговременной памяти: %s", exc)
            return False

    def import_markdown(self, path: str | Path) -> int:
        """Импортирует записи из Markdown (разделы ``## имя``, записи ``- **key**: value``).

        Returns:
            Число импортированных записей.
        """
        source = Path(path)
        try:
            text = source.read_text(encoding="utf-8")
        except OSError as exc:
            self._log.error("Ошибка чтения файла импорта: %s", exc)
            return 0
        current: Optional[str] = None
        imported = 0
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                name = stripped[3:].strip().lower()
                try:
                    current = normalize_section(name)
                except MemoryError:
                    current = None
                continue
            match = re.match(r"^[-*]\s+\*\*(.+?)\*\*\s*:\s*(.+)$", stripped)
            if match and current is not None:
                key = match.group(1).strip()
                value = match.group(2).strip()
                if key and value and self.add(current, key, value, source="import"):
                    imported += 1
        return imported

    def close(self) -> None:
        """Закрывает хранилище (идемпотентно)."""
        self._closed = True


# ---------------------------------------------------------------------------
# Предложения классификации (pending)
# ---------------------------------------------------------------------------

class PendingMemoryStore:
    """Предложения классификации от LLM (таблица ``pending_memory``).

    Кандидаты живут в том же файле, что и долговременная память, но в слои
    **не попадают** до явного ``accept``.
    """

    def __init__(self, long_store: LongTermStore) -> None:
        self._long = long_store
        self._log = utils.get_logger("deepseek-chat.memory.pending")

    def add(self, items: Sequence[Dict[str, Any]],
            source_message: Optional[int] = None) -> int:
        """Добавляет кандидатов (дедупликация по (section|layer, key, value)).

        Returns:
            Число реально добавленных кандидатов.
        """
        if not self._long.enabled or self._long._closed:
            return 0
        added = 0
        existing = self._existing_signatures()
        try:
            self._long._prepare_disk()
            with self._long._session() as conn:
                self._long._ensure_schema(conn)
                now = _now()
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    layer = item.get("layer", LONG_TERM)
                    section = str(item.get("section") or "").strip().lower() or None
                    key = str(item.get("key") or "").strip()
                    value = _clean(item.get("value"))
                    if not value:
                        continue
                    signature = (section, layer, key, value)
                    if signature in existing:
                        continue
                    conn.execute(
                        "INSERT INTO pending_memory(layer, section, key, value, "
                        "kind, confidence, source_message, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (layer, section, key, value,
                         str(item.get("kind") or "").strip() or None,
                         item.get("confidence"), source_message, now),
                    )
                    existing.add(signature)
                    added += 1
            return added
        except sqlite3.Error as exc:
            self._log.error("Ошибка записи предложений памяти: %s", exc)
            return added

    def _existing_signatures(self) -> set:
        existing = set()
        for item in self.list():
            existing.add((
                str(item.get("section") or "").strip().lower() or None,
                item.get("layer"),
                item.get("key"),
                item.get("value"),
            ))
        return existing

    def list(self) -> List[Dict[str, Any]]:
        """Все кандидаты (id, слой, раздел, ключ, значение, confidence)."""
        if not self._long.enabled or self._long._closed or not self._long.exists:
            return []
        try:
            with self._long._session() as conn:
                self._long._ensure_schema(conn)
                rows = conn.execute(
                    "SELECT id, layer, section, key, value, kind, confidence, "
                    "source_message, created_at FROM pending_memory ORDER BY id"
                ).fetchall()
            return [
                {
                    "id": int(row[0]), "layer": row[1], "section": row[2],
                    "key": row[3], "value": row[4], "kind": row[5],
                    "confidence": row[6], "source_message": row[7],
                    "created_at": row[8],
                }
                for row in rows
            ]
        except sqlite3.Error as exc:
            self._log.error("Ошибка чтения предложений памяти: %s", exc)
            return []

    def get(self, pending_id: int) -> Optional[Dict[str, Any]]:
        """Кандидат по id (None, если нет)."""
        for item in self.list():
            if int(item.get("id") or -1) == int(pending_id):
                return item
        return None

    def _delete(self, pending_id: int) -> bool:
        if not self._long.enabled or self._long._closed or not self._long.exists:
            return False
        try:
            with self._long._session() as conn:
                self._long._ensure_schema(conn)
                cursor = conn.execute(
                    "DELETE FROM pending_memory WHERE id = ?", (int(pending_id),)
                )
            return cursor.rowcount > 0
        except sqlite3.Error as exc:
            self._log.error("Ошибка удаления предложения памяти: %s", exc)
            return False

    def accept(self, pending_id: int) -> Optional[Dict[str, Any]]:
        """Удаляет кандидата из pending и возвращает его (для записи в слой)."""
        item = self.get(pending_id)
        if item is None:
            return None
        self._delete(pending_id)
        return item

    def reject(self, pending_id: int) -> bool:
        """Отклоняет кандидата (удаляет без записи)."""
        return self._delete(pending_id)

    def clear(self) -> bool:
        """Отклоняет все предложения."""
        if not self._long.enabled or self._long._closed or not self._long.exists:
            return False
        try:
            with self._long._session() as conn:
                self._long._ensure_schema(conn)
                conn.execute("DELETE FROM pending_memory")
            return True
        except sqlite3.Error as exc:
            self._log.error("Ошибка очистки предложений памяти: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Классификация: локальные правила + LLM
# ---------------------------------------------------------------------------

class MemoryClassifier:
    """Классификация текста по слоям памяти.

    Локальные правила разбирают явные маркеры («навсегда» → long_term,
    «по этой задаче/дедлайн» → working, «на этот диалог» → short_term).
    LLM-классификация выполняется одним служебным запросом с маркером
    :data:`MEMORY_MARKER` и строгим JSON.
    """

    #: Маркеры, указывающие на долговременную память
    _LONG_RE = re.compile(
        r"\b(навсегда|всегда|долговременн\w+|профиль|запомни\s+меня\b)\b",
        re.IGNORECASE,
    )
    #: Маркеры рабочей памяти (текущая задача)
    _WORKING_RE = re.compile(
        r"\b(по\s+этой\s+задаче|для\s+задачи|задач\w+|план|дедлайн|срок|"
        r"статус|ограничени\w+|открыты\w+\s+вопрос\w*|артефакт)\b",
        re.IGNORECASE,
    )
    #: Маркеры краткосрочной памяти (текущий диалог)
    _SHORT_RE = re.compile(
        r"\b(на\s+этот\s+диалог|учти\s+в\s+диалоге|в\s+этом\s+диалоге|"
        r"здесь\s+и\s+сейчас)\b",
        re.IGNORECASE,
    )

    @staticmethod
    def classify_local(text: str) -> Dict[str, Any]:
        """Предложения по слоям локальными правилами (без LLM).

        Returns:
            Словарь в формате ``{short_term: [...], working: [...],
            long_term: [...]}`` (списки могут быть пустыми).
        """
        proposals: Dict[str, Any] = {
            SHORT_TERM: [], WORKING: [], LONG_TERM: [],
        }
        cleaned = _clean(text)
        if not cleaned:
            return proposals

        long_hit = bool(MemoryClassifier._LONG_RE.search(cleaned))
        working_hit = bool(MemoryClassifier._WORKING_RE.search(cleaned))
        short_hit = bool(MemoryClassifier._SHORT_RE.search(cleaned))

        # Разбор «ключ — значение» (как в FactsStore.extract_local)
        for match in re.finditer(
            r"\b(запомни|сохрани)\b[^,.;!?]{0,40}?[:—–-]\s*([^,.;!?]{1,120})",
            cleaned, re.IGNORECASE,
        ):
            key = match.group(1)
            value = match.group(2).strip(" .,;:—–-")
            if not value:
                continue
            if long_hit:
                proposals[LONG_TERM].append({
                    "section": "profile", "key": key, "value": value,
                    "confidence": 0.9,
                })
            elif working_hit:
                proposals[WORKING].append({
                    "key": key, "value": value,
                    "kind": "decision" if "реши" in cleaned.lower() else "note",
                    "confidence": 0.9,
                })
            else:
                proposals[SHORT_TERM].append({
                    "text": f"{key}: {value}", "confidence": 0.9,
                })

        # Имя/город/предпочтения → долговременный профиль (маркер «меня зовут»)
        name_match = re.search(
            r"(?:меня\s+зовут|мо[её]\s+имя)\s+([^\s,.!?;]+)",
            cleaned, re.IGNORECASE,
        )
        if name_match and (long_hit or "профиль" in cleaned.lower()
                           or "навсегда" in cleaned.lower()):
            proposals[LONG_TERM].append({
                "section": "profile", "key": "имя",
                "value": name_match.group(1), "confidence": 0.9,
            })

        # Дедлайн → рабочая память
        deadline = re.search(
            r"дедлайн[:\s-]*([0-9]{1,2}\s+[а-яА-Я]+\w*)", cleaned, re.IGNORECASE,
        )
        if deadline:
            proposals[WORKING].append({
                "key": "дедлайн", "value": deadline.group(1),
                "kind": "note", "confidence": 0.9,
            })

        # Если маркер есть, но ничего не распарсили — предложить текст целиком
        if not proposals[LONG_TERM] and not proposals[WORKING] \
                and not proposals[SHORT_TERM]:
            if long_hit:
                proposals[LONG_TERM].append({
                    "section": "profile", "key": "",
                    "value": cleaned, "confidence": 0.6,
                })
            elif working_hit:
                proposals[WORKING].append({
                    "key": "", "value": cleaned, "kind": "note",
                    "confidence": 0.6,
                })
            elif short_hit:
                proposals[SHORT_TERM].append({
                    "text": cleaned, "confidence": 0.6,
                })

        return proposals

    @staticmethod
    def build_request(text: str, context_hint: str = "") -> Dict[str, Any]:
        """Формирует служебный запрос классификации (для ``_call_service_api``).

        Args:
            text: сообщение для классификации.
            context_hint: дополнительный контекст (например, последний обмен).

        Returns:
            Словарь параметров ``chat.completions.create``.
        """
        parts = [f"Фрагмент диалога:\n{text}"]
        if context_hint:
            parts.append(f"Контекст:\n{context_hint}")
        parts.append("Верни кандидатов для памяти одним JSON-объектом.")
        return {
            "messages": [
                {"role": "system", "content": DEFAULT_MEMORY_PROMPT},
                {"role": "user", "content": "\n\n".join(parts)},
            ],
            "temperature": 0.0,
            "max_tokens": DEFAULT_MEMORY_REQUEST_MAX_TOKENS,
        }

    @staticmethod
    def parse(raw_text: str) -> Dict[str, Any]:
        """Разбирает JSON-ответ классификатора.

        При мусоре возвращает ``{"error": ...}`` и пустые списки — прежние
        данные при этом не трогаются.
        """
        empty = {SHORT_TERM: [], WORKING: [], LONG_TERM: []}
        if not raw_text or not raw_text.strip():
            return {**empty, "error": "Модель вернула пустой ответ классификации."}
        parsed = extract_json_object(raw_text)
        if not isinstance(parsed, dict):
            return {**empty, "error": "Не удалось разобрать JSON классификации."}
        result = {**empty}
        for layer in LAYERS:
            raw_list = parsed.get(layer)
            if not isinstance(raw_list, list):
                continue
            for item in raw_list:
                if not isinstance(item, dict):
                    continue
                cleaned = {k: v for k, v in item.items() if v is not None}
                result[layer].append(cleaned)
        return result

    @staticmethod
    def classify_llm(text: str, call_service, settings: Dict[str, Any],
                     context_hint: str = "") -> Dict[str, Any]:
        """Только LLM-предложение (без локальных правил).

        Используется для генерации ``pending`` после хода. Ошибка служебного
        запроса не пробрасывается; кандидаты фильтруются по
        ``MEMORY_MIN_CONFIDENCE`` и лимитируются по числу.

        Returns:
            ``{short_term, working, long_term, error, tokens}``.
        """
        empty = {SHORT_TERM: [], WORKING: [], LONG_TERM: []}
        result: Dict[str, Any] = {
            SHORT_TERM: [], WORKING: [], LONG_TERM: [],
            "error": None, "tokens": None,
        }
        if call_service is None:
            return result
        request = MemoryClassifier.build_request(text, context_hint)
        model = str(settings.get("model") or "").strip()
        if model:
            request["model"] = model
        request["temperature"] = float(settings.get("temperature", 0.0))
        try:
            payload = call_service(request) or {}
        except Exception as exc:
            result["error"] = str(exc)
            logger.warning("Классификация памяти не удалась: %s", exc)
            return result
        tokens = dict(payload.get("tokens") or {})
        tokens.setdefault("estimated", False)
        tokens["model"] = payload.get("model") or model
        result["tokens"] = tokens
        parsed = MemoryClassifier.parse(str(payload.get("content") or ""))
        if parsed.get("error"):
            result["error"] = parsed["error"]
            return result
        min_conf = float(settings.get("min_confidence", DEFAULT_MIN_CONFIDENCE))
        for layer in LAYERS:
            filtered = [
                item for item in parsed.get(layer, [])
                if isinstance(item, dict)
                and (float(item.get("confidence") or 1.0) >= min_conf)
            ]
            result[layer] = filtered[:20]
        return result

    @staticmethod
    def classify(text: str, call_service, settings: Dict[str, Any],
                 context_hint: str = "") -> Dict[str, Any]:
        """LLM-предложение + локальные правила (для предзаполнения диалога)."""
        result = MemoryClassifier.classify_llm(text, call_service, settings,
                                               context_hint)
        local = MemoryClassifier.classify_local(text)
        for layer in LAYERS:
            existing_keys = {
                str(item.get("key") or item.get("text") or "").strip()
                for item in result[layer]
            }
            for item in local.get(layer, []):
                candidate_key = str(item.get("key") or item.get("text") or "").strip()
                if candidate_key and candidate_key in existing_keys:
                    continue
                result[layer].append(item)
        return result


# ---------------------------------------------------------------------------
# Роутер и владелец слоёв
# ---------------------------------------------------------------------------

class MemoryManager:
    """Единственный роутер и владелец слоёв памяти.

    Через :meth:`route` проходят все записи в слои (кнопки у сообщений, окно
    «Память…», текстовые команды, подтверждение предложений). Никакой код не
    пишет в слои в обход роутера.

    Attributes:
        settings: настройки памяти (см. ``config.get_memory_settings``).
        working: рабочая память активной ветки (:class:`WorkingMemory`).
        long: долговременная память (:class:`LongTermStore`).
        pending: предложения классификации (:class:`PendingMemoryStore`).
        pins: закреплённые сообщения ``{index: note|None}`` (без system).
        counter: счётчик токенов.
        facts_store: ссылка на ``FactsStore`` (для sticky_facts, опционально).
    """

    def __init__(self, settings: Optional[Dict[str, Any]] = None,
                 long_store: Optional[LongTermStore] = None,
                 working: Optional[WorkingMemory] = None,
                 counter: Any = None,
                 facts_store: Any = None) -> None:
        self.settings: Dict[str, Any] = dict(settings or {})
        self.counter = counter
        self.facts_store = facts_store
        self.pins: Dict[int, Optional[str]] = {}
        self.working = working if working is not None else WorkingMemory(counter)
        self.long = long_store if long_store is not None else LongTermStore(
            ":memory:", enabled=False
        )
        self.pending = PendingMemoryStore(self.long)

    # ------------------------------------------------------------------
    # Роутер (единственная точка записи)
    # ------------------------------------------------------------------

    def route(self, layer: Any, text: str, *, key: Optional[str] = None,
              section: Optional[str] = None, kind: Optional[str] = None,
              source: str = "manual", message_index: Optional[int] = None,
              branch_id: Optional[int] = None) -> Dict[str, Any]:
        """Записывает ``text`` строго в выбранный слой.

        Args:
            layer: слой (строка или русская подпись кнопки).
            text: значение для сохранения.
            key: ключ (пустой → вывод из текста/автогенерация).
            section: раздел долговременной памяти.
            kind: вид элемента рабочей памяти.
            source: источник ("manual"/"confirmed"/"local"/"import").
            message_index: индекс сообщения-источника (без system).
            branch_id: для лога (id ветки).

        Returns:
            ``{"ok", "layer", "key", "value", "section", "error"}``.
        """
        try:
            canonical = normalize_layer(layer)
        except MemoryError as exc:
            return {"ok": False, "layer": str(layer), "error": str(exc)}

        if canonical == SHORT_TERM:
            if message_index is None:
                return {
                    "ok": False, "layer": canonical,
                    "error": "Для краткосрочной памяти нужен индекс сообщения.",
                }
            result = self.pin(int(message_index), note=_clean(text) or None)
            result["layer"] = canonical
            return result

        value = _clean(text)
        if not value:
            return {"ok": False, "layer": canonical, "error": "Пустое значение."}

        resolved_key = self._resolve_key(key, value)
        if canonical == WORKING:
            try:
                self.working.remember(
                    resolved_key, value, kind or "note",
                    source=source, message_index=message_index,
                )
            except MemoryError as exc:
                return {"ok": False, "layer": canonical, "error": str(exc)}
            self._sync_working_to_facts(resolved_key, value, kind or "note",
                                        source, message_index)
            logger.info(
                "Память: слой=working, ключ=%r, источник=%s, ветка=%s.",
                resolved_key, source, branch_id,
            )
            return {
                "ok": True, "layer": canonical, "key": resolved_key,
                "value": value, "section": None, "error": None,
            }

        # long_term
        try:
            section_name = normalize_section(section or "profile")
        except MemoryError as exc:
            return {"ok": False, "layer": canonical, "error": str(exc)}
        ok = self.long.add(section_name, resolved_key, value,
                           source=source, source_message=message_index)
        if not ok:
            return {
                "ok": False, "layer": canonical, "key": resolved_key,
                "value": value, "section": section_name,
                "error": "Не удалось записать в долговременную память.",
            }
        logger.info(
            "Память: слой=long_term, раздел=%s, ключ=%r, источник=%s.",
            section_name, resolved_key, source,
        )
        return {
            "ok": True, "layer": canonical, "key": resolved_key,
            "value": value, "section": section_name, "error": None,
        }

    def _resolve_key(self, key: Optional[str], value: str) -> str:
        """Нормализует ключ: пустой → вывод из текста/автогенерация ``заметка-N``."""
        key = " ".join(str(key or "").split()).strip().lower().replace(" ", "_")
        if key:
            return key
        words = [w for w in re.split(r"\W+", value) if w][:3]
        if words:
            return "_".join(words).lower()
        used = {k for k in self.working.items}
        index = 1
        while f"заметка-{index}" in used:
            index += 1
        return f"заметка-{index}"

    def _sync_working_to_facts(self, key: str, value: str, kind: str,
                               source: str, message_index: Optional[int]) -> None:
        """Зеркалит decision/artifact в FactsStore (для стратегии sticky_facts).

        Ключ→значение рабочей памяти — те же объекты, что и ``FactsStore``:
        блок фактов уже содержит их, поэтому рабочая память не дублирует их в
        своём блоке (см. ``build_blocks``).
        """
        category = _KIND_TO_FACTS_CATEGORY.get(kind)
        if category is None or self.facts_store is None:
            return
        try:
            self.facts_store.remember(key, value, category)
        except Exception as exc:
            logger.warning("Не удалось зеркалировать рабочий элемент в факты: %s", exc)

    # ------------------------------------------------------------------
    # Краткосрочная память: закрепление (pin)
    # ------------------------------------------------------------------

    def pin(self, message_index: int, note: Optional[str] = None) -> Dict[str, Any]:
        """Закрепляет сообщение (всегда уходит в запрос).

        При превышении ``MEMORY_PIN_MAX`` вытесняется самое старое закрепление
        (WARNING в лог).
        """
        index = int(message_index)
        pin_max = int(self.settings.get("pin_max", DEFAULT_PIN_MAX))
        note = _clean(note) or None
        if index in self.pins:
            self.pins[index] = note
        else:
            if len(self.pins) >= pin_max:
                oldest = min(self.pins)
                logger.warning(
                    "Превышен лимит закреплений (%d): вытесняю сообщение №%d.",
                    pin_max, oldest,
                )
                del self.pins[oldest]
            self.pins[index] = note
        logger.info("Закреплено сообщение №%d (всего %d).", index, len(self.pins))
        return {
            "ok": True, "layer": SHORT_TERM, "key": str(index),
            "value": note or "", "section": None, "error": None,
        }

    def unpin(self, message_index: int) -> Dict[str, Any]:
        """Открепляет сообщение."""
        index = int(message_index)
        removed = index in self.pins
        self.pins.pop(index, None)
        return {
            "ok": removed, "layer": SHORT_TERM, "key": str(index),
            "value": "", "section": None,
            "error": None if removed else f"Сообщение №{index} не было закреплено.",
        }

    def toggle_pin(self, message_index: int,
                   note: Optional[str] = None) -> Dict[str, Any]:
        """Закрепляет/открепляет сообщение."""
        if int(message_index) in self.pins:
            return self.unpin(int(message_index))
        return self.pin(int(message_index), note=note)

    def list_pinned(self) -> List[Dict[str, Any]]:
        """Список закреплений (в хронологическом порядке индексов)."""
        return [
            {"index": index, "note": self.pins[index]}
            for index in sorted(self.pins)
        ]

    # ------------------------------------------------------------------
    # Рабочая память: публичные операции
    # ------------------------------------------------------------------

    def working_set_field(self, field: str, value: Any) -> Dict[str, Any]:
        try:
            self.working.set_field(field, value)
            return {"ok": True, "error": None}
        except MemoryError as exc:
            return {"ok": False, "error": str(exc)}

    def working_drop_field(self, field: str) -> Dict[str, Any]:
        try:
            self.working.drop_field(field)
            return {"ok": True, "error": None}
        except MemoryError as exc:
            return {"ok": False, "error": str(exc)}

    def working_remember(self, key: str, value: str, kind: str = "note",
                         source: str = "manual",
                         message_index: Optional[int] = None) -> Dict[str, Any]:
        return self.route(WORKING, value, key=key, kind=kind, source=source,
                          message_index=message_index)

    def working_forget(self, key: str) -> Dict[str, Any]:
        ok = self.working.forget(key)
        return {"ok": ok, "error": None if ok else f"Элемент {key!r} не найден."}

    def working_clear(self) -> Dict[str, Any]:
        self.working.clear()
        return {"ok": True, "error": None}

    def start_new_task(self, carry_to_long: Optional[List[str]] = None) -> Dict[str, Any]:
        """«Новая задача»: очищает рабочую память.

        Args:
            carry_to_long: ключи элементов (``decision``), которые нужно
                перенести в ``long_term.decisions`` перед очисткой.

        Returns:
            ``{"ok", "carried", "error"}``.
        """
        carried = 0
        for key in (carry_to_long or []):
            item = self.working.items.get(str(key).strip())
            if item is None:
                continue
            if self.long.enabled:
                if self.long.add("decisions", str(key).strip(),
                                 item.get("value", ""),
                                 source="confirmed",
                                 source_message=item.get("source_message")):
                    carried += 1
        self.working.clear()
        return {"ok": True, "carried": carried, "error": None}

    # ------------------------------------------------------------------
    # Долговременная память: публичные операции
    # ------------------------------------------------------------------

    def long_add(self, section: str, key: str, value: str,
                 source: str = "manual",
                 message_index: Optional[int] = None) -> Dict[str, Any]:
        return self.route(LONG_TERM, value, key=key, section=section,
                          source=source, message_index=message_index)

    def long_forget(self, section: str, key: str) -> Dict[str, Any]:
        try:
            section_name = normalize_section(section)
        except MemoryError as exc:
            return {"ok": False, "error": str(exc)}
        ok = self.long.forget(section_name, key)
        return {"ok": ok, "error": None if ok else "Запись не найдена."}

    def long_clear_section(self, section: str) -> Dict[str, Any]:
        try:
            section_name = normalize_section(section)
        except MemoryError as exc:
            return {"ok": False, "error": str(exc)}
        ok = self.long.clear_section(section_name)
        return {"ok": ok, "error": None}

    def long_clear_all(self) -> Dict[str, Any]:
        ok = self.long.clear_all()
        return {"ok": ok, "error": None}

    def long_export(self, path: str | Path) -> Dict[str, Any]:
        ok = self.long.export_markdown(path)
        return {"ok": ok, "error": None if ok else "Не удалось экспортировать."}

    def long_import(self, path: str | Path) -> Dict[str, Any]:
        imported = self.long.import_markdown(path)
        return {"ok": imported > 0, "imported": imported, "error": None}

    # ------------------------------------------------------------------
    # Предложения (pending)
    # ------------------------------------------------------------------

    def pending_list(self) -> List[Dict[str, Any]]:
        return self.pending.list()

    def pending_accept(self, pending_id: int,
                       layer: Optional[str] = None) -> Dict[str, Any]:
        """Принимает кандидата и пишет в выбранный слой (по умолчанию — его слой)."""
        item = self.pending.accept(int(pending_id))
        if item is None:
            return {"ok": False, "error": f"Предложение id={pending_id} не найдено."}
        target = normalize_layer(layer or item.get("layer") or LONG_TERM)
        if target == SHORT_TERM:
            result = {
                "ok": True, "layer": target, "key": None,
                "value": item.get("value", ""), "section": None, "error": None,
            }
            # Краткосрочная: закрепляем, если есть source_message
            if item.get("source_message") is not None:
                result.update(self.pin(int(item["source_message"])))
            return result
        if target == WORKING:
            return self.route(
                WORKING, item.get("value", ""),
                key=item.get("key"), kind=item.get("kind") or "note",
                source="confirmed", message_index=item.get("source_message"),
            )
        return self.route(
            LONG_TERM, item.get("value", ""),
            key=item.get("key"), section=item.get("section") or "profile",
            source="confirmed", message_index=item.get("source_message"),
        )

    def pending_reject(self, pending_id: int) -> Dict[str, Any]:
        ok = self.pending.reject(int(pending_id))
        return {"ok": ok, "error": None if ok else "Предложение не найдено."}

    def pending_clear(self) -> Dict[str, Any]:
        ok = self.pending.clear()
        return {"ok": ok, "error": None}

    # ------------------------------------------------------------------
    # Отчёт
    # ------------------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        """«Какие данные в каком слое»: элементы, количество, токены, источник."""
        long_items = self.long.list_items() if self.long.enabled else []
        long_count = len(long_items)
        long_tokens = self.long.token_count(self.counter) if self.long.enabled else 0

        working_items = [
            {"key": key, "value": item.get("value", ""),
             "kind": item.get("kind", "note"),
             "source": item.get("source", "manual"),
             "source_message": item.get("source_message"),
             "updated_at": item.get("updated_at")}
            for key, item in self.working.items.items()
        ]
        working_tokens = self.working.token_count(self.counter)

        return {
            SHORT_TERM: {
                "messages": None,  # число сообщений знает агент
                "pinned": self.list_pinned(),
                "pinned_count": len(self.pins),
                "items": self.list_pinned(),
                "count": len(self.pins),
                "tokens": 0,
                "source": "manual",
            },
            WORKING: {
                "task": self.working.as_dict()["task"],
                "items": working_items,
                "count": len(working_items),
                "tokens": working_tokens,
                "source": "manual",
            },
            LONG_TERM: {
                "items": long_items,
                "count": long_count,
                "sections": sorted({i.get("section") for i in long_items}),
                "tokens": long_tokens,
                "source": "manual",
            },
            "pending": self.pending_list(),
            "budget": {
                "max_tokens": int(self.settings.get("max_tokens",
                                                    DEFAULT_MEMORY_MAX_TOKENS)),
                "long_max_tokens": int(self.settings.get("long_max_tokens",
                                                         DEFAULT_LONG_MAX_TOKENS)),
                "working_max_tokens": int(self.settings.get(
                    "working_max_tokens", DEFAULT_WORKING_MAX_TOKENS)),
            },
        }

    # ------------------------------------------------------------------
    # Сборка блоков запроса
    # ------------------------------------------------------------------

    def build_blocks(self, counter: Any, settings: Dict[str, Any],
                     facts: Any = None, facts_active: bool = False,
                     pinned_messages: Optional[List[Dict[str, Any]]] = None,
                     include_pins: bool = True) -> List[Message]:
        """Три ``system``-сообщения в фиксированном порядке.

        Порядок: долговременная → рабочая → закреплённые сообщения. Пустые
        слои не добавляются. Блок рабочей памяти не повторяет пары, которые
        уже уходят блоком фактов ``sticky_facts``.

        Args:
            counter: счётчик токенов.
            settings: настройки памяти.
            facts: ``FactsStore`` (для проверки дублей).
            facts_active: True, если блок фактов уйдёт в запрос.
            pinned_messages: список ``{index, role, content, note}``.
            include_pins: False — не добавлять блок закреплённых (для тестов
                сравнения контекста без памяти).

        Returns:
            Список system-сообщений (может быть пустым).
        """
        blocks: List[Message] = []
        long_max = int(settings.get("long_max_tokens", DEFAULT_LONG_MAX_TOKENS))
        working_max = int(settings.get("working_max_tokens",
                                       DEFAULT_WORKING_MAX_TOKENS))
        max_tokens = int(settings.get("max_tokens", DEFAULT_MEMORY_MAX_TOKENS))

        # 1. Долговременная
        if settings.get("long_enabled", True) and self.long.enabled:
            long_text = self.long.to_text(counter, long_max)
            if long_text.strip() and long_text.strip() != LONG_HEADER:
                blocks.append({"role": "system", "content": long_text})

        # 2. Рабочая (без дублей пар facts)
        if settings.get("working_enabled", True):
            skip_keys = self._facts_skip_keys(facts, facts_active)
            working_text = self.working.to_text(counter, working_max,
                                                skip_keys=skip_keys)
            if working_text.strip() and working_text.strip() != WORKING_HEADER:
                blocks.append({"role": "system", "content": working_text})

        # 3. Закреплённые сообщения
        if include_pins and pinned_messages:
            lines = [PINNED_HEADER]
            for message in pinned_messages:
                role = message.get("role", "user")
                role_label = "Пользователь" if role == "user" else "Ассистент"
                content = str(message.get("content") or "")
                note = message.get("note")
                entry = f"[{message.get('index')}] {role_label}: {content}"
                if note:
                    entry += f" (заметка: {note})"
                lines.append(entry)
            blocks.append({"role": "system", "content": "\n".join(lines)})

        # Общий бюджет: если превышен — усекаем рабочий, затем долговременный
        if counter is not None and max_tokens > 0:
            total = sum(_count_text(counter, b["content"]) for b in blocks)
            if total > max_tokens:
                logger.warning(
                    "Блоки памяти превышают MEMORY_MAX_TOKENS=%d "
                    "(≈%d токенов) — усекаю.", max_tokens, total,
                )
                # Усекаем рабочий и долговременный блоки (пины не трогаем)
                for index, block in enumerate(blocks):
                    if block["content"].startswith(WORKING_HEADER):
                        block["content"] = _trim_lines(
                            block["content"], counter, max(1, working_max // 2)
                        )
                    elif block["content"].startswith(LONG_HEADER):
                        block["content"] = _trim_lines(
                            block["content"], counter, max(1, long_max // 2)
                        )

        return blocks

    def _facts_skip_keys(self, facts: Any, facts_active: bool) -> set:
        """Ключи рабочих элементов, которые уже уходят блоком фактов."""
        if not facts_active or facts is None:
            return set()
        skip = set()
        for key in list(self.working.items):
            item = self.working.items[key]
            if item.get("kind") not in _KIND_TO_FACTS_CATEGORY:
                continue
            try:
                fact_value = facts.get(key)
            except Exception:
                fact_value = None
            if fact_value == item.get("value"):
                skip.add(key)
        return skip

    def close(self) -> None:
        """Закрывает долговременную память (идемпотентно)."""
        self.long.close()
