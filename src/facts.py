"""Sticky Facts: блок «ключ → значение» с важными данными диалога.

Модуль реализует стратегию **Sticky Facts / Key-Value Memory**:

- факты хранятся плоско, по категориям ``goal``, ``constraints``,
  ``preferences``, ``decisions``, ``agreements``;
- обновление выполняется служебным LLM-запросом после каждого хода
  (строгий JSON), а явные команды пользователя («запомни, что …»,
  «меня зовут …», «мой … — …») разбираются локально и работают даже
  при сбое LLM;
- блок фактов уходит в запрос отдельным system-сообщением сразу после
  основного системного промпта и перед последними N сообщениями.

Маркер ``FACTS_MARKER`` обязательно присутствует в служебном запросе: по нему
служебный вызов отличается от обычного (тесты, sim-режим бенчмарка, логи).
В обычные запросы диалога маркер не попадает.

Модуль не импортирует tkinter и не выполняет сетевых вызовов: служебный запрос
выполняет переданный снаружи callable (в тестах — подмена).
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import utils

logger = utils.get_logger("deepseek-chat.facts")

#: Тип сообщения чата
Message = Dict[str, str]

#: Маркер служебного запроса извлечения фактов (см. модульный docstring)
FACTS_MARKER = "[[ОБНОВЛЕНИЕ_ФАКТОВ]]"

#: Фиксированный набор категорий фактов
FACT_CATEGORIES: Tuple[str, ...] = (
    "goal", "constraints", "preferences", "decisions", "agreements",
)

#: Значения по умолчанию
DEFAULT_FACTS_MAX_ITEMS = 50
DEFAULT_FACTS_MAX_TOKENS = 300
DEFAULT_FACTS_TEMPERATURE = 0.0
DEFAULT_FACTS_REQUEST_MAX_TOKENS = 600

#: Заголовок блока фактов в system-сообщении
FACTS_HEADER = "Известные факты о диалоге:"

#: Промпт-инструкция извлечения фактов по умолчанию
DEFAULT_FACTS_PROMPT = (
    "Ты — модуль извлечения фактов из диалога. По текущим фактам и новому "
    "обмену сообщениями верни ОБНОВЛЁННЫЙ полный набор фактов.\n"
    "Требования:\n"
    "— ответь СТРОГО одним JSON-объектом, без пояснений и без markdown;\n"
    "— допустимые категории (ключи верхнего уровня): goal, constraints, "
    "preferences, decisions, agreements;\n"
    "— внутри категории — плоские пары «ключ»: «значение», значения только "
    "строки (например: {\"preferences\": {\"имя\": \"Алекс\"}});\n"
    "— сохраняй прежние факты, если они не устарели и не противоречат новым;\n"
    "— новое значение для того же ключа заменяет старое;\n"
    "— чтобы удалить факт, передай для его ключа null;\n"
    "— не выдумывай факты, которых не было в диалоге;\n"
    "— пиши на языке диалога, ключи — короткие и понятные.\n"
    f"Служебная метка запроса: {FACTS_MARKER}"
)


def _now() -> str:
    """Текущее локальное время в формате ISO (секунды)."""
    return datetime.now().isoformat(timespec="seconds")


def _strip_code_fence(text: str) -> str:
    """Убирает markdown-ограждение ```json … ``` вокруг ответа модели."""
    stripped = (text or "").strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Извлекает первый сбалансированный JSON-объект из текста ответа.

    Устойчиво к markdown-ограждению, вступительным фразам модели и мусору
    после объекта. Кавычки и экранирование внутри строк учитываются, поэтому
    фигурные скобки в значениях не ломают разбор.

    Args:
        text: сырой текст ответа модели.

    Returns:
        Разобранный словарь либо None, если объекта нет/он невалиден.
    """
    if not text:
        return None
    cleaned = _strip_code_fence(text)

    start = cleaned.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(cleaned)):
            char = cleaned[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start:index + 1]
                    try:
                        parsed = json.loads(candidate)
                    except (ValueError, TypeError):
                        break
                    return parsed if isinstance(parsed, dict) else None
        start = cleaned.find("{", start + 1)
    return None


# ---------------------------------------------------------------------------
# Локальный разбор явных команд пользователя
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(r"(?:меня\s+зовут|мо[её]\s+имя)\s+([^\s,.!?;]+)", re.IGNORECASE)
_MY_RE = re.compile(
    r"\bмо[йяе]\s+([^,.;!?]{2,40}?)\s*[—–\-:]\s*([^,.;!?]{1,80})",
    re.IGNORECASE,
)
_CITY_RE = re.compile(r"\bя\s+живу\s+в\s+([^\s,.!?;]+)", re.IGNORECASE)
_REMEMBER_RE = re.compile(
    r"\bзапомни\b[\s,:]*(?:что\s+)?(.+)", re.IGNORECASE | re.DOTALL
)
_POLITE_RE = re.compile(r"^это\b", re.IGNORECASE)
_SPLIT_RE = re.compile(r"\s*[:—–]\s*|\s+-\s+")


class FactsStore:
    """Хранилище фактов диалога: ``{категория: {ключ: значение}}``.

    Объект владеет «плоским» словарём фактов (:attr:`data`) и параллельно
    хранит отметки времени обновления каждого ключа (:attr:`updated_at`) —
    по ним вытесняются самые старые факты при превышении лимитов.

    Атрибут :attr:`data` — тот же объект, который агент отдаёт наружу как
    «факты»: мутации выполняются на месте, поэтому ссылки не рассинхронизируются.

    Attributes:
        categories: допустимые категории фактов.
        max_items: максимум пар «ключ → значение».
        max_tokens: лимит блока фактов в токенах.
        counter: счётчик токенов (может быть задан позже).
        data: плоский словарь фактов.
        updated_at: отметки времени обновления ключей.
    """

    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        updated_at: Optional[Dict[str, Any]] = None,
        categories: Iterable[str] = FACT_CATEGORIES,
        max_items: int = DEFAULT_FACTS_MAX_ITEMS,
        max_tokens: int = DEFAULT_FACTS_MAX_TOKENS,
        counter: Any = None,
    ) -> None:
        self.categories: Tuple[str, ...] = tuple(categories)
        self.max_items = int(max_items)
        self.max_tokens = int(max_tokens)
        self.counter = counter
        #: Плоские факты: {категория: {ключ: значение}}
        self.data: Dict[str, Dict[str, str]] = {}
        #: Отметки обновления: {категория: {ключ: iso-время}}
        self.updated_at: Dict[str, Dict[str, str]] = {}
        self.from_dict({"data": data, "updated_at": updated_at})

    # ------------------------------------------------------------------
    # Сериализация
    # ------------------------------------------------------------------

    @staticmethod
    def normalize(raw: Any) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
        """Приводит произвольные данные к паре (факты, отметки времени).

        Устойчиво к мусору в JSON: неизвестные категории, не-словари и
        не-строковые значения молча отбрасываются (с предупреждением в лог).

        Args:
            raw: либо ``{"data": {...}, "updated_at": {...}}``, либо просто
                ``{категория: {ключ: значение}}``.

        Returns:
            Кортеж (data, updated_at).
        """
        payload = raw if isinstance(raw, dict) else {}
        if "data" in payload or "updated_at" in payload:
            data_raw = payload.get("data")
            updated_raw = payload.get("updated_at")
        else:
            data_raw = payload
            updated_raw = None

        data = FactsStore.merge({}, data_raw)
        updated_at: Dict[str, Dict[str, str]] = {}
        if isinstance(updated_raw, dict):
            for raw_cat, entries in updated_raw.items():
                category = str(raw_cat).strip().lower()
                if category not in data or not isinstance(entries, dict):
                    continue
                for raw_key, value in entries.items():
                    key = str(raw_key).strip()
                    if key in data.get(category, {}) and isinstance(value, str) and value:
                        updated_at.setdefault(category, {})[key] = value
        return data, updated_at

    def from_dict(self, raw: Any) -> "FactsStore":
        """Загружает факты из словаря (заменяет текущее содержимое).

        Args:
            raw: сериализованные факты (см. :meth:`normalize`).

        Returns:
            self (для цепочек вызовов).
        """
        data, updated_at = self.normalize(raw)
        self.data.clear()
        self.data.update(data)
        self.updated_at.clear()
        self.updated_at.update(updated_at)
        return self

    def as_dict(self) -> Dict[str, Any]:
        """Возвращает сериализуемое представление фактов (копию)."""
        return {
            "data": {cat: dict(values) for cat, values in self.data.items()},
            "updated_at": {
                cat: dict(values) for cat, values in self.updated_at.items()
            },
        }

    # ------------------------------------------------------------------
    # Доступ
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Общее число пар «ключ → значение»."""
        return sum(len(values) for values in self.data.values())

    def is_empty(self) -> bool:
        """True, если фактов нет."""
        return self.count() == 0

    def items(self) -> List[Tuple[str, str, str, str]]:
        """Все факты списком: (категория, ключ, значение, updated_at)."""
        rows: List[Tuple[str, str, str, str]] = []
        for category in self._category_order():
            for key in sorted(self.data.get(category, {})):
                rows.append((
                    category,
                    key,
                    self.data[category][key],
                    self.updated_at.get(category, {}).get(key, ""),
                ))
        return rows

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Значение первого найденного ключа (поиск по всем категориям)."""
        target = str(key).strip()
        for category in self._category_order():
            values = self.data.get(category, {})
            if target in values:
                return values[target]
        return default

    def remember(self, key: str, value: str,
                 category: str = "decisions") -> bool:
        """Запоминает факт вручную (публичное API агента).

        Args:
            key: ключ факта.
            value: значение (непустая строка).
            category: категория из :data:`FACT_CATEGORIES`.

        Returns:
            True, если факт сохранён; False — при некорректных аргументах.
        """
        category = str(category).strip().lower()
        key = str(key).strip()
        if category not in self.categories:
            logger.warning("Неизвестная категория факта %r — не сохраняю.", category)
            return False
        if not key or not isinstance(value, str) or not value.strip():
            logger.warning("Пустой ключ/значение факта — не сохраняю.")
            return False
        self.apply({category: {key: value}})
        return True

    def forget(self, key: str) -> bool:
        """Удаляет факт с указанным ключом из любой категории.

        Returns:
            True, если факт был найден и удалён.
        """
        target = str(key).strip()
        found = False
        for category, values in list(self.data.items()):
            if target in values:
                found = True
                self.apply({category: {target: None}})
        return found

    def clear(self) -> None:
        """Полностью очищает факты."""
        self.data.clear()
        self.updated_at.clear()

    # ------------------------------------------------------------------
    # Слияние
    # ------------------------------------------------------------------

    @staticmethod
    def merge(current: Any, incoming: Any,
              categories: Iterable[str] = FACT_CATEGORIES) -> Dict[str, Dict[str, str]]:
        """Сливает текущие факты с пришедшими (чистая функция).

        Правила:
        - новое строковое значение перезаписывает старое;
        - ``null`` или пустая строка удаляет ключ;
        - неизвестные категории и не-строковые значения отбрасываются
          с предупреждением в лог;
        - ключи, которых нет в ``incoming``, сохраняются как были.

        Args:
            current: текущие факты.
            incoming: новые факты (ответ модели или локальный разбор).
            categories: допустимый набор категорий.

        Returns:
            Новый словарь фактов (входные данные не изменяются).
        """
        allowed = tuple(categories)
        merged: Dict[str, Dict[str, str]] = {}
        if isinstance(current, dict):
            for raw_cat, values in current.items():
                category = str(raw_cat).strip().lower()
                if category not in allowed or not isinstance(values, dict):
                    continue
                cleaned = {
                    str(k).strip(): v.strip()
                    for k, v in values.items()
                    if isinstance(v, str) and v.strip() and str(k).strip()
                }
                if cleaned:
                    merged[category] = cleaned

        if not isinstance(incoming, dict):
            if incoming is not None:
                logger.warning("Ответ с фактами не является объектом JSON — игнорирую.")
            return merged

        for raw_cat, entries in incoming.items():
            category = str(raw_cat).strip().lower()
            if category not in allowed:
                logger.warning("Неизвестная категория фактов %r — игнорирую.", raw_cat)
                continue
            if not isinstance(entries, dict):
                logger.warning("Категория %r не является объектом — игнорирую.", category)
                continue
            target = merged.setdefault(category, {})
            for raw_key, raw_value in entries.items():
                key = str(raw_key).strip()
                if not key:
                    continue
                if raw_value is None or (
                    isinstance(raw_value, str) and not raw_value.strip()
                ):
                    target.pop(key, None)  # null/"" — удаление факта
                    continue
                if not isinstance(raw_value, str):
                    logger.warning(
                        "Не-строковое значение факта %r=%r отброшено.",
                        key, raw_value,
                    )
                    continue
                target[key] = raw_value.strip()
            if not target:
                merged.pop(category, None)
        return {cat: values for cat, values in merged.items() if values}

    def apply(self, incoming: Any, source: str = "llm") -> Dict[str, int]:
        """Применяет новые факты к хранилищу (мутация на месте).

        Args:
            incoming: новые факты (см. :meth:`merge`).
            source: пометка источника для лога ("llm", "local", "manual").

        Returns:
            Словарь {"added", "updated", "removed", "total"}.
        """
        before = {cat: dict(values) for cat, values in self.data.items()}
        merged = self.merge(self.data, incoming, self.categories)
        now = _now()

        added = updated = removed = 0
        for category, values in merged.items():
            for key, value in values.items():
                previous = before.get(category, {}).get(key)
                if previous is None:
                    added += 1
                    self.updated_at.setdefault(category, {})[key] = now
                elif previous != value:
                    updated += 1
                    self.updated_at.setdefault(category, {})[key] = now

        for category, values in before.items():
            for key in values:
                if key not in merged.get(category, {}):
                    removed += 1

        # Отметки времени исчезнувших ключей больше не нужны
        for category in list(self.updated_at):
            if category not in merged:
                self.updated_at.pop(category, None)
                continue
            for key in list(self.updated_at[category]):
                if key not in merged[category]:
                    self.updated_at[category].pop(key, None)

        self.data.clear()
        self.data.update(merged)
        if added or updated or removed:
            logger.info(
                "Факты обновлены (%s): +%d, ~%d, -%d, всего %d.",
                source, added, updated, removed, self.count(),
            )
        return {
            "added": added, "updated": updated,
            "removed": removed, "total": self.count(),
        }

    # ------------------------------------------------------------------
    # Вывод в запрос и лимиты
    # ------------------------------------------------------------------

    def _category_order(self) -> List[str]:
        """Порядок категорий: сначала фиксированный, затем прочие."""
        ordered = [c for c in self.categories if c in self.data]
        ordered += [c for c in sorted(self.data) if c not in self.categories]
        return ordered

    def to_text(self) -> str:
        """Текст блока фактов для system-сообщения.

        Формат::

            Известные факты о диалоге:
            [goal] цель: написать отчёт
            [preferences] любимый_цвет: синий

        Returns:
            Текст блока фактов (пустая строка, если фактов нет).
        """
        rows = self.items()
        if not rows:
            return ""
        lines = [FACTS_HEADER]
        lines += [f"[{category}] {key}: {value}" for category, key, value, _ in rows]
        return "\n".join(lines)

    def to_messages(self) -> List[Message]:
        """Блок фактов отдельным system-сообщением.

        Returns:
            Список из одного system-сообщения либо пустой список.
        """
        text = self.to_text()
        if not text:
            return []
        return [{"role": "system", "content": text}]

    def token_count(self, counter: Any = None) -> int:
        """Оценка токенов блока фактов."""
        engine = counter if counter is not None else self.counter
        text = self.to_text()
        if not text or engine is None:
            return 0
        try:
            return int(engine.count_text(text))
        except Exception as exc:  # счётчик не должен ломать вызывающий код
            logger.warning("Ошибка счётчика токенов при подсчёте фактов: %s", exc)
            return 0

    def _drop_oldest(self) -> bool:
        """Удаляет самый старый факт (по updated_at, затем по категории/ключу).

        Returns:
            True, если что-то удалено.
        """
        candidates: List[Tuple[str, int, str, str]] = []
        for category in self._category_order():
            index = self.categories.index(category) if category in self.categories else 99
            for key in self.data.get(category, {}):
                stamp = self.updated_at.get(category, {}).get(key, "")
                candidates.append((stamp, index, category, key))
        if not candidates:
            return False
        candidates.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
        _, _, category, key = candidates[0]
        self.apply({category: {key: None}}, source="limit")
        return True

    def limit(self, counter: Any = None, max_items: Optional[int] = None,
              max_tokens: Optional[int] = None) -> int:
        """Вытесняет самые старые факты при превышении лимитов.

        Args:
            counter: счётчик токенов (по умолчанию — сохранённый в объекте).
            max_items: лимит числа пар (по умолчанию — self.max_items).
            max_tokens: лимит токенов блока (по умолчанию — self.max_tokens).

        Returns:
            Сколько фактов вытеснено.
        """
        items_limit = self.max_items if max_items is None else int(max_items)
        tokens_limit = self.max_tokens if max_tokens is None else int(max_tokens)
        removed = 0

        guard = 0
        while items_limit > 0 and self.count() > items_limit and guard < 1000:
            if not self._drop_oldest():
                break
            removed += 1
            guard += 1

        engine = counter if counter is not None else self.counter
        guard = 0
        while (engine is not None and tokens_limit > 0 and self.count() > 0
               and self.token_count(engine) > tokens_limit and guard < 1000):
            if not self._drop_oldest():
                break
            removed += 1
            guard += 1

        if removed:
            logger.info(
                "Факты вытеснены по лимитам (%d шт.), осталось %d.",
                removed, self.count(),
            )
        return removed

    # ------------------------------------------------------------------
    # Локальный fallback: явные команды пользователя
    # ------------------------------------------------------------------

    @staticmethod
    def extract_local(text: str) -> Dict[str, Dict[str, str]]:
        """Разбирает явные команды запоминания из сообщения пользователя.

        Понимает: «запомни, что …», «запомни: ключ — значение», «меня зовут …»,
        «мой … — …», «я живу в …». Используется как гарантия выполнения явной
        просьбы «запомни» и как запасной путь при сбое LLM-обновления.

        Args:
            text: сообщение пользователя.

        Returns:
            Словарь фактов в формате :meth:`merge` (может быть пустым).
        """
        if not isinstance(text, str) or not text.strip():
            return {}
        found: Dict[str, Dict[str, str]] = {}

        def put(category: str, key: str, value: str) -> None:
            key = " ".join(str(key).split()).strip().lower().replace(" ", "_")
            value = " ".join(str(value).split()).strip(" .,;:—–-")
            if key and value:
                found.setdefault(category, {})[key] = value

        for match in _NAME_RE.finditer(text):
            put("preferences", "имя", match.group(1))
        for match in _CITY_RE.finditer(text):
            put("preferences", "город", match.group(1))
        for match in _MY_RE.finditer(text):
            put("preferences", match.group(1), match.group(2))
        for match in _REMEMBER_RE.finditer(text):
            rest = match.group(1).strip()
            if _POLITE_RE.match(rest):
                continue  # «запомни это, пожалуйста» — не факт
            parts = _SPLIT_RE.split(rest, maxsplit=1)
            if len(parts) == 2:
                put("decisions", parts[0], parts[1])
            else:
                put("decisions", "заметка", rest)
        return found


# ---------------------------------------------------------------------------
# Служебный запрос извлечения фактов
# ---------------------------------------------------------------------------

def build_request(
    store: FactsStore,
    user_text: str,
    assistant_text: str,
    model: str,
    prompt: Optional[str] = None,
    temperature: float = DEFAULT_FACTS_TEMPERATURE,
    max_tokens: int = DEFAULT_FACTS_REQUEST_MAX_TOKENS,
) -> Dict[str, Any]:
    """Формирует служебный запрос обновления фактов.

    В запрос попадают текущие факты (JSON) и новый обмен «пользователь →
    ассистент». Служебная инструкция содержит :data:`FACTS_MARKER`.

    Args:
        store: текущее хранилище фактов.
        user_text: сообщение пользователя за прошедший ход.
        assistant_text: ответ ассистента за прошедший ход.
        model: модель для служебного запроса.
        prompt: своя инструкция (по умолчанию — :data:`DEFAULT_FACTS_PROMPT`).
        temperature: температура служебного запроса.
        max_tokens: лимит длины ответа.

    Returns:
        Словарь параметров для chat.completions.create.
    """
    current = json.dumps(
        {cat: dict(values) for cat, values in store.data.items()},
        ensure_ascii=False, indent=2,
    )
    parts = [
        "Текущие факты (JSON):\n" + (current if store.data else "{}"),
        f"Новый обмен:\nПользователь: {user_text}\nАссистент: {assistant_text}",
        "Верни обновлённый набор фактов одним JSON-объектом.",
    ]
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": (prompt or "").strip() or DEFAULT_FACTS_PROMPT},
            {"role": "user", "content": "\n\n".join(parts)},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }


def parse_response(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Разбирает ответ модели с фактами.

    Args:
        text: сырой текст ответа.

    Returns:
        Кортеж (факты | None, ошибка | None).
    """
    if not text or not text.strip():
        return None, "Модель вернула пустой ответ с фактами."
    parsed = extract_json_object(text)
    if parsed is None:
        return None, "Не удалось разобрать JSON с фактами в ответе модели."
    return parsed, None
