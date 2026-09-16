"""Доменные модели профиля пользователя (третий слой персонализации).

Модуль содержит dataclass-модели, которые используются слоями выше
(репозиторий, сборщик промпта, агент, UI) и НЕ знают про SQL и UI:

- :class:`UserProfile` — профиль пользователя (стиль, формат, язык,
  ограничения, контекст);
- :class:`ProfileAgentSettings` — настройки агента, привязанные к профилю
  (модель, temperature, max_tokens, базовый системный промпт).

Сложные поля сериализуются в SQLite как JSON:

- ``UserProfile.constraints`` — JSON-массив строк;
- ``UserProfile.context``     — JSON-объект ``{ключ: значение}``.

Здесь же живут вспомогательные функции разбора этих полей
(:func:`parse_constraints`, :func:`parse_context`, :func:`parse_context_lines`),
используемые и репозиторием (чтение из БД), и UI (парсинг многострочного
текста формы «каждая строка = ограничение» / «ключ: значение»).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Модель агента по умолчанию для новых профилей (см. utils.AVAILABLE_MODELS)
DEFAULT_PROFILE_MODEL = "deepseek-v4-flash"
#: Базовый промпт по умолчанию для новых профилей
DEFAULT_PROFILE_PROMPT = "Ты полезный AI-ассистент."


@dataclass
class UserProfile:
    """Профиль пользователя — правила персонализации ответов агента.

    Attributes:
        id: идентификатор в таблице ``profiles`` (None до сохранения).
        name: уникальное имя профиля ("Senior Dev", "Default").
        display_name: отображаемое имя (например, имя пользователя).
        style: стиль ответов ("краткий, технический, без воды").
        format: формат ответов ("markdown, код в блоках").
        language: язык ответов.
        constraints: список жёстких ограничений (нельзя нарушать никогда).
        context: произвольный контекст пользователя {"role": ..., "stack": [...]}.
        is_active: True для активного профиля (заполняется репозиторием).
    """

    id: Optional[int] = None
    name: str = ""
    display_name: str = ""
    style: str = ""
    format: str = ""
    language: str = "русский"
    constraints: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = False


@dataclass
class ProfileAgentSettings:
    """Настройки агента, привязанные к конкретному профилю.

    Хранятся в таблице ``profile_agent_settings`` (1:1 с профилем) и
    применяются к :class:`~src.agent.DeepSeekAgent` при активации профиля.

    Attributes:
        profile_id: идентификатор профиля (внешний ключ).
        model: имя модели (deepseek-v4-flash | deepseek-v4-pro).
        temperature: температура сэмплирования (0.0–2.0).
        max_tokens: лимит токенов ответа.
        base_system_prompt: базовая инструкция промпта (фон профиля).
    """

    profile_id: int = 0
    model: str = DEFAULT_PROFILE_MODEL
    temperature: float = 0.7
    max_tokens: int = 1000
    base_system_prompt: str = DEFAULT_PROFILE_PROMPT


# ---------------------------------------------------------------------------
# Сериализация сложных полей профиля (JSON <-> Python)
# ---------------------------------------------------------------------------

def serialize_constraints(constraints: List[str]) -> str:
    """Сериализует список ограничений в JSON-строку для БД."""
    return json.dumps(
        [str(item) for item in (constraints or []) if str(item).strip()],
        ensure_ascii=False,
    )


def serialize_context(context: Dict[str, Any]) -> str:
    """Сериализует словарь контекста в JSON-строку для БД."""
    return json.dumps(context or {}, ensure_ascii=False)


def parse_constraints(raw: Any) -> List[str]:
    """Разбирает поле ``constraints`` из БД в список строк (устойчиво к мусору).

    Args:
        raw: JSON-строка, список или None.

    Returns:
        Список непустых строк; None/битый JSON → пустой список.
    """
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        items = raw
    else:
        try:
            items = json.loads(str(raw))
        except (ValueError, TypeError):
            return []
    if not isinstance(items, list):
        return []
    return [str(item) for item in items if str(item).strip()]


def parse_context(raw: Any) -> Dict[str, Any]:
    """Разбирает поле ``context`` из БД в словарь (устойчиво к мусору)."""
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(str(raw))
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def parse_context_lines(text: str) -> Tuple[List[str], Dict[str, Any]]:
    """Разбирает многострочный текст формы профиля.

    Каждая непустая строка трактуется как:

    - ``ключ: значение`` — пара контекста (значение пытается приводиться
      к JSON-типу: числа, списки ``[a, b]``, true/false);
    - любая другая строка — ограничение (constraints).

    Args:
        text: многострочный текст (одно поле формы «constraints или context»).

    Returns:
        Кортеж ``(constraints, context)``.
    """
    constraints: List[str] = []
    context: Dict[str, Any] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        key, sep, value = line.partition(":")
        if sep and key.strip():
            context[key.strip()] = _coerce_value(value.strip())
        else:
            constraints.append(line)
    return constraints, context


def _coerce_value(raw: str) -> Any:
    """Пытается привести строковое значение контекста к JSON-типу."""
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw
