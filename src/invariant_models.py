"""Доменные модели подсистемы инвариантов.

Модуль содержит dataclass-модели, которые используются слоями выше
(репозиторий, сборщик промпта, валидатор, агент, UI) и НЕ знают про SQL и UI:

- :class:`InvariantCategory` — категория инвариантов (группировка +
  приоритет: меньше = важнее);
- :class:`Invariant` — инвариант, жёсткое правило, которое всегда истинно
  для проекта; ключевое для LLM поле — ``rule``;
- :class:`InvariantViolation` — запись лога нарушений (пост-проверка,
  отказы и перегенерации).

Двухуровневая система областей действия (:attr:`Invariant.scope`):

- ``global``  — фундаментальные правила (``profile_id IS NULL``), действуют
  для всех профилей;
- ``profile`` — правила конкретного профиля (``profile_id IS NOT NULL``).

Отличие от профиля: профиль описывает, КАК отвечать (мягкие правила,
уступают запросу), инвариант — что МОЖНО делать (жёсткие правила,
непереопределяемые даже прямым запросом пользователя).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

#: Область действия: глобальные инварианты (для всех профилей)
SCOPE_GLOBAL = "global"
#: Область действия: инварианты конкретного профиля
SCOPE_PROFILE = "profile"

#: Допустимые значения scope (для валидации)
VALID_SCOPES = (SCOPE_GLOBAL, SCOPE_PROFILE)


@dataclass
class InvariantCategory:
    """Категория инвариантов (группировка и приоритизация в промпте).

    Attributes:
        id: идентификатор в таблице ``invariant_categories`` (None до
            сохранения).
        name: уникальное латинское имя ("security", "architecture").
        display_name: отображаемое имя ("Безопасность").
        description: произвольное описание категории.
        priority: приоритет в system-промпте (меньше = важнее, выше).
        is_system: True для предустановленных категорий (нельзя удалить).
    """

    id: Optional[int] = None
    name: str = ""
    display_name: str = ""
    description: str = ""
    priority: int = 100
    is_system: bool = False


@dataclass
class Invariant:
    """Инвариант — жёсткое правило, действующее для проекта.

    Attributes:
        id: идентификатор в таблице ``invariants`` (None до сохранения).
        category_id: категория (внешний ключ ``invariant_categories.id``).
        title: короткое название ("UI только на Tkinter").
        description: человекочитаемое описание (для UI, не для LLM).
        rule: формулировка правила для system-промпта — ключевое поле.
        scope: 'global' | 'profile'.
        profile_id: профиль (None для scope='global', обязателен для
            scope='profile').
    """

    id: Optional[int] = None
    category_id: int = 0
    title: str = ""
    description: str = ""
    rule: str = ""
    scope: str = SCOPE_PROFILE
    profile_id: Optional[int] = None


@dataclass
class InvariantViolation:
    """Запись лога нарушений (таблица ``invariant_violations``).

    Создаётся пост-проверкой ответа агента: при отказе по инварианту,
    при обнаруженном нарушении и перегенерации.

    Attributes:
        id: идентификатор записи (None до сохранения).
        invariant_id: какой инвариант затронут.
        profile_id: активный профиль в момент проверки (может быть None).
        user_query: запрос пользователя.
        agent_response: ответ агента (финальный на момент записи).
        regenerated: 1, если была перегенерация.
        resolved: 1, если нарушение устранено (отказ или успешная
            перегенерация); 0 — нарушение осталось после лимита попыток.
    """

    id: Optional[int] = None
    invariant_id: int = 0
    profile_id: Optional[int] = None
    user_query: str = ""
    agent_response: str = ""
    regenerated: bool = False
    resolved: bool = False
