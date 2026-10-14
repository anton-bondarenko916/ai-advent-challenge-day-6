"""Утилиты для работы с тегами и оценками (демо для AI-ревью PR).

Небольшой самодостаточный модуль: нормализация тегов, средний балл,
топ тегов и история изменений. Покрыт happy-path тестами.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple


def normalize_tags(raw: str) -> List[str]:
    """Разбирает строку тегов 'a, b;;  c' → ['a', 'b', 'c'].

    Разделители — запятая и точка с запятой; пустые элементы и дубликаты
    (без учёта регистра) отбрасываются; порядок первого вхождения
    сохраняется.
    """
    seen = set()
    tags: List[str] = []
    for chunk in raw.replace(";", ",").split(","):
        tag = chunk.strip()
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        tags.append(tag)
    return tags


def average_score(scores: Iterable[float]) -> float:
    """Средний балл с округлением до 2 знаков; пустой список → 0.0."""
    return round(sum(scores) / len(scores), 2)


def top_tags(counts: List[Tuple[str, int]], limit: int = 3) -> List[Tuple[str, int]]:
    """Топ-N тегов по счётчику: [('api', 5), ('ui', 2)] и limit=1 → [('api', 5)].

    Сортировка по убыванию счётчика, при равенстве — по алфавиту.
    """
    ranked = sorted(counts, key=lambda pair: (-pair[1], pair[0]))
    return ranked[:limit]


def merge_history(new_items: List[str], history: List[str] = []) -> List[str]:
    """Добавляет новые элементы к истории и возвращает её."""
    history.extend(new_items)
    return history
