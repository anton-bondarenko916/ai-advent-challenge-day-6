"""Парсер ответов LLM для подсистемы задач (маркеры + эвристика).

Класс :class:`TaskResponseParser` извлекает из ответа модели **намерение
перехода** (intent) по гибридной схеме:

1. явный маркер (``[PLAN_READY]``, ``[STEP_DONE]``…) — приоритет;
2. эвристика — поиск устойчивых формулировок («план готов», «все шаги
   выполнены»…);
3. fail-safe — ``None``: никуда не переходим, агент логирует
   ``unresolved_transition`` и ждёт следующего запроса/уточнения.

Ключевой принцип подсистемы: парсер только ПРЕДЛАГАЕТ намерение —
решение о переходе принимает :class:`~src.task_fsm.TaskFSM` на основе
матрицы :data:`~src.task_fsm.ALLOWED_TRANSITIONS`. LLM не может изменить
состояние задачи напрямую.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

#: Маркеры состояния задачи (ставятся LLM в конце ответа, по одному)
MARKERS: Dict[str, str] = {
    "[PLAN_READY]": "plan_ready",
    "[STEP_DONE]": "step_done",
    "[ALL_STEPS_DONE]": "all_steps_done",
    "[VALIDATION_OK]": "validation_ok",
    "[VALIDATION_FAIL]": "validation_fail",
    "[NEED_INPUT]": "need_input",
}

#: Эвристики (fallback, когда маркера нет): intent → регулярные выражения
HEURISTICS: Dict[str, List[str]] = {
    "plan_ready": [
        r"план\s+готов",
        r"вот\s+план",
        r"предлагаю\s+следующие\s+шаги",
    ],
    "step_done": [
        r"шаг\s+выполнен",
        r"готово\s+по\s+шагу",
        r"сделано",
    ],
    "all_steps_done": [
        r"все\s+шаги\s+выполнены",
        r"задача\s+выполнена",
        r"можно\s+переходить\s+к\s+проверке",
    ],
    "validation_ok": [
        r"валидация\s+пройдена",
        r"результат\s+корректен",
        r"всё\s+верно",
    ],
    "validation_fail": [
        r"валидация\s+не\s+пройдена",
        r"нужно\s+переделать",
        r"результат\s+не\s+соответствует",
    ],
}

#: Компилированные эвристики (готовятся один раз при импорте)
_COMPILED_HEURISTICS: Dict[str, List[re.Pattern]] = {
    intent: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for intent, patterns in HEURISTICS.items()
}


class TaskResponseParser:
    """Извлекает из ответа LLM намерение перехода задачи.

    Приоритет: явный маркер → эвристика → None (fail-safe).
    """

    MARKERS = MARKERS
    HEURISTICS = HEURISTICS

    def parse(self, response_text: Optional[str]) -> Optional[str]:
        """Возвращает intent (строку) или None, если не удалось определить.

        Args:
            response_text: текст ответа модели (может быть None/пустым).

        Returns:
            Имя намерения (ключ из ``MARKERS.values()`` / ``HEURISTICS``)
            или None — сигнал «маркера нет» (переход не выполняется).
        """
        text = str(response_text or "")
        if not text.strip():
            return None

        # 1. Явный маркер (приоритет)
        for marker, intent in self.MARKERS.items():
            if marker in text:
                return intent

        # 2. Эвристика (fallback)
        for intent, patterns in _COMPILED_HEURISTICS.items():
            for pattern in patterns:
                if pattern.search(text):
                    return intent

        # 3. Fail-safe
        return None

    def find_marker(self, response_text: Optional[str]) -> Optional[str]:
        """Возвращает первый найденный маркер в тексте (None — маркеров нет).

        Используется для удаления маркеров из текста, показываемого
        пользователю, и для диагностики.
        """
        text = str(response_text or "")
        for marker in self.MARKERS:
            if marker in text:
                return marker
        return None

    @classmethod
    def strip_markers(cls, response_text: Optional[str]) -> str:
        """Убирает служебные маркеры из текста ответа (для показа в чате)."""
        text = str(response_text or "")
        for marker in cls.MARKERS:
            text = text.replace(marker, "")
        return text.rstrip()
