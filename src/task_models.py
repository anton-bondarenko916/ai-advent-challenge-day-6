"""Доменные модели подсистемы задач (FSM).

Модуль содержит только данные и перечисления — без SQL и без логики LLM:

- :class:`TaskStage` — состояния конечного автомата задачи;
- :class:`TaskMode` — режимы работы (автоматический / ручной);
- :class:`TaskStep` — один шаг плана задачи;
- :class:`Task` — задача целиком (состояние FSM + план + результат);
- :class:`InvalidTransitionError` / :class:`ActiveTaskExistsError` —
  исключения домена.

Все остальные модули подсистемы задач (репозиторий, FSM, агент, UI)
работают исключительно с этими объектами.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class TaskStage(str, Enum):
    """Состояния конечного автомата задачи."""

    IDLE = "idle"                                      # задача не создана
    PLANNING = "planning"                              # LLM формирует план
    AWAITING_PLAN_APPROVAL = "awaiting_plan_approval"  # ждём подтверждения плана (manual)
    EXECUTION = "execution"                            # выполняем шаги
    AWAITING_STEP_APPROVAL = "awaiting_step_approval"  # ждём подтверждения шага (manual)
    VALIDATION = "validation"                          # проверка результата
    AWAITING_FINAL_APPROVAL = "awaiting_final_approval"  # ждём финального подтверждения (manual)
    PAUSED = "paused"                                  # пауза пользователем
    DONE = "done"                                      # успешно завершена
    FAILED = "failed"                                  # ошибка
    CANCELLED = "cancelled"                            # отменена пользователем


class TaskMode(str, Enum):
    """Режим работы задачи."""

    AUTO = "auto"       # переходы выполняются автоматически
    MANUAL = "manual"   # каждый переход требует подтверждения пользователя


# Стадии, в которых задача считается активной (требует внимания/работы).
# Задачи в остальных стадиях восстанавливать при запуске не нужно.
ACTIVE_STAGES = frozenset({
    TaskStage.PLANNING,
    TaskStage.AWAITING_PLAN_APPROVAL,
    TaskStage.EXECUTION,
    TaskStage.AWAITING_STEP_APPROVAL,
    TaskStage.VALIDATION,
    TaskStage.AWAITING_FINAL_APPROVAL,
    TaskStage.PAUSED,
})

# Стадии, в которых агент реально работает (LLM-вызовы возможны).
WORKING_STAGES = frozenset({
    TaskStage.PLANNING,
    TaskStage.EXECUTION,
    TaskStage.VALIDATION,
})

# Стадии ожидания подтверждения пользователя (ручной режим).
# Пользователь управляет ТОЛЬКО переходами между этапами: подтверждает
# план и итоговый результат. Шаги внутри выполнения идут автоматически
# в обоих режимах; AWAITING_STEP_APPROVAL оставлен в enum только для
# совместимости со старыми записями в БД (в новой логике недостижим).
AWAITING_STAGES = frozenset({
    TaskStage.AWAITING_PLAN_APPROVAL,
    TaskStage.AWAITING_FINAL_APPROVAL,
})

# Человекочитаемые названия стадий (для UI и статус-бара).
STAGE_TITLES = {
    TaskStage.IDLE: "Нет задачи",
    TaskStage.PLANNING: "Планирование",
    TaskStage.AWAITING_PLAN_APPROVAL: "Ожидание плана",
    TaskStage.EXECUTION: "Выполнение",
    TaskStage.AWAITING_STEP_APPROVAL: "Ожидание шага",
    TaskStage.VALIDATION: "Проверка",
    TaskStage.AWAITING_FINAL_APPROVAL: "Финальное ожидание",
    TaskStage.PAUSED: "Пауза",
    TaskStage.DONE: "Завершена",
    TaskStage.FAILED: "Ошибка",
    TaskStage.CANCELLED: "Отменена",
}

# Человекочитаемые названия режимов.
MODE_TITLES = {
    TaskMode.AUTO: "Авто",
    TaskMode.MANUAL: "Ручной",
}

# Что ожидается от пользователя/агента на каждой стадии (expected_action).
EXPECTED_ACTIONS = {
    TaskStage.PLANNING: "формирование плана",
    TaskStage.AWAITING_PLAN_APPROVAL: "подтверждение плана",
    TaskStage.EXECUTION: "выполнение шага",
    TaskStage.AWAITING_STEP_APPROVAL: "подтверждение шага",
    TaskStage.VALIDATION: "проверка результата",
    TaskStage.AWAITING_FINAL_APPROVAL: "финальное подтверждение",
    TaskStage.PAUSED: "продолжение задачи",
    TaskStage.DONE: "—",
    TaskStage.FAILED: "—",
    TaskStage.CANCELLED: "—",
    TaskStage.IDLE: "—",
}

# Возможные статусы шага плана.
STEP_STATUSES = ("pending", "in_progress", "done", "rejected", "failed")


class InvalidTransitionError(Exception):
    """Попытка запрещённого матрицей перехода FSM."""


class ActiveTaskExistsError(Exception):
    """У профиля уже есть активная задача (правило «одна задача на профиль»)."""


@dataclass
class TaskStep:
    """Один шаг плана задачи.

    Attributes:
        id: идентификатор в БД (None — шаг ещё не сохранён).
        task_id: идентификатор задачи-владельца.
        step_index: порядковый номер шага (с нуля).
        description: описание шага (что нужно сделать).
        status: pending | in_progress | done | rejected | failed.
        result: результат выполнения шага (текст).
        rejection_reason: причина отклонения шага пользователем.
    """

    id: Optional[int] = None
    task_id: Optional[int] = None
    step_index: int = 0
    description: str = ""
    status: str = "pending"
    result: Optional[str] = None
    rejection_reason: Optional[str] = None


@dataclass
class Task:
    """Задача: состояние FSM + план из шагов + результат.

    Attributes:
        id: идентификатор в БД (None — задача не сохранена).
        profile_id: профиль-владелец задачи.
        title: краткое название (показывается в UI).
        description: полное описание задачи.
        mode: режим работы (auto/manual); меняется на лету.
        stage: текущее состояние FSM.
        current_step_index: индекс текущего шага плана (с нуля).
        expected_action: что ожидается на текущей стадии.
        rejection_reason: причина последнего отклонения (план/шаг/итог).
        retry_count: число повторов текущего шага; > 3 — задача FAILED.
        stage_before_pause: стадия, сохранённая при паузе (для resume).
        plan: шаги плана (TaskStep).
        result: итоговый результат задачи (заполняется при DONE).
        error: текст последней ошибки (заполняется при FAILED).
    """

    id: Optional[int] = None
    profile_id: int = 0
    title: str = ""
    description: str = ""
    mode: TaskMode = TaskMode.MANUAL
    stage: TaskStage = TaskStage.IDLE
    current_step_index: int = 0
    expected_action: str = ""
    rejection_reason: Optional[str] = None
    retry_count: int = 0
    stage_before_pause: Optional[TaskStage] = None
    plan: List[TaskStep] = field(default_factory=list)
    result: Optional[str] = None
    error: Optional[str] = None

    # ------------------------------------------------------------------
    # Производные значения (удобно для UI и промптов)
    # ------------------------------------------------------------------

    def current_step(self) -> Optional[TaskStep]:
        """Текущий шаг плана (None — плана нет или индекс вне диапазона)."""
        if 0 <= self.current_step_index < len(self.plan):
            return self.plan[self.current_step_index]
        return None

    def done_steps_count(self) -> int:
        """Число выполненных шагов (для прогресс-бара)."""
        return sum(1 for step in self.plan if step.status == "done")

    def is_terminal(self) -> bool:
        """True, если задача в конечном состоянии (done/failed/cancelled)."""
        return self.stage in (
            TaskStage.DONE, TaskStage.FAILED, TaskStage.CANCELLED,
        )
