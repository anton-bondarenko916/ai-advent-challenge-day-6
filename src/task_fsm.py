"""Конечный автомат задачи (FSM).

Класс :class:`TaskFSM` управляет переходами между стадиями
:class:`~src.task_models.TaskStage` по явной матрице переходов и
сохраняет каждое изменение в БД (:class:`~src.task_repository.TaskRepository`)
и в отдельный файл ``fsm.log``.

Модель управления: пользователь задаёт задачу сообщением в чате и
управляет ТОЛЬКО переходами между этапами — подтверждает план
(``approve_plan``) и итоговый результат (``approve_final``), а также
может поставить паузу, продолжить или остановить задачу. Шаги внутри
выполнения идут автоматически в обоих режимах.

Ключевой принцип: FSM **не содержит бизнес-логики LLM**. Он только
проверяет допустимость перехода, выполняет его, пишет событие в журнал
и обновляет состояние задачи в SQLite. Вся логика (сформировать план,
выполнить шаг, проверить результат) — в агенте.

Все методы повышения уровня (``approve_plan``, ``reject_final``, ``pause``…)
безопасно вызывать только на соответствующей стадии: попытка запрещённого
перехода выбрасывает :class:`~src.task_models.InvalidTransitionError` и
состояние задачи не меняется.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .task_models import (
    EXPECTED_ACTIONS,
    AWAITING_STAGES,
    InvalidTransitionError,
    Task,
    TaskMode,
    TaskStage,
    TaskStep,
)
from .task_repository import TaskRepository

# Лимит повторов одного шага: превышение — автоматический FAILED.
TASK_RETRY_LIMIT = 3

# Матрица переходов: {из} → {множество допустимых «в»}.
# Пользователь управляет только переходами между ЭТАПАМИ (подтверждение
# плана и итога); шаги внутри EXECUTION идут автоматически в обоих
# режимах. Переходы «любое → PAUSED/FAILED/CANCELLED» из активных стадий
# добавляются ниже отдельными правилами. AWAITING_STEP_APPROVAL оставлен
# в матрице только как «исходящая» стадия для старых записей в БД — в
# новой логике он недостижим.
_TRANSITIONS: Dict[TaskStage, set] = {
    TaskStage.IDLE: {
        TaskStage.PLANNING,
    },
    TaskStage.PLANNING: {
        TaskStage.AWAITING_PLAN_APPROVAL,   # план готов (manual)
        TaskStage.EXECUTION,                # план готов (auto)
    },
    TaskStage.AWAITING_PLAN_APPROVAL: {
        TaskStage.EXECUTION,                # approve_plan
        TaskStage.PLANNING,                 # reject_plan → новый план
    },
    TaskStage.EXECUTION: {
        TaskStage.VALIDATION,               # все шаги выполнены
    },
    # Легаси: старые задачи могли застрять в ожидании шага — из них
    # разрешены только глобальные переходы (пауза/стоп/ошибка), которые
    # добавляет цикл ниже. В новую стадию попасть нельзя.
    TaskStage.AWAITING_STEP_APPROVAL: set(),
    TaskStage.VALIDATION: {
        TaskStage.AWAITING_FINAL_APPROVAL,  # валидация пройдена (manual)
        TaskStage.DONE,                     # валидация пройдена (auto)
        TaskStage.EXECUTION,                # валидация не пройдена — переделать
    },
    TaskStage.AWAITING_FINAL_APPROVAL: {
        TaskStage.DONE,                     # approve_final
        TaskStage.EXECUTION,                # reject_final → переделать
    },
    TaskStage.PAUSED: {
        # resume(): возвращаемся в стадию, сохранённую при паузе
        TaskStage.PLANNING,
        TaskStage.AWAITING_PLAN_APPROVAL,
        TaskStage.EXECUTION,
        TaskStage.AWAITING_STEP_APPROVAL,   # легаси-восстановление
        TaskStage.VALIDATION,
        TaskStage.AWAITING_FINAL_APPROVAL,
    },
    TaskStage.DONE: set(),
    TaskStage.FAILED: {
        TaskStage.CANCELLED,                # «Стоп» после ошибки
    },
    TaskStage.CANCELLED: set(),
}

# Глобальные правила: из любой активной стадии можно поставить паузу,
# отменить задачу или пометить её ошибочной.
_GLOBAL_FROM_ACTIVE = {
    TaskStage.PAUSED,
    TaskStage.CANCELLED,
    TaskStage.FAILED,
}
_ACTIVE_SOURCES = (
    TaskStage.PLANNING,
    TaskStage.AWAITING_PLAN_APPROVAL,
    TaskStage.EXECUTION,
    TaskStage.AWAITING_STEP_APPROVAL,
    TaskStage.VALIDATION,
    TaskStage.AWAITING_FINAL_APPROVAL,
)
for _src in _ACTIVE_SOURCES:
    _TRANSITIONS[_src] |= _GLOBAL_FROM_ACTIVE

# Из паузы задачу можно отменить или пометить ошибочной
# (повторная пауза и «пауза из паузы» бессмысленны).
_TRANSITIONS[TaskStage.PAUSED] |= {TaskStage.CANCELLED, TaskStage.FAILED}


def _fsm_logger() -> logging.Logger:
    """Логгер FSM, пишущий все переходы в отдельный файл ``fsm.log``.

    Файл лежит в корне проекта. Обработчик добавляется один раз
    (повторные вызовы безопасны); в консоль лог не дублируется.
    """
    import logging as _logging

    logger = _logging.getLogger("deepseek-chat.task_fsm")
    if not logger.handlers:
        try:
            from .config import PROJECT_ROOT

            log_path = PROJECT_ROOT / "fsm.log"
        except Exception:  # pragma: no cover — конфиг всегда доступен
            log_path = Path("fsm.log")  # type: ignore[name-defined]
        handler = _logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(_logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(_logging.INFO)
        logger.propagate = False
    return logger


class TaskFSM:
    """Конечный автомат одной задачи.

    Attributes:
        task: объект задачи (изменяется вместе с БД).
        repo: репозиторий (все SQL — внутри него).

    Приватные атрибуты:
        _log: логгер fsm.log.
    """

    def __init__(self, task: Task, repo: TaskRepository) -> None:
        """Привязывает автомат к задаче и репозиторию.

        Args:
            task: задача (рекомендуется свежепрочитанная из БД).
            repo: репозиторий задач.
        """
        self.task = task
        self.repo = repo
        self._log = _fsm_logger()

    # ------------------------------------------------------------------
    # Базовые переходы
    # ------------------------------------------------------------------

    def can_transition(self, to_stage: TaskStage) -> bool:
        """Проверяет, разрешён ли переход по матрице переходов."""
        return to_stage in _TRANSITIONS.get(self.task.stage, set())

    def transition(self, to_stage: TaskStage, event_type: str,
                   payload: Optional[Dict[str, Any]] = None) -> None:
        """Выполняет переход, логирует событие и обновляет БД.

        Args:
            to_stage: целевая стадия.
            event_type: имя события (для журнала: start_task, approve_plan…).
            payload: дополнительные данные события (JSON).

        Raises:
            InvalidTransitionError: переход запрещён матрицей — состояние
                задачи (в памяти и в БД) не меняется.
        """
        if not self.can_transition(to_stage):
            raise InvalidTransitionError(
                f"Переход {self.task.stage.value} → {to_stage.value} "
                f"запрещён матрицей переходов (событие {event_type!r})."
            )
        from_stage = self.task.stage
        self.repo.log_event(self.task.id, from_stage, to_stage,
                            event_type, payload)
        self.task.stage = to_stage
        expected = EXPECTED_ACTIONS.get(to_stage, "")
        self.task.expected_action = expected
        self.repo.update_task_stage(self.task.id, to_stage, expected)
        self._log.info(
            "task #%s [%s]: %s -> %s (%s)%s",
            self.task.id, self.task.mode.value,
            from_stage.value, to_stage.value, event_type,
            f" payload={payload}" if payload else "",
        )

    # ------------------------------------------------------------------
    # Пауза / продолжение / отмена / ошибка
    # ------------------------------------------------------------------

    def pause(self) -> None:
        """Ставит задачу на паузу, сохраняя текущую стадию для resume()."""
        if self.task.stage == TaskStage.PAUSED:
            raise InvalidTransitionError("Задача уже на паузе.")
        if self.task.is_terminal():
            raise InvalidTransitionError(
                f"Задачу в стадии {self.task.stage.value} нельзя поставить "
                f"на паузу."
            )
        self.task.stage_before_pause = self.task.stage
        self.repo.update_task_fields(
            self.task.id, stage_before_pause=self.task.stage_before_pause
        )
        self.transition(TaskStage.PAUSED, "pause")

    def resume(self) -> None:
        """Возвращает задачу в стадию, сохранённую при паузе."""
        if self.task.stage != TaskStage.PAUSED:
            raise InvalidTransitionError(
                "Продолжить можно только задачу на паузе."
            )
        target = self.task.stage_before_pause
        if target is None:
            # Запасной путь: стадия до паузы потеряна (ручная правка БД) —
            # возвращаем задачу в выполнение с текущего шага.
            target = TaskStage.EXECUTION
        self.task.stage_before_pause = None
        self.repo.update_task_fields(self.task.id, stage_before_pause=None)
        self.transition(target, "resume",
                        payload={"resumed_to": target.value})

    def cancel(self) -> None:
        """Отменяет задачу (из любой стадии, кроме done/cancelled/idle)."""
        if self.task.stage in (TaskStage.CANCELLED, TaskStage.IDLE):
            raise InvalidTransitionError(
                f"Задачу в стадии {self.task.stage.value} нельзя отменить."
            )
        self.transition(TaskStage.CANCELLED, "cancel")
        self.repo.cancel_task(self.task.id)

    def fail(self, error: str) -> None:
        """Помечает задачу ошибочной (из любой активной стадии)."""
        if self.task.is_terminal():
            raise InvalidTransitionError(
                f"Задачу в стадии {self.task.stage.value} нельзя перевести "
                f"в failed."
            )
        self.task.error = error
        self.repo.set_task_error(self.task.id, error)
        self.transition(TaskStage.FAILED, "fail", payload={"error": error})

    # ------------------------------------------------------------------
    # Подтверждения (ручной режим)
    # ------------------------------------------------------------------

    def approve_plan(self) -> None:
        """Подтверждает план: ожидание → выполнение (шаги пойдут сами)."""
        if self.task.stage != TaskStage.AWAITING_PLAN_APPROVAL:
            raise InvalidTransitionError(
                "Подтвердить план можно только в стадии ожидания плана."
            )
        self.task.current_step_index = 0
        self.repo.update_task_fields(self.task.id, current_step_index=0)
        self.transition(TaskStage.EXECUTION, "approve_plan")

    def reject_plan(self, reason: str = "") -> None:
        """Отклоняет план: возврат в планирование для нового плана."""
        if self.task.stage != TaskStage.AWAITING_PLAN_APPROVAL:
            raise InvalidTransitionError(
                "Отклонить план можно только в стадии ожидания плана."
            )
        self._save_rejection(reason)
        # План отклонён: старые шаги больше не актуальны
        self.repo.update_task_plan(self.task.id, [])
        self.task.plan = []
        self.task.current_step_index = 0
        self.repo.update_task_fields(self.task.id, current_step_index=0)
        self.transition(TaskStage.PLANNING, "reject_plan",
                        payload={"reason": reason})

    def approve_final(self, result: Optional[str] = None) -> None:
        """Финальное подтверждение: задача завершена (DONE)."""
        if self.task.stage != TaskStage.AWAITING_FINAL_APPROVAL:
            raise InvalidTransitionError(
                "Финальное подтверждение доступно только в стадии "
                "финального ожидания."
            )
        if result:
            self.task.result = result
            self.repo.set_task_result(self.task.id, result)
        self.transition(TaskStage.DONE, "approve_final")

    def reject_final(self, reason: str = "") -> None:
        """Отклоняет итог: возврат в выполнение для переделки."""
        if self.task.stage != TaskStage.AWAITING_FINAL_APPROVAL:
            raise InvalidTransitionError(
                "Отклонить итог можно только в стадии финального ожидания."
            )
        self._save_rejection(reason)
        self.transition(TaskStage.EXECUTION, "reject_final",
                        payload={"reason": reason})

    # ------------------------------------------------------------------
    # Шаги плана
    # ------------------------------------------------------------------

    def next_step(self) -> bool:
        """Переходит к следующему шагу плана.

        В auto-режиме вызывается автоматически после выполнения шага.

        Returns:
            True — есть следующий шаг для выполнения (стадия EXECUTION,
            индекс увеличен); False — шагов больше нет и задача переведена
            в VALIDATION.

        Raises:
            InvalidTransitionError: вызван вне стадии EXECUTION.
        """
        if self.task.stage != TaskStage.EXECUTION:
            raise InvalidTransitionError(
                "next_step() доступен только в стадии выполнения."
            )
        if self.task.current_step_index + 1 < len(self.task.plan):
            self.task.current_step_index += 1
            self.repo.update_task_fields(
                self.task.id,
                current_step_index=self.task.current_step_index,
            )
            return True
        self.transition(TaskStage.VALIDATION, "steps_completed")
        return False

    def mark_step_started(self, step: TaskStep) -> None:
        """Помечает шаг выполняющимся (in_progress) в БД и в памяти."""
        step.status = "in_progress"
        if step.id is not None:
            self.repo.update_step_status(step.id, "in_progress")

    def mark_step_done(self, step: TaskStep, result: str) -> None:
        """Помечает шаг выполненным и сохраняет его результат."""
        step.status = "done"
        step.result = result
        if step.id is not None:
            self.repo.update_step_status(step.id, "done", result=result)

    # ------------------------------------------------------------------
    # Служебное
    # ------------------------------------------------------------------

    def _save_rejection(self, reason: str) -> None:
        """Сохраняет причину отклонения в объекте задачи и в БД."""
        self.task.rejection_reason = reason or None
        self.repo.update_task_fields(
            self.task.id, rejection_reason=reason or None
        )

    def set_mode(self, mode: TaskMode) -> None:
        """Меняет режим задачи на лету (auto/manual) с записью в журнал."""
        if mode == self.task.mode:
            return
        old = self.task.mode
        self.task.mode = mode
        self.repo.set_task_mode(self.task.id, mode)
        self.repo.log_event(self.task.id, self.task.stage,
                            self.task.stage, "mode_changed",
                            {"from": old.value, "to": mode.value})
        self._log.info("task #%s: mode %s -> %s",
                       self.task.id, old.value, mode.value)

    def awaiting_decision(self) -> bool:
        """True, если задача ждёт решения пользователя (ручной режим)."""
        return self.task.stage in AWAITING_STAGES

    def steps_overview(self) -> List[Dict[str, Any]]:
        """Краткое описание шагов (для контекстного блока и UI-логов)."""
        return [
            {"index": step.step_index, "description": step.description,
             "status": step.status}
            for step in self.task.plan
        ]
