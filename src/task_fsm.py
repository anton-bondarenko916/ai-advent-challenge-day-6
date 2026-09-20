"""Конечный автомат задачи (FSM): явная матрица и единый контроллер.

Модуль — единственная точка, через которую меняется стадия задачи
(:class:`~src.task_models.TaskStage`). Три уровня защиты (по ТЗ):

1. **Промпт** — объясняет LLM, какие маркеры использовать
   (см. ``DeepSeekAgent._task_markers_block``);
2. **Контроллер** — :class:`TaskFSM` проверяет матрицу
   :data:`ALLOWED_TRANSITIONS`, отклоняет недопустимые переходы и пишет
   в журнал и успешные, и отклонённые попытки;
3. **БД** — :class:`~src.task_repository.TaskRepository` транзакционно
   обновляет состояние.

Ключевой принцип: LLM предлагает действие, но решение о переходе
принимает код на основе матрицы. Никто, кроме TaskFSM, не меняет
``task.stage`` в БД.

Модель управления (решение пользователя из прошлой итерации, не менять):
задача задаётся сообщением в чате, пользователь управляет ТОЛЬКО
переходами между этапами — подтверждает план (``approve_plan``) и итог
(``approve_final``), а также может поставить паузу, продолжить или
остановить задачу. Шаги внутри выполнения идут автоматически в обоих
режимах; ``AWAITING_STEP_APPROVAL`` присутствует в матрице как «исходная»
стадия для старых записей в БД, но в новой логике недостижим.

Пауза/резюме обрабатываются отдельно от матрицы: ``pause`` разрешён из
любой активной стадии (он есть в матрице), а ``resume`` возвращает задачу
в стадию, сохранённую при паузе — в памяти, в колонке
``tasks.stage_before_pause`` или, после перезапуска приложения, из
журнала ``task_events``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional, Set, Tuple

from .task_models import (
    EXPECTED_ACTIONS,
    InvalidTransitionError,
    TASK_STAGE_VALUES,
    Task,
    TaskMode,
    TaskStage,
    TaskStep,
    TerminalStateError,
)
from .task_repository import TaskRepository

# Лимит повторов (откатов validation → execution): превышение — FAILED.
# Единый с конфигом TASKS_MAX_RETRIES (по умолчанию 3).
TASK_RETRY_LIMIT = 3

# ---------------------------------------------------------------------------
# Матрица переходов (по ТЗ): {из} → {множество допустимых «в»}.
# Декларативная, в одном месте. Терминальные стадии — пустые множества.
# ---------------------------------------------------------------------------

ALLOWED_TRANSITIONS: Dict[TaskStage, Set[TaskStage]] = {
    TaskStage.IDLE: {
        TaskStage.PLANNING,
        TaskStage.CANCELLED,
    },
    TaskStage.PLANNING: {
        TaskStage.AWAITING_PLAN_APPROVAL,
        TaskStage.EXECUTION,
        TaskStage.PAUSED,
        TaskStage.CANCELLED,
        TaskStage.FAILED,
    },
    TaskStage.AWAITING_PLAN_APPROVAL: {
        TaskStage.EXECUTION,
        TaskStage.PLANNING,            # отклонение плана
        TaskStage.PAUSED,
        TaskStage.CANCELLED,
    },
    TaskStage.EXECUTION: {
        # AWAITING_STEP_APPROVAL — только как исходная легаси-стадия;
        # в новую логику шаги подтверждений не проходят
        TaskStage.AWAITING_STEP_APPROVAL,
        TaskStage.VALIDATION,
        TaskStage.PAUSED,
        TaskStage.CANCELLED,
        TaskStage.FAILED,
    },
    TaskStage.AWAITING_STEP_APPROVAL: {
        TaskStage.EXECUTION,
        TaskStage.PAUSED,
        TaskStage.CANCELLED,
    },
    TaskStage.VALIDATION: {
        TaskStage.AWAITING_FINAL_APPROVAL,
        TaskStage.DONE,
        TaskStage.EXECUTION,           # откат при провале валидации
        TaskStage.PAUSED,
        TaskStage.CANCELLED,
        TaskStage.FAILED,
    },
    TaskStage.AWAITING_FINAL_APPROVAL: {
        TaskStage.DONE,
        TaskStage.EXECUTION,           # отклонение финала
        TaskStage.PAUSED,
        TaskStage.CANCELLED,
    },
    TaskStage.PAUSED: {
        # резюме обрабатывается отдельно, не через матрицу
        TaskStage.CANCELLED,
    },
    TaskStage.DONE: set(),             # терминальное
    TaskStage.FAILED: {
        TaskStage.PLANNING,            # можно перезапустить
        # «Стоп» после ошибки (решение пользователя из прошлой итерации:
        # кнопка «Стоп» активна для failed-задачи)
        TaskStage.CANCELLED,
    },
    TaskStage.CANCELLED: set(),        # терминальное
}

#: Терминальные стадии: переходы ИЗ них запрещены (пустые множества).
TERMINAL_STAGES = frozenset({
    TaskStage.DONE,
    TaskStage.CANCELLED,
})


def get_allowed_transitions(stage: TaskStage,
                            mode: TaskMode) -> Set[TaskStage]:
    """Возвращает допустимые переходы с учётом режима задачи.

    В auto-режиме убираются состояния ожидания подтверждений. В manual-режиме
    дополнительно запрещён прямой переход PLANNING → EXECUTION: план должен
    пройти через подтверждение (инвариант ТЗ «нельзя в execution без
    утверждённого плана»).

    Args:
        stage: текущая стадия.
        mode: режим задачи (auto/manual).

    Returns:
        Множество допустимых целевых стадий.
    """
    transitions = set(ALLOWED_TRANSITIONS.get(stage, set()))
    if mode == TaskMode.AUTO:
        transitions.discard(TaskStage.AWAITING_PLAN_APPROVAL)
        transitions.discard(TaskStage.AWAITING_STEP_APPROVAL)
        transitions.discard(TaskStage.AWAITING_FINAL_APPROVAL)
    elif stage == TaskStage.PLANNING:
        # Ручной режим: план должен пройти подтверждение — прямой
        # переход PLANNING → EXECUTION запрещён (инвариант ТЗ
        # «нельзя в execution без утверждённого плана»)
        transitions.discard(TaskStage.EXECUTION)
    return transitions


def _fsm_logger() -> logging.Logger:
    """Логгер FSM, пишущий все переходы в отдельный файл ``fsm.log``.

    Файл лежит в корне проекта. Обработчик добавляется один раз
    (повторные вызовы безопасны); в консоль лог не дублируется.
    """
    logger = logging.getLogger("deepseek-chat.task_fsm")
    if not logger.handlers:
        try:
            from .config import PROJECT_ROOT

            log_path = PROJECT_ROOT / "fsm.log"
        except Exception:  # pragma: no cover — конфиг всегда доступен
            from pathlib import Path as _Path
            log_path = _Path("fsm.log")
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _fsm_warnings_logger() -> logging.Logger:
    """Логгер отклонённых переходов — отдельный файл ``fsm_warnings.log``.

    Аналитика по ТЗ: много записей здесь — сигнал переработать промпт
    для LLM (модель предлагает недопустимые переходы).
    """
    logger = logging.getLogger("deepseek-chat.task_fsm_warnings")
    if not logger.handlers:
        try:
            from .config import PROJECT_ROOT

            log_path = PROJECT_ROOT / "fsm_warnings.log"
        except Exception:  # pragma: no cover — конфиг всегда доступен
            from pathlib import Path as _Path
            log_path = _Path("fsm_warnings.log")
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        logger.propagate = False
    return logger


class TaskFSM:
    """Единый контроллер переходов одной задачи.

    Только через :meth:`TaskFSM.transition_to` меняется ``task.stage``:
    контроллер проверяет матрицу, логирует попытку (успешную или
    отклонённую) в ``task_events`` и транзакционно обновляет БД.

    Attributes:
        task: объект задачи (изменяется вместе с БД).
        repo: репозиторий (все SQL — внутри него).
        max_retries: лимит откатов validation → execution (по умолчанию
            ``TASK_RETRY_LIMIT``; конфиг ``TASKS_MAX_RETRIES``).

    Потокобезопасен: все публичные методы берут ``RLock`` (переходы из
    разных потоков атомарны).
    """

    #: Лимит откатов validation → execution (по ТЗ)
    MAX_RETRIES = TASK_RETRY_LIMIT

    def __init__(self, task: Task, repo: TaskRepository,
                 max_retries: Optional[int] = None) -> None:
        """Привязывает автомат к задаче и репозиторию.

        Args:
            task: задача (рекомендуется свежепрочитанная из БД).
            repo: репозиторий задач.
            max_retries: лимит откатов валидации; None — ``MAX_RETRIES``.
        """
        self.task = task
        self.repo = repo
        try:
            self.max_retries = (int(max_retries) if max_retries is not None
                                else self.MAX_RETRIES)
        except (TypeError, ValueError):
            self.max_retries = self.MAX_RETRIES
        self._lock = threading.RLock()
        # Стадия до паузы (память текущего процесса; после перезапуска
        # восстанавливается из task_events — см. _load_stage_before_pause)
        self._stage_before_pause: Optional[TaskStage] = None
        self._log = _fsm_logger()
        self._warnings_log = _fsm_warnings_logger()

    # ------------------------------------------------------------------
    # Проверка перехода
    # ------------------------------------------------------------------

    def can_transition(self, to_stage: TaskStage) -> Tuple[bool, str]:
        """Проверяет, разрешён ли переход.

        Args:
            to_stage: целевая стадия.

        Returns:
            Кортеж ``(allowed, reason)``: разрешён ли переход и причина
            отказа (пустая строка, если переход разрешён).
        """
        if self.task.stage in TERMINAL_STAGES:
            return False, "Задача в терминальном состоянии"
        allowed = get_allowed_transitions(self.task.stage, self.task.mode)
        if to_stage not in allowed:
            return False, (
                f"Переход {self.task.stage.value} → {to_stage.value} "
                f"не разрешён матрицей"
            )
        # Лимит откатов валидации: validation → execution не бесконечен
        if (self.task.stage == TaskStage.VALIDATION
                and to_stage == TaskStage.EXECUTION
                and self.task.retry_count >= self.max_retries):
            return False, (
                f"Превышен лимит откатов ({self.max_retries})"
            )
        return True, ""

    def transition_to(
        self,
        to_stage: TaskStage,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Выполняет переход. Единственный способ изменить ``task.stage``.

        Порядок: проверка матрицы → запись попытки в ``task_events``
        (успешной или отклонённой — с ``allowed`` и ``reason`` в payload) →
        при отказе :class:`InvalidTransitionError` (из терминальной стадии —
        :class:`TerminalStateError`) → при откате валидации инкремент
        ``retry_count`` → транзакционное обновление стадии в БД.

        Args:
            to_stage: целевая стадия.
            event_type: имя события для журнала (start_task, approve_plan,
                llm_validation_ok…).
            payload: дополнительные данные события (JSON).

        Raises:
            TerminalStateError: попытка выхода из done/cancelled.
            InvalidTransitionError: переход запрещён матрицей/лимитом —
                состояние задачи (в памяти и в БД) не меняется.
        """
        with self._lock:
            from_stage = self.task.stage
            allowed, reason = self.can_transition(to_stage)

            if not allowed:
                # Логируем отклонённую попытку: аналитика по
                # rejected_transition показывает, где LLM «пролетает»
                # мимо матрицы
                self.repo.log_event(
                    self.task.id, from_stage, to_stage,
                    "rejected_transition",
                    {
                        **(payload or {}),
                        "allowed": False,
                        "reason": reason,
                        "event_type": event_type,
                    },
                )
                if from_stage in TERMINAL_STAGES:
                    self._log.warning(
                        "task #%s: отклонён переход из терминальной "
                        "стадии %s -> %s (%s)",
                        self.task.id, from_stage.value, to_stage.value,
                        event_type,
                    )
                    raise TerminalStateError(from_stage, to_stage, reason)
                self._warnings_log.warning(
                    "task #%s [%s]: REJECTED %s -> %s (%s): %s",
                    self.task.id, self.task.mode.value,
                    from_stage.value, to_stage.value, event_type, reason,
                )
                self._log.info(
                    "task #%s: отклонён переход %s -> %s (%s): %s",
                    self.task.id, from_stage.value, to_stage.value,
                    event_type, reason,
                )
                raise InvalidTransitionError(from_stage, to_stage, reason)

            self._commit_transition(
                to_stage, event_type,
                {**(payload or {}), "allowed": True, "reason": ""},
            )

    def _commit_transition(self, to_stage: TaskStage, event_type: str,
                           payload: Optional[Dict[str, Any]] = None) -> None:
        """Пишет событие в журнал и обновляет стадию в БД и в памяти.

        Внутренний метод: вызывается из :meth:`transition_to` (после
        проверки матрицы) и из :meth:`resume` (резюме обрабатывается
        отдельно от матрицы, по ТЗ). Вызывающий код уже решил, что
        переход разрешён, и подготовил payload (с ``allowed``/``reason``).

        Args:
            to_stage: целевая стадия (проверена вызывающим кодом).
            event_type: имя события для журнала.
            payload: данные события.
        """
        with self._lock:
            from_stage = self.task.stage
            # Журнал перехода: одна запись на успешный переход
            self.repo.log_event(self.task.id, from_stage, to_stage,
                                event_type, payload)
            # Откат валидации: retry_count растёт централизованно в FSM
            if (from_stage == TaskStage.VALIDATION
                    and to_stage == TaskStage.EXECUTION):
                retries = self.repo.increment_retry_count(self.task.id)
                self.task.retry_count = max(0, retries)
            # Перезапуск после ошибки: счётчик откатов начинается заново
            if (from_stage == TaskStage.FAILED
                    and to_stage == TaskStage.PLANNING):
                self.repo.update_task_fields(self.task.id, retry_count=0)
                self.task.retry_count = 0

            self.task.stage = to_stage
            expected = self._expected_action_for(to_stage)
            self.task.expected_action = expected
            self.repo.update_task_stage(self.task.id, to_stage, expected)
            self._log.info(
                "task #%s [%s]: %s -> %s (%s)%s",
                self.task.id, self.task.mode.value,
                from_stage.value, to_stage.value, event_type,
                f" payload={payload}" if payload else "",
            )

    @staticmethod
    def _expected_action_for(stage: TaskStage) -> str:
        """Возвращает текст для UI: что ожидается в этом состоянии."""
        return EXPECTED_ACTIONS.get(stage, "")

    # Совместимость со старым кодом: transition() = transition_to()
    def transition(self, to_stage: TaskStage, event_type: str,
                   payload: Optional[Dict[str, Any]] = None) -> None:
        """Псевдоним :meth:`transition_to` (совместимость со старым кодом)."""
        self.transition_to(to_stage, event_type, payload)

    # ------------------------------------------------------------------
    # Пауза / продолжение / отмена / ошибка
    # ------------------------------------------------------------------

    def pause(self) -> None:
        """Ставит задачу на паузу, сохраняя текущую стадию для resume().

        Стадия до паузы хранится в трёх местах: в памяти контроллера,
        в колонке ``tasks.stage_before_pause`` и в payload события паузы
        в ``task_events`` — восстановление работает и после перезапуска.
        """
        with self._lock:
            if self.task.stage == TaskStage.PAUSED:
                raise InvalidTransitionError(
                    self.task.stage, TaskStage.PAUSED, "Задача уже на паузе"
                )
            if self.task.stage in TERMINAL_STAGES or \
                    self.task.stage == TaskStage.IDLE:
                raise InvalidTransitionError(
                    self.task.stage, TaskStage.PAUSED,
                    f"Задачу в стадии {self.task.stage.value} нельзя "
                    f"поставить на паузу",
                )
            self._stage_before_pause = self.task.stage
            self.task.stage_before_pause = self.task.stage
            self.repo.update_task_fields(
                self.task.id, stage_before_pause=self.task.stage_before_pause
            )
            self.transition_to(TaskStage.PAUSED, "pause", payload={
                "stage_before_pause": self.task.stage_before_pause.value,
            })

    def resume(self) -> TaskStage:
        """Возвращает задачу в стадию, сохранённую при паузе.

        Резюме идёт мимо матрицы (по ТЗ): целевая стадия — та, из которой
        задачу поставили на паузу. Источник истины: память контроллера →
        колонка ``tasks.stage_before_pause`` → журнал ``task_events``
        (перезапуск приложения).

        Returns:
            Стадия, в которую вернулась задача.

        Raises:
            InvalidTransitionError: задача не на паузе.
        """
        with self._lock:
            if self.task.stage != TaskStage.PAUSED:
                raise InvalidTransitionError(
                    self.task.stage, "resume", "Задача не на паузе"
                )
            target = (self._stage_before_pause
                      or self.task.stage_before_pause)
            if target is None:
                # Восстановление из журнала (после перезапуска приложения)
                target = self._load_stage_before_pause()
            self._stage_before_pause = None
            self.task.stage_before_pause = None
            self.repo.update_task_fields(self.task.id, stage_before_pause=None)
            self._commit_transition(
                target, "resume",
                {"resumed_to": target.value, "allowed": True, "reason": ""},
            )
            return target

    def _load_stage_before_pause(self) -> TaskStage:
        """Восстанавливает стадию до паузы из журнала ``task_events``.

        Ищет последнее событие паузы (to_stage = paused) и берёт
        ``stage_before_pause`` из его payload. Запасной вариант — EXECUTION:
        продолжить работу с текущего шага безопаснее, чем планировать заново.
        """
        events = self.repo.get_task_events(self.task.id)
        for event in reversed(events):
            if (event.get("to_stage") == TaskStage.PAUSED.value
                    and isinstance(event.get("payload"), dict)
                    and event["payload"].get("stage_before_pause")):
                raw = str(event["payload"]["stage_before_pause"])
                if raw in TASK_STAGE_VALUES:
                    return TaskStage(raw)
        return TaskStage.EXECUTION

    def cancel(self) -> None:
        """Отменяет задачу (переход через матрицу: CANCELLED почти отовсюду)."""
        with self._lock:
            if self.task.stage in (TaskStage.CANCELLED, TaskStage.IDLE):
                raise InvalidTransitionError(
                    self.task.stage, TaskStage.CANCELLED,
                    f"Задачу в стадии {self.task.stage.value} нельзя отменить",
                )
            self.transition_to(TaskStage.CANCELLED, "cancel")
            self.repo.cancel_task(self.task.id)

    def fail(self, error: str) -> None:
        """Помечает задачу ошибочной (из любой активной стадии)."""
        with self._lock:
            if self.task.stage in TERMINAL_STAGES:
                raise TerminalStateError(
                    self.task.stage, TaskStage.FAILED,
                    f"Задачу в стадии {self.task.stage.value} нельзя "
                    f"перевести в failed",
                )
            self.task.error = error
            self.repo.set_task_error(self.task.id, error)
            self.transition_to(TaskStage.FAILED, "fail",
                               payload={"error": error})

    # ------------------------------------------------------------------
    # Подтверждения (ручной режим)
    # ------------------------------------------------------------------

    def approve_plan(self) -> None:
        """Подтверждает план: ожидание → выполнение (шаги пойдут сами)."""
        with self._lock:
            if self.task.stage != TaskStage.AWAITING_PLAN_APPROVAL:
                raise InvalidTransitionError(
                    self.task.stage, TaskStage.EXECUTION,
                    "Подтвердить план можно только в стадии ожидания плана",
                )
            self.task.current_step_index = 0
            self.repo.update_task_fields(self.task.id, current_step_index=0)
            self.transition_to(TaskStage.EXECUTION, "approve_plan")

    def reject_plan(self, reason: str = "") -> None:
        """Отклоняет план: возврат в планирование для нового плана."""
        with self._lock:
            if self.task.stage != TaskStage.AWAITING_PLAN_APPROVAL:
                raise InvalidTransitionError(
                    self.task.stage, TaskStage.PLANNING,
                    "Отклонить план можно только в стадии ожидания плана",
                )
            self._save_rejection(reason)
            # План отклонён: старые шаги больше не актуальны
            self.repo.update_task_plan(self.task.id, [])
            self.task.plan = []
            self.task.current_step_index = 0
            self.repo.update_task_fields(self.task.id, current_step_index=0)
            self.transition_to(TaskStage.PLANNING, "reject_plan",
                               payload={"reason": reason})

    def approve_final(self, result: Optional[str] = None) -> None:
        """Финальное подтверждение: задача завершена (DONE)."""
        with self._lock:
            if self.task.stage != TaskStage.AWAITING_FINAL_APPROVAL:
                raise InvalidTransitionError(
                    self.task.stage, TaskStage.DONE,
                    "Финальное подтверждение доступно только в стадии "
                    "финального ожидания",
                )
            if result:
                self.task.result = result
                self.repo.set_task_result(self.task.id, result)
            self.transition_to(TaskStage.DONE, "approve_final")

    def reject_final(self, reason: str = "") -> None:
        """Отклоняет итог: возврат в выполнение для переделки."""
        with self._lock:
            if self.task.stage != TaskStage.AWAITING_FINAL_APPROVAL:
                raise InvalidTransitionError(
                    self.task.stage, TaskStage.EXECUTION,
                    "Отклонить итог можно только в стадии финального "
                    "ожидания",
                )
            self._save_rejection(reason)
            self.transition_to(TaskStage.EXECUTION, "reject_final",
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
        with self._lock:
            if self.task.stage != TaskStage.EXECUTION:
                raise InvalidTransitionError(
                    self.task.stage, TaskStage.EXECUTION,
                    "next_step() доступен только в стадии выполнения",
                )
            if self.task.current_step_index + 1 < len(self.task.plan):
                self.task.current_step_index += 1
                self.repo.update_task_fields(
                    self.task.id,
                    current_step_index=self.task.current_step_index,
                )
                return True
            self.transition_to(TaskStage.VALIDATION, "steps_completed")
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
        return self.task.stage in (
            TaskStage.AWAITING_PLAN_APPROVAL,
            TaskStage.AWAITING_STEP_APPROVAL,
            TaskStage.AWAITING_FINAL_APPROVAL,
        )

    def steps_overview(self) -> list:
        """Краткое описание шагов (для контекстного блока и UI-логов)."""
        return [
            {"index": step.step_index, "description": step.description,
             "status": step.status}
            for step in self.task.plan
        ]
