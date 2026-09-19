"""Тесты подсистемы задач (FSM).

Модель управления: задача задаётся сообщением чата (``/задача …``),
пользователь управляет ТОЛЬКО переходами между этапами — подтверждает
план и итог; шаги внутри выполнения идут автоматически в обоих режимах.

Проверяются:

1. Пауза на этапе planning → продолжение: задача не теряется, план
   досчитывается;
2. Пауза на этапе execution (шаг 2 из 3) → продолжение: задача
   продолжается с текущего шага, а не с начала;
3. Пауза → закрытие приложения → запуск → восстановление: задача
   восстанавливается в состоянии paused;
4. Ручной режим: план → отклонить → вернуться в planning → новый план;
5. Ручной режим: после подтверждения плана все шаги выполняются
   автоматически, задача останавливается только на переходах между
   этапами (план → итог); превышение лимита повторов валидации → FAILED;
6. Авто-режим: задача проходит planning → execution → validation → done
   без вмешательства;
7. Смена режима на лету: в середине execution переключить на manual →
   финальное подтверждение потребуется;
8. Продолжение без повторных объяснений: контекст задачи подтягивается
   из БД (блок [Состояние задачи] содержит историю шагов);
9. Одна активная задача на профиль: вторая не создаётся, пока активна
   первая;
10. Каскадное удаление: удаление задачи удаляет steps и events.

Дополнительно: матрица переходов FSM (запрещённые переходы), журнал
событий в БД и fsm.log, команда ``/задача`` в чате, состояния кнопок
панели задачи в UI.

Все тесты идут в mock-режиме (без реального API). Запуск:
    python -m unittest tests.test_task_fsm -v
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

from src.agent import DeepSeekAgent
from src.task_fsm import TASK_RETRY_LIMIT, TaskFSM
from src.task_models import (
    ActiveTaskExistsError,
    InvalidTransitionError,
    TaskMode,
    TaskStage,
)
from src.task_repository import TaskRepository

_TESTS_TMP = Path(__file__).resolve().parent / "tmp_tasks"


def _make_repo(name: str) -> TaskRepository:
    """Репозиторий задач на временной БД."""
    return TaskRepository(_TESTS_TMP / f"{name}_{uuid.uuid4().hex}.db")


def _make_agent(repo: TaskRepository) -> DeepSeekAgent:
    """Mock-агент с подсистемой задач на той же БД."""
    return DeepSeekAgent(api_key="", mock_mode=True,
                         task_repository=repo)


class TaskFSMTestBase(unittest.TestCase):
    """Общая база: временный каталог, репозиторий и агент."""

    def setUp(self) -> None:
        _TESTS_TMP.mkdir(parents=True, exist_ok=True)
        self.repo = _make_repo(type(self).__name__)
        self.agent = _make_agent(self.repo)

    def tearDown(self) -> None:
        shutil.rmtree(_TESTS_TMP, ignore_errors=True)


# ---------------------------------------------------------------------------
# Сценарий 1: пауза на planning → продолжение
# ---------------------------------------------------------------------------

class PauseOnPlanningTest(TaskFSMTestBase):
    """Пауза/продолжение не теряют задачу: план досчитывается."""

    def test_pause_on_planning_resume_completes(self) -> None:
        # FSM-уровень: из PLANNING можно встать на паузу и вернуться
        task = self.repo.create_task(0, "План с паузой", TaskMode.AUTO)
        fsm = TaskFSM(task, self.repo)
        fsm.transition(TaskStage.PLANNING, "start_task")
        fsm.pause()
        self.assertEqual(task.stage, TaskStage.PAUSED)
        fsm.resume()
        self.assertEqual(task.stage, TaskStage.PLANNING)
        # Стадия до паузы сохраняется в БД и очищается при resume
        fresh = self.repo.get_task_by_id(task.id)
        self.assertIsNone(fresh.stage_before_pause)
        # Убираем служебную задачу, чтобы не блокировать следующие тесты
        self.repo.delete_task(task.id)

        # Агент-уровень: пауза сразу после старта → resume → задача
        # доводится до DONE без повторного старта
        result = self.agent.start_task("Дописать README", TaskMode.AUTO)
        self.agent.pause_task()
        result = self.agent.resume_task()
        result = self.agent.run_task()
        self.assertEqual(result["task"].stage, TaskStage.DONE)
        self.assertEqual(len(result["task"].plan), 3)  # план целый


# ---------------------------------------------------------------------------
# Сценарий 2: пауза на execution → продолжение с текущего шага
# ---------------------------------------------------------------------------

class PauseOnExecutionTest(TaskFSMTestBase):
    """Продолжение после паузы идёт с текущего шага, не с начала."""

    def test_resume_continues_from_current_step(self) -> None:
        result = self.agent.start_task("Написать парсер логов",
                                       TaskMode.AUTO)
        # После старта план готов, но шаги ещё не выполнялись
        task = result["task"]
        self.assertEqual(task.stage, TaskStage.EXECUTION)
        self.assertEqual(task.current_step_index, 0)

        # Авто-режим: один вызов continue — один выполненный шаг
        result = self.agent.continue_task()
        task = result["task"]
        self.assertEqual(task.stage, TaskStage.EXECUTION)
        self.assertEqual(task.current_step_index, 1)  # сейчас «шаг 2 из 3»
        first_step_result = task.plan[0].result
        self.assertTrue(first_step_result)

        # Пауза → продолжение: индекс не сбрасывается, работа идёт
        # с текущего (второго) шага, а не с начала
        self.agent.pause_task()
        result = self.agent.resume_task()
        task = result["task"]
        self.assertEqual(task.stage, TaskStage.EXECUTION)
        self.assertEqual(task.current_step_index, 2)
        self.assertEqual(task.plan[0].result, first_step_result)  # история цела
        self.assertTrue(task.plan[1].result)  # второй шаг выполнен после паузы

        # Дорабатываем до конца
        result = self.agent.run_task()
        self.assertEqual(result["task"].stage, TaskStage.DONE)
        self.assertEqual(result["task"].done_steps_count(), 3)


# ---------------------------------------------------------------------------
# Сценарий 3: пауза → перезапуск → восстановление в paused
# ---------------------------------------------------------------------------

class RestartRecoveryTest(TaskFSMTestBase):
    """После перезапуска приложения задача восстанавливается из БД."""

    def test_paused_task_survives_restart(self) -> None:
        self.agent.start_task("Переезд на новый сервер", TaskMode.AUTO)
        self.agent.pause_task()

        # «Перезапуск»: новый агент над той же БД задач
        agent2 = _make_agent(self.repo)
        result = agent2.restore_task()
        self.assertIsNotNone(result["task"])
        self.assertEqual(result["task"].stage, TaskStage.PAUSED)
        self.assertFalse(result["interrupted"])

        # После восстановления задача дорабатывается до конца
        result = agent2.resume_task()
        result = agent2.run_task()
        self.assertEqual(result["task"].stage, TaskStage.DONE)

    def test_interrupted_execution_waits_for_user(self) -> None:
        # Прерванная в рабочей стадии задача НЕ запускается сама
        self.agent.start_task("Проверить бэкапы", TaskMode.AUTO)
        self.agent.continue_task()  # шаг 1 выполнен, но паузы не было
        self.agent.pause_task()     # закрываем «на паузе» — как при краше

        agent2 = _make_agent(self.repo)
        result = agent2.restore_task()
        self.assertEqual(result["task"].stage, TaskStage.PAUSED)
        # Задача не выполнила ни одного нового шага сама
        self.assertEqual(result["task"].current_step_index, 1)


# ---------------------------------------------------------------------------
# Сценарии 4–5: ручной режим — управление только переходами между этапами
# ---------------------------------------------------------------------------

class RejectPlanTest(TaskFSMTestBase):
    """Отклонение плана возвращает задачу в планирование."""

    def test_reject_plan_generates_new_plan(self) -> None:
        result = self.agent.start_task("Собрать отчёт", TaskMode.MANUAL)
        task = result["task"]
        self.assertEqual(task.stage, TaskStage.AWAITING_PLAN_APPROVAL)

        result = self.agent.reject_plan("Слишком крупные шаги")
        task = result["task"]
        # Вернулись в ожидание нового плана (planning → новый план)
        self.assertEqual(task.stage, TaskStage.AWAITING_PLAN_APPROVAL)
        self.assertEqual(task.rejection_reason, "Слишком крупные шаги")
        self.assertEqual(len(task.plan), 3)
        # Причина отклонения записана в журнал событий
        events = self.agent.get_task_events()
        self.assertTrue(any(e["event_type"] == "reject_plan" for e in events))


class ManualModeStagesTest(TaskFSMTestBase):
    """Ручной режим: шаги идут сами, остановки только на этапах."""

    def test_steps_run_automatically_between_stage_gates(self) -> None:
        result = self.agent.start_task("Настроить CI", TaskMode.MANUAL)
        self.assertEqual(result["task"].stage,
                         TaskStage.AWAITING_PLAN_APPROVAL)

        # Единственная точка контроля внутри выполнения — подтверждение
        # плана; дальше все шаги и проверка проходят без пользователя
        result = self.agent.approve_plan()
        result = self.agent.run_task()
        task = result["task"]
        self.assertEqual(task.stage, TaskStage.AWAITING_FINAL_APPROVAL)
        self.assertEqual(task.done_steps_count(), len(task.plan))
        # Подтверждений шагов не было и быть не могло: стадия ожидания
        # шага в новой логике недостижима
        event_types = [e["event_type"] for e in self.agent.get_task_events()]
        self.assertNotIn("step_done", event_types)
        self.assertNotIn("approve_step", event_types)

        # Отклонение итога возвращает задачу на переделку: агент успевает
        # переделать текущий шаг, и цепочка снова доходит до проверки
        result = self.agent.reject_final("итог не полный")
        self.assertIn(result["task"].stage,
                      (TaskStage.EXECUTION, TaskStage.VALIDATION))
        result = self.agent.run_task()
        self.assertEqual(result["task"].stage,
                         TaskStage.AWAITING_FINAL_APPROVAL)

        result = self.agent.approve_final()
        self.assertEqual(result["task"].stage, TaskStage.DONE)

    def test_validation_retry_limit_fails_task(self) -> None:
        # Валидация «проваливается»: retry_count растёт, после лимита — FAILED.
        # В mock-режиме валидация всегда «pass», поэтому временно включаем
        # реальный путь разбора вердикта и подменяем ответ LLM.
        self.agent.start_task("Починить сборку", TaskMode.AUTO)
        self.agent._mock_mode = False
        try:
            with mock.patch.object(
                DeepSeekAgent, "_task_llm_call",
                return_value='{"verdict": "fail", "comment": "не готово"}',
            ):
                result = self.agent.run_task()
        finally:
            self.agent._mock_mode = True
        self.assertEqual(result["task"].stage, TaskStage.FAILED)
        self.assertIn("лимит повторов", result["task"].error)
        self.assertGreater(result["task"].retry_count, TASK_RETRY_LIMIT)


# ---------------------------------------------------------------------------
# Сценарий 6: авто-режим без вмешательства
# ---------------------------------------------------------------------------

class AutoModeTest(TaskFSMTestBase):
    """Авто-режим: planning → execution → validation → done."""

    def test_auto_pipeline_without_intervention(self) -> None:
        result = self.agent.start_task("Автоматическая задача",
                                       TaskMode.AUTO)
        result = self.agent.run_task()
        task = result["task"]
        self.assertEqual(task.stage, TaskStage.DONE)
        self.assertEqual(task.done_steps_count(), len(task.plan))
        self.assertTrue(task.result)

        # Весь путь отражён в журнале событий
        event_types = [e["event_type"] for e in self.agent.get_task_events()]
        self.assertIn("start_task", event_types)
        self.assertIn("plan_ready_auto", event_types)
        self.assertIn("steps_completed", event_types)
        self.assertIn("validation_passed", event_types)
        # Подтверждений пользователя в авто-режиме не было
        self.assertNotIn("approve_plan", event_types)
        self.assertNotIn("approve_final", event_types)


# ---------------------------------------------------------------------------
# Сценарий 7: смена режима на лету
# ---------------------------------------------------------------------------

class ModeSwitchTest(TaskFSMTestBase):
    """Переключение auto → manual в середине execution."""

    def test_switch_to_manual_requires_final_approval(self) -> None:
        self.agent.start_task("Разложить файлы", TaskMode.AUTO)
        result = self.agent.continue_task()  # шаг 1 в авто-цепочке
        self.assertEqual(result["task"].stage, TaskStage.EXECUTION)

        result = self.agent.set_task_mode(TaskMode.MANUAL)
        self.assertEqual(result["task"].mode, TaskMode.MANUAL)

        # Шаги продолжают идти сами, но финальное подтверждение
        # (переход этапа validation → done) теперь требуется
        result = self.agent.run_task()
        self.assertEqual(result["task"].stage,
                         TaskStage.AWAITING_FINAL_APPROVAL)
        self.assertFalse(result["auto"])

        # Возврат в авто НЕ подтверждает этап молча: решение остаётся
        # за пользователем; после подтверждения задача завершена
        self.agent.set_task_mode(TaskMode.AUTO)
        result = self.agent.run_task()
        self.assertEqual(result["task"].stage,
                         TaskStage.AWAITING_FINAL_APPROVAL)
        result = self.agent.approve_final()
        self.assertEqual(result["task"].stage, TaskStage.DONE)


# ---------------------------------------------------------------------------
# Сценарий 8: возобновление без повторных объяснений
# ---------------------------------------------------------------------------

class ContextRestorationTest(TaskFSMTestBase):
    """Контекст задачи при возобновлении берётся из БД."""

    def test_context_block_contains_task_state(self) -> None:
        self.agent.start_task("Миграция базы данных", TaskMode.AUTO)
        self.agent.continue_task()
        self.agent.pause_task()

        # «Перезапуск»: свежий агент + восстановление из БД
        agent2 = _make_agent(self.repo)
        result = agent2.restore_task()
        task = result["task"]
        block = agent2._task_context_block(task)
        # В контексте есть всё, чтобы агент не переспрашивал «что мы делаем?»
        self.assertIn("[Состояние задачи]", block)
        self.assertIn("Миграция базы данных", block)
        self.assertIn(f"Этап: {task.stage.value}", block)
        self.assertIn("Текущий шаг", block)
        self.assertIn("История выполнения", block)
        # Результаты уже выполненных шагов восстановлены из БД
        done_steps = [s for s in task.plan if s.status == "done"]
        self.assertTrue(done_steps)
        for step in done_steps:
            self.assertIn(step.result, block)

    def test_resume_does_not_reset_stage(self) -> None:
        self.agent.start_task("Обновить зависимости", TaskMode.MANUAL)
        stage_after_plan = TaskStage.AWAITING_PLAN_APPROVAL
        self.agent.pause_task()
        result = self.agent.resume_task()
        self.assertEqual(result["task"].stage, stage_after_plan)
        # План не сгенерирован заново: те же шаги, тот же порядок
        events = [e["event_type"] for e in self.agent.get_task_events()]
        self.assertEqual(events.count("start_task"), 1)
        self.assertEqual(events.count("resume"), 1)


# ---------------------------------------------------------------------------
# Сценарий 9: одна активная задача на профиль
# ---------------------------------------------------------------------------

class SingleActiveTaskTest(TaskFSMTestBase):
    """Вторая активная задача для того же профиля запрещена."""

    def test_second_active_task_rejected(self) -> None:
        result = self.agent.start_task("Первая задача", TaskMode.AUTO)
        self.agent.pause_task()  # задача остаётся активной (paused)

        result = self.agent.start_task("Вторая задача", TaskMode.MANUAL)
        self.assertIsNone(result["task"])
        self.assertIn("уже есть активная задача", result["error"])

        # Репозиторий тоже защищается напрямую
        with self.assertRaises(ActiveTaskExistsError):
            self.repo.create_task(0, "Третья", TaskMode.AUTO)

        # После отмены первой — новая задача создаётся
        self.agent.cancel_task()
        result = self.agent.start_task("Новая задача", TaskMode.MANUAL)
        self.assertIsNotNone(result["task"])
        self.assertEqual(result["task"].stage,
                         TaskStage.AWAITING_PLAN_APPROVAL)


# ---------------------------------------------------------------------------
# Сценарий 10: каскадное удаление
# ---------------------------------------------------------------------------

class CascadeDeleteTest(TaskFSMTestBase):
    """Удаление задачи удаляет её шаги и события."""

    def test_delete_task_cascades_steps_and_events(self) -> None:
        result = self.agent.start_task("Удаляемая задача", TaskMode.AUTO)
        task_id = result["task"].id
        self.agent.run_task()
        self.assertTrue(self.repo.get_task_events(task_id))
        self.assertTrue(self.repo.get_task_by_id(task_id).plan)

        self.assertTrue(self.repo.delete_task(task_id))
        self.assertIsNone(self.repo.get_task_by_id(task_id))
        self.assertEqual(self.repo.get_task_events(task_id), [])

        # В самой БД не осталось ни шагов, ни событий
        with sqlite3.connect(self.repo.db_path) as conn:
            steps = conn.execute(
                "SELECT COUNT(*) FROM task_steps WHERE task_id = ?",
                (task_id,)).fetchone()[0]
            events = conn.execute(
                "SELECT COUNT(*) FROM task_events WHERE task_id = ?",
                (task_id,)).fetchone()[0]
        self.assertEqual(steps, 0)
        self.assertEqual(events, 0)


# ---------------------------------------------------------------------------
# Очистка задач: удаление одной и всех
# ---------------------------------------------------------------------------

class TaskCleanupTest(TaskFSMTestBase):
    """Удаление задачи с панели и полная очистка истории задач."""

    def test_delete_finished_task(self) -> None:
        result = self.agent.start_task("Завершённая", TaskMode.AUTO)
        finished_id = result["task"].id
        self.agent.run_task()  # → done

        result = self.agent.start_task("Активная", TaskMode.AUTO)
        active_id = result["task"].id
        # Незавершённую задачу удалить нельзя — сначала «Стоп»
        result = self.agent.delete_task()
        self.assertIn("остановите", result["error"])
        self.assertIsNotNone(self.repo.get_task_by_id(active_id))

        self.agent.cancel_task()
        result = self.agent.delete_task()
        self.assertTrue(result["deleted"])
        self.assertIsNone(self.repo.get_task_by_id(active_id))
        self.assertEqual(self.repo.get_task_events(active_id), [])
        self.assertIsNone(self.agent.active_task)  # панель сброшена
        # Завершённая задача не тронута
        self.assertIsNotNone(self.repo.get_task_by_id(finished_id))

    def test_delete_by_id_keeps_current_task(self) -> None:
        result = self.agent.start_task("Первая", TaskMode.AUTO)
        first_id = result["task"].id
        self.agent.run_task()
        self.agent.start_task("Вторая", TaskMode.MANUAL)

        result = self.agent.delete_task(first_id)
        self.assertTrue(result["deleted"])
        # Текущая задача панели осталась привязанной
        self.assertIsNotNone(self.agent.active_task)
        self.assertEqual(self.agent.active_task.id, first_id + 1)

    def test_clear_tasks_removes_everything(self) -> None:
        result = self.agent.start_task("Первая", TaskMode.AUTO)
        first_id = result["task"].id
        self.agent.run_task()
        self.agent.start_task("Вторая активная", TaskMode.AUTO)

        result = self.agent.clear_tasks()
        self.assertEqual(result["cleared"], 2)
        self.assertEqual(self.repo.count_tasks(), 0)
        self.assertEqual(self.repo.get_task_events(first_id), [])
        self.assertIsNone(self.agent.active_task)
        # После очистки можно создать новую задачу
        result = self.agent.start_task("После очистки", TaskMode.AUTO)
        self.assertIsNotNone(result["task"])

    def test_clear_tasks_when_empty(self) -> None:
        result = self.agent.clear_tasks()
        self.assertEqual(result["cleared"], 0)
        self.assertIsNone(result["error"])


# ---------------------------------------------------------------------------
# Матрица переходов и журнал
# ---------------------------------------------------------------------------

class TransitionMatrixTest(TaskFSMTestBase):
    """Запрещённые переходы бросают InvalidTransitionError, БД не меняется."""

    def test_forbidden_transition_raises_and_keeps_stage(self) -> None:
        task = self.repo.create_task(0, "Матрица", TaskMode.AUTO)
        fsm = TaskFSM(task, self.repo)
        fsm.transition(TaskStage.PLANNING, "start_task")
        with self.assertRaises(InvalidTransitionError):
            fsm.transition(TaskStage.DONE, "skip_everything")
        self.assertEqual(task.stage, TaskStage.PLANNING)
        fresh = self.repo.get_task_by_id(task.id)
        self.assertEqual(fresh.stage, TaskStage.PLANNING)

    def test_approve_wrong_stage_raises(self) -> None:
        self.agent.start_task("Лишний approve", TaskMode.MANUAL)
        # План не подтверждён — финальное подтверждение невозможно
        with self.assertRaises(InvalidTransitionError):
            self.agent._fsm.approve_final()

    def test_events_logged_to_db(self) -> None:
        result = self.agent.start_task("Журнал", TaskMode.MANUAL)
        task_id = result["task"].id
        events = self.repo.get_task_events(task_id)
        self.assertTrue(events)
        first = events[0]
        self.assertEqual(first["event_type"], "start_task")
        self.assertEqual(first["from_stage"], TaskStage.IDLE.value)
        self.assertEqual(first["to_stage"], TaskStage.PLANNING.value)
        self.assertTrue(first["created_at"])


# ---------------------------------------------------------------------------
# UI: панель задачи отражает состояние FSM (пропускается без дисплея)
# ---------------------------------------------------------------------------

class TaskPanelUITest(TaskFSMTestBase):
    """Кнопки панели активны/неактивны по состоянию FSM."""

    def setUp(self) -> None:
        super().setUp()
        from tests import gui_helpers
        self.root = gui_helpers.get_root()
        if self.root is None:
            raise unittest.SkipTest("Tkinter недоступен (нет дисплея).")
        gui_helpers.reset_children(self.root)
        from src.ui import ChatApplication
        self.app = ChatApplication(root=self.root, agent=self.agent)

    def test_panel_hidden_without_task(self) -> None:
        self.assertFalse(self.app.task_panel.winfo_ismapped())

    def test_panel_buttons_follow_fsm(self) -> None:
        result = self.agent.start_task("UI-задача", TaskMode.MANUAL)
        self.app._refresh_task_panel(result["task"])

        # Ожидание плана: принять/отклонить активны, пауза — нет
        self.assertEqual(self.app.task_approve_btn.cget("state"), "normal")
        self.assertEqual(self.app.task_reject_btn.cget("state"), "normal")
        self.assertEqual(self.app.task_pause_btn.cget("state"), "disabled")
        self.assertEqual(self.app.task_resume_btn.cget("state"), "disabled")
        self.assertEqual(self.app.task_stop_btn.cget("state"), "normal")
        # Статус-бар дублирует информацию о задаче
        self.assertIn("Задача:", self.app.status_label.cget("text"))

        # После подтверждения плана шаги идут сами: кнопок подтверждения
        # на этапе выполнения нет — пользователь управляет этапами
        result = self.agent.approve_plan()
        self.app._refresh_task_panel(result["task"])
        self.assertEqual(self.app.task_approve_btn.cget("state"), "disabled")
        self.assertEqual(self.app.task_reject_btn.cget("state"), "disabled")
        self.assertEqual(self.app.task_pause_btn.cget("state"), "normal")

        result = self.agent.pause_task()
        self.app._refresh_task_panel(result["task"])
        self.assertEqual(self.app.task_pause_btn.cget("state"), "disabled")
        self.assertEqual(self.app.task_resume_btn.cget("state"), "normal")

        result = self.agent.cancel_task()
        self.app._refresh_task_panel(result["task"])
        self.assertEqual(self.app.task_stop_btn.cget("state"), "disabled")

    def test_panel_delete_button_and_clearing(self) -> None:
        result = self.agent.start_task("UI-удаление", TaskMode.AUTO)
        self.app._refresh_task_panel(result["task"])
        # Активная задача: удаление заблокировано
        self.assertEqual(self.app.task_delete_btn.cget("state"), "disabled")

        result = self.agent.run_task()  # авто-цепочка до done
        self.app._refresh_task_panel(result["task"])
        self.assertEqual(self.app.task_delete_btn.cget("state"), "normal")

        # Удаление (результат агента обрабатывается как в UI-потоке):
        # панель скрывается, статус-бар очищается от информации о задаче
        self.app._on_task_result(self.agent.delete_task())
        self.assertFalse(self.app.task_panel.winfo_ismapped())
        self.assertEqual(self.app._task_status_text, "")


# ---------------------------------------------------------------------------
# Команда создания задачи в чате
# ---------------------------------------------------------------------------

class TaskCommandParsingTest(unittest.TestCase):
    """Разбор сообщения «/задача <описание>» (без GUI)."""

    def test_parse_task_command(self) -> None:
        from src.ui import ChatApplication

        parse = ChatApplication._parse_task_command
        self.assertEqual(parse("/задача написать парсер логов"),
                         "написать парсер логов")
        self.assertEqual(parse("/Задача  собрать отчёт "), "собрать отчёт")
        self.assertEqual(parse("/task deploy"), "deploy")
        self.assertEqual(parse("/задача"), "")          # без описания
        self.assertEqual(parse("/ЗАДАЧА"), "")
        self.assertIsNone(parse("обычное сообщение"))   # не команда
        self.assertIsNone(parse("/задачачто-то"))       # не команда
        self.assertIsNone(parse(""))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
