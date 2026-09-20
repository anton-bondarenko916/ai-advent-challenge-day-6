"""Тесты явных переходов FSM задач (матрица, контроллер, парсер, UI).

Покрывают 14 тестов из ТЗ (офлайн, без сети):

1. Прямой запрещённый переход: IDLE → EXECUTION → InvalidTransitionError,
   в журнале rejected_transition.
2. Пропуск этапа в manual: PLANNING → EXECUTION без
   AWAITING_PLAN_APPROVAL → отказ.
3. Финал без валидации: EXECUTION → DONE → отказ.
4. Терминальное состояние: DONE → PLANNING → TerminalStateError.
5. Пауза из любого состояния: planning/execution/validation → PAUSED,
   resume возвращает в исходную стадию.
6. Пауза → перезапуск → resume: stage_before_pause восстанавливается
   из task_events.
7. Лимит откатов: validation → execution 3 раза, 4-й → FAILED.
8. Auto-режим: PLANNING → EXECUTION без подтверждения.
9. Manual-режим: PLANNING → AWAITING_PLAN_APPROVAL.
10. Парсинг маркера: [PLAN_READY] → intent plan_ready.
11. Парсинг эвристикой: «план готов» без маркера → intent найден.
12. Fail-safe: без маркера и эвристик → None, стадия не меняется,
    в журнале unresolved_transition.
13. UI граф: текущее состояние подсвечено, обновляется при переходах.
14. Потокобезопасность: параллельные transition_to не ломают состояние.

Запуск из корня проекта:
    python -m unittest tests.test_task_transitions -v
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import unittest
import uuid
from pathlib import Path
from unittest import mock

from src.agent import DeepSeekAgent
from src.config import DEFAULTS, get_task_settings
from src.task_fsm import (
    ALLOWED_TRANSITIONS,
    TASK_RETRY_LIMIT,
    TaskFSM,
    get_allowed_transitions,
)
from src.task_models import (
    InvalidTransitionError,
    Task,
    TaskMode,
    TaskStage,
    TaskStep,
    TerminalStateError,
)
from src.task_repository import TaskRepository
from src.task_response_parser import TaskResponseParser

_TESTS_TMP = Path(__file__).resolve().parent / "tmp_transitions"


def _make_repo(name: str) -> TaskRepository:
    """Репозиторий задач на временной БД."""
    return TaskRepository(_TESTS_TMP / f"{name}_{uuid.uuid4().hex}.db")


class TransitionsBase(unittest.TestCase):
    """Общая база: временный каталог + репозиторий задач."""

    def setUp(self) -> None:
        _TESTS_TMP.mkdir(parents=True, exist_ok=True)
        self._tmp_dir = Path(tempfile.mkdtemp(dir=_TESTS_TMP))
        self.repo = TaskRepository(self._tmp_dir / "transitions.db")

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def make_fsm(self, stage: TaskStage = TaskStage.PLANNING,
                 mode: TaskMode = TaskMode.MANUAL,
                 profile_id: int = 0,
                 **task_fields) -> TaskFSM:
        """Задача в заданной стадии + привязанный FSM."""
        task = self.repo.create_task(profile_id, "Тестовая задача", mode)
        self.repo.update_task_stage(task.id, stage)
        if task_fields:
            self.repo.update_task_fields(task.id, **task_fields)
        fresh = self.repo.get_task_by_id(task.id)
        assert fresh is not None
        return TaskFSM(fresh, self.repo)


# ---------------------------------------------------------------------------
# Матрица и режимы (ТЗ: тесты 1–3, 8, 9)
# ---------------------------------------------------------------------------

class MatrixTest(TransitionsBase):
    """Явная матрица: декларативность и защита от перепрыгивания."""

    def test_idle_to_execution_rejected_and_logged(self) -> None:
        """Тест 1 ТЗ: IDLE → EXECUTION запрещён, отклонённый переход в журнале."""
        fsm = self.make_fsm(TaskStage.IDLE)
        with self.assertRaises(InvalidTransitionError) as ctx:
            fsm.transition_to(TaskStage.EXECUTION, "skip_everything")
        self.assertIn("не разрешён матрицей", ctx.exception.reason)
        # Стадия не изменилась
        self.assertEqual(fsm.task.stage, TaskStage.IDLE)
        # В журнале — отклонённый переход с причиной
        events = self.repo.get_task_events(fsm.task.id)
        rejected = [e for e in events
                    if e["event_type"] == "rejected_transition"]
        self.assertEqual(len(rejected), 1)
        payload = rejected[0]["payload"]
        self.assertFalse(payload["allowed"])
        self.assertIn("не разрешён матрицей", payload["reason"])
        self.assertEqual(rejected[0]["from_stage"], TaskStage.IDLE.value)
        self.assertEqual(rejected[0]["to_stage"], TaskStage.EXECUTION.value)

    def test_manual_planning_to_execution_rejected(self) -> None:
        """Тест 2 ТЗ: в manual нельзя перескочить подтверждение плана."""
        fsm = self.make_fsm(TaskStage.PLANNING, TaskMode.MANUAL)
        allowed, reason = fsm.can_transition(TaskStage.EXECUTION)
        self.assertFalse(allowed)
        with self.assertRaises(InvalidTransitionError):
            fsm.transition_to(TaskStage.EXECUTION, "plan_ready_skip")
        # Через подтверждение — можно
        allowed, _ = fsm.can_transition(TaskStage.AWAITING_PLAN_APPROVAL)
        self.assertTrue(allowed)

    def test_execution_to_done_rejected(self) -> None:
        """Тест 3 ТЗ: финал без валидации запрещён (в EXECUTION нет → DONE)."""
        self.assertNotIn(TaskStage.DONE,
                         ALLOWED_TRANSITIONS[TaskStage.EXECUTION])
        fsm = self.make_fsm(TaskStage.EXECUTION, TaskMode.AUTO)
        with self.assertRaises(InvalidTransitionError):
            fsm.transition_to(TaskStage.DONE, "skip_validation")

    def test_terminal_states_have_no_outgoing(self) -> None:
        """Терминальные стадии — пустые множества в матрице."""
        self.assertEqual(ALLOWED_TRANSITIONS[TaskStage.DONE], set())
        self.assertEqual(ALLOWED_TRANSITIONS[TaskStage.CANCELLED], set())

    def test_mode_aware_transitions(self) -> None:
        """Тесты 8–9 ТЗ: режимы учтены в матрице."""
        # Auto: PLANNING → EXECUTION без подтверждения; awaiting-стадии убраны
        auto = get_allowed_transitions(TaskStage.PLANNING, TaskMode.AUTO)
        self.assertIn(TaskStage.EXECUTION, auto)
        self.assertNotIn(TaskStage.AWAITING_PLAN_APPROVAL, auto)
        # Auto: валидация → DONE напрямую
        self.assertIn(TaskStage.DONE, get_allowed_transitions(
            TaskStage.VALIDATION, TaskMode.AUTO))
        # Manual: PLANNING → AWAITING_PLAN_APPROVAL (и НЕ напрямую в execution)
        manual = get_allowed_transitions(TaskStage.PLANNING, TaskMode.MANUAL)
        self.assertIn(TaskStage.AWAITING_PLAN_APPROVAL, manual)
        self.assertNotIn(TaskStage.EXECUTION, manual)
        # Manual: валидация → ожидание финального подтверждения
        self.assertIn(TaskStage.AWAITING_FINAL_APPROVAL,
                      get_allowed_transitions(TaskStage.VALIDATION,
                                              TaskMode.MANUAL))


# ---------------------------------------------------------------------------
# Терминальные состояния и перезапуск после ошибки (ТЗ: тест 4)
# ---------------------------------------------------------------------------

class TerminalStateTest(TransitionsBase):
    """Из done/cancelled переходов нет; failed можно перезапустить."""

    def test_done_to_planning_raises_terminal(self) -> None:
        """Тест 4 ТЗ: DONE → PLANNING → TerminalStateError."""
        fsm = self.make_fsm(TaskStage.DONE, TaskMode.AUTO)
        with self.assertRaises(TerminalStateError):
            fsm.transition_to(TaskStage.PLANNING, "restart_done")
        self.assertEqual(fsm.task.stage, TaskStage.DONE)

    def test_cancelled_is_terminal(self) -> None:
        fsm = self.make_fsm(TaskStage.CANCELLED, TaskMode.AUTO)
        with self.assertRaises(TerminalStateError):
            fsm.transition_to(TaskStage.PLANNING, "restart_cancelled")

    def test_failed_can_restart_to_planning_and_resets_retries(self) -> None:
        """FAILED → PLANNING разрешён (перезапуск), retry_count обнуляется."""
        fsm = self.make_fsm(TaskStage.FAILED, TaskMode.AUTO,
                            retry_count=TASK_RETRY_LIMIT)
        fsm.transition_to(TaskStage.PLANNING, "restart")
        self.assertEqual(fsm.task.stage, TaskStage.PLANNING)
        self.assertEqual(fsm.task.retry_count, 0)
        fresh = self.repo.get_task_by_id(fsm.task.id)
        assert fresh is not None
        self.assertEqual(fresh.retry_count, 0)


# ---------------------------------------------------------------------------
# Пауза/резюме (ТЗ: тесты 5–6)
# ---------------------------------------------------------------------------

class PauseResumeTest(TransitionsBase):
    """Пауза с любой стадии; resume возвращает в исходную; события в журнале."""

    def test_pause_from_any_stage_and_resume_back(self) -> None:
        """Тест 5 ТЗ: planning/execution/validation → PAUSED → назад."""
        for index, stage in enumerate((
                TaskStage.PLANNING, TaskStage.EXECUTION,
                TaskStage.VALIDATION, TaskStage.AWAITING_PLAN_APPROVAL)):
            fsm = self.make_fsm(stage, TaskMode.AUTO, profile_id=index + 1)
            fsm.pause()
            self.assertEqual(fsm.task.stage, TaskStage.PAUSED)
            resumed_to = fsm.resume()
            self.assertEqual(resumed_to, stage)
            self.assertEqual(fsm.task.stage, stage)
            # Повторная пауза сразу после resume работает
            fsm.pause()
            self.assertEqual(fsm.task.stage, TaskStage.PAUSED)
            fsm.resume()

    def test_pause_rejected_from_terminal(self) -> None:
        fsm = self.make_fsm(TaskStage.DONE, TaskMode.AUTO)
        with self.assertRaises(InvalidTransitionError):
            fsm.pause()

    def test_resume_without_pause_rejected(self) -> None:
        fsm = self.make_fsm(TaskStage.EXECUTION, TaskMode.AUTO)
        with self.assertRaises(InvalidTransitionError):
            fsm.resume()

    def test_stage_before_pause_recovered_from_events(self) -> None:
        """Тест 6 ТЗ: после перезапуска стадия до паузы берётся из task_events."""
        fsm = self.make_fsm(TaskStage.EXECUTION, TaskMode.AUTO)
        fsm.pause()
        task_id = fsm.task.id

        # «Перезапуск»: новый контроллер над свежепрочитанной задачей,
        # стадия до паузы потеряна (как в старых записях БД)
        fresh = self.repo.get_task_by_id(task_id)
        assert fresh is not None
        self.assertEqual(fresh.stage, TaskStage.PAUSED)
        fsm2 = TaskFSM(fresh, self.repo)
        # Память процесса пуста, колонка тоже — остаётся журнал
        self.assertIsNone(fsm2._stage_before_pause)
        recovered = fsm2._load_stage_before_pause()
        self.assertEqual(recovered, TaskStage.EXECUTION)
        self.assertEqual(fsm2.resume(), TaskStage.EXECUTION)
        self.assertEqual(fsm2.task.stage, TaskStage.EXECUTION)


# ---------------------------------------------------------------------------
# Лимит откатов валидации (ТЗ: тест 7)
# ---------------------------------------------------------------------------

class RetryLimitTest(TransitionsBase):
    """validation → execution: 3 отката можно, 4-й запрещён матрицей."""

    def test_three_rollbacks_then_fourth_denied_and_failed(self) -> None:
        """Тест 7 ТЗ: лимит MAX_RETRIES = 3, 4-й откат → FAILED."""
        fsm = self.make_fsm(TaskStage.VALIDATION, TaskMode.AUTO)
        task_id = fsm.task.id
        # Три отката подряд: каждый увеличивает retry_count
        for expected_count in (1, 2, 3):
            fsm.transition_to(TaskStage.EXECUTION, "validation_failed")
            self.assertEqual(fsm.task.retry_count, expected_count)
            # Возвращаемся в валидацию для следующей попытки
            self.repo.update_task_stage(task_id, TaskStage.VALIDATION)
            fsm.task.stage = TaskStage.VALIDATION
        # Четвёртая попытка: лимит исчерпан — отказ с причиной
        allowed, reason = fsm.can_transition(TaskStage.EXECUTION)
        self.assertFalse(allowed)
        self.assertIn("Превышен лимит откатов", reason)
        with self.assertRaises(InvalidTransitionError):
            fsm.transition_to(TaskStage.EXECUTION, "validation_failed")
        # Вызывающий код (агент) переводит задачу в FAILED
        fsm.fail(f"Превышен лимит повторов валидации ({TASK_RETRY_LIMIT})")
        self.assertEqual(fsm.task.stage, TaskStage.FAILED)
        self.assertEqual(fsm.task.retry_count, TASK_RETRY_LIMIT)
        # Отклонённая попытка зафиксирована в журнале
        events = self.repo.get_task_events(task_id)
        rejected = [e for e in events
                    if e["event_type"] == "rejected_transition"]
        self.assertEqual(len(rejected), 1)
        self.assertIn("Превышен лимит откатов", rejected[0]["payload"]["reason"])

    def test_agent_fails_task_on_fourth_validation_fail(self) -> None:
        """Интеграция с агентом: 4-й провал валидации → FAILED (по ТЗ)."""
        agent = DeepSeekAgent(api_key="", mock_mode=True,
                              task_repository=self.repo,
                              task_max_retries=TASK_RETRY_LIMIT)
        agent.start_task("Починить сборку", TaskMode.AUTO)
        agent._mock_mode = False
        try:
            with mock.patch.object(
                DeepSeekAgent, "_task_llm_call",
                return_value='{"verdict": "fail", "comment": "не готово"}',
            ):
                result = agent.run_task()
        finally:
            agent._mock_mode = True
        task = result["task"]
        self.assertEqual(task.stage, TaskStage.FAILED)
        self.assertIn("лимит повторов", task.error or "")
        self.assertEqual(task.retry_count, TASK_RETRY_LIMIT)


# ---------------------------------------------------------------------------
# Парсер ответов LLM (ТЗ: тесты 10–12)
# ---------------------------------------------------------------------------

class ParserTest(unittest.TestCase):
    """TaskResponseParser: маркер → эвристика → fail-safe."""

    def setUp(self) -> None:
        self.parser = TaskResponseParser()

    def test_marker_has_priority(self) -> None:
        """Тест 10 ТЗ: [PLAN_READY] распознаётся; маркер сильнее эвристики."""
        self.assertEqual(self.parser.parse("План готов:\n1. …\n[PLAN_READY]"),
                         "plan_ready")
        # Эвристика «сделано» указывает на step_done, но маркер важнее
        self.assertEqual(
            self.parser.parse("Шаг выполнен, всё сделано.\n[VALIDATION_OK]"),
            "validation_ok",
        )
        for marker, intent in TaskResponseParser.MARKERS.items():
            self.assertEqual(self.parser.parse(f"Текст.\n{marker}"), intent)

    def test_heuristic_fallback(self) -> None:
        """Тест 11 ТЗ: без маркера, но с «план готов» → intent найден."""
        self.assertEqual(
            self.parser.parse("Вот план для задачи: шаг 1, шаг 2."),
            "plan_ready",
        )
        self.assertEqual(self.parser.parse("Все шаги выполнены."),
                         "all_steps_done")
        self.assertEqual(self.parser.parse("Валидация не пройдена."),
                         "validation_fail")

    def test_failsafe_none(self) -> None:
        """Тест 12 ТЗ: нет ни маркера, ни эвристики → None."""
        self.assertIsNone(self.parser.parse("Обычный ответ без намёков."))
        self.assertIsNone(self.parser.parse(""))
        self.assertIsNone(self.parser.parse(None))

    def test_strip_markers(self) -> None:
        text = TaskResponseParser.strip_markers("Ответ.\n[PLAN_READY]")
        self.assertEqual(text, "Ответ.")


# ---------------------------------------------------------------------------
# Интеграция с агентом: ответ чата → переход через FSM
# ---------------------------------------------------------------------------

class AgentTransitionTest(TransitionsBase):
    """send_message: намерение LLM применяется только через матрицу."""

    def setUp(self) -> None:
        super().setUp()
        self.agent = DeepSeekAgent(api_key="", mock_mode=True,
                                   task_repository=self.repo)

    def test_llm_marker_drives_transition(self) -> None:
        """LLM предлагает [ALL_STEPS_DONE] — FSM выполняет переход."""
        self.agent.start_task("Задача", TaskMode.AUTO)
        agent_continue = self.agent.continue_task()  # шаг 1
        self.assertEqual(agent_continue["task"].stage, TaskStage.EXECUTION)
        result = self.agent.send_message("как дела с задачей?")
        # Mock-ответ без маркера: fail-safe, стадия не изменилась
        self.assertEqual(result["task_transition"]["intent"], None)
        self.assertIn("transition_hint", result)
        self.assertEqual(self.agent.active_task.stage, TaskStage.EXECUTION)

        # Подменяем ответ: LLM просит перехода — FSM выполняет
        with mock.patch.object(
            DeepSeekAgent, "_mock_response",
            return_value={"content": "Работа закончена.\n[ALL_STEPS_DONE]",
                          "model": "m", "tokens": {},
                          "finish_reason": "stop"},
        ):
            result = self.agent.send_message("что дальше?")
        self.assertEqual(result["task_transition"]["intent"],
                         "all_steps_done")
        self.assertEqual(result["task_transition"]["transitioned_to"],
                         TaskStage.VALIDATION.value)
        self.assertEqual(self.agent.active_task.stage, TaskStage.VALIDATION)
        # Маркер убран из текста, показанного пользователю
        self.assertNotIn("[ALL_STEPS_DONE]", result["text"])
        # Маркер убран и из истории диалога
        self.assertNotIn("[ALL_STEPS_DONE]",
                         self.agent.history[-1]["content"])

    def test_llm_rejected_transition_reported(self) -> None:
        """LLM просит запрещённое (EXECUTION → DONE) — матрица отклоняет."""
        self.agent.start_task("Задача", TaskMode.AUTO)
        self.agent.continue_task()
        self.assertEqual(self.agent.active_task.stage, TaskStage.EXECUTION)
        with mock.patch.object(
            DeepSeekAgent, "_mock_response",
            return_value={"content": "Готово, закрывай.\n[VALIDATION_OK]",
                          "model": "m", "tokens": {},
                          "finish_reason": "stop"},
        ):
            result = self.agent.send_message("закрываем?")
        # LLM предложила DONE, но валидации не было → отказ матрицы
        self.assertIn("transition_error", result)
        self.assertEqual(self.agent.active_task.stage, TaskStage.EXECUTION)
        # Отклонённая попытка в журнале
        events = self.repo.get_task_events(self.agent.active_task.id)
        rejected = [e for e in events
                    if e["event_type"] == "rejected_transition"]
        self.assertTrue(rejected)

    def test_failsafe_logs_unresolved_transition(self) -> None:
        """Тест 12 ТЗ (интеграция): None парсера → лог + стадия на месте."""
        self.agent.start_task("Задача", TaskMode.AUTO)
        self.agent.continue_task()
        stage_before = self.agent.active_task.stage
        result = self.agent.send_message("расскажи анекдот")
        self.assertEqual(result["task_transition"]["intent"], None)
        self.assertEqual(self.agent.active_task.stage, stage_before)
        events = self.repo.get_task_events(self.agent.active_task.id)
        self.assertTrue(any(e["event_type"] == "unresolved_transition"
                            for e in events))

    def test_markers_block_in_request(self) -> None:
        """Уровень 1 (промпт): блок маркеров с допустимыми переходами."""
        self.agent.start_task("Задача", TaskMode.AUTO)
        self.agent.continue_task()
        block = self.agent._task_markers_block()
        self.assertIn("[МАРКЕРЫ СОСТОЯНИЯ ЗАДАЧИ]", block)
        self.assertIn("[PLAN_READY]", block)
        self.assertIn(
            f"Текущее состояние задачи: "
            f"{self.agent.active_task.stage.value}", block,
        )
        self.assertIn("Допустимые следующие состояния:", block)
        # В запросе блок добавлен к system-сообщению (история не мутируется)
        request = self.agent._build_request()
        self.assertIn("[МАРКЕРЫ СОСТОЯНИЯ ЗАДАЧИ]",
                      request["messages"][0]["content"])
        self.assertNotIn("[МАРКЕРЫ СОСТОЯНИЯ ЗАДАЧИ]",
                         self.agent.history[0]["content"])
        # Без активной задачи блока нет
        self.agent.cancel_task()
        self.assertEqual(self.agent._task_markers_block(), "")

    def test_transition_target_respects_mode(self) -> None:
        """plan_ready/validation_ok ведут в разные стадии в auto и manual."""
        self.agent.start_task("Задача", TaskMode.AUTO)
        self.assertEqual(
            self.agent._transition_target_for("plan_ready"),
            TaskStage.EXECUTION,
        )
        self.agent.set_task_mode(TaskMode.MANUAL)
        self.assertEqual(
            self.agent._transition_target_for("plan_ready"),
            TaskStage.AWAITING_PLAN_APPROVAL,
        )
        self.assertEqual(
            self.agent._transition_target_for("validation_ok"),
            TaskStage.AWAITING_FINAL_APPROVAL,
        )
        self.agent.set_task_mode(TaskMode.AUTO)
        self.assertEqual(
            self.agent._transition_target_for("validation_ok"),
            TaskStage.DONE,
        )
        # need_input — без перехода
        self.assertIsNone(self.agent._transition_target_for("need_input"))


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

class TaskSettingsTest(unittest.TestCase):
    """get_task_settings: TASKS_MAX_RETRIES из конфига."""

    def test_default(self) -> None:
        self.assertEqual(get_task_settings(dict(DEFAULTS)), {"max_retries": 3})

    def test_env_override_and_garbage(self) -> None:
        config = dict(DEFAULTS)
        config["TASKS_MAX_RETRIES"] = "5"
        self.assertEqual(get_task_settings(config)["max_retries"], 5)
        config["TASKS_MAX_RETRIES"] = "abc"
        self.assertEqual(get_task_settings(config)["max_retries"], 3)
        config["TASKS_MAX_RETRIES"] = "999"  # вне [0, 100] → 3
        self.assertEqual(get_task_settings(config)["max_retries"], 3)

    def test_fsm_uses_configured_limit(self) -> None:
        repo = TaskRepository(Path(tempfile.mkdtemp(dir=_TESTS_TMP)) / "x.db")
        task = repo.create_task(0, "Лимит", TaskMode.AUTO)
        repo.update_task_stage(task.id, TaskStage.VALIDATION)
        fresh = repo.get_task_by_id(task.id)
        fsm = TaskFSM(fresh, repo, max_retries=1)
        fsm.transition_to(TaskStage.EXECUTION, "validation_failed")
        # Лимит 1: после первого отката второй уже запрещён
        fsm.task.stage = TaskStage.VALIDATION
        allowed, reason = fsm.can_transition(TaskStage.EXECUTION)
        self.assertFalse(allowed)
        self.assertIn("Превышен лимит откатов (1)", reason)


# ---------------------------------------------------------------------------
# Потокобезопасность (ТЗ: тест 14)
# ---------------------------------------------------------------------------

class ThreadSafetyTest(TransitionsBase):
    """Параллельные transition_to не ломают состояние."""

    def test_parallel_transitions_are_serialized(self) -> None:
        """Тест 14 ТЗ: N потоков дают ровно N успешных/отклонённых исходов,
        стадия в БД согласована с объектом задачи."""
        fsm = self.make_fsm(TaskStage.AWAITING_PLAN_APPROVAL,
                            TaskMode.MANUAL)
        task_id = fsm.task.id
        barrier = threading.Barrier(8)
        results = []

        def worker() -> None:
            barrier.wait()
            try:
                fsm.transition_to(TaskStage.EXECUTION, "approve_plan")
                results.append("ok")
            except InvalidTransitionError:
                results.append("rejected")

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Ровно один поток выполнил переход, остальные получили отказ
        self.assertEqual(results.count("ok"), 1)
        self.assertEqual(results.count("rejected"), 7)
        self.assertEqual(fsm.task.stage, TaskStage.EXECUTION)
        fresh = self.repo.get_task_by_id(task_id)
        assert fresh is not None
        self.assertEqual(fresh.stage, TaskStage.EXECUTION)
        # Каждая попытка записана: 1 успешная + 7 отклонённых
        events = self.repo.get_task_events(task_id)
        self.assertEqual(len([e for e in events
                              if e["event_type"] == "approve_plan"]), 1)
        self.assertEqual(len([e for e in events
                              if e["event_type"] == "rejected_transition"]),
                         7)


# ---------------------------------------------------------------------------
# UI: граф состояний (ТЗ: тест 13, пропускается без дисплея)
# ---------------------------------------------------------------------------

try:  # pragma: no cover — зависит от окружения
    import tkinter as tk
except ImportError:  # pragma: no cover
    tk = None  # type: ignore

from tests import gui_helpers  # noqa: E402


class FsmGraphGuiTest(unittest.TestCase):
    """Canvas-граф: текущая стадия подсвечена, обновляется при переходах."""

    root: "tk.Tk"

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = gui_helpers.get_root()
        if cls.root is None:
            raise unittest.SkipTest("графическая среда недоступна (tkinter/Tk)")
        from src.ui import ChatApplication
        cls._ChatApplication = ChatApplication

    def setUp(self) -> None:
        _TESTS_TMP.mkdir(parents=True, exist_ok=True)
        self._tmp_dir = Path(tempfile.mkdtemp(dir=_TESTS_TMP))
        self.repo = TaskRepository(self._tmp_dir / "gui_fsm.db")
        self.agent = DeepSeekAgent(api_key="", mock_mode=True,
                                   task_repository=self.repo)
        gui_helpers.reset_children(self.root)
        self.app = self._ChatApplication(root=self.root, agent=self.agent)
        self.root.update()

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def test_graph_highlights_current_stage_and_updates(self) -> None:
        """Тест 13 ТЗ: подсветка текущей стадии, обновление при переходах."""
        result = self.agent.start_task("Граф-задача", TaskMode.MANUAL)
        task = result["task"]
        self.assertEqual(task.stage, TaskStage.AWAITING_PLAN_APPROVAL)
        self.app._refresh_task_panel(task)
        self.root.update()

        # Один узел помечен как текущий; надпись допустимых переходов есть
        current_items = self.app.task_fsm_canvas.find_withtag("current")
        self.assertEqual(len(current_items), 1)
        label_text = self.app.task_fsm_allowed_label.cget("text")
        self.assertIn("Допустимые переходы:", label_text)
        self.assertIn("Выполнение", label_text)
        # AWAITING_STEP_APPROVAL в подписи не показывается (легаси-стадия)
        self.assertNotIn("Ожидание шага", label_text)

        # Переход (подтверждение плана): подсветка сместилась, подпись новая
        result = self.agent.approve_plan()
        self.app._refresh_task_panel(result["task"])
        self.root.update()
        current_items = self.app.task_fsm_canvas.find_withtag("current")
        self.assertEqual(len(current_items), 1)
        self.assertIn("Пауза",
                      self.app.task_fsm_allowed_label.cget("text"))

        # Пауза: единственная подсвеченная точка — узел ПАУЗА;
        # из паузы матрица разрешает только отмену
        result = self.agent.pause_task()
        self.app._refresh_task_panel(result["task"])
        self.root.update()
        self.assertEqual(
            len(self.app.task_fsm_canvas.find_withtag("current")), 1)
        self.assertIn("Отменена",
                      self.app.task_fsm_allowed_label.cget("text"))
        self.assertNotIn("Выполнение",
                         self.app.task_fsm_allowed_label.cget("text"))

    def test_graph_cleared_without_task(self) -> None:
        self.app._refresh_task_panel(None)
        self.assertEqual(self.app.task_fsm_canvas.find_all(), ())
        self.assertEqual(self.app.task_fsm_allowed_label.cget("text"), "")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
