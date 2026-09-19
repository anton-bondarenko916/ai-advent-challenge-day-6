"""Слой доступа к данным задач (SQLite).

Класс :class:`TaskRepository` — единственное место, где живут SQL-запросы
подсистемы задач. Таблицы:

- ``tasks`` — задачи (состояние FSM, режим, шаг, результат);
- ``task_steps`` — шаги плана (принадлежат задаче, удаляются каскадно);
- ``task_events`` — журнал переходов FSM (удаляются каскадно).

Каскадное удаление реализовано через ``ON DELETE CASCADE`` +
``PRAGMA foreign_keys = ON``: удаление задачи автоматически удаляет её
шаги и события.

Как и остальные хранилища проекта, репозиторий использует короткоживущие
соединения sqlite3 (безопасно при вызовах из разных потоков — запросы к
API выполняются в отдельном потоке). Ошибки БД не пробрасываются наружу
из методов записи: они логируются, а методы возвращают False/None.

UI и агент работают только с объектами :class:`~src.task_models.Task` и
:class:`~src.task_models.TaskStep` — SQL наружу не выходит.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from . import utils
from .task_models import (
    ACTIVE_STAGES,
    ActiveTaskExistsError,
    Task,
    TaskMode,
    TaskStage,
    TaskStep,
)


def _now() -> str:
    """Локальное время в ISO-формате (для created_at/updated_at)."""
    return datetime.now().isoformat(timespec="seconds")


class TaskRepository:
    """Репозиторий задач: CRUD задач, шагов и журнала событий.

    Attributes:
        db_path: путь к файлу БД (по умолчанию ``data/tasks.db``).
        enabled: False переводит все операции в no-op (как ContextStore).
    """

    def __init__(self, db_path: str | Path,
                 enabled: bool = True) -> None:
        """Создаёт репозиторий и схему БД (идемпотентно).

        Args:
            db_path: путь к файлу БД.
            enabled: False — репозиторий отключён (все операции no-op).
        """
        self.db_path = Path(db_path)
        self.enabled = bool(enabled)
        self._log = utils.get_logger("deepseek-chat.task_repo")
        if self.enabled:
            self._migrate()

    # ------------------------------------------------------------------
    # Соединения и схема
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Открывает короткоживущее соединение с включёнными FK."""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        """Контекст соединения: транзакция + гарантированное закрытие."""
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _migrate(self) -> None:
        """Создаёт таблицы задач (идемпотентно); ошибки не роняют запуск."""
        try:
            parent = self.db_path.parent
            if str(parent) and not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)
            with self._session() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tasks (
                        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                        profile_id         INTEGER NOT NULL,
                        title              TEXT NOT NULL DEFAULT '',
                        description        TEXT NOT NULL,
                        mode               TEXT NOT NULL DEFAULT 'manual',
                        stage              TEXT NOT NULL DEFAULT 'idle',
                        current_step_index INTEGER NOT NULL DEFAULT 0,
                        expected_action    TEXT NOT NULL DEFAULT '',
                        rejection_reason   TEXT,
                        retry_count        INTEGER NOT NULL DEFAULT 0,
                        stage_before_pause TEXT,
                        result             TEXT,
                        error              TEXT,
                        created_at         TEXT NOT NULL,
                        updated_at         TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS task_steps (
                        id               INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id          INTEGER NOT NULL
                            REFERENCES tasks(id) ON DELETE CASCADE,
                        step_index       INTEGER NOT NULL,
                        description      TEXT NOT NULL,
                        status           TEXT NOT NULL DEFAULT 'pending',
                        result           TEXT,
                        rejection_reason TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS task_events (
                        id         INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id    INTEGER NOT NULL
                            REFERENCES tasks(id) ON DELETE CASCADE,
                        from_stage TEXT,
                        to_stage   TEXT,
                        event_type TEXT NOT NULL,
                        payload    TEXT,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_task_steps_task "
                    "ON task_steps(task_id, step_index)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_task_events_task "
                    "ON task_events(task_id, id)"
                )
        except sqlite3.Error as exc:
            self._log.error("Не удалось создать схему БД задач: %s", exc)

    # ------------------------------------------------------------------
    # Разбор строк БД в доменные объекты
    # ------------------------------------------------------------------

    @staticmethod
    def _stage_of(value: Any) -> TaskStage:
        """Безопасно разбирает строку стадии (неизвестная → IDLE)."""
        try:
            return TaskStage(str(value))
        except ValueError:
            return TaskStage.IDLE

    @staticmethod
    def _mode_of(value: Any) -> TaskMode:
        """Безопасно разбирает строку режима (неизвестный → MANUAL)."""
        try:
            return TaskMode(str(value))
        except ValueError:
            return TaskMode.MANUAL

    def _task_from_row(self, row: sqlite3.Row,
                       conn: sqlite3.Connection) -> Task:
        """Собирает Task из строки tasks + её шаги (одним соединением)."""
        task = Task(
            id=int(row["id"]),
            profile_id=int(row["profile_id"]),
            title=str(row["title"] or ""),
            description=str(row["description"] or ""),
            mode=self._mode_of(row["mode"]),
            stage=self._stage_of(row["stage"]),
            current_step_index=int(row["current_step_index"] or 0),
            expected_action=str(row["expected_action"] or ""),
            rejection_reason=row["rejection_reason"],
            retry_count=int(row["retry_count"] or 0),
            stage_before_pause=(
                self._stage_of(row["stage_before_pause"])
                if row["stage_before_pause"] else None
            ),
            result=row["result"],
            error=row["error"],
        )
        step_rows = conn.execute(
            "SELECT id, task_id, step_index, description, status, result, "
            "rejection_reason FROM task_steps WHERE task_id = ? "
            "ORDER BY step_index, id",
            (task.id,),
        ).fetchall()
        task.plan = [
            TaskStep(
                id=int(step_row["id"]),
                task_id=int(step_row["task_id"]),
                step_index=int(step_row["step_index"]),
                description=str(step_row["description"] or ""),
                status=str(step_row["status"] or "pending"),
                result=step_row["result"],
                rejection_reason=step_row["rejection_reason"],
            )
            for step_row in step_rows
        ]
        return task

    # ------------------------------------------------------------------
    # Задачи
    # ------------------------------------------------------------------

    def create_task(self, profile_id: int, description: str,
                    mode: TaskMode) -> Task:
        """Создаёт задачу для профиля.

        Правило «одна активная задача на профиль»: если у профиля уже есть
        активная задача, выбрасывается :class:`ActiveTaskExistsError` —
        сначала нужно завершить/отменить существующую.

        Args:
            profile_id: идентификатор профиля-владельца.
            description: описание задачи.
            mode: режим работы (auto/manual).

        Returns:
            Созданная задача (со Stage=IDLE; FSM переведёт её в PLANNING).

        Raises:
            ActiveTaskExistsError: у профиля уже есть активная задача.
        """
        if not self.enabled:
            raise RuntimeError("TaskRepository отключён (enabled=False).")
        active = self.get_active_task(int(profile_id))
        if active is not None:
            raise ActiveTaskExistsError(
                f"У профиля id={profile_id} уже есть активная задача "
                f"#{active.id} («{active.title}»). Завершите или отмените её."
            )
        now = _now()
        with self._session() as conn:
            cursor = conn.execute(
                "INSERT INTO tasks(profile_id, title, description, mode, "
                "stage, current_step_index, expected_action, "
                "created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?)",
                (int(profile_id), "", str(description),
                 mode.value, TaskStage.IDLE.value, "", now, now),
            )
            task_id = int(cursor.lastrowid or 0)
        task = self.get_task_by_id(task_id)
        if task is None:  # pragma: no cover — только при ошибке БД
            raise RuntimeError(f"Задача id={task_id} не создана.")
        return task

    def get_active_task(self, profile_id: int) -> Optional[Task]:
        """Возвращает активную задачу профиля (None — активной нет).

        Активной считается задача, чья стадия входит в ACTIVE_STAGES
        (в том числе paused и awaiting_*).
        """
        if not self.enabled:
            return None
        placeholders = ",".join("?" for _ in ACTIVE_STAGES)
        try:
            with self._session() as conn:
                row = conn.execute(
                    f"SELECT * FROM tasks WHERE profile_id = ? AND stage "
                    f"IN ({placeholders}) ORDER BY id DESC LIMIT 1",
                    (int(profile_id),
                     *[stage.value for stage in ACTIVE_STAGES]),
                ).fetchone()
                return self._task_from_row(row, conn) if row else None
        except sqlite3.Error as exc:
            self._log.error("Ошибка чтения активной задачи: %s", exc)
            return None

    def get_task_by_id(self, task_id: int) -> Optional[Task]:
        """Возвращает задачу по идентификатору (None — не найдена)."""
        if not self.enabled:
            return None
        try:
            with self._session() as conn:
                row = conn.execute(
                    "SELECT * FROM tasks WHERE id = ?", (int(task_id),)
                ).fetchone()
                return self._task_from_row(row, conn) if row else None
        except sqlite3.Error as exc:
            self._log.error("Ошибка чтения задачи id=%s: %s", task_id, exc)
            return None

    def update_task_stage(self, task_id: int, new_stage: TaskStage,
                          expected_action: str = "") -> bool:
        """Обновляет стадию и expected_action задачи (вызывает FSM)."""
        if not self.enabled:
            return False
        try:
            with self._session() as conn:
                conn.execute(
                    "UPDATE tasks SET stage = ?, expected_action = ?, "
                    "updated_at = ? WHERE id = ?",
                    (new_stage.value, str(expected_action or ""),
                     _now(), int(task_id)),
                )
            return True
        except sqlite3.Error as exc:
            self._log.error("Ошибка обновления стадии задачи %s: %s",
                            task_id, exc)
            return False

    def update_task_fields(self, task_id: int, **fields: Any) -> bool:
        """Точечное обновление полей задачи (белый список имён).

        Поддерживаются: title, current_step_index, rejection_reason,
        retry_count, stage_before_pause, result, error, mode.
        Значения TaskMode/TaskStage сериализуются в строку.
        """
        if not self.enabled:
            return False
        allowed = {
            "title", "current_step_index", "rejection_reason",
            "retry_count", "stage_before_pause", "result", "error", "mode",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Неизвестные поля задачи: {sorted(unknown)}")
        serialized: Dict[str, Any] = {}
        for key, value in fields.items():
            if key == "mode" and isinstance(value, TaskMode):
                value = value.value
            elif key == "stage_before_pause" and isinstance(value, TaskStage):
                value = value.value
            serialized[key] = value
        if not serialized:
            return True
        assignments = ", ".join(f"{key} = ?" for key in serialized)
        values = list(serialized.values()) + [_now(), int(task_id)]
        try:
            with self._session() as conn:
                conn.execute(
                    f"UPDATE tasks SET {assignments}, updated_at = ? "
                    f"WHERE id = ?",
                    values,
                )
            return True
        except sqlite3.Error as exc:
            self._log.error("Ошибка обновления полей задачи %s: %s",
                            task_id, exc)
            return False

    def set_task_mode(self, task_id: int, mode: TaskMode) -> bool:
        """Меняет режим задачи на лету (auto/manual)."""
        return self.update_task_fields(task_id, mode=mode)

    def increment_retry_count(self, task_id: int) -> int:
        """Увеличивает retry_count на 1 и возвращает новое значение (-1 при ошибке)."""
        if not self.enabled:
            return -1
        try:
            with self._session() as conn:
                conn.execute(
                    "UPDATE tasks SET retry_count = retry_count + 1, "
                    "updated_at = ? WHERE id = ?",
                    (_now(), int(task_id)),
                )
                row = conn.execute(
                    "SELECT retry_count FROM tasks WHERE id = ?",
                    (int(task_id),),
                ).fetchone()
            return int(row[0]) if row else -1
        except sqlite3.Error as exc:
            self._log.error("Ошибка инкремента retry_count задачи %s: %s",
                            task_id, exc)
            return -1

    def set_task_result(self, task_id: int, result: str) -> bool:
        """Сохраняет итоговый результат задачи."""
        return self.update_task_fields(task_id, result=result)

    def set_task_error(self, task_id: int, error: str) -> bool:
        """Сохраняет текст ошибки задачи (стадию переводит FSM)."""
        return self.update_task_fields(task_id, error=error)

    def cancel_task(self, task_id: int) -> bool:
        """Помечает задачу отменённой (стадию и событие логирует FSM)."""
        return self.update_task_fields(task_id, stage_before_pause=None)

    def delete_task(self, task_id: int) -> bool:
        """Удаляет задачу каскадно (вместе с шагами и событиями)."""
        if not self.enabled:
            return False
        try:
            with self._session() as conn:
                conn.execute(
                    "DELETE FROM tasks WHERE id = ?", (int(task_id),)
                )
            return True
        except sqlite3.Error as exc:
            self._log.error("Ошибка удаления задачи %s: %s", task_id, exc)
            return False

    def delete_all_tasks(self) -> int:
        """Удаляет ВСЕ задачи (каскадно — со всеми шагами и журналами).

        Returns:
            Число удалённых задач, -1 при ошибке БД.
        """
        if not self.enabled:
            return -1
        try:
            with self._session() as conn:
                cursor = conn.execute("DELETE FROM tasks")
                return max(0, int(cursor.rowcount or 0))
        except sqlite3.Error as exc:
            self._log.error("Ошибка очистки задач: %s", exc)
            return -1

    def count_tasks(self) -> int:
        """Общее число задач в БД (-1 при ошибке)."""
        if not self.enabled:
            return -1
        try:
            with self._session() as conn:
                row = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()
            return int(row[0]) if row else 0
        except sqlite3.Error as exc:
            self._log.error("Ошибка подсчёта задач: %s", exc)
            return -1

    # ------------------------------------------------------------------
    # Шаги плана
    # ------------------------------------------------------------------

    def update_task_plan(self, task_id: int,
                         steps: List[TaskStep]) -> bool:
        """Полностью заменяет план задачи (используется при генерации плана).

        Шагам проставляются task_id и step_index по порядку списка;
        прежние шаги удаляются (вместе с их статусами).
        """
        if not self.enabled:
            return False
        try:
            with self._session() as conn:
                conn.execute(
                    "DELETE FROM task_steps WHERE task_id = ?",
                    (int(task_id),),
                )
                for index, step in enumerate(steps):
                    cursor = conn.execute(
                        "INSERT INTO task_steps(task_id, step_index, "
                        "description, status, result, rejection_reason) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (int(task_id), index, str(step.description or ""),
                         str(step.status or "pending"), step.result,
                         step.rejection_reason),
                    )
                    step.id = int(cursor.lastrowid or 0)
                    step.task_id = int(task_id)
                    step.step_index = index
            return True
        except sqlite3.Error as exc:
            self._log.error("Ошибка сохранения плана задачи %s: %s",
                            task_id, exc)
            return False

    def update_step_status(self, step_id: int, status: str,
                           result: Optional[str] = None,
                           rejection_reason: Optional[str] = None) -> bool:
        """Обновляет статус шага и (опционально) результат/причину отклонения."""
        if not self.enabled:
            return False
        try:
            with self._session() as conn:
                conn.execute(
                    "UPDATE task_steps SET status = ?, "
                    "result = COALESCE(?, result), "
                    "rejection_reason = COALESCE(?, rejection_reason) "
                    "WHERE id = ?",
                    (str(status), result, rejection_reason, int(step_id)),
                )
            return True
        except sqlite3.Error as exc:
            self._log.error("Ошибка обновления шага %s: %s", step_id, exc)
            return False

    # ------------------------------------------------------------------
    # Журнал событий FSM
    # ------------------------------------------------------------------

    def log_event(self, task_id: int,
                  from_stage: Optional[TaskStage],
                  to_stage: Optional[TaskStage],
                  event_type: str,
                  payload: Optional[Dict[str, Any]] = None) -> bool:
        """Добавляет запись в журнал переходов (payload — JSON)."""
        if not self.enabled:
            return False
        try:
            with self._session() as conn:
                conn.execute(
                    "INSERT INTO task_events(task_id, from_stage, to_stage, "
                    "event_type, payload, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (int(task_id),
                     from_stage.value if from_stage else None,
                     to_stage.value if to_stage else None,
                     str(event_type),
                     json.dumps(payload, ensure_ascii=False)
                     if payload else None,
                     _now()),
                )
            return True
        except sqlite3.Error as exc:
            self._log.error("Ошибка записи события задачи %s: %s",
                            task_id, exc)
            return False

    def get_task_events(self, task_id: int) -> List[Dict[str, Any]]:
        """Возвращает журнал событий задачи (в порядке добавления)."""
        if not self.enabled:
            return []
        try:
            with self._session() as conn:
                rows = conn.execute(
                    "SELECT id, from_stage, to_stage, event_type, payload, "
                    "created_at FROM task_events WHERE task_id = ? "
                    "ORDER BY id",
                    (int(task_id),),
                ).fetchall()
        except sqlite3.Error as exc:
            self._log.error("Ошибка чтения событий задачи %s: %s",
                            task_id, exc)
            return []
        events: List[Dict[str, Any]] = []
        for row in rows:
            payload = row["payload"]
            try:
                payload = json.loads(payload) if payload else None
            except (TypeError, ValueError):
                payload = None
            events.append({
                "id": int(row["id"]),
                "from_stage": row["from_stage"],
                "to_stage": row["to_stage"],
                "event_type": str(row["event_type"] or ""),
                "payload": payload,
                "created_at": row["created_at"],
            })
        return events
