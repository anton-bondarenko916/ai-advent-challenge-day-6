"""Проверка заданий через filesystem MCP (ТЗ ai_1_36, Модуль 4).

Сценарий (agentic-цепочка из ТЗ):

1. пользователь указывает путь к папке своего Android-проекта;
2. fs-инструмент ``tree`` отдаёт дерево проекта;
3. LLM по заданию выбирает 3–5 файлов, которые нужно прочитать
   (``SELECT_FILES_PROMPT``, JSON ``{"files": [...]}``);
4. ``read_multiple_files`` читает выбранные файлы;
5. LLM оценивает код по заданию и рубрике (``REVIEW_PROMPT``) —
   JSON ``{"score": 1..5, "feedback": "…"}``; без рубрики проверка
   не запускается вовсе (ТЗ «ЧЕГО НЕ ДЕЛАТЬ»);
6. результат сохраняется в progress (оценка ≥ 4 завершает подтему).

Транспорт: fs_server.py — но НЕ общий сервер приложения (порт 8005):
его ALLOWED_ROOT — корень этого проекта, а папка пользователя лежит
снаружи. :class:`EphemeralFsClient` поднимает ЭФЕМЕРНЫЙ экземпляр того
же fs_server.py с ``--root <папка проекта>`` на свободном порту и
останавливает его после проверки — доступ ограничен ровно папкой
пользователя, состояние общего сервера не затрагивается.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.utils import get_logger
from app.learning.db import LearningRepository
from app.learning.llm import REVIEW_MARKER, SELECT_FILES_MARKER

logger = get_logger("deepseek-chat.learning.review")

# Лимиты ТЗ/защиты контекста
MAX_SELECTED_FILES = 5      # LLM выбирает 3-5 файлов (ТЗ)
TREE_DEPTH = 3
READY_TIMEOUT = 20.0        # подъём эфемерного fs-сервера, секунд

SELECT_FILES_PROMPT = """{marker}
Задание:
{task}

Дерево проекта (JSON):
{tree}

Выбери 3-5 файлов, которые нужно прочитать для проверки задания.
Верни JSON-объект: {{"files": ["путь/к/файлу", ...]}} — пути
относительно корня проекта. Выбирай только файлы из дерева
(исходники Kotlin/Java, Gradle-скрипты, README); бинарные файлы
(изображения, *.apk) не выбирай.
"""

REVIEW_PROMPT = """{marker}
Ты — преподаватель Android-разработки. Проверь задание.

[Задание]
{task}

[Критерии оценки]
{rubric}

[Код пользователя]
{contents}

Оцени задание по шкале 1-5 и дай развёрнутый фидбек:
- Что сделано хорошо
- Что можно улучшить
- Конкретные рекомендации

Верни JSON-объект: {{"score": int, "feedback": "текст"}}
"""


class ReviewError(RuntimeError):
    """Ошибка проверки задания (путь/сервер/LLM/пустой выбор файлов)."""


def parse_tool_json(text: Any) -> Any:
    """Значение из JSON-результата MCP-инструмента (FastMCP оборачивает
    значение в ``{"result": …}``; не-JSON → None)."""
    try:
        payload = json.loads(str(text or ""))
    except (TypeError, ValueError):
        return None
    if isinstance(payload, dict) and set(payload) == {"result"}:
        payload = payload["result"]
    return payload


def deep_tool_value(text: Any) -> Any:
    """Разворачивает повторно закодированные ответы fs-сервера.

    Инструменты fs_server, возвращающие СТРОКУ (например ``tree`` —
    JSON-строка дерева), приходят дважды закодированными:
    ``structuredContent → {"result": "<json-строка>"}``. Поэтому
    после снятия обёртки разворачиваем ещё один уровень JSON.
    """
    value = parse_tool_json(text)
    if isinstance(value, str):
        value = parse_tool_json(value)
    return value


def normalize_selected_files(data: Any) -> List[str]:
    """Список путей из ответа LLM ({"files": [...]} или голый массив):
    строки, без дубликатов, не больше MAX_SELECTED_FILES."""
    raw: Any = data
    if isinstance(data, dict):
        raw = data.get("files")
    if not isinstance(raw, list):
        return []
    seen: set = set()
    files: List[str] = []
    for item in raw:
        path = str(item or "").strip().replace("\\", "/")
        if not path or path in seen:
            continue
        seen.add(path)
        files.append(path)
        if len(files) >= MAX_SELECTED_FILES:
            break
    return files


def _free_port() -> int:
    """Свободный TCP-порт на 127.0.0.1 (как в живых тестах fs-сервера)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class EphemeralFsClient:
    """fs_server.py как подпроцесс с FS_ROOT = папка проекта пользователя.

    Запускается на свободном порту (создание окна консоли подавлено),
    готовность — опрос TCP-порта; :meth:`close` останавливает процесс
    (безопасно вызывать повторно). Клиент MCP — ReminderMCPClient,
    каждый call_tool — отдельная HTTP-сессия (как в ядре).
    """

    def __init__(self, project_path: str, server_path: Optional[Path] = None,
                 ready_timeout: float = READY_TIMEOUT) -> None:
        self.project_path = str(project_path)
        self.server_path = Path(server_path or (
            Path(__file__).resolve().parents[2] / "fs_server.py"))
        self.ready_timeout = float(ready_timeout)
        self.port = _free_port()
        self.client: Optional[Any] = None
        self._process: Optional[subprocess.Popen] = None
        self._log_file = None

    def start(self) -> "EphemeralFsClient":
        """Запускает сервер и ждёт готовности (Иначе — ReviewError)."""
        from src.reminder_client import ReminderMCPClient

        self._log_file = open(self._log_path(), "a", encoding="utf-8")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) \
            if sys.platform == "win32" else 0
        self._process = subprocess.Popen(
            [
                sys.executable, "-u", str(self.server_path),
                "--root", self.project_path,
                "--port", str(self.port),
            ],
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,   # сервер не должен глотать ввод UI
            creationflags=creationflags,
        )
        deadline = time.monotonic() + self.ready_timeout
        while time.monotonic() < deadline:
            if self._tcp_ready():
                break
            if self._process.poll() is not None:
                raise ReviewError(
                    f"FS-сервер проверки упал при старте (код "
                    f"{self._process.returncode})")
            time.sleep(0.2)
        else:
            raise ReviewError(
                f"FS-сервер проверки не поднялся за {self.ready_timeout:.0f} с")
        self.client = ReminderMCPClient(
            f"http://127.0.0.1:{self.port}/mcp", label="FS-сервер проверки")
        return self

    def _log_path(self) -> Path:
        # Журнал — во временном каталоге системы, НЕ в папке проекта
        # пользователя (иначе он попал бы в дерево, которое читает LLM)
        return Path(tempfile.gettempdir()) / f"fs_review_{self.port}.log"

    def _tcp_ready(self) -> bool:
        """True, если порт сервера отвечает на TCP-подключение."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex(("127.0.0.1", self.port)) == 0

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """tools/call через HTTP MCP-клиент (прозрачный прокси)."""
        if self.client is None:
            raise ReviewError("FS-сервер проверки не запущен")
        return await self.client.call_tool(tool_name, arguments)

    async def close(self) -> None:
        """Останавливает эфемерный сервер (идемпотентно)."""
        process, self._process = self._process, None
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                process.kill()
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None


def default_fs_provider(project_path: str) -> EphemeralFsClient:
    """Провайдер по умолчанию: эфемерный fs-сервер с корнем в project_path."""
    return EphemeralFsClient(project_path).start()


class ReviewService:
    """Проверка задания: fs MCP читает проект, LLM оценивает по рубрике.

    Attributes:
        repo: LearningRepository (подтемы/прогресс).
        llm: JSON-клиент (выбор файлов + оценка).
        fs_provider: project_path → клиент с call_tool/close; по
            умолчанию эфемерный fs-сервер (тесты подставляют фейк).
    """

    def __init__(self, repo: LearningRepository, llm: Any,
                 fs_provider: Optional[Callable[[str], Any]] = None) -> None:
        self.repo = repo
        self.llm = llm
        self.fs_provider = fs_provider or default_fs_provider

    async def review_task(self, user_id: int, subtopic_id: int,
                          project_path: str) -> Dict[str, Any]:
        """Полная проверка (цепочка ТЗ). Returns: {"score", "feedback",
        "files", "completed", "subtopic_id", "topic_title"}.

        Raises:
            ReviewError: путь не папка, у подтемы нет рубрики, fs/LLM
                недоступны, модель не выбрала файлы.
        """
        subtopic = self.repo.get_subtopic_for_user(user_id, subtopic_id)
        if subtopic is None:
            raise ReviewError("Подтема не найдена")
        rubric = str(subtopic.get("rubric") or "").strip()
        if not rubric:
            raise ReviewError(
                "У подтемы нет критериев оценки (rubric) — проверка "
                "невозможна")
        folder = Path(str(project_path or "").strip().strip('"'))
        if not folder.is_dir():
            raise ReviewError(f"Папка проекта не найдена: {folder}")

        client = self.fs_provider(str(folder))
        try:
            tree = deep_tool_value(await client.call_tool(
                "tree", {"path": ".", "depth": TREE_DEPTH}))
            if isinstance(tree, dict) and "error" in tree:
                raise ReviewError(f"fs-сервер: {tree['error']}")
            if not isinstance(tree, dict) or not tree:
                raise ReviewError("fs-сервер вернул пустое дерево проекта")

            selection = await self.llm.generate_json_async(
                SELECT_FILES_PROMPT.format(
                    marker=SELECT_FILES_MARKER,
                    task=str(subtopic["task"]),
                    tree=json.dumps(tree, ensure_ascii=False, indent=2)))
            files = normalize_selected_files(selection)
            if not files:
                raise ReviewError(
                    "Модель не выбрала файлы для проверки — попробуйте "
                    "ещё раз")

            contents = deep_tool_value(await client.call_tool(
                "read_multiple_files", {"paths": files}))
            if not isinstance(contents, dict):
                raise ReviewError("fs-сервер вернул некорректное содержимое")
        finally:
            closer = getattr(client, "close", None)
            if closer is not None:
                result = closer()
                if hasattr(result, "__await__"):
                    await result

        review = await self.llm.generate_json_async(REVIEW_PROMPT.format(
            marker=REVIEW_MARKER,
            task=str(subtopic["task"]),
            rubric=rubric,
            contents=self._format_contents(contents),
        ))
        if not isinstance(review, dict):
            raise ReviewError("Ответ модели не является JSON-объектом")
        try:
            score = int(review.get("score"))
        except (TypeError, ValueError) as exc:
            raise ReviewError("Модель не вернула числовую оценку") from exc
        feedback = str(review.get("feedback") or "").strip()

        completed = self.repo.save_review(
            user_id, subtopic_id, feedback, score,
            task_answer=f"Проверены файлы: {', '.join(files)}")
        logger.info("Проверка подтемы %s: score=%s, файлы: %s",
                    subtopic_id, score, ", ".join(files))
        return {
            "score": max(1, min(5, score)),
            "feedback": feedback,
            "files": files,
            "completed": completed,
            "subtopic_id": subtopic_id,
            "topic_title": subtopic.get("topic_title") or "",
        }

    @staticmethod
    def _format_contents(contents: Dict[str, Any]) -> str:
        """Файлы для промпта проверки: «=== path ===» + содержимое."""
        blocks = []
        for path, text in contents.items():
            blocks.append(f"=== {path} ===\n{text}")
        return "\n\n".join(blocks)
