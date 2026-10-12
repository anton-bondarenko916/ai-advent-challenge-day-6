"""Тесты собственного MCP-сервера Git и вызова инструментов из приложения.

Покрывают сценарии приёмки ТЗ (my_git_server.py + call_tool + UI):

1. Сервер запускается: ``python -u my_git_server.py`` не падает (smoke);
2. Инструменты регистрируются: в FastMCP видны get_git_log/status/branches;
3. MCPClient.list_tools() возвращает 3 инструмента с описаниями и схемой;
4. call_tool("get_git_log", {..., "limit": 5}) возвращает 5 коммитов;
5. call_tool("get_git_status") возвращает modified/staged/untracked;
6. call_tool("get_git_branches") возвращает список веток;
7. несуществующий repo_path → результат с {"error": "..."};
8. UI: кнопка «Вызвать» открывает диалог, результат отображается;
9. Потокобезопасность: UI не зависает во время вызова.

Плюс юнит-покрытие: разбор git-вывода (юзеры, '|', пустой репозиторий,
зажим limit, переименования), конфиг (резолв python/.py-путей,
MCP_GIT_REPO_PATH), воркер вызова, валидация формы диалога.

Git-репозитории для тестов создаются внутри проекта
(``tests/tmp_git_server/`` — в .gitignore), системный TEMP не нужен.
Интеграционные тесты требуют SDK ``mcp`` и git в PATH, иначе пропускаются.

Запуск из корня проекта:
    python -m unittest tests.test_mcp_server -v
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

from src.config import PROJECT_ROOT, get_mcp_settings
from src.mcp_client import (
    MCPClient,
    MCPConnectionError,
    MCPTimeoutError,
)
from src.mcp_worker import MCPToolCallWorker

try:  # pragma: no cover — зависит от окружения
    import tkinter as tk
except ImportError:  # pragma: no cover
    tk = None  # type: ignore

from tests import gui_helpers

try:  # pragma: no cover — зависит от окружения
    import mcp  # noqa: F401

    MCP_SDK = True
except ImportError:  # pragma: no cover
    MCP_SDK = False

GIT_SERVER = str(PROJECT_ROOT / "my_git_server.py")
GIT_AVAILABLE = shutil.which("git") is not None

# Базовый каталог git-фикстур (внутри репозитория — см. .gitignore).
_FIXTURE_BASE = Path(__file__).resolve().parent / "tmp_git_server"


def _server_command() -> list:
    """Команда запуска собственного сервера (python -u, без буферизации)."""
    return [sys.executable, "-u", GIT_SERVER]


def _unwrap(payload):
    """Снимает обёртку annotated structured content FastMCP.

    structuredContent всегда ``{"result": <значение>}``; тесты принимают
    оба варианта (обёрнутый и нет) — устойчиво к версии SDK.
    """
    if isinstance(payload, dict) and set(payload) == {"result"}:
        return payload["result"]
    return payload


def _new_repo_dir(name: str) -> Path:
    """Уникальный каталог под git-репозиторий в tests/tmp_git_server/."""
    path = _FIXTURE_BASE / f"{name}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _git(path: Path, *args: str) -> subprocess.CompletedProcess:
    """Запускает git в репозитории (check=True)."""
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True, capture_output=True, text=True, encoding="utf-8",
        errors="replace",
    )


def _make_repo(path: Path, commits: int = 0, branches=()) -> Path:
    """Создаёт настоящий git-репозиторий с коммитами и ветками.

    Локальная конфигурация user.name/user.email — чтобы commit работал
    при любом глобальном .gitconfig машины.
    """
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init")
    _git(path, "config", "user.name", "Test User")
    _git(path, "config", "user.email", "test@example.com")
    for i in range(commits):
        (path / f"file_{i}.txt").write_text(f"content {i}\n", encoding="utf-8")
        _git(path, "add", ".")
        _git(path, "commit", "-m", f"commit number {i}")
    for branch in branches:
        _git(path, "branch", branch)
    return path


class _RepoCase(unittest.TestCase):
    """База для тестов с git-фикстурами: каталоги и очистка."""

    def setUp(self) -> None:
        super().setUp()
        _FIXTURE_BASE.mkdir(parents=True, exist_ok=True)
        self._paths: list = []

    def make_repo(self, name: str, commits: int = 0, branches=()) -> Path:
        path = _make_repo(_new_repo_dir(name), commits=commits, branches=branches)
        self._paths.append(path)
        return path

    def make_dir(self, name: str) -> Path:
        path = _new_repo_dir(name)
        self._paths.append(path)
        return path

    def tearDown(self) -> None:
        # Песочница может запретить удаление — тогда каталоги остаются
        # в gitignored tests/tmp_git_server/ и тест не падает.
        for path in self._paths:
            shutil.rmtree(path, ignore_errors=True)
        super().tearDown()


# ---------------------------------------------------------------------------
# 1-2. Сервер: запуск и регистрация инструментов
# ---------------------------------------------------------------------------

@unittest.skipUnless(MCP_SDK and GIT_AVAILABLE, "нужны SDK mcp и git")
class GitServerRegistrationTest(unittest.TestCase):
    """my_git_server: инструменты зарегистрированы, схемы корректны."""

    def test_server_module_registers_four_tools(self) -> None:
        """Сценарий 2: в FastMCP зарегистрированы все 4 инструмента."""
        import my_git_server

        tools = asyncio.run(my_git_server.mcp.list_tools())
        names = {t.name for t in tools}
        self.assertEqual(
            names,
            {"get_git_log", "get_git_status", "get_git_branches",
             "get_current_branch"},
        )
        by_name = {t.name: t for t in tools}
        for tool in tools:
            self.assertTrue(tool.description, f"{tool.name}: нет описания")
            self.assertEqual(tool.inputSchema.get("type"), "object")

        # get_git_log: repo_path обязателен (string), limit — integer=10.
        log_schema = by_name["get_git_log"].inputSchema
        self.assertEqual(log_schema.get("required"), ["repo_path"])
        self.assertEqual(log_schema["properties"]["repo_path"]["type"], "string")
        self.assertEqual(log_schema["properties"]["limit"]["type"], "integer")
        self.assertEqual(log_schema["properties"]["limit"]["default"], 10)

        # Остальные: единственный обязательный строковый repo_path.
        for name in ("get_git_status", "get_git_branches",
                     "get_current_branch"):
            schema = by_name[name].inputSchema
            self.assertEqual(schema.get("required"), ["repo_path"])
            self.assertEqual(schema["properties"]["repo_path"]["type"], "string")

    def test_server_process_starts_and_stays_alive(self) -> None:
        """Сценарий 1: ``python -u my_git_server.py`` запускается и живёт."""
        proc = subprocess.Popen(
            _server_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            time.sleep(2.0)
            self.assertIsNone(
                proc.poll(), "сервер завершился сам до подключения клиента"
            )
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover — страховка
                proc.kill()
            # Закрываем пайпы — иначе unittest ругается ResourceWarning.
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    stream.close()
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Инструменты: разбор реального git-вывода (прямые вызовы функций)
# ---------------------------------------------------------------------------

@unittest.skipUnless(GIT_AVAILABLE, "git недоступен")
class GitToolsDirectTest(_RepoCase):
    """Прямые вызовы функций сервера (без MCP-транспорта)."""

    def _import_server(self):
        import my_git_server

        return my_git_server

    def test_get_git_log_returns_commits(self) -> None:
        server = self._import_server()
        repo = self.make_repo("log", commits=7)
        commits = server.get_git_log(str(repo), 5)
        self.assertEqual(len(commits), 5)
        for item in commits:
            self.assertEqual(set(item), {"hash", "author", "date", "message"})
            self.assertEqual(len(item["hash"]), 8)
            self.assertEqual(item["author"], "Test User")
            self.assertTrue(item["message"].startswith("commit number "))
        # Порядок — от новых к старым.
        self.assertEqual(
            [c["message"] for c in commits],
            [f"commit number {i}" for i in (6, 5, 4, 3, 2)],
        )

    def test_get_git_log_default_limit_ten(self) -> None:
        server = self._import_server()
        repo = self.make_repo("log_default", commits=12)
        commits = server.get_git_log(str(repo))
        self.assertEqual(len(commits), 10)

    def test_get_git_log_limit_is_clamped(self) -> None:
        server = self._import_server()
        repo = self.make_repo("log_clamp", commits=3)
        # 0 и отрицательные — некорректный "git log -N": зажимаем к 1.
        self.assertEqual(len(server.get_git_log(str(repo), 0)), 1)
        self.assertEqual(len(server.get_git_log(str(repo), -5)), 1)
        # Заведомо больший лимит не ломает вывод.
        self.assertEqual(len(server.get_git_log(str(repo), 1000)), 3)

    def test_get_git_log_message_with_pipe(self) -> None:
        server = self._import_server()
        repo = self.make_repo("log_pipe")
        (repo / "f.txt").write_text("x\n", encoding="utf-8")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "тема | с деталью")
        commits = server.get_git_log(str(repo), 1)
        self.assertEqual(commits[0]["message"], "тема | с деталью")

    def test_get_git_log_missing_path(self) -> None:
        server = self._import_server()
        result = server.get_git_log(str(_FIXTURE_BASE / "no_such_repo_xyz"), 5)
        self.assertEqual(len(result), 1)
        self.assertIn("error", result[0])
        self.assertIn("не существует", result[0]["error"])

    def test_get_git_log_empty_repo(self) -> None:
        server = self._import_server()
        repo = self.make_repo("empty")  # init без коммитов
        self.assertEqual(server.get_git_log(str(repo), 10), [])

    def test_get_git_log_non_repo_directory(self) -> None:
        server = self._import_server()
        plain = self.make_dir("plain")
        # Каталог лежит внутри git-репозитория проекта: без потолка git
        # найдёт «родительский» репозиторий восходящим поиском. Потолок
        # запрещает подъём выше самого каталога.
        with mock.patch.dict(
            "os.environ", {"GIT_CEILING_DIRECTORIES": str(plain.parent)}
        ):
            result = server.get_git_log(str(plain), 5)
        self.assertEqual(len(result), 1)
        self.assertIn("error", result[0])

    def test_get_git_status_partitions(self) -> None:
        server = self._import_server()
        repo = self.make_repo("status", commits=1)
        # modified: изменение зафиксированного файла (не индексируется)
        (repo / "file_0.txt").write_text("changed\n", encoding="utf-8")
        # staged: новый файл в индексе
        (repo / "staged_file.txt").write_text("staged\n", encoding="utf-8")
        _git(repo, "add", "staged_file.txt")
        # untracked: новый файл вне индекса
        (repo / "untracked_file.txt").write_text("untracked\n", encoding="utf-8")

        status = server.get_git_status(str(repo))
        self.assertEqual(set(status), {"modified", "staged", "untracked"})
        self.assertIn("file_0.txt", status["modified"])
        self.assertIn("staged_file.txt", status["staged"])
        self.assertIn("untracked_file.txt", status["untracked"])

    def test_get_git_status_clean_repo(self) -> None:
        server = self._import_server()
        repo = self.make_repo("status_clean", commits=1)
        self.assertEqual(
            server.get_git_status(str(repo)),
            {"modified": [], "staged": [], "untracked": []},
        )

    def test_get_git_status_rename_shows_new_path(self) -> None:
        server = self._import_server()
        repo = self.make_repo("status_rename", commits=1)
        _git(repo, "mv", "file_0.txt", "renamed_file.txt")
        status = server.get_git_status(str(repo))
        self.assertIn("renamed_file.txt", status["staged"])
        self.assertNotIn("file_0.txt", status["staged"])

    def test_get_git_status_missing_path(self) -> None:
        server = self._import_server()
        result = server.get_git_status(str(_FIXTURE_BASE / "no_such_repo_abc"))
        self.assertIn("error", result)

    def test_get_git_branches_lists_all(self) -> None:
        server = self._import_server()
        repo = self.make_repo("branches", commits=1, branches=("feature_x",))
        branches = server.get_git_branches(str(repo))
        self.assertIn("feature_x", branches)
        self.assertEqual(len(branches), 2)  # текущая + feature_x

    def test_get_git_branches_missing_path(self) -> None:
        server = self._import_server()
        result = server.get_git_branches(str(_FIXTURE_BASE / "no_such_repo_b"))
        self.assertEqual(len(result), 1)
        self.assertIn("не существует", result[0])


# ---------------------------------------------------------------------------
# get_current_branch: git-контекст команды /help (прямые вызовы + MCP)
# ---------------------------------------------------------------------------

@unittest.skipUnless(GIT_AVAILABLE, "git недоступен")
class GetCurrentBranchDirectTest(_RepoCase):
    """get_current_branch: имя ветки, unborn/detached HEAD, ошибки."""

    def _import_server(self):
        import my_git_server

        return my_git_server

    def test_returns_checked_out_branch(self) -> None:
        server = self._import_server()
        repo = self.make_repo("current_branch", commits=1,
                              branches=("feature_x",))
        _git(repo, "checkout", "feature_x")
        self.assertEqual(
            server.get_current_branch(str(repo)), "feature_x")

    def test_returns_default_branch_without_commits(self) -> None:
        server = self._import_server()
        repo = self.make_repo("current_unborn")
        self.assertIn(
            server.get_current_branch(str(repo)), ("master", "main"))

    def test_detached_head_reports_hash(self) -> None:
        server = self._import_server()
        repo = self.make_repo("current_detached", commits=2)
        _git(repo, "checkout", "--detach", "HEAD")
        result = server.get_current_branch(str(repo))
        self.assertTrue(
            result.startswith("detached HEAD ("), result)

    def test_missing_path(self) -> None:
        server = self._import_server()
        result = server.get_current_branch(
            str(_FIXTURE_BASE / "no_such_repo_branch"))
        self.assertIn("не существует", result)

    def test_not_a_repo(self) -> None:
        server = self._import_server()
        # ВАЖНО: каталог ВНЕ репозитория проекта — внутри tests/tmp_git_server
        # git найдёт внешние .git (родительский репозиторий) и вернёт ветку
        import tempfile as _tempfile

        directory = Path(_tempfile.mkdtemp(prefix="not_a_repo_"))
        try:
            result = server.get_current_branch(str(directory))
            self.assertIn("Ошибка git", result)
        finally:
            shutil.rmtree(directory, ignore_errors=True)


@unittest.skipUnless(MCP_SDK and GIT_AVAILABLE, "нужны SDK mcp и git")
class GetCurrentBranchClientTest(_RepoCase):
    """get_current_branch через MCPClient (пайплайн /help зовёт так же)."""

    def test_call_returns_branch_name(self) -> None:
        repo = self.make_repo("client_branch", commits=1,
                              branches=("help_branch",))
        _git(repo, "checkout", "help_branch")
        client = MCPClient(
            server_command=_server_command(),
            timeout=30.0,
            log_path=_new_repo_dir("logs") / "mcp.log",
        )
        text = asyncio.run(client.call_tool(
            "get_current_branch", {"repo_path": str(repo)}))
        self.assertEqual(_unwrap(json.loads(text)), "help_branch")

    def test_call_missing_path_returns_error_text(self) -> None:
        client = MCPClient(
            server_command=_server_command(),
            timeout=30.0,
            log_path=_new_repo_dir("logs") / "mcp.log",
        )
        text = asyncio.run(client.call_tool(
            "get_current_branch",
            {"repo_path": str(_FIXTURE_BASE / "no_such_repo_cb")},
        ))
        self.assertIn("не существует", _unwrap(json.loads(text)))


# ---------------------------------------------------------------------------
# 3-7. Клиент: интеграция с собственным сервером (stdio-подпроцесс)
# ---------------------------------------------------------------------------

@unittest.skipUnless(MCP_SDK and GIT_AVAILABLE, "нужны SDK mcp и git")
class MCPClientGitServerTest(_RepoCase):
    """MCPClient против реального my_git_server.py (сценарии 3-7)."""

    @classmethod
    def setUpClass(cls) -> None:
        _FIXTURE_BASE.mkdir(parents=True, exist_ok=True)
        cls.shared_repo = _make_repo(
            _new_repo_dir("client_shared"), commits=7, branches=("feature_x",),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.shared_repo, ignore_errors=True)

    def _client(self) -> MCPClient:
        return MCPClient(
            server_command=_server_command(),
            timeout=30.0,
            log_path=_new_repo_dir("logs") / "mcp.log",
        )

    def test_list_tools_returns_four_git_tools(self) -> None:
        """Сценарий 3: tools/list отдаёт 4 инструмента с описаниями."""
        tools = asyncio.run(self._client().list_tools())
        names = [t.name for t in tools]
        # Сортировка по имени — как в панели.
        self.assertEqual(
            names,
            ["get_current_branch", "get_git_branches", "get_git_log",
             "get_git_status"],
        )
        for tool in tools:
            self.assertIn("git", tool.description.lower())
            self.assertIsNotNone(tool.input_schema)

    def test_call_get_git_log_returns_five_commits(self) -> None:
        """Сценарий 4: call_tool("get_git_log", limit=5) → 5 коммитов."""
        text = asyncio.run(self._client().call_tool(
            "get_git_log", {"repo_path": str(self.shared_repo), "limit": 5}
        ))
        commits = _unwrap(json.loads(text))
        self.assertEqual(len(commits), 5)
        self.assertEqual(set(commits[0]), {"hash", "author", "date", "message"})

    def test_call_get_git_status(self) -> None:
        """Сценарий 5: call_tool("get_git_status") → modified/staged/untracked."""
        (self.shared_repo / "untracked_probe.txt").write_text(
            "probe\n", encoding="utf-8"
        )
        text = asyncio.run(self._client().call_tool(
            "get_git_status", {"repo_path": str(self.shared_repo)}
        ))
        status = _unwrap(json.loads(text))
        self.assertEqual(set(status), {"modified", "staged", "untracked"})
        self.assertIn("untracked_probe.txt", status["untracked"])

    def test_call_get_git_branches(self) -> None:
        """Сценарий 6: call_tool("get_git_branches") → список имён веток."""
        text = asyncio.run(self._client().call_tool(
            "get_git_branches", {"repo_path": str(self.shared_repo)}
        ))
        branches = _unwrap(json.loads(text))
        self.assertIsInstance(branches, list)
        self.assertIn("feature_x", branches)

    def test_call_with_missing_path_returns_error(self) -> None:
        """Сценарий 7: несуществующий repo_path → [{"error": "..."}]."""
        text = asyncio.run(self._client().call_tool(
            "get_git_log",
            {"repo_path": str(_FIXTURE_BASE / "no_such_repo_client"), "limit": 3},
        ))
        payload = _unwrap(json.loads(text))
        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 1)
        self.assertIn("error", payload[0])

    def test_call_unknown_tool_returns_error_text(self) -> None:
        """Неизвестный инструмент: isError-ответ приходит текстом (не исключение)."""
        text = asyncio.run(self._client().call_tool("no_such_tool", {}))
        self.assertIn("no_such_tool", text)

    def test_call_with_empty_name_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            asyncio.run(self._client().call_tool("  ", {}))

    def test_call_with_missing_command_raises_connection_error(self) -> None:
        client = MCPClient(
            server_command=["mcp-no-such-executable-xyz"],
            timeout=5.0,
            log_path=_new_repo_dir("logs") / "mcp.log",
        )
        with self.assertRaises(MCPConnectionError):
            asyncio.run(client.call_tool("get_git_log", {}))

    def test_call_timeout_raises_timeout_error(self) -> None:
        client = MCPClient(
            server_command=[sys.executable, "-u", "-c", "import time; time.sleep(30)"],
            timeout=2.0,
            log_path=_new_repo_dir("logs") / "mcp.log",
        )
        with self.assertRaises(MCPTimeoutError):
            asyncio.run(client.call_tool("get_git_log", {}))


# ---------------------------------------------------------------------------
# Конфигурация: резолв локальной команды и git_repo_path
# ---------------------------------------------------------------------------

class GitServerConfigTest(unittest.TestCase):
    """get_mcp_settings: python/.py-резолв, MCP_GIT_REPO_PATH, MCP_ENABLED."""

    def test_python_command_resolves_interpreter_and_script(self) -> None:
        settings = get_mcp_settings({"MCP_SERVER_COMMAND": "python -u my_git_server.py"})
        self.assertEqual(settings["server_command"][0], sys.executable)
        self.assertEqual(settings["server_command"][1], "-u")
        self.assertEqual(
            settings["server_command"][2], str(PROJECT_ROOT / "my_git_server.py")
        )

    def test_npx_command_is_untouched(self) -> None:
        settings = get_mcp_settings({})
        self.assertEqual(
            settings["server_command"], ["npx", "-y", "open-internet-mcp"]
        )

    def test_absolute_python_is_not_replaced(self) -> None:
        settings = get_mcp_settings({
            "MCP_SERVER_COMMAND": f'"{sys.executable}" -u my_git_server.py'
        })
        self.assertEqual(settings["server_command"][0], sys.executable)

    def test_git_repo_path_defaults_to_project_root(self) -> None:
        settings = get_mcp_settings({})
        self.assertEqual(Path(settings["git_repo_path"]), PROJECT_ROOT)

    def test_git_repo_path_custom_relative_resolved(self) -> None:
        settings = get_mcp_settings({"MCP_GIT_REPO_PATH": "some/repo"})
        self.assertEqual(
            Path(settings["git_repo_path"]), PROJECT_ROOT / "some" / "repo"
        )

    def test_mcp_enabled_false_is_coerced(self) -> None:
        """MCP_ENABLED=false в .env действительно выключает панель."""
        from src.config import _coerce

        self.assertIs(_coerce("MCP_ENABLED", "false"), False)
        self.assertIs(_coerce("MCP_ENABLED", "true"), True)


# ---------------------------------------------------------------------------
# Воркер вызова
# ---------------------------------------------------------------------------

class MCPToolCallWorkerTest(unittest.TestCase):
    """MCPToolCallWorker: callback'и успеха/ошибки, daemon-поток."""

    def _run_worker(self, worker: MCPToolCallWorker, done: dict) -> None:
        self.assertTrue(worker.daemon)
        worker.start()
        worker.join(timeout=10)
        self.assertFalse(worker.is_alive(), "воркер завис")

    def test_success_delivers_text(self) -> None:
        done: dict = {}
        client = mock.Mock()

        async def _call(name, args):
            return f"результат {name}"

        client.call_tool = _call
        self._run_worker(
            MCPToolCallWorker(
                "get_git_log", {"limit": 5},
                on_success=lambda text: done.update(text=text),
                on_error=lambda msg: done.update(error=msg),
                client=client,
            ),
            done,
        )
        self.assertEqual(done["text"], "результат get_git_log")
        self.assertNotIn("error", done)

    def test_mcp_error_delivers_message(self) -> None:
        from src.mcp_client import MCPTimeoutError

        done: dict = {}
        client = mock.Mock()

        def _boom(name, args):
            raise MCPTimeoutError("Инструмент не ответил за 30 секунд")

        client.call_tool = _boom
        self._run_worker(
            MCPToolCallWorker(
                "get_git_log", {},
                on_success=lambda text: done.update(text=text),
                on_error=lambda msg: done.update(error=msg),
                client=client,
            ),
            done,
        )
        self.assertIn("не ответил", done["error"])

    def test_unexpected_error_delivers_generic_message(self) -> None:
        done: dict = {}
        client = mock.Mock()

        def _boom(name, args):
            raise RuntimeError("бум")

        client.call_tool = _boom
        self._run_worker(
            MCPToolCallWorker(
                "get_git_log", {},
                on_success=lambda text: done.update(text=text),
                on_error=lambda msg: done.update(error=msg),
                client=client,
            ),
            done,
        )
        self.assertIn("Неожиданная ошибка", done["error"])


# ---------------------------------------------------------------------------
# 8-9. UI: кнопка «Вызвать», диалог, результат, потокобезопасность
# ---------------------------------------------------------------------------

def _sync_dialog(dialog) -> None:
    """Дожидается доставки результата фейкового воркера в диалог.

    Вместо ожидания after-цикла вызываем обработчик очереди напрямую.
    """
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not dialog._busy and dialog._queue.empty():
            return
        dialog._poll()
        time.sleep(0.01)
    raise AssertionError("воркер вызова не ответил за 5 секунд")


@unittest.skipUnless(tk is not None, "tkinter недоступен")
class ToolCallDialogTest(unittest.TestCase):
    """Диалог вызова: форма по схеме, префилл, валидация, результат."""

    root = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = gui_helpers.get_root()
        if cls.root is None:
            raise unittest.SkipTest("графическая среда недоступна (tkinter/Tk)")

    def setUp(self) -> None:
        gui_helpers.reset_children(self.root)
        self.tmp = _new_repo_dir("dialog_state")
        self.panel = self._make_panel()
        self.root.update()
        self._reset_call()

    def tearDown(self) -> None:
        for dialog in list(self.root.winfo_children()):
            if isinstance(dialog, tk.Toplevel):
                try:
                    dialog.destroy()
                except tk.TclError:
                    pass
        try:
            self.panel.destroy()
        except tk.TclError:
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.root.update()

    def _make_panel(self):
        from src.mcp_panel import MCPPanel

        return MCPPanel(
            self.root,
            settings={
                "enabled": True,
                "server_command": _server_command(),
                "timeout": 30.0,
                "log_path": self.tmp / "mcp.log",
                "git_repo_path": str(PROJECT_ROOT / "some" / "repo"),
            },
            worker_factory=lambda on_success, on_error: mock.Mock(),
        )

    def _reset_call(self) -> None:
        """Сбрасывает управляемый фейк воркера вызова.

        ``_call_result`` — текст успеха, ``_call_error`` — исключение или
        текст ошибки; «висящий» воркер — оба None.
        """
        self._call_result = None
        self._call_error = None
        self._call_event = threading.Event()
        self._calls: list = []
        panel = self.panel

        def factory(tool_name, arguments, on_success, on_error):
            self._calls.append({"tool": tool_name, "args": arguments})

            def _start() -> None:
                def runner() -> None:
                    self._call_event.wait(15)
                    if self._call_error is not None:
                        on_error(str(self._call_error))
                    elif self._call_result is not None:
                        on_success(self._call_result)

                threading.Thread(target=runner, daemon=True).start()

            worker = mock.Mock()
            worker.start.side_effect = _start
            return worker

        panel._make_call_worker = factory

    def _deliver(self, result=None, error=None) -> None:
        """Подаёт результат (или ошибку) воркеру, как будто сервер ответил."""
        self._call_result = result
        self._call_error = error
        self._call_event.set()

    def _git_tool(self, **schema_overrides):
        from src.mcp_models import MCPTool

        schema = {
            "type": "object",
            "properties": {
                "repo_path": {"title": "Repo Path", "type": "string"},
                "limit": {"default": 10, "title": "Limit", "type": "integer"},
            },
            "required": ["repo_path"],
        }
        schema.update(schema_overrides)
        return MCPTool(
            name="get_git_log",
            description="Возвращает последние N коммитов в репозитории.",
            input_schema=schema,
        )

    # ---- сценарии -------------------------------------------------------

    def test_card_has_call_and_schema_buttons(self) -> None:
        """Сценарий 8 (начало): на карточке есть «Вызвать» и «Параметры»."""
        self.panel._add_tool_card(self._git_tool())
        self.root.update()
        buttons = self._find_button_texts(self.panel.tools_frame)
        self.assertIn("▶ Вызвать", buttons)
        self.assertIn("Параметры", buttons)

    def test_call_dialog_builds_form_and_prefills(self) -> None:
        """Диалог: поля по схеме, repo_path из настроек, limit из default."""
        dialog = self.panel.open_call_dialog(self._git_tool())
        self.addCleanup(dialog.destroy)
        self.root.update()
        self.assertIn("get_git_log", dialog.title())
        self.assertEqual(
            [f["name"] for f in dialog._fields], ["repo_path", "limit"]
        )
        repo_field = dialog._fields[0]
        limit_field = dialog._fields[1]
        self.assertTrue(repo_field["required"])
        self.assertFalse(limit_field["required"])
        self.assertEqual(
            repo_field["var"].get(), str(PROJECT_ROOT / "some" / "repo")
        )
        self.assertEqual(limit_field["var"].get(), "10")
        # Кнопки окна: Выполнить / Копировать (неактивен) / Закрыть
        buttons = self._find_button_texts(dialog)
        self.assertIn("Выполнить", buttons)
        self.assertIn("Копировать результат", buttons)
        self.assertIn("Закрыть", buttons)

    def test_execute_delivers_result_to_dialog(self) -> None:
        """Сценарий 8: «Выполнить» → результат отображается, статус «Готово»."""
        dialog = self.panel.open_call_dialog(self._git_tool())
        self.addCleanup(dialog.destroy)
        self.root.update()
        dialog._fields[1]["var"].set("5")
        dialog.run_btn.invoke()
        self.assertEqual(self._calls, [{
            "tool": "get_git_log",
            "args": {"repo_path": str(PROJECT_ROOT / "some" / "repo"), "limit": 5},
        }])
        self._deliver('[{"hash": "abc", "author": "a", "date": "2026-01-01", "message": "m"}]')
        _sync_dialog(dialog)
        self.root.update()
        self.assertIn('"abc"', dialog.get_result_text())
        self.assertIn("Готово", dialog.status_label.cget("text"))
        self.assertEqual(str(dialog.run_btn.cget("state")), "normal")
        self.assertEqual(str(dialog.copy_btn.cget("state")), "normal")

    def test_worker_error_shows_messagebox_and_red_status(self) -> None:
        dialog = self.panel.open_call_dialog(self._git_tool())
        self.addCleanup(dialog.destroy)
        self.root.update()
        dialog.run_btn.invoke()
        with mock.patch("src.mcp_panel.messagebox") as mb:
            self._deliver(error=MCPConnectionError("Сервер не ответил"))
            _sync_dialog(dialog)
            mb.showerror.assert_called_once()
        self.assertIn("Ошибка:", dialog.status_label.cget("text"))
        # После ошибки кнопку можно нажать снова
        self.assertEqual(str(dialog.run_btn.cget("state")), "normal")

    def test_missing_required_field_blocks_call(self) -> None:
        """Валидация: пустой обязательный repo_path — вызов не запускается."""
        dialog = self.panel.open_call_dialog(self._git_tool())
        self.addCleanup(dialog.destroy)
        self.root.update()
        dialog._fields[0]["var"].set("")
        with mock.patch("src.mcp_panel.messagebox") as mb:
            dialog.run_btn.invoke()
            mb.showwarning.assert_called_once()
        self.assertEqual(self._calls, [])
        self.assertTrue(dialog._busy is False)

    def test_bad_integer_blocks_call(self) -> None:
        """Валидация: limit="abc" — предупреждение, вызов не запускается."""
        dialog = self.panel.open_call_dialog(self._git_tool())
        self.addCleanup(dialog.destroy)
        self.root.update()
        dialog._fields[1]["var"].set("abc")
        with mock.patch("src.mcp_panel.messagebox") as mb:
            dialog.run_btn.invoke()
            mb.showwarning.assert_called_once()
        self.assertEqual(self._calls, [])

    def test_empty_optional_field_is_omitted(self) -> None:
        """Пустое необязательное поле не передаётся (сработает default)."""
        dialog = self.panel.open_call_dialog(self._git_tool())
        self.addCleanup(dialog.destroy)
        self.root.update()
        dialog._fields[1]["var"].set("")
        dialog.run_btn.invoke()
        self.assertEqual(
            self._calls[0]["args"],
            {"repo_path": str(PROJECT_ROOT / "some" / "repo")},
        )
        self._deliver("[]")
        _sync_dialog(dialog)

    def test_tool_without_schema_has_empty_arguments(self) -> None:
        from src.mcp_models import MCPTool

        tool = MCPTool(name="no_params", description="Без параметров")
        dialog = self.panel.open_call_dialog(tool)
        self.addCleanup(dialog.destroy)
        self.root.update()
        self.assertEqual(dialog._fields, [])
        dialog.run_btn.invoke()
        self.assertEqual(self._calls, [{"tool": "no_params", "args": {}}])
        self._deliver('"ok"')
        _sync_dialog(dialog)
        self.assertIn("ok", dialog.get_result_text())

    def test_destroy_during_pending_call_is_safe(self) -> None:
        """Закрытие диалога во время выполнения не падает."""
        dialog = self.panel.open_call_dialog(self._git_tool())
        self.root.update()
        dialog.run_btn.invoke()  # воркер «висит» (ответ не подан)
        dialog.destroy()
        self.root.update()  # after-задача отменена, исключений нет
        self._deliver("[]")  # поздний ответ уходит в брошенную очередь

    def test_ui_stays_responsive_during_call(self) -> None:
        """Сценарий 9: пока вызов идёт, поток событий Tk жив."""
        dialog = self.panel.open_call_dialog(self._git_tool())
        self.addCleanup(dialog.destroy)
        self.root.update()
        dialog.run_btn.invoke()  # воркер «висит»
        markers = []
        self.root.after(10, lambda: markers.append("tick"))
        self.root.update()
        deadline = time.monotonic() + 2.0
        while not markers and time.monotonic() < deadline:
            self.root.update()
        self.assertEqual(markers, ["tick"])
        self.assertIn("Выполняется", dialog.status_label.cget("text"))
        self._deliver('{"ok": true}')
        _sync_dialog(dialog)
        self.root.update()
        self.assertIn("Готово", dialog.status_label.cget("text"))

    def test_parse_argument_types(self) -> None:
        """_parse_argument: integer/number/boolean/string и ошибки."""
        from src.mcp_panel import _parse_argument

        self.assertEqual(_parse_argument("5", "integer"), 5)
        self.assertEqual(_parse_argument(" 7 ", "integer"), 7)
        self.assertEqual(_parse_argument("2.5", "number"), 2.5)
        self.assertIs(_parse_argument("true", "boolean"), True)
        self.assertIs(_parse_argument("Нет", "boolean"), False)
        self.assertEqual(_parse_argument("C:\\repo", "string"), "C:\\repo")
        self.assertEqual(_parse_argument("x", "array"), "x")  # неизвестный тип
        with self.assertRaises(ValueError):
            _parse_argument("abc", "integer")
        with self.assertRaises(ValueError):
            _parse_argument("maybe", "boolean")

    # ---- вспомогательные -------------------------------------------------

    def _find_button_texts(self, widget) -> list:
        found = []

        def _walk(w) -> None:
            for child in w.winfo_children():
                if isinstance(child, tk.Button):
                    found.append(str(child.cget("text")))
                _walk(child)

        _walk(widget)
        return found


@unittest.skipUnless(tk is not None and MCP_SDK and GIT_AVAILABLE,
                     "нужны tkinter, SDK mcp и git")
class PanelRealCallIntegrationTest(_RepoCase):
    """Сквозной вызов из UI: кнопка → диалог → настоящий сервер → результат."""

    root = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = gui_helpers.get_root()
        if cls.root is None:
            raise unittest.SkipTest("графическая среда недоступна (tkinter/Tk)")

    def setUp(self) -> None:
        super().setUp()
        gui_helpers.reset_children(self.root)
        self.repo = self.make_repo("ui_repo", commits=3)
        from src.mcp_panel import MCPPanel

        self.panel = MCPPanel(
            self.root,
            settings={
                "enabled": True,
                "server_command": _server_command(),
                "timeout": 30.0,
                "log_path": self.make_dir("ui_logs") / "mcp.log",
                "git_repo_path": str(self.repo),
            },
        )
        self.root.update()

    def tearDown(self) -> None:
        for dialog in list(self.root.winfo_children()):
            if isinstance(dialog, tk.Toplevel):
                try:
                    dialog.destroy()
                except tk.TclError:
                    pass
        try:
            self.panel.destroy()
        except tk.TclError:
            pass
        self.root.update()
        super().tearDown()

    def _wait_until(self, predicate, message: str, timeout: float = 20.0) -> None:
        """Крутит цикл событий Tk, пока не выполнится условие."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.root.update()
            if predicate():
                return
            time.sleep(0.02)
        raise AssertionError(message)

    def test_call_get_git_log_from_panel_dialog(self) -> None:
        """Сценарий 8 целиком: карточка → «Вызвать» → реальный вызов."""
        from src.mcp_models import MCPTool

        tool = MCPTool(
            name="get_git_log",
            description="Возвращает последние N коммитов в репозитории.",
            input_schema={
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string"},
                    "limit": {"default": 10, "type": "integer"},
                },
                "required": ["repo_path"],
            },
        )
        dialog = self.panel.open_call_dialog(tool)
        self.addCleanup(dialog.destroy)
        self.root.update()
        self.assertEqual(dialog._fields[0]["var"].get(), str(self.repo))
        dialog._fields[1]["var"].set("2")
        dialog.run_btn.invoke()
        self._wait_until(
            lambda: not dialog._busy and dialog._queue.empty(),
            "вызов через панель не завершился",
        )
        self.root.update()
        raw = dialog.get_result_text()
        result = _unwrap(json.loads(raw))
        self.assertEqual(len(result), 2)
        self.assertEqual(set(result[0]), {"hash", "author", "date", "message"})
        self.assertIn("Готово", dialog.status_label.cget("text"))


if __name__ == "__main__":
    unittest.main()
