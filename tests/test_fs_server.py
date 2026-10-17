"""Тесты MCP-сервера файловой системы (fs_server.py, ТЗ ai_1_35).

Проверяются: инструменты на реальных файлах (read_file /
read_multiple_files / write_file / search_files / search_within_files /
list_directory / tree — офлайн: ALLOWED_ROOT подменяется на фикстуру),
защита ALLOWED_ROOT (traversal, абсолютные пути наружу, «соседний»
каталог с похожим префиксом), настройки config (FS_*), лаунчер, живой
HTTP-сервер (tools/call через ReminderMCPClient — как дёргает ядро).

Запуск из корня проекта:
    python -m unittest tests.test_fs_server -v
"""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import fs_server
from src.config import get_fs_settings
from src.fs_launcher import FsServerLauncher
from src.mcp_client import MCP_SDK_AVAILABLE
from src.reminder_client import HTTP_AVAILABLE, ReminderMCPClient

try:  # pragma: no cover — зависит от окружения
    from mcp import ClientSession  # noqa: F401

    MCP_SDK = MCP_SDK_AVAILABLE and HTTP_AVAILABLE
except ImportError:  # pragma: no cover
    MCP_SDK = False

FS_SERVER = str(PROJECT_ROOT / "fs_server.py")

FS_TOOLS = {"read_file", "read_multiple_files", "write_file", "search_files",
            "search_within_files", "list_directory", "tree"}


def _free_port() -> int:
    """Свободный TCP-порт на 127.0.0.1 (для живого сервера в тестах)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_fixture(root: Path) -> None:
    """Проект-фикстура: код, документация, venv (должен пропускаться)."""
    src = root / "src"
    src.mkdir(exist_ok=True)
    (src / "agent.py").write_text(
        "class DeepSeekAgent:\n"
        "    def send_message(self, text):\n"
        "        return DeepSeekAgent\n",
        encoding="utf-8")
    (src / "main.py").write_text(
        "from agent import DeepSeekAgent\n"
        "agent = DeepSeekAgent()\n",
        encoding="utf-8")
    docs = root / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "adr").mkdir(exist_ok=True)
    (docs / "adr" / "0001-example.md").write_text(
        "# ADR-0001: Пример\n\nКонтекст: шаблон для новых ADR.\n",
        encoding="utf-8")
    venv = root / "venv" / "lib"
    venv.mkdir(parents=True, exist_ok=True)
    (venv / "junk.py").write_text("DeepSeekAgent = 'не проект'\n",
                                  encoding="utf-8")
    (root / "notes.txt").write_text("обычная заметка\n", encoding="utf-8")


class FsToolTest(unittest.TestCase):
    """Инструменты fs-сервера (офлайн: корень подменяется на фикстуру)."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        _make_fixture(self.tmp)
        patcher = mock.patch.object(fs_server, "ALLOWED_ROOT", self.tmp)
        self.addCleanup(patcher.stop)
        patcher.start()

    # ------------------------------------------------------------------
    # Чтение / запись
    # ------------------------------------------------------------------
    def test_read_file_relative(self) -> None:
        text = fs_server.read_file("src/agent.py")
        self.assertIn("class DeepSeekAgent", text)

    def test_read_file_missing_is_error_json(self) -> None:
        result = fs_server.read_file("src/nope.py")
        self.assertIn("error", json.loads(result))

    def test_read_file_long_is_capped(self) -> None:
        """Большой файл обрезается (защита контекста LLM)."""
        big = self.tmp / "big.md"
        big.write_text("x" * (fs_server.MAX_READ_CHARS + 5000),
                       encoding="utf-8")
        text = fs_server.read_file("big.md")
        self.assertLess(len(text), fs_server.MAX_READ_CHARS + 200)
        self.assertIn("обрезан", text)

    def test_write_file_creates_directories(self) -> None:
        """write_file с parent.mkdir — новые каталоги создаются (ТЗ)."""
        message = fs_server.write_file(
            "docs/adr/0002-new.md", "# ADR-0002")
        self.assertIn("Файл создан", message)
        self.assertEqual(
            (self.tmp / "docs/adr/0002-new.md").read_text(
                encoding="utf-8"),
            "# ADR-0002")

    def test_write_file_overwrites(self) -> None:
        fs_server.write_file("notes.txt", "обновлено")
        self.assertEqual(
            (self.tmp / "notes.txt").read_text(encoding="utf-8"),
            "обновлено")

    def test_read_multiple_files(self) -> None:
        result = fs_server.read_multiple_files(
            ["src/agent.py", "src/missing.py"])
        self.assertIn("class DeepSeekAgent", result["src/agent.py"])
        self.assertIn("error", json.loads(result["src/missing.py"]))

    def test_read_multiple_files_batch_limit(self) -> None:
        paths = [f"src/f{i}.py" for i in range(
            fs_server.MAX_BATCH_FILES + 1)]
        result = fs_server.read_multiple_files(paths)
        self.assertIn("error", result)

    # ------------------------------------------------------------------
    # Поиск
    # ------------------------------------------------------------------
    def test_search_files_glob(self) -> None:
        found = fs_server.search_files("*.md")
        self.assertEqual(found, [str(Path("docs/adr/0001-example.md"))])

    def test_search_within_files_finds_usages(self) -> None:
        """Сценарий ТЗ: поиск всех мест использования класса."""
        hits = fs_server.search_within_files("DeepSeekAgent")
        files = {hit["file"] for hit in hits}
        # venv пропущен, проектные файлы найдены
        self.assertEqual(files, {str(Path("src/agent.py")),
                                 str(Path("src/main.py"))})
        for hit in hits:
            self.assertIn("DeepSeekAgent", hit["content"])
            self.assertGreater(hit["line"], 0)

    def test_search_within_files_max_results(self) -> None:
        hits = fs_server.search_within_files("DeepSeekAgent",
                                             max_results=1)
        self.assertEqual(len(hits), 1)

    def test_search_within_default_cap_keeps_context_small(self) -> None:
        """Дефолт 30 записей и сниппет ≤120: результат не раздувает
        контекст LLM (прецедент: 5 поисков × 100 вхождений → запрос
        325K символов → пустые ответы)."""
        target = self.tmp / "wide.md"
        target.write_text(
            "\n".join(f"needle строка {i} " + "z" * 200
                      for i in range(50)),
            encoding="utf-8")
        hits = fs_server.search_within_files("needle")
        self.assertEqual(len(hits), fs_server.DEFAULT_SEARCH_RESULTS)
        self.assertTrue(all(len(hit["content"]) <=
                            fs_server.SEARCH_SNIPPET_CHARS
                            for hit in hits))

    def test_search_within_files_empty_substring(self) -> None:
        self.assertIn("error", fs_server.search_within_files("  ")[0])

    # ------------------------------------------------------------------
    # Навигация
    # ------------------------------------------------------------------
    def test_list_directory_marks_dirs(self) -> None:
        names = fs_server.list_directory(".")
        self.assertIn("docs/", names)
        self.assertIn("src/", names)
        self.assertIn("notes.txt", names)

    def test_tree_json(self) -> None:
        tree = json.loads(fs_server.tree(".", depth=2))
        self.assertEqual(tree["src"]["agent.py"], "file")
        self.assertEqual(tree["docs"]["adr"]["0001-example.md"], "file")
        self.assertNotIn("venv", tree)
        # Глубина исчерпана: нераскрытый каталог помечается "dir"
        shallow = json.loads(fs_server.tree(".", depth=1))
        self.assertEqual(shallow["docs"]["adr"], "dir")

    def test_tree_depth_cap(self) -> None:
        tree = json.loads(fs_server.tree(".", depth=99))
        self.assertEqual(tree["docs"]["adr"]["0001-example.md"], "file")


class FsSecurityTest(unittest.TestCase):
    """Защита ALLOWED_ROOT (ТЗ «ЧЕГО НЕ ДЕЛАТЬ»: доступ вне корня)."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "project"
        self.root.mkdir()
        _make_fixture(self.root)
        # «Соседний» каталог с похожим префиксом — НЕ должен быть доступен
        evil = self.tmp / "project-evil"
        evil.mkdir()
        (evil / "secret.txt").write_text("секрет", encoding="utf-8")
        patcher = mock.patch.object(fs_server, "ALLOWED_ROOT", self.root)
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_traversal_blocked(self) -> None:
        result = fs_server.read_file("../../etc/passwd")
        self.assertIn("Доступ запрещён", result)

    def test_absolute_outside_blocked(self) -> None:
        result = fs_server.read_file(str(self.tmp / "project-evil"
                                         / "secret.txt"))
        self.assertIn("Доступ запрещён", result)

    def test_sibling_prefix_directory_blocked(self) -> None:
        """«project-evil» начинается с «project» — но это другой каталог."""
        result = fs_server.read_file("../project-evil/secret.txt")
        self.assertIn("Доступ запрещён", result)

    def test_write_outside_blocked(self) -> None:
        result = fs_server.write_file("../outside.md", "взлом")
        self.assertIn("error", json.loads(result))
        self.assertFalse((self.tmp / "outside.md").exists())

    def test_search_outside_blocked(self) -> None:
        hits = fs_server.search_within_files("секрет",
                                             directory="../project-evil")
        self.assertIn("error", hits[0])

    def test_root_itself_allowed(self) -> None:
        names = fs_server.list_directory(".")
        self.assertIn("src/", names)


class FsSettingsTest(unittest.TestCase):
    """Настройки config (FS_*)."""

    def test_defaults(self) -> None:
        settings = get_fs_settings({
            "FS_ENABLED": True, "FS_SERVER_HOST": "127.0.0.1",
            "FS_SERVER_PORT": 8005,
        })
        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["url"], "http://127.0.0.1:8005/mcp")

    def test_unsafe_host_rejected(self) -> None:
        settings = get_fs_settings({
            "FS_ENABLED": True, "FS_SERVER_HOST": "0.0.0.0",
            "FS_SERVER_PORT": 8005,
        })
        self.assertEqual(settings["host"], "127.0.0.1")

    def test_invalid_port_falls_back(self) -> None:
        settings = get_fs_settings({
            "FS_ENABLED": True, "FS_SERVER_HOST": "127.0.0.1",
            "FS_SERVER_PORT": "не число",
        })
        self.assertEqual(settings["port"], 8005)


class FsLauncherTest(unittest.TestCase):
    """Лаунчер fs-сервера (автозапуск в отдельном окне)."""

    def test_defaults(self) -> None:
        launcher = FsServerLauncher()
        self.assertTrue(str(launcher.server_path).endswith("fs_server.py"))
        self.assertEqual(launcher.label, "FS-сервер")
        self.assertTrue(str(launcher.server_url).endswith(":8005/mcp"))

    def test_custom_port(self) -> None:
        launcher = FsServerLauncher(port=_free_port())
        self.assertIn("/mcp", launcher.server_url)


@unittest.skipUnless(MCP_SDK, "требуется SDK mcp")
class FsServerHTTPTest(unittest.TestCase):
    """Живой сервер на HTTP: tools/list и tools/call (как из ядра)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp())
        cls.root = cls.tmp / "project"
        cls.root.mkdir()
        _make_fixture(cls.root)
        cls.port = _free_port()
        cls.url = f"http://127.0.0.1:{cls.port}/mcp"
        cls.log_file = cls.tmp / "fs_http_server.log"
        cls.process = subprocess.Popen(
            [
                sys.executable, "-u", FS_SERVER,
                "--port", str(cls.port),
                "--root", str(cls.root),
            ],
            stdout=open(cls.log_file, "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        cls.client = ReminderMCPClient(cls.url, label="FS-сервер")
        deadline = time.monotonic() + 20.0
        launcher = FsServerLauncher(port=cls.port)
        while time.monotonic() < deadline:
            if launcher.is_server_running():
                return
            if cls.process.poll() is not None:
                raise RuntimeError(
                    f"FS-сервер упал при старте, код "
                    f"{cls.process.returncode}. Лог: {cls.log_file}"
                )
            time.sleep(0.3)
        raise RuntimeError("FS-сервер не поднялся за 20 секунд")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.process.terminate()
        try:
            cls.process.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover
            cls.process.kill()

    @staticmethod
    def _unwrap(payload_text: str):
        payload = json.loads(payload_text)
        if isinstance(payload, dict) and set(payload) == {"result"}:
            return payload["result"]
        return payload

    def _call(self, name: str, arguments=None):
        return asyncio.run(self.client.call_tool(name, arguments or {}))

    def test_list_tools_returns_seven(self) -> None:
        """tools/list возвращает ровно 7 инструментов ТЗ."""
        tools = asyncio.run(self.client.list_tools())
        self.assertEqual({t.name for t in tools}, FS_TOOLS)

    def test_write_and_read_roundtrip_over_http(self) -> None:
        write = self._call("write_file", {
            "path": "docs/adr/0002-generated.md",
            "content": "# ADR-0002: Сгенерирован по HTTP",
        })
        self.assertIn("Файл создан", write)
        text = self._call("read_file",
                          {"path": "docs/adr/0002-generated.md"})
        self.assertIn("ADR-0002", self._unwrap(text))

    def test_search_over_http(self) -> None:
        raw = self._call("search_within_files",
                         {"substring": "DeepSeekAgent", "directory": "src"})
        hits = self._unwrap(raw)
        self.assertTrue(hits)
        self.assertTrue(all("DeepSeekAgent" in hit["content"]
                            for hit in hits))

    def test_traversal_over_http_blocked(self) -> None:
        raw = self._call("read_file", {"path": "../../outside.txt"})
        self.assertIn("Доступ запрещён", raw)


if __name__ == "__main__":
    unittest.main()
