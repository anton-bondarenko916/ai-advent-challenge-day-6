"""Тесты MCP-сервера CRM (crm_server.py, ТЗ ai_1_34).

Проверяются: инструменты на JSON-данных (get_user / get_user_tickets /
get_user_context / list_users, офлайн: путь к данным подменяется),
настройки config (CRM_*), лаунчер, живой HTTP-сервер (tools/call через
ReminderMCPClient — как дёргает SupportPipeline).

Запуск из корня проекта:
    python -m unittest tests.test_crm_server -v
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import crm_server
from src.config import get_crm_settings
from src.crm_launcher import CrmServerLauncher
from src.mcp_client import MCP_SDK_AVAILABLE
from src.reminder_client import HTTP_AVAILABLE, ReminderMCPClient

try:  # pragma: no cover — зависит от окружения
    from mcp import ClientSession  # noqa: F401

    MCP_SDK = MCP_SDK_AVAILABLE and HTTP_AVAILABLE
except ImportError:  # pragma: no cover
    MCP_SDK = False

CRM_SERVER = str(PROJECT_ROOT / "crm_server.py")

FIXTURE_DATA = {
    "users": [
        {
            "id": 42,
            "name": "Иван Петров",
            "email": "ivan@example.com",
            "status": "active",
            "created_at": "2024-01-15",
            "tickets": [
                {
                    "id": 101,
                    "subject": "Не работает авторизация",
                    "status": "open",
                    "created_at": "2026-10-14",
                    "priority": "high",
                },
            ],
        },
        {
            "id": 43,
            "name": "Мария Сидорова",
            "email": "maria@example.com",
            "status": "pending_verification",
            "created_at": "2025-06-20",
            "tickets": [],
        },
    ],
}


def _write_fixture(root: Path) -> Path:
    """Фикстура CRM (пользователи 42/43 из ТЗ) во временном каталоге."""
    path = root / f"crm_{uuid.uuid4().hex[:8]}.json"
    path.write_text(json.dumps(FIXTURE_DATA, ensure_ascii=False),
                    encoding="utf-8")
    return path


def _free_port() -> int:
    """Свободный TCP-порт на 127.0.0.1 (для живого сервера в тестах)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class CrmToolTest(unittest.TestCase):
    """Инструменты CRM на JSON-данных (офлайн: путь подменяется)."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.data_path = _write_fixture(self.tmp)
        patcher = mock_patch_data(self.data_path)
        self.addCleanup(patcher.stop)
        patcher.start()

    # -- ТЗ: get_user_context(42) возвращает профиль + тикеты --

    def test_get_user_context_returns_profile_and_tickets(self) -> None:
        context = crm_server.get_user_context(42)
        self.assertEqual(context["user"]["name"], "Иван Петров")
        self.assertEqual(context["user"]["email"], "ivan@example.com")
        self.assertEqual(context["user"]["status"], "active")
        self.assertEqual(len(context["tickets"]), 1)
        ticket = context["tickets"][0]
        self.assertEqual(ticket["id"], 101)
        self.assertEqual(ticket["subject"], "Не работает авторизация")
        self.assertEqual(ticket["priority"], "high")

    def test_get_user_context_unknown_user(self) -> None:
        context = crm_server.get_user_context(999)
        self.assertIn("error", context)
        self.assertIn("999", context["error"])

    def test_get_user_context_accepts_string_id(self) -> None:
        """ID приходит из UI строкой — сравнение как int."""
        context = crm_server.get_user_context("42")
        self.assertEqual(context["user"]["id"], 42)

    def test_get_user_profile(self) -> None:
        user = crm_server.get_user(43)
        self.assertEqual(user["name"], "Мария Сидорова")
        self.assertEqual(user["status"], "pending_verification")
        self.assertEqual(user["created_at"], "2025-06-20")
        self.assertNotIn("tickets", user)

    def test_get_user_unknown(self) -> None:
        self.assertIn("error", crm_server.get_user(1000))

    def test_get_user_tickets(self) -> None:
        tickets = crm_server.get_user_tickets(42)
        self.assertEqual([t["id"] for t in tickets], [101])
        self.assertEqual(crm_server.get_user_tickets(43), [])
        self.assertEqual(crm_server.get_user_tickets(999), [])

    def test_list_users_for_dropdown(self) -> None:
        users = crm_server.list_users()
        self.assertEqual([u["id"] for u in users], [42, 43])
        self.assertEqual(users[0]["name"], "Иван Петров")
        self.assertNotIn("tickets", users[0])

    def test_missing_data_file_is_empty_crm(self) -> None:
        """Отсутствующий JSON — пустой CRM, инструменты не падают."""
        patcher = mock_patch_data(self.tmp / "no_such.json")
        self.addCleanup(patcher.stop)
        patcher.start()
        self.assertEqual(crm_server.list_users(), [])
        self.assertIn("error", crm_server.get_user_context(42))

    def test_broken_data_file_is_empty_crm(self) -> None:
        """Битый JSON — пустой CRM, сервер не падает."""
        broken = self.tmp / "broken.json"
        broken.write_text("{не json", encoding="utf-8")
        patcher = mock_patch_data(broken)
        self.addCleanup(patcher.stop)
        patcher.start()
        self.assertEqual(crm_server.list_users(), [])


def mock_patch_data(path: Path):
    """Подменяет путь к JSON-данным CRM (офлайн-тесты)."""
    from unittest import mock

    return mock.patch.object(crm_server, "CRM_DATA_PATH", path)


class CrmSettingsTest(unittest.TestCase):
    """Настройки config (CRM_*)."""

    def test_defaults(self) -> None:
        settings = get_crm_settings({
            "CRM_ENABLED": True, "CRM_SERVER_HOST": "127.0.0.1",
            "CRM_SERVER_PORT": 8004,
        })
        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["url"], "http://127.0.0.1:8004/mcp")

    def test_unsafe_host_rejected(self) -> None:
        settings = get_crm_settings({
            "CRM_ENABLED": True, "CRM_SERVER_HOST": "0.0.0.0",
            "CRM_SERVER_PORT": 8004,
        })
        self.assertEqual(settings["host"], "127.0.0.1")

    def test_invalid_port_falls_back(self) -> None:
        settings = get_crm_settings({
            "CRM_ENABLED": True, "CRM_SERVER_HOST": "127.0.0.1",
            "CRM_SERVER_PORT": "не число",
        })
        self.assertEqual(settings["port"], 8004)


class CrmLauncherTest(unittest.TestCase):
    """Лаунчер CRM-сервера (автозапуск в отдельном окне)."""

    def test_defaults(self) -> None:
        launcher = CrmServerLauncher()
        self.assertTrue(str(launcher.server_path).endswith("crm_server.py"))
        self.assertEqual(launcher.label, "CRM-сервер")
        self.assertTrue(str(launcher.server_url).endswith(":8004/mcp"))

    def test_custom_port(self) -> None:
        port = _free_port()
        launcher = CrmServerLauncher(port=port)
        self.assertEqual(
            launcher.server_url, f"http://127.0.0.1:{port}/mcp")


@unittest.skipUnless(MCP_SDK, "требуется SDK mcp")
class CrmServerHTTPTest(unittest.TestCase):
    """Живой сервер на HTTP: tools/call через ReminderMCPClient."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp())
        cls.data_path = _write_fixture(cls.tmp)
        cls.port = _free_port()
        cls.url = f"http://127.0.0.1:{cls.port}/mcp"
        cls.log_file = cls.tmp / "crm_http_server.log"
        cls.process = subprocess.Popen(
            [
                sys.executable, "-u", CRM_SERVER,
                "--port", str(cls.port),
                "--data", str(cls.data_path),
            ],
            stdout=open(cls.log_file, "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,  # WinError 32 без DEVNULL
        )
        cls.client = ReminderMCPClient(cls.url, label="CRM-сервер")
        # Ждём, пока HTTP-порт начнёт отвечать (старт uvicorn ~1-3 с)
        deadline = time.monotonic() + 20.0
        launcher = CrmServerLauncher(port=cls.port)
        while time.monotonic() < deadline:
            if launcher.is_server_running():
                return
            if cls.process.poll() is not None:
                raise RuntimeError(
                    f"CRM-сервер упал при старте, код "
                    f"{cls.process.returncode}. Лог: {cls.log_file}"
                )
            time.sleep(0.3)
        raise RuntimeError("CRM-сервер не поднялся за 20 секунд")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.process.terminate()
        try:
            cls.process.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover
            cls.process.kill()

    @staticmethod
    def _unwrap(payload_text: str):
        """Снимает обёртку annotated structured content FastMCP."""
        payload = json.loads(payload_text)
        if isinstance(payload, dict) and set(payload) == {"result"}:
            return payload["result"]
        return payload

    def _call(self, name: str, arguments=None):
        return asyncio.run(self.client.call_tool(name, arguments or {}))

    def test_get_user_context_over_http(self) -> None:
        """Сценарий /support: get_user_context(42) по HTTP."""
        context = self._unwrap(self._call(
            "get_user_context", {"user_id": 42}))
        self.assertEqual(context["user"]["name"], "Иван Петров")
        self.assertEqual(context["tickets"][0]["id"], 101)

    def test_list_users_over_http(self) -> None:
        users = self._unwrap(self._call("list_users"))
        self.assertEqual([u["id"] for u in users], [42, 43])

    def test_unknown_user_over_http(self) -> None:
        context = self._unwrap(self._call(
            "get_user_context", {"user_id": 999}))
        self.assertIn("error", context)


if __name__ == "__main__":
    unittest.main()
