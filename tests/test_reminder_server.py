"""Тесты MCP-сервера «Smart Reminder» (reminder_server.py, HTTP + tool calling).

Покрывают сценарии приёмки ТЗ:

 1. Сервер запускается: ``python reminder_server.py`` → HTTP-порт отвечает;
 2. list_tools() возвращает 5 инструментов (add/get_due/get_all/delete/stats);
 3. add_reminder("кофе", 1, False) → запись в SQLite;
 4. Одноразовое срабатывание: get_due_reminders возвращает напоминание,
    is_active = 0 (повторный вызов не возвращает его снова);
 5. Повторяющееся: next_due_at сдвигается на интервал, is_active = 1;
 6. LLM tool calling: агент получает tools, вызывает add_reminder через
    MCP-клиент и возвращает результат LLM (офлайн — фейковый API-клиент);
 7. SSE-уведомление: планировщик сервера рассылает REMINDER: …, слушатель
    клиента получает его, UI показывает сообщение в чате;
 8. Автозапуск: launcher стартует сервер (команда/флаги консоли);
 9. Не перезапускается: если порт занят, launch() не дублирует процесс;
10. Сервер живёт после закрытия: stop() — no-op, сервер продолжает отвечать;
11. get_stats() возвращает корректные агрегаты;
12. delete_reminder(id) удаляет запись.

Юнит-часть работает с impl-функциями сервера на временной БД (быстро, без
сети). HTTP-интеграция поднимает сервер подпроцессом на свободном порту с
``--check-seconds 2`` (ускоренный тик планировщика для тестов) и требует
SDK ``mcp`` — иначе тесты пропускаются.

Запуск из корня проекта:
    python -m unittest tests.test_reminder_server -v
"""

from __future__ import annotations

import asyncio
import json
import queue
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.config import DEFAULTS, PROJECT_ROOT, get_reminder_settings
from src.mcp_models import MCPTool

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

import reminder_server
from src.reminder_client import (
    ReminderMCPClient,
    ReminderNotificationListener,
    extract_call_text,
)
from src.reminder_launcher import ReminderServerLauncher

REMINDER_SERVER = str(PROJECT_ROOT / "reminder_server.py")

# Базовый каталог временных файлов (внутри репозитория — см. .gitignore)
_FIXTURE_BASE = Path(__file__).resolve().parent / "tmp_reminder"

TOOLS = [
    "add_reminder",
    "get_due_reminders",
    "get_all_reminders",
    "delete_reminder",
    "get_stats",
]


# ---------------------------------------------------------------------------
# Юнит-тесты impl-функций сервера (временная БД, без сети)
# ---------------------------------------------------------------------------


class ReminderImplTest(unittest.TestCase):
    """impl-функции reminder_server: переходы состояний в SQLite (ТЗ 3-5, 11, 12)."""

    def setUp(self) -> None:
        _FIXTURE_BASE.mkdir(parents=True, exist_ok=True)
        self.db_path = _FIXTURE_BASE / f"impl_{uuid.uuid4().hex[:8]}.db"
        patcher = mock.patch.object(reminder_server, "DB_PATH", self.db_path)
        patcher.start()
        self.addCleanup(patcher.stop)
        reminder_server.init_db()

    def tearDown(self) -> None:
        self.db_path.unlink(missing_ok=True)

    # -- ТЗ 3: создание напоминания --

    def test_add_reminder_writes_row(self) -> None:
        """add_reminder пишет запись и возвращает id/next_due_at."""
        result = reminder_server._add_reminder_impl("кофе", 5, False)
        self.assertNotIn("error", result)
        self.assertEqual(result["text"], "кофе")
        self.assertEqual(result["interval_minutes"], 5)
        self.assertFalse(result["repeat"])
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT text, interval_minutes, is_active "
                            "FROM reminders").fetchall()
        conn.close()
        self.assertEqual(rows, [("кофе", 5, 1)])

    def test_add_reminder_validation(self) -> None:
        """Некорректные аргументы дают {"error": ...} без записи в БД."""
        self.assertIn("error", reminder_server._add_reminder_impl("", 5))
        self.assertIn("error", reminder_server._add_reminder_impl("x", 0))
        self.assertIn("error", reminder_server._add_reminder_impl("x", -3))
        conn = sqlite3.connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0]
        conn.close()
        self.assertEqual(count, 0)

    # -- ТЗ 4: одноразовое срабатывание --

    def test_one_shot_reminder_fires_once(self) -> None:
        """Одноразовое: из-за срока → get_due_reminders → is_active = 0."""
        reminder_server._add_reminder_impl("выпить кофе", 1, False)
        # Сдвигаем next_due_at в прошлое (не ждём минуту)
        self._make_due(1)

        due = reminder_server._get_due_reminders_impl()
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0]["text"], "выпить кофе")
        # Транзакционность: второй вызов НЕ возвращает то же напоминание
        self.assertEqual(reminder_server._get_due_reminders_impl(), [])

        conn = sqlite3.connect(self.db_path)
        is_active, count = conn.execute(
            "SELECT is_active, trigger_count FROM reminders"
        ).fetchone()
        log_rows = conn.execute("SELECT COUNT(*) FROM reminder_log").fetchone()[0]
        conn.close()
        self.assertEqual((is_active, count, log_rows), (0, 1, 1))

    # -- ТЗ 5: повторяющееся срабатывание --

    def test_repeat_reminder_reschedules(self) -> None:
        """Повторяющееся: next_due_at сдвигается на интервал, остаётся активным."""
        result = reminder_server._add_reminder_impl("пить воду", 10, True)
        self._make_due(1)

        due = reminder_server._get_due_reminders_impl()
        self.assertEqual(len(due), 1)

        conn = sqlite3.connect(self.db_path)
        is_active, next_due, count = conn.execute(
            "SELECT is_active, next_due_at, trigger_count FROM reminders"
        ).fetchone()
        conn.close()
        self.assertEqual(is_active, 1)
        self.assertEqual(count, 1)
        # next_due_at ≈ now + 10 минут (сдвиг от момента срабатывания)
        self.assertGreater(next_due, result["next_due_at"])

    # -- ТЗ 10: get_stats --

    def test_stats_aggregates(self) -> None:
        """get_stats считает total/active/triggered_today/avg_interval."""
        reminder_server._add_reminder_impl("a", 10, False)
        reminder_server._add_reminder_impl("b", 30, True)
        reminder_server._add_reminder_impl("c", 20, False)
        self._make_due(1, 3)
        # Тик «планировщика»: все due-напоминания срабатывают
        self.assertEqual(len(reminder_server._get_due_reminders_impl()), 3)

        stats = reminder_server._get_stats_impl()
        self.assertEqual(stats["total"], 3)
        # a и c — одноразовые (деактивировались), b — повторное (активно)
        self.assertEqual(stats["active"], 1)
        # все три сработали (b — повторное) → 3 записи в логе
        self.assertEqual(stats["triggered_today"], 3)
        # средний интервал активных: только "b" (30 минут)
        self.assertEqual(stats["avg_interval"], 30.0)

    # -- ТЗ 11: удаление --

    def test_delete_reminder(self) -> None:
        """delete_reminder удаляет запись, несуществующий id — success=false."""
        created = reminder_server._add_reminder_impl("лишнее", 5, False)
        self.assertTrue(reminder_server._delete_reminder_impl(
            created["id"])["success"])
        self.assertFalse(reminder_server._delete_reminder_impl(999)["success"])
        conn = sqlite3.connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0]
        conn.close()
        self.assertEqual(count, 0)

    # -- Вспомогательные --

    def _make_due(self, minutes_ago: float, upto_id: int = None) -> None:
        """Сдвигает next_due_at активных напоминаний в прошлое."""
        conn = sqlite3.connect(self.db_path)
        from datetime import datetime, timedelta

        past = (datetime.now() - timedelta(minutes=minutes_ago)).isoformat()
        if upto_id is None:
            conn.execute("UPDATE reminders SET next_due_at = ? "
                         "WHERE is_active = 1", (past,))
        else:
            conn.execute(
                "UPDATE reminders SET next_due_at = ? WHERE id <= ?",
                (past, upto_id),
            )
        conn.commit()
        conn.close()


# ---------------------------------------------------------------------------
# HTTP-интеграция: сервер подпроцессом (ТЗ 1, 2, 6, 7, 9, 10, 12)
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Свободный TCP-порт на 127.0.0.1 (для экземпляра сервера в тестах)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@unittest.skipUnless(MCP_SDK, "требуется SDK mcp")
class ReminderServerHTTPTest(unittest.TestCase):
    """Сервер на реальном HTTP: tools/list, tools/call, SSE-уведомления."""

    @classmethod
    def setUpClass(cls) -> None:
        _FIXTURE_BASE.mkdir(parents=True, exist_ok=True)
        cls.db_path = _FIXTURE_BASE / f"http_{uuid.uuid4().hex[:8]}.db"
        cls.port = _free_port()
        cls.url = f"http://127.0.0.1:{cls.port}/mcp"
        cls.log_file = _FIXTURE_BASE / "http_server.log"
        # --check-seconds 2: планировщик тикает каждые 2 секунды —
        # SSE-сценарий проверяется без минутного ожидания
        cls.process = subprocess.Popen(
            [
                sys.executable, "-u", REMINDER_SERVER,
                "--port", str(cls.port),
                "--db", str(cls.db_path),
                "--check-seconds", "2",
            ],
            stdout=open(cls.log_file, "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        cls.client = ReminderMCPClient(cls.url)
        # Ждём, пока HTTP-порт начнёт отвечать (старт uvicorn ~1-3 с)
        deadline = time.monotonic() + 20.0
        launcher = ReminderServerLauncher(port=cls.port)
        while time.monotonic() < deadline:
            if launcher.is_server_running():
                return
            if cls.process.poll() is not None:
                raise RuntimeError(
                    f"Сервер упал при старте, код {cls.process.returncode}. "
                    f"Лог: {cls.log_file}"
                )
            time.sleep(0.3)
        raise RuntimeError("Сервер не поднялся за 20 секунд")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.process.terminate()
        try:
            cls.process.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover
            cls.process.kill()
        try:
            cls.db_path.unlink()
        except OSError:  # pragma: no cover
            pass

    @staticmethod
    def _unwrap(payload):
        """Снимает обёртку annotated structured content FastMCP."""
        if isinstance(payload, dict) and set(payload) == {"result"}:
            return payload["result"]
        return payload

    def _call(self, name: str, arguments=None):
        """Синхронный вызов инструмента по HTTP (для тестов)."""
        return asyncio.run(self.client.call_tool(name, arguments or {}))

    def _insert_due_reminder(self, text: str, repeat: bool = False) -> None:
        """Пишет due-напоминание напрямую в БД сервера (из этого процесса)."""
        from datetime import datetime, timedelta

        conn = sqlite3.connect(
            f"file:{self.db_path}?mode=rw", uri=True, timeout=10.0
        )
        past = (datetime.now() - timedelta(seconds=1)).isoformat()
        conn.execute(
            "INSERT INTO reminders (text, interval_minutes, repeat, "
            "next_due_at) VALUES (?, 1, ?, ?)",
            (text, int(repeat), past),
        )
        conn.commit()
        conn.close()

    # -- ТЗ 1: сервер запускается --

    def test_server_is_up(self) -> None:
        """Сервер отвечает по HTTP (setUpClass уже дождался порта)."""
        self.assertIsNone(self.process.poll())

    # -- ТЗ 2: список инструментов --

    def test_list_tools_returns_five(self) -> None:
        """tools/list возвращает ровно 5 инструментов ТЗ."""
        tools = asyncio.run(self.client.list_tools())
        self.assertEqual([t.name for t in tools], sorted(TOOLS))
        self.assertEqual(len(tools), 5)

    def test_tools_have_schemas_and_descriptions(self) -> None:
        """Каждый инструмент имеет inputSchema и описание для LLM."""
        tools = {t.name: t for t in asyncio.run(self.client.list_tools())}
        add_schema = tools["add_reminder"].input_schema
        self.assertEqual(add_schema["type"], "object")
        self.assertIn("text", add_schema["properties"])
        self.assertIn("interval_minutes", add_schema["properties"])
        self.assertIn("repeat", add_schema["properties"])
        self.assertIn("напоминани", tools["add_reminder"].description.lower())
        self.assertTrue(tools["get_due_reminders"].description)

    # -- ТЗ 3: создание через HTTP --

    def test_add_reminder_via_http(self) -> None:
        """add_reminder по HTTP создаёт запись в SQLite сервера."""
        payload = self._unwrap(json.loads(self._call(
            "add_reminder",
            {"text": "кофе", "interval_minutes": 30, "repeat": False},
        )))
        self.assertEqual(payload["text"], "кофе")
        self.assertEqual(payload["interval_minutes"], 30)

        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT text, interval_minutes FROM reminders"
        ).fetchall()
        conn.close()
        self.assertIn(("кофе", 30), rows)

    def test_add_reminder_error_via_http(self) -> None:
        """Некорректный interval_minutes возвращается как {"error": ...}."""
        payload = self._unwrap(json.loads(self._call(
            "add_reminder", {"text": "x", "interval_minutes": 0},
        )))
        self.assertIn("error", payload)

    # -- ТЗ 4: одноразовое срабатывание через HTTP --

    def test_due_reminder_fires_and_deactivates(self) -> None:
        """get_due_reminders возвращает due-запись; повторный вызов — нет."""
        self._insert_due_reminder("одноразовое")
        result = self._unwrap(json.loads(self._call("get_due_reminders")))
        texts = [r["text"] for r in result]
        self.assertIn("одноразовое", texts)

        # Транзакционность: сработавшее одноразовое не возвращается снова
        again = self._unwrap(json.loads(self._call("get_due_reminders")))
        self.assertNotIn("одноразовое", [r["text"] for r in again])

        conn = sqlite3.connect(self.db_path)
        is_active = conn.execute(
            "SELECT is_active FROM reminders WHERE text = ?", ("одноразовое",)
        ).fetchone()[0]
        conn.close()
        self.assertEqual(is_active, 0)

    # -- ТЗ 5: повторяющееся --

    def test_repeat_reminder_stays_active(self) -> None:
        """Повторяющееся остаётся активным с новым next_due_at."""
        self._insert_due_reminder("повторяющееся", repeat=True)
        self._call("get_due_reminders")

        conn = sqlite3.connect(self.db_path)
        is_active, next_due = conn.execute(
            "SELECT is_active, next_due_at FROM reminders "
            "WHERE text = ?", ("повторяющееся",)
        ).fetchone()
        conn.close()
        self.assertEqual(is_active, 1)
        # next_due_at в будущем (сдвинулся на интервал)
        from datetime import datetime

        self.assertGreater(
            datetime.fromisoformat(next_due), datetime.now()
        )

    # -- ТЗ 10: статистика через HTTP --

    def test_stats_via_http(self) -> None:
        """get_stats по HTTP возвращает согласованные агрегаты."""
        stats = self._unwrap(json.loads(self._call("get_stats")))
        for key in ("total", "active", "triggered_today", "avg_interval"):
            self.assertIn(key, stats)
        self.assertGreaterEqual(stats["total"], 1)
        self.assertGreaterEqual(stats["triggered_today"], 2)

    # -- ТЗ 12: удаление через HTTP --

    def test_delete_reminder_via_http(self) -> None:
        """delete_reminder по HTTP удаляет запись."""
        created = self._unwrap(json.loads(self._call(
            "add_reminder", {"text": "на удаление", "interval_minutes": 5},
        )))
        result = self._unwrap(json.loads(
            self._call("delete_reminder", {"reminder_id": created["id"]})
        ))
        self.assertTrue(result["success"])
        conn = sqlite3.connect(self.db_path)
        count = conn.execute(
            "SELECT COUNT(*) FROM reminders WHERE text = ?", ("на удаление",)
        ).fetchone()[0]
        conn.close()
        self.assertEqual(count, 0)

    # -- ТЗ 7: SSE-уведомление end-to-end --

    def test_sse_notification_delivered(self) -> None:
        """Планировщик рассылает REMINDER: …, слушатель клиента получает.

        Порядок важен: сначала слушатель подключается и регистрирует
        сессию (служебным get_stats), и только потом появляется
        due-напоминание — иначе первый тик планировщика поглотит его
        до регистрации подписчика.
        """
        events: "queue.Queue[tuple]" = queue.Queue()
        listener = ReminderNotificationListener(
            self.url,
            on_notification=lambda text: events.put(("notif", text)),
            on_status=lambda kind, msg: events.put(("status", kind)),
        )
        listener.start()
        # Один cleanup: сначала stop (иначе LIFO-порядок addCleanup вызвал
        # бы join раньше stop, и поток ждал бы конца 60-секундного цикла)
        self.addCleanup(lambda: (listener.stop(), listener.join(timeout=5)))

        # Ждём подключения (+ регистрация сессии на сервере)
        deadline = time.monotonic() + 10.0
        connected = False
        while time.monotonic() < deadline:
            try:
                kind, _payload = events.get(timeout=0.5)
                if kind == "status":
                    connected = True
                    break
            except Exception:  # queue.Empty
                pass
        self.assertTrue(connected, "слушатель не подключился")
        time.sleep(0.5)  # даём завершиться регистрации (get_stats)

        # due-напоминание ПОСЛЕ подключения — следующий тик (≤2 с) разошлёт
        self._insert_due_reminder("sse-проверка")

        deadline = time.monotonic() + 10.0
        notification = None
        while time.monotonic() < deadline:
            try:
                kind, payload = events.get(timeout=0.5)
                if kind == "notif":
                    notification = payload
                    break
            except Exception:  # queue.Empty
                pass
        self.assertIsNotNone(notification, "уведомление не получено")
        self.assertIn("REMINDER:", notification)
        self.assertIn("sse-проверка", notification)

    # -- ТЗ 9: сервер переживает отключение клиента --

    def test_server_outlives_client(self) -> None:
        """После отключения слушателя сервер продолжает отвечать."""
        self._call("get_stats")  # любой вызов — сервер жив
        # (полный сценарий «закрыть приложение — сервер работает» покрывает
        # лаунчер: stop() — no-op, см. ReminderLauncherTest)


# ---------------------------------------------------------------------------
# Лаунчер: автозапуск, защита от дублей, 24/7 (ТЗ 8, 9)
# ---------------------------------------------------------------------------


class ReminderLauncherTest(unittest.TestCase):
    """ReminderServerLauncher: запуск в новом окне и защита от дублей."""

    def test_launch_starts_console_process(self) -> None:
        """launch() стартует python reminder_server.py (новая консоль на Win)."""
        launcher = ReminderServerLauncher(port=_free_port())
        with mock.patch.object(subprocess, "Popen") as popen:
            started, message = launcher.launch()
        self.assertTrue(started)
        self.assertIn("запущен", message)
        args = popen.call_args[0][0]
        self.assertEqual(args[0], sys.executable)
        self.assertTrue(args[1].endswith("reminder_server.py"))
        # host/port передаются серверу явно (иначе он поднимется на
        # дефолтном порту, а не там, где его ждёт клиент)
        self.assertIn("--port", args)
        self.assertEqual(str(launcher.port), args[args.index("--port") + 1])
        if sys.platform == "win32":
            self.assertEqual(
                popen.call_args[1]["creationflags"],
                subprocess.CREATE_NEW_CONSOLE,
            )

    def test_launch_skips_when_port_busy(self) -> None:
        """Занятый порт → launch() не дублирует процесс (ТЗ 9)."""
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            blocker.bind(("127.0.0.1", 0))
            blocker.listen(1)
            port = blocker.getsockname()[1]
            launcher = ReminderServerLauncher(port=port)
            with mock.patch.object(subprocess, "Popen") as popen:
                started, message = launcher.launch()
            self.assertFalse(started)
            self.assertIn("уже запущен", message)
            popen.assert_not_called()
        finally:
            blocker.close()

    def test_stop_is_noop(self) -> None:
        """stop() ничего не делает: сервер живёт 24/7 (ТЗ «не останавливать»)."""
        launcher = ReminderServerLauncher(port=_free_port())
        self.assertIsNone(launcher.stop())  # не бросает и не убивает ничего

    def test_server_url(self) -> None:
        """server_url собирается из host/port для MCP-клиента."""
        launcher = ReminderServerLauncher(host="127.0.0.1", port=8123)
        self.assertEqual(launcher.server_url, "http://127.0.0.1:8123/mcp")


# ---------------------------------------------------------------------------
# Tool calling агента (ТЗ 6, офлайн: фейковые API- и MCP-клиенты)
# ---------------------------------------------------------------------------


class _FakeOpenAIClient:
    """Клиент OpenAI-формата: client.chat.completions.create → фейки."""

    def __init__(self, responses):
        self.completions = _FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)

    @property
    def calls(self):
        return self.completions.calls


class _FakeCompletions:
    """chat.completions.create — заглушка с очередью ответов."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:  # pragma: no cover — защита от зацикливания
            raise AssertionError("Неожиданный дополнительный вызов API")
        return self.responses.pop(0)


def _api_message(content=None, tool_calls=None, reasoning=None):
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=reasoning,
    )


def _api_response(message, finish_reason="stop", usage=None):
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage, model="test-model")


def _tool_call(call_id, name, arguments: dict):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


class _FakeReminderClient:
    """MCP-клиент напоминаний: список tools + журнал вызовов."""

    def __init__(self):
        self.calls = []
        self.list_tools_calls = 0

    async def list_tools(self):
        self.list_tools_calls += 1
        return [
            MCPTool(
                name="add_reminder",
                description="Создать напоминание",
                input_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "interval_minutes": {"type": "integer"},
                    },
                    "required": ["text", "interval_minutes"],
                },
            ),
            MCPTool(name="get_stats", description="Статистика"),
        ]

    async def call_tool(self, name, arguments=None):
        self.calls.append((name, dict(arguments or {})))
        if name == "add_reminder":
            return json.dumps(
                {"id": 42, "text": arguments["text"], "next_due_at": "soon"},
                ensure_ascii=False,
            )
        return json.dumps({"total": 1}, ensure_ascii=False)


class AgentToolCallingTest(unittest.TestCase):
    """DeepSeekAgent + tool calling: tools → LLM, tool_calls → MCP (ТЗ 6)."""

    def _make_agent(self, fake_client, reminder_client):
        from src.agent import DeepSeekAgent
        from src.tokens import HeuristicCounter

        agent = DeepSeekAgent(
            api_key="test-key",
            mock_mode=False,
            token_counter=HeuristicCounter(),
            reminder_client=reminder_client,
        )
        # Подменяем HTTP-клиент OpenAI фейком (реальной сети нет)
        agent._client = fake_client
        return agent

    def test_tools_passed_and_tool_call_executed(self) -> None:
        """tools уходят в API; tool_call выполняется через MCP; итог в чате."""
        api = _FakeOpenAIClient([
            _api_response(
                _api_message(
                    content=None,
                    tool_calls=[_tool_call(
                        "call_1", "add_reminder",
                        {"text": "выпить кофе", "interval_minutes": 5},
                    )],
                    reasoning="пользователь просит напоминание",
                ),
                finish_reason="tool_calls",
            ),
            _api_response(
                _api_message(content="Готово: напомню через 5 минут."),
                usage=SimpleNamespace(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15,
                    prompt_cache_hit_tokens=None,
                    prompt_cache_miss_tokens=None,
                ),
            ),
        ])
        reminder = _FakeReminderClient()
        agent = self._make_agent(api, reminder)

        result = agent.send_message("напомни выпить кофе через 5 минут")

        self.assertIsNone(result["error"])
        self.assertEqual(result["text"], "Готово: напомню через 5 минут.")
        # tools напоминаний переданы в ПЕРВЫЙ запрос
        first_tools = api.calls[0].get("tools")
        self.assertIsNotNone(first_tools)
        self.assertEqual(first_tools[0]["type"], "function")
        self.assertEqual(
            first_tools[0]["function"]["name"], "add_reminder"
        )
        # MCP-инструмент вызван с аргументами LLM
        self.assertEqual(
            reminder.calls,
            [("add_reminder", {"text": "выпить кофе", "interval_minutes": 5})],
        )
        # Второй запрос содержит assistant(tool_calls)+reasoning и tool-результат
        second_messages = api.calls[1]["messages"]
        assistant = [m for m in second_messages
                     if isinstance(m, dict) and "tool_calls" in m]
        tool = [m for m in second_messages
                if isinstance(m, dict) and m.get("role") == "tool"]
        self.assertEqual(len(assistant), 1)
        self.assertEqual(
            assistant[0]["reasoning_content"],
            "пользователь просит напоминание",
        )
        self.assertEqual(len(tool), 1)
        self.assertEqual(tool[0]["tool_call_id"], "call_1")
        self.assertIn("id", tool[0]["content"])
        # Результат для UI: список выполненных вызовов
        self.assertEqual(result["tool_calls"][0]["name"], "add_reminder")
        self.assertIn("result", result["tool_calls"][0])
        # В истории диалога остаётся ТОЛЬКО финальный текст
        roles = [m["role"] for m in agent.history]
        self.assertNotIn("tool", roles)

    def test_no_tools_without_reminder_client(self) -> None:
        """Без reminder_client запрос уходит без tools (обычный диалог)."""
        api = _FakeOpenAIClient([
            _api_response(_api_message(content="Ответ."))
        ])
        agent = self._make_agent(api, None)
        result = agent.send_message("привет")
        self.assertIsNone(result["error"])
        self.assertNotIn("tools", api.calls[0])
        self.assertNotIn("tool_calls", result)

    def test_mock_mode_skips_mcp(self) -> None:
        """В mock-режиме MCP-клиент не опрашивается (офлайн-тесты без сети)."""
        from src.agent import DeepSeekAgent
        from src.tokens import HeuristicCounter

        reminder = _FakeReminderClient()
        agent = DeepSeekAgent(
            api_key="", mock_mode=True,
            token_counter=HeuristicCounter(),
            reminder_client=reminder,
        )
        agent.send_message("напомни что-нибудь")
        self.assertEqual(reminder.list_tools_calls, 0)
        self.assertEqual(reminder.calls, [])

    def test_unavailable_server_does_not_break_dialog(self) -> None:
        """Сервер напоминаний недоступен → запрос уходит без tools (не падает)."""
        from src.agent import DeepSeekAgent
        from src.mcp_client import MCPConnectionError
        from src.tokens import HeuristicCounter

        class _DownClient:
            async def list_tools(self):
                raise MCPConnectionError("Сервер напоминаний не запущен")

            async def call_tool(self, name, arguments=None):
                raise MCPConnectionError("Сервер напоминаний не запущен")

        api = _FakeOpenAIClient([
            _api_response(_api_message(content="Отвечаю без напоминаний."))
        ])
        agent = DeepSeekAgent(
            api_key="test-key", mock_mode=False,
            token_counter=HeuristicCounter(),
            reminder_client=_DownClient(),
        )
        agent._client = api
        result = agent.send_message("напомни кофе")
        self.assertIsNone(result["error"])
        self.assertNotIn("tools", api.calls[0])

    def test_mcp_error_reaches_llm_as_tool_result(self) -> None:
        """Ошибка MCP-вызова превращается в результат-инструмент с error."""
        from src.agent import DeepSeekAgent
        from src.mcp_client import MCPConnectionError
        from src.tokens import HeuristicCounter

        class _HalfDownClient:
            async def list_tools(self):
                return [MCPTool(name="add_reminder", description="")]

            async def call_tool(self, name, arguments=None):
                raise MCPConnectionError("сервер упал во время вызова")

        api = _FakeOpenAIClient([
            _api_response(
                _api_message(
                    content=None,
                    tool_calls=[_tool_call("c1", "add_reminder", {"text": "x",
                                                                 "interval_minutes": 1})],
                ),
                finish_reason="tool_calls",
            ),
            _api_response(_api_message(content="Сервер напоминаний недоступен.")),
        ])
        agent = DeepSeekAgent(
            api_key="test-key", mock_mode=False,
            token_counter=HeuristicCounter(),
            reminder_client=_HalfDownClient(),
        )
        agent._client = api
        result = agent.send_message("напомни")
        self.assertIsNone(result["error"])
        tool_messages = [
            m for m in api.calls[1]["messages"]
            if isinstance(m, dict) and m.get("role") == "tool"
        ]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("error", tool_messages[0]["content"])

    def test_tool_rounds_limit(self) -> None:
        """Лимит циклов: бесконечные tool_calls обрываются лимитом раундов."""
        from src.agent import DeepSeekAgent
        from src.tokens import HeuristicCounter

        def looping_response():
            return _api_response(
                _api_message(
                    content=None,
                    tool_calls=[_tool_call("c", "get_stats", {})],
                ),
                finish_reason="tool_calls",
            )

        api = _FakeOpenAIClient([looping_response() for _ in range(10)])
        agent = DeepSeekAgent(
            api_key="test-key", mock_mode=False,
            token_counter=HeuristicCounter(),
            reminder_client=_FakeReminderClient(),
            max_tool_rounds=3,
        )
        agent._client = api
        result = agent.send_message("напоминай бесконечно")
        # Лимит соблюдён: 3 раунда + финальный ответ из последнего вызова
        self.assertLessEqual(len(api.calls), 4)
        self.assertIsNone(result["error"])

    def test_tokens_summed_across_rounds(self) -> None:
        """Usage нескольких вызовов API суммируется (ТЗ: учёт токенов)."""
        from src.agent import DeepSeekAgent
        from src.tokens import HeuristicCounter

        usage = SimpleNamespace(
            prompt_tokens=100, completion_tokens=20, total_tokens=120,
            prompt_cache_hit_tokens=None, prompt_cache_miss_tokens=None,
        )
        api = _FakeOpenAIClient([
            _api_response(
                _api_message(content=None, tool_calls=[
                    _tool_call("c1", "add_reminder",
                               {"text": "a", "interval_minutes": 1}),
                ]),
                finish_reason="tool_calls", usage=usage,
            ),
            _api_response(
                _api_message(content="Готово."),
                usage=usage,
            ),
        ])
        agent = DeepSeekAgent(
            api_key="test-key", mock_mode=False,
            token_counter=HeuristicCounter(),
            reminder_client=_FakeReminderClient(),
        )
        agent._client = api
        result = agent.send_message("напомни")
        self.assertEqual(result["tokens"]["prompt_tokens"], 200)
        self.assertEqual(result["tokens"]["total_tokens"], 240)
        self.assertFalse(result["tokens"]["estimated"])


# ---------------------------------------------------------------------------
# Конфигурация напоминаний
# ---------------------------------------------------------------------------


class ReminderSettingsTest(unittest.TestCase):
    """get_reminder_settings: дефолты и валидация REMINDER_*."""

    def test_defaults(self) -> None:
        settings = get_reminder_settings({})
        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["host"], "127.0.0.1")
        self.assertEqual(settings["port"], 8000)
        self.assertEqual(settings["url"], "http://127.0.0.1:8000/mcp")
        self.assertEqual(settings["max_tool_rounds"], 5)

    def test_unsafe_host_rejected(self) -> None:
        """Хостить сервер в интернете нельзя — внешний хост → 127.0.0.1."""
        settings = get_reminder_settings({"REMINDER_SERVER_HOST": "0.0.0.0"})
        self.assertEqual(settings["host"], "127.0.0.1")

    def test_bad_port_falls_back(self) -> None:
        """Некорректный порт → 8000 (приложение не падает из-за конфига)."""
        self.assertEqual(
            get_reminder_settings({"REMINDER_SERVER_PORT": "abc"})["port"],
            8000,
        )
        self.assertEqual(
            get_reminder_settings({"REMINDER_SERVER_PORT": "70000"})["port"],
            8000,
        )

    def test_disabled_flag(self) -> None:
        settings = get_reminder_settings({"REMINDER_ENABLED": "false"})
        self.assertFalse(settings["enabled"])


# ---------------------------------------------------------------------------
# UI: кнопка/индикатор, уведомления в чате (ТЗ 7, «Сервер не запущен»)
# ---------------------------------------------------------------------------


class ReminderUITest(unittest.TestCase):
    """ChatApplication: напоминания в верхней панели и в чате."""

    root = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = gui_helpers.get_root()
        if cls.root is None:
            raise unittest.SkipTest("графическая среда недоступна (tkinter/Tk)")

    def _make_app(self, config_overrides=None):
        from src.agent import DeepSeekAgent
        from src.tokens import HeuristicCounter
        from src.ui import ChatApplication

        gui_helpers.reset_children(self.root)
        agent = DeepSeekAgent(
            api_key="", mock_mode=True, token_counter=HeuristicCounter()
        )
        config = dict(DEFAULTS)
        config.update({
            "APP_STATE_PATH": str(Path(tempfile.mkdtemp()) / "state.db"),
        })
        config.update(config_overrides or {})
        app = ChatApplication(root=self.root, agent=agent, config=config)
        self.addCleanup(self.root.update)
        return app

    def test_reminder_controls_present(self) -> None:
        """Индикатор и кнопка «Перезапустить сервер» создаются по умолчанию."""
        app = self._make_app()
        self.assertIsNotNone(app.reminder_status_label)
        self.assertIsNotNone(app.restart_reminder_btn)
        self.assertIn("напоминания",
                      app.reminder_status_label.cget("text").lower())

    def test_reminder_controls_hidden_when_disabled(self) -> None:
        """REMINDER_ENABLED=false — ни индикатора, ни кнопки."""
        app = self._make_app({"REMINDER_ENABLED": False})
        self.assertIsNone(app.reminder_status_label)
        self.assertIsNone(app.restart_reminder_btn)

    def test_status_label_states(self) -> None:
        """Индикатор различает «запущен» и «не запущен» (по ТЗ — в UI)."""
        app = self._make_app()
        app._set_reminder_status("running")
        self.assertIn("запущен", app.reminder_status_label.cget("text"))
        self.assertNotIn("не", app.reminder_status_label.cget("text"))
        app._set_reminder_status("down")
        self.assertIn("не запущен", app.reminder_status_label.cget("text"))

    def test_notification_shown_in_chat(self) -> None:
        """SSE-уведомление → строка «🔔 Напоминание: …» в области чата."""
        app = self._make_app()
        app._on_reminder_notification("REMINDER: выпить кофе")
        chat_text = app.chat.get("1.0", "end-1c")
        self.assertIn("Напоминание", chat_text)
        self.assertIn("выпить кофе", chat_text)
        # Префикс REMINDER: не дублируется в UI
        self.assertNotIn("REMINDER:", chat_text)
        # Статус-бар сообщает о срабатывании
        self.assertIn("Напоминание", app.status_label.cget("text"))

    def test_notification_survives_chat_refresh(self) -> None:
        """Уведомление остаётся в чате после перерисовки (источник — UI-слой)."""
        app = self._make_app()
        app._on_reminder_notification("REMINDER: тест")
        app._refresh_chat()
        self.assertIn("тест", app.chat.get("1.0", "end-1c"))

    def test_notification_not_in_agent_history(self) -> None:
        """Уведомления НЕ попадают в историю диалога агента."""
        app = self._make_app()
        before = len(app.agent.history)
        app._on_reminder_notification("REMINDER: тест")
        self.assertEqual(len(app.agent.history), before)


# ---------------------------------------------------------------------------
# Мелкие утилиты клиента
# ---------------------------------------------------------------------------


class ReminderClientUtilsTest(unittest.TestCase):
    """extract_call_text и обработчик сообщений слушателя."""

    def test_extract_structured(self) -> None:
        """structuredContent приоритетнее текстовых блоков."""
        result = SimpleNamespace(
            structuredContent={"result": {"total": 3}},
            content=[],
        )
        self.assertEqual(extract_call_text(result), '{\n  "result": {\n    "total": 3\n  }\n}')

    def test_extract_text_blocks(self) -> None:
        """Без structuredContent склеиваются текстовые блоки."""
        result = SimpleNamespace(
            structuredContent=None,
            content=[SimpleNamespace(text="строка 1"),
                     SimpleNamespace(text="строка 2")],
        )
        self.assertEqual(extract_call_text(result), "строка 1\nстрока 2")

    def test_extract_empty(self) -> None:
        """Пустой результат даёт понятную заглушку."""
        self.assertEqual(extract_call_text(SimpleNamespace(
            structuredContent=None, content=[])), "Пустой результат")

    def test_message_handler_parses_logging_notification(self) -> None:
        """Обработчик извлекает data из notifications/message."""
        from src.reminder_client import _make_message_handler

        received = []
        handler = _make_message_handler(SimpleNamespace(
            _on_notification=received.append))

        notification = SimpleNamespace(
            method="notifications/message",
            params=SimpleNamespace(level="info", data="REMINDER: кофе"),
        )
        message = SimpleNamespace(root=notification)
        asyncio.run(handler(message))
        self.assertEqual(received, ["REMINDER: кофе"])

    def test_message_handler_ignores_other_messages(self) -> None:
        """Сообщения прочих типов (ответы, другие уведомления) игнорируются."""
        from src.reminder_client import _make_message_handler

        received = []
        handler = _make_message_handler(SimpleNamespace(
            _on_notification=received.append))

        asyncio.run(handler(SimpleNamespace(root=SimpleNamespace(
            method="notifications/initialized", params=None))))
        asyncio.run(handler(SimpleNamespace(root=None)))
        self.assertEqual(received, [])


class ReminderClientTimeoutLabelTest(unittest.TestCase):
    """Таймауты tools/call и подпись сервера в тексте ошибок HTTP-клиента.

    Регрессия «Суммаризатор не доступен — MCP-сервер напоминаний не
    запущен»: (1) текст ошибки был зашит «Сервер напоминаний не запущен»
    в общем для напоминаний и новостей клиенте — LLM сообщала не о том
    сервере; (2) tools/call ждал 15 с (а httpx SDK ~30 с), хотя
    summarize_text с thinking-моделью идёт дольше, и обрыв выглядел как
    «сервер не запущен».
    """

    def test_tool_call_timeout_is_long(self) -> None:
        """call_tool ждёт минуты, а не 15 с: summarize_text долгий."""
        from src.reminder_client import _TOOL_CALL_TIMEOUT

        self.assertGreaterEqual(_TOOL_CALL_TIMEOUT, 120.0)

    def test_dead_server_error_uses_client_label(self) -> None:
        """Недоступный news-сервер подписывается «News-сервер», не «напоминаний»."""
        from src.mcp_client import MCPConnectionError
        from src.reminder_client import ReminderMCPClient

        news = ReminderMCPClient("http://127.0.0.1:9/mcp",
                                 label="News-сервер")
        with self.assertRaises(MCPConnectionError) as ctx:
            asyncio.run(news.call_tool("summarize_text", {"text": "x"}))
        message = str(ctx.exception)
        self.assertIn("News-сервер не запущен", message)
        self.assertNotIn("напоминаний", message.lower())
        self.assertIn("http://127.0.0.1:9/mcp", message)

    def test_default_label_is_neutral(self) -> None:
        """Без подписи клиент говорит нейтрально «MCP-сервер»."""
        from src.mcp_client import MCPConnectionError
        from src.reminder_client import ReminderMCPClient

        client = ReminderMCPClient("http://127.0.0.1:9/mcp")
        with self.assertRaises(MCPConnectionError) as ctx:
            asyncio.run(client.call_tool("x", {}))
        self.assertIn("MCP-сервер не запущен", str(ctx.exception))

    def test_restart_hook_retries_call_tool(self) -> None:
        """Ошибка подключения → хук поднимает сервер → вызов повторяется."""
        from src.mcp_client import MCPConnectionError
        from src.reminder_client import ReminderMCPClient

        calls = {"attempts": 0, "hook": 0}

        async def once(tool_name, arguments, seconds):
            calls["attempts"] += 1
            if calls["attempts"] == 1:
                raise MCPConnectionError("News-сервер не запущен")
            return "готово"

        client = ReminderMCPClient("http://127.0.0.1:9/mcp")
        client._call_tool_once = once  # type: ignore[method-assign]

        def hook() -> bool:
            calls["hook"] += 1
            return True

        client.restart_hook = hook
        result = asyncio.run(client.call_tool("summarize_text", {"text": "x"}))
        self.assertEqual(result, "готово")
        self.assertEqual(calls["attempts"], 2)
        self.assertEqual(calls["hook"], 1)

    def test_restart_hook_failure_raises_original(self) -> None:
        """Хук не помог (сервер не поднялся) — исключение первой попытки."""
        from src.mcp_client import MCPConnectionError
        from src.reminder_client import ReminderMCPClient

        async def once(tool_name, arguments, seconds):
            raise MCPConnectionError("News-сервер не запущен")

        client = ReminderMCPClient("http://127.0.0.1:9/mcp")
        client._call_tool_once = once  # type: ignore[method-assign]
        client.restart_hook = lambda: False

        with self.assertRaises(MCPConnectionError):
            asyncio.run(client.call_tool("summarize_text", {"text": "x"}))

    def test_timeout_error_does_not_trigger_hook(self) -> None:
        """Таймаут (инструмент ещё работает) — сервер не перезапускаем."""
        from src.mcp_client import MCPTimeoutError
        from src.reminder_client import ReminderMCPClient

        async def once(tool_name, arguments, seconds):
            raise MCPTimeoutError("Инструмент не ответил за 180 с")

        client = ReminderMCPClient("http://127.0.0.1:9/mcp")
        client._call_tool_once = once  # type: ignore[method-assign]
        hook_calls = []
        client.restart_hook = lambda: hook_calls.append(1) or True

        with self.assertRaises(MCPTimeoutError):
            asyncio.run(client.call_tool("summarize_text", {"text": "x"}))
        self.assertEqual(hook_calls, [])


class LauncherEnsureRunningTest(unittest.TestCase):
    """HttpMCPServerLauncher.ensure_running: поднять сервер и дождаться порта."""

    def test_running_port_returns_true_immediately(self) -> None:
        """Порт занят → True без запуска процессов."""
        from src.reminder_launcher import HttpMCPServerLauncher

        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            blocker.bind(("127.0.0.1", 0))
            blocker.listen(1)
            port = blocker.getsockname()[1]
            launcher = HttpMCPServerLauncher(
                server_path=Path("whatever.py"), port=port,
            )
            with mock.patch.object(subprocess, "Popen") as popen:
                self.assertTrue(launcher.ensure_running(timeout=1.0))
            popen.assert_not_called()
        finally:
            blocker.close()

    def test_dead_port_starts_server_and_waits(self) -> None:
        """Мёртвый порт: ensure_running стартует реальный сервер и ждёт."""
        port = _free_port()
        from src.reminder_launcher import HttpMCPServerLauncher

        launcher = HttpMCPServerLauncher(
            server_path=Path(REMINDER_SERVER), port=port,
        )
        try:
            self.assertTrue(launcher.ensure_running(timeout=20.0))
            self.assertTrue(launcher.is_server_running())
        finally:
            # Гасим запущенный сервер даже при провале теста: он держит
            # наследованный pipe вывода и «висит» в фоне. netstat пишет
            # кириллицу в CP866 — декодируем отказоустойчиво.
            out = subprocess.run(
                ["netstat", "-ano"], capture_output=True,
                text=True, errors="replace",
            ).stdout or ""
            for line in out.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    pid = line.split()[-1]
                    subprocess.run(["taskkill", "/PID", pid, "/F"],
                                   capture_output=True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
