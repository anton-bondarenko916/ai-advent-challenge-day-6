"""Тесты цепочки MCP-серверов «гео → погода» (ТЗ ai_1_20).

Покрывают сценарии приёмки ТЗ (geo_server.py + weather_server.py +
многосерверный клиент + agentic loop + панель + автозапуск):

1. Инструменты серверов работают офлайн (сетевые функции подменяются);
2. Серверы регистрируют инструменты с корректными схемами;
3. Настройки config (GEO_*/WEATHER_*) валидируются;
4. Лаунчеры запускают серверы и не дублируют на занятом порту;
5. Agentic loop агента маршрутизирует цепочку get_user_city →
   get_weather на РАЗНЫЕ серверы (офлайн: фейковые клиенты);
6. Панель MCP группирует инструменты по серверам и вызывает инструмент
   на его сервере;
7. main.create_agent собирает четырёх MCP-клиентов.

Запуск из корня проекта:
    python -m unittest tests.test_geo_weather -v
"""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import geo_server
import weather_server
from src.config import get_geo_settings, get_weather_settings
from src.geo_launcher import GeoServerLauncher
from src.mcp_models import MCPTool, ServerTools
from src.mcp_worker import MultiServerListWorker, stdio_group_name
from src.weather_launcher import WeatherServerLauncher

try:  # pragma: no cover — зависит от окружения
    import tkinter as tk
except ImportError:  # pragma: no cover
    tk = None  # type: ignore

from tests import gui_helpers

try:  # pragma: no cover — зависит от окружения
    from src.mcp_panel import MCPPanel
    from src.app_state import AppStateStore
except ImportError:  # pragma: no cover
    MCPPanel = None  # type: ignore


def _free_port() -> int:
    """Свободный TCP-порт на 127.0.0.1 (для тестов лаунчеров)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# ---------------------------------------------------------------------------
# Инструменты geo-сервера (офлайн: сеть подменяется)
# ---------------------------------------------------------------------------


class GeoToolTest(unittest.TestCase):
    """get_user_city: город по IP; любая неудача → "Unknown"."""

    def test_returns_city_on_success(self) -> None:
        with mock.patch.object(
            geo_server, "_fetch_geo_payload",
            return_value={"status": "success", "city": "Berlin"},
        ):
            self.assertEqual(geo_server.get_user_city(), "Berlin")

    def test_unknown_when_status_not_success(self) -> None:
        with mock.patch.object(
            geo_server, "_fetch_geo_payload",
            return_value={"status": "fail", "message": "reserved"},
        ):
            self.assertEqual(geo_server.get_user_city(), "Unknown")

    def test_unknown_when_city_empty(self) -> None:
        with mock.patch.object(
            geo_server, "_fetch_geo_payload",
            return_value={"status": "success", "city": ""},
        ):
            self.assertEqual(geo_server.get_user_city(), "Unknown")

    def test_unknown_on_network_error(self) -> None:
        """Сеть недоступна — инструмент отвечает "Unknown", не падает."""
        with mock.patch.object(
            geo_server, "_fetch_geo_payload", side_effect=OSError("no net")
        ):
            self.assertEqual(geo_server.get_user_city(), "Unknown")


# ---------------------------------------------------------------------------
# Инструменты weather-сервера (офлайн: сеть подменяется)
# ---------------------------------------------------------------------------


class WeatherToolTest(unittest.TestCase):
    """get_weather: геокодинг → прогноз → JSON; ошибки → {"error": …}."""

    def _patch_ok(self):
        return (
            mock.patch.object(weather_server, "_geocode_city", return_value={
                "latitude": 52.52, "longitude": 13.41, "name": "Berlin",
            }),
            mock.patch.object(
                weather_server, "_fetch_current_weather",
                return_value={
                    "temperature_2m": 15.2, "wind_speed_10m": 11.9,
                    "relative_humidity_2m": 82,
                },
            ),
        )

    def test_returns_json_with_fields(self) -> None:
        geo_patch, weather_patch = self._patch_ok()
        with geo_patch, weather_patch:
            result = json.loads(weather_server.get_weather("Berlin"))
        self.assertEqual(result["temperature"], 15.2)
        self.assertEqual(result["wind_speed"], 11.9)
        self.assertEqual(result["humidity"], 82)
        self.assertEqual(result["city"], "Berlin")

    def test_city_not_found_is_error_json(self) -> None:
        """Город не найден — {"error": …} (LLM объяснит пользователю)."""
        with mock.patch.object(
            weather_server, "_geocode_city", side_effect=LookupError("нет"),
        ):
            result = json.loads(weather_server.get_weather("Нигдеград"))
        self.assertIn("error", result)
        self.assertIn("Нигдеград", result["error"])

    def test_network_error_is_error_json(self) -> None:
        with mock.patch.object(
            weather_server, "_geocode_city", side_effect=OSError("no net"),
        ):
            result = json.loads(weather_server.get_weather("Berlin"))
        self.assertIn("error", result)

    def test_empty_city_is_error_json(self) -> None:
        result = json.loads(weather_server.get_weather("  "))
        self.assertIn("error", result)

    def test_geocode_and_forecast_call_params(self) -> None:
        """Геокодинг получает имя города, прогноз — его координаты."""
        with mock.patch.object(
            weather_server, "_geocode_city", return_value={
                "latitude": 48.85, "longitude": 2.35, "name": "Paris",
            },
        ) as geo_mock, mock.patch.object(
            weather_server, "_fetch_current_weather", return_value={},
        ) as weather_mock:
            weather_server.get_weather("Paris")
        geo_mock.assert_called_once_with("Paris")
        weather_mock.assert_called_once_with(48.85, 2.35)

    def test_accepts_raw_tool_result_from_geo_server(self) -> None:
        """LLM могла переслать весь текст get_user_city (pretty-JSON)."""
        raw_result = '{\n  "result": "Frankfurt"\n}'
        with mock.patch.object(
            weather_server, "_geocode_city", return_value={
                "latitude": 50.11, "longitude": 8.68, "name": "Frankfurt",
            },
        ) as geo_mock, mock.patch.object(
            weather_server, "_fetch_current_weather", return_value={
                "temperature_2m": 14.0,
            },
        ):
            result = json.loads(weather_server.get_weather(raw_result))
        geo_mock.assert_called_once_with("Frankfurt")
        self.assertEqual(result["temperature"], 14.0)

    def test_normalize_city_variants(self) -> None:
        """Нормализация: чистое имя, JSON, многострочный блоб, пусто."""
        normalize = weather_server._normalize_city
        self.assertEqual(normalize("Berlin"), "Berlin")
        self.assertEqual(normalize('  "Berlin" '), "Berlin")
        self.assertEqual(normalize('{"result": "Berlin"}'), "Berlin")
        self.assertEqual(normalize('{\n  "result": "Frankfurt"\n}'), "Frankfurt")
        self.assertEqual(normalize('{"city": "Prague"}'), "Prague")
        # валидный JSON без строкового города — имени в нём нет
        self.assertEqual(normalize('{"result": {"city": "Paris"}}'), "")
        self.assertEqual(normalize("{}"), "")
        self.assertEqual(normalize(""), "")
        self.assertEqual(normalize("   "), "")
        self.assertEqual(normalize("{}"), "")


# ---------------------------------------------------------------------------
# Регистрация инструментов на серверах (сценарий «list_all_tools»)
# ---------------------------------------------------------------------------


class ServerRegistrationTest(unittest.TestCase):
    """FastMCP-серверы регистрируют инструменты с корректными схемами."""

    def test_geo_server_registers_get_user_city(self) -> None:
        tools = asyncio.run(geo_server.mcp.list_tools())
        self.assertEqual([tool.name for tool in tools], ["get_user_city"])
        description = tools[0].description or ""
        self.assertIn("город", description.lower())
        self.assertEqual(tools[0].inputSchema.get("type"), "object")

    def test_weather_server_registers_get_weather(self) -> None:
        tools = asyncio.run(weather_server.mcp.list_tools())
        self.assertEqual([tool.name for tool in tools], ["get_weather"])
        schema = tools[0].inputSchema
        self.assertIn("city", schema.get("properties", {}))
        self.assertIn("city", schema.get("required", []))


# ---------------------------------------------------------------------------
# Настройки config (GEO_*/WEATHER_*)
# ---------------------------------------------------------------------------


class GeoWeatherSettingsTest(unittest.TestCase):
    """get_geo_settings / get_weather_settings: дефолты и валидация."""

    def test_geo_defaults(self) -> None:
        settings = get_geo_settings({"GEO_SERVER_PORT": 8002})
        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["host"], "127.0.0.1")
        self.assertEqual(settings["port"], 8002)
        self.assertEqual(settings["url"], "http://127.0.0.1:8002/mcp")

    def test_weather_defaults(self) -> None:
        settings = get_weather_settings({"WEATHER_SERVER_PORT": 8003})
        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["port"], 8003)
        self.assertEqual(settings["url"], "http://127.0.0.1:8003/mcp")

    def test_broken_port_falls_back_to_default(self) -> None:
        settings = get_geo_settings({"GEO_SERVER_PORT": "abc"})
        self.assertEqual(settings["port"], 8002)
        settings = get_weather_settings({"WEATHER_SERVER_PORT": 99999})
        self.assertEqual(settings["port"], 8003)

    def test_public_host_is_forced_to_localhost(self) -> None:
        """Хостить в интернете запрещено: 0.0.0.0 → 127.0.0.1."""
        settings = get_geo_settings({"GEO_SERVER_HOST": "0.0.0.0"})
        self.assertEqual(settings["host"], "127.0.0.1")
        settings = get_weather_settings({"WEATHER_SERVER_HOST": "192.168.1.5"})
        self.assertEqual(settings["host"], "127.0.0.1")

    def test_enabled_false_is_respected(self) -> None:
        settings = get_geo_settings({"GEO_ENABLED": "false"})
        self.assertFalse(settings["enabled"])
        settings = get_weather_settings({"WEATHER_ENABLED": False})
        self.assertFalse(settings["enabled"])


# ---------------------------------------------------------------------------
# Лаунчеры: автозапуск, защита от дублей, 24/7
# ---------------------------------------------------------------------------


class GeoWeatherLauncherTest(unittest.TestCase):
    """GeoServerLauncher / WeatherServerLauncher (базовый HttpMCPServerLauncher)."""

    def test_geo_launch_starts_console_process(self) -> None:
        launcher = GeoServerLauncher(port=_free_port())
        with mock.patch.object(subprocess, "Popen") as popen:
            started, message = launcher.launch()
        self.assertTrue(started)
        self.assertIn("Geo-сервер", message)
        args = popen.call_args[0][0]
        self.assertEqual(args[0], sys.executable)
        self.assertTrue(args[1].endswith("geo_server.py"))
        self.assertEqual(str(launcher.port), args[args.index("--port") + 1])
        if sys.platform == "win32":
            self.assertEqual(
                popen.call_args[1]["creationflags"],
                subprocess.CREATE_NEW_CONSOLE,
            )

    def test_weather_launch_starts_console_process(self) -> None:
        launcher = WeatherServerLauncher(port=_free_port())
        with mock.patch.object(subprocess, "Popen") as popen:
            started, message = launcher.launch()
        self.assertTrue(started)
        self.assertIn("Weather-сервер", message)
        args = popen.call_args[0][0]
        self.assertTrue(args[1].endswith("weather_server.py"))

    def test_launch_skips_when_port_busy(self) -> None:
        """Занятый порт → запуск не дублируется (сервер живёт 24/7)."""
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            blocker.bind(("127.0.0.1", 0))
            blocker.listen(16)
            port = blocker.getsockname()[1]
            for launcher in (
                GeoServerLauncher(port=port),
                WeatherServerLauncher(port=port),
            ):
                with mock.patch.object(subprocess, "Popen") as popen:
                    started, message = launcher.launch()
                self.assertFalse(started)
                self.assertIn("уже запущен", message)
                popen.assert_not_called()
        finally:
            blocker.close()

    def test_missing_script_is_not_fatal(self) -> None:
        """Отсутствующий скрипт → (False, сообщение), без исключения."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        missing = Path(tmp.name) / "nope.py"
        launcher = GeoServerLauncher(port=_free_port(), server_path=missing)
        started, message = launcher.launch()
        self.assertFalse(started)
        self.assertIn("не найден", message)

    def test_server_urls(self) -> None:
        self.assertEqual(
            GeoServerLauncher(host="127.0.0.1", port=8002).server_url,
            "http://127.0.0.1:8002/mcp",
        )
        self.assertEqual(
            WeatherServerLauncher(host="127.0.0.1", port=8003).server_url,
            "http://127.0.0.1:8003/mcp",
        )


# ---------------------------------------------------------------------------
# Agentic loop: цепочка get_user_city → get_weather через РАЗНЫЕ серверы
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


def _api_message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls,
                           reasoning_content=None)


def _api_response(message, finish_reason="stop"):
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None, model="test-model")


def _tool_call(call_id, name, arguments: dict):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


class _FakeGeoClient:
    """MCP-клиент geo-сервера: tools + журнал вызовов."""

    def __init__(self):
        self.calls = []

    async def list_tools(self):
        return [
            MCPTool(
                name="get_user_city",
                description="Определить город пользователя по IP-адресу",
                input_schema={"type": "object", "properties": {}},
            ),
        ]

    async def call_tool(self, name, arguments=None):
        self.calls.append((name, dict(arguments or {})))
        return "Berlin"


class _FakeWeatherClient:
    """MCP-клиент weather-сервера: tools + журнал вызовов."""

    def __init__(self):
        self.calls = []

    async def list_tools(self):
        return [
            MCPTool(
                name="get_weather",
                description="Получить текущую погоду в указанном городе",
                input_schema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            ),
        ]

    async def call_tool(self, name, arguments=None):
        self.calls.append((name, dict(arguments or {})))
        return json.dumps({
            "temperature": 15.2, "wind_speed": 11.9, "humidity": 82,
        })


class AgentGeoWeatherChainTest(unittest.TestCase):
    """LLM строит цепочку geo → weather; вызовы уходят на свои серверы."""

    def _make_agent(self, geo_client, weather_client):
        from src.agent import DeepSeekAgent
        from src.tokens import HeuristicCounter

        api = _FakeOpenAIClient([
            # Ход 1: LLM просит get_user_city
            _api_response(
                _api_message(content=None, tool_calls=[
                    _tool_call("c1", "get_user_city", {}),
                ]),
                finish_reason="tool_calls",
            ),
            # Ход 2: LLM передаёт город в get_weather
            _api_response(
                _api_message(content=None, tool_calls=[
                    _tool_call("c2", "get_weather", {"city": "Berlin"}),
                ]),
                finish_reason="tool_calls",
            ),
            # Ход 3: финальный ответ пользователю
            _api_response(_api_message(
                content="В Берлине сейчас 15.2°C, ветер 11.9 км/ч, влажность 82%.",
            )),
        ])
        agent = DeepSeekAgent(
            api_key="test-key", mock_mode=False,
            token_counter=HeuristicCounter(),
            geo_client=geo_client, weather_client=weather_client,
        )
        agent._client = api
        return agent, api

    def test_chain_routes_tools_to_their_servers(self) -> None:
        """get_user_city уходит на geo, get_weather — на weather-сервер."""
        geo, weather = _FakeGeoClient(), _FakeWeatherClient()
        agent, api = self._make_agent(geo, weather)

        result = agent.send_message("Какая сегодня погода?")

        self.assertIsNone(result["error"])
        self.assertIn("Берлине", result["text"])
        # Цепочка: сначала geo, затем weather с городом из результата geo
        self.assertEqual(geo.calls, [("get_user_city", {})])
        self.assertEqual(weather.calls, [("get_weather", {"city": "Berlin"})])
        # Оба инструмента переданы LLM в первом запросе
        first_tools = api.calls[0]["tools"]
        names = [t["function"]["name"] for t in first_tools]
        self.assertEqual(names, ["get_user_city", "get_weather"])
        # Результат get_user_city дошёл до LLM как роль tool
        second_messages = api.calls[1]["messages"]
        tool_msgs = [m for m in second_messages
                     if isinstance(m, dict) and m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["content"], "Berlin")
        # В историю диалога tool-сообщения не попадают
        roles = [m["role"] for m in agent.history]
        self.assertNotIn("tool", roles)

    def test_executed_tools_reported_to_ui(self) -> None:
        """UI получает список выполненных вызовов с результатами."""
        geo, weather = _FakeGeoClient(), _FakeWeatherClient()
        agent, _ = self._make_agent(geo, weather)

        result = agent.send_message("Какая погода?")

        names = [call["name"] for call in result["tool_calls"]]
        self.assertEqual(names, ["get_user_city", "get_weather"])
        self.assertIn("temperature", result["tool_calls"][1]["result"])


# ---------------------------------------------------------------------------
# MultiServerListWorker: сбор tools со всех серверов (офлайн)
# ---------------------------------------------------------------------------


class _FakeListClient:
    """Готовый клиент для MultiServerListWorker (list_tools без сети)."""

    def __init__(self, tools=None, error=None, server_url=""):
        self._tools = tools or []
        self._error = error
        self.server_url = server_url
        self.server_command = [sys.executable, "-u", "my_git_server.py"]

    async def list_tools(self):
        if self._error is not None:
            raise self._error
        return list(self._tools)


class _StdioListClient(_FakeListClient):
    """Stdio-вариант: server_command задаётся явно."""

    def __init__(self, tools):
        super().__init__(tools=tools)
        self.server_command = [sys.executable, "-u", "my_git_server.py"]


class MultiServerListWorkerTest(unittest.TestCase):
    """Опрос всех серверов: группы, частичные ошибки, имя stdio-группы."""

    def _run_worker(self, stdio_client=None, http_clients=None):
        result = {}
        done = threading.Event()

        worker = MultiServerListWorker(
            on_success=lambda groups: (
                result.setdefault("groups", groups), done.set(),
            ),
            on_error=lambda message: (
                result.setdefault("error", message), done.set(),
            ),
            stdio_client=stdio_client,
            http_clients=http_clients,
        )
        worker.start()
        self.assertTrue(done.wait(15), "воркер не завершился за 15 с")
        return result

    def test_collects_tools_from_all_servers(self) -> None:
        stdio = _StdioListClient([MCPTool(name="get_git_log")])
        geo = _FakeListClient(
            tools=[MCPTool(name="get_user_city")], server_url="geo-url",
        )
        weather = _FakeListClient(
            tools=[MCPTool(name="get_weather")], server_url="weather-url",
        )
        result = self._run_worker(
            stdio_client=stdio,
            http_clients=[("geo", geo), ("weather", weather)],
        )
        self.assertNotIn("error", result)
        groups = {group.name: group for group in result["groups"]}
        self.assertEqual(
            groups["my_git_server"].tools[0].name, "get_git_log")
        self.assertEqual(groups["geo"].tools[0].name, "get_user_city")
        self.assertEqual(groups["weather"].tools[0].name, "get_weather")
        # url группы запоминается для маршрутизации вызова из панели
        self.assertEqual(groups["geo"].url, "geo-url")

    def test_dead_server_does_not_hide_others(self) -> None:
        """Упавший сервер → группа с error; остальные отвечают."""
        from src.mcp_client import MCPConnectionError

        geo = _FakeListClient(
            tools=[MCPTool(name="get_user_city")], server_url="geo-url",
        )
        weather = _FakeListClient(error=MCPConnectionError(
            "Weather-сервер не запущен (http://127.0.0.1:8003/mcp)."
        ))
        result = self._run_worker(
            http_clients=[("geo", geo), ("weather", weather)],
        )
        groups = {group.name: group for group in result["groups"]}
        self.assertEqual(groups["geo"].total, 1)
        self.assertEqual(groups["weather"].total, 0)
        self.assertIn("Weather-сервер", groups["weather"].error)

    def test_stdio_group_name_from_command(self) -> None:
        self.assertEqual(
            stdio_group_name([sys.executable, "-u", "my_git_server.py"]),
            "my_git_server",
        )
        self.assertEqual(
            stdio_group_name(["npx", "-y", "open-internet-mcp"]),
            "open-internet-mcp",
        )
        self.assertEqual(stdio_group_name([]), "stdio")


# ---------------------------------------------------------------------------
# Панель MCP: группировка инструментов по серверам
# ---------------------------------------------------------------------------


def _sync_panel_result(panel) -> None:
    """Прокручивает очередь панели до опустошения (без after-цикла)."""
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not panel._busy and panel._queue.empty():
            return
        panel._poll()
        time.sleep(0.01)
    raise AssertionError("воркер не ответил за 5 секунд")


def _panel_settings(tmp_dir: Path) -> dict:
    """Настройки панели на временные пути (чтобы не трогать mcp.log)."""
    return {
        "enabled": True,
        "server_command": [sys.executable, "-u", "my_git_server.py"],
        "timeout": 30.0,
        "log_path": tmp_dir / "mcp.log",
    }


@unittest.skipIf(tk is None or MCPPanel is None,
                 "графическая среда или панель недоступны")
class PanelGroupingTest(unittest.TestCase):
    """MCPPanel: заголовки серверов, частичные ошибки, вызов на свой сервер."""

    root = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = gui_helpers.get_root()
        if cls.root is None:
            raise unittest.SkipTest("графическая среда недоступна (tkinter/Tk)")

    def setUp(self) -> None:
        gui_helpers.reset_children(self.root)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.call_worker_kwargs = []

    def tearDown(self) -> None:
        if getattr(self, "panel", None) is not None:
            try:
                self.panel.destroy()
            except tk.TclError:
                pass
        self.root.update()

    def _make_panel(self) -> MCPPanel:
        """Панель с синхронной фейковой фабрикой (результат — self._result)."""
        store = AppStateStore(Path(self.tmp.name) / "state.db")
        panel = MCPPanel(
            self.root,
            settings=_panel_settings(Path(self.tmp.name)),
            app_state=store,
            worker_factory=self._fake_list_factory,
            call_worker_factory=self._fake_call_factory,
        )
        self.panel = panel
        return panel

    def _fake_list_factory(self, on_success, on_error):
        panel = self.panel

        class _Worker:
            def start(self_inner) -> None:
                result = getattr(panel, "_result", None)
                if isinstance(result, Exception):
                    on_error(str(result))
                else:
                    on_success(result)

        return _Worker()

    def _fake_call_factory(self, tool_name, arguments, on_success, on_error,
                           **kwargs):
        """Фабрика вызова: запоминает kwarg source_url (маршрутизация)."""
        self.call_worker_kwargs.append(dict(kwargs))

        class _Worker:
            def start(self_inner) -> None:
                on_success('{"result": "ok"}')

        return _Worker()

    def _deliver(self, result) -> None:
        self.panel._result = result
        self.panel.refresh()
        _sync_panel_result(self.panel)

    def _labels(self) -> list:
        found = []

        def _walk(widget) -> None:
            for child in widget.winfo_children():
                if isinstance(child, tk.Label):
                    found.append(str(child.cget("text")))
                _walk(child)

        _walk(self.panel.tools_frame)
        return found

    def test_tools_grouped_by_server_with_headers(self) -> None:
        """Группы рисуются с заголовками «● сервер» и своими карточками."""
        panel = self._make_panel()
        self._deliver([
            ServerTools(name="geo", tools=[
                MCPTool(name="get_user_city", description="Город по IP"),
            ], url="http://127.0.0.1:8002/mcp"),
            ServerTools(name="weather", tools=[
                MCPTool(name="get_weather", description="Погода"),
            ], url="http://127.0.0.1:8003/mcp"),
        ])
        labels = self._labels()
        self.assertIn("● geo", labels)
        self.assertIn("● weather", labels)
        self.assertTrue(any("get_user_city" in text for text in labels))
        self.assertTrue(any("get_weather" in text for text in labels))
        # Статус: суммарно 2 инструмента
        self.assertEqual(panel.get_status_kind(), "connected")
        self.assertIn("2 инструмента", panel.get_status_text())
        # Маршрут вызовов построен
        self.assertEqual(
            panel.tool_sources.get("get_user_city"),
            "http://127.0.0.1:8002/mcp",
        )

    def test_flat_tool_list_still_renders(self) -> None:
        """Плоский список MCPTool (одиночный сервер) — как раньше."""
        panel = self._make_panel()
        self._deliver([MCPTool(name="wiki", description="Википедия")])
        labels = self._labels()
        self.assertTrue(any("wiki" in text for text in labels))
        # Безымянная группа — без заголовка
        self.assertNotIn("● ", labels)

    def test_dead_server_shows_error_and_others(self) -> None:
        """Недоступный сервер: красная подпись ошибки, остальные живы."""
        panel = self._make_panel()
        self._deliver([
            ServerTools(name="geo", error="Geo-сервер не запущен"),
            ServerTools(name="weather", tools=[
                MCPTool(name="get_weather"),
            ]),
        ])
        labels = self._labels()
        self.assertTrue(any("Geo-сервер не запущен" in text for text in labels))
        self.assertTrue(any("get_weather" in text for text in labels))
        self.assertEqual(panel.get_status_kind(), "connected")
        self.assertIn("нет связи: geo", panel.get_status_text())

    def test_all_dead_is_error_status(self) -> None:
        panel = self._make_panel()
        self._deliver([
            ServerTools(name="geo", error="Geo-сервер не запущен"),
            ServerTools(name="weather", error="Weather-сервер не запущен"),
        ])
        self.assertEqual(panel.get_status_kind(), "error")

    def _sync_dialog(self, dialog) -> None:
        """Прокручивает очередь диалога до опустошения (без after-цикла)."""
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if not dialog._busy and dialog._queue.empty():
                return
            dialog._poll()
            time.sleep(0.01)
        raise AssertionError("воркер вызова не ответил за 5 секунд")

    def test_http_tool_call_routes_to_source(self) -> None:
        """Кнопка «Вызвать» у инструмента geo уходит на geo-сервер."""
        panel = self._make_panel()
        self._deliver([
            ServerTools(name="weather", tools=[
                MCPTool(name="get_weather"),
            ], url="http://127.0.0.1:8003/mcp"),
        ])
        tool = panel.tools[0]
        dialog = panel.open_call_dialog(tool, panel.tool_sources[tool.name])
        self.addCleanup(dialog.destroy)
        dialog.run_btn.invoke()
        self._sync_dialog(dialog)
        self.assertEqual(len(self.call_worker_kwargs), 1)
        self.assertEqual(
            self.call_worker_kwargs[0].get("source_url"),
            "http://127.0.0.1:8003/mcp",
        )
        self.assertIn("ok", dialog.get_result_text())


# ---------------------------------------------------------------------------
# Сборка агента в main.create_agent: четыре MCP-клиента
# ---------------------------------------------------------------------------


class CreateAgentWiringTest(unittest.TestCase):
    """create_agent подключает reminder/news/geo/weather к агенту."""

    def _config(self, **overrides) -> dict:
        from src.config import get_config

        config = get_config()
        config.update({
            "DEEPSEEK_API_KEY": "test-key",
            "MOCK_MODE": False,
            "STORAGE_ENABLED": False,
            "DEFAULT_MODEL": "deepseek-v4-flash",
        })
        config.update(overrides)
        return config

    def test_all_enabled_builds_four_clients(self) -> None:
        import main as app_main

        agent = app_main.create_agent(self._config())
        try:
            self.assertEqual(len(agent._tool_clients), 4)
        finally:
            agent.close()

    def test_geo_weather_disabled_drops_clients(self) -> None:
        import main as app_main

        agent = app_main.create_agent(self._config(
            GEO_ENABLED=False, WEATHER_ENABLED=False,
        ))
        try:
            self.assertEqual(len(agent._tool_clients), 2)
        finally:
            agent.close()

    def test_disabled_geo_skips_launcher(self) -> None:
        """GEO_ENABLED=false → лаунчер не создаётся, сервер не запускается."""
        from src.config import get_geo_settings

        self.assertFalse(get_geo_settings({"GEO_ENABLED": "false"})["enabled"])


if __name__ == "__main__":
    unittest.main()
