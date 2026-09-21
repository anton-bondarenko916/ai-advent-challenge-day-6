"""Тесты подсистемы MCP: клиент, воркер, панель UI, app_state.

Покрывают сценарии приёмки ТЗ (адаптированы под офлайн-прогон):

- подключение: у панели статус «Не подключён», запрос не выполняется сам;
- обновление: реальный FastMCP-сервер (stdio, без сети) отдаёт список;
  у панели после «Обновить» — зелёный статус и карточки инструментов;
- содержимое: фикстурный сервер отдаёт wikipedia_summary, github_search;
- inputSchema: клик по карточке открывает модалку с JSON-схемой;
- повторное обновление: список перестраивается, дублей нет;
- ошибка: несуществующая команда → MCPConnectionError и красный статус;
- таймаут: «зависший» сервер → MCPTimeoutError;
- закрытие во время загрузки: воркер daemon, разрушение панели безопасно;
- npx не найден: сообщение с ссылкой https://nodejs.org;
- UI не блокируется: пока идёт запрос, поток событий Tk жив;
- сворачивание: тело скрывается, состояние сохраняется в app_state.

Интеграционные тесты используют фикстуру
``tests/fixtures/mcp_echo_server.py`` (FastMCP, stdio-транспорт) — они
не требуют ни сети, ни Node.js. Если SDK ``mcp`` не установлен, тесты
аккуратно пропускаются.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import queue
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from src.config import DEFAULTS, get_mcp_settings
from src.mcp_client import (
    MCPClient,
    MCPConnectionError,
    MCPError,
    MCPTimeoutError,
)
from src.mcp_models import MCPTool, sort_tools
from src.mcp_panel import (
    STATUS_CONNECTED,
    STATUS_ERROR,
    STATUS_IDLE,
    STATUS_LOADING,
    MCPPanel,
)
from src.mcp_worker import MCPWorker
from src.app_state import AppStateStore

try:  # pragma: no cover — зависит от окружения
    import tkinter as tk
except ImportError:  # pragma: no cover
    tk = None  # type: ignore

from tests import gui_helpers

# SDK mcp нужен интеграционным тестам (фикстура-сервер + клиент).
try:  # pragma: no cover — зависит от окружения
    import mcp  # noqa: F401

    MCP_SDK = True
except ImportError:  # pragma: no cover
    MCP_SDK = False

_SERVER = str(Path(__file__).resolve().parent / "fixtures" / "mcp_echo_server.py")


def _server_command() -> list:
    """Команда запуска фикстурного MCP-сервера (python -u, без буферизации)."""
    return [sys.executable, "-u", _SERVER]


def _fake_response(tools):
    """Подменяет ListToolsResult SDK (только поле .tools)."""
    return mock.Mock(tools=tools)


def _fake_sdk_tool(name, description=None, input_schema=None):
    """Подменяет mcp.types.Tool (name/description/inputSchema)."""
    tool = mock.Mock()
    tool.name = name
    tool.description = description
    tool.inputSchema = input_schema
    return tool


def _panel_settings(tmp_dir: Path) -> dict:
    """Настройки панели на временные пути (чтобы не трогать mcp.log)."""
    return {
        "enabled": True,
        "server_command": _server_command(),
        "timeout": 30.0,
        "log_path": tmp_dir / "mcp.log",
    }


def _sync_panel_result(panel: MCPPanel) -> None:
    """Дожидается доставки результата фейкового воркера в панель.

    Фейковая фабрика из другого потока кладёт результат в очередь; вместо
    ожидания after-цикла вызываем обработчик очереди напрямую. Успех:
    запрос обработан (busy=False) и очередь пуста.
    """
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not panel._busy and panel._queue.empty():
            return
        panel._poll()
        time.sleep(0.01)
    raise AssertionError("воркер не ответил за 5 секунд")


# ---------------------------------------------------------------------------
# Доменная модель
# ---------------------------------------------------------------------------

class MCPToolTest(unittest.TestCase):
    """MCPTool: поля по умолчанию, краткое описание, JSON схемы."""

    def test_defaults(self) -> None:
        tool = MCPTool(name="wikipedia_summary")
        self.assertEqual(tool.description, "")
        self.assertIsNone(tool.input_schema)

    def test_short_description_truncates_to_first_line(self) -> None:
        tool = MCPTool(name="t", description="Первая строка\nВторая строка")
        self.assertEqual(tool.short_description(), "Первая строка")
        long = "х" * 200
        cut = MCPTool(name="t", description=long).short_description(limit=80)
        self.assertEqual(len(cut), 80)
        self.assertTrue(cut.endswith("…"))

    def test_schema_json_pretty_print(self) -> None:
        schema = {"type": "object", "required": ["title"]}
        tool = MCPTool(name="t", input_schema=schema)
        parsed = __import__("json").loads(tool.schema_json())
        self.assertEqual(parsed, schema)

    def test_sort_tools_is_case_insensitive_and_not_inplace(self) -> None:
        tools = [MCPTool(name="Wikipedia"), MCPTool(name="arxiv"), MCPTool(name="GitHub")]
        ordered = sort_tools(tools)
        self.assertEqual([t.name for t in ordered], ["arxiv", "GitHub", "Wikipedia"])
        self.assertEqual([t.name for t in tools], ["Wikipedia", "arxiv", "GitHub"])


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

class MCPSettingsTest(unittest.TestCase):
    """config.get_mcp_settings: разбор команды, таймаут, выключатель."""

    def test_defaults(self) -> None:
        s = get_mcp_settings({})
        self.assertTrue(s["enabled"])
        self.assertEqual(s["server_command"], ["npx", "-y", "open-internet-mcp"])
        self.assertEqual(s["timeout"], 30.0)
        self.assertTrue(str(s["log_path"]).endswith("mcp.log"))

    def test_custom_command_is_split(self) -> None:
        s = get_mcp_settings({"MCP_SERVER_COMMAND": "uvx -y other-mcp --flag"})
        self.assertEqual(s["server_command"], ["uvx", "-y", "other-mcp", "--flag"])

    def test_quoted_command_keeps_path(self) -> None:
        s = get_mcp_settings({"MCP_SERVER_COMMAND": '"C:\\Program Files\\x.exe" serve'})
        self.assertEqual(len(s["server_command"]), 2)
        self.assertTrue(s["server_command"][0].endswith("x.exe"))

    def test_broken_command_falls_back_to_default(self) -> None:
        s = get_mcp_settings({"MCP_SERVER_COMMAND": '  '})
        self.assertEqual(s["server_command"], ["npx", "-y", "open-internet-mcp"])

    def test_timeout_out_of_range_is_clamped(self) -> None:
        self.assertEqual(get_mcp_settings({"MCP_TIMEOUT": "9999"})["timeout"], 30.0)
        self.assertEqual(get_mcp_settings({"MCP_TIMEOUT": "abc"})["timeout"], 30.0)

    def test_disabled_flag(self) -> None:
        # Как и остальные *_ENABLED, ключ приводится к bool на уровне
        # get_config/_coerce; get_mcp_settings получает готовый bool.
        self.assertFalse(get_mcp_settings({"MCP_ENABLED": False})["enabled"])


# ---------------------------------------------------------------------------
# Клиент: разбор ответа и подготовка ошибок (без сети)
# ---------------------------------------------------------------------------

class MCPClientOfflineTest(unittest.TestCase):
    """MCPClient без запуска сервера: разбор, ошибки, конфигурация."""

    def test_empty_command_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            MCPClient(server_command=[])

    def test_default_command_and_timeout(self) -> None:
        client = MCPClient()
        self.assertEqual(client.server_command, ["npx", "-y", "open-internet-mcp"])
        self.assertEqual(client.timeout, 30.0)

    def test_parse_tools_maps_sdk_fields(self) -> None:
        client = MCPClient()
        response = _fake_response([
            _fake_sdk_tool("b_tool", "второй", {"type": "object"}),
            _fake_sdk_tool("a_tool"),
        ])
        tools = sort_tools(client._parse_tools(response))
        self.assertEqual([t.name for t in tools], ["a_tool", "b_tool"])
        self.assertEqual(tools[0].description, "")
        self.assertIsNone(tools[0].input_schema)
        self.assertEqual(tools[1].description, "второй")
        self.assertEqual(tools[1].input_schema, {"type": "object"})

    def test_parse_tools_skips_nameless(self) -> None:
        client = MCPClient()
        response = _fake_response([_fake_sdk_tool(""), _fake_sdk_tool("ok")])
        tools = client._parse_tools(response)
        self.assertEqual([t.name for t in tools], ["ok"])

    def test_parse_tools_rejects_response_without_tools(self) -> None:
        client = MCPClient()
        with self.assertRaises(MCPConnectionError):
            client._parse_tools(mock.Mock(spec=[]))  # нет поля .tools

    def test_non_dict_schema_becomes_none(self) -> None:
        client = MCPClient()
        response = _fake_response([_fake_sdk_tool("t", "d", "not-a-dict")])
        self.assertIsNone(client._parse_tools(response)[0].input_schema)

    def test_npx_failure_message_contains_nodejs_link(self) -> None:
        client = MCPClient()
        error = client._connection_error_from(FileNotFoundError("npx"))
        self.assertIn("https://nodejs.org", str(error))

    def test_unknown_command_failure_has_no_nodejs_link(self) -> None:
        client = MCPClient(server_command=["definitely-missing-cmd"])
        error = client._connection_error_from(FileNotFoundError("x"))
        self.assertNotIn("nodejs.org", str(error))


# ---------------------------------------------------------------------------
# Клиент: интеграция с реальным stdio-сервером (без сети)
# ---------------------------------------------------------------------------

@unittest.skipUnless(MCP_SDK, "SDK mcp не установлен")
class MCPClientServerTest(unittest.TestCase):
    """Реальные сессии с фикстурным FastMCP-сервером (stdio)."""

    def test_list_tools_from_real_server(self) -> None:
        """tools/list через SDK: handshake + разбор списка инструментов."""
        client = MCPClient(
            server_command=_server_command(),
            timeout=30.0,
            log_path=tempfile.mktemp(suffix=".log"),
        )
        tools = asyncio_run_safe(client.list_tools())
        names = [t.name for t in tools]
        self.assertIn("wikipedia_summary", names)
        self.assertIn("github_search", names)
        self.assertEqual(names, sorted(names, key=str.lower))
        wiki = next(t for t in tools if t.name == "wikipedia_summary")
        self.assertIn("Википедии", wiki.description)
        self.assertEqual(wiki.input_schema.get("type"), "object")
        self.assertIn("title", wiki.input_schema.get("properties", {}))

    def test_list_tools_twice_no_state_between_sessions(self) -> None:
        """Повторный запрос — новая сессия, тот же результат (без дублей)."""
        client = MCPClient(
            server_command=_server_command(),
            timeout=30.0,
            log_path=tempfile.mktemp(suffix=".log"),
        )
        first = asyncio_run_safe(client.list_tools())
        second = asyncio_run_safe(client.list_tools())
        self.assertEqual([t.name for t in first], [t.name for t in second])

    def test_missing_command_raises_connection_error(self) -> None:
        client = MCPClient(
            server_command=["mcp-no-such-executable-xyz"],
            timeout=5.0,
            log_path=tempfile.mktemp(suffix=".log"),
        )
        with self.assertRaises(MCPConnectionError):
            asyncio_run_safe(client.list_tools())

    def test_hanging_server_raises_timeout_error(self) -> None:
        """Сервер-«молчун»: initialize не отвечает → MCPTimeoutError."""
        client = MCPClient(
            server_command=[sys.executable, "-u", "-c", "import time; time.sleep(30)"],
            timeout=2.0,
            log_path=tempfile.mktemp(suffix=".log"),
        )
        started = time.monotonic()
        with self.assertRaises(MCPTimeoutError):
            asyncio_run_safe(client.list_tools())
        # Отбой по таймауту, а не по полному ожиданию (30 с молчания)
        self.assertLess(time.monotonic() - started, 15.0)

    def test_stderr_is_written_to_log(self) -> None:
        """stderr сервера не глушится: попадает в mcp.log."""
        log_path = Path(tempfile.mktemp(suffix=".log"))
        client = MCPClient(
            server_command=_server_command(),
            timeout=30.0,
            log_path=log_path,
        )
        asyncio_run_safe(client.list_tools())
        self.assertTrue(log_path.exists())
        # FastMCP пишет в stderr служебные строки о запросах
        content = log_path.read_text(encoding="utf-8", errors="replace")
        self.assertIn("ListToolsRequest", content)


def asyncio_run_safe(coro):
    """Запускает корутину в свежем event loop (как MCPWorker.run)."""
    return __import__("asyncio").run(coro)


# ---------------------------------------------------------------------------
# Воркер
# ---------------------------------------------------------------------------

class MCPWorkerTest(unittest.TestCase):
    """MCPWorker: callback'и успеха/ошибки, daemon-поток."""

    def _run_worker(self, client: MCPClient, done: dict) -> None:
        worker = MCPWorker(
            on_success=lambda tools: done.update(tools=tools),
            on_error=lambda msg: done.update(error=msg),
            client=client,
        )
        self.assertTrue(worker.daemon)
        worker.start()
        worker.join(timeout=10)
        self.assertFalse(worker.is_alive(), "воркер завис")

    def test_success_delivers_tools(self) -> None:
        done: dict = {}
        client = mock.Mock()
        client.list_tools = lambda: _fake_future([MCPTool(name="t1")])
        self._run_worker(client, done)
        self.assertEqual([t.name for t in done["tools"]], ["t1"])
        self.assertNotIn("error", done)

    def test_mcp_error_delivers_message(self) -> None:
        done: dict = {}
        client = mock.Mock()

        def _boom():
            raise MCPTimeoutError("Сервер не ответил за 30 секунд")

        client.list_tools = _boom
        self._run_worker(client, done)
        self.assertIn("не ответил", done["error"])

    def test_unexpected_error_delivers_generic_message(self) -> None:
        done: dict = {}
        client = mock.Mock()

        def _boom():
            raise RuntimeError("бум")

        client.list_tools = _boom
        self._run_worker(client, done)
        self.assertIn("Неожиданная ошибка", done["error"])
        self.assertIn("бум", done["error"])


def _fake_future(value):
    """Асинхронная заглушка: coroutine, возвращающая value."""
    import asyncio

    async def _coro():
        return value

    return _coro()


# ---------------------------------------------------------------------------
# AppStateStore
# ---------------------------------------------------------------------------

class AppStateStoreTest(unittest.TestCase):
    """Таблица app_state: upsert, bool-хелперы, устойчивость к ошибкам."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = AppStateStore(Path(self.tmp.name) / "state.db")

    def test_set_get_roundtrip(self) -> None:
        self.assertIsNone(self.store.get("missing"))
        self.store.set("k", "v1")
        self.store.set("k", "v2")  # upsert без дублей
        self.assertEqual(self.store.get("k"), "v2")

    def test_bool_helpers(self) -> None:
        self.assertFalse(self.store.get_bool("k", default=False))
        self.store.set_bool("k", True)
        self.assertTrue(self.store.get_bool("k"))
        self.store.set_bool("k", False)
        self.assertFalse(self.store.get_bool("k"))

    def test_unreadable_db_is_safe(self) -> None:
        # Каталог вместо файла БД: sqlite не сможет открыть
        broken = AppStateStore(Path(self.tmp.name))
        self.assertIsNone(broken.get("k"))
        self.assertFalse(broken.set("k", "v"))


# ---------------------------------------------------------------------------
# Панель (GUI; без дисплея тесты пропускаются)
# ---------------------------------------------------------------------------

class MCPPanelTest(unittest.TestCase):
    """MCPPanel: статусы, список, модалка схемы, сворачивание."""

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
        self._reset_deliver()
        self.panel = self._make_panel()
        self.root.update()

    def tearDown(self) -> None:
        if self.panel is not None:
            try:
                self.panel.destroy()
            except tk.TclError:
                pass
        self.root.update()

    # ---- фабрики -------------------------------------------------------

    def _make_panel(self, settings=None) -> MCPPanel:
        """Панель с фейковым воркером (результат подаётся тестом)."""
        store = AppStateStore(Path(self.tmp.name) / "state.db")
        panel = MCPPanel(
            self.root,
            settings=settings or _panel_settings(Path(self.tmp.name)),
            app_state=store,
            worker_factory=self._fake_factory,
        )
        return panel

    def _fake_factory(self, on_success, on_error):
        """Фабрика «висящего» воркера: ответ доставляет deliver()."""
        panel = self.panel
        worker = mock.Mock()

        def _start() -> None:
            def runner() -> None:
                # Реальная асинхронность: ждём тестовое событие
                self._deliver_event.wait(15)
                result = getattr(self, "_result", None)
                if isinstance(result, Exception):
                    on_error(str(result))
                elif isinstance(result, str):
                    on_error(result)
                elif result is not None:
                    on_success(result)
                # None — ответ так и не пришёл (воркер «висит»)

            threading.Thread(target=runner, daemon=True).start()

        worker.start.side_effect = _start
        panel._fake_worker = worker
        return worker

    def _deliver(self, result, panel=None) -> None:
        """Подаёт результат воркеру (как будто сервер ответил)."""
        self._result = result
        self._deliver_event.set()
        _sync_panel_result(panel or self.panel)

    def _reset_deliver(self) -> None:
        self._result = None
        self._deliver_event = threading.Event()

    # ---- сценарии -------------------------------------------------------

    def test_initial_status_is_disconnected(self) -> None:
        """На старте: статус «Не подключён», кнопка активна, список пуст."""
        self.assertEqual(self.panel.get_status_kind(), STATUS_IDLE)
        self.assertEqual(self.panel.get_status_text(), "Не подключён")
        self.assertEqual(str(self.panel.refresh_btn.cget("state")), "normal")
        self.assertEqual(self.panel.tools, [])

    def test_refresh_shows_loading_then_tools(self) -> None:
        """«Обновить» → жёлтый «Загрузка...» → зелёный статус и карточки."""
        self.panel.refresh()
        # Пока воркер не ответил — блокировка кнопки и статус загрузки
        self.assertEqual(str(self.panel.refresh_btn.cget("state")), "disabled")
        self.assertEqual(self.panel.get_status_kind(), STATUS_LOADING)
        self.assertIn("Загрузка", self.panel.get_status_text())
        # Сервер «ответил»
        self._deliver([MCPTool(name="wiki", description="Википедия")])
        self.assertEqual(self.panel.get_status_kind(), STATUS_CONNECTED)
        self.assertIn("1 инструмент ", self.panel.get_status_text() + " ")
        names = self._card_names()
        self.assertTrue(any("wiki" in t for t in names))

    def _card_names(self) -> list:
        """Имена инструментов с карточек (рекурсивный обход меток)."""
        found = []

        def _walk(widget) -> None:
            for child in widget.winfo_children():
                if isinstance(child, tk.Label):
                    found.append(str(child.cget("text")))
                _walk(child)

        _walk(self.panel.tools_frame)
        return found

    def test_status_plural_forms(self) -> None:
        """Формы слов: 1 инструмент / 2 инструмента / 5 инструментов."""
        from src.mcp_panel import _plural_tools

        self.assertEqual(_plural_tools(1), "инструмент")
        self.assertEqual(_plural_tools(2), "инструмента")
        self.assertEqual(_plural_tools(5), "инструментов")
        self.assertEqual(_plural_tools(11), "инструментов")
        self.assertEqual(_plural_tools(21), "инструмент")
        self.assertEqual(_plural_tools(26), "инструментов")

    def test_second_refresh_rebuilds_without_duplicates(self) -> None:
        """Повторный запрос перестраивает список: без дублей и хвостов."""
        self.panel.refresh()
        self._deliver([MCPTool(name=f"tool_{i}") for i in range(3)])
        first_count = len(self.panel.tools_frame.winfo_children())
        # Второй запрос: другой набор (проверка, что старые карточки стёрты)
        self._reset_deliver()
        self.panel.refresh()
        self._deliver([MCPTool(name="only_one")])
        self.assertEqual(len(self.panel.tools), 1)
        second_count = len(self.panel.tools_frame.winfo_children())
        self.assertEqual(second_count, 1)
        self.assertEqual(first_count, 3)

    def test_error_status_and_messagebox(self) -> None:
        """Ошибка воркера: красный статус + messagebox, кнопка разблокирована."""
        with mock.patch("src.mcp_panel.messagebox") as mb:
            self.panel.refresh()
            self._deliver(MCPConnectionError("Не удалось подключиться"))
            mb.showerror.assert_called_once()
        self.assertEqual(self.panel.get_status_kind(), STATUS_ERROR)
        self.assertIn("Ошибка:", self.panel.get_status_text())
        self.assertEqual(str(self.panel.refresh_btn.cget("state")), "normal")

    def test_nodejs_error_appends_download_link(self) -> None:
        """Ошибка «npx не найден» дополняется ссылкой на nodejs.org."""
        with mock.patch("src.mcp_panel.messagebox") as mb:
            self.panel.refresh()
            self._deliver(
                "Команда npx не найдена. Установите Node.js: https://nodejs.org"
            )
            args, _kwargs = mb.showerror.call_args
            self.assertIn("https://nodejs.org", args[1])

    def test_empty_tool_list_is_not_error(self) -> None:
        """Пустой список — не ошибка: зелёный статус «0 инструментов»."""
        self.panel.refresh()
        self._deliver([])
        self.assertEqual(self.panel.get_status_kind(), STATUS_CONNECTED)
        self.assertIn("0 инструментов", self.panel.get_status_text())

    def test_schema_dialog_shows_name_description_json(self) -> None:
        """Модалка inputSchema: имя, описание, JSON, кнопка копирования."""
        tool = MCPTool(
            name="wikipedia_summary",
            description="Сводка статьи",
            input_schema={"type": "object", "properties": {"title": {"type": "string"}}},
        )
        dialog = self.panel.open_schema_dialog(tool)
        self.addCleanup(dialog.destroy)
        self.root.update()
        self.assertIn("wikipedia_summary", dialog.title())
        texts = [w for w in dialog.winfo_children() if isinstance(w, tk.Text)]
        self.assertEqual(len(texts), 1)
        content = texts[0].get("1.0", "end-1c")
        self.assertIn('"title"', content)
        buttons = self._find_buttons(dialog)
        self.assertTrue(any("Копировать JSON" in b for b in buttons))
        self.assertTrue(any("Закрыть" in b for b in buttons))

    def test_copy_button_puts_json_to_clipboard(self) -> None:
        tool = MCPTool(name="t", input_schema={"type": "object"})
        dialog = self.panel.open_schema_dialog(tool)
        self.addCleanup(dialog.destroy)
        self.root.update()
        copy_btn = next(
            btn for btn in self._find_button_widgets(dialog)
            if "Копировать" in str(btn.cget("text"))
        )
        copy_btn.invoke()
        self.root.update()
        self.assertIn('"type"', self.root.clipboard_get())

    def _find_button_widgets(self, widget) -> list:
        found = []
        for child in widget.winfo_children():
            if isinstance(child, tk.Button):
                found.append(child)
            found.extend(self._find_button_widgets(child))
        return found

    def _find_buttons(self, widget) -> list:
        return [str(b.cget("text")) for b in self._find_button_widgets(widget)]

    def test_collapse_hides_body_and_persists(self) -> None:
        """Сворачивание: тело скрыто, состояние записано в app_state."""
        store = self.panel.app_state
        self.panel.set_collapsed(True)
        self.root.update()
        self.assertEqual(self.panel.body.winfo_manager(), "")  # не упакован
        self.assertEqual(self.panel.toggle_btn.cget("text"), "▶")
        self.assertTrue(store.get_bool("mcp_panel_collapsed"))
        # Разворачивание возвращает всё как было
        self.panel.set_collapsed(False)
        self.root.update()
        self.assertNotEqual(self.panel.body.winfo_manager(), "")
        self.assertFalse(store.get_bool("mcp_panel_collapsed"))

    def test_collapsed_state_restored_for_new_panel(self) -> None:
        """Новая панель с тем же app_state стартует свёрнутой."""
        self.panel.set_collapsed(True)
        settings = _panel_settings(Path(self.tmp.name))
        restored = MCPPanel(
            self.root, settings=settings,
            app_state=self.panel.app_state,
            worker_factory=self._fake_factory,
        )
        self.addCleanup(restored.destroy)
        restored.pack()
        restored.apply_collapsed_state()
        self.assertTrue(restored.collapsed)
        self.assertEqual(restored.body.winfo_manager(), "")

    def test_destroy_during_loading_is_safe(self) -> None:
        """Разрушение панели во время «зависшего» запроса не падает."""
        self._result = None  # воркер «висит»
        self.panel.refresh()
        self.panel.destroy()
        self.root.update()  # после-задачи отменены, исключений нет

    def test_busy_refresh_ignored(self) -> None:
        """Повторный клик во время загрузки игнорируется (одна сессия)."""
        self.panel.refresh()  # воркер «висит» (ответ не подан)
        first_poll_job = self.panel._poll_job
        self.panel.refresh()  # должен быть проигнорирован
        self.assertIs(self.panel._poll_job, first_poll_job)
        self._deliver([MCPTool(name="late")])
        self.assertEqual(len(self.panel.tools), 1)

    def test_ui_stays_responsive_during_loading(self) -> None:
        """Пока запрос идёт, поток событий Tk жив (UI не блокируется)."""
        self.panel.refresh()  # воркер «висит» (ответ не подан)
        markers = []
        self.root.after(10, lambda: markers.append("tick"))
        self.root.update()  # обычный цикл событий работает
        deadline = time.monotonic() + 2.0
        while not markers and time.monotonic() < deadline:
            self.root.update()
        self.assertEqual(markers, ["tick"])
        self._deliver([MCPTool(name="done")])
        self.assertEqual(self.panel.get_status_kind(), STATUS_CONNECTED)


# ---------------------------------------------------------------------------
# Панель внутри ChatApplication (интеграция со вкладкой «Чат»)
# ---------------------------------------------------------------------------

class ChatAppMCPIntegrationTest(unittest.TestCase):
    """MCP-панель встроена в чат-вкладку и уважает MCP_ENABLED."""

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
            "MCP_ENABLED": True,
            "APP_STATE_PATH": str(Path(tempfile.mkdtemp()) / "state.db"),
        })
        config.update(config_overrides or {})
        app = ChatApplication(root=self.root, agent=agent, config=config)
        self.addCleanup(self.root.update)
        return app

    def test_panel_visible_next_to_chat(self) -> None:
        """Панель существует, размещена в чат-вкладке, статус «Не подключён»."""
        app = self._make_app()
        self.assertIsNotNone(app.mcp_panel)
        panel = app.mcp_panel
        self.assertEqual(panel.winfo_manager(), "pack")
        pack_info = panel.pack_info()
        self.assertEqual(pack_info["side"], "right")
        self.assertEqual(panel.get_status_kind(), STATUS_IDLE)
        self.assertEqual(panel.get_status_text(), "Не подключён")

    def test_panel_left_of_chat_scrollbar(self) -> None:
        """Панель правее области чата: x-координата панели больше координаты чата."""
        app = self._make_app()
        self.root.update()
        self.assertGreater(
            app.mcp_panel.winfo_rootx(), app.chat.winfo_rootx()
        )

    def test_disabled_flag_removes_panel(self) -> None:
        """MCP_ENABLED=false — панель не создаётся, чат работает."""
        app = self._make_app({"MCP_ENABLED": False})
        self.assertIsNone(app.mcp_panel)
        self.assertTrue(app.chat.winfo_exists())

    def test_app_close_with_busy_panel_is_safe(self) -> None:
        """Закрытие приложения во время загрузки не блокирует выход."""
        app = self._make_app()
        panel = app.mcp_panel
        # Имитируем «висящий» воркер через реальный поток с длинным клиентом
        started = threading.Event()

        class _HangingClient:
            def list_tools(self):
                started.set()
                time.sleep(5)
                return []

        worker = MCPWorker(
            on_success=lambda tools: None,
            on_error=lambda msg: None,
            client=_HangingClient(),
        )
        panel._busy = True
        panel._make_worker = lambda on_success, on_error: worker
        worker.start()
        self.assertTrue(started.wait(2.0))
        panel.destroy()  # закрытие окна приложения
        self.root.update()
        # daemon-поток не мешает процессу завершиться
        self.assertTrue(worker.daemon)


if __name__ == "__main__":
    unittest.main()
