"""Фоновые потоки для операций MCP.

Tkinter не потокобезопасен, а ``asyncio.run`` нельзя вызывать в главном
потоке GUI. Воркеры совмещают оба требования: выполняют асинхронный
клиент в отдельном daemon-потоке и доставляют результат через callback.
Потокобезопасную доставку в UI обеспечивает панель (MCPPanel кладёт
callback-результат в queue.Queue и опрашивает её через ``root.after`` —
тот же паттерн, что у подсистемы задач в ui.py).

Четыре воркера:

* :class:`MCPWorker` — запрос списка инструментов (tools/list) одного
  stdio-сервера;
* :class:`MultiServerListWorker` — tools/list со ВСЕХ серверов (stdio +
  HTTP): напоминания, news, geo, weather; возвращает группы
  :class:`src.mcp_models.ServerTools` — панель рисует их с заголовками
  серверов;
* :class:`MCPToolCallWorker` — вызов инструмента на stdio-сервере
  (tools/call), используется диалогом «Вызвать» в панели MCP;
* :class:`MCPRemoteCallWorker` — вызов инструмента на HTTP-сервере
  (тот же диалог, но для групп с url).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable, Dict, List, Optional

from .mcp_client import MCPClient, MCPError
from .mcp_models import MCPTool, ServerTools, sort_tools


class MCPWorker(threading.Thread):
    """Запускает async MCP-клиент в отдельном потоке.

    По завершении вызывает ровно один callback:

    * ``on_success(tools)`` — сервер ответил, ``tools`` — список MCPTool;
    * ``on_error(message)`` — любая ошибка (MCPError или неожиданная).

    Callbacks вызываются ИЗ потока воркера: UI обязан маршалинговать их
    в главный поток (очередь + root.after, см. MCPPanel).

    Attributes:
        client: экземпляр MCPClient (создаётся из аргументов конструктора
            либо передан готовый — удобно в тестах).
    """

    def __init__(
        self,
        on_success: Callable[[List[MCPTool]], None],
        on_error: Callable[[str], None],
        server_command: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        log_path: Optional[str] = None,
        client: Optional[MCPClient] = None,
    ) -> None:
        """Настраивает поток.

        Args:
            on_success: callback успешного получения списка инструментов.
            on_error: callback ошибки (текст сообщения для пользователя).
            server_command: команда запуска сервера (см. MCPClient).
            timeout: таймаут операций (см. MCPClient).
            log_path: путь к mcp.log (см. MCPClient).
            client: готовый клиент; перекрывает предыдущие три аргумента.
        """
        super().__init__(daemon=True, name="mcp-client")
        self.on_success = on_success
        self.on_error = on_error
        self.client = client or MCPClient(
            server_command=server_command,
            timeout=timeout,
            log_path=log_path,
        )

    def run(self) -> None:
        """Точка входа потока: list_tools → callback.

        Никогда не бросает исключения наружу: поток не должен «падать»
        молча — любая ошибка уходит в on_error.
        """
        try:
            tools = asyncio.run(self.client.list_tools())
            self.on_success(tools)
        except MCPError as exc:
            self.on_error(str(exc))
        except Exception as exc:  # страховка: ошибки SDK и окружения
            self.on_error(f"Неожиданная ошибка MCP: {exc}")


class MCPToolCallWorker(threading.Thread):
    """Вызывает один инструмент MCP-сервера в отдельном потоке.

    По завершении вызывает ровно один callback:

    * ``on_success(text)`` — инструмент ответил, ``text`` — текст
      результата (JSON-строка для инструментов FastMCP);
    * ``on_error(message)`` — ошибка подключения/таймаут/неожиданная.

    Callbacks вызываются ИЗ потока воркера (см. маршалинг в MCPPanel).

    Attributes:
        tool_name: имя вызываемого инструмента.
        arguments: словарь аргументов инструмента.
        client: экземпляр MCPClient (аналогично MCPWorker).
    """

    def __init__(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]],
        on_success: Callable[[str], None],
        on_error: Callable[[str], None],
        server_command: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        log_path: Optional[str] = None,
        client: Optional[MCPClient] = None,
    ) -> None:
        """Настраивает поток.

        Args:
            tool_name: имя инструмента (например, ``get_git_log``).
            arguments: аргументы инструмента (None — без аргументов).
            on_success: callback текста результата.
            on_error: callback ошибки (текст сообщения для пользователя).
            server_command: команда запуска сервера (см. MCPClient).
            timeout: таймаут операций (см. MCPClient).
            log_path: путь к mcp.log (см. MCPClient).
            client: готовый клиент; перекрывает предыдущие три аргумента.
        """
        super().__init__(daemon=True, name="mcp-tool-call")
        self.tool_name = tool_name
        self.arguments = dict(arguments) if arguments else {}
        self.on_success = on_success
        self.on_error = on_error
        self.client = client or MCPClient(
            server_command=server_command,
            timeout=timeout,
            log_path=log_path,
        )

    def run(self) -> None:
        """Точка входа потока: call_tool → callback (без исключений наружу)."""
        try:
            text = asyncio.run(self.client.call_tool(self.tool_name, self.arguments))
            self.on_success(text)
        except MCPError as exc:
            self.on_error(str(exc))
        except Exception as exc:  # страховка: ошибки SDK и окружения
            self.on_error(f"Неожиданная ошибка MCP: {exc}")


def stdio_group_name(server_command: Optional[List[str]]) -> str:
    """Отображаемое имя stdio-сервера для заголовка группы панели.

    Из команды берётся последний токен: для скриптов — имя файла без
    ``.py`` (``python -u my_git_server.py`` → ``my_git_server``), для
    npx-пакетов — имя пакета (``npx -y open-internet-mcp`` →
    ``open-internet-mcp``).
    """
    command = [str(part) for part in (server_command or []) if str(part)]
    if not command:
        return "stdio"
    tail = command[-1].replace("\\", "/").rsplit("/", 1)[-1]
    if tail.lower().endswith(".py"):
        tail = tail[:-3]
    return tail or "stdio"


class MultiServerListWorker(threading.Thread):
    """Собирает tools/list со всех MCP-серверов (stdio + HTTP).

    Возвращает группы даже при частичных ошибках: сервер, не ответивший
    на tools/list, попадает в результат с ``error`` и пустым списком —
    панель показывает остальные серверы и помечает упавший красным
    (диалог живёт без его инструментов, как и в agentic loop агента).

    По завершении вызывает ровно один callback:

    * ``on_success(groups)`` — опрос завершён (успешные и упавшие
      группы вместе), ``groups`` — список ServerTools;
    * ``on_error(message)`` — только неожиданная ошибка самого потока.

    Callbacks вызываются ИЗ потока воркера (см. маршалинг в MCPPanel).

    Attributes:
        stdio_client: MCPClient stdio-сервера; None — панели без stdio.
        http_clients: список (имя, ReminderMCPClient) HTTP-серверов.
    """

    def __init__(
        self,
        on_success: Callable[[List[ServerTools]], None],
        on_error: Callable[[str], None],
        server_command: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        log_path: Optional[str] = None,
        http_sources: Optional[List[Dict[str, Any]]] = None,
        stdio_client: Optional[MCPClient] = None,
        http_clients: Optional[List[Any]] = None,
    ) -> None:
        """Настраивает поток.

        Args:
            on_success: callback списка групп ServerTools.
            on_error: callback неожиданной ошибки (текст для пользователя).
            server_command: команда stdio-сервера; None/пусто — без stdio.
            timeout: таймаут операций stdio-клиента.
            log_path: путь к mcp.log stdio-клиента.
            http_sources: список ``{"name", "url"}`` HTTP-серверов.
            stdio_client: готовый stdio-клиент; перекрывает команду.
            http_clients: готовые HTTP-клиенты (имя, клиент) — для тестов;
                перекрывает http_sources.
        """
        super().__init__(daemon=True, name="mcp-multi-list")
        self.on_success = on_success
        self.on_error = on_error
        self.stdio_client = stdio_client or (
            MCPClient(
                server_command=list(server_command),
                timeout=timeout,
                log_path=log_path,
            )
            if server_command else None
        )
        if http_clients is not None:
            self.http_clients = list(http_clients)
        else:
            # Ленивый импорт: reminder_client тянет HTTP-транспорт SDK
            from .reminder_client import ReminderMCPClient

            self.http_clients = [
                (str(source.get("name") or "mcp"),
                 ReminderMCPClient(server_url=str(source.get("url") or "")))
                for source in (http_sources or [])
                if source.get("url")
            ]

    def run(self) -> None:
        """Точка входа потока: опрос всех серверов → callback.

        Никогда не бросает исключения наружу: ошибки отдельных серверов
        становятся полями ``error`` их групп.
        """
        groups: List[ServerTools] = []
        try:
            if self.stdio_client is not None:
                name = stdio_group_name(
                    getattr(self.stdio_client, "server_command", None)
                )
                try:
                    tools = asyncio.run(self.stdio_client.list_tools())
                    groups.append(ServerTools(name=name, tools=tools))
                except Exception as exc:
                    # MCPError и «странные» ошибки одинаково помечают группу
                    groups.append(ServerTools(name=name, error=str(exc)))
            for name, client in self.http_clients:
                url = getattr(client, "server_url", "")
                try:
                    tools = asyncio.run(client.list_tools())
                    groups.append(ServerTools(
                        name=name,
                        tools=sort_tools(tools),
                        url=url,
                    ))
                except MCPError as exc:
                    groups.append(ServerTools(
                        name=name, error=str(exc), url=url,
                    ))
                except Exception as exc:
                    # «Странный» сервер не должен ронять опрос остальных
                    groups.append(ServerTools(
                        name=name, error=str(exc), url=url,
                    ))
        except Exception as exc:  # страховка: ошибки SDK и окружения
            self.on_error(f"Неожиданная ошибка MCP: {exc}")
            return
        self.on_success(groups)


class MCPRemoteCallWorker(threading.Thread):
    """Вызывает инструмент на HTTP MCP-сервере в отдельном потоке.

    Аналог :class:`MCPToolCallWorker`, но клиент — HTTP
    (:class:`src.reminder_client.ReminderMCPClient`): кнопка «Вызвать»
    в панели работает и для инструментов geo/weather/напоминаний/news.

    По завершении вызывает ровно один callback:

    * ``on_success(text)`` — инструмент ответил;
    * ``on_error(message)`` — ошибка подключения/таймаут/неожиданная.

    Callbacks вызываются ИЗ потока воркера (см. маршалинг в MCPPanel).

    Attributes:
        client: HTTP-клиент сервера.
        tool_name: имя вызываемого инструмента.
        arguments: словарь аргументов инструмента.
    """

    def __init__(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]],
        on_success: Callable[[str], None],
        on_error: Callable[[str], None],
        server_url: str = "",
        client: Optional[Any] = None,
    ) -> None:
        """Настраивает поток.

        Args:
            tool_name: имя инструмента на сервере.
            arguments: аргументы инструмента (None — без аргументов).
            on_success: callback текста результата.
            on_error: callback ошибки (текст сообщения для пользователя).
            server_url: MCP-эндпоинт сервера; клиент создаётся из него,
                если не передан готовый (удобно в тестах).
            client: готовый HTTP-клиент; перекрывает server_url.
        """
        super().__init__(daemon=True, name="mcp-remote-call")
        self.tool_name = tool_name
        self.arguments = dict(arguments) if arguments else {}
        self.on_success = on_success
        self.on_error = on_error
        if client is not None:
            self.client = client
        else:
            from .reminder_client import ReminderMCPClient

            self.client = ReminderMCPClient(
                server_url=server_url,
                label="MCP-сервер",
            )

    def run(self) -> None:
        """Точка входа потока: call_tool → callback (без исключений наружу)."""
        try:
            text = asyncio.run(
                self.client.call_tool(self.tool_name, self.arguments)
            )
            self.on_success(text)
        except MCPError as exc:
            self.on_error(str(exc))
        except Exception as exc:  # страховка: ошибки SDK и окружения
            self.on_error(f"Неожиданная ошибка MCP: {exc}")
