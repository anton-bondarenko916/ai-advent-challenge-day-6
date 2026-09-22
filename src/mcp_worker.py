"""Фоновые потоки для операций MCP.

Tkinter не потокобезопасен, а ``asyncio.run`` нельзя вызывать в главном
потоке GUI. Воркеры совмещают оба требования: выполняют асинхронный
клиент в отдельном daemon-потоке и доставляют результат через callback.
Потокобезопасную доставку в UI обеспечивает панель (MCPPanel кладёт
callback-результат в queue.Queue и опрашивает её через ``root.after`` —
тот же паттерн, что у подсистемы задач в ui.py).

Два воркера:

* :class:`MCPWorker` — запрос списка инструментов (tools/list);
* :class:`MCPToolCallWorker` — вызов одного инструмента (tools/call),
  используется диалогом «Вызвать» в панели MCP.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable, Dict, List, Optional

from .mcp_client import MCPClient, MCPError
from .mcp_models import MCPTool


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
