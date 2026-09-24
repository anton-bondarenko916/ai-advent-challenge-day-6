"""MCP-клиент напоминаний: HTTP-транспорт (Streamable HTTP) + SSE.

Отличие от :mod:`src.mcp_client` (stdio): сервер напоминаний — это
долгоживущий процесс ``reminder_server.py`` на ``http://127.0.0.1:<порт>/mcp``,
он НЕ запускается клиентом как подпроцесс. Подключение — официальный
``streamablehttp_client`` + ``ClientSession``:

* :meth:`ReminderMCPClient.list_tools` — tools/list (для LLM и панели);
* :meth:`ReminderMCPClient.call_tool` — tools/call (tool calling агента);
* :class:`ReminderNotificationListener` — постоянная сессия, слушающая
  SSE-поток: сервер рассылает ``notifications/message`` при срабатывании
  напоминаний (поток планировщика → session.send_log_message).

Каждый вызов list_tools/call_tool — отдельная HTTP-сессия (подключение →
initialize → запрос → закрытие): предсказуемо и не оставляет соединений.
Слушатель, наоборот, держит одну сессию до остановки; потеря связи
лечится переподключением с паузой (сервер живёт 24/7 и может
перезапускаться независимо от приложения).

Все ошибки SDK оборачиваются в :class:`MCPError` и наследники — те же
исключения, что использует stdio-клиент, поэтому UI показывает понятные
сообщения («Сервер напоминаний не запущен…»).
"""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional

from .mcp_client import (
    MCP_SDK_AVAILABLE,
    MCPConnectionError,
    MCPError,
    MCPTimeoutError,
)
from .mcp_models import MCPTool, sort_tools

try:  # HTTP-транспорт SDK (модуль должен импортироваться и без mcp)
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    HTTP_AVAILABLE = True
except ImportError:  # pragma: no cover — окружение без mcp
    ClientSession = None  # type: ignore[assignment]
    streamablehttp_client = None  # type: ignore[assignment]
    HTTP_AVAILABLE = False

logger_name = "deepseek-chat.reminder"

DEFAULT_URL = "http://127.0.0.1:8000/mcp"

# Таймаут handshake и tools/call: сервер локальный, долгих ответов нет.
_REQUEST_TIMEOUT = 15.0

# Таймаут простоя SSE-потока слушателя: уведомления могут не приходить
# часами, поэтому стандартные 5 минут SDK не подходят.
_LISTEN_READ_TIMEOUT = timedelta(hours=24)

# Пауза между переподключениями слушателя (сервер мог быть не поднят ещё).
_RECONNECT_DELAY = 5.0

# Интервал повторной регистрации сессии на сервере (see _listen_once):
# заодно служит health-check — упавший сервер рвёт сессию, слушатель
# переподключается и снова встаёт в реестр подписчиков.
_REGISTRATION_INTERVAL = 60.0


def extract_call_text(result: Any) -> str:
    """Извлекает текст результата CallToolResult SDK (аналог MCPClient).

    Приоритет: ``structuredContent`` (FastMCP оборачивает значение в
    ``{"result": ...}``) → текстовые блоки ``content`` → «Пустой результат».
    """
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return json.dumps(structured, ensure_ascii=False, indent=2)
    content = getattr(result, "content", None)
    blocks = list(content) if content else []
    parts = [
        str(getattr(block, "text", ""))
        for block in blocks
        if getattr(block, "text", None)
    ]
    if not parts:
        return "Пустой результат"
    return "\n".join(parts)


def _connection_error(url: str, exc: Exception) -> MCPConnectionError:
    """Понятное сообщение об ошибке подключения (подсказка для UI)."""
    return MCPConnectionError(
        f"Сервер напоминаний не запущен ({url}). "
        f"Нажмите «Перезапустить сервер» в окне приложения. "
        f"Детали: {exc}"
    )


class ReminderMCPClient:
    """HTTP-клиент MCP-сервера напоминаний (tools/list, tools/call).

    Attributes:
        server_url: адрес MCP-эндпоинта (например,
            ``http://127.0.0.1:8000/mcp``).
    """

    def __init__(self, server_url: str = DEFAULT_URL) -> None:
        """Настраивает клиент.

        Args:
            server_url: URL MCP-эндпоинта сервера.
        """
        self.server_url = str(server_url or DEFAULT_URL)

    def _require_sdk(self) -> None:
        """Проверяет доступность SDK (иначе — понятная ошибка)."""
        if not (MCP_SDK_AVAILABLE and HTTP_AVAILABLE):
            raise MCPConnectionError(
                "Python SDK 'mcp' не установлен. Выполните: "
                "pip install 'mcp>=1.28,<2'"
            )

    async def list_tools(self) -> List[MCPTool]:
        """Запрашивает tools/list по HTTP (отдельная сессия).

        Returns:
            Список MCPTool, отсортированный по имени.

        Raises:
            MCPTimeoutError: сервер не ответил за таймаут запроса.
            MCPConnectionError: сервер недоступен или ответ некорректен.
        """
        self._require_sdk()
        try:
            async with streamablehttp_client(self.server_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await asyncio.wait_for(
                        session.initialize(), timeout=_REQUEST_TIMEOUT
                    )
                    response = await asyncio.wait_for(
                        session.list_tools(), timeout=_REQUEST_TIMEOUT
                    )
        except asyncio.TimeoutError as exc:
            raise MCPTimeoutError(
                f"Сервер напоминаний не ответил за {_REQUEST_TIMEOUT:.0f} с"
            ) from exc
        except MCPError:
            raise
        except Exception as exc:
            raise _connection_error(self.server_url, exc) from exc

        raw_tools = getattr(response, "tools", None)
        if raw_tools is None:
            raise MCPConnectionError(
                "Сервер напоминаний вернул некорректный ответ: "
                "отсутствует поле tools"
            )
        tools: List[MCPTool] = []
        for tool in raw_tools:
            name = str(getattr(tool, "name", "") or "").strip()
            if not name:
                continue
            schema = getattr(tool, "inputSchema", None)
            tools.append(MCPTool(
                name=name,
                description=str(getattr(tool, "description", "") or ""),
                input_schema=dict(schema) if isinstance(schema, dict) else None,
            ))
        return sort_tools(tools)

    async def call_tool(self, tool_name: str,
                        arguments: Optional[Dict[str, Any]] = None) -> str:
        """Вызывает инструмент по HTTP (отдельная сессия).

        Args:
            tool_name: имя инструмента на сервере (``add_reminder`` и др.).
            arguments: словарь аргументов; None/пустой — без аргументов.

        Returns:
            Текст результата (pretty-JSON structuredContent либо склеенные
            текстовые блоки). ``isError``-ответы возвращаются как обычный
            текст — решение принимает вызывающий код (агент/UI).

        Raises:
            MCPTimeoutError: сервер не ответил за таймаут запроса.
            MCPConnectionError: сервер недоступен.
        """
        self._require_sdk()
        args = dict(arguments) if arguments else {}
        try:
            async with streamablehttp_client(self.server_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await asyncio.wait_for(
                        session.initialize(), timeout=_REQUEST_TIMEOUT
                    )
                    result = await asyncio.wait_for(
                        session.call_tool(tool_name, args),
                        timeout=_REQUEST_TIMEOUT,
                    )
        except asyncio.TimeoutError as exc:
            raise MCPTimeoutError(
                f"Инструмент {tool_name} не ответил за "
                f"{_REQUEST_TIMEOUT:.0f} с"
            ) from exc
        except MCPError:
            raise
        except Exception as exc:
            raise _connection_error(self.server_url, exc) from exc
        return extract_call_text(result)


class ReminderNotificationListener(threading.Thread):
    """Фоновый поток: держит MCP-сессию и слушает SSE-уведомления.

    Уведомления приходят как ``notifications/message`` (LoggingMessage)
    из потока планировщика reminder_server. Поток-слушатель переживает
    разрывы связи: при ошибке переподключается через ``_RECONNECT_DELAY``
    секунд, пока не установлен stop-флаг.

    Callbacks вызываются ИЗ потока слушателя (UI обязан маршалинговать
    их в главный поток — очередь + root.after):

    * ``on_notification(text)`` — пришло уведомление (текст ``data``);
    * ``on_status(kind, message)`` — смена состояния: ``kind`` = 
      ``"connected"`` | ``"error"`` | ``"stopped"``.

    Attributes:
        server_url: адрес MCP-эндпоинта.
    """

    def __init__(
        self,
        server_url: str = DEFAULT_URL,
        on_notification: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """Настраивает поток (запуск — метод start(), остановка — stop()).

        Args:
            server_url: URL MCP-эндпоинта.
            on_notification: callback уведомления (текст для чата).
            on_status: callback смены состояния (kind, message).
        """
        super().__init__(daemon=True, name="reminder-notifications")
        self.server_url = str(server_url or DEFAULT_URL)
        self._on_notification = on_notification or (lambda text: None)
        self._on_status = on_status or (lambda kind, message: None)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Просит поток завершиться (идемпотентно, не ждёт завершения)."""
        self._stop_event.set()

    def run(self) -> None:
        """Точка входа потока: цикл «подключиться → слушать → переподключиться».

        Никогда не бросает исключения наружу: всё уходит в on_status.
        """
        if not (MCP_SDK_AVAILABLE and HTTP_AVAILABLE):
            self._on_status("error", "Python SDK 'mcp' не установлен")
            return
        while not self._stop_event.is_set():
            try:
                asyncio.run(self._listen_once())
                # _listen_once вернулся — остановка по флагу
                break
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                logger = _logger()
                logger.warning("Слушатель напоминаний: %s", exc)
                self._on_status("error", f"Нет связи с сервером: {exc}")
                if self._stop_event.wait(_RECONNECT_DELAY):
                    break
        self._on_status("stopped", "Слушатель остановлен")

    async def _listen_once(self) -> None:
        """Одно подключение: initialize → регистрация → приём уведомлений.

        Регистрация сессии — вызов ``get_stats``: напоминание серверу
        включить эту сессию в SSE-рассылку (см. reminder_server._register_session).
        Дальше раз в ``_REGISTRATION_INTERVAL`` регистрация повторяется
        (сервер мог перезапуститься и потерять реестр подписчиков).
        """
        async with streamablehttp_client(
            self.server_url, sse_read_timeout=_LISTEN_READ_TIMEOUT
        ) as (read, write, _):
            async with ClientSession(
                read, write, message_handler=_make_message_handler(self)
            ) as session:
                await asyncio.wait_for(
                    session.initialize(), timeout=_REQUEST_TIMEOUT
                )
                self._on_status(
                    "connected", f"Подключён к {self.server_url}"
                )
                # Первичная регистрация сессии в реестре подписчиков
                await asyncio.wait_for(
                    session.call_tool("get_stats", {}),
                    timeout=_REQUEST_TIMEOUT,
                )
                while not self._stop_event.is_set():
                    # Спим короткими шагами, чтобы stop() отзывался быстро
                    if await self._sleep_interruptible(_REGISTRATION_INTERVAL):
                        break
                    if self._stop_event.is_set():
                        break
                    # Повторная регистрация + health-check соединения
                    await asyncio.wait_for(
                        session.call_tool("get_stats", {}),
                        timeout=_REQUEST_TIMEOUT,
                    )

    async def _sleep_interruptible(self, seconds: float) -> bool:
        """Спит ``seconds`` секунды, отзываясь на stop().

        Returns:
            True — поступил сигнал остановки, False — время истекло.
        """
        steps = max(1, int(seconds))
        for _ in range(steps):
            if self._stop_event.is_set():
                return True
            await asyncio.sleep(min(1.0, seconds / steps))
        return self._stop_event.is_set()


def _make_message_handler(
    listener: ReminderNotificationListener,
):
    """Фабрика обработчика входящих сообщений ClientSession.

    Извлекает текст уведомлений ``notifications/message``
    (LoggingMessageNotification.params.data) и вызывает callback
    слушателя. Прочие сообщения (ответы на запросы и т.п.) игнорируются —
    их обрабатывает сам ClientSession.
    """

    async def handler(message: Any) -> None:
        try:
            root = getattr(message, "root", None)
            if root is None:
                return
            method = str(getattr(root, "method", "") or "")
            if method != "notifications/message":
                return
            params = getattr(root, "params", None)
            data = getattr(params, "data", None)
            if data is None and isinstance(params, dict):
                data = params.get("data")
            if data is None:
                return
            listener._on_notification(str(data))
        except Exception:  # уведомление не должно ломать сессию
            _logger().exception("Ошибка обработки уведомления")

    return handler


def _logger():
    """Логгер модуля (лениво, чтобы не зависеть от utils при импорте)."""
    import logging

    return logging.getLogger(logger_name)
