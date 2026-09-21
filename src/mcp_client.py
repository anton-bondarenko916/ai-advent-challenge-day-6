"""MCP-клиент: подключение к серверу и запрос списка инструментов.

Обёртка над официальным Python SDK ``mcp`` (ClientSession + stdio_client).
Поддерживается stdio-транспорт: MCP-сервер запускается подпроцессом
(по умолчанию ``npx -y open-internet-mcp``), обмен идёт через stdin/stdout.

На текущем этапе реализован только ``tools/list`` — получение и разбор
списка инструментов. Вызов инструментов (tools/call) добавляется позже.

Все ошибки SDK оборачиваются в собственные исключения
(:class:`MCPError` и наследники), чтобы UI показывал понятные сообщения.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

from . import utils
from .config import PROJECT_ROOT
from .mcp_models import MCPTool, sort_tools

try:  # SDK обязателен для работы клиента, но модуль должен импортироваться
    # и без него (панель UI тогда показывает подсказку об установке).
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    MCP_SDK_AVAILABLE = True
except ImportError:  # pragma: no cover — окружение без mcp
    ClientSession = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]
    stdio_client = None  # type: ignore[assignment]
    MCP_SDK_AVAILABLE = False

logger = utils.get_logger("deepseek-chat.mcp")


class MCPError(Exception):
    """Базовое исключение подсистемы MCP."""


class MCPConnectionError(MCPError):
    """Ошибка подключения к MCP-серверу (запуск, протокол, некорректный ответ)."""


class MCPTimeoutError(MCPError):
    """Сервер не ответил за отведённое время."""


class MCPClient:
    """Управляет stdio-соединением с MCP-сервером.

    Каждый вызов :meth:`list_tools` — отдельная сессия: процесс сервера
    запускается, проходит handshake (initialize), отдаёт tools/list и
    завершается. Для panel refresh это простое и предсказуемое поведение
    (нет фонового процесса между обновлениями).

    Attributes:
        server_command: команда запуска сервера, например
            ``["npx", "-y", "open-internet-mcp"]``.
        timeout: таймаут на handshake и на tools/list, секунды
            (первый ``npx`` скачивает пакет, поэтому по умолчанию 30 с).
        log_path: путь к mcp.log — сюда пишутся события клиента и
            stderr сервера (последний не глушится — нужен при отладке).
    """

    DEFAULT_TIMEOUT = 30.0  # секунд; первый npx может быть медленным

    def __init__(
        self,
        server_command: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        log_path: Optional[str | Path] = None,
    ) -> None:
        """Настраивает клиент.

        Args:
            server_command: команда запуска сервера; None — дефолт
                (open-internet-mcp через npx).
            timeout: таймаут операций; None — DEFAULT_TIMEOUT.
            log_path: файл журнала; None — mcp.log в корне проекта.

        Raises:
            ValueError: команда пуста.
        """
        if server_command is None:
            self.server_command: List[str] = ["npx", "-y", "open-internet-mcp"]
        else:
            self.server_command = list(server_command)
        if not self.server_command or not str(self.server_command[0]).strip():
            raise ValueError("Команда запуска MCP-сервера не задана")
        self.timeout = float(timeout) if timeout else self.DEFAULT_TIMEOUT
        self.log_path = Path(log_path) if log_path else (
            PROJECT_ROOT / "mcp.log"
        )

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    async def list_tools(self) -> List[MCPTool]:
        """Подключается к серверу и запрашивает ``tools/list``.

        Порядок: запуск подпроцесса → initialize (handshake) → list_tools
        → разбор ответа → завершение процесса.

        Returns:
            Список MCPTool, отсортированный по имени. Пустой список не
            считается ошибкой.

        Raises:
            MCPTimeoutError: сервер не ответил за ``self.timeout``.
            MCPConnectionError: команда не найдена, сервер упал, вернул
                некорректный ответ или SDK ``mcp`` не установлен.
        """
        if not MCP_SDK_AVAILABLE:
            raise MCPConnectionError(
                "Python SDK 'mcp' не установлен. Выполните: "
                "pip install 'mcp>=1.28,<2'"
            )

        started = time.monotonic()
        deadline = started + self.timeout
        logger.info(
            "MCP: подключение к серверу (команда: %s, таймаут: %.0f с)",
            " ".join(self.server_command), self.timeout,
        )

        # Наследуем окружение процесса: SDK по умолчанию передаёт серверу
        # урезанный набор переменных (get_default_environment), а npx/npm
        # на Windows нужны APPDATA и LOCALAPPDATA.
        params = StdioServerParameters(
            command=self.server_command[0],
            args=self.server_command[1:],
            env=dict(os.environ),
        )

        errlog = self._open_errlog()
        try:
            async with stdio_client(params, errlog=errlog) as (read, write):
                async with ClientSession(read, write) as session:
                    # Handshake: согласование версии протокола и возможностей.
                    await asyncio.wait_for(
                        session.initialize(), timeout=self.timeout
                    )
                    response = await asyncio.wait_for(
                        session.list_tools(), timeout=self.timeout
                    )
            tools = sort_tools(self._parse_tools(response))
            elapsed = time.monotonic() - started
            logger.info(
                "MCP: получено инструментов: %d за %.1f с", len(tools), elapsed
            )
            return tools
        except asyncio.TimeoutError as exc:
            logger.error("MCP: таймаут %.0f с", self.timeout)
            raise MCPTimeoutError(
                f"Сервер не ответил за {self.timeout:.0f} секунд "
                f"(первый запуск npx может скачивать пакет — попробуйте ещё раз)"
            ) from exc
        except FileNotFoundError as exc:
            logger.error("MCP: команда не найдена: %s", exc)
            raise self._connection_error_from(exc) from exc
        except OSError as exc:
            logger.error("MCP: не удалось запустить сервер: %s", exc)
            raise self._connection_error_from(exc) from exc
        except asyncio.CancelledError:
            # Отмена (закрытие приложения во время загрузки): пробрасываем,
            # чтобы не маскировать её под ошибку подключения.
            raise
        except MCPError:
            raise
        except Exception as exc:
            # SDK может завернуть ошибку в ExceptionGroup при разборке
            # сессии; если дедлайн истёк — это наш таймаут.
            if time.monotonic() >= deadline:
                raise MCPTimeoutError(
                    f"Сервер не ответил за {self.timeout:.0f} секунд"
                ) from exc
            logger.error("MCP: ошибка подключения: %s", exc)
            raise MCPConnectionError(
                f"Не удалось подключиться к MCP-серверу: {exc}"
            ) from exc
        finally:
            if errlog is not sys.stderr:
                errlog.close()

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _parse_tools(self, response: Any) -> List[MCPTool]:
        """Разбирает ответ SDK (ListToolsResult) в список MCPTool.

        Args:
            response: объект с полем ``.tools`` — коллекция объектов с
                полями ``name``, ``description``, ``inputSchema``.

        Returns:
            Список MCPTool.

        Raises:
            MCPConnectionError: ответ не имеет поля ``tools`` (нарушение
                протокола).
        """
        raw_tools = getattr(response, "tools", None)
        if raw_tools is None:
            raise MCPConnectionError(
                "Сервер вернул некорректный ответ: отсутствует поле tools"
            )
        tools: List[MCPTool] = []
        for tool in raw_tools:
            name = str(getattr(tool, "name", "") or "").strip()
            if not name:
                continue  # инструмент без имени не показываем
            schema = getattr(tool, "inputSchema", None)
            tools.append(MCPTool(
                name=name,
                description=str(getattr(tool, "description", "") or ""),
                input_schema=dict(schema) if isinstance(schema, dict) else None,
            ))
        return tools

    def _connection_error_from(self, exc: Exception) -> MCPConnectionError:
        """Преобразует ошибку запуска процесса в понятное сообщение.

        Для npx-команд добавляется подсказка про установку Node.js —
        самая частая причина FileNotFoundError на машине пользователя.
        """
        base = self.server_command[0].lower()
        is_npx = base in ("npx", "npx.cmd", "npx.exe", "npx.bat")
        if isinstance(exc, FileNotFoundError) and is_npx:
            return MCPConnectionError(
                "Команда npx не найдена. Установите Node.js: "
                "https://nodejs.org"
            )
        return MCPConnectionError(
            f"Не удалось запустить MCP-сервер "
            f"({' '.join(self.server_command)}): {exc}"
        )

    def _open_errlog(self):
        """Открывает файл журнала для stderr сервера (или stderr процесса).

        stderr сервера НЕ глушится: туда npm/npx пишут прогресс скачивания
        и ошибки запуска — без этого «тихие» падения невозможно отладить.
        """
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            return open(
                self.log_path, "a", encoding="utf-8", errors="replace"
            )
        except OSError:
            logger.warning(
                "MCP: не удалось открыть журнал %s — stderr идёт в stderr "
                "процесса", self.log_path,
            )
            return sys.stderr
