"""Автозапуск MCP-серверов на HTTP в отдельном окне консоли.

Базовый класс :class:`HttpMCPServerLauncher` управляет запуском
долгоживущего FastMCP-сервера (reminder_server.py, news_server.py):

* :meth:`HttpMCPServerLauncher.is_server_running` — TCP-проверка порта
  (сокет, без HTTP-запроса): мгновенно и не требует SDK;
* :meth:`HttpMCPServerLauncher.launch` — если порт свободен, стартует
  сервер в НОВОМ окне консоли (``CREATE_NEW_CONSOLE``): uvicorn пишет
  туда логи, окно видно пользователю. Если сервер уже работает —
  ничего не делает (повторный запуск приложения не дублирует процесс);
* :meth:`HttpMCPServerLauncher.stop` — пустой метод: закрытие
  приложения сервер НЕ останавливает (серверы живут 24/7).

Почему ``CREATE_NEW_CONSOLE`` вместо ``start cmd /k``: дочерний процесс
получает собственную консоль независимо от родителя и переживает
закрытие приложения без job-объектов и shell-кавычек; логи uvicorn
видны в этом окне. На не-Windows платформах консоль не выделяется
(сервер уходит в фон).
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

from .config import PROJECT_ROOT

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


class HttpMCPServerLauncher:
    """Управляет запуском MCP-сервера на HTTP (без его остановки).

    Attributes:
        host: адрес сервера (только 127.0.0.1 — хостить в интернете
            запрещено).
        port: порт HTTP-сервера.
        server_path: путь к скрипту сервера.
        label: человекочитаемое имя для сообщений («Сервер напоминаний»,
            «News-сервер»).
    """

    def __init__(
        self,
        server_path: Path,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        label: str = "MCP-сервер",
    ) -> None:
        """Настраивает лаунчер.

        Args:
            server_path: путь к скрипту сервера.
            host: адрес сервера.
            port: порт сервера.
            label: имя сервера для сообщений пользователю.
        """
        self.host = str(host or DEFAULT_HOST)
        self.port = int(port or DEFAULT_PORT)
        self.server_path = Path(server_path)
        self.label = str(label or "MCP-сервер")

    @property
    def server_url(self) -> str:
        """MCP-эндпоинт сервера (для HTTP MCP-клиента)."""
        return f"http://{self.host}:{self.port}/mcp"

    def is_server_running(self) -> bool:
        """True, если порт сервера отвечает (TCP-соединение устанавливается).

        Сокет с таймаутом 0.5 с: закрытый порт на localhost отвечает
        моментально, занятый — значит сервер (или кто-то другой) слушает.
        """
        try:
            with socket.create_connection((self.host, self.port), timeout=0.5):
                return True
        except OSError:
            return False

    def launch(self) -> Tuple[bool, str]:
        """Запускает сервер, если он ещё не работает.

        Returns:
            (started, message): started=True — процесс запущен;
            started=False — сервер уже работает (message объясняет это).
            Ошибки запуска не пробрасываются: (False, текст ошибки) —
            приложение должно продолжать работать без инструментов.
        """
        if self.is_server_running():
            return False, (
                f"{self.label} уже запущен на {self.server_url} — "
                f"не перезапускаем."
            )
        if not self.server_path.exists():
            return False, f"Скрипт сервера не найден: {self.server_path}"

        # ВАЖНО: host/port передаются серверу явно — иначе он поднимется
        # на дефолтном порту скрипта, а лаунчер будет ждать его здесь
        command = [
            sys.executable, str(self.server_path),
            "--host", self.host,
            "--port", str(self.port),
        ]
        try:
            if sys.platform == "win32":
                # Новое окно консоли с логами сервера; процесс живёт
                # независимо от приложения (сервер 24/7).
                subprocess.Popen(
                    command,
                    creationflags=subprocess.CREATE_NEW_CONSOLE,  # type: ignore[attr-defined]
                    stdin=subprocess.DEVNULL,
                    cwd=str(PROJECT_ROOT),
                )
            else:
                subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    cwd=str(PROJECT_ROOT),
                )
        except OSError as exc:
            return False, f"Не удалось запустить {self.label.lower()}: {exc}"
        return True, (
            f"{self.label} запущен в новом окне ({self.server_url})."
        )

    def restart(self) -> Tuple[bool, str]:
        """«Перезапуск» по кнопке UI: запуск, если сервер не работает.

        Работающий сервер намеренно НЕ убивается (сервер 24/7, данные не
        должны пострадать); кнопка нужна на случай падения.

        Returns:
            (started, message) — как у :meth:`launch`.
        """
        return self.launch()

    def stop(self) -> None:
        """Ничего не делает: сервер должен жить после закрытия приложения."""
        return None

    def ensure_running(self, timeout: float = 20.0,
                       poll: float = 0.3) -> bool:
        """Гарантирует, что сервер работает: проверяет, при необходимости
        запускает и ждёт выхода на порт.

        Используется как «лечебный» хук HTTP-клиента: когда инструмент
        упёрся в мёртвый сервер, клиент вызывает хук, сервер поднимается,
        и вызов повторяется уже в рамках того же хода диалога.

        Args:
            timeout: сколько секунд ждать выхода сервера на порт.
            poll: интервал опроса порта.

        Returns:
            True — сервер отвечает (работал или поднят), False — не удалось.
        """
        if self.is_server_running():
            return True
        self.launch()
        deadline = time.monotonic() + max(0.0, float(timeout))
        while time.monotonic() < deadline:
            if self.is_server_running():
                return True
            time.sleep(poll)
        return False


class ReminderServerLauncher(HttpMCPServerLauncher):
    """Лаунчер reminder_server.py (напоминания, порт 8000 по умолчанию)."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        server_path: Optional[Path] = None,
    ) -> None:
        """Настраивает лаунчер напоминаний.

        Args:
            host: адрес сервера.
            port: порт сервера.
            server_path: путь к reminder_server.py; None — корень проекта.
        """
        super().__init__(
            server_path=server_path
            or (PROJECT_ROOT / "reminder_server.py"),
            host=host,
            port=port,
            label="Сервер напоминаний",
        )
