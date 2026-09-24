"""Автозапуск MCP-сервера напоминаний в отдельном окне консоли.

Сервер ``reminder_server.py`` живёт 24/7 независимо от приложения:

* :meth:`ReminderServerLauncher.is_server_running` — TCP-проверка порта
  (сокет, без HTTP-запроса): мгновенно и не требует SDK;
* :meth:`ReminderServerLauncher.launch` — если порт свободен, стартует
  сервер в НОВОМ окне консоли (``CREATE_NEW_CONSOLE``): uvicorn пишет
  туда логи, окно видно пользователю. Если сервер уже работает —
  ничего не делает (повторный запуск приложения не дублирует процесс);
* :meth:`ReminderServerLauncher.stop` — пустой метод: закрытие
  приложения сервер НЕ останавливает.

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
from pathlib import Path
from typing import Optional, Tuple

from .config import PROJECT_ROOT

SERVER_SCRIPT = PROJECT_ROOT / "reminder_server.py"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


class ReminderServerLauncher:
    """Управляет запуском reminder_server.py (без его остановки).

    Attributes:
        host: адрес сервера (только 127.0.0.1 — хостить в интернете
            запрещено).
        port: порт HTTP-сервера.
        server_path: путь к reminder_server.py.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        server_path: Optional[Path] = None,
    ) -> None:
        """Настраивает лаунчер.

        Args:
            host: адрес сервера.
            port: порт сервера.
            server_path: путь к скрипту сервера; None — reminder_server.py
                в корне проекта.
        """
        self.host = str(host or DEFAULT_HOST)
        self.port = int(port or DEFAULT_PORT)
        self.server_path = Path(server_path) if server_path else SERVER_SCRIPT

    @property
    def server_url(self) -> str:
        """MCP-эндпоинт сервера (для ReminderMCPClient)."""
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
            приложение должно продолжать работать без напоминаний.
        """
        if self.is_server_running():
            return False, (
                f"Сервер напоминаний уже запущен на {self.server_url} — "
                f"не перезапускаем."
            )
        if not self.server_path.exists():
            return False, f"Скрипт сервера не найден: {self.server_path}"

        command = [sys.executable, str(self.server_path)]
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
            return False, f"Не удалось запустить сервер напоминаний: {exc}"
        return True, (
            f"Сервер напоминаний запущен в новом окне ({self.server_url})."
        )

    def restart(self) -> Tuple[bool, str]:
        """«Перезапуск» по кнопке UI: запуск, если сервер не работает.

        Работающий сервер намеренно НЕ убивается (сервер 24/7, данные в
        SQLite не должны пострадать); кнопка нужна на случай падения.

        Returns:
            (started, message) — как у :meth:`launch`.
        """
        return self.launch()

    def stop(self) -> None:
        """Ничего не делает: сервер должен жить после закрытия приложения."""
        return None
