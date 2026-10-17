"""Автозапуск MCP-сервера файловой системы (fs_server.py) в отдельном окне.

Тонкая настройка базового :class:`HttpMCPServerLauncher`
(см. src/reminder_launcher.py) под fs-сервер: скрипт в корне проекта,
порт из настроек (FS_SERVER_PORT, по умолчанию 8005 — 8000–8004 заняты
reminder/news/geo/weather/crm), сервер живёт 24/7 — повторный запуск
приложения его не дублирует, закрытие приложения не останавливает.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import PROJECT_ROOT
from .reminder_launcher import DEFAULT_HOST, HttpMCPServerLauncher

DEFAULT_PORT = 8005


class FsServerLauncher(HttpMCPServerLauncher):
    """Лаунчер fs_server.py (чтение/запись/поиск по файлам проекта)."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        server_path: Optional[Path] = None,
    ) -> None:
        """Настраивает лаунчер fs-сервера.

        Args:
            host: адрес сервера.
            port: порт сервера.
            server_path: путь к fs_server.py; None — корень проекта.
        """
        super().__init__(
            server_path=server_path or (PROJECT_ROOT / "fs_server.py"),
            host=host,
            port=port,
            label="FS-сервер",
        )
