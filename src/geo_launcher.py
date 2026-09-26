"""Автозапуск MCP-сервера геолокации (geo_server.py) в отдельном окне.

Тонкая настройка базового :class:`HttpMCPServerLauncher`
(см. src/reminder_launcher.py) под geo-сервер: скрипт в корне проекта,
порт из настроек (GEO_SERVER_PORT, по умолчанию 8002 — порт 8001 занят
news-сервером), сервер живёт 24/7 — повторный запуск приложения его не
дублирует, закрытие приложения не останавливает.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import PROJECT_ROOT
from .reminder_launcher import DEFAULT_HOST, HttpMCPServerLauncher

DEFAULT_PORT = 8002


class GeoServerLauncher(HttpMCPServerLauncher):
    """Лаунчер geo_server.py (город пользователя по IP, ip-api.com)."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        server_path: Optional[Path] = None,
    ) -> None:
        """Настраивает лаунчер geo-сервера.

        Args:
            host: адрес сервера.
            port: порт сервера.
            server_path: путь к geo_server.py; None — корень проекта.
        """
        super().__init__(
            server_path=server_path or (PROJECT_ROOT / "geo_server.py"),
            host=host,
            port=port,
            label="Geo-сервер",
        )
