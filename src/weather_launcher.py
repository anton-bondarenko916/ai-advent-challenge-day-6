"""Автозапуск MCP-сервера погоды (weather_server.py) в отдельном окне.

Тонкая настройка базового :class:`HttpMCPServerLauncher`
(см. src/reminder_launcher.py) под weather-сервер: скрипт в корне
проекта, порт из настроек (WEATHER_SERVER_PORT, по умолчанию 8003),
сервер живёт 24/7 — повторный запуск приложения его не дублирует,
закрытие приложения не останавливает.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import PROJECT_ROOT
from .reminder_launcher import DEFAULT_HOST, HttpMCPServerLauncher

DEFAULT_PORT = 8003


class WeatherServerLauncher(HttpMCPServerLauncher):
    """Лаунчер weather_server.py (погода по городу, Open-Meteo)."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        server_path: Optional[Path] = None,
    ) -> None:
        """Настраивает лаунчер weather-сервера.

        Args:
            host: адрес сервера.
            port: порт сервера.
            server_path: путь к weather_server.py; None — корень проекта.
        """
        super().__init__(
            server_path=server_path or (PROJECT_ROOT / "weather_server.py"),
            host=host,
            port=port,
            label="Weather-сервер",
        )
