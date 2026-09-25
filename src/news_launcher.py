"""Автозапуск MCP-сервера новостей (news_server.py) в отдельном окне.

Тонкая настройка базового :class:`HttpMCPServerLauncher`
(см. src/reminder_launcher.py) под news-сервер: скрипт в корне проекта,
порт из настроек (NEWS_SERVER_PORT, по умолчанию 8001), порт 24/7 —
повторный запуск приложения сервер не дублирует, закрытие приложения
его не останавливает.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import PROJECT_ROOT
from .reminder_launcher import DEFAULT_HOST, HttpMCPServerLauncher

DEFAULT_PORT = 8001


class NewsServerLauncher(HttpMCPServerLauncher):
    """Лаунчер news_server.py (новости, сводка, сохранение в output/)."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        server_path: Optional[Path] = None,
    ) -> None:
        """Настраивает лаунчер новостей.

        Args:
            host: адрес сервера.
            port: порт сервера.
            server_path: путь к news_server.py; None — корень проекта.
        """
        super().__init__(
            server_path=server_path or (PROJECT_ROOT / "news_server.py"),
            host=host,
            port=port,
            label="News-сервер",
        )
