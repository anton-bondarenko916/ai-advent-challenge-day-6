"""Автозапуск MCP-сервера CRM (crm_server.py) в отдельном окне.

Тонкая настройка базового :class:`HttpMCPServerLauncher`
(см. src/reminder_launcher.py) под CRM-сервер: скрипт в корне проекта,
порт из настроек (CRM_SERVER_PORT, по умолчанию 8004 — 8000–8003 заняты
reminder/news/geo/weather), сервер живёт 24/7 — повторный запуск
приложения его не дублирует, закрытие приложения не останавливает.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import PROJECT_ROOT
from .reminder_launcher import DEFAULT_HOST, HttpMCPServerLauncher

DEFAULT_PORT = 8004


class CrmServerLauncher(HttpMCPServerLauncher):
    """Лаунчер crm_server.py (пользователи и тикеты для /support)."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        server_path: Optional[Path] = None,
    ) -> None:
        """Настраивает лаунчер CRM-сервера.

        Args:
            host: адрес сервера.
            port: порт сервера.
            server_path: путь к crm_server.py; None — корень проекта.
        """
        super().__init__(
            server_path=server_path or (PROJECT_ROOT / "crm_server.py"),
            host=host,
            port=port,
            label="CRM-сервер",
        )
