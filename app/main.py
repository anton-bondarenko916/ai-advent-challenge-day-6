"""Точка входа Flet-приложения DeepSeek Chat (миграция с Tkinter).

Экраны (вкладки): 💬 Чат (RAG-режим, источники), 📚 База знаний
(индексация MD), ⚖️ Сравнение (10 вопросов × 2 режима), 🔀 Сравнение
RAG-режимов (5 режимов + автоматические метрики), ⚙️ Настройки
(профиль, температура, max_tokens, модель).

Ядро агента — то же, что в Tkinter-версии (src.agent.DeepSeekAgent:
профили, память, FSM задач, инварианты, MCP-инструменты).

Запуск из корня проекта:

    python app/main.py            # нативное окно (нужен DEEPSEEK_API_KEY)
    python app/main.py --mock     # mock-режим без обращения к API
    flet run app/main.py          # через flet CLI

Асинхронность: фоновые задачи (LLM, Ollama, индексация) запускаются
через page.run_task, блокирующие вызовы — asyncio.to_thread внутри
RAGChatAgent; UI обновляется только из event loop Flet.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Запуск "python app/main.py": корень проекта — в sys.path для
# абсолютных импортов app.* и src.*
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import flet as ft

from app.agent import RAGChatAgent
from app.ui.chat_view import ChatView
from app.ui.comparison_view import ComparisonView
from app.ui.knowledge_view import KnowledgeView
from app.ui.rag_comparison_view import RAGModesView
from app.ui.settings_view import SettingsView
from src.config import (
    get_geo_settings,
    get_news_settings,
    get_reminder_settings,
    get_weather_settings,
    load_config,
    validate_api_key,
)
from src.utils import get_logger

logger = get_logger("deepseek-chat.flet")

TAB_TITLES = ("💬 Чат", "📚 База знаний", "⚖️ Сравнение",
              "🔀 RAG-режимы", "⚙️ Настройки")

DEFAULT_DOCS_DIR = PROJECT_ROOT / "app" / "data" / "docs"


class ChatApp:
    """Flet-приложение: вкладки + общий RAGChatAgent.

    Attributes:
        page: страница Flet.
        agent: RAGChatAgent (ядро + RAG-пайплайн).
        file_picker: сервис выбора папки (страница «База знаний»).
    """

    def __init__(
        self,
        page: ft.Page,
        agent: RAGChatAgent,
        config: Dict[str, Any],
    ) -> None:
        self.page = page
        self.agent = agent
        self.config = config

        page.title = str(config.get("UI_TITLE", "DeepSeek Chat")) + " (Flet)"
        page.padding = 16
        page.window.width = int(config.get("UI_WIDTH", 1000) or 1000)
        page.window.height = int(config.get("UI_HEIGHT", 760) or 760)

        # FilePicker — сервис страницы (диалог выбора папки с документами)
        self.file_picker = ft.FilePicker()
        page.services.append(self.file_picker)

        self.chat_view = ChatView(self)
        self.knowledge_view = KnowledgeView(
            self, default_folder=str(DEFAULT_DOCS_DIR))
        self.comparison_view = ComparisonView(self)
        self.rag_modes_view = RAGModesView(self)
        self.settings_view = SettingsView(self)
        self._views = (
            self.chat_view, self.knowledge_view,
            self.comparison_view, self.rag_modes_view,
            self.settings_view,
        )

        self.tabs = ft.Tabs(
            content=ft.TabBar(tabs=[ft.Tab(label=title) for title in TAB_TITLES]),
            length=len(TAB_TITLES),
            selected_index=0,
            on_change=self._on_tab_change,
        )
        self.view_containers = [
            ft.Container(content=view.build(), expand=True, visible=(i == 0))
            for i, view in enumerate(self._views)
        ]
        body = ft.Column(self.view_containers, expand=True, spacing=8)

        # Восстановление истории диалога из хранилища ядра
        self.chat_view.restore_history(self._history_messages())
        self.chat_view.refresh_rag_status()

        page.add(ft.Column([self.tabs, body], expand=True, spacing=8))
        page.on_disconnect = self._on_disconnect

    # ------------------------------------------------------------------
    # Сервисные методы для вкладок
    # ------------------------------------------------------------------
    def notify(self, message: str) -> None:
        """SnackBar-уведомление (только из event loop)."""
        self.page.show_dialog(ft.SnackBar(content=ft.Text(message)))

    def _history_messages(self) -> list:
        """История ядра без системного сообщения (для восстановления чата)."""
        try:
            history = self.agent.core._message_history  # noqa: SLF001
        except AttributeError:
            return []
        return [
            message for message in history
            if message.get("role") in ("user", "assistant")
        ]

    def _on_tab_change(self, event=None) -> None:
        """Переключение вкладок: видимость + refresh активной.

        Ошибка refresh не блокирует интерфейс: traceback — в лог,
        пользователю — SnackBar (без полноэкранного error-overlay Flet).
        """
        try:
            index = int(self.tabs.selected_index or 0)
            for i, container in enumerate(self.view_containers):
                container.visible = i == index
            view = self._views[index]
            on_show = getattr(view, "on_show", None)
            if on_show is not None:
                on_show()
        except Exception:
            logger.error("Ошибка переключения вкладки:\n%s",
                         traceback.format_exc())
            try:
                self.notify("⚠️ Ошибка вкладки (подробности — в логе)")
            except Exception:
                logger.error("Не удалось показать SnackBar об ошибке")
        try:
            self.page.update()
        except Exception:
            logger.error("Ошибка перерисовки после смены вкладки:\n%s",
                         traceback.format_exc())

    def _on_disconnect(self, event=None) -> None:
        """Финальный flush контекста при закрытии окна."""
        try:
            self.agent.close()
        except Exception:
            logger.error("Ошибка при закрытии агента:\n%s",
                         traceback.format_exc())


def launch_mcp_servers(config: Dict[str, Any]) -> List[str]:
    """Автозапуск локальных MCP-серверов (как в Tkinter-версии).

    Без этого серверы стартуют только при первом ходе LLM через
    self-healing хук (restart_hook → ensure_running) — инструменты
    недоступны до первого запроса. Каждый launch(): порт уже отвечает →
    сервер не трогаем (живёт 24/7); иначе процесс в отдельном окне, не
    дожидаясь бинда. Ошибки запуска приложение не ломают.

    Args:
        config: конфигурация load_config.

    Returns:
        Список сообщений для лога (метка сервера: результат).
    """
    from src.geo_launcher import GeoServerLauncher
    from src.news_launcher import NewsServerLauncher
    from src.reminder_launcher import ReminderServerLauncher
    from src.weather_launcher import WeatherServerLauncher

    specs = []
    reminder = get_reminder_settings(config)
    if reminder["enabled"]:
        specs.append(ReminderServerLauncher(host=reminder["host"],
                                            port=reminder["port"]))
    news = get_news_settings(config)
    if news["enabled"]:
        specs.append(NewsServerLauncher(host=news["host"], port=news["port"]))
    geo = get_geo_settings(config)
    if geo["enabled"]:
        specs.append(GeoServerLauncher(host=geo["host"], port=geo["port"]))
    weather = get_weather_settings(config)
    if weather["enabled"]:
        specs.append(WeatherServerLauncher(host=weather["host"],
                                           port=weather["port"]))

    messages: List[str] = []
    for launcher in specs:
        try:
            _started, message = launcher.launch()
        except Exception as exc:  # noqa: BLE001 — старт не должен падать
            message = f"Ошибка автозапуска: {exc}"
        messages.append(f"{launcher.label}: {message}")
        logger.info(message)
    return messages


def run_app(argv: Optional[list] = None) -> None:
    """Загружает конфиг, собирает агента и запускает Flet-приложение.

    Args:
        argv: аргументы командной строки (поддерживается --mock).

    Raises:
        SystemExit: при ошибке конфигурации (код возврата 1).
    """
    argv = list(argv or [])
    try:
        config = load_config()
    except Exception as exc:
        print(f"[ОШИБКА] Не удалось загрузить конфигурацию: {exc}",
              file=sys.stderr)
        raise SystemExit(1)
    if "--mock" in argv:
        config["MOCK_MODE"] = True
    try:
        validate_api_key(config)
    except ValueError as exc:
        print(f"[ОШИБКА] {exc}", file=sys.stderr)
        print("Подсказка: скопируйте .env.example в .env и укажите ключ, "
              "либо запустите с флагом --mock.", file=sys.stderr)
        raise SystemExit(1)
    try:
        agent = RAGChatAgent.with_config(config).bind_config(config)
    except Exception as exc:
        print(f"[ОШИБКА] Не удалось создать агента: {exc}", file=sys.stderr)
        raise SystemExit(1)

    # Автозапуск MCP-серверов (напоминания/новости/гео/погода): порты
    # уже отвечают → мгновенный пропуск; иначе процессы в новых окнах.
    # Не блокирует открытие окна: launch() только спавнит процесс.
    launch_mcp_servers(config)

    def start(page: ft.Page) -> None:
        ChatApp(page, agent, config)

    ft.run(start)


if __name__ == "__main__":
    run_app(sys.argv[1:])
