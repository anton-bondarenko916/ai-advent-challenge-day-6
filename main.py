"""Точка входа в приложение DeepSeek Chat.

Загружает конфигурацию из .env, создаёт агента DeepSeekAgent
с параметрами из конфигурации, запускает GUI (ChatApplication).

Примеры запуска:

    python main.py                 # обычный режим (нужен DEEPSEEK_API_KEY)
    python main.py --mock          # mock-режим без обращения к API
"""

from __future__ import annotations

import sys
import traceback
from typing import List

from src import utils
from src.agent_factory import create_agent  # noqa: F401 — реэкспорт для тестов
from src.config import load_config, validate_api_key
from src.utils import AgentConfigError, get_logger

logger = get_logger("deepseek-chat")


def parse_args(argv: List[str]) -> bool:
    """Разбирает аргументы командной строки.

    Args:
        argv: список аргументов (без имени программы).

    Returns:
        True, если запрошен mock-режим (флаг --mock).
    """
    return "--mock" in argv


def main(argv: List[str]) -> int:
    """Запускает приложение.

    Args:
        argv: аргументы командной строки (без имени программы).

    Returns:
        Код возврата: 0 — успех, 1 — ошибка инициализации.
    """
    mock_requested = parse_args(argv)

    # 1. Конфигурация
    try:
        config = load_config()
    except Exception as exc:
        print(f"[ОШИБКА] Не удалось загрузить конфигурацию: {exc}", file=sys.stderr)
        return 1

    # 2. Проверка API-ключа (пропускается в mock-режиме)
    if mock_requested:
        config["MOCK_MODE"] = True
    try:
        validate_api_key(config)
    except ValueError as exc:
        print(f"[ОШИБКА] {exc}", file=sys.stderr)
        print("Подсказка: скопируйте .env.example в .env и укажите ключ, "
              "либо запустите с флагом --mock.", file=sys.stderr)
        return 1

    # 3. Создание агента (хранилище, счётчик токенов, цены, сжатие контекста)
    try:
        agent = create_agent(config)
    except AgentConfigError as exc:
        print(f"[ОШИБКА] Некорректная конфигурация агента: {exc}", file=sys.stderr)
        return 1

    # 4. GUI
    try:
        # Импортируем tkinter-модуль здесь, чтобы CLI/тесты могли
        # пользоваться агентом без установленного графического стека.
        import tkinter as tk
        from src.ui import ChatApplication
    except ImportError as exc:
        print(f"[ОШИБКА] Tkinter недоступен: {exc}", file=sys.stderr)
        print("Установите python3-tk (Linux) или используйте стандартный "
              "сборщик Python (Windows/macOS).", file=sys.stderr)
        return 1

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"[ОШИБКА] Не удалось создать графическое окно: {exc}", file=sys.stderr)
        print("Убедитесь, что доступна графическая среда (дисплей).", file=sys.stderr)
        return 1

    try:
        app = ChatApplication(root=root, agent=agent, config=config)
        # Напоминания: автозапуск MCP-сервера (не повторяется, если сервер
        # уже работает — он живёт 24/7) и слушатель SSE-уведомлений
        app.launch_reminder_server()
        app.start_reminder_listener()
        # Новости: автозапуск news-сервера (search_news → summarize_text →
        # save_to_file); цепочку вызовов строит LLM в agentic loop
        app.launch_news_server()
        # Гео и погода: автозапуск серверов цепочки
        # get_user_city (geo, порт 8002) → get_weather (weather, порт 8003)
        app.launch_geo_server()
        app.launch_weather_server()
        app.root.mainloop()
        return 0
    except Exception:
        logger.error("Необработанная ошибка приложения:\n%s",
                     traceback.format_exc())
        print("[ОШИБКА] См. лог выше.", file=sys.stderr)
        return 1
    finally:
        # Финальный flush контекста при любом исходе завершения GUI
        agent.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
