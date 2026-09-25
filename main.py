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
from pathlib import Path
from typing import Any, Dict, List

from src import utils
from src.agent import DeepSeekAgent
from src.config import (
    PROJECT_ROOT,
    get_invariant_settings,
    get_memory_settings,
    get_news_settings,
    get_prices,
    get_reminder_settings,
    get_strategy_settings,
    get_task_settings,
    load_config,
    validate_api_key,
)
from src.reminder_client import ReminderMCPClient
from src.storage import ContextStore
from src.profile_repository import ProfileRepository
from src.invariant_repository import InvariantRepository
from src.invariant_validator import InvariantValidator
from src.task_repository import TaskRepository
from src.tokens import create_counter
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


def create_agent(config: Dict[str, Any]) -> DeepSeekAgent:
    """Собирает агента со всеми зависимостями из конфигурации.

    Вынесено из :func:`main`, чтобы стартовую конфигурацию можно было
    проверить тестами без запуска GUI (именно здесь ломались запуски при
    изменении настроек).

    Args:
        config: словарь конфигурации (результат load_config).

    Returns:
        Готовый DeepSeekAgent (хранилище, счётчик токенов, цены, стратегия
        управления контекстом — из конфигурации).

    Raises:
        AgentConfigError: если параметры агента некорректны.
    """
    # Стратегия управления контекстом: sliding_window | sticky_facts | branching
    strategy_settings = get_strategy_settings(config)

    # Хранилище контекста (SQLite). В mock-режиме агент сам его игнорирует.
    storage = None
    if config.get("STORAGE_ENABLED", True):
        raw_path = str(config.get("STORAGE_PATH", "data/chat_history.db"))
        db_path = Path(raw_path)
        if not db_path.is_absolute():
            db_path = PROJECT_ROOT / db_path
        storage = ContextStore(
            db_path=db_path,
            enabled=True,
            default_strategy=str(strategy_settings["branch_strategy"]),
        )

    # Персонализация: репозиторий профилей (SQLite, тот же файл БД, что и
    # у хранилища контекста). Миграция выполняется в конструкторе: создаёт
    # таблицы profiles/profile_agent_settings, колонку messages.profile_id
    # и профиль Default с настройками из .env (лог — migration.log).
    profile_repository = None
    if storage is not None:
        profile_repository = ProfileRepository(
            db_path=storage.db_path,
            config=config,
        )

    # Инварианты: жёсткие правила «что можно делать» (SQLite, тот же файл
    # БД). Миграция в конструкторе создаёт таблицы, предустановленные
    # категории и 1–2 инварианта-примера (лог — migration.log). Валидатор
    # пост-проверки настраивается из .env (ENABLE_POST_CHECK и лимиты).
    invariant_repository = None
    if storage is not None:
        invariant_repository = InvariantRepository(db_path=storage.db_path)
    invariant_validator = InvariantValidator(
        **get_invariant_settings(config)
    )

    # Подсистема задач (FSM): отдельная БД SQLite, состояние задачи
    # переживает перезапуск приложения. В mock-режиме задачи тоже работают
    # (LLM-этапы имитируются) — это удобно для тестов и демонстрации.
    task_repository = None
    if config.get("TASKS_ENABLED", True):
        raw_task_path = str(config.get("TASKS_PATH", "data/tasks.db"))
        task_db_path = Path(raw_task_path)
        if not task_db_path.is_absolute():
            task_db_path = PROJECT_ROOT / task_db_path
        task_repository = TaskRepository(db_path=task_db_path)

    # Счётчик токенов и цены: tiktoken при наличии, иначе встроенная эвристика
    token_counter = create_counter(
        model=config["DEFAULT_MODEL"],
        enabled=bool(config.get("TOKENIZER_ENABLED", True)),
        encoding_name=str(config.get("TOKENIZER_ENCODING", "cl100k_base")),
    )
    prices = get_prices(config)
    # Модель памяти (short_term | working | long_term): настройки из конфигурации
    memory_settings = get_memory_settings(config)

    # MCP-сервер напоминаний: HTTP-клиент для tool calling (tools → LLM,
    # tool_calls → MCP-сервер). Сам сервер запускается лаунчером при старте
    # UI (ChatApplication); недоступный сервер НЕ ломает диалог — запрос
    # уходит без tools (см. DeepSeekAgent._get_tools_for_llm).
    reminder_client = None
    reminder_settings = get_reminder_settings(config)
    if reminder_settings["enabled"]:
        reminder_client = ReminderMCPClient(server_url=reminder_settings["url"])

    # MCP-сервер новостей: search_news (RSS) → summarize_text (DeepSeek
    # внутри сервера) → save_to_file (output/). Цепочку строит LLM в
    # agentic loop агента; недоступный сервер тоже не ломает диалог.
    news_client = None
    news_settings = get_news_settings(config)
    if news_settings["enabled"]:
        news_client = ReminderMCPClient(server_url=news_settings["url"])
    max_tool_rounds = reminder_settings["max_tool_rounds"]

    return DeepSeekAgent(
        api_key=config["DEEPSEEK_API_KEY"],
        model=config["DEFAULT_MODEL"],
        system_prompt=config["DEFAULT_SYSTEM_PROMPT"],
        temperature=config["DEFAULT_TEMPERATURE"],
        max_tokens=config["DEFAULT_MAX_TOKENS"],
        mock_mode=config["MOCK_MODE"],
        max_history_messages=config["MAX_HISTORY_MESSAGES"],
        storage=storage,
        token_counter=token_counter,
        prices=prices,
        # В приложении используется переключатель стратегий; суммаризация
        # (summary) не входит в него и оставлена для обратной совместимости.
        strategy=strategy_settings["name"],
        strategy_settings=strategy_settings,
        memory_settings=memory_settings,
        # Персонализация: активный профиль подставляется в каждый запрос,
        # а его настройки перекрывают DEFAULT_* из .env
        profile_repository=profile_repository,
        # Инварианты: блок в system-промпте + пост-проверка ответов
        # (репозиторий создаётся лениво из хранилища, если не передан)
        invariant_repository=invariant_repository,
        invariant_validator=invariant_validator,
        # Подсистема задач (FSM): план → шаги → проверка, пауза/восстановление
        task_repository=task_repository,
        # Лимит откатов валидации для FSM задач (TASKS_MAX_RETRIES)
        task_max_retries=get_task_settings(config)["max_retries"],
        # Tool calling напоминаний: tools reminder-сервера передаются в LLM,
        # tool_calls выполняются через HTTP MCP-клиент
        reminder_client=reminder_client,
        # Tool calling новостей: search_news → summarize_text → save_to_file
        # (цепочку строит LLM в agentic loop)
        news_client=news_client,
        max_tool_rounds=max_tool_rounds,
    )


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
