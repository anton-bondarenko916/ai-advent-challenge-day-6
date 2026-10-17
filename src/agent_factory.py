"""Фабрика DeepSeekAgent: сборка агента со всеми зависимостями из конфигурации.

Функция :func:`create_agent` переехала сюда из ``main.py`` (миграция UI на
Flet): и Tkinter-приложение, и новое Flet-приложение (``app/main.py``)
собирают один и тот же агент — профили, память, FSM задач, инварианты,
MCP-клиенты напоминаний/новостей/гео/погоды/файловой системы.

``main.create_agent`` сохранён как реэкспорт для обратной совместимости
с тестами (tests/test_startup.py, tests/test_geo_weather.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.agent import DeepSeekAgent
from src.config import (
    PROJECT_ROOT,
    get_fs_settings,
    get_geo_settings,
    get_invariant_settings,
    get_memory_settings,
    get_news_settings,
    get_prices,
    get_reminder_settings,
    get_strategy_settings,
    get_task_settings,
    get_weather_settings,
)
from src.fs_launcher import FsServerLauncher
from src.geo_launcher import GeoServerLauncher
from src.news_launcher import NewsServerLauncher
from src.reminder_client import ReminderMCPClient
from src.reminder_launcher import ReminderServerLauncher
from src.storage import ContextStore
from src.profile_repository import ProfileRepository
from src.invariant_repository import InvariantRepository
from src.invariant_validator import InvariantValidator
from src.task_repository import TaskRepository
from src.tokens import create_counter
from src.utils import AgentConfigError
from src.weather_launcher import WeatherServerLauncher

__all__ = ["create_agent"]


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
    # UI; недоступный сервер НЕ ломает диалог — запрос уходит без tools
    # (см. DeepSeekAgent._get_tools_for_llm). Подпись label попадает в
    # тексты ошибок — LLM сообщает верный сервер.
    reminder_client = None
    reminder_settings = get_reminder_settings(config)
    if reminder_settings["enabled"]:
        reminder_client = ReminderMCPClient(
            server_url=reminder_settings["url"],
            label="Сервер напоминаний",
        )
        # Самовосстановление: если инструмент упёрся в мёртвый сервер —
        # поднять его и повторить вызов в рамках того же хода
        reminder_client.restart_hook = ReminderServerLauncher(
            host=reminder_settings["host"],
            port=reminder_settings["port"],
        ).ensure_running

    # MCP-сервер новостей: search_news (RSS) → summarize_text (DeepSeek
    # внутри сервера) → save_to_file (output/). Цепочку строит LLM в
    # agentic loop агента; недоступный сервер тоже не ломает диалог.
    news_client = None
    news_settings = get_news_settings(config)
    if news_settings["enabled"]:
        news_client = ReminderMCPClient(
            server_url=news_settings["url"],
            label="News-сервер",
        )
        news_client.restart_hook = NewsServerLauncher(
            host=news_settings["host"],
            port=news_settings["port"],
        ).ensure_running

    # MCP-серверы цепочки «гео → погода»: LLM в agentic loop сама строит
    # цепочку get_user_city (geo, ip-api.com) → get_weather(city)
    # (weather, Open-Meteo). Недоступный сервер не ломает диалог.
    geo_client = None
    geo_settings = get_geo_settings(config)
    if geo_settings["enabled"]:
        geo_client = ReminderMCPClient(
            server_url=geo_settings["url"],
            label="Geo-сервер",
        )
        geo_client.restart_hook = GeoServerLauncher(
            host=geo_settings["host"],
            port=geo_settings["port"],
        ).ensure_running

    weather_client = None
    weather_settings = get_weather_settings(config)
    if weather_settings["enabled"]:
        weather_client = ReminderMCPClient(
            server_url=weather_settings["url"],
            label="Weather-сервер",
        )
        weather_client.restart_hook = WeatherServerLauncher(
            host=weather_settings["host"],
            port=weather_settings["port"],
        ).ensure_running

    # MCP-сервер файловой системы (ТЗ ai_1_35): read/write/search/tree —
    # LLM в agentic loop сама решает, какие файловые операции выполнить
    # (сценарии «найти использования компонента» и «сгенерировать файл»).
    # Доступ ограничен корнем проекта на стороне fs_server.py.
    fs_client = None
    fs_settings = get_fs_settings(config)
    if fs_settings["enabled"]:
        fs_client = ReminderMCPClient(
            server_url=fs_settings["url"],
            label="FS-сервер",
        )
        fs_client.restart_hook = FsServerLauncher(
            host=fs_settings["host"],
            port=fs_settings["port"],
        ).ensure_running
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
        # Tool calling гео/погоды: get_user_city → get_weather (цепочку
        # строит LLM в agentic loop, маршрутизация — по имени инструмента)
        geo_client=geo_client,
        weather_client=weather_client,
        # Tool calling файловой системы (ТЗ ai_1_35): read/write/search
        fs_client=fs_client,
        max_tool_rounds=max_tool_rounds,
    )
