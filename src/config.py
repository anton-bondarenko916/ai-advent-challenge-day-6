"""Конфигурация приложения и загрузка переменных окружения.

Модуль отвечает за:
- поиск и загрузку .env файла (через python-dotenv);
- чтение настроек с безопасными значениями по умолчанию;
- проверку наличия API-ключа;
- предоставление единой точки доступа к настройкам (get_config).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import AVAILABLE_MODELS, get_logger

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover — запасной вариант без python-dotenv
    def load_dotenv(
        dotenv_path: str | os.PathLike | None = None,
        override: bool = False,
    ) -> bool:
        """Минимальная замена python-dotenv: читает .env вручную.

        Используется только если пакет python-dotenv не установлен.
        """
        path = Path(dotenv_path) if dotenv_path else None
        if path is None or not path.exists():
            return False
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if override or not os.getenv(key):
                os.environ[key] = value
        return True

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------

# Корень проекта: каталог, где лежит main.py (на уровень выше src/)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
ENV_PATH: Path = PROJECT_ROOT / ".env"

# ---------------------------------------------------------------------------
# Значения по умолчанию
# ---------------------------------------------------------------------------

DEFAULTS: Dict[str, Any] = {
    "DEEPSEEK_API_KEY": "",
    "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
    "DEFAULT_MODEL": "deepseek-v4-flash",
    "DEFAULT_TEMPERATURE": 0.7,
    "DEFAULT_MAX_TOKENS": 1000,
    "DEFAULT_SYSTEM_PROMPT": (
        "Ты полезный AI-ассистент. Отвечай четко и по делу."
    ),
    "UI_TITLE": "DeepSeek Chat",
    "UI_WIDTH": 900,
    "UI_HEIGHT": 700,
    "MOCK_MODE": False,  # True — имитация ответов без обращения к API
    "MAX_HISTORY_MESSAGES": 50,  # ограничение истории диалога
    # Персистентность контекста (SQLite)
    "STORAGE_ENABLED": True,  # False — отключить сохранение контекста
    "STORAGE_PATH": "data/chat_history.db",  # путь к БД (относительно корня)
    # Подсчёт токенов
    "TOKENIZER_ENABLED": True,  # False — всегда использовать эвристику
    "TOKENIZER_ENCODING": "cl100k_base",  # резервная кодировка tiktoken
    # Цены за 1M токенов в USD (0.0 — цена не задана, стоимость не считается)
    "PRICE_INPUT_PER_1M": 0.0,
    "PRICE_OUTPUT_PER_1M": 0.0,
    # Управление контекстом: сжатие старой части диалога в конспект (summary).
    # Оставлено для обратной совместимости: в приложении не используется,
    # в переключателе стратегий суммаризации нет.
    "SUMMARY_ENABLED": True,
    "SUMMARY_KEEP_LAST_N": 10,   # сколько последних сообщений идут «как есть»
    "SUMMARY_EVERY_N": 10,       # размер блока сжатия (K сообщений)
    "SUMMARY_MAX_TOKENS": 400,   # лимит длины конспекта
    "SUMMARY_TEMPERATURE": 0.2,  # температура служебного запроса
    "SUMMARY_MODEL": "",         # "" — использовать модель диалога
    "SUMMARY_PROMPT": "",        # "" — встроенная инструкция сжатия
    # Управление контекстом: стратегии (sliding_window | sticky_facts | branching)
    "STRATEGY": "sliding_window",   # стратегия по умолчанию
    "WINDOW_KEEP_LAST_N": 20,       # размер окна последних сообщений
    # Sticky Facts: блок «ключ → значение», обновляемый после каждого хода
    "FACTS_ENABLED": True,          # обновлять факты LLM-запросом
    "FACTS_MODEL": "",              # "" — использовать модель диалога
    "FACTS_TEMPERATURE": 0.0,       # температура служебного запроса фактов
    "FACTS_MAX_TOKENS": 300,        # лимит блока фактов в токенах
    "FACTS_MAX_ITEMS": 50,          # максимум пар «ключ → значение»
    "FACTS_PROMPT": "",             # "" — встроенная инструкция извлечения
    # Ветки диалога (режим branching)
    "BRANCH_MAX": 20,               # максимум веток в одном хранилище
    # ------------------------------------------------------------------
    # Явная модель памяти агента (short_term | working | long_term)
    # ------------------------------------------------------------------
    "MEMORY_ENABLED": True,          # общий выключатель модели памяти
    "MEMORY_WORKING_ENABLED": True,  # рабочая память (задача, привязана к ветке)
    "MEMORY_LONG_ENABLED": True,     # долговременная память (профиль/решения/знания)
    "MEMORY_LONG_PATH": "data/long_memory.db",  # отдельный файл БД (от корня)
    "MEMORY_LONG_MAX_ITEMS": 200,    # максимум записей долговременной памяти
    "MEMORY_MAX_TOKENS": 600,        # общий бюджет блоков памяти в запросе
    "MEMORY_LONG_MAX_TOKENS": 300,   # бюджет долговременной памяти
    "MEMORY_WORKING_MAX_TOKENS": 250,  # бюджет рабочей памяти
    "MEMORY_PIN_MAX": 5,             # максимум закреплённых сообщений
    "MEMORY_PROPOSE_LLM": True,      # предлагать классификацию после хода
    "MEMORY_MODEL": "",              # "" — модель диалога
    "MEMORY_TEMPERATURE": 0.0,       # температура служебного запроса
    "MEMORY_PROMPT": "",             # "" — встроенная инструкция
    "MEMORY_MIN_CONFIDENCE": 0.6,    # порог уверенности кандидатов
    # ВНИМАНИЕ: true нарушает принцип явного выбора слоя (не рекомендуется).
    "MEMORY_AUTO_APPLY": False,
    "MEMORY_EXPORT_PATH": "data/memory_export.md",  # путь экспорта по умолчанию
    # ------------------------------------------------------------------
    # Подсистема задач (FSM): состояние задачи в отдельной БД SQLite
    # ------------------------------------------------------------------
    "TASKS_ENABLED": True,           # False — задачи недоступны в UI
    "TASKS_PATH": "data/tasks.db",   # путь к БД задач (относительно корня)
    "TASKS_MAX_RETRIES": 3,          # лимит откатов validation → execution
    # ------------------------------------------------------------------
    # Инварианты: пост-проверка ответов агента (второй LLM-вызов)
    # ------------------------------------------------------------------
    "ENABLE_POST_CHECK": True,       # False — без пост-проверки ответов
    "MAX_REGENERATIONS": 2,          # лимит перегенераций при нарушении
    "MIN_RESPONSE_LENGTH": 100,      # ответы короче не проверяются
    # ------------------------------------------------------------------
    # MCP-клиент: список инструментов сервера (stdio-транспорт)
    # ------------------------------------------------------------------
    "MCP_ENABLED": True,             # False — панель MCP не создаётся
    "MCP_SERVER_COMMAND": "npx -y open-internet-mcp",  # команда запуска
    "MCP_TIMEOUT": 30,               # таймаут handshake/tools/list, сек
    "MCP_LOG_PATH": "mcp.log",       # журнал клиента и stderr сервера
    "MCP_GIT_REPO_PATH": "",         # репозиторий по умолчанию для вызова
                                     # get_git_* ("" — корень проекта)
    # ------------------------------------------------------------------
    # MCP-сервер напоминаний (reminder_server.py, HTTP-транспорт 24/7)
    # ------------------------------------------------------------------
    "REMINDER_ENABLED": True,        # False — без напоминаний и автозапуска
    "REMINDER_SERVER_HOST": "127.0.0.1",  # только локально, не хостить!
    "REMINDER_SERVER_PORT": 8000,    # порт http://127.0.0.1:<порт>/mcp
    "REMINDER_TOOLCALL_MAX_ROUNDS": 5,  # лимит циклов tool calling за ход
    # ------------------------------------------------------------------
    # Состояние UI (сворачивание панели MCP и т.п.): SQLite key-value
    # ------------------------------------------------------------------
    "APP_STATE_PATH": "data/app_state.db",  # путь к БД (относительно корня)
}

# Переменные, значения которых нужно приводить к логическому типу
_BOOL_KEYS = (
    "MOCK_MODE", "STORAGE_ENABLED", "TOKENIZER_ENABLED", "SUMMARY_ENABLED",
    "FACTS_ENABLED", "TASKS_ENABLED",
    "MEMORY_ENABLED", "MEMORY_WORKING_ENABLED", "MEMORY_LONG_ENABLED",
    "MEMORY_PROPOSE_LLM", "MEMORY_AUTO_APPLY",
    "ENABLE_POST_CHECK",
    "MCP_ENABLED",
    "REMINDER_ENABLED",
)

# Переменные, значения которых нужно приводить к числу
_INT_KEYS = (
    "UI_WIDTH", "UI_HEIGHT", "DEFAULT_MAX_TOKENS", "MAX_HISTORY_MESSAGES",
    "SUMMARY_KEEP_LAST_N", "SUMMARY_EVERY_N", "SUMMARY_MAX_TOKENS",
    "WINDOW_KEEP_LAST_N", "FACTS_MAX_TOKENS", "FACTS_MAX_ITEMS", "BRANCH_MAX",
    "MEMORY_LONG_MAX_ITEMS", "MEMORY_MAX_TOKENS", "MEMORY_LONG_MAX_TOKENS",
    "MEMORY_WORKING_MAX_TOKENS", "MEMORY_PIN_MAX",
    "MAX_REGENERATIONS", "MIN_RESPONSE_LENGTH",
    "TASKS_MAX_RETRIES",
    "REMINDER_SERVER_PORT", "REMINDER_TOOLCALL_MAX_ROUNDS",
)
_FLOAT_KEYS = (
    "DEFAULT_TEMPERATURE", "PRICE_INPUT_PER_1M", "PRICE_OUTPUT_PER_1M",
    "SUMMARY_TEMPERATURE", "FACTS_TEMPERATURE", "MEMORY_TEMPERATURE",
    "MEMORY_MIN_CONFIDENCE", "MCP_TIMEOUT",
)

# Суффиксы переменных окружения с ценами по конкретной модели:
# PRICE_<MODEL>_INPUT_PER_1M / PRICE_<MODEL>_OUTPUT_PER_1M
_PRICE_INPUT_SUFFIX = "INPUT_PER_1M"
_PRICE_OUTPUT_SUFFIX = "OUTPUT_PER_1M"


def load_config(env_file: str | os.PathLike | None = None) -> Dict[str, Any]:
    """Загружает переменные окружения из .env в окружение процесса.

    Повторные вызовы безопасны: python-dotenv не перезаписывает уже
    установленные переменные (override=False), поэтому реальные переменные
    окружения имеют приоритет над значениями из .env.

    Args:
        env_file: путь к файлу .env. По умолчанию PROJECT_ROOT / ".env".

    Returns:
        Словарь с полной конфигурацией (см. get_config).
    """
    path: Path = Path(env_file) if env_file else ENV_PATH
    if path.exists():
        load_dotenv(dotenv_path=path, override=False)
    return get_config()


def get_config() -> Dict[str, Any]:
    """Собирает конфигурацию из переменных окружения и значений по умолчанию.

    Возвращает словарь, где каждый ключ из DEFAULTS присутствует
    и приведён к корректному типу.

    Returns:
        Dict[str, Any]: итоговая конфигурация приложения.
    """
    config: Dict[str, Any] = {}

    for key, default in DEFAULTS.items():
        raw = os.getenv(key)
        if raw is None or raw == "":
            config[key] = default
            continue
        config[key] = _coerce(key, raw)

    return config


def _coerce(key: str, raw: str) -> Any:
    """Приводит строковое значение из окружения к нужному типу.

    Args:
        key: имя переменной.
        raw: строковое значение из окружения.

    Returns:
        Значение, приведённое к типу, объявленному для данного ключа.
        Если приведение невозможно, возвращается строка без изменений.
    """
    if key in _BOOL_KEYS:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    try:
        if key in _INT_KEYS:
            return int(raw)
        if key in _FLOAT_KEYS:
            return float(raw)
    except ValueError:
        return raw
    return raw


def validate_api_key(config: Dict[str, Any]) -> None:
    """Проверяет наличие API-ключа в конфигурации.

    В mock-режиме ключ не обязателен.

    Args:
        config: словарь конфигурации (результат get_config).

    Raises:
        ValueError: если ключ отсутствует/пуст и mock-режим выключен.
    """
    if config.get("MOCK_MODE"):
        return
    api_key = (config.get("DEEPSEEK_API_KEY") or "").strip()
    if not api_key:
        raise ValueError(
            "API-ключ DeepSeek не найден.\n\n"
            "1. Создайте файл .env в корне проекта (см. .env.example).\n"
            "2. Укажите в нём: DEEPSEEK_API_KEY=ваш_ключ\n\n"
            "Либо задайте переменную окружения DEEPSEEK_API_KEY.\n"
            "Для тестирования без ключа можно включить mock-режим: MOCK_MODE=true"
        )


def _price_prefix(model: str) -> str:
    """Возвращает префикс переменных окружения с ценами для модели.

    Пример: "deepseek-v4-flash" → "PRICE_DEEPSEEK_V4_FLASH".
    """
    normalized = re.sub(r"[^A-Za-z0-9]", "_", model).upper()
    return f"PRICE_{normalized}"


def get_prices(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Собирает цены токенов (USD за 1M) из конфигурации и окружения.

    Приоритет: цена по конкретной модели → общая цена → 0.0 (не задана).

    Args:
        config: словарь конфигурации; None — взять из get_config().

    Returns:
        Словарь вида::

            {
              "default": {"input": float, "output": float},
              "models": {"deepseek-v4-flash": {"input": float, "output": float}},
            }
    """
    cfg = config if config is not None else get_config()

    def _to_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    prices: Dict[str, Any] = {
        "default": {
            "input": _to_float(cfg.get("PRICE_INPUT_PER_1M", 0.0)),
            "output": _to_float(cfg.get("PRICE_OUTPUT_PER_1M", 0.0)),
        },
        "models": {},
    }

    # Переопределения по моделям читаем из окружения: имена переменных
    # зависят от имени модели и не перечислены в DEFAULTS.
    for model in AVAILABLE_MODELS:
        prefix = _price_prefix(model)
        entry: Dict[str, float] = {}
        raw_input = os.getenv(f"{prefix}_{_PRICE_INPUT_SUFFIX}")
        raw_output = os.getenv(f"{prefix}_{_PRICE_OUTPUT_SUFFIX}")
        if raw_input not in (None, ""):
            entry["input"] = _to_float(raw_input)
        if raw_output not in (None, ""):
            entry["output"] = _to_float(raw_output)
        if entry:
            prices["models"][model] = entry

    return prices


def get_summary_settings(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Собирает и валидирует настройки сжатия контекста (summary).

    Некорректные значения заменяются значениями по умолчанию с записью
    предупреждения в лог (приложение не должно падать из-за конфигурации).

    Args:
        config: словарь конфигурации; None — взять из get_config().

    Returns:
        Словарь {"enabled", "keep_last", "every_n", "max_tokens",
        "temperature", "model", "prompt"}.
    """
    cfg = config if config is not None else get_config()
    logger = get_logger("deepseek-chat.config")

    def _int(key: str, default: int, low: int, high: int) -> int:
        raw = cfg.get(key, default)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            logger.warning("Некорректное значение %s=%r — беру %d.", key, raw, default)
            return default
        if not (low <= value <= high):
            logger.warning(
                "%s=%d вне диапазона [%d, %d] — беру %d.",
                key, value, low, high, default,
            )
            return default
        return value

    def _float(key: str, default: float, low: float, high: float) -> float:
        raw = cfg.get(key, default)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            logger.warning("Некорректное значение %s=%r — беру %.2f.", key, raw, default)
            return default
        if not (low <= value <= high):
            logger.warning(
                "%s=%.2f вне диапазона [%.2f, %.2f] — беру %.2f.",
                key, value, low, high, default,
            )
            return default
        return value

    return {
        "enabled": bool(cfg.get("SUMMARY_ENABLED", True)),
        "keep_last": _int("SUMMARY_KEEP_LAST_N", 10, 1, 1000),
        "every_n": _int("SUMMARY_EVERY_N", 10, 1, 200),
        "max_tokens": _int("SUMMARY_MAX_TOKENS", 400, 50, 4000),
        "temperature": _float("SUMMARY_TEMPERATURE", 0.2, 0.0, 2.0),
        "model": str(cfg.get("SUMMARY_MODEL") or "").strip(),
        "prompt": str(cfg.get("SUMMARY_PROMPT") or "").strip(),
    }


def get_strategy_settings(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Собирает и валидирует настройки стратегий управления контекстом.

    Некорректные значения заменяются значениями по умолчанию с записью
    предупреждения в лог (приложение не должно падать из-за конфигурации).
    Неизвестное имя стратегии заменяется на ``sliding_window``.

    Args:
        config: словарь конфигурации; None — взять из get_config().

    Returns:
        Словарь {"name", "window", "branch_strategy", "branch_max",
        "facts_enabled", "facts_model", "facts_temperature",
        "facts_max_tokens", "facts_max_items", "facts_prompt"}.
    """
    # Импорт внутри функции: config не должен зависеть от стратегий на старте
    from .strategies import (
        BRANCHING,
        DEFAULT_WINDOW_KEEP_LAST_N,
        SLIDING_WINDOW,
        normalize_name,
    )

    cfg = config if config is not None else get_config()
    logger = get_logger("deepseek-chat.config")

    def _int(key: str, default: int, low: int, high: int) -> int:
        raw = cfg.get(key, default)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            logger.warning("Некорректное значение %s=%r — беру %d.", key, raw, default)
            return default
        if not (low <= value <= high):
            logger.warning(
                "%s=%d вне диапазона [%d, %d] — беру %d.",
                key, value, low, high, default,
            )
            return default
        return value

    def _float(key: str, default: float, low: float, high: float) -> float:
        raw = cfg.get(key, default)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            logger.warning("Некорректное значение %s=%r — беру %.2f.", key, raw, default)
            return default
        if not (low <= value <= high):
            logger.warning(
                "%s=%.2f вне диапазона [%.2f, %.2f] — беру %.2f.",
                key, value, low, high, default,
            )
            return default
        return value

    raw_name = str(cfg.get("STRATEGY", SLIDING_WINDOW) or "").strip()
    try:
        name = normalize_name(raw_name)
    except Exception:
        logger.warning(
            "Неизвестная стратегия STRATEGY=%r — беру %s.",
            raw_name, SLIDING_WINDOW,
        )
        name = SLIDING_WINDOW

    # Подстратегия для режима веток: у режима branching своей сборки контекста
    # нет, поэтому запоминаем, какая стратегия действует внутри ветки.
    branch_strategy = name if name != BRANCHING else SLIDING_WINDOW

    return {
        "name": name,
        "window": _int("WINDOW_KEEP_LAST_N", DEFAULT_WINDOW_KEEP_LAST_N, 1, 1000),
        "branch_strategy": branch_strategy,
        "branch_max": _int("BRANCH_MAX", 20, 1, 500),
        "facts_enabled": bool(cfg.get("FACTS_ENABLED", True)),
        "facts_model": str(cfg.get("FACTS_MODEL") or "").strip(),
        "facts_temperature": _float("FACTS_TEMPERATURE", 0.0, 0.0, 2.0),
        "facts_max_tokens": _int("FACTS_MAX_TOKENS", 300, 20, 4000),
        "facts_max_items": _int("FACTS_MAX_ITEMS", 50, 1, 500),
        "facts_prompt": str(cfg.get("FACTS_PROMPT") or "").strip(),
    }


def get_memory_settings(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Собирает и валидирует настройки модели памяти (memory layers).

    Некорректные значения заменяются значениями по умолчанию с записью
    предупреждения в лог (приложение не должно падать из-за конфигурации).
    Пустая строка ``MEMORY_LONG_PATH`` выключает долговременный слой.

    Args:
        config: словарь конфигурации; None — взять из get_config().

    Returns:
        Словарь {"enabled", "working_enabled", "long_enabled", "long_path",
        "long_max_items", "max_tokens", "long_max_tokens",
        "working_max_tokens", "pin_max", "propose_llm", "model",
        "temperature", "prompt", "min_confidence", "auto_apply",
        "export_path"}.
    """
    cfg = config if config is not None else get_config()
    logger = get_logger("deepseek-chat.config")

    def _int(key: str, default: int, low: int, high: int) -> int:
        raw = cfg.get(key, default)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            logger.warning("Некорректное значение %s=%r — беру %d.", key, raw, default)
            return default
        if not (low <= value <= high):
            logger.warning(
                "%s=%d вне диапазона [%d, %d] — беру %d.",
                key, value, low, high, default,
            )
            return default
        return value

    def _float(key: str, default: float, low: float, high: float) -> float:
        raw = cfg.get(key, default)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            logger.warning("Некорректное значение %s=%r — беру %.2f.", key, raw, default)
            return default
        if not (low <= value <= high):
            logger.warning(
                "%s=%.2f вне диапазона [%.2f, %.2f] — беру %.2f.",
                key, value, low, high, default,
            )
            return default
        return value

    max_tokens = _int("MEMORY_MAX_TOKENS", 600, 100, 4000)
    long_max_tokens = _int("MEMORY_LONG_MAX_TOKENS", 300, 20, max_tokens)
    working_max_tokens = _int("MEMORY_WORKING_MAX_TOKENS", 250, 20, max_tokens)

    # Сумма бюджетов слоёв не должна превышать общий бюджет: иначе усекаем
    # бюджеты слоёв пропорционально (с предупреждением).
    if long_max_tokens + working_max_tokens > max_tokens:
        logger.warning(
            "MEMORY_LONG_MAX_TOKENS(%d) + MEMORY_WORKING_MAX_TOKENS(%d) > "
            "MEMORY_MAX_TOKENS(%d) — усекаю бюджеты слоёв.",
            long_max_tokens, working_max_tokens, max_tokens,
        )
        long_max_tokens = max(20, max_tokens // 2)
        working_max_tokens = max(20, max_tokens - long_max_tokens)

    # Путь к БД долговременной памяти: отсутствие ключа — значение по
    # умолчанию, явная пустая строка — слой выключен.
    raw_long_path = cfg.get("MEMORY_LONG_PATH")
    if raw_long_path is None:
        raw_long_path = DEFAULTS["MEMORY_LONG_PATH"]
    long_path = str(raw_long_path or "").strip()
    long_enabled = bool(cfg.get("MEMORY_LONG_ENABLED", True))

    return {
        "enabled": bool(cfg.get("MEMORY_ENABLED", True)),
        "working_enabled": bool(cfg.get("MEMORY_WORKING_ENABLED", True)),
        "long_enabled": long_enabled and bool(long_path),
        "long_path": long_path,
        "long_max_items": _int("MEMORY_LONG_MAX_ITEMS", 200, 1, 100000),
        "max_tokens": max_tokens,
        "long_max_tokens": long_max_tokens,
        "working_max_tokens": working_max_tokens,
        "pin_max": _int("MEMORY_PIN_MAX", 5, 1, 50),
        "propose_llm": bool(cfg.get("MEMORY_PROPOSE_LLM", True)),
        "model": str(cfg.get("MEMORY_MODEL") or "").strip(),
        "temperature": _float("MEMORY_TEMPERATURE", 0.0, 0.0, 2.0),
        "prompt": str(cfg.get("MEMORY_PROMPT") or "").strip(),
        "min_confidence": _float("MEMORY_MIN_CONFIDENCE", 0.6, 0.0, 1.0),
        "auto_apply": bool(cfg.get("MEMORY_AUTO_APPLY", False)),
        "export_path": str(cfg.get("MEMORY_EXPORT_PATH")
                           or "data/memory_export.md").strip(),
    }


def get_invariant_settings(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Собирает и валидирует настройки пост-проверки инвариантов.

    Некорректные значения заменяются значениями по умолчанию с записью
    предупреждения в лог (приложение не должно падать из-за конфигурации).

    Args:
        config: словарь конфигурации; None — взять из get_config().

    Returns:
        Словарь {"enabled", "max_regenerations", "min_response_length"}.
    """
    cfg = config if config is not None else get_config()
    logger = get_logger("deepseek-chat.config")

    def _int(key: str, default: int, low: int, high: int) -> int:
        raw = cfg.get(key, default)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            logger.warning("Некорректное значение %s=%r — беру %d.", key, raw, default)
            return default
        if not (low <= value <= high):
            logger.warning(
                "%s=%d вне диапазона [%d, %d] — беру %d.",
                key, value, low, high, default,
            )
            return default
        return value

    return {
        "enabled": bool(cfg.get("ENABLE_POST_CHECK", True)),
        "max_regenerations": _int("MAX_REGENERATIONS", 2, 0, 10),
        "min_response_length": _int("MIN_RESPONSE_LENGTH", 100, 0, 100000),
    }


def get_task_settings(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Собирает и валидирует настройки подсистемы задач (FSM).

    Некорректные значения заменяются значениями по умолчанию с записью
    предупреждения в лог (приложение не должно падать из-за конфигурации).

    Args:
        config: словарь конфигурации; None — взять из get_config().

    Returns:
        Словарь {"max_retries"} — лимит откатов validation → execution.
    """
    cfg = config if config is not None else get_config()
    logger = get_logger("deepseek-chat.config")

    raw = cfg.get("TASKS_MAX_RETRIES", 3)
    try:
        max_retries = int(raw)
    except (TypeError, ValueError):
        logger.warning("Некорректное значение TASKS_MAX_RETRIES=%r — беру 3.", raw)
        max_retries = 3
    if not (0 <= max_retries <= 100):
        logger.warning(
            "TASKS_MAX_RETRIES=%d вне диапазона [0, 100] — беру 3.",
            max_retries,
        )
        max_retries = 3

    return {"max_retries": max_retries}


def get_project_path(config: Dict[str, Any], key: str) -> Path:
    """Разрешает путь из конфигурации относительно корня проекта.

    Args:
        config: словарь конфигурации.
        key: ключ конфигурации со строкой-путём (например, APP_STATE_PATH).

    Returns:
        Path: абсолютный путь. Относительные значения отсчитываются от
        PROJECT_ROOT (как это делает main.py для STORAGE_PATH/TASKS_PATH).
    """
    raw = str(config.get(key) or DEFAULTS.get(key) or "")
    path = Path(raw) if raw else Path(DEFAULTS.get(key, ""))
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_mcp_settings(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Собирает и валидирует настройки MCP-клиента (панель инструментов).

    Команда сервера разбирается через shlex: ``MCP_SERVER_COMMAND`` —
    одна строка вида ``npx -y open-internet-mcp`` или
    ``python -u my_git_server.py`` (собственный сервер). Для локальных
    серверов применяются два правила-resolve, чтобы команда работала из
    любого рабочего каталога и любого интерпретатора:

    * голый ``python``/``python3`` заменяется на ``sys.executable`` —
      тот интерпретатор, под которым запущено приложение (у него есть
      все зависимости, включая SDK ``mcp``);
    * относительный путь к ``*.py``-файлу в последнем аргументе
      разрешается от корня проекта (PROJECT_ROOT).

    Некорректные значения заменяются значениями по умолчанию с записью
    предупреждения в лог (приложение не должно падать из-за конфигурации).

    Args:
        config: словарь конфигурации; None — взять из get_config().

    Returns:
        Словарь {"enabled", "server_command" (список аргументов),
        "timeout" (сек), "log_path" (абсолютный Path),
        "git_repo_path" (абсолютный Path — репозиторий по умолчанию
        для инструментов get_git_*)}.
    """
    import shlex

    cfg = config if config is not None else get_config()
    logger = get_logger("deepseek-chat.config")

    default_command = str(DEFAULTS["MCP_SERVER_COMMAND"])
    raw_command = str(cfg.get("MCP_SERVER_COMMAND") or "").strip()
    if not raw_command:
        raw_command = default_command
    # На Windows posix-режим shlex «съедает» обратные слэши в путях,
    # поэтому там разбор в не-posix режиме; он оставляет кавычки в токенах —
    # срезаем их (пути вида "C:\Program Files\x.exe" должны сохраниться).
    try:
        parts = shlex.split(raw_command, posix=(os.name != "nt"))
    except ValueError:
        logger.warning(
            "Некорректная команда MCP_SERVER_COMMAND=%r — беру %r.",
            raw_command, default_command,
        )
        parts = default_command.split()
    parts = [part.strip('"').strip("'") for part in parts]
    if not parts or not parts[0]:
        logger.warning(
            "Пустая команда MCP_SERVER_COMMAND=%r — беру %r.",
            raw_command, default_command,
        )
        parts = default_command.split()
    parts = _resolve_local_server_command(parts)

    raw_timeout = cfg.get("MCP_TIMEOUT", DEFAULTS["MCP_TIMEOUT"])
    try:
        timeout = float(raw_timeout)
    except (TypeError, ValueError):
        logger.warning(
            "Некорректное значение MCP_TIMEOUT=%r — беру 30.", raw_timeout
        )
        timeout = 30.0
    if not (1.0 <= timeout <= 600.0):
        logger.warning(
            "MCP_TIMEOUT=%.0f вне диапазона [1, 600] — беру 30.", timeout
        )
        timeout = 30.0

    return {
        "enabled": bool(cfg.get("MCP_ENABLED", True)),
        "server_command": parts,
        "timeout": timeout,
        "log_path": get_project_path(cfg, "MCP_LOG_PATH"),
        "git_repo_path": get_project_path(cfg, "MCP_GIT_REPO_PATH"),
    }


def _resolve_local_server_command(parts: List[str]) -> List[str]:
    """Разрешает локальную команду MCP-сервера (python-скрипт проекта).

    Правила (см. get_mcp_settings): голый ``python``/``python3`` →
    ``sys.executable``; относительный ``*.py`` в последнем аргументе →
    абсолютный путь от PROJECT_ROOT. Команды вида ``npx ...`` и пути
    с разделителями каталогов не трогаются.

    Args:
        parts: разобранные shlex-токены команды.

    Returns:
        Список токенов после подстановок (копия).
    """
    resolved = list(parts)
    if resolved[0].lower() in ("python", "python3", "python.exe", "python3.exe"):
        resolved[0] = sys.executable
    last = resolved[-1]
    if last.lower().endswith(".py"):
        path = Path(last)
        if not path.is_absolute():
            resolved[-1] = str(PROJECT_ROOT / path)
    return resolved


def get_reminder_settings(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Собирает и валидирует настройки подсистемы напоминаний.

    Некорректные значения заменяются значениями по умолчанию с записью
    предупреждения в лог (приложение не должно падать из-за конфигурации).

    Args:
        config: словарь конфигурации; None — взять из get_config().

    Returns:
        Словарь {"enabled", "host", "port", "url", "max_tool_rounds"}:
        host — только 127.0.0.1 (хостить в интернете запрещено по ТЗ);
        url — MCP-эндпоинт ``http://<host>:<port>/mcp``.
    """
    cfg = config if config is not None else get_config()
    logger = get_logger("deepseek-chat.config")

    host = str(cfg.get("REMINDER_SERVER_HOST") or DEFAULTS["REMINDER_SERVER_HOST"])
    host = host.strip() or str(DEFAULTS["REMINDER_SERVER_HOST"])
    if host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning(
            "REMINDER_SERVER_HOST=%s небезопасен — сервер напоминаний "
            "не должен быть доступен из сети, беру 127.0.0.1.", host,
        )
        host = "127.0.0.1"

    raw_port = cfg.get("REMINDER_SERVER_PORT", DEFAULTS["REMINDER_SERVER_PORT"])
    try:
        port = int(raw_port)
    except (TypeError, ValueError):
        logger.warning(
            "Некорректное значение REMINDER_SERVER_PORT=%r — беру 8000.",
            raw_port,
        )
        port = 8000
    if not (1 <= port <= 65535):
        logger.warning(
            "REMINDER_SERVER_PORT=%d вне диапазона [1, 65535] — беру 8000.",
            port,
        )
        port = 8000

    raw_rounds = cfg.get(
        "REMINDER_TOOLCALL_MAX_ROUNDS",
        DEFAULTS["REMINDER_TOOLCALL_MAX_ROUNDS"],
    )
    try:
        max_rounds = int(raw_rounds)
    except (TypeError, ValueError):
        logger.warning(
            "Некорректное значение REMINDER_TOOLCALL_MAX_ROUNDS=%r — беру 5.",
            raw_rounds,
        )
        max_rounds = 5
    if not (1 <= max_rounds <= 20):
        logger.warning(
            "REMINDER_TOOLCALL_MAX_ROUNDS=%d вне диапазона [1, 20] — беру 5.",
            max_rounds,
        )
        max_rounds = 5

    return {
        "enabled": _coerce_bool(cfg.get("REMINDER_ENABLED", True)),
        "host": host,
        "port": port,
        "url": f"http://{host}:{port}/mcp",
        "max_tool_rounds": max_rounds,
    }


def _coerce_bool(value: Any) -> bool:
    """Приводит значение настройки к bool (строки «true»/«false» тоже)."""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)
