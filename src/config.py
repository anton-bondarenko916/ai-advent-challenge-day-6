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
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import AVAILABLE_MODELS

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
}

# Переменные, значения которых нужно приводить к логическому типу
_BOOL_KEYS = ("MOCK_MODE", "STORAGE_ENABLED", "TOKENIZER_ENABLED")

# Переменные, значения которых нужно приводить к числу
_INT_KEYS = ("UI_WIDTH", "UI_HEIGHT", "DEFAULT_MAX_TOKENS", "MAX_HISTORY_MESSAGES")
_FLOAT_KEYS = ("DEFAULT_TEMPERATURE", "PRICE_INPUT_PER_1M", "PRICE_OUTPUT_PER_1M")

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
