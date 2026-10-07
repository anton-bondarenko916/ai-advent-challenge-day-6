"""Вспомогательные функции: валидация, логирование, утилиты времени.

Содержит:
- параметры-константы для валидации модели;
- функцию get_logger для настройки логирования;
- форматирование времени;
- исключение, транслирующее ошибки API в понятные сообщения.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Допустимые параметры
# ---------------------------------------------------------------------------

# Список моделей, доступных в выпадающем списке (DeepSeek — облако,
# qwen/llama — локальные модели Ollama, провайдеры src.llm_providers)
AVAILABLE_MODELS = (
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "qwen2.5:7b",
    "llama3.1:8b",
)

TEMPERATURE_RANGE = (0.0, 2.0)  # допустимый диапазон температуры
MAX_TOKENS_RANGE = (1, 4096)  # допустимый диапазон max_tokens
MAX_HISTORY_LIMIT = 50  # максимальное число сообщений в истории (без system)


class AgentConfigError(ValueError):
    """Ошибка некорректной конфигурации агента (параметров модели)."""


class ApiError(Exception):
    """Ошибка обращения к DeepSeek API с «человекочитаемым» текстом.

    Attributes:
        message: текст ошибки для пользователя.
        original: исходное исключение (при наличии).
    """

    def __init__(self, message: str, original: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.original = original


def validate_model(model: str) -> str:
    """Проверяет, что модель входит в список поддерживаемых.

    Args:
        model: имя модели.

    Returns:
        Модель без изменений (для цепочек вызовов).

    Raises:
        AgentConfigError: если модель не поддерживается.
    """
    if not isinstance(model, str) or not model.strip():
        raise AgentConfigError("Модель не может быть пустой.")
    if model not in AVAILABLE_MODELS:
        raise AgentConfigError(
            f"Неизвестная модель: {model!r}. "
            f"Доступные модели: {', '.join(AVAILABLE_MODELS)}"
        )
    return model


def validate_temperature(value: float) -> float:
    """Проверяет температуру на попадание в диапазон [0.0, 2.0].

    Args:
        value: температура.

    Returns:
        Значение, округлённое до одного знака после запятой.

    Raises:
        AgentConfigError: если значение вне допустимого диапазона.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise AgentConfigError(f"Температура должна быть числом, получено: {value!r}")
    low, high = TEMPERATURE_RANGE
    if not (low <= number <= high):
        raise AgentConfigError(
            f"Температура должна быть в диапазоне [{low}, {high}], получено: {number}"
        )
    return round(number, 1)


def validate_max_tokens(value: int) -> int:
    """Проверяет max_tokens: целое число не меньше 1.

    Args:
        value: максимальное число токенов.

    Returns:
        Проверенное целое значение.

    Raises:
        AgentConfigError: если значение некорректно.
    """
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise AgentConfigError(f"max_tokens должен быть целым числом, получено: {value!r}")
    low, high = MAX_TOKENS_RANGE
    if number < low or number > high:
        raise AgentConfigError(
            f"max_tokens должен быть в диапазоне [{low}, {high}], получено: {number}"
        )
    return number


def validate_system_prompt(prompt: str) -> str:
    """Проверяет системный промпт на непустоту.

    Args:
        prompt: текст системного промпта.

    Returns:
        Промпт с обрезанными краевыми пробелами.

    Raises:
        AgentConfigError: если промпт пуст.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise AgentConfigError("Системный промпт не может быть пустым.")
    return prompt.strip()


def format_time(timestamp: Optional[datetime] = None) -> str:
    """Форматирует время в виде ЧЧ:ММ:СС (по умолчанию — текущее).

    Args:
        timestamp: момент времени; None — взять текущее время.

    Returns:
        Строка вида "12:34:56".
    """
    moment = timestamp or datetime.now()
    return moment.strftime("%H:%M:%S")


def format_tokens(tokens: Optional[dict]) -> str:
    """Форматирует информацию об использованных токенах для статус-бара.

    Args:
        tokens: словарь с ключами prompt_tokens/completion_tokens/total_tokens.

    Returns:
        Строка вида "in: 12 · out: 34 · total: 46" либо "—".
    """
    if not tokens:
        return "—"
    prompt = tokens.get("prompt_tokens")
    completion = tokens.get("completion_tokens")
    total = tokens.get("total_tokens")
    if prompt is None and completion is None and total is None:
        return "—"
    parts = []
    if prompt is not None:
        parts.append(f"in: {prompt}")
    if completion is not None:
        parts.append(f"out: {completion}")
    if total is not None:
        parts.append(f"total: {total}")
    return " · ".join(parts)


def get_logger(name: str = "deepseek-chat", level: int = logging.INFO) -> logging.Logger:
    """Возвращает настроенный логгер приложения.

    Вывод идёт в stderr. Повторные вызовы не дублируют обработчики.

    Args:
        name: имя логгера.
        level: уровень логирования.

    Returns:
        logging.Logger: логгер.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # логгер уже настроен
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def humanize_api_error(exc: Exception) -> str:
    """Преобразует исключение (например, из openai SDK) в понятный текст.

    Ключ API никогда не выводится. Распознаются типовые ошибки:
    неверный ключ, превышение лимитов, сетевая недоступность.

    Args:
        exc: исходное исключение.

    Returns:
        Строка с понятным описанием ошибки.
    """
    text = str(exc)
    lowered = text.lower()

    if "api key" in lowered or "authentication" in lowered or "401" in text:
        return ("Ошибка аутентификации: неверный или отсутствующий API-ключ. "
                "Проверьте DEEPSEEK_API_KEY в файле .env.")
    if "insufficient_quota" in lowered or "429" in text or "rate limit" in lowered:
        return ("Превышен лимит запросов или исчерпана квота. "
                "Повторите позже или проверьте баланс аккаунта.")
    if "invalid_request" in lowered or "400" in text:
        return f"Некорректный запрос к API: {text[:300]}"
    if "connection" in lowered or "network" in lowered or "timed out" in lowered \
            or "timeout" in lowered or "econnrefused" in lowered \
            or "econnreset" in lowered or "socket" in lowered:
        return ("Сетевая ошибка: не удалось связаться с API. "
                "Проверьте подключение к интернету и повторите попытку.")
    return text[:500] if text else "Неизвестная ошибка при обращении к API."
