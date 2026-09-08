"""Инкапсулированный AI-агент для работы с DeepSeek API.

Класс DeepSeekAgent скрывает все детали взаимодействия с HTTP API
(клиент, формат запроса, разбор ответа) и предоставляет простой
интерфейс: send_message / clear_history / update_config.

Агент:
- хранит историю диалога (системный промпт + сообщения) между вызовами;
- ограничивает историю последними MAX_HISTORY_MESSAGES сообщениями;
- потокобезопасен (все операции под RLock);
- не выбрасывает исключений наружу из send_message — ошибки
  возвращаются в результирующем словаре под ключом "error";
- поддерживает mock-режим (имитация ответа без сети) для тестов.
"""

from __future__ import annotations

import random
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import OpenAI

from . import utils
from .utils import AgentConfigError, ApiError, humanize_api_error

# Тип сообщения истории: {"role": "system"|"user"|"assistant", "content": str}
Message = Dict[str, str]

# Ответ по умолчанию для mock-режима
_MOCK_ANSWERS = [
    "Это ответ в mock-режиме: реальный запрос к DeepSeek API не отправлялся.",
    "Mock-режим активен. Настройте DEEPSEEK_API_KEY в .env и перезапустите "
    "приложение для реальных ответов.",
]


class DeepSeekAgent:
    """AI-агент, инкапсулирующий диалог и запросы к DeepSeek API.

    Attributes (публичные, изменяемые через update_config):
        model: имя модели.
        system_prompt: системный промпт.
        temperature: температура (0.0–2.0).
        max_tokens: максимальное число токенов ответа (>= 1).

    Приватные атрибуты:
        _api_key: ключ API.
        _client: клиент openai.OpenAI.
        _message_history: история сообщений {role, content}.
        _lock: блокировка потокобезопасности.
        _base_url: приватная константа адреса API.
    """

    # Приватная константа адреса API
    _base_url: str = "https://api.deepseek.com"

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        mock_mode: bool = False,
        max_history_messages: int = utils.MAX_HISTORY_LIMIT,
    ) -> None:
        """Инициализирует агента и сохраняет системный промпт в истории.

        Args:
            api_key: ключ DeepSeek API (обязателен вне mock-режима).
            model: имя модели; по умолчанию "deepseek-v4-flash".
            system_prompt: системный промпт; по умолчанию стандартный.
            temperature: температура; по умолчанию 0.7.
            max_tokens: лимит токенов ответа; по умолчанию 1000.
            mock_mode: если True, ответы имитируются без обращения к API.
            max_history_messages: максимум сообщений истории без system.

        Raises:
            AgentConfigError: если какой-либо параметр некорректен.
        """
        # Заполняем значения по умолчанию
        self.model = model or "deepseek-v4-flash"
        self.system_prompt = system_prompt or (
            "Ты полезный AI-ассистент. Отвечай четко и по делу."
        )
        self.temperature = temperature if temperature is not None else 0.7
        self.max_tokens = max_tokens if max_tokens is not None else 1000

        # Проверяем параметры ДО создания клиента
        self._validate_parameters()

        self._api_key: str = api_key or ""
        self._mock_mode: bool = bool(mock_mode)
        self._max_history: int = int(max_history_messages)
        self._lock = threading.RLock()

        # Клиент создаётся только вне mock-режима
        self._client: Optional[OpenAI] = None
        if not self._mock_mode:
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=DeepSeekAgent._base_url,
                timeout=60.0,
                max_retries=1,
            )

        # История начинается с системного промпта
        self._message_history: List[Message] = []
        self._init_history()

    # ------------------------------------------------------------------
    # Публичное API
    # ------------------------------------------------------------------

    def send_message(self, user_input: str) -> Dict[str, Any]:
        """Отправляет сообщение пользователя агенту и возвращает ответ.

        Сообщение добавляется в историю, формируется запрос со всей
        историей (включая системный промпт), измеряется время ответа.
        Результат ассистента дописывается в историю.

        При любой ошибке (сеть, ключ, API) исключение не пробрасывается:
        возвращается словарь с заполненным полем "error".

        Args:
            user_input: текст сообщения пользователя.

        Returns:
            Словарь с полями:
            - text: текст ответа (None при ошибке);
            - model: использованная модель;
            - tokens: {prompt_tokens, completion_tokens, total_tokens}
              или None, если API не вернул статистику;
            - response_time: время запроса в секундах;
            - finish_reason: причина завершения (stop, length и т.п.);
            - error: None либо текст ошибки;
            - timestamp: локальное время ответа (строка).
        """
        if not user_input or not user_input.strip():
            return self._error_result("Пустое сообщение.")

        with self._lock:
            self._append_message({"role": "user", "content": user_input.strip()})

            start = time.perf_counter()
            try:
                if self._mock_mode:
                    payload = self._mock_response(user_input)
                else:
                    payload = self._call_api()

                elapsed = round(time.perf_counter() - start, 2)
                self._append_message(
                    {"role": "assistant", "content": payload["content"]}
                )
                self._trim_history()

                return {
                    "text": payload["content"],
                    "model": payload.get("model", self.model),
                    "tokens": payload.get("tokens"),
                    "response_time": elapsed,
                    "finish_reason": payload.get("finish_reason"),
                    "error": None,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            except ApiError as exc:
                # При ошибке откатываем сообщение пользователя из истории,
                # чтобы повторная отправка того же текста не дублировала его.
                self._rollback_user_message()
                return self._error_result(
                    exc.message,
                    response_time=round(time.perf_counter() - start, 2),
                )
            except Exception as exc:  # любая непредвиденная ошибка
                self._rollback_user_message()
                return self._error_result(
                    humanize_api_error(exc),
                    response_time=round(time.perf_counter() - start, 2),
                )

    def clear_history(self) -> None:
        """Очищает историю диалога, оставляя только системный промпт."""
        with self._lock:
            self._init_history()

    def update_config(self, **kwargs: Any) -> None:
        """Обновляет параметры агента «на лету» с проверкой валидности.

        Поддерживаемые ключи: model, system_prompt, temperature,
        max_tokens, mock_mode, max_history_messages.

        Args:
            **kwargs: обновляемые параметры.

        Raises:
            AgentConfigError: если значение какого-либо параметра некорректно.
        """
        allowed = {
            "model", "system_prompt", "temperature",
            "max_tokens", "mock_mode", "max_history_messages",
        }
        unknown = set(kwargs) - allowed
        if unknown:
            raise AgentConfigError(
                f"Неизвестные параметры конфигурации: {', '.join(sorted(unknown))}"
            )

        with self._lock:
            # Валидируем всё до применения изменений
            model = utils.validate_model(kwargs.get("model", self.model))
            system_prompt = utils.validate_system_prompt(
                kwargs.get("system_prompt", self.system_prompt)
            )
            temperature = utils.validate_temperature(
                kwargs.get("temperature", self.temperature)
            )
            max_tokens = utils.validate_max_tokens(
                kwargs.get("max_tokens", self.max_tokens)
            )
            new_mock = bool(kwargs.get("mock_mode", self._mock_mode))
            new_max_history = int(
                kwargs.get("max_history_messages", self._max_history)
            )
            if new_max_history < 1:
                raise AgentConfigError("max_history_messages должен быть >= 1.")

            # Применяем изменения
            if "model" in kwargs:
                self.model = model
            if "system_prompt" in kwargs:
                # Смена системного промпта сбрасывает историю диалога
                self.system_prompt = system_prompt
                self._init_history()
            if "temperature" in kwargs:
                self.temperature = temperature
            if "max_tokens" in kwargs:
                self.max_tokens = max_tokens
            if "mock_mode" in kwargs and new_mock != self._mock_mode:
                self._mock_mode = new_mock
                self._client = None if new_mock else OpenAI(
                    api_key=self._api_key,
                    base_url=DeepSeekAgent._base_url,
                    timeout=60.0,
                    max_retries=1,
                )
            if "max_history_messages" in kwargs:
                self._max_history = new_max_history
                self._trim_history()

    # ------------------------------------------------------------------
    # Приватные методы
    # ------------------------------------------------------------------

    def _validate_parameters(self) -> None:
        """Проверяет корректность текущих параметров агента.

        Raises:
            AgentConfigError: с описанием первой найденной проблемы.
        """
        utils.validate_model(self.model)
        utils.validate_system_prompt(self.system_prompt)
        utils.validate_temperature(self.temperature)
        utils.validate_max_tokens(self.max_tokens)

    def _build_request(self) -> Dict[str, Any]:
        """Формирует структуру запроса к API.

        Returns:
            Словарь с параметрами chat.completions.create: модель,
            полная история сообщений, температура, max_tokens.
        """
        return {
            "model": self.model,
            "messages": list(self._message_history),  # вся история, включая system
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _call_api(self) -> Dict[str, Any]:
        """Выполняет реальный запрос к DeepSeek API и разбирает ответ.

        Returns:
            Словарь с ключами content, model, tokens, finish_reason.

        Raises:
            ApiError: при ошибке аутентификации, сети, лимитов и т.п.
        """
        if self._client is None:
            raise ApiError("Клиент API не инициализирован (mock-режим?).")

        request = self._build_request()
        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            raise ApiError(humanize_api_error(exc), original=exc)

        return self._handle_response(response)

    def _handle_response(self, response: Any) -> Dict[str, Any]:
        """Обрабатывает ответ API и извлекает данные.

        Args:
            response: объект ответа openai (chat.completions).

        Returns:
            Словарь с ключами content, model, tokens, finish_reason.

        Raises:
            ApiError: если структура ответа неожиданная (нет choices).
        """
        if response is None:
            raise ApiError("API вернул пустой ответ.")

        choices = getattr(response, "choices", None)
        if not choices:
            raise ApiError("API вернул ответ без choices.")

        choice = choices[0]
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None) if message else None
        if content is None:
            content = ""  # например, если включён stream и контент в другом поле

        usage = getattr(response, "usage", None)
        tokens = None
        if usage is not None:
            tokens = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }

        return {
            "content": content,
            "model": getattr(response, "model", None) or self.model,
            "tokens": tokens,
            "finish_reason": getattr(choice, "finish_reason", None),
        }

    def _mock_response(self, user_input: str) -> Dict[str, Any]:
        """Имитирует ответ ассистента без обращения к API (для тестов).

        Args:
            user_input: текст сообщения пользователя (для эха).

        Returns:
            Словарь той же структуры, что и _handle_response.
        """
        text = random.choice(_MOCK_ANSWERS) + f"\n\nВы написали: {user_input.strip()}"
        return {
            "content": text,
            "model": self.model,
            "tokens": {
                "prompt_tokens": len(self._message_history),
                "completion_tokens": len(text.split()),
                "total_tokens": len(self._message_history) + len(text.split()),
            },
            "finish_reason": "stop",
        }

    def _init_history(self) -> None:
        """Сбрасывает историю: остаётся только системный промпт."""
        self._message_history = [
            {"role": "system", "content": self.system_prompt}
        ]

    def _append_message(self, message: Message) -> None:
        """Добавляет сообщение в историю (вызывается только под блокировкой)."""
        self._message_history.append(message)

    def _trim_history(self) -> None:
        """Ограничивает историю последними _max_history сообщениями.

        Системный промпт (первый элемент) никогда не удаляется.
        """
        system = self._message_history[0] if self._message_history else None
        rest = self._message_history[1:]
        if len(rest) > self._max_history:
            rest = rest[-self._max_history:]
        self._message_history = [system] + rest if system else rest

    def _rollback_user_message(self) -> None:
        """Удаляет последнее сообщение пользователя при неудачном запросе."""
        if self._message_history and self._message_history[-1].get("role") == "user":
            self._message_history.pop()

    def _error_result(self, message: str, response_time: Optional[float] = None) -> Dict[str, Any]:
        """Формирует результирующий словарь ошибки."""
        return {
            "text": None,
            "model": self.model,
            "tokens": None,
            "response_time": response_time if response_time is not None else 0.0,
            "finish_reason": None,
            "error": message,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    # ------------------------------------------------------------------
    # Служебные свойства (для UI и отладки)
    # ------------------------------------------------------------------

    @property
    def history(self) -> List[Message]:
        """Копия текущей истории сообщений (только для чтения)."""
        with self._lock:
            return list(self._message_history)

    @property
    def is_mock(self) -> bool:
        """True, если агент работает в mock-режиме (без реального API)."""
        return self._mock_mode
