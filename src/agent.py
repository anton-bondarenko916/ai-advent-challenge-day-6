"""Инкапсулированный AI-агент для работы с DeepSeek API.

Класс DeepSeekAgent скрывает все детали взаимодействия с HTTP API
(клиент, формат запроса, разбор ответа) и предоставляет простой
интерфейс: send_message / clear_history / update_config.

Агент:
- хранит историю диалога (системный промпт + сообщения) между вызовами;
- ограничивает историю последними MAX_HISTORY_MESSAGES сообщениями
  (когда сжатие контекста выключено);
- потокобезопасен (все операции под RLock);
- не выбрасывает исключений наружу из send_message — ошибки
  возвращаются в результирующем словаре под ключом "error";
- поддерживает mock-режим (имитация ответа без сети) для тестов;
- умеет сохранять контекст (историю + параметры) в SQLite через
  ContextStore и восстанавливать его при следующем запуске; в
  mock-режиме хранилище не читается и не пишется;
- считает токены: точные значения из usage API для запроса и ответа,
  локальную оценку — для контекста истории и (при отсутствии usage)
  для запроса/ответа; накапливает статистику сессии и считает стоимость
  по ценам из конфигурации;
- управляет контекстом: последние N сообщений уходят в модель «как есть»,
  всё старшее заменяется конспектом (summary), который обновляется
  инкрементально каждые K сообщений, хранится отдельно и подставляется
  в запрос отдельным system-сообщением.
"""

from __future__ import annotations

import random
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import OpenAI

from . import utils
from .storage import ContextStore
from .summarizer import (
    ContextSummarizer,
    get_default_settings as get_default_summary_settings,
)
from .tokens import TokenCounter, create_counter
from .utils import AgentConfigError, ApiError, humanize_api_error

logger = utils.get_logger("deepseek-chat.agent")

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
        storage: Optional[ContextStore] = None,
        token_counter: Optional[TokenCounter] = None,
        prices: Optional[Dict[str, Any]] = None,
        summary_settings: Optional[Dict[str, Any]] = None,
        summarizer: Optional[ContextSummarizer] = None,
    ) -> None:
        """Инициализирует агента и сохраняет системный промпт в истории.

        Если передано хранилище (ContextStore), агент пытается загрузить
        сохранённый ранее контекст: восстановленные параметры переопределяют
        значения, переданные в конструктор. В mock-режиме хранилище
        игнорируется (ни чтение, ни запись).

        Args:
            api_key: ключ DeepSeek API (обязателен вне mock-режима).
            model: имя модели; по умолчанию "deepseek-v4-flash".
            system_prompt: системный промпт; по умолчанию стандартный.
            temperature: температура; по умолчанию 0.7.
            max_tokens: лимит токенов ответа; по умолчанию 1000.
            mock_mode: если True, ответы имитируются без обращения к API.
            max_history_messages: максимум сообщений истории без system.
            storage: хранилище контекста (SQLite); None — без персиста.
            token_counter: счётчик токенов; None — создать автоматически
                (tiktoken при наличии, иначе встроенная эвристика).
            prices: цены токенов (USD за 1M) вида
                {"default": {"input": .., "output": ..},
                 "models": {"<model>": {"input": .., "output": ..}}};
                None/нули — стоимость не считается.
            summary_settings: настройки сжатия контекста (см.
                config.get_summary_settings); None — сжатие выключено.
            summarizer: готовый суммаризатор; None — создать из настроек.

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
        self._storage: Optional[ContextStore] = storage
        self._closed: bool = False

        # Учёт токенов: счётчик, цены, статистика сессии и последний обмен
        self._counter: TokenCounter = (
            token_counter if token_counter is not None
            else create_counter(model=self.model)
        )
        self._prices: Dict[str, Any] = dict(prices) if prices else {}
        self._session_stats: Dict[str, Any] = self._default_session_stats()
        self._last_exchange: Dict[str, Any] = self._default_last_exchange()

        # Управление контекстом: конспект старой части диалога
        self._summary_settings: Dict[str, Any] = (
            dict(summary_settings) if summary_settings is not None
            else get_default_summary_settings()
        )
        self._summarizer: Optional[ContextSummarizer] = summarizer
        self._summary: Optional[Dict[str, Any]] = None
        self._last_summary_error: Optional[str] = None
        if self._summarizer is None:
            self._summarizer = self._create_summarizer()

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

        # Восстановление контекста из хранилища (только вне mock-режима)
        if not self._mock_mode:
            self._load_context()

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
            - tokens: словарь статистики последнего обмена:
              prompt_tokens/completion_tokens/total_tokens (значения API,
              либо оценка, если API не вернул usage), estimated (True —
              значения оценочные), context_tokens (оценка всего контекста),
              session_total_tokens, cost_usd, session_cost_usd,
              cache_hit_tokens, cache_miss_tokens;
            - response_time: время запроса в секундах;
            - finish_reason: причина завершения (stop, length и т.п.);
            - error: None либо текст ошибки;
            - timestamp: локальное время ответа (строка).

            При ошибке запроса поле tokens = None, а счётчики сессии
            не изменяются.
        """
        if not user_input or not user_input.strip():
            return self._error_result("Пустое сообщение.")

        with self._lock:
            # Сжатие старой части диалога выполняется ДО добавления нового
            # сообщения, чтобы границы блока всегда совпадали с парами
            # «вопрос-ответ» и в запрос уходили ровно последние N сообщений.
            summarized = self._maybe_summarize()

            self._append_message({"role": "user", "content": user_input.strip()})
            # Снимок того, что реально уйдёт в API (system + конспект +
            # непокрытая история) — используется для оценки токенов, если
            # API не вернул usage.
            request_messages = self._build_context_messages()

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
                # При включённом сжатии «сырая» история не урезается:
                # единственный способ «забыть» старое — конспект.
                if not self._summary_active():
                    self._trim_history()
                # Учитываем токены обмена (точные из API или оценку)
                token_info = self._update_token_stats(
                    payload, request_messages,
                    summarized_this_turn=summarized,
                )
                # Сохраняем контекст после каждого успешного обмена
                self._persist()

                return {
                    "text": payload["content"],
                    "model": payload.get("model", self.model),
                    "tokens": token_info,
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
        """Очищает историю диалога, оставляя только системный промпт.

        Сохранённый на диске контекст также очищается (остаётся только
        системный промпт и параметры агента). Накопленная статистика
        токенов сессии при этом сбрасывается — и в памяти, и на диске.
        """
        with self._lock:
            self._init_history()
            self._session_stats = self._default_session_stats()
            self._last_exchange = self._default_last_exchange()
            # Конспект относится к удалённой истории — очищаем и его
            self._summary = None
            self._last_summary_error = None
            # Перезаписываем хранилище согласованным состоянием
            self._persist()

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
                # Суммаризатор следует за моделью диалога, если для сжатия
                # не задана отдельная модель
                if self._summarizer is not None and not str(
                    (self._summary_settings or {}).get("model") or ""
                ):
                    self._summarizer.model = self.model
            if "system_prompt" in kwargs:
                # Смена системного промпта сбрасывает историю диалога
                self.system_prompt = system_prompt
                self._init_history()
                # ...и вместе с ней — конспект удалённой истории
                self._summary = None
                self._last_summary_error = None
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
                # При включённом сжатии «сырая» история не урезается:
                # ограничение действует только на то, что уходит в API
                if not self._summary_active():
                    self._trim_history()

            # Персист: смена системного промпта перезаписывает диалог
            # на диске, смена остальных параметров обновляет параметры.
            if self._persistence_active():
                if "system_prompt" in kwargs:
                    self._persist()
                elif any(k in kwargs for k in
                         ("model", "temperature", "max_tokens")):
                    self._persist_params()

    def close(self) -> None:
        """Закрывает агента: финальный flush контекста и хранилища.

        Метод идемпотентен и безопасен для повторного вызова. Финальный
        flush выполняется только если на диске уже есть сохранённый
        контекст или в памяти есть диалог — чтобы пустой запуск не
        создавал файл БД «просто так».
        """
        with self._lock:
            if self._closed:
                return
            # Сначала финальный flush, только потом закрытие хранилища:
            # установка _closed раньше отключила бы персист.
            if self._persistence_active() and self._storage is not None:
                has_dialog = len(self._message_history) > 1
                if self._storage.exists or has_dialog:
                    self._persist()
                self._storage.close()
            self._closed = True

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

    def _build_context_messages(self) -> List[Message]:
        """Собирает сообщения, которые реально уйдут в API.

        Схема: [system-промпт] → [system: конспект] → [непокрытые сообщения].
        Покрытая конспектом часть истории в запрос не попадает; «сырая»
        история при этом сохраняется в памяти и в БД.

        Returns:
            Список сообщений для API.
        """
        system = self._message_history[0] if self._message_history else None
        dialog = self._message_history[1:]
        messages: List[Message] = [system] if system else []

        start = 0
        summary_text = ""
        if self._summary:
            summary_text = str(self._summary.get("text") or "")
        if summary_text:
            messages.append({
                "role": "system",
                "content": (
                    "Краткое содержание предыдущего диалога:\n" + summary_text
                ),
            })
            try:
                start = int(self._summary.get("covered", 0))
            except (TypeError, ValueError):
                start = 0
            start = max(0, min(start, len(dialog)))

        tail = dialog[start:]
        # Аварийное ограничение: если сжатие недоступно (ошибки) и история
        # слишком велика, ограничиваем только отправляемую часть.
        if len(tail) > self._max_history:
            logger.warning(
                "Контекст превышает MAX_HISTORY_MESSAGES=%d: в API уйдут "
                "только последние %d сообщений (сжатие недоступно).",
                self._max_history, self._max_history,
            )
            tail = tail[-self._max_history:]

        messages.extend(tail)
        return messages

    def _build_request(self) -> Dict[str, Any]:
        """Формирует структуру запроса к API.

        Returns:
            Словарь с параметрами chat.completions.create: модель,
            контекст (system + конспект + непокрытая часть истории),
            температура, max_tokens.
        """
        return {
            "model": self.model,
            "messages": self._build_context_messages(),
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
                "estimated": False,
                # Поля кэша промпта (если API их возвращает)
                "cache_hit_tokens": getattr(usage, "prompt_cache_hit_tokens", None),
                "cache_miss_tokens": getattr(usage, "prompt_cache_miss_tokens", None),
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
                # mock-значения синтетические — помечаем как оценочные
                "estimated": True,
                "cache_hit_tokens": None,
                "cache_miss_tokens": None,
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
    # Персистентность контекста
    # ------------------------------------------------------------------

    def _persistence_active(self) -> bool:
        """True, если хранилище подключено и агент не в mock-режиме."""
        return (
            self._storage is not None
            and not self._mock_mode
            and not self._closed
        )

    def _current_params(self) -> Dict[str, Any]:
        """Текущие параметры агента для сохранения в хранилище."""
        return {
            "model": self.model,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _persist(self) -> None:
        """Сохраняет историю, параметры, статистику и конспект.

        Ошибки хранилища не пробрасываются: агент продолжает работу в памяти.
        """
        if not self._persistence_active() or self._storage is None:
            return
        try:
            self._storage.save_state(
                self._message_history,
                self._current_params(),
                stats=dict(self._session_stats),
                # {} — явно стереть конспект, None — не трогать (см. storage)
                summary=dict(self._summary) if self._summary else {},
            )
        except Exception as exc:
            logger.error("Не удалось сохранить контекст: %s", exc)

    def _persist_params(self) -> None:
        """Сохраняет только параметры агента (ошибки не пробрасываются)."""
        if not self._persistence_active() or self._storage is None:
            return
        try:
            self._storage.save_params(self._current_params())
        except Exception as exc:
            logger.error("Не удалось сохранить параметры: %s", exc)

    def _load_context(self) -> None:
        """Загружает сохранённый контекст (историю и параметры) из хранилища.

        Вызывается в конце __init__ (вне mock-режима). Восстановленные
        параметры проходят валидацию: невалидные значения игнорируются
        с предупреждением в логе. Если хранилище отсутствует/пусто/бито —
        агент остаётся в исходном состоянии.
        """
        if self._storage is None:
            return
        try:
            state = self._storage.load_state()
        except Exception as exc:  # хранилище не должно ронять запуск
            logger.warning("Не удалось прочитать сохранённый контекст: %s", exc)
            return
        if not state:
            return

        params = state.get("params") or {}
        messages = state.get("messages") or []

        # 1. Восстанавливаем параметры с валидацией и мягким fallback
        for key, validator in (
            ("model", utils.validate_model),
            ("temperature", utils.validate_temperature),
            ("max_tokens", utils.validate_max_tokens),
            ("system_prompt", utils.validate_system_prompt),
        ):
            if key not in params:
                continue
            try:
                setattr(self, key, validator(params[key]))
            except (AgentConfigError, TypeError, ValueError) as exc:
                logger.warning(
                    "Сохранённый параметр %r=%r невалиден (%s) — "
                    "оставлено текущее значение.",
                    key, params.get(key), exc,
                )

        # 2. Если параметра system_prompt нет — берём из первого сообщения
        if "system_prompt" not in params and messages:
            first = messages[0]
            if first.get("role") == "system" and first.get("content"):
                try:
                    self.system_prompt = utils.validate_system_prompt(
                        str(first["content"])
                    )
                except AgentConfigError:
                    pass

        # 3. Пересоздаём историю: system из параметра, диалог — из БД
        self._init_history()
        restored = 0
        for message in messages:
            role = message.get("role")
            if role in ("user", "assistant"):
                self._message_history.append(
                    {"role": role, "content": str(message.get("content", ""))}
                )
                restored += 1
        self._trim_history()

        # 4. Восстанавливаем накопленную статистику токенов
        self._restore_session_stats(state.get("stats"))

        # 5. Восстанавливаем конспект диалога (если он сохранён)
        self._restore_summary(state.get("summary"))

        if restored:
            logger.info(
                "Контекст восстановлен: %d сообщений, модель=%s.",
                restored, self.model,
            )

    # ------------------------------------------------------------------
    # Учёт токенов и стоимости
    # ------------------------------------------------------------------

    @staticmethod
    def _default_session_stats() -> Dict[str, Any]:
        """Пустая накопленная статистика сессии."""
        return {
            # Точные значения из usage API
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
            "exact_requests": 0,
            # Оценочные значения (usage недоступен)
            "estimated_prompt_tokens": 0,
            "estimated_completion_tokens": 0,
            "estimated_total_tokens": 0,
            # Стоимость (USD): точная и по оценкам
            "cost_usd": 0.0,
            "estimated_cost_usd": 0.0,
            # Служебные запросы сжатия контекста (учитываются отдельно)
            "summary_requests": 0,
            "summary_prompt_tokens": 0,
            "summary_completion_tokens": 0,
            "summary_estimated_tokens": 0,
            "summary_cost_usd": 0.0,
        }

    @staticmethod
    def _default_last_exchange() -> Dict[str, Any]:
        """Пустая статистика последнего обмена."""
        return {
            "request_tokens": None,
            "response_tokens": None,
            "total_tokens": None,
            "estimated": False,
            "cost_usd": None,
            "cache_hit_tokens": None,
            "cache_miss_tokens": None,
        }

    def _restore_session_stats(self, stats: Optional[Dict[str, Any]]) -> None:
        """Восстанавливает статистику сессии из хранилища (с валидацией).

        Неизвестные ключи и значения неверного типа игнорируются, поэтому
        старая/повреждённая статистика не ломает запуск.
        """
        if not isinstance(stats, dict):
            return
        restored = self._default_session_stats()
        for key, default in restored.items():
            value = stats.get(key)
            if isinstance(default, float):
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    restored[key] = float(value)
            elif isinstance(value, int) and not isinstance(value, bool):
                restored[key] = value
        self._session_stats = restored

    def _resolve_prices(self, model: Optional[str] = None) -> tuple:
        """Возвращает (цена_входа, цена_выхода) за 1M токенов для модели.

        Приоритет: цена по модели → общая цена → None (не задана).

        Args:
            model: модель, для которой нужны цены (None — текущая модель агента).
        """
        cfg = self._prices or {}
        default = cfg.get("default") or {}
        target = model or self.model
        per_model = (cfg.get("models") or {}).get(target) or {}

        def _norm(value: Any) -> Optional[float]:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        price_in = per_model.get("input", default.get("input"))
        price_out = per_model.get("output", default.get("output"))
        return _norm(price_in), _norm(price_out)

    def _prices_set(self, model: Optional[str] = None) -> bool:
        """True, если задана хотя бы одна ненулевая цена."""
        price_in, price_out = self._resolve_prices(model)
        return bool(price_in) or bool(price_out)

    def _compute_cost(self, prompt_tokens: int, completion_tokens: int,
                      model: Optional[str] = None) -> Optional[float]:
        """Считает стоимость обмена в USD (None, если цены не заданы)."""
        price_in, price_out = self._resolve_prices(model)
        if not price_in and not price_out:
            return None
        try:
            cost = (
                prompt_tokens / 1_000_000 * (price_in or 0.0)
                + completion_tokens / 1_000_000 * (price_out or 0.0)
            )
        except (TypeError, ValueError):
            return None
        return round(cost, 6)

    def _update_token_stats(self, payload: Dict[str, Any],
                            request_messages: List[Message],
                            summarized_this_turn: bool = False) -> Dict[str, Any]:
        """Обновляет статистику по завершённому обмену и возвращает её срез.

        Точные значения берутся из usage API; если usage нет (или значения
        помечены как оценочные, например в mock-режиме) — используется
        локальная оценка счётчиком токенов.

        Args:
            payload: результат _handle_response/_mock_response.
            request_messages: снимок истории, отправленной в API.
            summarized_this_turn: был ли в этом ходе обновлён конспект.

        Returns:
            Словарь статистики обмена для поля "tokens" результата.
        """
        raw = payload.get("tokens") or {}
        api_prompt = raw.get("prompt_tokens")
        api_completion = raw.get("completion_tokens")
        api_total = raw.get("total_tokens")
        marked_estimated = bool(raw.get("estimated", False))

        estimated = (
            marked_estimated
            or api_prompt is None
            or api_completion is None
        )

        if estimated:
            # Локальная оценка: снимок запроса + текст ответа
            prompt_tokens = self._counter.count_messages(request_messages)
            completion_tokens = self._counter.count_text(
                payload.get("content") or ""
            )
            total_tokens = prompt_tokens + completion_tokens
            self._session_stats["estimated_prompt_tokens"] += prompt_tokens
            self._session_stats["estimated_completion_tokens"] += completion_tokens
            self._session_stats["estimated_total_tokens"] += total_tokens
            cost = self._compute_cost(prompt_tokens, completion_tokens)
            if cost is not None:
                self._session_stats["estimated_cost_usd"] = round(
                    self._session_stats["estimated_cost_usd"] + cost, 6
                )
        else:
            prompt_tokens = int(api_prompt)
            completion_tokens = int(api_completion)
            if api_total is None:
                total_tokens = prompt_tokens + completion_tokens
            else:
                total_tokens = int(api_total)
            self._session_stats["prompt_tokens"] += prompt_tokens
            self._session_stats["completion_tokens"] += completion_tokens
            self._session_stats["total_tokens"] += total_tokens
            cost = self._compute_cost(prompt_tokens, completion_tokens)
            if cost is not None:
                self._session_stats["cost_usd"] = round(
                    self._session_stats["cost_usd"] + cost, 6
                )

        self._session_stats["requests"] += 1
        if not estimated:
            self._session_stats["exact_requests"] += 1

        self._last_exchange = {
            "request_tokens": prompt_tokens,
            "response_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated": estimated,
            "cost_usd": cost,
            "cache_hit_tokens": raw.get("cache_hit_tokens"),
            "cache_miss_tokens": raw.get("cache_miss_tokens"),
        }

        # Контекст (system + конспект + непокрытая история) считаем один раз
        context_messages = self._build_context_messages()
        context_tokens = self._counter.count_messages(context_messages)

        return {
            # «Старые» ключи: точные значения API, либо оценка (см. estimated)
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            # Новые ключи
            "estimated": estimated,
            "context_tokens": context_tokens,
            "context_messages": max(0, len(context_messages) - 1),
            "summary_covered": self._summary_covered(),
            "summarized_this_turn": bool(summarized_this_turn),
            "session_total_tokens": self._session_total_with_estimates(),
            "cost_usd": cost,
            "session_cost_usd": self._session_cost_total(),
            "cache_hit_tokens": raw.get("cache_hit_tokens"),
            "cache_miss_tokens": raw.get("cache_miss_tokens"),
        }

    def _session_total_with_estimates(self) -> int:
        """Суммарные токены сессии: точные + оценочные."""
        return int(
            self._session_stats.get("total_tokens", 0)
            + self._session_stats.get("estimated_total_tokens", 0)
        )

    def _session_cost_total(self) -> Optional[float]:
        """Суммарная стоимость сессии (точная + оценочная) или None."""
        if not self._prices_set():
            return None
        return round(
            float(self._session_stats.get("cost_usd", 0.0))
            + float(self._session_stats.get("estimated_cost_usd", 0.0)),
            6,
        )

    def count_tokens(self, text: str) -> int:
        """Оценивает число токенов в произвольном тексте.

        Args:
            text: текст для оценки.

        Returns:
            Оценка токенов (0 для пустой строки).
        """
        try:
            return int(self._counter.count_text(str(text or "")))
        except Exception as exc:  # счётчик не должен ломать вызывающий код
            logger.warning("Ошибка счётчика токенов: %s", exc)
            return 0

    def count_history_tokens(self) -> int:
        """Оценивает токены всего текущего контекста диалога (включая system).

        Returns:
            Оценка токенов истории, которая уйдёт в следующий запрос.
        """
        with self._lock:
            try:
                return int(self._counter.count_messages(self._message_history))
            except Exception as exc:
                logger.warning("Ошибка счётчика токенов: %s", exc)
                return 0

    def get_token_stats(self) -> Dict[str, Any]:
        """Возвращает полную статистику токенов для UI.

        Returns:
            Словарь:
            - request: {"tokens", "estimated"} — последний запрос;
            - response: {"tokens", "estimated"} — последний ответ;
            - context: {"tokens", "messages", "estimated"} — текущий контекст;
            - session: накопленные суммы (точные и оценочные), стоимость,
              признак prices_set, total_with_estimates;
            - counter: имя активного счётчика ("tiktoken:..." | "heuristic").
        """
        with self._lock:
            context_messages = self._build_context_messages()
            context_tokens = self.count_context_tokens()
            summary_text = str((self._summary or {}).get("text") or "")
            session = dict(self._session_stats)
            session["requests"] = int(session.get("requests", 0))
            session["total_with_estimates"] = self._session_total_with_estimates()
            session["prices_set"] = self._prices_set()
            session["cost_usd"] = (
                round(float(session.get("cost_usd", 0.0)), 6)
                if session["prices_set"] else None
            )
            session["estimated_cost_usd"] = (
                round(float(session.get("estimated_cost_usd", 0.0)), 6)
                if session["prices_set"] else None
            )
            session["summary_cost_usd"] = (
                round(float(session.get("summary_cost_usd", 0.0)), 6)
                if session["prices_set"] else None
            )
            return {
                "request": {
                    "tokens": self._last_exchange.get("request_tokens"),
                    "estimated": bool(self._last_exchange.get("estimated", False)),
                },
                "response": {
                    "tokens": self._last_exchange.get("response_tokens"),
                    "estimated": bool(self._last_exchange.get("estimated", False)),
                },
                "context": {
                    "tokens": context_tokens,
                    "messages": max(0, len(context_messages) - 1),
                    "messages_raw": max(0, len(self._message_history) - 1),
                    "messages_sent": len(context_messages),
                    "summary_tokens": self.count_tokens(summary_text) if summary_text else 0,
                    "summary_covered": self._summary_covered(),
                    "summary_active": bool(summary_text),
                    "estimated": True,  # контекст всегда оценивается локально
                },
                "summary": {
                    "active": bool(summary_text),
                    "enabled": self._summary_active(),
                    "text": summary_text,
                    "covered": self._summary_covered(),
                    "tokens": self.count_tokens(summary_text) if summary_text else 0,
                    "updated_at": (self._summary or {}).get("updated_at"),
                    "model": (self._summary or {}).get("model"),
                    "error": self._last_summary_error,
                },
                "session": session,
                "counter": self._counter.name,
            }

    @property
    def session_stats(self) -> Dict[str, Any]:
        """Копия накопленной статистики сессии (только для чтения)."""
        with self._lock:
            return dict(self._session_stats)

    # ------------------------------------------------------------------
    # Управление контекстом (сжатие в конспект)
    # ------------------------------------------------------------------

    def _create_summarizer(self) -> ContextSummarizer:
        """Создаёт суммаризатор по текущим настройкам сжатия."""
        settings = self._summary_settings or {}
        return ContextSummarizer(
            prompt=str(settings.get("prompt") or "") or None,
            temperature=float(settings.get("temperature", 0.2)),
            max_tokens=int(settings.get("max_tokens", 400)),
            model=str(settings.get("model") or self.model),
        )

    def _summary_active(self) -> bool:
        """True, если сжатие контекста включено и доступно (не mock)."""
        return bool((self._summary_settings or {}).get("enabled")) \
            and not self._mock_mode

    def _dialog_messages(self) -> List[Message]:
        """Сообщения диалога без системного промпта."""
        return self._message_history[1:]

    def _summary_covered(self) -> int:
        """Сколько сообщений диалога покрыто конспектом."""
        if not self._summary:
            return 0
        try:
            return max(0, int(self._summary.get("covered", 0)))
        except (TypeError, ValueError):
            return 0

    def _maybe_summarize(self) -> bool:
        """Сжимает историю, если сработал триггер (по числу сообщений).

        Returns:
            True, если конспект был обновлён в этом вызове.
        """
        if not self._summary_active() or self._summarizer is None:
            return False
        settings = self._summary_settings or {}
        dialog_length = len(self._dialog_messages())
        keep_last = int(settings.get("keep_last", 10))
        every_n = int(settings.get("every_n", 10))
        if not ContextSummarizer.should_summarize(
            dialog_length, self._summary_covered(), keep_last, every_n
        ):
            return False
        return bool(self._run_summarization(force=False).get("updated"))

    def _call_service_api(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Служебный запрос к API (сжатие контекста).

        Не учитывается в статистике диалога и не порождает рекурсивного
        сжатия. В тестах и sim-режиме бенчмарка подменяется целиком.

        Args:
            request: параметры для chat.completions.create.

        Returns:
            Разобранный ответ (см. _handle_response).

        Raises:
            ApiError: при любой ошибке обращения к API.
        """
        if self._client is None:
            raise ApiError("Клиент API не инициализирован (mock-режим?).")
        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            raise ApiError(humanize_api_error(exc), original=exc)
        return self._handle_response(response)

    def _run_summarization(self, force: bool = False) -> Dict[str, Any]:
        """Сжимает всё, кроме последних keep_last сообщений.

        При ошибке служебного запроса **ничего не меняется**: старый конспект
        и его границы остаются, ошибка запоминается для UI, а повторная
        попытка произойдёт при следующем триггере.

        Args:
            force: True — вызвано пользователем («Обновить» в окне «Контекст»).

        Returns:
            Словарь {"updated", "covered", "error"}.
        """
        if not self._summary_active():
            return {
                "updated": False,
                "covered": self._summary_covered(),
                "error": "Сжатие контекста отключено.",
            }
        if self._summarizer is None:
            return {
                "updated": False,
                "covered": self._summary_covered(),
                "error": "Суммаризатор недоступен.",
            }

        settings = self._summary_settings or {}
        keep_last = int(settings.get("keep_last", 10))
        max_tokens = int(settings.get("max_tokens", 400))
        dialog = self._dialog_messages()
        start, end = ContextSummarizer.chunk_bounds(
            len(dialog), self._summary_covered(), keep_last
        )
        if start >= end:
            return {"updated": False, "covered": start, "error": None}

        chunk = dialog[start:end]
        previous = str((self._summary or {}).get("text") or "") or None
        result = self._summarizer.summarize(
            previous, chunk, self._call_service_api
        )
        if result.get("error"):
            self._last_summary_error = str(result["error"])
            logger.warning("Сжатие контекста не удалось: %s",
                           self._last_summary_error)
            return {
                "updated": False,
                "covered": start,
                "error": self._last_summary_error,
            }

        text = ContextSummarizer.truncate(
            str(result.get("text") or ""), max_tokens, self._counter
        )
        model_used = (result.get("model")
                      or (self._summarizer.model if self._summarizer else "")
                      or self.model)

        # Служебные токены: отдельные счётчики, статистика диалога не меняется
        estimated = bool(result.get("estimated")) \
            or result.get("prompt_tokens") is None \
            or result.get("completion_tokens") is None
        if estimated:
            request = self._summarizer.build_request(previous, chunk)
            prompt_tokens = self._counter.count_messages(request["messages"])
            completion_tokens = self.count_tokens(text)
            self._session_stats["summary_estimated_tokens"] += (
                prompt_tokens + completion_tokens
            )
        else:
            prompt_tokens = int(result.get("prompt_tokens") or 0)
            completion_tokens = int(result.get("completion_tokens") or 0)
            self._session_stats["summary_prompt_tokens"] += prompt_tokens
            self._session_stats["summary_completion_tokens"] += completion_tokens

        cost = self._compute_cost(prompt_tokens, completion_tokens,
                                  model=model_used)
        if cost is not None:
            self._session_stats["summary_cost_usd"] = round(
                float(self._session_stats.get("summary_cost_usd", 0.0)) + cost, 6
            )

        self._summary = {
            "text": text,
            "covered": end,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "model": model_used,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "estimated": estimated,
        }
        self._session_stats["summary_requests"] += 1
        self._last_summary_error = None
        logger.info(
            "Контекст сжат: покрыто %d сообщений (с %d), конспект ≈%d токенов.",
            end, start, self.count_tokens(text),
        )
        self._persist()
        return {"updated": True, "covered": end, "error": None}

    def _restore_summary(self, raw: Optional[Dict[str, Any]]) -> None:
        """Восстанавливает конспект из хранилища (с проверкой согласованности).

        Некорректный или противоречащий истории конспект игнорируется —
        сжатие просто начнётся заново.
        """
        if not isinstance(raw, dict):
            return
        text = str(raw.get("text") or "").strip()
        if not text:
            return
        try:
            covered = int(raw.get("covered", 0))
        except (TypeError, ValueError):
            covered = 0
        dialog_length = len(self._dialog_messages())
        if covered < 0 or covered > dialog_length:
            logger.warning(
                "Сохранённый конспект противоречит истории (covered=%s из %d) "
                "— игнорирую.", raw.get("covered"), dialog_length,
            )
            return
        self._summary = {
            "text": text,
            "covered": covered,
            "updated_at": raw.get("updated_at"),
            "model": raw.get("model"),
            "prompt_tokens": raw.get("prompt_tokens"),
            "completion_tokens": raw.get("completion_tokens"),
        }
        logger.info("Конспект восстановлен: покрыто %d сообщений.", covered)

    def get_summary(self) -> Dict[str, Any]:
        """Возвращает состояние конспекта для UI и тестов.

        Returns:
            Словарь {"text", "covered", "updated_at", "model", "tokens",
            "active", "enabled", "error"}.
        """
        with self._lock:
            text = str((self._summary or {}).get("text") or "")
            return {
                "text": text,
                "covered": self._summary_covered(),
                "updated_at": (self._summary or {}).get("updated_at"),
                "model": (self._summary or {}).get("model"),
                "tokens": self.count_tokens(text) if text else 0,
                "active": bool(text),
                "enabled": self._summary_active(),
                "error": self._last_summary_error,
            }

    def summarize_now(self, force: bool = False) -> Dict[str, Any]:
        """Принудительно сжимает историю (кнопка «Обновить» в окне «Контекст»).

        Args:
            force: оставлен для явности вызова (сжатие всегда выполняется
                для всего блока, кроме последних keep_last сообщений).

        Returns:
            Словарь {"updated", "covered", "error"}.
        """
        with self._lock:
            if self._mock_mode:
                return {
                    "updated": False,
                    "covered": 0,
                    "error": "Сжатие недоступно в mock-режиме.",
                }
            if not (self._summary_settings or {}).get("enabled"):
                return {
                    "updated": False,
                    "covered": self._summary_covered(),
                    "error": "Сжатие контекста отключено "
                             "(SUMMARY_ENABLED=false).",
                }
            return self._run_summarization(force=force)

    def clear_summary(self) -> None:
        """Очищает конспект (история диалога при этом не трогается)."""
        with self._lock:
            self._summary = None
            self._last_summary_error = None
            self._persist()

    def count_context_tokens(self) -> int:
        """Оценивает токены того, что реально уйдёт в следующий запрос.

        Returns:
            Оценка токенов (system + конспект + непокрытая часть истории).
        """
        with self._lock:
            try:
                messages = self._build_context_messages()
                return int(self._counter.count_messages(messages))
            except Exception as exc:
                logger.warning("Ошибка счётчика токенов: %s", exc)
                return 0

    @property
    def summary_settings(self) -> Dict[str, Any]:
        """Копия настроек сжатия контекста (только для чтения)."""
        with self._lock:
            return dict(self._summary_settings or {})

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
