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
- управляет контекстом: стратегия (``sliding_window`` | ``sticky_facts``)
  решает, что уходит в API; режим ``branching`` позволяет вести несколько
  независимых веток диалога от одной точки (checkpoint/развилка);
- поддерживает блок фактов (key → value): факты обновляются после каждого хода
  и отправляются отдельным system-сообщением вместе с последними N сообщениями;
- поддерживает персонализацию через профиль пользователя (ProfileRepository):
  системный промпт собирается из активного профиля, настройки агента
  (модель/temperature/max_tokens) привязаны к профилю, история диалога
  изолирована по профилям (messages.profile_id) и переключается вместе
  с профилем без потери;
- поддерживает инварианты (InvariantRepository): жёсткие правила
  «что можно делать» (global + активный профиль) попадают в system-промпт
  перед блоком профиля; после каждого ответа работает пост-проверка
  (InvariantValidator, второй LLM-вызов) — при обнаружении нарушения ответ
  перегенерируется с явным напоминанием про инвариант (лимит попыток из
  конфига), отказы и перегенерации пишутся в лог нарушений;
- сохраняет совместимость со старым режимом сжатия в конспект (summary):
  он работает только если настройки summary переданы явно, а стратегия не задана.
"""

from __future__ import annotations

import asyncio
import json
import random
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from . import utils
from .branches import BranchManager
from .facts import FactsStore
from .memory import (
    LONG_HEADER,
    LONG_TERM,
    LAYERS,
    LAYER_TITLES,
    PINNED_HEADER,
    SHORT_TERM,
    WORKING,
    WORKING_HEADER,
    LongTermStore,
    MemoryClassifier,
    MemoryError,
    MemoryManager,
    WorkingMemory,
    normalize_layer,
)
from .models import ProfileAgentSettings, UserProfile
from .invariant_models import Invariant, InvariantCategory
from .invariant_prompt_builder import build_invariants_block
from .invariant_repository import InvariantRepository
from .invariant_validator import Conflict, InvariantValidator
from .llm_providers import (
    BaseLLMProvider,
    LLMProviderFactory,
    ProviderUnavailableError,
)
from .profile_prompt_builder import build_system_prompt
from .profile_repository import ProfileRepository
from .storage import ContextStore
from .task_fsm import TASK_RETRY_LIMIT, TaskFSM, get_allowed_transitions
from .task_models import (
    ACTIVE_STAGES,
    WORKING_STAGES,
    ActiveTaskExistsError,
    InvalidTransitionError,
    Task,
    TaskMode,
    TaskStage,
    TaskStep,
)
from .task_repository import TaskRepository
from .task_response_parser import TaskResponseParser
from .strategies import (
    AVAILABLE_STRATEGIES,
    BRANCHING,
    SLIDING_WINDOW,
    STICKY_FACTS,
    ContextStrategy,
    create_strategy,
    normalize_name,
)
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

# Блок правил инструментов: добавляется в конец system-сообщения запроса,
# когда у агента есть tools напоминаний/новостей. Без него LLM может
# ограничиться текстом сводки в чате и не вызвать save_to_file — тогда
# пользователь не получит файл (и кнопку «Открыть файл»).
_TOOL_USAGE_RULES = (
    "[ПРАВИЛА РАБОТЫ С ИНСТРУМЕНТАМИ]\n"
    "Тебе доступны инструменты напоминаний, новостей, геолокации и погоды "
    "— вызывай их сразу, без переспрашивания.\n"
    "- Напоминание: add_reminder(text, interval_minutes, repeat).\n"
    "- Новости: search_news(topic, limit) — поиск; summarize_text(text) — "
    "краткая сводка; save_to_file(content) — сохранение текста в файл.\n"
    "- Гео и погода: get_user_city() — город пользователя; "
    "get_weather(city) — текущая погода в городе.\n"
    "Сценарий «найди новости / сделай сводку (и сохрани)»: search_news → "
    "summarize_text → save_to_file.\n"
    "Сценарий «какая погода»: get_user_city() → get_weather(city).\n"
    "ВАЖНО: полный текст сводки сохраняй в ФАЙЛ через save_to_file; "
    "в чате после этого отвечай кратко — что сделано, путь к файлу и "
    "1–2 предложения сути, не дублируя весь текст сводки.\n"
    "Если инструмент вернул ошибку — кратко сообщи о ней и продолжи без "
    "этого шага или предложи альтернативу."
)


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
        strategy: Optional[str] = None,
        strategy_settings: Optional[Dict[str, Any]] = None,
        branch_manager: Optional[BranchManager] = None,
        memory_settings: Optional[Dict[str, Any]] = None,
        memory_manager: Optional[MemoryManager] = None,
        long_memory: Optional[LongTermStore] = None,
        profile_repository: Optional[ProfileRepository] = None,
        profile_id: Optional[int] = None,
        task_repository: Optional["TaskRepository"] = None,
        invariant_repository: Optional[InvariantRepository] = None,
        invariant_validator: Optional[InvariantValidator] = None,
        task_max_retries: Optional[int] = None,
        reminder_client: Optional[Any] = None,
        news_client: Optional[Any] = None,
        geo_client: Optional[Any] = None,
        weather_client: Optional[Any] = None,
        max_tool_rounds: Optional[int] = None,
        provider: Optional[BaseLLMProvider] = None,
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
            strategy: имя стратегии управления контекстом (``sliding_window``,
                ``sticky_facts``, ``branching``); None вместе с
                ``strategy_settings`` означает режим обратной совместимости
                (стратегии не используются, работает summary).
            strategy_settings: настройки стратегий (см.
                config.get_strategy_settings): окно, факты, ветки.
            branch_manager: готовый менеджер веток (для тестов); None — создать.
            memory_settings: настройки модели памяти (см.
                config.get_memory_settings); None — модель памяти выключена
                (поведение ровно как раньше).
            memory_manager: готовый менеджер памяти (для тестов); None — создать.
            long_memory: готовое хранилище долговременной памяти (для тестов);
                None — создать из настроек (не создаётся в mock-режиме).
            profile_repository: репозиторий профилей (персонализация); None —
                агент работает без профилей, поведение как раньше.
            profile_id: профиль для активации; None — активный профиль из
                репозитория (is_active=1, обычно Default после миграции).
            task_repository: репозиторий задач (подсистема FSM); None —
                создаётся лениво при первой операции с задачами, если
                подключено хранилище контекста.
            invariant_repository: репозиторий инвариантов (SQLite); None —
                создаётся лениво при первом обращении, если подключено
                хранилище контекста. Без репозитория инварианты выключены
                (промпт и пост-проверка работают как раньше).
            invariant_validator: валидатор пост-проверки ответов; None —
                создаётся с настройками по умолчанию (ENABLE_POST_CHECK
                и лимиты обычно передаются из конфига через main.py).
            task_max_retries: лимит откатов validation → execution для FSM
                задач (конфиг TASKS_MAX_RETRIES); None — ``TASK_RETRY_LIMIT``.
            reminder_client: HTTP MCP-клиент напоминаний
                (:class:`src.reminder_client.ReminderMCPClient`); None —
                напоминания не участвуют в tool calling.
            news_client: HTTP MCP-клиент новостей
                (:class:`src.reminder_client.ReminderMCPClient` на
                news_server.py); None — инструменты новостей выключены.
            geo_client: HTTP MCP-клиент геолокации
                (:class:`src.reminder_client.ReminderMCPClient` на
                geo_server.py); None — инструмент get_user_city выключен.
            weather_client: HTTP MCP-клиент погоды
                (:class:`src.reminder_client.ReminderMCPClient` на
                weather_server.py); None — инструмент get_weather выключен.
            max_tool_rounds: лимит циклов «LLM → tool_calls → результат →
                LLM» за один ход (agentic loop); None — 5.
            provider: провайдер LLM (BaseLLMProvider, ТЗ ai_1_25); None —
                облако DeepSeek (клиент создаётся как раньше). Провайдер
                поставляет клиента, модель и base_url; переключение на
                лету — :meth:`switch_provider`.

        Raises:
            AgentConfigError: если какой-либо параметр некорректен.
        """
        # Провайдер LLM: источник клиента/модели/base_url. Задаётся ДО
        # профиля, чтобы guard в _apply_profile_settings не дал настройкам
        # профиля подменить модель чужого провайдера.
        self._provider: Optional[BaseLLMProvider] = provider
        self._provider_base_url: str = (
            provider.base_url if provider is not None
            else DeepSeekAgent._base_url
        )
        # MCP-инструменты включены; switch_provider отключает их, если
        # модель провайдера не умеет tool calling
        self._provider_tools_enabled: bool = True

        # Настройки оптимизации локальной модели (ТЗ ai_1_28): глобальные
        # (вне профилей), применяются к Ollama-провайдерам — текущему и
        # всем созданным позже через switch_provider
        self._ollama_optimization: Dict[str, Any] = {}

        # Заполняем значения по умолчанию
        self.model = model or "deepseek-v4-flash"
        if self._provider is not None:
            self.model = self._provider.model
        self.system_prompt = system_prompt or (
            "Ты полезный AI-ассистент. Отвечай четко и по делу."
        )
        self.temperature = temperature if temperature is not None else 0.7
        self.max_tokens = max_tokens if max_tokens is not None else 1000

        # Хранилище и подсистемы данных, нужные уже при сборке первого
        # system-промпта (инварианты читаются из БД наравне с профилем).
        self._mock_mode: bool = bool(mock_mode)
        self._storage: Optional[ContextStore] = storage
        self._invariant_repo: Optional[InvariantRepository] = \
            invariant_repository
        self._invariant_validator: InvariantValidator = (
            invariant_validator if invariant_validator is not None
            else InvariantValidator()
        )

        # Персонализация: репозиторий профилей, активный профиль и его
        # настройки. Параметры из конструктора перекрываются настройками
        # профиля (model/temperature/max_tokens из profile_agent_settings),
        # а системный промпт собирается из профиля (build_system_prompt).
        # Кэш историй позволяет переключать профили без потери диалогов.
        self._profile_repo: Optional[ProfileRepository] = profile_repository
        self._profile: Optional[UserProfile] = None
        self._profile_settings: Optional[ProfileAgentSettings] = None
        self._profile_histories: Dict[int, Dict[str, Any]] = {}
        if self._profile_repo is not None:
            profile = (
                self._profile_repo.get_by_id(int(profile_id))
                if profile_id is not None
                else self._profile_repo.get_active()
            )
            if profile_id is not None and profile is None:
                raise AgentConfigError(
                    f"Профиль id={profile_id} не найден в репозитории."
                )
            if profile is not None:
                self._profile = profile
                self._profile_settings = self._profile_repo.get_settings(
                    profile.id
                )
                self._apply_profile_settings()
                self.system_prompt = build_system_prompt(
                    self._profile, self._profile_settings,
                    invariants_block=self._build_invariants_block(),
                )

        # Проверяем параметры ДО создания клиента
        self._validate_parameters()

        self._api_key: str = api_key or ""
        self._max_history: int = int(max_history_messages)
        self._lock = threading.RLock()
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

        # Стратегии управления контекстом, факты и ветки диалога.
        # Стратегия включается только если передана явно (или переданы её
        # настройки): без них агент работает в режиме обратной совместимости
        # (summary + ограничение «сырой» истории), как раньше.
        self._strategy_settings: Dict[str, Any] = dict(strategy_settings or {})
        self._strategy_name: Optional[str] = None   # значение переключателя
        self._sub_strategy_name: str = SLIDING_WINDOW  # подстратегия ветки
        self._strategy: Optional[ContextStrategy] = None
        self._facts_store = FactsStore(
            max_items=self._setting_int("facts_max_items", 50),
            max_tokens=self._setting_int("facts_max_tokens", 300),
            counter=self._counter,
        )
        #: Плоские факты диалога ({категория: {ключ: значение}})
        self._facts: Dict[str, Dict[str, str]] = self._facts_store.data
        self._facts_error: Optional[str] = None
        self._facts_updated_at: Optional[str] = None
        self._active_branch_id: int = 1
        self._resolve_strategy_mode(strategy, strategy_settings)

        # Явная модель памяти (short_term | working | long_term).
        # Включена только если настройки памяти переданы явно (или передан
        # готовый менеджер/хранилище) — иначе поведение ровно как раньше.
        self._memory_settings: Dict[str, Any] = dict(memory_settings or {})
        self._memory_enabled: bool = bool(self._memory_settings.get("enabled", True))
        if memory_settings is None and memory_manager is None and long_memory is None:
            self._memory_enabled = False
        self._memory_error: Optional[str] = None
        self._memory_updated_at: Optional[str] = None

        # Долговременная память: отдельный файл, создаётся лениво и НЕ
        # создаётся в mock-режиме (там LongTermStore отключён).
        long_path = str(self._memory_settings.get("long_path") or "").strip()
        long_enabled = bool(self._memory_settings.get("long_enabled", True))
        if long_memory is not None:
            self._long_memory: LongTermStore = long_memory
        elif (self._memory_enabled and long_enabled and long_path
                and not self._mock_mode):
            from pathlib import Path as _Path
            from .config import PROJECT_ROOT as _PROJECT_ROOT
            p = _Path(long_path)
            if not p.is_absolute():
                p = _PROJECT_ROOT / p
            self._long_memory = LongTermStore(
                p, enabled=True,
                max_items=int(self._memory_settings.get("long_max_items", 200)),
            )
        else:
            self._long_memory = LongTermStore(":memory:", enabled=False)

        self._memory_manager: MemoryManager = memory_manager or MemoryManager(
            settings=self._memory_settings,
            long_store=self._long_memory,
            counter=self._counter,
            facts_store=self._facts_store,
        )
        # Публичная ссылка на рабочую память (для UI/тестов)
        self._working_memory: WorkingMemory = self._memory_manager.working

        self._branch_manager: BranchManager = branch_manager or BranchManager(
            # В mock-режиме хранилище полностью игнорируется (как и раньше)
            store=None if self._mock_mode else self._storage,
            max_branches=self._setting_int("branch_max", 20),
            default_strategy=self._sub_strategy_name,
        )

        # Подсистема задач (FSM): репозиторий создаётся лениво при первой
        # операции с задачами (или передаётся явно), fsm привязывается к
        # конкретной задаче в start_task/continue_task/restore_task.
        self._task_repo: Optional[TaskRepository] = task_repository
        self._task_max_retries: int = (
            int(task_max_retries) if task_max_retries is not None
            else TASK_RETRY_LIMIT
        )
        self._task_parser = TaskResponseParser()
        self._fsm: Optional[TaskFSM] = None

        # Tool calling (агентный цикл по MCP-серверам: напоминания и
        # новости). Список tools запрашивается у серверов лениво и
        # кэшируется по клиенту (TTL 60 с) — localhost, но на каждый ход
        # дергать HTTP не нужно. Ошибка одного сервера не ломает диалог:
        # его tools просто не попадают в запрос.
        self._reminder_client = reminder_client
        self._news_client = news_client
        self._geo_client = geo_client
        self._weather_client = weather_client
        self._tool_clients: List[Any] = [
            client for client in (
                reminder_client, news_client, geo_client, weather_client,
            )
            if client is not None
        ]
        self._max_tool_rounds: int = (
            int(max_tool_rounds) if max_tool_rounds is not None
            else 5
        )
        # Кэш tools по клиенту: id(client) → (timestamp, tools в формате LLM)
        self._tools_cache: Dict[int, Tuple[float, List[Dict[str, Any]]]] = {}
        self._tools_ttl: float = 60.0
        # Маршрутизация вызовов: имя инструмента → клиент его сервера
        # (обновляется при каждом успешном list_tools)
        self._tool_routes: Dict[str, Any] = {}

        # RAG-хук (миграция на Flet, app/rag/): одноразовый контекст для
        # следующего хода. set_rag_context задаёт полный текст последнего
        # user-сообщения запроса к API (промпт с чанками базы знаний) и
        # список источников; история диалога при этом хранит вопрос
        # пользователя как есть. Контекст расходуется одним ходом: в
        # конце send_message сбрасывается.
        self._rag_block: Optional[str] = None
        self._rag_sources: Optional[List[Dict[str, Any]]] = None

        # Клиент создаётся только вне mock-режима; при активном провайдере
        # используется его клиент (base_url из конфига провайдера)
        self._client: Optional[OpenAI] = None
        if not self._mock_mode:
            if self._provider is not None:
                self._client = self._provider.client
            else:
                self._client = OpenAI(
                    api_key=self._api_key,
                    base_url=self._provider_base_url,
                    timeout=60.0,
                    max_retries=1,
                )

        # История начинается с системного промпта
        self._message_history: List[Message] = []
        self._init_history()

        # Восстановление контекста из хранилища (только вне mock-режима)
        # Сначала ветки: они определяют активную ветку и её стратегию.
        # В mock-режиме менеджер создан без хранилища, поэтому обращений к БД нет.
        self._branch_manager.load()
        self._active_branch_id = self._branch_manager.active_id()
        if not self._mock_mode:
            self._load_context()

        # Профиль перекрывает параметры, восстановленные из хранилища:
        # источник истины для модели/промпта — настройки активного профиля.
        if self._profile_repo is not None and self._profile is not None:
            self._apply_profile_settings()
            prompt = build_system_prompt(
                self._profile, self._profile_settings,
                invariants_block=self._build_invariants_block(),
            )
            self.system_prompt = prompt
            if (self._message_history
                    and self._message_history[0].get("role") == "system"):
                self._message_history[0]["content"] = prompt
            else:
                self._message_history.insert(
                    0, {"role": "system", "content": prompt}
                )

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
            - timestamp: локальное время ответа (строка);
            - regenerations: число перегенераций пост-проверки инвариантов
              (ключ появляется, только если проверка выполнялась);
            - invariant_warning: True — лимит перегенераций исчерпан,
              нарушение могло остаться (предупреждение для UI);
            - task_transition: результат разбора маркеров задачи
              ``{"intent", "transitioned_to"?, "transition_error"?, "hint"?}``
              (ключ появляется, только когда есть активная задача);
            - transition_error: LLM предложила переход, запрещённый
              матрицей FSM (текст ошибки — для статус-бара);
            - transition_hint: LLM не дала маркер задачи (fail-safe);
            - tool_calls: список выполненных вызовов инструментов
              напоминаний ``{"name", "arguments", "result"}`` (ключ
              появляется, только если LLM запросила tool_calls, а
              подключён MCP-клиент напоминаний);
            - file_path: абсолютный путь файла, сохранённого инструментом
              save_to_file news-сервера (ключ появляется, только если
              вызов состоялся; UI показывает кнопку «📂 Открыть файл»).

            При ошибке запроса поле tokens = None, а счётчики сессии
            не изменяются.
        """
        if not user_input or not user_input.strip():
            return self._error_result("Пустое сообщение.")

        with self._lock:
            # Профиль мог измениться в БД (редактирование профиля или его
            # настроек в UI): перечитываем его и обновляем system-сообщение
            # в позиции [0] — без очистки истории.
            self._refresh_profile_locked()
            # Сжатие старой части диалога выполняется ДО добавления нового
            # сообщения, чтобы границы блока всегда совпадали с парами
            # «вопрос-ответ» и в запрос уходили ровно последние N сообщений.
            summarized = self._maybe_summarize()

            self._append_message({"role": "user", "content": user_input.strip()})
            # Снимок того, что реально уйдёт в API (system + конспект +
            # непокрытая история) — используется для оценки токенов, если
            # API не вернул usage. RAG-контекст подмешивается и в снимок:
            # оценка токенов должна включать чанки базы знаний.
            request_messages = self._build_context_messages()
            if self._rag_block:
                request_messages = self._apply_rag_block(request_messages)

            start = time.perf_counter()
            try:
                if self._mock_mode:
                    payload = self._mock_response(user_input)
                else:
                    payload = self._call_api()

                elapsed = round(time.perf_counter() - start, 2)
                # Явные переходы FSM (по ТЗ): LLM предлагает переход маркером
                # в конце ответа, решение принимает TaskFSM по матрице.
                # Маркеры — служебные: из текста, уходящего в историю и чат,
                # они убираются.
                transition_info = self._apply_task_response_transition(
                    user_input.strip(), str(payload.get("content") or "")
                )
                if transition_info is not None:
                    payload["content"] = TaskResponseParser.strip_markers(
                        str(payload.get("content") or "")
                    )
                self._append_message(
                    {"role": "assistant", "content": payload["content"]}
                )
                # В режиме стратегий «сырая» история не урезается: sliding
                # window отбрасывает сообщения только в самом запросе.
                if self._trimming_active():
                    self._trim_history()
                # Пост-проверка инвариантов (второй LLM-вызов): при
                # нарушении ответ перегенерируется (лимит попыток из
                # конфига), финальный текст заменяет ответ в истории.
                invariant_info = self._post_check_invariants(
                    user_input.strip(), payload
                )
                # Стратегия обновляет свои артефакты после хода
                # (для sticky_facts — служебный запрос обновления фактов).
                facts_info = self._after_turn(
                    user_input.strip(), payload["content"]
                )
                # Классификация памяти: один служебный LLM-запрос предлагает
                # кандидатов в pending (в слои не пишет). Ошибка не ломает ход.
                self._classify_memory_after_turn(
                    user_input.strip(), payload["content"]
                )
                # Учитываем токены обмена (точные из API или оценку)
                token_info = self._update_token_stats(
                    payload, request_messages,
                    summarized_this_turn=summarized,
                )
                if facts_info is not None:
                    token_info["facts_updated"] = bool(facts_info.get("updated"))
                    token_info["strategy"] = self._strategy_name
                # Сохраняем контекст после каждого успешного обмена
                self._persist()

                result = {
                    "text": payload["content"],
                    "model": payload.get("model", self.model),
                    "tokens": token_info,
                    "response_time": elapsed,
                    "finish_reason": payload.get("finish_reason"),
                    "error": None,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
                if invariant_info is not None:
                    result["regenerations"] = invariant_info["regenerations"]
                    result["invariant_warning"] = \
                        invariant_info["warning"]
                if transition_info is not None:
                    result["task_transition"] = transition_info
                    if transition_info.get("transition_error"):
                        result["transition_error"] = \
                            transition_info["transition_error"]
                    if transition_info.get("hint"):
                        result["transition_hint"] = transition_info["hint"]
                # Tool calling: какие инструменты вызвала LLM и что они
                # вернули (UI показывает это под ответом ассистента)
                executed_tools = payload.pop("tool_calls_executed", [])
                if executed_tools:
                    result["tool_calls"] = executed_tools
                # Путь сохранённого файла (save_to_file news-сервера) —
                # для кнопки «📂 Открыть файл»
                saved_path = payload.pop("saved_file_path", None)
                if saved_path:
                    result["file_path"] = saved_path
                # RAG: источники чанков, использованных в промпте (панель
                # «Источники» в UI). Появляется, только если контекст был
                # установлен через set_rag_context.
                if self._rag_sources is not None:
                    result["sources"] = list(self._rag_sources)
                    result["rag_used"] = bool(self._rag_block)
                return result
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
            finally:
                # Одноразовость RAG-контекста: блок расходуется одним ходом
                self._rag_block = None
                self._rag_sources = None

    def set_rag_context(
        self,
        user_content: Optional[str],
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Устанавливает одноразовый RAG-контекст для следующего хода.

        Вызывается RAG-обёрткой (app/agent.py) перед send_message:
        ``user_content`` — полный текст последнего user-сообщения запроса
        к API (промпт с чанками базы знаний, собранный retriever'ом);
        ``sources`` — метаданные чанков для панели «Источники».

        История диалога не изменяется: вопрос пользователя сохраняется
        как есть, контекст уходит только в копию запроса (см.
        _apply_rag_block). Контекст расходуется одним ходом send_message
        и сбрасывается в его конце.

        Args:
            user_content: текст промпта с контекстом (None — сбросить).
            sources: список словарей-источников (source, section, preview,
                strategy, distance и т.п.) или None.
        """
        with self._lock:
            self._rag_block = user_content
            self._rag_sources = list(sources) if sources is not None else (
                [] if user_content is not None else None
            )

    def clear_rag_context(self) -> None:
        """Сбрасывает RAG-контекст, установленный set_rag_context."""
        with self._lock:
            self._rag_block = None
            self._rag_sources = None

    def _apply_rag_block(self, messages: List[Message]) -> List[Message]:
        """Подставляет RAG-контекст в копию сообщений запроса.

        Последнее user-сообщение заменяется на self._rag_block (полный
        промпт с контекстом и вопросом). История диалога не мутируется:
        заменяются копии словарей, исходные списки/словари не изменяются.

        Args:
            messages: сообщения запроса (system + конспект + история).

        Returns:
            Копия списка с заменённым последним user-сообщением.
        """
        augmented = list(messages)
        for index in range(len(augmented) - 1, -1, -1):
            if augmented[index].get("role") == "user":
                augmented[index] = dict(augmented[index])
                augmented[index]["content"] = self._rag_block
                break
        return augmented

    def clear_history(self) -> None:
        """Очищает историю диалога активной ветки.

        Сохранённый на диске контекст также очищается (остаётся только
        системный промпт и параметры агента). Накопленная статистика
        токенов сессии при этом сбрасывается — и в памяти, и на диске.
        Артефакты активной ветки (конспект и факты) очищаются вместе с
        историей: они описывают удалённый диалог. Закрепления и рабочая
        память активной ветки очищаются тоже. **Долговременная память не
        трогается** — она живёт в отдельном файле (``data/long_memory.db``).
        Удаление самой ветки — отдельная операция (:meth:`delete_branch`).
        """
        with self._lock:
            self._init_history()
            self._session_stats = self._default_session_stats()
            self._last_exchange = self._default_last_exchange()
            # Конспект и факты относятся к удалённой истории — очищаем
            self._summary = None
            self._last_summary_error = None
            self._facts_store.clear()
            self._facts_error = None
            self._facts_updated_at = None
            # Закрепления и рабочая память относятся к удалённой истории
            self._memory_manager.pins.clear()
            self._memory_manager.working.clear()
            self._memory_error = None
            self._memory_updated_at = None
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
                provider = self._provider
                if provider is not None and not provider.owns_model(model):
                    raise AgentConfigError(
                        f"Модель {model!r} не относится к провайдеру "
                        f"{provider.name}. Доступные модели: "
                        f"{', '.join(provider.model_choices())}."
                    )
                self.model = model
                # Суммаризатор следует за моделью диалога, если для сжатия
                # не задана отдельная модель
                if self._summarizer is not None and not str(
                    (self._summary_settings or {}).get("model") or ""
                ):
                    self._summarizer.model = self.model
                # Стратегия получает модель как зависимость (служебные запросы)
                if self._strategy is not None:
                    self._strategy = self._make_strategy()
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
                if new_mock:
                    self._client = None
                elif self._provider is not None:
                    self._client = self._provider.client
                else:
                    self._client = OpenAI(
                        api_key=self._api_key,
                        base_url=self._provider_base_url,
                        timeout=60.0,
                        max_retries=1,
                    )
            if "max_history_messages" in kwargs:
                self._max_history = new_max_history
                # В режиме стратегий «сырая» история не урезается:
                # ограничение действует только на то, что уходит в API
                if self._trimming_active():
                    self._trim_history()

            # Персист: смена системного промпта перезаписывает диалог
            # на диске, смена остальных параметров обновляет параметры.
            if self._persistence_active():
                if "system_prompt" in kwargs:
                    self._persist()
                elif any(k in kwargs for k in
                         ("model", "temperature", "max_tokens")):
                    self._persist_params()

    # ------------------------------------------------------------------
    # Провайдер LLM: облако DeepSeek ↔ локальная Ollama (ТЗ ai_1_25)
    # ------------------------------------------------------------------

    @property
    def provider(self) -> Optional[BaseLLMProvider]:
        """Активный провайдер LLM (None — облако DeepSeek напрямую)."""
        return self._provider

    @property
    def provider_id(self) -> str:
        """Идентификатор провайдера (deepseek / ollama_qwen / …)."""
        return self._provider.id if self._provider is not None else "deepseek"

    def switch_provider(self, provider_name: str,
                        api_key: Optional[str] = None) -> Dict[str, Any]:
        """Переключает провайдера LLM на лету (агент не пересоздаётся).

        Порядок (ТЗ): фабрика создаёт провайдера → проверка доступности
        (для Ollama — сервер запущен и модель загружена) → замена
        клиента, модели и base_url ядра. История диалога, профили,
        память и хранилища не затрагиваются.

        MCP-инструменты включаются только для облачных провайдеров с
        tool calling. Локальные провайдеры (``local = True``) работают
        БЕЗ инструментов независимо от capability модели: небольшие
        модели склонны к ложным вызовам (например, search_news на
        вопрос «факт о космосе»), поэтому запрос идёт только к модели —
        без tools и без блока правил инструментов в system-промпте.
        В mock-режиме проверка доступности пропускается (сети нет),
        политика инструментов детерминирована: local-флаг не требует
        HTTP, DeepSeek поддерживает tool calling всегда.

        Args:
            provider_name: ключ реестра LLMProviderFactory.PROVIDERS
                (deepseek / ollama_qwen / ollama_llama).
            api_key: ключ DeepSeek API; None — из окружения/агента.

        Returns:
            {"success": True, "provider", "name", "model"} либо тот же
            словарь с "warning" (локальный провайдер или нет tool
            calling — MCP отключён).

        Raises:
            ProviderUnavailableError: провайдер недоступен; состояние
                агента при этом не меняется.
            ValueError: неизвестное имя провайдера.
        """
        new_provider = LLMProviderFactory.create(
            provider_name,
            # Без явного ключа — ключ агента (переключение обратно на
            # DeepSeek не должно зависеть от переменных окружения)
            api_key=api_key if api_key is not None
            else (self._api_key or None),
            # Настройки оптимизации следуют за Ollama-провайдерами
            # (ТЗ ai_1_28): панель в UI пишет их в _ollama_optimization
            optimization=dict(self._ollama_optimization),
        )
        if not self._mock_mode and not new_provider.is_available():
            raise ProviderUnavailableError(
                f"Провайдер '{new_provider.name}' недоступен. Проверьте, "
                f"запущен ли Ollama ({new_provider.base_url}) и загружена "
                f"ли модель {new_provider.model}."
            )
        with self._lock:
            previous = self.provider_id
            self._provider = new_provider
            self._provider_base_url = new_provider.base_url
            self.model = new_provider.model
            if not self._mock_mode:
                self._client = new_provider.client
            # Суммаризатор и стратегия следуют за моделью (как update_config)
            if (self._summarizer is not None and not str(
                    (self._summary_settings or {}).get("model") or "")):
                self._summarizer.model = self.model
            if self._strategy is not None:
                self._strategy = self._make_strategy()
            # Порядок условий важен: для локальных провайдеров capability
            # НЕ проверяется (лишний HTTP-запрос к /api/show не нужен)
            if not new_provider.local and new_provider.supports_tool_calling():
                self._provider_tools_enabled = True
                logger.info(
                    "Провайдер переключён: %s → %s (%s), MCP-инструменты "
                    "включены.", previous, new_provider.id,
                    new_provider.model,
                )
                return {
                    "success": True,
                    "provider": new_provider.id,
                    "name": new_provider.name,
                    "model": new_provider.model,
                }
            self._provider_tools_enabled = False
            warning = (
                "Локальная модель: запросы идут только к ней — "
                "MCP-инструменты (напоминания, новости, погода) отключены."
                if new_provider.local
                else ("Модель не поддерживает tool calling. "
                      "MCP-инструменты отключены.")
            )
            logger.warning(
                "Провайдер %s: MCP-инструменты отключены (%s).",
                new_provider.id, warning,
            )
            return {
                "warning": warning,
                "provider": new_provider.id,
                "name": new_provider.name,
                "model": new_provider.model,
            }

    # ------------------------------------------------------------------
    # Оптимизация локальной модели Ollama (ТЗ ai_1_28)
    # ------------------------------------------------------------------

    @property
    def ollama_optimization(self) -> Dict[str, Any]:
        """Настройки оптимизации локальной модели (глобальные)."""
        return dict(self._ollama_optimization)

    def set_ollama_optimization(
            self, settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Обновляет настройки оптимизации локальной модели.

        Настройки глобальные (вне профилей): применяются к активному
        Ollama-провайдеру сразу (горячо, через ``set_optimization``) и
        передаются фабрикой всем провайдерам, созданным позже через
        :meth:`switch_provider`. Для DeepSeek-провайдера значения просто
        запоминаются — до переключения на Ollama они ни на что не влияют.
        Санитизация (типы и диапазоны) — OptimizationSettings.sanitize.

        Args:
            settings: новые настройки; None/пусто — оптимизация снята.

        Returns:
            Санитизированный словарь, который сохранён в ядре.
        """
        from .llm_optimization import OptimizationSettings

        clean = OptimizationSettings.sanitize(settings)
        with self._lock:
            self._ollama_optimization = clean
            apply_settings = getattr(self._provider, "set_optimization",
                                     None)
            if callable(apply_settings):
                apply_settings(clean)
        logger.info(
            "Оптимизация локальной модели: enabled=%s, temperature=%s, "
            "num_predict=%s, num_ctx=%s, top_p=%s, repeat_penalty=%s",
            clean["enabled"], clean["temperature"], clean["max_tokens"],
            clean["num_ctx"], clean["top_p"], clean["repeat_penalty"],
        )
        return clean

    # ------------------------------------------------------------------
    # Персонализация: профиль пользователя (третий слой памяти)
    # ------------------------------------------------------------------

    @property
    def profile_repository(self) -> Optional[ProfileRepository]:
        """Репозиторий профилей (None — агент работает без персонализации)."""
        return self._profile_repo

    @property
    def profile(self) -> Optional[UserProfile]:
        """Активный профиль пользователя (None — персонализация выключена)."""
        with self._lock:
            return self._profile

    @property
    def profile_settings(self) -> Optional[ProfileAgentSettings]:
        """Настройки агента активного профиля (None — без персонализации)."""
        with self._lock:
            return self._profile_settings

    def _active_profile_id(self) -> Optional[int]:
        """Идентификатор активного профиля (None — без персонализации)."""
        return self._profile.id if self._profile is not None else None

    def _apply_profile_settings(self) -> None:
        """Применяет настройки профиля к параметрам агента (с валидацией).

        Некорректные значения в БД не роняют агента: параметр игнорируется
        с предупреждением в логе. Суммаризатор и стратегия следуют за
        моделью (как в :meth:`update_config`).
        """
        settings = self._profile_settings
        if settings is None:
            return
        provider = getattr(self, "_provider", None)
        if provider is not None and not provider.owns_model(settings.model):
            # Профиль хранит модель другого провайдера (например, DeepSeek
            # при активной Ollama): модель не трогаем, остальные параметры
            # применяются как обычно
            logger.info(
                "Профиль: модель %r не относится к провайдеру %s — "
                "оставлена модель %s.",
                settings.model, provider.name, self.model,
            )
        else:
            try:
                self.model = utils.validate_model(settings.model)
            except AgentConfigError as exc:
                logger.warning(
                    "Профиль: некорректная модель %r (%s) — оставлено %s.",
                    settings.model, exc, self.model,
                )
        try:
            self.temperature = utils.validate_temperature(settings.temperature)
        except (AgentConfigError, TypeError, ValueError) as exc:
            logger.warning(
                "Профиль: некорректная temperature=%r (%s) — оставлено %s.",
                settings.temperature, exc, self.temperature,
            )
        try:
            self.max_tokens = utils.validate_max_tokens(settings.max_tokens)
        except (AgentConfigError, TypeError, ValueError) as exc:
            logger.warning(
                "Профиль: некорректный max_tokens=%r (%s) — оставлено %s.",
                settings.max_tokens, exc, self.max_tokens,
            )
        # Суммаризатор следует за моделью диалога (если нет отдельной)
        if (getattr(self, "_summarizer", None) is not None
                and not str((self._summary_settings or {}).get("model") or "")):
            self._summarizer.model = self.model
        # Стратегия получает модель как зависимость (служебные запросы)
        if getattr(self, "_strategy", None) is not None:
            self._strategy = self._make_strategy()

    def _replace_system_message(self, prompt: str) -> None:
        """Заменяет system-сообщение в позиции [0], не создавая дубликатов."""
        if (self._message_history
                and self._message_history[0].get("role") == "system"):
            self._message_history[0]["content"] = prompt
        else:
            self._message_history.insert(
                0, {"role": "system", "content": prompt}
            )

    def _replace_last_assistant_message(self, content: str) -> None:
        """Заменяет текст последнего ответа ассистента (пост-проверка).

        Вызывается, когда валидатор инвариантов перегенерировал ответ:
        в истории (и в БД при следующем персисте) остаётся финальный текст.
        """
        for index in range(len(self._message_history) - 1, -1, -1):
            if self._message_history[index].get("role") == "assistant":
                self._message_history[index]["content"] = content
                return

    def _refresh_profile_locked(self) -> None:
        """Перечитывает профиль из БД и обновляет промпт (вызов под lock).

        Вызывается в начале каждого :meth:`send_message`: если профиль или
        его настройки были изменены (UI, другой процесс), новый
        system-промпт и параметры применяются к текущему запросу без
        перезапуска. История диалога при этом не очищается.
        """
        if self._profile_repo is None or self._profile is None:
            return
        profile = self._profile_repo.get_by_id(self._profile.id)
        if profile is None:
            # Профиль удалён извне — пробуем переключиться на активный
            active = self._profile_repo.get_active()
            if active is not None and active.id != self._profile.id:
                self.set_profile(active.id)
            return
        self._profile = profile
        self._profile_settings = self._profile_repo.get_settings(profile.id)
        self._apply_profile_settings()
        prompt = build_system_prompt(
            self._profile, self._profile_settings,
            invariants_block=self._build_invariants_block(),
        )
        self.system_prompt = prompt
        self._replace_system_message(prompt)

    def refresh_profile(self) -> Dict[str, Any]:
        """Публичная перечитка профиля (редактирование профиля в UI).

        Загружает профиль и его настройки из БД заново, применяет параметры
        (модель/temperature/max_tokens) и пересобирает системный промпт,
        заменяя system-сообщение в позиции [0] без очистки истории.

        Returns:
            ``{"profile", "settings", "error"}``.
        """
        with self._lock:
            if self._profile_repo is None:
                return {
                    "profile": None,
                    "settings": None,
                    "error": "Профили не подключены (нет ProfileRepository).",
                }
            if self._profile is None:
                active = self._profile_repo.get_active()
                if active is not None:
                    return self.set_profile(active.id)
                return {"profile": None, "settings": None, "error": None}
            before = self._profile.id
            self._refresh_profile_locked()
            if self._profile is None or self._profile.id != before:
                return {
                    "profile": self._profile,
                    "settings": self._profile_settings,
                    "error": "Активный профиль был удалён.",
                }
            return {
                "profile": self._profile,
                "settings": self._profile_settings,
                "error": None,
            }

    def set_profile(self, profile_id: int) -> Dict[str, Any]:
        """Переключается на другой профиль (история не очищается — меняется).

        Порядок действий:
        1. текущая история/конспект/факты кэшируются за активным профилем
           и сохраняются на диск (с его ``profile_id``);
        2. загружаются целевой профиль и его настройки, применяются
           параметры (модель/temperature/max_tokens);
        3. восстанавливается история целевого профиля: из кэша этой сессии,
           либо из БД (сообщения с его ``profile_id``), либо пустая.

        Args:
            profile_id: идентификатор целевого профиля.

        Returns:
            ``{"profile", "settings", "error"}``.
        """
        with self._lock:
            if self._profile_repo is None:
                return {
                    "profile": None,
                    "settings": None,
                    "error": "Профили не подключены (нет ProfileRepository).",
                }
            profile = self._profile_repo.get_by_id(int(profile_id))
            if profile is None:
                return {
                    "profile": None,
                    "settings": None,
                    "error": f"Профиль id={profile_id} не найден.",
                }
            if self._profile is not None and profile.id == self._profile.id:
                return {
                    "profile": self._profile,
                    "settings": self._profile_settings,
                    "error": None,
                }

            # 1. Кэшируем и сохраняем состояние текущего профиля
            current_id = self._active_profile_id()
            if current_id is not None:
                self._profile_histories[current_id] = {
                    "messages": [dict(m) for m in self._message_history],
                    "summary": dict(self._summary) if self._summary else None,
                    "facts": self._facts_store.as_dict(),
                }
            self._persist()  # пишет сообщения с profile_id текущего профиля

            # 2. Переключаем профиль и применяем его настройки
            self._profile = profile
            self._profile_settings = self._profile_repo.get_settings(profile.id)
            self._apply_profile_settings()
            # Активный профиль фиксируется в БД: при следующем запуске
            # приложение восстановит последний использованный профиль.
            self._profile_repo.set_active(profile.id)
            prompt = build_system_prompt(
                self._profile, self._profile_settings,
                invariants_block=self._build_invariants_block(),
            )
            self.system_prompt = prompt

            # Артефакты диалога (конспект, факты, закрепления) относятся
            # к истории конкретного профиля — сбрасываем перед загрузкой.
            self._summary = None
            self._last_summary_error = None
            self._facts_store = self._new_facts_store({})
            self._facts = self._facts_store.data
            self._facts_error = None
            self._facts_updated_at = None
            self._memory_manager.pins.clear()

            # 3. История целевого профиля: кэш → БД → пустая
            cached = self._profile_histories.pop(profile.id, None)
            messages_loaded = False
            if cached is not None:
                self._message_history = [
                    dict(m) for m in cached.get("messages") or []
                ]
                if cached.get("summary"):
                    self._restore_summary(cached["summary"])
                if cached.get("facts"):
                    self._facts_store = self._new_facts_store(cached["facts"])
                    self._facts = self._facts_store.data
                messages_loaded = True
            elif not self._mock_mode and self._storage is not None:
                try:
                    state = self._storage.load_state(profile_id=profile.id)
                except Exception as exc:
                    logger.warning(
                        "Не удалось загрузить историю профиля id=%s: %s",
                        profile.id, exc,
                    )
                    state = None
                if state and state.get("messages"):
                    self._message_history = [
                        {"role": str(m.get("role", "")),
                         "content": str(m.get("content", ""))}
                        for m in state["messages"]
                    ]
                    messages_loaded = True
            if not messages_loaded:
                self._message_history = []

            # Ровно одно system-сообщение в позиции [0] — собранный промпт
            self._replace_system_message(prompt)
            if self._trimming_active():
                self._trim_history()

            self._persist()
            logger.info(
                "Профиль переключён: %r (id=%s), модель=%s, "
                "сообщений в истории: %d.",
                profile.name, profile.id, self.model,
                max(0, len(self._message_history) - 1),
            )
            return {
                "profile": self._profile,
                "settings": self._profile_settings,
                "error": None,
            }

    # ------------------------------------------------------------------
    # Инварианты: жёсткие правила в промпте и пост-проверка ответов
    # ------------------------------------------------------------------

    @property
    def invariant_repository(self) -> Optional[InvariantRepository]:
        """Репозиторий инвариантов (None — подсистема выключена).

        Создаётся лениво при первом обращении, если подключено хранилище
        контекста (миграция схемы выполняется в конструкторе репозитория).
        В mock-режиме репозиторий НЕ создаётся автоматически: инварианты
        работают только с явно переданным репозиторием (БД не создаётся).
        """
        if (self._invariant_repo is None and self._storage is not None
                and not self._mock_mode):
            self._invariant_repo = InvariantRepository(self._storage.db_path)
        return self._invariant_repo

    @property
    def invariant_validator(self) -> Optional[InvariantValidator]:
        """Валидатор пост-проверки ответов (всегда доступен)."""
        return self._invariant_validator

    def _build_invariants_block(self) -> str:
        """Собирает блок инвариантов для system-промпта (или "").

        Читает global + инварианты активного профиля из репозитория.
        Любая ошибка не должна ломать диалог: возвращается пустая строка.
        """
        repo = self.invariant_repository
        if repo is None or not repo.enabled:
            return ""
        try:
            invariants = repo.get_invariants_for_profile(
                self._active_profile_id()
            )
            if not invariants:
                return ""
            categories = repo.get_all_categories()
            return build_invariants_block(invariants, categories)
        except Exception as exc:  # подсистема не должна ронять агента
            logger.warning("Не удалось собрать блок инвариантов: %s", exc)
            return ""

    def _get_active_invariants(self) -> List[Invariant]:
        """Действующие инварианты (global + активный профиль)."""
        repo = self.invariant_repository
        if repo is None or not repo.enabled:
            return []
        try:
            return repo.get_invariants_for_profile(self._active_profile_id())
        except Exception as exc:
            logger.warning("Не удалось прочитать инварианты: %s", exc)
            return []

    def _invariant_call_api(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Служебный запрос пост-проверки инвариантов.

        Подставляет модель диалога, если у валидатора не задана своя
        (``model = ""`` означает «модель диалога»).
        """
        if not request.get("model"):
            request = dict(request, model=self.model)
        return self._call_service_api(request)

    def regenerate_last_response(self, user_query: str,
                                 violated_rule: str) -> str:
        """Перегенерирует последний ответ с напоминанием про инвариант.

        Используется пост-проверкой (:meth:`InvariantValidator.
        validate_and_regenerate`). Запрос строится из текущего контекста,
        но без нарушившего ответа; в конец добавляется явное напоминание.

        Args:
            user_query: исходный запрос пользователя.
            violated_rule: формулировка нарушенного инварианта.

        Returns:
            Новый текст ответа.

        Raises:
            ApiError: при ошибке запроса (в mock-режиме — всегда:
                клиента API нет, перегенерация невозможна).
        """
        with self._lock:
            messages = [dict(m) for m in self._build_context_messages()]
        # Убираем нарушивший ответ: перегенерация должна выдать текст заново
        while messages and messages[-1].get("role") == "assistant":
            messages.pop()
        reminder = (
            f"Предыдущий ответ нарушил инвариант: {violated_rule}.\n"
            "Сгенерируй ответ заново, строго соблюдая инварианты.\n"
            f"Пользовательский запрос: {user_query}"
        )
        messages.append({"role": "user", "content": reminder})
        request = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        payload = self._call_service_api(request)
        return str(payload.get("content") or "")

    def _post_check_invariants(self, user_query: str,
                               payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Пост-проверка ответа на нарушение инвариантов (после генерации).

        Условия запуска (по ТЗ): проверка включена (ENABLE_POST_CHECK),
        ответ достаточно длинный (MIN_RESPONSE_LENGTH) и есть действующие
        инварианты. При нарушении ответ перегенерируется валидатором;
        финальный текст заменяет ответ в истории, отказы и перегенерации
        пишутся в лог нарушений. Ошибки валидатора не ломают ответ.

        Returns:
            ``{"text", "regenerations", "warning"}`` — если проверка
            выполнялась, иначе None.
        """
        validator = self._invariant_validator
        text = str(payload.get("content") or "")
        if validator is None or not validator.enabled:
            return None
        if len(text) < validator.min_response_length:
            return None
        invariants = self._get_active_invariants()
        if not invariants:
            return None

        try:
            final_text, regenerations = validator.validate_and_regenerate(
                agent=self,
                user_query=user_query,
                first_response=text,
                invariants=invariants,
                call_api=self._invariant_call_api,
            )
        except Exception as exc:  # проверка не должна ломать диалог
            logger.warning("Пост-проверка инвариантов не удалась: %s", exc)
            return None

        final_text = str(final_text or text)
        if final_text != text:
            payload["content"] = final_text
            self._replace_last_assistant_message(final_text)

        warning = regenerations >= validator.max_regenerations
        self._log_invariant_outcome(user_query, final_text,
                                    regenerations, invariants)
        return {
            "text": final_text,
            "regenerations": regenerations,
            "warning": warning,
        }

    def _log_invariant_outcome(self, user_query: str, final_text: str,
                               regenerations: int,
                               invariants: List[Invariant]) -> None:
        """Пишет отказ/перегенерацию в лог нарушений (invariant_violations).

        - отказ по инварианту (без нарушения) → запись resolved=1;
        - нарушение с перегенерацией → запись regenerated=1, resolved
          по итогу (устранено/осталось после лимита попыток).
        """
        validator = self._invariant_validator
        repo = self.invariant_repository
        if validator is None or repo is None or not repo.enabled:
            return
        result = validator.last_result
        if result is None:
            return
        fallback_id = int(invariants[0].id or 0) if invariants else 0
        invariant_id = (result.invariant_id
                        if result.invariant_id is not None else fallback_id)
        if invariant_id <= 0:
            return
        profile_id = self._active_profile_id()
        try:
            if result.refusal and regenerations == 0:
                repo.log_violation(
                    invariant_id, profile_id, user_query, final_text,
                    regenerated=False, resolved=True,
                )
            elif regenerations > 0:
                repo.log_violation(
                    invariant_id, profile_id, user_query, final_text,
                    regenerated=True, resolved=not result.violates,
                )
        except Exception as exc:  # лог не должен ломать диалог
            logger.warning("Не удалось записать нарушение инварианта: %s", exc)

    def check_invariant_conflicts(self, new_invariant: Invariant,
                                  existing: List[Invariant],
                                  categories: List[InvariantCategory]
                                  ) -> List[Conflict]:
        """LLM-проверка противоречий инварианта (для UI при сохранении).

        Возвращает список :class:`~src.invariant_validator.Conflict`;
        ошибки дают пустой список — сохранение не блокируется.

        Args:
            new_invariant: сохраняемый инвариант.
            existing: существующие инварианты для сравнения (без самого
                сохраняемого).
            categories: категории (для подписей в промпте).
        """
        validator = self._invariant_validator
        if validator is None or not existing:
            return []
        try:
            return validator.find_conflicts(
                new_invariant, existing, categories,
                call_api=self._invariant_call_api,
            )
        except Exception as exc:
            logger.warning("Проверка конфликтов инвариантов не удалась: %s",
                           exc)
            return []

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
            # Закрываем долговременную память (идемпотентно)
            self._memory_manager.close()
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

    def _setting_int(self, key: str, default: int) -> int:
        """Читает целочисленную настройку стратегий с мягким fallback."""
        try:
            return int(self._strategy_settings.get(key, default))
        except (TypeError, ValueError):
            return int(default)

    def _resolve_strategy_mode(self, strategy: Optional[str],
                               strategy_settings: Optional[Dict[str, Any]]) -> None:
        """Определяет, работает ли агент через стратегии контекста.

        Режим включается только при явном выборе стратегии или передаче её
        настроек — иначе сохраняется прежнее поведение (summary).

        Raises:
            AgentConfigError: если имя стратегии неизвестно.
        """
        if strategy is None and strategy_settings is None:
            self._strategy_name = None
            self._strategy = None
            return
        settings = self._strategy_settings
        raw = strategy if strategy is not None else settings.get("name")
        name = normalize_name(raw or SLIDING_WINDOW)
        sub_raw = settings.get("branch_strategy") or SLIDING_WINDOW
        if name != BRANCHING:
            sub_raw = name
        sub = normalize_name(sub_raw)
        if sub == BRANCHING:  # защита от вложенного режима веток
            sub = SLIDING_WINDOW
        self._strategy_name = name
        self._sub_strategy_name = sub
        self._strategy = self._make_strategy()

    def _make_strategy(self) -> ContextStrategy:
        """Создаёт объект стратегии для подстратегии активной ветки."""
        deps = {"counter": self._counter, "model": self.model}
        return create_strategy(self._sub_strategy_name,
                               self._strategy_settings, deps)

    def _strategy_mode(self) -> bool:
        """True, если контекст собирает стратегия (а не legacy-режим)."""
        return self._strategy is not None

    def _uses_facts(self) -> bool:
        """True, если активная стратегия использует блок фактов."""
        return self._strategy_mode() and self._sub_strategy_name == STICKY_FACTS

    def _artifacts(self) -> Dict[str, Any]:
        """Артефакты диалога для стратегии (факты, конспект, счётчик)."""
        return {
            "facts": self._facts_store,
            "summary": self._summary,
            "counter": self._counter,
        }

    def _trimming_active(self) -> bool:
        """True, если «сырая» история ограничивается по MAX_HISTORY_MESSAGES.

        В режиме стратегий история в памяти не урезается: sliding window
        отбрасывает сообщения только в запросе, а sticky facts — тем более.
        """
        return not self._strategy_mode() and not self._summary_active()

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

        # Режим стратегий: что уходит в API, решает стратегия
        # (sliding_window: system + последние N; sticky_facts: system +
        # блок фактов + последние N).
        if self._strategy is not None:
            built = self._strategy.build_context(
                str((system or {}).get("content") or ""), dialog,
                self._artifacts(),
            )
            return self._limit_context(self._insert_memory_blocks(built))

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
        return self._insert_memory_blocks(messages)

    def _memory_blocks(self) -> List[Message]:
        """Три system-сообщения памяти в фиксированном порядке.

        Порядок: долговременная → рабочая → закреплённые сообщения. Пустые
        слои не добавляются. Без ``MEMORY_ENABLED`` возвращает пустой список
        (контекст побайтово совпадает с текущим поведением).
        """
        if not self._memory_enabled:
            return []
        dialog = self._dialog_messages()
        pinned_messages: List[Dict[str, Any]] = []
        for pin in self._memory_manager.list_pinned():
            index = int(pin["index"])
            if 0 <= index < len(dialog):
                message = dialog[index]
                pinned_messages.append({
                    "index": index,
                    "role": message.get("role", "user"),
                    "content": message.get("content", ""),
                    "note": pin["note"],
                })
        return self._memory_manager.build_blocks(
            self._counter, self._memory_settings,
            facts=self._facts_store, facts_active=self._uses_facts(),
            pinned_messages=pinned_messages,
        )

    def _insert_memory_blocks(self, messages: List[Message]) -> List[Message]:
        """Вставляет блоки памяти сразу после системного промпта."""
        blocks = self._memory_blocks()
        if not blocks:
            return messages
        insert_at = 1 if messages and messages[0].get("role") == "system" else 0
        return list(messages[:insert_at]) + blocks + list(messages[insert_at:])

    def _limit_context(self, messages: List[Message]) -> List[Message]:
        """Аварийно ограничивает отправляемый контекст (MAX_HISTORY_MESSAGES).

        Системные сообщения (промпт и блок фактов) никогда не удаляются —
        ограничение касается только сообщений диалога.

        Args:
            messages: собранный стратегией контекст.

        Returns:
            Контекст с не более чем ``_max_history`` сообщениями диалога.
        """
        positions = [index for index, message in enumerate(messages)
                     if message.get("role") != "system"]
        if len(positions) <= self._max_history:
            return messages
        drop = len(positions) - self._max_history
        keep_from = positions[drop]
        logger.warning(
            "Контекст превышает MAX_HISTORY_MESSAGES=%d: в API уйдут "
            "только последние %d сообщений диалога.",
            self._max_history, self._max_history,
        )
        result = [m for m in messages[:keep_from] if m.get("role") == "system"]
        result.extend(messages[keep_from:])
        return result

    def _build_request(self) -> Dict[str, Any]:
        """Формирует структуру запроса к API.

        Если есть активная задача, к system-сообщению добавляется блок
        маркеров состояния (LLM сообщает о смене этапа, решение о
        переходе принимает FSM по матрице).

        Если установлен RAG-контекст (set_rag_context), последнее
        user-сообщение запроса заменяется промптом с чанками базы
        знаний; история диалога не изменяется.

        Returns:
            Словарь параметров для chat.completions.create: модель,
            контекст (system + конспект + непокрытая часть истории),
            температура, max_tokens.
        """
        messages = self._build_context_messages()
        if self._rag_block:
            messages = self._apply_rag_block(messages)
        markers_block = self._task_markers_block()
        if markers_block and messages:
            # Копируем system-сообщение: не мутируем историю диалога
            first = dict(messages[0])
            first["content"] = (
                str(first.get("content") or "") + "\n\n" + markers_block
            )
            messages = [first] + list(messages[1:])
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _task_markers_block(self) -> str:
        """Блок «[МАРКЕРЫ СОСТОЯНИЯ ЗАДАЧИ]» для system-промпта (или "").

        Уровень 1 защиты (по ТЗ): объясняет LLM, какой маркер поставить
        в конце ответа, и сообщает текущую стадию с допустимыми
        переходами. Решение о переходе всё равно принимает FSM — маркер
        лишь предложение.
        """
        task = self.active_task
        if task is None or task.stage not in ACTIVE_STAGES:
            return ""
        allowed = get_allowed_transitions(task.stage, task.mode)
        allowed_text = ", ".join(
            sorted(stage.value for stage in allowed)
        ) or "нет (стадия без выходов)"
        return (
            "[МАРКЕРЫ СОСТОЯНИЯ ЗАДАЧИ]\n\n"
            "В конце каждого ответа, относящегося к задаче, добавь ОДИН "
            "из маркеров:\n\n"
            "[PLAN_READY]        — план сформирован и готов к утверждению\n"
            "[STEP_DONE]         — текущий шаг выполнен\n"
            "[ALL_STEPS_DONE]    — все шаги выполнены, пора валидировать\n"
            "[VALIDATION_OK]     — валидация пройдена\n"
            "[VALIDATION_FAIL]   — валидация не пройдена, нужна переделка\n"
            "[NEED_INPUT]        — нужен ввод пользователя\n\n"
            "Маркер ставится на отдельной строке в самом конце ответа.\n\n"
            f"Текущее состояние задачи: {task.stage.value}\n"
            f"Допустимые следующие состояния: {allowed_text}"
        )

    def _apply_task_response_transition(
        self, user_query: str, response_text: str,
    ) -> Optional[Dict[str, Any]]:
        """Определяет намерение перехода из ответа LLM и применяет его.

        Вызывается из :meth:`send_message`, когда есть активная задача.
        Парсер: маркер → эвристика → None (fail-safe). None — перехода
        нет, в журнал пишется ``unresolved_transition`` (по ТЗ это сигнал,
        что LLM не дала маркер). Намерение применяется только через
        FSM: недопустимые переходы отклоняются матрицей.

        Args:
            user_query: запрос пользователя (для журнала).
            response_text: текст ответа ассистента.

        Returns:
            None — активной задачи нет (проверка не выполнялась); иначе
            словарь с ключами ``intent``, и по обстоятельствам:
            ``transitioned_to``, ``transition_error``, ``hint``.
        """
        task = self.active_task
        if task is None or task.stage not in ACTIVE_STAGES:
            return None
        intent = self._task_parser.parse(response_text)
        if intent is None:
            # Fail-safe: не переходим никуда, логируем сигнал для анализа
            self.repo_log_task_event(
                task.id, task.stage, None, "unresolved_transition",
                {"query": user_query[:200]},
            )
            return {
                "intent": None,
                "hint": ("LLM не указала явный маркер задачи — "
                         "переход не выполнен."),
            }

        target = self._transition_target_for(intent)
        if target is None:
            # need_input и прочие намерения без смены стадии
            return {"intent": intent}
        if target == task.stage:
            # Самопереход (например, step_done остаётся в execution):
            # событие фиксируем, стадию не трогаем
            self.repo_log_task_event(
                task.id, task.stage, target, f"llm_{intent}",
                {"allowed": True, "reason": "самопереход — стадия не меняется"},
            )
            return {"intent": intent}

        assert self._fsm is not None
        try:
            self._fsm.transition_to(target, f"llm_{intent}")
        except InvalidTransitionError as exc:
            return {
                "intent": intent,
                "transition_error": str(exc),
            }
        return {"intent": intent, "transitioned_to": target.value}

    def _transition_target_for(self, intent: str) -> Optional[TaskStage]:
        """Маппинг намерения LLM → целевая стадия (None — без перехода).

        Учёт режима: в manual план и итог проходят через подтверждения
        (AWAITING_*), в auto — напрямую; step_done остаётся в EXECUTION
        (шаги идут автоматически в обоих режимах).
        """
        manual = (self._fsm is not None
                  and self._fsm.task.mode == TaskMode.MANUAL)
        if intent == "plan_ready":
            return (TaskStage.AWAITING_PLAN_APPROVAL if manual
                    else TaskStage.EXECUTION)
        if intent == "step_done":
            return TaskStage.EXECUTION
        if intent == "all_steps_done":
            return TaskStage.VALIDATION
        if intent == "validation_ok":
            return (TaskStage.AWAITING_FINAL_APPROVAL if manual
                    else TaskStage.DONE)
        if intent == "validation_fail":
            return TaskStage.EXECUTION
        return None  # need_input и неизвестные намерения

    def repo_log_task_event(self, task_id: int,
                            from_stage: Optional[TaskStage],
                            to_stage: Optional[TaskStage],
                            event_type: str,
                            payload: Optional[Dict[str, Any]] = None) -> None:
        """Пишет событие в журнал задач (обёртка для служебных записей)."""
        repo = self.task_repository
        if repo is not None:
            repo.log_event(task_id, from_stage, to_stage, event_type, payload)

    def _call_api(self) -> Dict[str, Any]:
        """Выполняет запрос к DeepSeek API и обрабатывает tool_calls.

        Если подключён MCP-клиент напоминаний (:attr:`_reminder_client`),
        в запрос добавляется список tools; ответ с ``tool_calls``
        обрабатывается циклом: инструменты вызываются через MCP-клиент,
        результаты (роль ``tool`` с ``tool_call_id``) возвращаются LLM
        для финального ответа. Цикл ограничен ``_max_tool_rounds``.
        Промежуточные сообщения (assistant с tool_calls, tool) существуют
        только внутри этого цикла — в историю диалога попадает финальный
        текст, поэтому персистентность и стратегии контекста не меняются.

        Returns:
            Словарь с ключами content, model, tokens, finish_reason;
            при выполнении инструментов добавляется ``tool_calls_executed``
            (список ``{"name", "arguments", "result"}``), а при Thinking
            Mode — ``reasoning_content`` последнего ответа.

        Raises:
            ApiError: при ошибке аутентификации, сети, лимитов и т.п.
        """
        if self._client is None:
            raise ApiError("Клиент API не инициализирован (mock-режим?).")

        # Сначала tools: от их наличия зависит и system-блок правил
        tools = self._get_tools_for_llm()
        request = self._build_request()
        # Параметры оптимизации локальной модели (ТЗ ai_1_28): провайдер
        # Ollama добавляет extra_body с options и кастомный system-промпт
        # (копия сообщений — история не мутируется). Только для основного
        # хода: служебные вызовы (суммаризатор, инварианты, задачи) имеют
        # собственные бюджеты токенов и промпты.
        apply_optimization = getattr(self._provider,
                                     "apply_optimization", None)
        if callable(apply_optimization):
            request = apply_optimization(request)
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"
            # Правила использования инструментов — в конец system-промпта
            messages = list(request.get("messages") or [])
            if messages:
                first = dict(messages[0])
                first["content"] = (
                    str(first.get("content") or "") + "\n\n" + _TOOL_USAGE_RULES
                )
                request["messages"] = [first] + list(messages[1:])

        response = self._send_completion(request)
        payload = self._handle_response(response)

        # Цикл tool calling: пока LLM просит инструменты и лимит не исчерпан
        executed: List[Dict[str, Any]] = []
        tokens_acc: Optional[Dict[str, Any]] = payload.get("tokens")
        rounds = 0
        while payload.get("tool_calls") and rounds < self._max_tool_rounds:
            rounds += 1
            assistant_msg, tool_msgs, events = self._execute_tool_calls(payload)
            executed.extend(events)
            request["messages"] = (
                list(request.get("messages") or []) + [assistant_msg] + tool_msgs
            )
            response = self._send_completion(request)
            payload = self._handle_response(response)
            tokens_acc = self._merge_tokens(tokens_acc, payload.get("tokens"))

        payload["tokens"] = tokens_acc
        payload["tool_calls_executed"] = executed
        # Путь сохранённого файла (инструмент save_to_file news-сервера) —
        # UI показывает по нему кнопку «Открыть файл»
        payload["saved_file_path"] = self._extract_saved_file_path(executed)
        return payload

    @staticmethod
    def _extract_saved_file_path(
        events: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Путь из последнего успешного вызова save_to_file (или None).

        Результат инструмента — текст; у FastMCP это pretty-JSON
        ``{"result": "<путь>"}``, на всякий случай принимается и «сырой»
        путь. Строки с ошибками («Ошибка сохранения: …») путём не
        считаются.
        """
        for event in reversed(events):
            if event.get("name") != "save_to_file":
                continue
            result = str(event.get("result") or "")
            try:
                parsed = json.loads(result)
            except (TypeError, ValueError):
                parsed = None
            if (isinstance(parsed, dict)
                    and isinstance(parsed.get("result"), str)):
                result = parsed["result"]
            result = result.strip()
            if result and not result.lower().startswith("ошибка"):
                return result
        return None

    # ------------------------------------------------------------------
    # Tool calling: tools от MCP-серверов (напоминания, новости)
    # ------------------------------------------------------------------

    def _get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Собирает tools всех подключённых MCP-серверов (формат OpenAI).

        Для каждого клиента: кэш на ``_tools_ttl`` секунд, при промахе —
        tools/list по HTTP; сервер недоступен → его предыдущий кэш или
        пустой вклад (диалог живёт без его инструментов). Попутно
        строится маршрут вызовов «имя инструмента → клиент»
        (:attr:`_tool_routes`); при совпадении имён у разных серверов
        побеждает первый (предупреждение в логе).

        Returns:
            Список словарей ``{"type": "function", "function": {...}}``.
        """
        if (not self._tool_clients or self._mock_mode
                or not self._provider_tools_enabled):
            # Без клиентов, в mock-режиме или когда модель провайдера не
            # поддерживает tool calling — запрос уходит без tools
            return []
        now = time.monotonic()
        tools: List[Dict[str, Any]] = []
        routes: Dict[str, Any] = {}
        for client in self._tool_clients:
            cached = self._tools_cache.get(id(client))
            if cached is not None and now - cached[0] < self._tools_ttl:
                client_tools = cached[1]
            else:
                try:
                    mcp_tools = asyncio.run(client.list_tools())
                except Exception as exc:
                    logger.warning(
                        "Tools MCP-сервера недоступны (сервер не запущен?): %s",
                        exc,
                    )
                    client_tools = cached[1] if cached is not None else []
                else:
                    client_tools = [
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.input_schema
                                or {"type": "object", "properties": {}},
                            },
                        }
                        for tool in mcp_tools
                    ]
                    self._tools_cache[id(client)] = (now, client_tools)
                    logger.info(
                        "Tools MCP-сервера получены: %d шт.", len(client_tools)
                    )
            for tool in client_tools:
                name = tool["function"]["name"]
                if name in routes:
                    logger.warning(
                        "Инструмент %r есть у нескольких MCP-серверов — "
                        "используется первый.", name,
                    )
                    continue
                routes[name] = client
                tools.append(tool)
        self._tool_routes = routes
        return tools

    def _call_tool_via_mcp(self, name: str,
                           arguments: Dict[str, Any]) -> str:
        """Вызывает MCP-инструмент по HTTP (маршрут — по имени инструмента).

        Ошибки MCP (сервер недоступен, таймаут, isError) превращаются в
        JSON-строку с полем ``error`` — она уходит LLM как результат
        инструмента, чтобы модель могла объяснить проблему пользователю,
        а ход диалога не прерывался (по ТЗ: «не игнорировать ошибки
        MCP-вызовов», но и не ронять ответ).

        Args:
            name: имя инструмента (``add_reminder``, ``save_to_file``, …).
            arguments: аргументы из tool_calls.

        Returns:
            Текст результата (JSON) либо JSON с ошибкой.
        """
        client = self._tool_routes.get(name)
        if client is None:
            # Маршруты заполняются при сборке tools того же хода; попадание
            # сюда — LLM выдумала инструмент без tools → ошибка модели
            return json.dumps(
                {"error": f"Инструмент {name!r} не найден ни на одном "
                          f"MCP-сервере"},
                ensure_ascii=False,
            )
        try:
            return asyncio.run(client.call_tool(name, arguments))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Вызов инструмента %s не удался: %s", name, exc)
            return json.dumps(
                {"error": f"MCP-сервер недоступен: {exc}"},
                ensure_ascii=False,
            )

    def _execute_tool_calls(
        self, payload: Dict[str, Any]
    ) -> Tuple[Message, List[Message], List[Dict[str, Any]]]:
        """Выполняет все tool_calls текущего ответа.

        Args:
            payload: ответ :meth:`_handle_response` с ключом
                ``tool_calls`` (список ``{"id", "name", "arguments"}``).

        Returns:
            Кортеж: сообщение ассистента с tool_calls (добавляется в
            запрос перед результатами), список сообщений роли ``tool``,
            события для UI ``{"name", "arguments", "result"}``.
        """
        raw_calls = payload.get("tool_calls") or []
        assistant_msg: Message = {
            "role": "assistant",
            "content": str(payload.get("content") or ""),
        }
        # Thinking Mode: reasoning_content возвращается вместе с
        # tool_calls (по ТЗ). Если API отвергает это поле — _send_completion
        # повторит запрос без него (совместимость с версиями API).
        reasoning = payload.get("reasoning_content")
        if reasoning:
            assistant_msg["reasoning_content"] = str(reasoning)

        tool_msgs: List[Message] = []
        events: List[Dict[str, Any]] = []
        for call in raw_calls:
            name = str(call.get("name") or "")
            raw_arguments = call.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_arguments)  # type: ignore[arg-type]
                if not isinstance(arguments, dict):
                    arguments = {}
            except (TypeError, ValueError):
                arguments = {}
            result_text = self._call_tool_via_mcp(name, arguments)
            tool_msgs.append({
                "role": "tool",
                "tool_call_id": str(call.get("id") or ""),
                "content": result_text,
            })
            events.append({
                "name": name,
                "arguments": arguments,
                "result": result_text,
            })
            logger.info("Tool call: %s(%s)", name, arguments)
        # tool_calls уходит в API в исходном виде (id/type/function)
        assistant_msg["tool_calls"] = [
            {
                "id": str(call.get("id") or ""),
                "type": "function",
                "function": {
                    "name": str(call.get("name") or ""),
                    "arguments": str(call.get("arguments") or "{}"),
                },
            }
            for call in raw_calls
        ]
        return assistant_msg, tool_msgs, events

    def _send_completion(self, request: Dict[str, Any]) -> Any:
        """Отправляет chat.completions.create с повтором без reasoning_content.

        DeepSeek Thinking Mode: reasoning_content передаётся обратно по ТЗ,
        но некоторые версии API отвечают на него ошибкой 400 — тогда запрос
        повторяется один раз без этого поля (совместимость без потери хода).

        Args:
            request: параметры запроса (модель, messages, tools, ...).

        Returns:
            Объект ответа OpenAI SDK.

        Raises:
            ApiError: при ошибке аутентификации, сети, лимитов и т.п.
        """
        try:
            return self._client.chat.completions.create(**request)
        except Exception as exc:
            message = humanize_api_error(exc)
            if "reasoning_content" in message.lower() and any(
                isinstance(m, dict) and "reasoning_content" in m
                for m in request.get("messages") or []
            ):
                logger.warning(
                    "API отверг reasoning_content — повтор запроса без него."
                )
                retry = dict(request)
                retry["messages"] = self._strip_reasoning_content(
                    request["messages"]
                )
                try:
                    return self._client.chat.completions.create(**retry)
                except Exception as retry_exc:
                    raise ApiError(
                        humanize_api_error(retry_exc), original=retry_exc
                    )
            raise ApiError(message, original=exc)

    @staticmethod
    def _strip_reasoning_content(messages: List[Any]) -> List[Any]:
        """Убирает reasoning_content из сообщений запроса (fallback)."""
        cleaned: List[Any] = []
        for message in messages:
            if isinstance(message, dict) and "reasoning_content" in message:
                message = {
                    key: value for key, value in message.items()
                    if key != "reasoning_content"
                }
            cleaned.append(message)
        return cleaned

    @staticmethod
    def _merge_tokens(acc: Optional[Dict[str, Any]],
                      new: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Суммирует usage нескольких вызовов API (цикл tool_calls).

        Args:
            acc: накопленная статистика (None — ещё не было).
            new: usage очередного ответа (None — API не вернул).

        Returns:
            Объединённая статистика: числовые поля суммируются, флаг
            ``estimated`` остаётся True только если ВСЕ ответы оценочные.
        """
        if new is None:
            return acc
        if acc is None:
            return dict(new)
        merged = dict(acc)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens",
                    "cache_hit_tokens", "cache_miss_tokens"):
            merged[key] = (merged.get(key) or 0) + (new.get(key) or 0)
        merged["estimated"] = bool(acc.get("estimated")) and bool(
            new.get("estimated")
        )
        return merged

    def _handle_response(self, response: Any) -> Dict[str, Any]:
        """Обрабатывает ответ API и извлекает данные.

        Args:
            response: объект ответа openai (chat.completions).

        Returns:
            Словарь с ключами content, model, tokens, finish_reason;
            при запросе с tools добавляются ``tool_calls`` (список
            ``{"id", "name", "arguments"}``) и ``reasoning_content``
            (Thinking Mode), если они присутствуют в ответе.

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

        # Tool calling: вызовы инструментов, запрошенные LLM. Нормализуем
        # к словарям (аргументы остаются JSON-строкой — их парсит
        # _execute_tool_calls, в API уходят как есть).
        tool_calls: List[Dict[str, Any]] = []
        raw_tool_calls = getattr(message, "tool_calls", None) if message else None
        for call in raw_tool_calls or []:
            function = getattr(call, "function", None)
            tool_calls.append({
                "id": str(getattr(call, "id", "") or ""),
                "name": str(getattr(function, "name", "") or ""),
                "arguments": str(getattr(function, "arguments", "") or "{}"),
            })

        reasoning = getattr(message, "reasoning_content", None) if message else None

        payload = {
            "content": content,
            "model": getattr(response, "model", None) or self.model,
            "tokens": tokens,
            "finish_reason": getattr(choice, "finish_reason", None),
        }
        if tool_calls:
            payload["tool_calls"] = tool_calls
        if reasoning:
            payload["reasoning_content"] = str(reasoning)
        return payload

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
        params = {
            "model": self.model,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        # Выбор в переключателе стратегий тоже сохраняется (восстанавливается
        # при следующем запуске приложения).
        if self._strategy is not None:
            params["strategy"] = self._strategy_name
            params["branch_strategy"] = self._sub_strategy_name
        return params

    def _persist(self) -> None:
        """Сохраняет историю, параметры, статистику и артефакты.

        Сообщения и факты пишутся в активную ветку; ошибки хранилища не
        пробрасываются — агент продолжает работу в памяти.
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
                branch_id=self._active_branch_id,
                # {} — явно стереть факты, None — не трогать
                facts=self._facts_store.as_dict(),
                pinned=self._memory_manager.pins,
                working=self._memory_manager.working.as_dict()
                if self._memory_enabled else None,
                # Сообщения пишутся с profile_id: история изолирована
                # по профилям (другие профили той же ветки не трогаются)
                profile_id=self._active_profile_id(),
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
            state = self._storage.load_state(
                profile_id=self._active_profile_id()
            )
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
        if self._trimming_active():
            self._trim_history()

        # 4. Восстанавливаем накопленную статистику токенов
        self._restore_session_stats(state.get("stats"))

        # 5. Восстанавливаем конспект диалога (если он сохранён)
        self._restore_summary(state.get("summary"))

        # 6. Восстанавливаем факты и подстратегию активной ветки
        self._restore_facts(state.get("facts"))
        active = state.get("active_branch")
        if active is not None:
            try:
                self._active_branch_id = int(active)
            except (TypeError, ValueError):
                pass
        self._restore_branch_strategy()

        # 6б. Восстанавливаем рабочую память и закрепления активной ветки
        self._restore_working_memory(state.get("working_memory"))
        self._restore_pinned(state.get("pinned"), restored)

        # 7. Восстанавливаем выбор переключателя стратегий (режим веток)
        if self._strategy is not None:
            saved = str(params.get("strategy") or "").strip()
            if saved:
                try:
                    canonical = normalize_name(saved)
                except AgentConfigError:
                    canonical = None
                if canonical == BRANCHING:
                    self._strategy_name = BRANCHING
                elif canonical is not None:
                    self._strategy_name = self._sub_strategy_name

        if restored:
            logger.info(
                "Контекст восстановлен: %d сообщений, модель=%s, ветка=%s.",
                restored, self.model,
                self._branch_manager.get_active().get("name"),
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
            # Служебные запросы обновления фактов (sticky facts)
            "facts_requests": 0,
            "facts_prompt_tokens": 0,
            "facts_completion_tokens": 0,
            "facts_estimated_tokens": 0,
            "facts_cost_usd": 0.0,
            # Служебные запросы классификации памяти (memory layers)
            "memory_requests": 0,
            "memory_prompt_tokens": 0,
            "memory_completion_tokens": 0,
            "memory_estimated_tokens": 0,
            "memory_cost_usd": 0.0,
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
            session["facts_cost_usd"] = (
                round(float(session.get("facts_cost_usd", 0.0)), 6)
                if session["prices_set"] else None
            )
            session["memory_cost_usd"] = (
                round(float(session.get("memory_cost_usd", 0.0)), 6)
                if session["prices_set"] else None
            )
            facts_items = self._facts_store.items()
            facts_text = self._facts_store.to_text()
            memory = self._memory_stats_block(context_messages)
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
                    "memory_tokens": memory.get("total_tokens", 0),
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
                "strategy": {
                    "name": self._strategy_name,
                    "mode": self._strategy_name or "legacy_summary",
                    "sub_strategy": self._sub_strategy_name
                    if self._strategy_mode() else None,
                    "branching": self._strategy_name == BRANCHING,
                    "window": self._strategy._window() if self._strategy else None,
                    "available": list(AVAILABLE_STRATEGIES),
                    "error": self._facts_error,
                },
                "facts": {
                    "active": self._uses_facts(),
                    "enabled": bool(
                        self._strategy_settings.get("facts_enabled", True)
                    ),
                    "count": self._facts_store.count(),
                    "tokens": self._facts_store.token_count(self._counter),
                    "updated_at": self._facts_updated_at,
                    "error": self._facts_error,
                    "text": facts_text,
                    "items": [
                        {"category": category, "key": key, "value": value,
                         "updated_at": stamp}
                        for category, key, value, stamp in facts_items
                    ],
                },
                "branch": self._branch_manager.describe(),
                "memory": memory,
                "session": session,
                "counter": self._counter.name,
            }

    @property
    def session_stats(self) -> Dict[str, Any]:
        """Копия накопленной статистики сессии (только для чтения)."""
        with self._lock:
            return dict(self._session_stats)

    def _memory_stats_block(self, context_messages: List[Message]) -> Dict[str, Any]:
        """Срез статистики памяти для ``get_token_stats`` (обратносовместимо)."""
        if not self._memory_enabled:
            return {
                "enabled": False,
                "short_term": {"messages": 0, "pinned": 0, "tokens": 0},
                "working": {"items": 0, "tokens": 0},
                "long_term": {"items": 0, "sections": [], "tokens": 0},
                "pending": 0,
                "total_tokens": 0,
                "budget": 0,
                "truncated": False,
                "error": None,
            }
        report = self._memory_manager.report()
        pending_count = len(report.get("pending", []))
        budget = report.get("budget", {})
        long_count = int(report[LONG_TERM].get("count", 0))
        long_tokens = int(report[LONG_TERM].get("tokens", 0))
        working_count = int(report[WORKING].get("count", 0))
        working_tokens = int(report[WORKING].get("tokens", 0))
        pinned_count = int(report[SHORT_TERM].get("pinned_count", 0))
        short_messages = max(0, len(self._message_history) - 1)

        # Суммарные токены реально ушедших в запрос блоков памяти
        sent_tokens = 0
        for message in context_messages:
            content = message.get("content") or ""
            if message.get("role") == "system" and (
                content.startswith(LONG_HEADER)
                or content.startswith(WORKING_HEADER)
                or content.startswith(PINNED_HEADER)
            ):
                sent_tokens += self.count_tokens(content)

        return {
            "enabled": True,
            "short_term": {
                "messages": short_messages,
                "pinned": pinned_count,
                "tokens": sent_tokens if pinned_count else 0,
            },
            "working": {"items": working_count, "tokens": working_tokens},
            "long_term": {
                "items": long_count,
                "sections": list(report[LONG_TERM].get("sections", [])),
                "tokens": long_tokens,
            },
            "pending": pending_count,
            "total_tokens": sent_tokens,
            "budget": int(budget.get("max_tokens", 0)),
            "truncated": bool((working_tokens + long_tokens)
                              > int(budget.get("max_tokens", 0))),
            "error": self._memory_error,
        }

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
        """True, если сжатие контекста включено и доступно (не mock).

        В режиме стратегий суммаризация не используется: она осталась только
        для обратной совместимости (когда настройки summary переданы явно,
        а стратегия не выбрана).
        """
        if self._strategy is not None:
            return False
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

    @property
    def strategy_settings(self) -> Dict[str, Any]:
        """Копия настроек стратегий управления контекстом (только для чтения)."""
        with self._lock:
            return dict(self._strategy_settings or {})

    # ------------------------------------------------------------------
    # Стратегии управления контекстом
    # ------------------------------------------------------------------

    def available_strategies(self) -> List[str]:
        """Список стратегий для переключателя (sliding_window, sticky_facts,
        branching)."""
        return list(AVAILABLE_STRATEGIES)

    def get_strategy(self) -> Dict[str, Any]:
        """Текущее состояние стратегии для UI и тестов.

        Returns:
            Словарь {"name", "mode", "sub_strategy", "branching", "window",
            "facts", "facts_count", "facts_tokens", "branch", "available",
            "error"}.
        """
        with self._lock:
            return {
                "name": self._strategy_name,
                "mode": self._strategy_name or "legacy_summary",
                "sub_strategy": (self._sub_strategy_name
                                 if self._strategy_mode() else None),
                "branching": self._strategy_name == BRANCHING,
                "window": self._strategy._window() if self._strategy else None,
                "facts": self._uses_facts(),
                "facts_count": self._facts_store.count(),
                "facts_tokens": self._facts_store.token_count(self._counter),
                "branch": self._branch_manager.describe(),
                "available": list(AVAILABLE_STRATEGIES),
                "error": self._facts_error,
            }

    def set_strategy(self, name: str) -> Dict[str, Any]:
        """Переключает стратегию управления контекстом.

        Выбор ``sliding_window``/``sticky_facts`` меняет подстратегию активной
        ветки и запоминается для неё; выбор ``branching`` включает режим веток
        и оставляет подстратегию ветки как есть. При переходе на стратегию с
        фактами выполняется начальное извлечение фактов по текущему диалогу.

        Args:
            name: имя стратегии.

        Returns:
            Новое состояние стратегии (см. :meth:`get_strategy`).

        Raises:
            AgentConfigError: если имя стратегии неизвестно.
        """
        with self._lock:
            canonical = normalize_name(name)
            self._strategy_name = canonical
            if canonical != BRANCHING:
                self._sub_strategy_name = canonical
            self._strategy_settings["name"] = canonical
            self._strategy_settings["branch_strategy"] = self._sub_strategy_name
            self._strategy = self._make_strategy()
            self._persist_branch_strategy()
            if self._uses_facts() and self._dialog_messages():
                # Начальное извлечение: иначе блок фактов будет пустым
                # до следующего хода диалога.
                self.update_facts_now(force=True)
            self._persist()
            logger.info(
                "Стратегия контекста: %s (подстратегия ветки: %s).",
                canonical, self._sub_strategy_name,
            )
            return self.get_strategy()

    def _persist_branch_strategy(self) -> None:
        """Запоминает подстратегию контекста у активной ветки."""
        branch = self._branch_manager.get(self._active_branch_id)
        if branch is None:
            return
        branch["strategy"] = self._sub_strategy_name
        if not self._branch_manager.persistence or self._storage is None:
            return
        try:
            self._storage.save_branch(
                name=str(branch.get("name") or "ветка"),
                parent_id=branch.get("parent_id"),
                forked_at=int(branch.get("forked_at") or 0),
                strategy=self._sub_strategy_name,
                branch_id=self._active_branch_id,
            )
        except Exception as exc:
            logger.warning("Не удалось сохранить стратегию ветки: %s", exc)

    def _after_turn(self, user_text: str,
                    assistant_text: str) -> Optional[Dict[str, Any]]:
        """Даёт стратегии обновить свои артефакты после хода.

        Returns:
            Результат стратегии либо None, если стратегии нет.
        """
        if self._strategy is None:
            return None
        call_service = None if self._mock_mode else self._call_service_api
        try:
            info = self._strategy.after_turn(
                user_text, assistant_text, self._dialog_messages(),
                self._artifacts(), call_service,
            )
        except Exception as exc:  # стратегия не должна ломать ход
            logger.warning(
                "Ошибка обновления артефактов стратегии %s: %s",
                self._strategy_name, exc,
            )
            self._facts_error = str(exc)
            return {"updated": False, "error": str(exc), "tokens": None}

        error = info.get("error")
        self._facts_error = str(error) if error else None
        if info.get("updated"):
            self._facts_updated_at = datetime.now().isoformat(timespec="seconds")
        self._account_facts(info)
        return info

    def _account_facts(self, info: Optional[Dict[str, Any]]) -> None:
        """Учитывает служебный запрос фактов отдельными счётчиками."""
        if not info:
            return
        tokens = info.get("tokens")
        if not isinstance(tokens, dict):
            return
        try:
            prompt = int(tokens.get("prompt_tokens") or 0)
            completion = int(tokens.get("completion_tokens") or 0)
        except (TypeError, ValueError):
            return
        estimated = (
            bool(tokens.get("estimated", True))
            or tokens.get("prompt_tokens") is None
            or tokens.get("completion_tokens") is None
        )
        if estimated:
            self._session_stats["facts_estimated_tokens"] += prompt + completion
        else:
            self._session_stats["facts_prompt_tokens"] += prompt
            self._session_stats["facts_completion_tokens"] += completion
        cost = self._compute_cost(
            prompt, completion, model=tokens.get("model") or self.model
        )
        if cost is not None:
            self._session_stats["facts_cost_usd"] = round(
                float(self._session_stats.get("facts_cost_usd", 0.0)) + cost, 6
            )
        self._session_stats["facts_requests"] += 1

    def _classify_memory_after_turn(self, user_text: str,
                                    assistant_text: str) -> None:
        """Предлагает кандидатов памяти после успешного хода (LLM).

        Выполняет один служебный запрос классификации (``MEMORY_MARKER``) и
        кладёт кандидатов в ``pending`` — в слои они **не** попадают до
        явного принятия. Ошибка классификации не ломает ход: WARNING,
        ``_memory_error``, повтор на следующем ходу.
        """
        if not self._memory_enabled or self._mock_mode:
            return
        if not self._memory_settings.get("propose_llm", True):
            return
        if self._client is None:
            return
        context_hint = f"Пользователь: {user_text}\nАссистент: {assistant_text}"
        try:
            result = MemoryClassifier.classify_llm(
                user_text, self._call_service_api, self._memory_settings,
                context_hint=context_hint,
            )
        except Exception as exc:  # классификация не должна ломать ход
            self._memory_error = str(exc)
            logger.warning("Классификация памяти не удалась: %s", exc)
            return
        self._account_memory_classification(result.get("tokens"))
        if result.get("error"):
            self._memory_error = str(result["error"])
            logger.warning("Классификация памяти: %s", result["error"])
            return
        items: List[Dict[str, Any]] = []
        for layer in LAYERS:
            for candidate in result.get(layer, []):
                if not isinstance(candidate, dict):
                    continue
                item = dict(candidate)
                item["layer"] = layer
                items.append(item)
        if items:
            dialog = self._dialog_messages()
            source_message = max(0, len(dialog) - 2)  # индекс user-сообщения
            added = self._memory_manager.pending.add(
                items, source_message=source_message
            )
            if added:
                self._memory_updated_at = datetime.now().isoformat(timespec="seconds")

    def _account_memory_classification(self, tokens: Optional[Dict[str, Any]]) -> None:
        """Учитывает служебный запрос классификации отдельными счётчиками."""
        if not isinstance(tokens, dict):
            return
        try:
            prompt = int(tokens.get("prompt_tokens") or 0)
            completion = int(tokens.get("completion_tokens") or 0)
        except (TypeError, ValueError):
            return
        estimated = (
            bool(tokens.get("estimated", True))
            or tokens.get("prompt_tokens") is None
            or tokens.get("completion_tokens") is None
        )
        if estimated:
            self._session_stats["memory_estimated_tokens"] += prompt + completion
        else:
            self._session_stats["memory_prompt_tokens"] += prompt
            self._session_stats["memory_completion_tokens"] += completion
        cost = self._compute_cost(
            prompt, completion, model=tokens.get("model") or self.model
        )
        if cost is not None:
            self._session_stats["memory_cost_usd"] = round(
                float(self._session_stats.get("memory_cost_usd", 0.0)) + cost, 6
            )
        self._session_stats["memory_requests"] += 1

    def _restore_facts(self, raw: Optional[Dict[str, Any]]) -> None:
        """Восстанавливает факты из хранилища (устойчиво к мусору)."""
        if raw is None:
            return
        self._facts_store = self._new_facts_store(raw)
        self._facts = self._facts_store.data
        stamps = [
            stamp
            for per_category in self._facts_store.updated_at.values()
            for stamp in per_category.values()
        ]
        self._facts_updated_at = max(stamps) if stamps else None

    def _restore_working_memory(self, raw: Optional[Dict[str, Any]]) -> None:
        """Восстанавливает рабочую память активной ветки (устойчиво к мусору)."""
        if not self._memory_enabled:
            return
        if raw is None:
            return
        self._memory_manager.working.from_dict(raw)

    def _restore_pinned(self, pinned: Optional[List[Dict[str, Any]]],
                        dialog_length: int) -> None:
        """Восстанавливает закрепления (индексы вне диапазона игнорируются)."""
        if not self._memory_enabled:
            return
        self._memory_manager.pins.clear()
        for entry in (pinned or []):
            if not isinstance(entry, dict):
                continue
            try:
                index = int(entry.get("index", -1))
            except (TypeError, ValueError):
                continue
            if 0 <= index < dialog_length:
                self._memory_manager.pins[index] = entry.get("note")
            else:
                logger.warning(
                    "Закрепление №%d вне диапазона диалога (%d) — игнорирую.",
                    index, dialog_length,
                )

    def _restore_branch_strategy(self) -> None:
        """Применяет подстратегию контекста, сохранённую у активной ветки."""
        branch = self._branch_manager.get(self._active_branch_id)
        if branch is None:
            return
        raw = str(branch.get("strategy") or "").strip()
        if not raw:
            return
        try:
            sub = normalize_name(raw)
        except AgentConfigError:
            logger.warning("Неизвестная стратегия ветки %r — игнорирую.", raw)
            return
        if sub == BRANCHING:
            sub = SLIDING_WINDOW
        self._sub_strategy_name = sub
        if self._strategy is not None:
            if self._strategy_name != BRANCHING:
                self._strategy_name = sub
            self._strategy = self._make_strategy()

    # ------------------------------------------------------------------
    # Блок фактов (sticky facts)
    # ------------------------------------------------------------------

    def get_facts(self) -> Dict[str, Any]:
        """Возвращает факты диалога и их состояние.

        Returns:
            Словарь {"facts", "items", "text", "count", "tokens",
            "updated_at", "active", "enabled", "error"}.
        """
        with self._lock:
            return {
                "facts": {cat: dict(values)
                          for cat, values in self._facts_store.data.items()},
                "items": [
                    {"category": category, "key": key, "value": value,
                     "updated_at": stamp}
                    for category, key, value, stamp in self._facts_store.items()
                ],
                "text": self._facts_store.to_text(),
                "count": self._facts_store.count(),
                "tokens": self._facts_store.token_count(self._counter),
                "updated_at": self._facts_updated_at,
                "active": self._uses_facts(),
                "enabled": bool(self._strategy_settings.get("facts_enabled", True)),
                "error": self._facts_error,
            }

    def remember(self, key: str, value: str,
                 category: str = "decisions") -> Dict[str, Any]:
        """Запоминает факт вручную."""
        with self._lock:
            ok = self._facts_store.remember(key, value, category)
            if ok:
                self._apply_facts_limits()
                self._facts_updated_at = datetime.now().isoformat(timespec="seconds")
                self._facts_error = None
                self._persist()
            result = self.get_facts()
            result["ok"] = ok
            return result

    def forget(self, key: str) -> Dict[str, Any]:
        """Удаляет факт по ключу."""
        with self._lock:
            ok = self._facts_store.forget(key)
            if ok:
                self._facts_updated_at = datetime.now().isoformat(timespec="seconds")
                self._persist()
            result = self.get_facts()
            result["ok"] = ok
            return result

    def clear_facts(self) -> Dict[str, Any]:
        """Полностью очищает блок фактов (история не трогается)."""
        with self._lock:
            self._facts_store.clear()
            self._facts_error = None
            self._facts_updated_at = None
            self._persist()
            return self.get_facts()

    def update_facts_now(self, force: bool = True) -> Dict[str, Any]:
        """Принудительно обновляет факты по текущему диалогу.

        Используется кнопкой «Обновить факты» в окне «Контекст» и при переходе
        на стратегию ``sticky_facts``.

        Args:
            force: True — извлечь факты из всего текущего диалога (иначе —
                только из последнего обмена, если он есть).

        Returns:
            Словарь {"updated", "error", "count"}.
        """
        with self._lock:
            if not self._uses_facts():
                return {
                    "updated": False,
                    "count": self._facts_store.count(),
                    "error": "Факты не используются в этой стратегии.",
                }
            dialog = self._dialog_messages()
            if not dialog:
                return {
                    "updated": False,
                    "count": self._facts_store.count(),
                    "error": "Диалог пуст — извлекать факты не из чего.",
                }

            if force:
                digest = "\n".join(
                    f"{'Пользователь' if m.get('role') == 'user' else 'Ассистент'}: "
                    f"{m.get('content', '')}"
                    for m in dialog[-40:]
                )
                user_text = ("Извлеки факты из текущего диалога:\n" + digest)
                assistant_text = "(начальное извлечение фактов)"
            else:
                last_user = ""
                last_assistant = ""
                for message in reversed(dialog):
                    if not last_assistant and message.get("role") == "assistant":
                        last_assistant = str(message.get("content") or "")
                    elif not last_user and message.get("role") == "user":
                        last_user = str(message.get("content") or "")
                    if last_user and last_assistant:
                        break
                user_text = last_user
                assistant_text = last_assistant

            call_service = None if self._mock_mode else self._call_service_api
            try:
                info = self._strategy.after_turn(
                    user_text, assistant_text, dialog,
                    self._artifacts(), call_service,
                ) if self._strategy is not None else {}
            except Exception as exc:
                self._facts_error = str(exc)
                return {"updated": False, "count": self._facts_store.count(),
                        "error": str(exc)}

            error = info.get("error")
            self._facts_error = str(error) if error else None
            if info.get("updated"):
                self._facts_updated_at = datetime.now().isoformat(timespec="seconds")
            self._account_facts(info)
            self._persist()
            return {
                "updated": bool(info.get("updated")),
                "count": self._facts_store.count(),
                "error": self._facts_error,
            }

    def _apply_facts_limits(self) -> int:
        """Применяет лимиты фактов (число пар и токены)."""
        return self._facts_store.limit(
            counter=self._counter,
            max_items=self._setting_int("facts_max_items", 50),
            max_tokens=self._setting_int("facts_max_tokens", 300),
        )

    def _new_facts_store(self, raw: Any) -> FactsStore:
        """Создаёт хранилище фактов из сериализованных данных."""
        if isinstance(raw, FactsStore):
            clone = FactsStore(
                max_items=self._setting_int("facts_max_items", 50),
                max_tokens=self._setting_int("facts_max_tokens", 300),
                counter=self._counter,
            )
            clone.from_dict(raw.as_dict())
            return clone
        payload = raw if isinstance(raw, dict) else {}
        if "data" in payload or "updated_at" in payload:
            data = payload.get("data")
            updated_at = payload.get("updated_at")
        else:
            data = payload
            updated_at = None
        return FactsStore(
            data=data if isinstance(data, dict) else {},
            updated_at=updated_at if isinstance(updated_at, dict) else None,
            max_items=self._setting_int("facts_max_items", 50),
            max_tokens=self._setting_int("facts_max_tokens", 300),
            counter=self._counter,
        )

    # ------------------------------------------------------------------
    # Ветки диалога (branching)
    # ------------------------------------------------------------------

    def list_branches(self) -> List[Dict[str, Any]]:
        """Список веток (для панели «Ветки»)."""
        with self._lock:
            return self._branch_manager.list_branches()

    def get_active_branch(self) -> Dict[str, Any]:
        """Описание активной ветки."""
        with self._lock:
            return self._branch_manager.describe()

    def checkpoint(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Создаёт checkpoint — ветку от конца текущего диалога."""
        return self.create_branch(from_index=None, name=name)

    def create_branch(self, from_index: Optional[int] = None,
                      name: Optional[str] = None,
                      strategy: Optional[str] = None) -> Dict[str, Any]:
        """Создаёт ветку от текущего места или от выбранного сообщения.

        Args:
            from_index: индекс сообщения диалога (без system), от которого
                ответвляемся; None — checkpoint в конце диалога.
            name: имя ветки (None — автоимя).
            strategy: подстратегия ветки (None — как у текущей).

        Returns:
            Словарь {"branch", "strategy", "error"}; при ошибке валидации
            возвращается заполненное поле "error".
        """
        with self._lock:
            working_snapshot = (
                self._memory_manager.working.as_dict()
                if self._memory_enabled else None
            )
            try:
                self._persist()  # фиксируем текущую ветку до развилки
                self._branch_manager.remember_messages(
                    self._active_branch_id, self._message_history
                )
                self._branch_manager.remember_facts(
                    self._active_branch_id, self._facts_store
                )
                result = self._branch_manager.create(
                    self._message_history, self._artifacts(),
                    from_index=from_index, name=name, strategy=strategy,
                )
            except AgentConfigError as exc:
                return {"branch": None, "strategy": self._strategy_name,
                        "error": str(exc)}

            self._apply_branch_state(
                result["branch"], result["messages"],
                result["artifacts"].get("facts"),
            )
            # Новая ветка копирует рабочую память префикса.
            if working_snapshot is not None:
                self._memory_manager.working.from_dict(working_snapshot)
            self._persist()
            return {
                "branch": self._branch_manager.get_active(),
                "strategy": self._strategy_name,
                "error": None,
            }

    def switch_branch(self, branch_id: int) -> Dict[str, Any]:
        """Переключается на другую ветку (сообщения и артефакты — свои).

        Returns:
            Словарь {"branch", "strategy", "error"}.
        """
        with self._lock:
            target = self._branch_manager.get(branch_id)
            if target is None:
                return {"branch": None, "strategy": self._strategy_name,
                        "error": f"Ветка id={branch_id} не найдена."}
            try:
                self._persist()
                self._branch_manager.remember_messages(
                    self._active_branch_id, self._message_history
                )
                self._branch_manager.remember_facts(
                    self._active_branch_id, self._facts_store
                )
                self._branch_manager.switch(branch_id)
            except AgentConfigError as exc:
                return {"branch": None, "strategy": self._strategy_name,
                        "error": str(exc)}

            messages = self._branch_manager.load_messages(branch_id)
            facts_raw = self._branch_manager.load_facts(branch_id)
            self._apply_branch_state(target, messages, facts_raw)
            self._persist()
            return {
                "branch": self._branch_manager.get_active(),
                "strategy": self._strategy_name,
                "error": None,
            }

    def rename_branch(self, branch_id: int, name: str) -> Dict[str, Any]:
        """Переименовывает ветку."""
        with self._lock:
            try:
                branch = self._branch_manager.rename(branch_id, name)
            except AgentConfigError as exc:
                return {"branch": None, "error": str(exc)}
            return {"branch": branch, "error": None}

    def delete_branch(self, branch_id: int,
                      cascade: bool = False) -> Dict[str, Any]:
        """Удаляет ветку (при ``cascade`` — вместе с дочерними)."""
        with self._lock:
            try:
                result = self._branch_manager.delete(branch_id, cascade=cascade)
            except AgentConfigError as exc:
                return {"deleted": [], "branch": None, "error": str(exc)}
            self._active_branch_id = self._branch_manager.active_id()
            if int(branch_id) in list(result.get("deleted") or []):
                messages = self._branch_manager.load_messages(self._active_branch_id)
                facts_raw = self._branch_manager.load_facts(self._active_branch_id)
                self._apply_branch_state(
                    self._branch_manager.get_active(), messages, facts_raw
                )
            self._persist()
            return {
                "deleted": list(result.get("deleted") or []),
                "branch": self._branch_manager.get_active(),
                "error": None,
            }

    def _apply_branch_state(self, branch: Dict[str, Any],
                            messages: Optional[List[Message]],
                            facts: Any) -> None:
        """Применяет состояние ветки к агенту (история, факты, стратегия, память)."""
        self._active_branch_id = int(branch.get("id") or self._active_branch_id)
        if messages:
            self._message_history = [
                {"role": str(m.get("role", "")),
                 "content": str(m.get("content", ""))}
                for m in messages
            ]
        else:
            self._init_history()
        self._facts_store = self._new_facts_store(facts)
        self._facts = self._facts_store.data
        self._facts_error = None
        self._facts_updated_at = None
        self._restore_branch_strategy()
        # Рабочая память принадлежит ветке: загружаем её копию (или очищаем).
        if self._memory_enabled and self._storage is not None:
            try:
                raw = self._storage.load_branch_working(self._active_branch_id)
            except Exception as exc:
                logger.warning("Не удалось прочитать рабочую память ветки: %s", exc)
                raw = None
            self._memory_manager.working.from_dict(raw or {})
        else:
            self._memory_manager.working.clear()
        # Закрепления привязаны к индексам сообщений конкретной ветки —
        # при переключении сбрасываем (источник истины — активная ветка).
        self._memory_manager.pins.clear()
        # Сообщения ветки — источник истины для её артефактов
        self._branch_manager.remember_messages(self._active_branch_id,
                                               self._message_history)

    # ------------------------------------------------------------------
    # Модель памяти (short_term | working | long_term)
    # ------------------------------------------------------------------

    def get_memory_settings(self) -> Dict[str, Any]:
        """Копия настроек модели памяти (см. ``config.get_memory_settings``)."""
        with self._lock:
            return dict(self._memory_settings)

    def save_to_memory(self, layer: str, text: str, *, key: Optional[str] = None,
                       section: Optional[str] = None, kind: Optional[str] = None,
                       source: str = "manual",
                       message_index: Optional[int] = None) -> Dict[str, Any]:
        """Сохраняет ``text`` в выбранный слой через единый роутер.

        Единственная точка записи в слои (кнопки, окно «Память…», команды,
        подтверждение предложений). Ошибки не бросаются.

        Returns:
            ``{"ok", "layer", "key", "value", "section", "error"}``.
        """
        with self._lock:
            result = self._memory_manager.route(
                layer, text, key=key, section=section, kind=kind,
                source=source, message_index=message_index,
                branch_id=self._active_branch_id,
            )
            if result.get("ok"):
                self._memory_error = None
                self._memory_updated_at = datetime.now().isoformat(timespec="seconds")
                self._persist()
            return result

    def pin_message(self, index: int, note: Optional[str] = None) -> Dict[str, Any]:
        """Закрепляет сообщение (краткосрочная память)."""
        with self._lock:
            result = self._memory_manager.pin(int(index), note=note)
            self._persist()
            return result

    def unpin_message(self, index: int) -> Dict[str, Any]:
        """Открепляет сообщение."""
        with self._lock:
            result = self._memory_manager.unpin(int(index))
            self._persist()
            return result

    def list_pinned(self) -> List[Dict[str, Any]]:
        """Список закреплённых сообщений ``{index, note}``."""
        with self._lock:
            return self._memory_manager.list_pinned()

    def get_working_memory(self) -> Dict[str, Any]:
        """Рабочая память активной ветки (task + items)."""
        with self._lock:
            return self._memory_manager.working.as_dict()

    def set_task_field(self, field: str, value: Any) -> Dict[str, Any]:
        """Устанавливает поле задачи рабочей памяти."""
        with self._lock:
            result = self._memory_manager.working_set_field(field, value)
            self._persist()
            return result

    def add_working_item(self, key: str, value: str,
                         kind: str = "note") -> Dict[str, Any]:
        """Добавляет элемент рабочей памяти (``decision``/``artifact``/``note``)."""
        return self.save_to_memory(
            WORKING, value, key=key, kind=kind, source="manual",
        )

    def clear_working_memory(self) -> Dict[str, Any]:
        """Очищает рабочую память активной ветки."""
        with self._lock:
            result = self._memory_manager.working_clear()
            self._persist()
            return result

    def forget_working_item(self, key: str) -> Dict[str, Any]:
        """Удаляет элемент рабочей памяти по ключу."""
        with self._lock:
            result = self._memory_manager.working_forget(key)
            self._persist()
            return result

    def start_new_task(self,
                       carry_to_long: Optional[List[str]] = None) -> Dict[str, Any]:
        """«Новая задача»: очищает рабочую память.

        Args:
            carry_to_long: ключи элементов (``decision``), которые переносятся
                в ``long_term.decisions`` перед очисткой.

        Returns:
            ``{"ok", "carried", "error"}``.
        """
        with self._lock:
            result = self._memory_manager.start_new_task(carry_to_long)
            self._persist()
            return result

    def get_long_memory(self, section: Optional[str] = None) -> List[Dict[str, Any]]:
        """Записи долговременной памяти (``section`` None — все разделы)."""
        with self._lock:
            return self._memory_manager.long.list_items(section)

    def remember_forever(self, section: str, key: str,
                         value: str) -> Dict[str, Any]:
        """Сохраняет запись в долговременную память."""
        return self.save_to_memory(
            LONG_TERM, value, key=key, section=section, source="manual",
        )

    def forget_forever(self, section: str, key: str) -> Dict[str, Any]:
        """Удаляет запись из долговременной памяти."""
        with self._lock:
            result = self._memory_manager.long_forget(section, key)
            return result

    def clear_long_memory(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Очищает раздел (или всю долговременную память, если ``section`` None)."""
        with self._lock:
            if section is None:
                return self._memory_manager.long_clear_all()
            return self._memory_manager.long_clear_section(section)

    def export_long_memory(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Экспортирует долговременную память в Markdown."""
        with self._lock:
            target = path or self._memory_settings.get("export_path")
            if not target:
                return {"ok": False, "error": "Не задан путь экспорта."}
            from pathlib import Path as _Path
            from .config import PROJECT_ROOT as _PROJECT_ROOT
            p = _Path(str(target))
            if not p.is_absolute():
                p = _PROJECT_ROOT / p
            return self._memory_manager.long_export(p)

    def import_long_memory(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Импортирует записи из Markdown."""
        with self._lock:
            if not path:
                return {"ok": False, "error": "Не задан путь импорта."}
            from pathlib import Path as _Path
            p = _Path(str(path))
            return self._memory_manager.long_import(p)

    def classify_message(self, index: int) -> Dict[str, Any]:
        """Предложения по одному сообщению (локальные правила + LLM).

        Используется для предзаполнения диалога сохранения. В слои ничего не
        пишется.
        """
        with self._lock:
            dialog = self._dialog_messages()
            if not (0 <= int(index) < len(dialog)):
                return {
                    SHORT_TERM: [], WORKING: [], LONG_TERM: [],
                    "error": f"Сообщение №{index} не найдено.",
                }
            text = str(dialog[int(index)].get("content") or "")
            call_service = (
                None if self._mock_mode or not self._memory_enabled
                else self._call_service_api
            )
            result = MemoryClassifier.classify(
                text, call_service, self._memory_settings,
            )
            return {
                SHORT_TERM: result[SHORT_TERM],
                WORKING: result[WORKING],
                LONG_TERM: result[LONG_TERM],
                "error": result.get("error"),
            }

    def get_pending_memory(self) -> List[Dict[str, Any]]:
        """Список предложений классификации (pending)."""
        with self._lock:
            return self._memory_manager.pending_list()

    def accept_pending(self, pending_id: int,
                       layer: Optional[str] = None) -> Dict[str, Any]:
        """Принимает предложение и пишет в выбранный слой."""
        with self._lock:
            result = self._memory_manager.pending_accept(int(pending_id), layer)
            if result.get("ok"):
                self._memory_error = None
                self._persist()
            return result

    def reject_pending(self, pending_id: int) -> Dict[str, Any]:
        """Отклоняет предложение (без записи в слой)."""
        with self._lock:
            return self._memory_manager.pending_reject(int(pending_id))

    def clear_pending_memory(self) -> Dict[str, Any]:
        """Отклоняет все предложения."""
        with self._lock:
            return self._memory_manager.pending_clear()

    def get_memory_report(self) -> Dict[str, Any]:
        """«Какие данные в каком слое»: отчёт + бюджет + pending."""
        with self._lock:
            report = self._memory_manager.report()
            report["enabled"] = self._memory_enabled
            report[SHORT_TERM]["messages"] = max(0, len(self._message_history) - 1)
            report["error"] = self._memory_error
            report["updated_at"] = self._memory_updated_at
            return report

    def preview_context(self) -> List[Dict[str, Any]]:
        """Реальный список сообщений будущего запроса с пометкой слоя.

        Returns:
            Список ``{"layer", "role", "content", "tokens"}``; слой для
            диагностики: ``system``, ``long_term``, ``working``, ``pinned``,
            ``short_term``.
        """
        with self._lock:
            result: List[Dict[str, Any]] = []
            for message in self._build_context_messages():
                role = message.get("role", "")
                content = str(message.get("content") or "")
                layer = "short_term"
                if role == "system":
                    layer = "system"
                    if content.startswith(LONG_HEADER):
                        layer = LONG_TERM
                    elif content.startswith(WORKING_HEADER):
                        layer = WORKING
                    elif content.startswith(PINNED_HEADER):
                        layer = "pinned"
                result.append({
                    "layer": layer,
                    "role": role,
                    "content": content,
                    "tokens": self.count_tokens(content),
                })
            return result

    def clear_memory(self, layer: str) -> Dict[str, Any]:
        """Явная очистка слоя.

        Для ``short_term`` — закрепления (сообщения очищает ``clear_history``),
        для ``working`` — рабочая память, для ``long_term`` — вся
        долговременная память.
        """
        with self._lock:
            try:
                canonical = normalize_layer(layer)
            except MemoryError as exc:
                return {"ok": False, "error": str(exc)}
            if canonical == SHORT_TERM:
                self._memory_manager.pins.clear()
                self._persist()
                return {"ok": True, "layer": canonical}
            if canonical == WORKING:
                self._memory_manager.working.clear()
                self._persist()
                return {"ok": True, "layer": canonical}
            return {"ok": True, "layer": canonical,
                    **self._memory_manager.long_clear_all()}

    # ------------------------------------------------------------------
    # Подсистема задач (FSM): план → шаги → проверка
    # ------------------------------------------------------------------
    #
    # FSM (task_fsm.py) управляет переходами и сохранением состояния,
    # агент — бизнес-логикой LLM: сформировать план, выполнить шаг,
    # проверить результат. Всё состояние задачи живёт в SQLite
    # (task_repository.py), поэтому пауза/перезапуск не теряют контекст.

    @property
    def task_repository(self) -> Optional[TaskRepository]:
        """Репозиторий задач (создаётся лениво; None — хранилища нет)."""
        if self._task_repo is None and self._storage is not None:
            self._task_repo = TaskRepository(self._storage.db_path)
        return self._task_repo

    @property
    def active_task(self) -> Optional[Task]:
        """Текущая задача, привязанная к FSM (None — задачи нет)."""
        with self._lock:
            return self._fsm.task if self._fsm is not None else None

    def start_task(self, description: str,
                   mode: TaskMode = TaskMode.MANUAL) -> Dict[str, Any]:
        """Создаёт задачу и запускает FSM (первый этап — планирование).

        Правило «одна активная задача на профиль» enforced репозиторием:
        при наличии активной задачи возвращается ошибка.

        Args:
            description: описание задачи.
            mode: режим работы (auto/manual).

        Returns:
            ``{"task": Task|None, "acted": bool, "auto": bool, "error": str|None}``.
            ``auto=True`` означает, что после выполненного этапа задача
            продолжает работать сама (авто-режим) — вызывающий должен
            вызвать :meth:`continue_task` ещё раз.
        """
        with self._lock:
            repo = self.task_repository
            if repo is None:
                return self._task_result(None, error="Задачи недоступны: "
                                                     "хранилище не подключено.")
            if not description or not description.strip():
                return self._task_result(None, error="Пустое описание задачи.")
            try:
                task = repo.create_task(
                    self._active_profile_id() or 0,
                    description.strip(), mode,
                )
            except ActiveTaskExistsError as exc:
                return self._task_result(None, error=str(exc))
            self._fsm = TaskFSM(task, repo,
                                 max_retries=self._task_max_retries)
            self._fsm.transition(TaskStage.PLANNING, "start_task")
            result = self._advance_task(task)
            return self._task_result(task, **result)

    def continue_task(self, task_id: Optional[int] = None) -> Dict[str, Any]:
        """Выполняет следующий этап задачи для текущей стадии FSM.

        Вызывается после паузы, подтверждения или (в авто-режиме) в цикле —
        пока не встретится ожидание пользователя/пауза/конечная стадия.
        Один вызов = один этап (план / один шаг / валидация), чтобы UI
        оставался отзывчивым и пауза работала между шагами.

        Args:
            task_id: идентификатор задачи; None — текущая/активная задача.

        Returns:
            ``{"task": Task|None, "acted": bool, "auto": bool, "error": str|None}``.
        """
        with self._lock:
            task, error = self._ensure_fsm(task_id)
            if error:
                return self._task_result(task, error=error)
            assert task is not None
            result = self._advance_if_working(task)
            return self._task_result(task, **result)

    def run_task(self, task_id: Optional[int] = None,
                 max_steps: int = 50) -> Dict[str, Any]:
        """Прогоняет задачу до остановки (ожидание/пауза/конец), без UI.

        Удобно для тестов и авто-режима «под ключ»: циклически вызывает
        :meth:`continue_task`, пока задача продолжает работать сама.

        Args:
            task_id: идентификатор задачи; None — текущая.
            max_steps: защитный лимит итераций.

        Returns:
            Результат последнего вызова :meth:`continue_task`.
        """
        result: Dict[str, Any] = self.continue_task(task_id)
        steps = 0
        while result.get("auto") and steps < max_steps:
            result = self.continue_task(task_id)
            steps += 1
        return result

    def pause_task(self) -> Dict[str, Any]:
        """Ставит активную задачу на паузу (стадия сохраняется в БД)."""
        with self._lock:
            if self._fsm is None:
                return self._task_result(None, error="Активной задачи нет.")
            try:
                self._fsm.pause()
            except InvalidTransitionError as exc:
                return self._task_result(self._fsm.task, error=str(exc))
            return self._task_result(self._fsm.task)

    def resume_task(self) -> Dict[str, Any]:
        """Продолжает задачу после паузы и выполняет следующий этап."""
        with self._lock:
            if self._fsm is None:
                return self._task_result(None, error="Активной задачи нет.")
            task = self._fsm.task
            try:
                self._fsm.resume()
            except InvalidTransitionError as exc:
                return self._task_result(task, error=str(exc))
            result = self._advance_if_working(task)
            return self._task_result(task, **result)

    def cancel_task(self) -> Dict[str, Any]:
        """Отменяет активную задачу (стадия CANCELLED, событие в журнале)."""
        with self._lock:
            if self._fsm is None:
                return self._task_result(None, error="Активной задачи нет.")
            try:
                self._fsm.cancel()
            except InvalidTransitionError as exc:
                return self._task_result(self._fsm.task, error=str(exc))
            return self._task_result(self._fsm.task)

    def approve_plan(self) -> Dict[str, Any]:
        """Подтверждает план (ручной режим) и выполняет первый шаг."""
        with self._lock:
            error = self._require_stage(TaskStage.AWAITING_PLAN_APPROVAL,
                                        "Подтвердить план")
            if error:
                return error
            assert self._fsm is not None
            task = self._fsm.task
            self._fsm.approve_plan()
            result = self._advance_if_working(task)
            return self._task_result(task, **result)

    def reject_plan(self, reason: str = "") -> Dict[str, Any]:
        """Отклоняет план: возврат в планирование за новым планом."""
        with self._lock:
            error = self._require_stage(TaskStage.AWAITING_PLAN_APPROVAL,
                                        "Отклонить план")
            if error:
                return error
            assert self._fsm is not None
            task = self._fsm.task
            self._fsm.reject_plan(reason)
            result = self._advance_if_working(task)
            return self._task_result(task, **result)

    def approve_final(self) -> Dict[str, Any]:
        """Финальное подтверждение: задача завершена (DONE)."""
        with self._lock:
            error = self._require_stage(TaskStage.AWAITING_FINAL_APPROVAL,
                                        "Финальное подтверждение")
            if error:
                return error
            assert self._fsm is not None
            task = self._fsm.task
            self._fsm.approve_final(result=task.result)
            return self._task_result(task)

    def reject_final(self, reason: str = "") -> Dict[str, Any]:
        """Отклоняет итог: возврат в выполнение для переделки."""
        with self._lock:
            error = self._require_stage(TaskStage.AWAITING_FINAL_APPROVAL,
                                        "Отклонить итог")
            if error:
                return error
            assert self._fsm is not None
            task = self._fsm.task
            self._fsm.reject_final(reason)
            result = self._advance_if_working(task)
            return self._task_result(task, **result)

    def set_task_mode(self, mode: TaskMode) -> Dict[str, Any]:
        """Меняет режим задачи на лету (auto ↔ manual)."""
        with self._lock:
            if self._fsm is None:
                return self._task_result(None, error="Активной задачи нет.")
            self._fsm.set_mode(mode)
            return self._task_result(self._fsm.task)

    def get_task_events(self, task_id: Optional[int] = None) -> List[dict]:
        """Журнал событий задачи (для UI/диагностики)."""
        with self._lock:
            repo = self.task_repository
            if repo is None:
                return []
            target = task_id
            if target is None and self._fsm is not None:
                target = self._fsm.task.id
            if target is None:
                return []
            return repo.get_task_events(int(target))

    def note_task_message(self, text: str, role: str = "user") -> None:
        """Добавляет сообщение о задаче в историю чата (без вызова LLM).

        Задача создаётся сообщением в чате: это сообщение остаётся в
        истории диалога, а итог завершённой задачи дописывается как
        сообщение ассистента. Состояние задачи при этом живёт в БД
        задач, а не в истории чата (разные слои).
        """
        with self._lock:
            self._append_message({"role": role, "content": text})
            if self._trimming_active():
                self._trim_history()
            self._persist()

    def delete_task(self, task_id: Optional[int] = None) -> Dict[str, Any]:
        """Удаляет задачу из БД (каскадно: шаги и журнал событий).

        Без аргумента удаляется текущая задача панели. Незавершённую
        (активную) задачу сначала нужно остановить — защита от потери
        работающего прогресса одним кликом.

        Args:
            task_id: идентификатор задачи; None — текущая задача панели.

        Returns:
            Стандартный результат задачных методов; ``deleted=True`` при
            успехе.
        """
        with self._lock:
            repo = self.task_repository
            if repo is None:
                return self._task_result(
                    None, error="Задачи недоступны: хранилище не подключено.")
            target = task_id
            if target is None and self._fsm is not None:
                target = self._fsm.task.id
            if target is None:
                return self._task_result(None, error="Активной задачи нет.")
            task = repo.get_task_by_id(int(target))
            if task is not None and not task.is_terminal():
                return self._task_result(
                    task,
                    error="Сначала остановите задачу («Стоп»), "
                          "затем удаляйте.",
                )
            if not repo.delete_task(int(target)):
                return self._task_result(
                    None, error=f"Не удалось удалить задачу #{target}.")
            if self._fsm is not None and self._fsm.task.id == int(target):
                self._fsm = None  # панель скрывается (задачи больше нет)
            result = self._task_result(None)
            result["deleted"] = True
            return result

    def clear_tasks(self) -> Dict[str, Any]:
        """Удаляет ВСЕ задачи (каскадно, со всеми шагами и журналами).

        Используется кнопкой «Очистить задачи»: полная очистка истории
        задач профиля(-ей), включая активную задачу.

        Returns:
            Стандартный результат задачных методов; ``cleared`` — число
            удалённых задач.
        """
        with self._lock:
            repo = self.task_repository
            if repo is None:
                return self._task_result(
                    None, error="Задачи недоступны: хранилище не подключено.")
            deleted = repo.delete_all_tasks()
            if deleted < 0:
                return self._task_result(
                    None, error="Не удалось очистить задачи (ошибка БД).")
            self._fsm = None
            result = self._task_result(None)
            result["cleared"] = deleted
            return result

    def restore_task(self) -> Dict[str, Any]:
        """Восстанавливает активную задачу профиля после запуска приложения.

        Задача НЕ запускается автоматически: если она была прервана в
        рабочей стадии (planning/execution/validation), вызывающий должен
        показать «Задача прервана. Продолжить?» и ждать пользователя.
        Для стадий paused/awaiting_* достаточно показать панель задачи —
        они сами ждут действия пользователя.

        Returns:
            ``{"task": Task|None, "interrupted": bool, "error": str|None}``.
            ``interrupted=True`` — задача остановилась в рабочей стадии.
        """
        with self._lock:
            repo = self.task_repository
            if repo is None:
                return {"task": None, "interrupted": False,
                        "error": "Задачи недоступны: хранилище не подключено."}
            task = repo.get_active_task(self._active_profile_id() or 0)
            if task is None:
                self._fsm = None
                return {"task": None, "interrupted": False, "error": None}
            self._fsm = TaskFSM(task, repo,
                                 max_retries=self._task_max_retries)
            interrupted = task.stage in WORKING_STAGES
            if interrupted:
                # Не запускаем автоматически: ждём явного «Продолжить».
                message = "Задача прервана. Продолжить?"
                task.expected_action = message
                repo.update_task_stage(task.id, task.stage, message)
            return {"task": task, "interrupted": interrupted, "error": None}

    # ---- внутренняя механика подсистемы задач -------------------------

    def _task_result(self, task: Optional[Task], acted: bool = False,
                     auto: bool = False,
                     error: Optional[str] = None) -> Dict[str, Any]:
        """Единая форма результата задачных методов агента."""
        return {"task": task, "acted": acted, "auto": auto, "error": error}

    def _ensure_fsm(self, task_id: Optional[int]) -> tuple:
        """Гарантирует, что self._fsm привязан к запрошенной задаче из БД.

        Состояние всегда перечитывается из БД — агент не держит единственную
        копию состояния задачи в памяти (перезапуск/пауза безопасны).

        Returns:
            ``(task, error)`` — ровно одно из значений не None.
        """
        repo = self.task_repository
        if repo is None:
            return None, "Задачи недоступны: хранилище не подключено."
        target_id = task_id
        if target_id is None and self._fsm is not None:
            target_id = self._fsm.task.id
        if target_id is None:
            return None, "Активной задачи нет."
        task = repo.get_task_by_id(int(target_id))
        if task is None:
            return None, f"Задача id={target_id} не найдена."
        if (self._fsm is None
                or self._fsm.task.id != task.id
                or task_id is not None):
            self._fsm = TaskFSM(task, repo,
                                 max_retries=self._task_max_retries)
        else:
            self._fsm.task = task  # свежая копия из БД
        return task, None

    def _require_stage(self, stage: TaskStage,
                       action: str) -> Optional[Dict[str, Any]]:
        """Проверяет, что активная задача находится в ожидаемой стадии.

        Returns:
            None — проверка пройдена; иначе готовый результат с ошибкой.
        """
        if self._fsm is None:
            return self._task_result(None, error="Активной задачи нет.")
        if self._fsm.task.stage != stage:
            return self._task_result(
                self._fsm.task,
                error=(f"{action} можно только в стадии "
                       f"{stage.value} (сейчас: {self._fsm.task.stage.value})."),
            )
        return None

    def _advance_if_working(self, task: Task) -> Dict[str, Any]:
        """Выполняет этап, если стадия рабочая; иначе просто ждёт."""
        if task.stage in WORKING_STAGES:
            return self._advance_task(task)
        return {"acted": False, "auto": False}

    def _advance_task(self, task: Task) -> Dict[str, Any]:
        """Диспетчер этапов: выполняет работу текущей стадии.

        Один вызов — один этап LLM (план / один шаг / валидация).
        Пользователь управляет только переходами между этапами, поэтому
        шаги внутри EXECUTION и проверка цепочкой продолжаются сами
        (auto=True) в обоих режимах — пока не встретится ожидание
        подтверждения (manual), пауза или конечная стадия.

        Returns:
            ``{"acted": True, "auto": bool}`` — auto=True, если задача
            продолжает работать сама (нужен ещё вызов continue_task).
        """
        try:
            if task.stage == TaskStage.PLANNING:
                self._generate_plan(task)
            elif task.stage == TaskStage.EXECUTION:
                self._execute_current_step(task)
            elif task.stage == TaskStage.VALIDATION:
                self._validate_result(task)
            else:
                return {"acted": False, "auto": False}
        except Exception as exc:  # любая ошибка этапа — задача FAILED
            logger.error("Ошибка этапа задачи #%s (%s): %s",
                         task.id, task.stage.value, exc)
            try:
                self._fsm.fail(humanize_api_error(exc))
            except InvalidTransitionError:
                pass
            return {"acted": False, "auto": False}
        return {"acted": True, "auto": task.stage in WORKING_STAGES}

    def _generate_plan(self, task: Task) -> None:
        """Этап PLANNING: LLM разбивает задачу на шаги (JSON-план).

        В manual-режиме после плана задача переходит в ожидание
        подтверждения, в auto — сразу в выполнение.
        """
        assert self._fsm is not None
        system = self.system_prompt + "\n\n" + self._task_context_block(task)
        user = (
            "Разбей задачу на 3–7 последовательных шагов. Верни СТРОГО JSON "
            "без пояснений в формате: {\"steps\": [\"шаг 1\", \"шаг 2\", ...]}. "
            f"Задача: {task.description}"
        )
        if self._mock_mode:
            steps_text = [
                f"Проанализировать задачу: {task.description[:80]}",
                "Выполнить основную работу",
                "Проверить и оформить результат",
            ]
        else:
            raw = self._task_llm_call(system, user)
            steps_text = self._parse_plan_steps(raw)
        steps = [TaskStep(description=text, status="pending")
                 for text in steps_text]
        if not steps:
            raise ValueError("LLM вернул пустой план.")
        self.task_repository.update_task_plan(task.id, steps)
        task.plan = steps
        if not task.title:
            task.title = task.description.strip().splitlines()[0][:60]
            self.task_repository.update_task_fields(task.id, title=task.title)
        if task.mode == TaskMode.MANUAL:
            self._fsm.transition(TaskStage.AWAITING_PLAN_APPROVAL,
                                 "plan_ready")
        else:
            self._fsm.transition(TaskStage.EXECUTION, "plan_ready_auto")

    def _execute_current_step(self, task: Task) -> None:
        """Этап EXECUTION: выполняет текущий шаг плана одним LLM-вызовом.

        Контекст запроса: задача + текущий шаг + история выполнения
        предыдущих шагов (из БД) + причина отклонения (при переделке).
        Пользователь шаги не подтверждает: после выполненного шага
        автоматом переходим к следующему (или в валидацию, если шаги
        кончились).
        """
        assert self._fsm is not None
        step = task.current_step()
        if step is None:
            if task.plan:
                self._fsm.transition(TaskStage.VALIDATION,
                                     "steps_completed")
                return
            raise ValueError("План задачи пуст — нечего выполнять.")
        self._fsm.mark_step_started(step)
        history = "\n".join(
            f"  {i + 1}. {s.description} — {s.status}: {s.result or '—'}"
            for i, s in enumerate(task.plan[:task.current_step_index])
        ) or "  (шаг первый — истории ещё нет)"
        redo = f"\nЗамечание от пользователя (переделать): {step.rejection_reason}" \
            if step.rejection_reason else ""
        system = self.system_prompt + "\n\n" + self._task_context_block(task)
        user = (
            f"Выполни текущий шаг задачи и верни краткий результат работы "
            f"(без вопросов и уточнений).\n"
            f"Текущий шаг ({task.current_step_index + 1}/{len(task.plan)}): "
            f"{step.description}\n"
            f"История выполнения предыдущих шагов:\n{history}{redo}"
        )
        if self._mock_mode:
            result_text = (f"Шаг {task.current_step_index + 1} выполнен "
                           f"(mock): {step.description}")
            if step.rejection_reason:
                result_text += f" (с учётом замечания: {step.rejection_reason})"
        else:
            result_text = self._task_llm_call(system, user)
        self._fsm.mark_step_done(step, result_text)
        self._fsm.next_step()

    def _validate_result(self, task: Task) -> None:
        """Этап VALIDATION: LLM проверяет результат по шагам.

        Вердикт JSON: {"verdict": "pass"|"fail", "comment": "..."}.
        pass: manual → ожидание финального подтверждения, auto → DONE.
        fail: откат в EXECUTION через FSM (retry_count растёт в контроллере;
        когда откатов сделано TASKS_MAX_RETRIES, матрица запрещает следующий —
        задача переводится в FAILED).
        """
        assert self._fsm is not None
        steps_report = "\n".join(
            f"  {i + 1}. {s.description} — {s.status}: {s.result or '—'}"
            for i, s in enumerate(task.plan)
        )
        system = self.system_prompt + "\n\n" + self._task_context_block(task)
        user = (
            "Проверь, достигнута ли цель задачи по истории выполнения шагов. "
            "Верни СТРОГО JSON: {\"verdict\": \"pass\"|\"fail\", "
            "\"comment\": \"краткое обоснование\"}.\n"
            f"Задача: {task.description}\nВыполненные шаги:\n{steps_report}"
        )
        if self._mock_mode:
            verdict, comment = "pass", "Все шаги выполнены (mock-проверка)."
        else:
            raw = self._task_llm_call(system, user)
            verdict, comment = self._parse_verdict(raw)
        summary = "\n".join(
            f"{i + 1}. {s.description}: {s.result or '—'}"
            for i, s in enumerate(task.plan)
        )
        task.result = f"{comment}\n\nИтоги шагов:\n{summary}"
        self.task_repository.set_task_result(task.id, task.result)
        if verdict == "pass":
            if task.mode == TaskMode.MANUAL:
                self._fsm.transition_to(TaskStage.AWAITING_FINAL_APPROVAL,
                                        "validation_passed")
            else:
                self._fsm.transition_to(TaskStage.DONE, "validation_passed")
            return
        # Валидация не пройдена: откат в EXECUTION. retry_count растёт
        # централизованно в FSM (_commit_transition); когда лимит откатов
        # исчерпан (по ТЗ: «3 отката — можно, 4-й — FAILED»), матрица
        # запрещает переход — задача переводится в FAILED.
        try:
            self._fsm.transition_to(TaskStage.EXECUTION, "validation_failed",
                                    payload={"comment": comment})
        except InvalidTransitionError as exc:
            self._fsm.fail(
                f"Превышен лимит повторов валидации "
                f"({self._task_retry_limit()}): {comment} ({exc.reason})"
            )

    def _task_retry_limit(self) -> int:
        """Лимит откатов валидации (конфиг TASKS_MAX_RETRIES, единый с FSM)."""
        return self._task_max_retries

    def _task_llm_call(self, system: str, user: str) -> str:
        """Служебный LLM-вызов для задач (мимо истории диалога).

        История чата не загрязняется: контекст задачи собирается из БД
        (состояние задачи + шаги), а не из message_history.

        Returns:
            Текст ответа модели.

        Raises:
            ApiError: при любой ошибке API (этап переведёт задачу в FAILED).
        """
        if self._client is None:
            raise ApiError("Клиент API не инициализирован (mock-режим?).")
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            raise ApiError(humanize_api_error(exc), original=exc)
        return str(self._handle_response(response).get("content") or "")

    def _task_context_block(self, task: Task) -> str:
        """Блок «[Состояние задачи]» для system-промпта задачных вызовов.

        Ключ к возобновлению без повторных вопросов: агент каждый раз
        получает полное состояние из БД — название, этап, текущий шаг,
        ожидаемое действие и историю выполнения.
        """
        current = task.current_step()
        history = "\n".join(
            f"  {i + 1}. {s.description} — {s.status}: {s.result or '—'}"
            for i, s in enumerate(task.plan)
        ) or "  (пока пусто)"
        current_line = (
            f"Текущий шаг ({task.current_step_index + 1}/{len(task.plan)}): "
            f"{current.description}" if current
            else "Текущий шаг: —"
        )
        return (
            "[Состояние задачи]\n"
            f"Название: {task.title or task.description}\n"
            f"Описание: {task.description}\n"
            f"Режим: {task.mode.value}\n"
            f"Этап: {task.stage.value}\n"
            f"{current_line}\n"
            f"Ожидаемое действие: {task.expected_action or '—'}\n"
            f"История выполнения:\n{history}"
        )

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """Извлекает первый JSON-объект из ответа LLM (с учётом ```-блоков)."""
        cleaned = text.strip()
        if "```" in cleaned:
            parts = cleaned.split("```")
            for part in parts:
                candidate = part.strip()
                if candidate.startswith("json"):
                    candidate = candidate[4:].strip()
                if candidate.startswith("{") or candidate.startswith("["):
                    cleaned = candidate
                    break
        start = cleaned.find("{")
        if start == -1:
            return None
        depth = 0
        for index in range(start, len(cleaned)):
            if cleaned[index] == "{":
                depth += 1
            elif cleaned[index] == "}":
                depth -= 1
                if depth == 0:
                    return cleaned[start:index + 1]
        return None

    @staticmethod
    def _parse_plan_steps(raw: str) -> List[str]:
        """Разбирает JSON-план LLM в список описаний шагов."""
        import json as _json

        blob = DeepSeekAgent._extract_json(raw)
        if not blob:
            return []
        try:
            data = _json.loads(blob)
        except ValueError:
            return []
        steps_raw = data.get("steps") if isinstance(data, dict) else data
        if not isinstance(steps_raw, list):
            return []
        steps: List[str] = []
        for item in steps_raw:
            if isinstance(item, str) and item.strip():
                steps.append(item.strip())
            elif isinstance(item, dict):
                text = str(item.get("step") or item.get("description") or "")
                if text.strip():
                    steps.append(text.strip())
        return steps

    @staticmethod
    def _parse_verdict(raw: str) -> tuple:
        """Разбирает вердикт валидации; неразбираемый ответ считается fail."""
        import json as _json

        blob = DeepSeekAgent._extract_json(raw)
        if blob:
            try:
                data = _json.loads(blob)
                if isinstance(data, dict):
                    verdict = str(data.get("verdict") or "").strip().lower()
                    comment = str(data.get("comment") or "").strip()
                    if verdict in ("pass", "fail"):
                        return verdict, comment
                    return "fail", f"Неизвестный вердикт: {raw[:200]}"
            except ValueError:
                pass
        return "fail", f"Не удалось разобрать вердикт: {raw[:200]}"

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
