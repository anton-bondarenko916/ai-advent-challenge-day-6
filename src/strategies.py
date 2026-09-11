"""Стратегии управления контекстом: что именно уходит в запрос к модели.

Модуль реализует переключаемые стратегии сборки контекста:

* ``sliding_window`` — в API уходят только последние N сообщений диалога;
  история при этом **не удаляется** (она остаётся в памяти, в БД и в чате);
* ``sticky_facts`` — в API уходят блок фактов (key → value) и последние
  N сообщений; блок фактов обновляется после каждого хода;
* ``branching`` — не отдельный способ сборки контекста, а **режим работы с
  ветками**: контекст собирает подстратегия активной ветки
  (``sliding_window`` или ``sticky_facts``). Поэтому :func:`create_strategy`
  для имени ``branching`` возвращает стратегию подстратегии ветки.

Контракт стратегии::

    build_context(system_prompt, dialog, artifacts) -> list[Message]
    after_turn(user_text, assistant_text, dialog, artifacts, call_service) -> dict
    describe() -> dict

``artifacts`` — словарь артефактов диалога: ``{"facts": FactsStore,
"summary": dict | None, "counter": TokenCounter}``. Стратегия **не хранит**
состояние: факты живут в :class:`src.facts.FactsStore`, история — в агенте.

Модуль не импортирует tkinter и не обращается к сети: служебный запрос
выполняет переданный callable (в тестах подменяется).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from . import utils
from .facts import (
    DEFAULT_FACTS_MAX_ITEMS,
    DEFAULT_FACTS_MAX_TOKENS,
    DEFAULT_FACTS_REQUEST_MAX_TOKENS,
    DEFAULT_FACTS_TEMPERATURE,
    FactsStore,
    build_request as build_facts_request,
    parse_response as parse_facts_response,
)
from .utils import AgentConfigError

logger = utils.get_logger("deepseek-chat.strategies")

#: Тип сообщения чата
Message = Dict[str, str]

#: Имя стратегии «скользящее окно»
SLIDING_WINDOW = "sliding_window"
#: Имя стратегии «липкие факты»
STICKY_FACTS = "sticky_facts"
#: Имя режима «ветки диалога»
BRANCHING = "branching"

#: Стратегии, которые реально собирают контекст (без режима веток)
CONTEXT_STRATEGIES = (SLIDING_WINDOW, STICKY_FACTS)
#: Полный список значений переключателя (в порядке отображения в UI)
AVAILABLE_STRATEGIES = (SLIDING_WINDOW, STICKY_FACTS, BRANCHING)

#: Размер окна по умолчанию (сообщений диалога без system)
DEFAULT_WINDOW_KEEP_LAST_N = 20

#: Псевдонимы имён стратегий (упрощают CLI и конфигурацию)
_ALIASES = {
    "window": SLIDING_WINDOW,
    "sliding": SLIDING_WINDOW,
    "sliding-window": SLIDING_WINDOW,
    "slidingwindow": SLIDING_WINDOW,
    "facts": STICKY_FACTS,
    "sticky": STICKY_FACTS,
    "sticky-facts": STICKY_FACTS,
    "stickyfacts": STICKY_FACTS,
    "branch": BRANCHING,
    "branches": BRANCHING,
}

#: Сигнатура callable служебного запроса к API
ServiceCall = Callable[[Dict[str, Any]], Dict[str, Any]]


def normalize_name(name: Any) -> str:
    """Приводит имя стратегии к каноническому виду.

    Args:
        name: имя или псевдоним стратегии.

    Returns:
        Каноническое имя ("sliding_window" | "sticky_facts" | "branching").

    Raises:
        AgentConfigError: если имя неизвестно.
    """
    raw = str(name or "").strip().lower()
    if raw in AVAILABLE_STRATEGIES:
        return raw
    if raw in _ALIASES:
        return _ALIASES[raw]
    raise AgentConfigError(
        f"Неизвестная стратегия контекста: {name!r}. "
        f"Доступные: {', '.join(AVAILABLE_STRATEGIES)}"
    )


def available_strategies() -> List[str]:
    """Список стратегий для переключателя (в порядке отображения)."""
    return list(AVAILABLE_STRATEGIES)


def is_context_strategy(name: Any) -> bool:
    """True, если имя — стратегия сборки контекста (не режим веток)."""
    try:
        return normalize_name(name) in CONTEXT_STRATEGIES
    except AgentConfigError:
        return False


# ---------------------------------------------------------------------------
# Базовая стратегия
# ---------------------------------------------------------------------------

class ContextStrategy:
    """Базовый интерфейс стратегии управления контекстом.

    Attributes:
        name: имя стратегии.
        settings: настройки стратегии (плоский словарь из конфигурации).
        deps: зависимости (счётчик токенов, модель диалога, логгер).
    """

    name: str = "context"

    def __init__(self, settings: Optional[Dict[str, Any]] = None,
                 deps: Optional[Dict[str, Any]] = None) -> None:
        self.settings: Dict[str, Any] = dict(settings or {})
        self.deps: Dict[str, Any] = dict(deps or {})

    # -- вспомогательное -------------------------------------------------

    def _window(self) -> int:
        """Размер окна последних сообщений (не меньше 1)."""
        try:
            window = int(self.settings.get("window", DEFAULT_WINDOW_KEEP_LAST_N))
        except (TypeError, ValueError):
            window = DEFAULT_WINDOW_KEEP_LAST_N
        return max(1, window)

    def _counter(self, artifacts: Dict[str, Any]) -> Any:
        """Счётчик токенов из артефактов или зависимостей."""
        return (artifacts or {}).get("counter") or self.deps.get("counter")

    def _model(self) -> str:
        """Модель для служебных запросов (по умолчанию — модель диалога)."""
        return str(self.settings.get("facts_model") or "").strip() \
            or str(self.deps.get("model") or "")

    @staticmethod
    def _system_messages(system_prompt: str) -> List[Message]:
        """Основной системный промпт (если он непустой)."""
        text = str(system_prompt or "").strip()
        return [{"role": "system", "content": text}] if text else []

    # -- контракт --------------------------------------------------------

    def build_context(self, system_prompt: str, dialog: Sequence[Message],
                      artifacts: Optional[Dict[str, Any]] = None) -> List[Message]:
        """Собирает сообщения, которые уйдут в API.

        Args:
            system_prompt: основной системный промпт.
            dialog: сообщения диалога без системного промпта.
            artifacts: артефакты диалога (факты и т.п.).

        Returns:
            Список сообщений для API.
        """
        raise NotImplementedError

    def after_turn(self, user_text: str, assistant_text: str,
                   dialog: Sequence[Message],
                   artifacts: Optional[Dict[str, Any]] = None,
                   call_service: Optional[ServiceCall] = None) -> Dict[str, Any]:
        """Обновляет артефакты стратегии после завершённого хода.

        Ошибки не пробрасываются: они возвращаются в поле ``error``, чтобы
        сбой служебного запроса не ломал ход диалога.

        Returns:
            Словарь {"updated", "error", "artifacts", "tokens"}.
        """
        return {
            "updated": False,
            "error": None,
            "artifacts": artifacts or {},
            "tokens": None,
        }

    def describe(self) -> Dict[str, Any]:
        """Описание стратегии для UI и статистики."""
        return {"name": self.name, "window": self._window(), "facts": False}


# ---------------------------------------------------------------------------
# Стратегия 1: скользящее окно
# ---------------------------------------------------------------------------

class SlidingWindowStrategy(ContextStrategy):
    """В запрос уходят только последние N сообщений диалога.

    История **не удаляется**: она остаётся в памяти агента, в БД и в чате,
    поэтому при переключении стратегии полный контекст возвращается.
    """

    name = SLIDING_WINDOW

    def build_context(self, system_prompt: str, dialog: Sequence[Message],
                      artifacts: Optional[Dict[str, Any]] = None) -> List[Message]:
        """``[system] + dialog[-N:]``."""
        messages = self._system_messages(system_prompt)
        messages.extend(list(dialog or [])[-self._window():])
        return messages

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mode": "window",
            "window": self._window(),
            "facts": False,
        }


# ---------------------------------------------------------------------------
# Стратегия 2: липкие факты
# ---------------------------------------------------------------------------

class StickyFactsStrategy(ContextStrategy):
    """В запрос уходят блок фактов и последние N сообщений.

    Блок фактов обновляется после каждого хода: сначала локальным разбором
    явных команд пользователя, затем служебным LLM-запросом (строгий JSON).
    """

    name = STICKY_FACTS

    def __init__(self, settings: Optional[Dict[str, Any]] = None,
                 deps: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(settings, deps)
        self.facts_enabled = bool(self.settings.get("facts_enabled", True))
        try:
            self.facts_max_tokens = int(
                self.settings.get("facts_max_tokens", DEFAULT_FACTS_MAX_TOKENS)
            )
        except (TypeError, ValueError):
            self.facts_max_tokens = DEFAULT_FACTS_MAX_TOKENS
        try:
            self.facts_max_items = int(
                self.settings.get("facts_max_items", DEFAULT_FACTS_MAX_ITEMS)
            )
        except (TypeError, ValueError):
            self.facts_max_items = DEFAULT_FACTS_MAX_ITEMS
        try:
            self.facts_temperature = float(
                self.settings.get("facts_temperature", DEFAULT_FACTS_TEMPERATURE)
            )
        except (TypeError, ValueError):
            self.facts_temperature = DEFAULT_FACTS_TEMPERATURE
        self.facts_prompt = str(self.settings.get("facts_prompt") or "")

    # -- сборка контекста ------------------------------------------------

    def build_context(self, system_prompt: str, dialog: Sequence[Message],
                      artifacts: Optional[Dict[str, Any]] = None) -> List[Message]:
        """``[system] + [system: факты] + dialog[-N:]``."""
        messages = self._system_messages(system_prompt)
        store = (artifacts or {}).get("facts")
        if isinstance(store, FactsStore):
            messages.extend(store.to_messages())
        messages.extend(list(dialog or [])[-self._window():])
        return messages

    # -- обновление артефактов -------------------------------------------

    def after_turn(self, user_text: str, assistant_text: str,
                   dialog: Sequence[Message],
                   artifacts: Optional[Dict[str, Any]] = None,
                   call_service: Optional[ServiceCall] = None) -> Dict[str, Any]:
        """Обновляет блок фактов по завершённому обмену сообщениями.

        Порядок:
        1. локальный разбор явных команд («запомни…») — работает всегда;
        2. служебный LLM-запрос (если включён и доступен) — строгий JSON;
        3. применение лимитов по числу фактов и токенам.

        Сбой шага 2 не ломает ход: факты остаются прежними, ошибка попадает
        в результат.
        """
        artifacts = artifacts if artifacts is not None else {}
        store = artifacts.get("facts")
        result: Dict[str, Any] = {
            "updated": False,
            "error": None,
            "artifacts": artifacts,
            "tokens": None,
        }
        if not isinstance(store, FactsStore):
            result["error"] = "Хранилище фактов недоступно."
            return result

        # 1. Явные команды пользователя (гарантия даже при сбое LLM)
        local = FactsStore.extract_local(user_text)
        if local:
            stats = store.apply(local, source="local")
            result["updated"] = bool(
                stats["added"] or stats["updated"] or stats["removed"]
            )

        # 2. Служебное обновление через LLM
        if self.facts_enabled and call_service is not None:
            model = self._model()
            if not model:
                result["error"] = "Не задана модель для обновления фактов."
            else:
                request = build_facts_request(
                    store, user_text, assistant_text, model,
                    prompt=self.facts_prompt or None,
                    temperature=self.facts_temperature,
                    max_tokens=DEFAULT_FACTS_REQUEST_MAX_TOKENS,
                )
                tokens: Optional[Dict[str, Any]] = None
                try:
                    payload = call_service(request) or {}
                except Exception as exc:  # ApiError и прочие неожиданные
                    message = getattr(exc, "message", None) \
                        or utils.humanize_api_error(exc)
                    result["error"] = str(message)
                    logger.warning(
                        "Обновление фактов не удалось: %s", result["error"]
                    )
                    payload = None

                if payload is not None:
                    tokens = dict(payload.get("tokens") or {})
                    # Как и у сжатия: отсутствие пометки — это точные значения
                    # usage из API, оценочные помечаются явно.
                    tokens.setdefault("estimated", False)
                    tokens["model"] = payload.get("model") or model
                    parsed, parse_error = parse_facts_response(
                        str(payload.get("content") or "")
                    )
                    if parse_error:
                        result["error"] = parse_error
                        logger.warning("Обновление фактов: %s", parse_error)
                    else:
                        stats = store.apply(parsed or {}, source="llm")
                        result["updated"] = bool(
                            result["updated"] or stats["added"]
                            or stats["updated"] or stats["removed"]
                        )
                    result["tokens"] = tokens

        # 3. Лимиты по числу фактов и токенам
        counter = self._counter(artifacts)
        store.limit(
            counter=counter,
            max_items=self.facts_max_items,
            max_tokens=self.facts_max_tokens,
        )
        return result

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mode": "facts",
            "window": self._window(),
            "facts": True,
            "facts_enabled": self.facts_enabled,
            "facts_max_items": self.facts_max_items,
            "facts_max_tokens": self.facts_max_tokens,
        }


# ---------------------------------------------------------------------------
# Реестр и фабрика
# ---------------------------------------------------------------------------

#: Реестр стратегий сборки контекста
STRATEGIES: Dict[str, type] = {
    SLIDING_WINDOW: SlidingWindowStrategy,
    STICKY_FACTS: StickyFactsStrategy,
}


def create_strategy(name: str,
                    settings: Optional[Dict[str, Any]] = None,
                    deps: Optional[Dict[str, Any]] = None) -> ContextStrategy:
    """Создаёт стратегию по имени.

    Для имени ``branching`` создаётся стратегия подстратегии активной ветки
    (``settings["branch_strategy"]``, по умолчанию ``sliding_window``):
    «ветки» — это режим работы с историей, а не способ сборки контекста.

    Args:
        name: имя стратегии (см. :data:`AVAILABLE_STRATEGIES`).
        settings: настройки стратегии (плоский словарь).
        deps: зависимости (``counter``, ``model``, ...).

    Returns:
        Экземпляр стратегии.

    Raises:
        AgentConfigError: если имя неизвестно.
    """
    canonical = normalize_name(name)
    if canonical == BRANCHING:
        sub_name = normalize_name(
            (settings or {}).get("branch_strategy") or SLIDING_WINDOW
        )
        if sub_name == BRANCHING:  # защита от рекурсии
            sub_name = SLIDING_WINDOW
        logger.debug("Режим веток: контекст собирает подстратегия %s.", sub_name)
        canonical = sub_name
    factory = STRATEGIES.get(canonical)
    if factory is None:  # pragma: no cover — недостижимо при валидном normalize
        raise AgentConfigError(
            f"Неизвестная стратегия контекста: {name!r}. "
            f"Доступные: {', '.join(AVAILABLE_STRATEGIES)}"
        )
    return factory(settings=settings, deps=deps)
