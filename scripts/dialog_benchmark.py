"""Бенчмарк диалогов с агентом: legacy-сценарии и стратегии контекста.

Скрипт ведёт несколько диалогов с ``DeepSeekAgent`` и сравнивает расход
токенов, стоимость, «память» агента и работу стратегий управления контекстом.

**Legacy-сценарии** (работают без стратегий, как раньше — summary +
``MAX_HISTORY_MESSAGES``):

1. **short** — короткий диалог (3 хода): всё помещается в контекст;
2. **long** — длинный диалог (14 ходов): контекст растёт линейно, но в окно
   модели помещается;
3. **over_limit** — тот же длинный диалог, но окно **искусственно уменьшено**
   (по умолчанию 6 сообщений и 400 токенов): часть истории вытесняется, и агент
   «забывает» факт, сказанный в начале диалога.

**Сценарии стратегий** (агент создаётся с ``strategy=...`` и
``strategy_settings=...``):

4. **window** — ``sliding_window``: в запрос уходят только последние
   ``--strategy-window`` сообщений (по умолчанию 20). История не удаляется, но
   факт из начала диалога в контекст не попадает — «Помнит: нет»;
5. **facts** — ``sticky_facts``: в запрос уходят блок фактов (key → value) и
   последние N сообщений. Блок обновляется служебным запросом после каждого
   хода, поэтому факт из начала диалога сохраняется даже при узком окне —
   «Помнит: да», видны служебные запросы и их токены;
6. **branching** — ``branching`` (подстратегия — ``sticky_facts``): несколько
   ходов в корневой ветке, затем ``checkpoint`` («ветка-A»), своё продолжение в
   каждой ветке и проверка изоляции: сообщения одной ветки не встречаются в
   другой, и каждая ветка «помнит своё».

В первом ходе каждого диалога сообщается факт («Меня зовут Алекс… любимый
цвет — синий»), в последнем ходе задаётся проверочный вопрос. Это позволяет
наглядно увидеть разницу между «помнит» и «не помнит».

Режимы работы:

* ``--mode sim`` (по умолчанию) — офлайн-имитация модели: детерминированные
  ответы, токены считаются локальным счётчиком, сеть не используется;
* ``--mode live`` — реальные запросы к DeepSeek API (нужен ключ и ``--yes``);
* ``--mode auto`` — live, если в конфигурации есть ключ, иначе sim.

Примеры запуска (из корня проекта)::

    python scripts/dialog_benchmark.py                  # офлайн-сравнение всего
    python scripts/dialog_benchmark.py --strategy-turns 30
    python scripts/dialog_benchmark.py --strategies window,facts,branching
    python scripts/dialog_benchmark.py --no-strategies   # только legacy-сценарии
    python scripts/dialog_benchmark.py --strategies facts --strategy-window 10
    python scripts/dialog_benchmark.py --dry-run        # только план диалогов
    python scripts/dialog_benchmark.py --verbose        # по каждому ходу
    python scripts/dialog_benchmark.py --json report.json
    python scripts/dialog_benchmark.py --mode live --yes

Скрипт по умолчанию **не трогает** вашу историю: хранилище выключено. Чтобы
проверить персистентность, укажите ``--storage-db data/bench.db``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Скрипт лежит в scripts/, а пакет src/ — в корне проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import DeepSeekAgent  # noqa: E402
from src.config import get_config, get_prices  # noqa: E402
from src.facts import FACTS_MARKER  # noqa: E402
from src.storage import ContextStore  # noqa: E402
from src.strategies import (  # noqa: E402
    BRANCHING,
    SLIDING_WINDOW,
    STICKY_FACTS,
    normalize_name,
)
from src.summarizer import SUMMARY_MARKER  # noqa: E402
from src.tokens import create_counter  # noqa: E402

# --- Данные «памяти» диалога -------------------------------------------------
FACT_NAME = "Алекс"
FACT_CITY = "Казани"
FACT_COLOR = "синий"
PROBE_MARKER = "ПРОВЕРКА ПАМЯТИ"

# --- Данные сценария ветвления ----------------------------------------------
#: Маркер сообщения, которое есть только в ветке A
BRANCH_A_MARKER = "ВЕТКА-А"
#: Маркер сообщения, которое есть только в ветке B
BRANCH_B_MARKER = "ВЕТКА-Б"
#: «Свой» факт ветки A (проверяется память именно этой ветки)
BRANCH_A_PROJECT = "ракета «Альфа»"
#: «Свой» факт ветки B
BRANCH_B_PROJECT = "лодка «Бета»"
#: Проверочный вопрос внутри ветки (ответ зависит только от её истории)
BRANCH_PROBE_MARKER = "ПРОВЕРКА ВЕТКИ"
#: Имя checkpoint-ветки в сценарии ветвления
BRANCH_A_NAME = "ветка-A"
#: Сколько ходов сценария ветвления уходит на «разогрев» корневой ветки
BRANCH_WARMUP_TURNS = 4

# --- Настройки сценариев стратегий ------------------------------------------
#: Размер окна (сообщений) по умолчанию для сценариев стратегий
DEFAULT_STRATEGY_WINDOW = 20
#: Число ходов по умолчанию для сценариев window/facts
DEFAULT_STRATEGY_TURNS = 30
#: Подстратегия контекста для режима веток (по умолчанию — sticky_facts)
DEFAULT_BRANCH_SUB_STRATEGY = STICKY_FACTS

# Темы-заполнители для «длинных» ходов
_FILLER_TOPICS = [
    "как работает HTTP и почему он stateless",
    "чем векторные базы данных отличаются от реляционных",
    "как устроен механизм внимания в трансформерах",
    "что такое идемпотентность и зачем она нужна",
    "как кэширование влияет на latency приложения",
    "чем отличаются процессы от потоков",
    "как проектировать схему БД для чата",
    "что такое backpressure в очередях сообщений",
]

# Фразы-заполнители: для «длинных» вопросов пользователя и для ответов имитатора
_USER_FILLER = "Также приведи пример и объясни, почему это важно на практике."
_REPLY_FILLER = "Отвечаю кратко и по делу, без лишних деталей."


@dataclass
class Scenario:
    """Описание одного сценария диалога.

    Attributes:
        key: короткий идентификатор (short/long/over_limit/long_with_summary/
            window/facts/branching).
        title: человекочитаемое название.
        turns: число ходов (запросов пользователя).
        words_per_turn: примерная длина «длинных» сообщений в словах.
        window_messages: искусственное окно в сообщениях (None — из конфигурации).
        window_tokens: искусственное окно в токенах (None — не ограничивать).
        reply_words: длина синтетического ответа имитатора.
        summary: включить сжатие контекста (конспект старой части диалога).
        summary_keep_last: сколько последних сообщений идут «как есть».
        summary_every_n: размер блока сжатия.
        strategy: имя стратегии управления контекстом; ``None`` — старое
            поведение (без стратегий, summary + MAX_HISTORY_MESSAGES).
        strategy_window: размер окна стратегии (сообщений диалога).
        branch_strategy: подстратегия контекста для режима веток
            (пустая строка — взять ``strategy``).
        branch_mode: True — сценарий ветвления (checkpoint, изоляция веток).
    """

    key: str
    title: str
    turns: int
    words_per_turn: int = 20
    window_messages: Optional[int] = None
    window_tokens: Optional[int] = None
    reply_words: int = 30
    summary: bool = False
    summary_keep_last: int = 10
    summary_every_n: int = 10
    strategy: Optional[str] = None
    strategy_window: int = DEFAULT_STRATEGY_WINDOW
    branch_strategy: str = ""
    branch_mode: bool = False


@dataclass
class Options:
    """Параметры запуска (заполняются из CLI или в тестах)."""

    mode: str = "sim"
    yes: bool = False
    pause: float = 0.0
    verbose: bool = False
    json_path: Optional[str] = None
    storage_db: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    model_limit: int = 0
    api_key: str = ""
    no_summary: bool = False


# ---------------------------------------------------------------------------
# Генерация диалогов
# ---------------------------------------------------------------------------

def _pad_to_words(text: str, words: int, filler: str = _USER_FILLER) -> str:
    """Дополняет текст повтором нейтральной фразы до нужного числа слов."""
    result = text
    while len(result.split()) < words:
        result += " " + filler
    return result


def build_dialog(scenario: Scenario) -> List[str]:
    """Строит список сообщений пользователя для сценария.

    Первый ход содержит факт для проверки памяти, последний — проверочный
    вопрос.

    Args:
        scenario: параметры сценария.

    Returns:
        Список текстов сообщений пользователя (длина = scenario.turns).
    """
    turns: List[str] = [
        f"Привет! Меня зовут {FACT_NAME}, я живу в {FACT_CITY}, "
        f"мой любимый цвет — {FACT_COLOR}. Запомни это, пожалуйста."
    ]
    # «Рабочие» ходы (если ходов ≥ 3)
    for index in range(2, max(2, scenario.turns)):
        topic = _FILLER_TOPICS[(index - 2) % len(_FILLER_TOPICS)]
        text = f"Вопрос №{index}: расскажи подробнее про {topic}."
        turns.append(_pad_to_words(text, scenario.words_per_turn))
    # Проверочный ход
    if scenario.turns >= 2:
        turns.append(
            f"{PROBE_MARKER}: как меня зовут и какой у меня любимый цвет? "
            "Ответь одной строкой."
        )
    return turns[: scenario.turns]


def check_remembered(answer: str) -> bool:
    """Проверяет, что в ответе есть и имя, и цвет из первого хода."""
    lowered = (answer or "").lower()
    return FACT_NAME.lower() in lowered and FACT_COLOR.lower() in lowered


# ---------------------------------------------------------------------------
# Офлайн-имитация модели
# ---------------------------------------------------------------------------

class SimulatedModel:
    """Детерминированная замена ``_call_api`` для офлайн-сравнения.

    Имитатор видит ровно те сообщения, которые агент отправляет в API:
    если факт из первого хода вытеснен из истории — он отвечает «не помню».
    Токены считаются локальным счётчиком агента и помечаются как оценочные.
    """

    def __init__(self, agent: DeepSeekAgent, counter: Any,
                 reply_words: int = 30) -> None:
        self.agent = agent
        self.counter = counter
        self.reply_words = reply_words

    def __call__(self) -> Dict[str, Any]:
        """Возвращает ответ в формате ``_handle_response``.

        Имитатор видит ровно то, что агент отправляет в API (system-промпт,
        конспект, блок фактов и непокрытую часть истории), — поэтому «помнит»
        он только за счёт того, что реально попало в запрос. В режиме веток
        ответ на ``BRANCH_PROBE_MARKER`` зависит от истории активной ветки.
        """
        messages = self.agent._build_request()["messages"]
        last_user = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                last_user = message.get("content", "")
                break

        if BRANCH_PROBE_MARKER.lower() in last_user.lower():
            # Память ветки: отвечаем по тому, что реально есть в контексте
            # этой ветки (в чужие ветки сообщения не попадают).
            sent = "\n".join(
                str(m.get("content") or "") for m in messages
            ).lower()
            if BRANCH_A_PROJECT.lower() in sent:
                content = (f"В этой ветке проект — {BRANCH_A_PROJECT} "
                           f"({BRANCH_A_MARKER}).")
            elif BRANCH_B_PROJECT.lower() in sent:
                content = (f"В этой ветке проект — {BRANCH_B_PROJECT} "
                           f"({BRANCH_B_MARKER}).")
            else:
                content = "Не помню, о каком проекте речь в этой ветке."
        elif PROBE_MARKER.lower() in last_user.lower():
            known = any(
                FACT_NAME.lower() in (m.get("content") or "").lower()
                and FACT_COLOR.lower() in (m.get("content") or "").lower()
                for m in messages
            )
            if known:
                content = (f"Тебя зовут {FACT_NAME}, "
                           f"твой любимый цвет — {FACT_COLOR}.")
            else:
                content = ("Не помню ни имени, ни любимого цвета — в текущем "
                           "контексте этих данных нет.")
        else:
            preview = " ".join(last_user.split()[:12])
            content = _pad_to_words(
                f"Понял вопрос: «{preview}…». {_REPLY_FILLER}",
                self.reply_words,
                filler=_REPLY_FILLER,
            )

        prompt_tokens = self.counter.count_messages(messages)
        completion_tokens = self.counter.count_text(content)
        return {
            "content": content,
            "model": self.agent.model,
            "tokens": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                # Имитация: значения синтетические
                "estimated": True,
                "cache_hit_tokens": None,
                "cache_miss_tokens": None,
            },
            "finish_reason": "stop",
        }

    def service(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Имитация служебного запроса (сжатие контекста или факты).

        Различает запросы по маркерам:

        * ``SUMMARY_MARKER`` — возвращает синтетический конспект: он сохраняет
          факт из первого хода, если тот присутствует в переданном фрагменте
          или в предыдущем конспекте, — так офлайн-прогон показывает, что
          модель «помнит» благодаря сжатию, а не полной истории;
        * ``FACTS_MARKER`` — возвращает **валидный JSON** фактов
          (``{"goal": {...}, "preferences": {"имя": ..., "любимый_цвет": ...}}``):
          имя и цвет берутся из констант, поэтому проверочный вопрос проходит
          даже когда первое сообщение вытеснено узким окном.

        Returns:
            Ответ в формате ``_handle_response`` (content — строка).
        """
        joined = "\n".join(
            str(m.get("content") or "") for m in (request.get("messages") or [])
        )
        if FACTS_MARKER in joined:
            content = json.dumps(
                {
                    "goal": {"тема": "диалог с пользователем"},
                    "preferences": {
                        "имя": FACT_NAME,
                        "любимый_цвет": FACT_COLOR,
                    },
                },
                ensure_ascii=False,
            )
        elif SUMMARY_MARKER in joined:
            parts: List[str] = []
            if FACT_NAME.lower() in joined.lower() \
                    or FACT_COLOR.lower() in joined.lower():
                parts.append(
                    f"Факты: пользователя зовут {FACT_NAME}, любимый цвет — "
                    f"{FACT_COLOR} (сохранено из начала диалога)."
                )
            else:
                parts.append("Факты: существенные факты предыдущего диалога.")
            parts.append("Решения: продолжать диалог по темам пользователя.")
            parts.append("Открытые вопросы: нет.")
            content = " ".join(parts)
        else:
            # Страховка: служебный запрос обязан содержать маркер
            print("  [!] служебный запрос без SUMMARY_MARKER/FACTS_MARKER")
            content = json.dumps({}, ensure_ascii=False)

        prompt_tokens = self.counter.count_messages(request.get("messages") or [])
        completion_tokens = self.counter.count_text(content)
        return {
            "content": content,
            "model": self.agent.model,
            "tokens": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "estimated": True,
                "cache_hit_tokens": None,
                "cache_miss_tokens": None,
            },
            "finish_reason": "stop",
        }


# ---------------------------------------------------------------------------
# Искусственное ограничение контекстного окна
# ---------------------------------------------------------------------------

def enforce_token_window(agent: DeepSeekAgent, budget: int) -> int:
    """Урезает историю агента, пока оценка контекста не влезет в бюджет.

    Реализует искусственное «контекстное окно» в токенах: уменьшает лимит
    сообщений агента, пока оценка контекста превышает ``budget``.

    Важно: агент всегда оставляет хотя бы одно последнее сообщение диалога
    (``max_history_messages >= 1``), поэтому если бюджет меньше одного
    сообщения, история урезается до system + 1 сообщение — ровно как в
    реальных моделях, которые всё равно отправляют последний запрос.

    Args:
        agent: агент, чью историю нужно урезать.
        budget: бюджет контекста в токенах.

    Returns:
        Сколько сообщений диалога было вытеснено на этом шаге.
    """
    if budget <= 0:
        return 0
    trimmed = 0
    dialog_len = len(agent.history) - 1  # без system
    while dialog_len > 0 and agent.count_history_tokens() > budget:
        keep = max(1, dialog_len - 2)  # вытесняем самый старый обмен
        agent.update_config(max_history_messages=keep)
        new_len = len(agent.history) - 1
        if new_len >= dialog_len:  # страховка от зацикливания
            break
        trimmed += dialog_len - new_len
        dialog_len = new_len
    return trimmed


# ---------------------------------------------------------------------------
# Прогон сценариев
# ---------------------------------------------------------------------------

def _strategy_settings_for(scenario: Scenario) -> Optional[Dict[str, Any]]:
    """Собирает настройки стратегии контекста для сценария.

    Для режима веток (``strategy="branching"``) подстратегию задаёт
    ``branch_strategy`` (например ``sticky_facts``) — именно она собирает
    контекст каждой ветки.

    Args:
        scenario: параметры сценария.

    Returns:
        Словарь настроек либо None, если стратегия не задана (старое
        поведение: summary + MAX_HISTORY_MESSAGES).
    """
    if not scenario.strategy:
        return None
    return {
        "name": scenario.strategy,
        "window": int(scenario.strategy_window),
        "branch_strategy": scenario.branch_strategy or scenario.strategy,
        "branch_max": 20,
        "facts_enabled": True,
        "facts_model": "",
        "facts_temperature": 0.0,
        "facts_max_tokens": 300,
        "facts_max_items": 50,
        "facts_prompt": "",
    }


def _make_agent(scenario: Scenario, options: Options, config: Dict[str, Any],
                counter: Any, prices: Dict[str, Any]) -> DeepSeekAgent:
    """Создаёт агента под конкретный сценарий (окно, модель, хранилище).

    Если у сценария задана стратегия, агент переводится в режим стратегий
    (``strategy`` + ``strategy_settings``); сжатие контекста при этом не
    включается. Для сценариев без стратегии поведение прежнее: summary
    (если он нужен сценарию) и ограничение «сырой» истории.
    """
    storage = None
    if options.storage_db:
        db_path = Path(options.storage_db)
        if not db_path.is_absolute():
            db_path = PROJECT_ROOT / db_path
        storage = ContextStore(db_path=db_path, enabled=True)

    window = scenario.window_messages or int(config["MAX_HISTORY_MESSAGES"])
    # В sim-режиме реальных запросов нет, но OpenAI-клиент требует непустой
    # ключ на этапе конструирования — подставляем фиктивный.
    api_key = options.api_key or ("simulation-key" if options.mode == "sim" else "")

    strategy_settings = _strategy_settings_for(scenario)

    # Сжатие контекста включаем только для legacy-сценария со summary
    # (в режиме стратегий контекстом управляет стратегия).
    summary_settings = None
    if strategy_settings is None and scenario.summary \
            and not getattr(options, "no_summary", False):
        summary_settings = {
            "enabled": True,
            "keep_last": int(scenario.summary_keep_last),
            "every_n": int(scenario.summary_every_n),
            "max_tokens": 300,
            "temperature": 0.2,
            "model": "",
            "prompt": "",
        }

    return DeepSeekAgent(
        api_key=api_key,
        model=options.model or str(config["DEFAULT_MODEL"]),
        system_prompt=options.system_prompt or str(config["DEFAULT_SYSTEM_PROMPT"]),
        temperature=float(config["DEFAULT_TEMPERATURE"]),
        max_tokens=int(config["DEFAULT_MAX_TOKENS"]),
        max_history_messages=window,
        storage=storage,
        token_counter=counter,
        prices=prices,
        summary_settings=summary_settings,
        strategy=scenario.strategy,
        strategy_settings=strategy_settings,
    )


def _turn_metrics(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Собирает метрики управления контекстом после очередного хода.

    Args:
        stats: срез ``agent.get_token_stats()``.

    Returns:
        Словарь {"strategy", "facts", "branch"} — состояние стратегии,
        блока фактов и активной ветки на текущий момент.
    """
    session = stats.get("session") or {}
    strategy = stats.get("strategy") or {}
    facts = stats.get("facts") or {}
    branch = stats.get("branch") or {}
    return {
        "strategy": {
            "name": strategy.get("name"),
            "mode": strategy.get("mode"),
            "sub_strategy": strategy.get("sub_strategy"),
            "branching": bool(strategy.get("branching")),
            "window": strategy.get("window"),
            "error": strategy.get("error"),
        },
        "facts": {
            "active": bool(facts.get("active")),
            "enabled": bool(facts.get("enabled")),
            "count": int(facts.get("count") or 0),
            "tokens": int(facts.get("tokens") or 0),
            "requests": int(session.get("facts_requests", 0) or 0),
            "prompt_tokens": int(session.get("facts_prompt_tokens", 0) or 0),
            "completion_tokens": int(session.get("facts_completion_tokens", 0) or 0),
            "estimated_tokens": int(session.get("facts_estimated_tokens", 0) or 0),
            "cost_usd": session.get("facts_cost_usd"),
            "updated_at": facts.get("updated_at"),
            "error": facts.get("error"),
        },
        "branch": {
            "id": branch.get("id"),
            "name": branch.get("name"),
            "strategy": branch.get("strategy"),
            "messages": branch.get("messages"),
            "branches_count": branch.get("branches_count"),
        },
    }


def _do_turn(agent: DeepSeekAgent, question: str, number: int,
             scenario: Scenario, options: Options,
             turns: List[Dict[str, Any]], errors: List[str],
             state: Dict[str, int]) -> bool:
    """Выполняет один ход диалога и записывает его метрики в ``turns``.

    Args:
        agent: агент сценария.
        question: сообщение пользователя.
        number: номер хода (сквозной по сценарию).
        scenario: параметры сценария (искусственное окно в токенах).
        options: параметры запуска (пауза между ходами).
        turns: накопитель ходов (дополняется на месте).
        errors: накопитель ошибок (дополняется на месте).
        state: счётчики вытеснений/срабатываний окна (мутируется на месте).

    Returns:
        True, если ход выполнен; False — если агент вернул ошибку.
    """
    result = agent.send_message(question)
    if result.get("error"):
        errors.append(f"ход {number}: {result['error']}")
        return False

    # Искусственное ограничение окна в токенах (после обмена)
    if scenario.window_tokens:
        trimmed = enforce_token_window(agent, scenario.window_tokens)
        if trimmed:
            state["trimmed_total"] = state.get("trimmed_total", 0) + int(trimmed)
            state["window_hits"] = state.get("window_hits", 0) + 1

    stats = agent.get_token_stats()
    tokens = result.get("tokens") or {}
    turn: Dict[str, Any] = {
        "turn": number,
        "question": question,
        "answer": result.get("text") or "",
        "prompt_tokens": tokens.get("prompt_tokens"),
        "completion_tokens": tokens.get("completion_tokens"),
        "total_tokens": tokens.get("total_tokens"),
        "estimated": tokens.get("estimated"),
        "context_tokens": stats["context"]["tokens"],
        "session_total": stats["session"]["total_with_estimates"],
        "cost_usd": tokens.get("cost_usd"),
        "response_time": result.get("response_time"),
    }
    # Метрики стратегии/фактов/веток после каждого хода
    turn.update(_turn_metrics(stats))
    turns.append(turn)

    if options.pause:
        time.sleep(options.pause)
    return True


# ---------------------------------------------------------------------------
# Сценарий ветвления
# ---------------------------------------------------------------------------

def _root_branch_id(agent: DeepSeekAgent) -> int:
    """Идентификатор корневой ветки (``main``, обычно id=1)."""
    branches = agent.list_branches() or []
    for branch in branches:
        if int(branch.get("id") or 0) == 1:
            return 1
    for branch in branches:
        if branch.get("parent_id") in (None, 0):
            return int(branch.get("id") or 1)
    return int(branches[0].get("id") or 1) if branches else 1


def _branch_messages() -> Dict[str, List[str]]:
    """Сообщения продолжения каждой ветки: свой факт, затем проверка памяти."""
    return {
        "a": [
            f"{BRANCH_A_MARKER}: в этой ветке мой проект — {BRANCH_A_PROJECT}. "
            f"Запомни: проект — {BRANCH_A_PROJECT}.",
            f"{BRANCH_PROBE_MARKER}: какой проект в этой ветке? "
            "Ответь одной строкой.",
        ],
        "b": [
            f"{BRANCH_B_MARKER}: в этой ветке мой проект — {BRANCH_B_PROJECT}. "
            f"Запомни: проект — {BRANCH_B_PROJECT}.",
            f"{BRANCH_PROBE_MARKER}: какой проект в этой ветке? "
            "Ответь одной строкой.",
        ],
    }


def _branch_text(messages: List[Dict[str, Any]]) -> str:
    """Склеенный текст сообщений ветки (в нижнем регистре)."""
    return "\n".join(str(m.get("content") or "") for m in messages).lower()


def check_branch_isolation(history_a: List[Dict[str, Any]],
                           history_b: List[Dict[str, Any]]) -> bool:
    """Проверяет изоляцию веток: сообщения одной не встречаются в другой.

    Дополнительно требуется, чтобы каждая ветка содержала **свой** факт
    (маркер и проект), то есть «помнила своё».

    Args:
        history_a: сообщения ветки A.
        history_b: сообщения ветки B.

    Returns:
        True, если ветки изолированы и каждая помнит свой факт.
    """
    text_a = _branch_text(history_a)
    text_b = _branch_text(history_b)
    a_marker = BRANCH_A_MARKER.lower()
    b_marker = BRANCH_B_MARKER.lower()
    a_project = BRANCH_A_PROJECT.lower()
    b_project = BRANCH_B_PROJECT.lower()
    a_own = a_marker in text_a and a_project in text_a
    a_clean = b_marker not in text_a and b_project not in text_a
    b_own = b_marker in text_b and b_project in text_b
    b_clean = a_marker not in text_b and a_project not in text_b
    return bool(a_own and a_clean and b_own and b_clean)


def _run_branch_flow(agent: DeepSeekAgent, scenario: Scenario, options: Options,
                     turns: List[Dict[str, Any]], errors: List[str],
                     state: Dict[str, int]) -> Dict[str, Any]:
    """Выполняет сценарий ветвления: checkpoint, две ветки, изоляция.

    Порядок:
    1. несколько ходов в корневой ветке (``main``) — «разогрев»;
    2. ``agent.checkpoint(name="ветка-A")`` — ветка-копия диалога;
    3. продолжение ветки A своим набором сообщений;
    4. ``agent.switch_branch(1)`` — возврат в корневую ветку (B) и её
       собственное продолжение;
    5. переключение обратно в A и проверка изоляции: сообщения одной ветки
       не встречаются в другой, и каждая ветка «помнит своё».

    Returns:
        Отчёт о ветках: id/имя веток, число веток, список веток
        (name, messages), результаты проверки изоляции и памяти веток.
    """
    output: Dict[str, Any] = {
        "mode": True,
        "root_branch_id": _root_branch_id(agent),
        "branch_a": None,
        "branch_b": None,
        "warmup_turns": 0,
        "turns_planned": 0,
        "branches": [],
        "branches_count": 0,
        "isolation_ok": False,
        "isolation_errors": [],
        "remembers_own": {"a": False, "b": False},
        "answers": {"a": "", "b": ""},
    }
    dialog = build_dialog(scenario)
    # Оставляем место под продолжения веток A и B (по два хода каждое)
    warmup = max(1, min(BRANCH_WARMUP_TURNS, int(scenario.turns) - 4))
    messages = _branch_messages()
    output["warmup_turns"] = warmup
    output["turns_planned"] = (warmup + len(messages["a"])
                              + len(messages["b"]))

    number = 0
    # 1. Разогрев в корневой ветке: факт из начала диалога + рабочие ходы
    for question in dialog[:warmup]:
        number += 1
        if not _do_turn(agent, question, number, scenario, options,
                        turns, errors, state):
            return output

    # 2. Checkpoint: ветка A — копия текущего диалога
    created = agent.checkpoint(name=BRANCH_A_NAME)
    branch = created.get("branch") or {}
    if created.get("error") or not branch.get("id"):
        errors.append(f"checkpoint: {created.get('error') or 'ветка не создана'}")
        return output
    branch_a_id = int(branch["id"])
    output["branch_a"] = {"id": branch_a_id, "name": branch.get("name")}

    # 3. Продолжение ветки A своим набором сообщений
    for question in messages["a"]:
        number += 1
        if not _do_turn(agent, question, number, scenario, options,
                        turns, errors, state):
            return output
    output["answers"]["a"] = turns[-1]["answer"] if turns else ""

    # 4. Возврат в корневую ветку (B) и её собственное продолжение
    root_id = int(output["root_branch_id"])
    switched = agent.switch_branch(root_id)
    branch_b = switched.get("branch") or {}
    if switched.get("error"):
        errors.append(f"switch_branch({root_id}): {switched['error']}")
        return output
    output["branch_b"] = {"id": int(branch_b.get("id") or root_id),
                          "name": branch_b.get("name")}
    for question in messages["b"]:
        number += 1
        if not _do_turn(agent, question, number, scenario, options,
                        turns, errors, state):
            return output
    history_b = [dict(m) for m in agent.history]
    output["answers"]["b"] = turns[-1]["answer"] if turns else ""

    # 5. Переключение обратно в A и проверка изоляции
    back = agent.switch_branch(branch_a_id)
    if back.get("error"):
        errors.append(f"switch_branch({branch_a_id}): {back['error']}")
        return output
    history_a = [dict(m) for m in agent.history]

    output["branches"] = [
        {
            "id": int(item.get("id") or 0),
            "name": item.get("name"),
            "parent_id": item.get("parent_id"),
            "messages": int(item.get("messages") or 0),
            "active": bool(item.get("active")),
            "facts_count": int(item.get("facts_count") or 0),
        }
        for item in (agent.list_branches() or [])
    ]
    output["branches_count"] = len(output["branches"])
    output["isolation_ok"] = check_branch_isolation(history_a, history_b)
    if not output["isolation_ok"]:
        output["isolation_errors"].append(
            "сообщения веток пересекаются или ветка не помнит свой факт"
        )
    output["remembers_own"] = {
        "a": BRANCH_A_PROJECT.lower() in str(output["answers"]["a"]).lower(),
        "b": BRANCH_B_PROJECT.lower() in str(output["answers"]["b"]).lower(),
    }
    return output


def run_scenario(scenario: Scenario, options: Options, config: Dict[str, Any],
                 counter: Any, prices: Dict[str, Any]) -> Dict[str, Any]:
    """Прогоняет один сценарий диалога и собирает метрики.

    Для сценариев со стратегией агент создаётся с ``strategy`` и
    ``strategy_settings``; для ``branch_mode`` дополнительно выполняется
    сценарий ветвления (checkpoint, две ветки, проверка изоляции).

    Returns:
        Словарь с полями: ключ/название сценария, список ходов (вопрос,
        ответ, токены, время, метрики стратегии/фактов/веток), сводка
        (контекст, сессия, стоимость, вытесненные сообщения, «помнит ли
        факт», стратегия, факты, ветки, изоляция).
    """
    agent = _make_agent(scenario, options, config, counter, prices)
    simulated: Optional[SimulatedModel] = None
    if options.mode == "sim":
        simulated = SimulatedModel(agent, counter, reply_words=scenario.reply_words)
        agent._call_api = simulated           # обычный ответ диалога
        agent._call_service_api = simulated.service  # служебный запрос

    turns: List[Dict[str, Any]] = []
    errors: List[str] = []
    state: Dict[str, int] = {"trimmed_total": 0, "window_hits": 0}

    dialog = build_dialog(scenario)
    branch_report: Optional[Dict[str, Any]] = None
    if scenario.branch_mode:
        branch_report = _run_branch_flow(agent, scenario, options, turns,
                                         errors, state)
        planned = int(branch_report.get("turns_planned") or len(dialog))
    else:
        planned = len(dialog)
        for number, question in enumerate(dialog, start=1):
            if not _do_turn(agent, question, number, scenario, options,
                            turns, errors, state):
                break

    max_context = max((int(t["context_tokens"]) for t in turns), default=0)
    dialog_messages = max(0, len(agent.history) - 1)
    # Всего добавлено сообщений: вопрос + ответ за каждый успешный ход.
    # Разница с текущей историей — вытесненные сообщения (окно агента
    # и/или искусственное ограничение в токенах).
    dropped = max(0, 2 * len(turns) - dialog_messages)
    stats = agent.get_token_stats()
    metrics = _turn_metrics(stats)

    if branch_report is not None:
        probe_answer = str(branch_report["answers"].get("a") or "")
        remembered: Optional[bool] = bool(
            branch_report["remembers_own"].get("a")
            and branch_report["remembers_own"].get("b")
        )
        isolation_ok: Optional[bool] = bool(branch_report["isolation_ok"])
    else:
        probe_answer = turns[-1]["answer"] if turns else ""
        remembered = check_remembered(probe_answer) if turns else None
        isolation_ok = None

    # Блок веток: срез активной ветки + (для branch_mode) полный отчёт
    branch_block: Dict[str, Any] = dict(metrics["branch"])
    if branch_report is not None:
        branch_block.update({
            "mode": True,
            "root_id": branch_report["root_branch_id"],
            "warmup_turns": branch_report["warmup_turns"],
            "list": branch_report["branches"],
            "branches_count": branch_report["branches_count"],
            "isolation_ok": branch_report["isolation_ok"],
            "isolation_errors": branch_report["isolation_errors"],
            "remembers_own": branch_report["remembers_own"],
            "answers": branch_report["answers"],
        })
        for item in branch_report["branches"]:
            if int(item.get("id") or 0) == int(branch_block.get("id") or 0):
                branch_block["messages"] = item.get("messages")
                branch_block["branches_count"] = branch_report["branches_count"]

    summary = {
        "turns_done": len(turns),
        "turns_planned": planned,
        "messages_kept": dialog_messages,
        "messages_dropped": dropped,
        "last_prompt_tokens": turns[-1]["prompt_tokens"] if turns else None,
        "last_completion_tokens": turns[-1]["completion_tokens"] if turns else None,
        "max_context_tokens": max_context,
        "final_context_tokens": stats["context"]["tokens"],
        "session_total_tokens": stats["session"]["total_with_estimates"],
        "session_cost_usd": stats["session"]["cost_usd"],
        "session_cost_estimated_usd": stats["session"]["estimated_cost_usd"],
        "prices_set": stats["session"]["prices_set"],
        "estimated_any": stats["session"]["estimated_total_tokens"] > 0,
        "remembered": remembered,
        "probe_answer": probe_answer,
        "window_messages": scenario.window_messages,
        "window_tokens": scenario.window_tokens,
        "window_tokens_hits": state.get("window_hits", 0),
        "token_window_trimmed": state.get("trimmed_total", 0),
        # Стратегия управления контекстом
        "strategy": dict(metrics["strategy"]),
        # Блок фактов (sticky facts)
        "facts": dict(metrics["facts"]),
        "facts_requests": int(metrics["facts"]["requests"]),
        "facts_service_tokens": int(metrics["facts"]["prompt_tokens"])
        + int(metrics["facts"]["completion_tokens"])
        + int(metrics["facts"]["estimated_tokens"]),
        "facts_cost_usd": metrics["facts"]["cost_usd"],
        # Ветки диалога
        "branch": branch_block,
        "branch_mode": bool(scenario.branch_mode),
        "isolation_ok": isolation_ok,
        # Сжатие контекста (summary)
        "summary_enabled": bool(stats["summary"].get("enabled")),
        "summary_active": bool(stats["summary"].get("active")),
        "summary_covered": int(stats["summary"].get("covered", 0) or 0),
        "summary_tokens": int(stats["summary"].get("tokens", 0) or 0),
        "summary_requests": int(stats["session"].get("summary_requests", 0) or 0),
        "summary_service_tokens": int(
            stats["session"].get("summary_prompt_tokens", 0) or 0
        ) + int(stats["session"].get("summary_completion_tokens", 0) or 0)
        + int(stats["session"].get("summary_estimated_tokens", 0) or 0),
        "summary_cost_usd": stats["session"].get("summary_cost_usd"),
        "summary_error": stats["summary"].get("error"),
        "errors": errors,
    }
    agent.close()
    return {
        "key": scenario.key,
        "title": scenario.title,
        "scenario": asdict(scenario),
        "turns": turns,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------------

def _fmt(value: Any, suffix: str = "") -> str:
    """Форматирует значение для отчёта («—» для None)."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value}{suffix}"
    return f"{value}{suffix}"


def _fmt_cost(summary: Dict[str, Any]) -> str:
    """Стоимость сессии (диалог + служебные запросы) или «—»."""
    if not summary.get("prices_set"):
        return "—"
    cost = (
        float(summary.get("session_cost_usd") or 0.0)
        + float(summary.get("session_cost_estimated_usd") or 0.0)
        + float(summary.get("summary_cost_usd") or 0.0)
        + float(summary.get("facts_cost_usd") or 0.0)
    )
    prefix = "≈" if summary.get("estimated_any") else ""
    return f"{prefix}${cost:.6f}"


def _strategy_label(summary: Dict[str, Any]) -> str:
    """Короткая подпись стратегии для сравнительной таблицы."""
    short_names = {SLIDING_WINDOW: "window", STICKY_FACTS: "facts",
                   BRANCHING: "branching"}
    strategy = summary.get("strategy") or {}
    name = strategy.get("name")
    if not name:
        return "summary" if summary.get("summary_enabled") else "legacy"
    label = short_names.get(str(name), str(name))
    sub = strategy.get("sub_strategy")
    if strategy.get("branching") and sub:
        label = f"{label}/{short_names.get(str(sub), str(sub))}"
    return label


def service_requests(summary: Dict[str, Any]) -> int:
    """Число служебных запросов (сжатие + обновление фактов)."""
    return int(summary.get("summary_requests", 0) or 0) \
        + int(summary.get("facts_requests", 0) or 0)


def print_scenario_report(result: Dict[str, Any], options: Options) -> None:
    """Печатает отчёт по одному сценарию."""
    summary = result["summary"]
    print(f"\n=== {result['key']} — {result['title']} ===")
    strategy = summary.get("strategy") or {}
    if strategy.get("name"):
        window_note = ""
        if strategy.get("window"):
            window_note = f", окно {strategy['window']} сообщений"
        branching_note = ""
        if strategy.get("branching"):
            branching_note = f", подстратегия {strategy.get('sub_strategy')}"
        print(f"  Стратегия: {strategy['name']}{branching_note}{window_note}")
        if strategy.get("error"):
            print(f"  Ошибка стратегии: {strategy['error']}")
    window = []
    if summary["window_messages"]:
        window.append(f"{summary['window_messages']} сообщений")
    if summary["window_tokens"]:
        window.append(f"{summary['window_tokens']} токенов")
    if window:
        print(f"  Искусственное окно: {', '.join(window)}"
              f" (сработало {summary['window_tokens_hits']} раз)")
    print(f"  Ходов: {summary['turns_done']}/{summary['turns_planned']} | "
          f"сообщений в истории: {summary['messages_kept']} | "
          f"вытеснено: {summary['messages_dropped']}")
    print(f"  Последний запрос: {_fmt(summary['last_prompt_tokens'])} токенов | "
          f"ответ: {_fmt(summary['last_completion_tokens'])} токенов")
    print(f"  Контекст: ≈{summary['final_context_tokens']} токенов "
          f"(максимум за диалог: ≈{summary['max_context_tokens']})")
    facts = summary.get("facts") or {}
    if facts.get("enabled") or facts.get("count"):
        cost_note = ""
        if facts.get("cost_usd") is not None:
            cost_note = f", факты ≈${float(facts['cost_usd']):.6f}"
        print(f"  Факты: {facts.get('count', 0)} шт., "
              f"≈{facts.get('tokens', 0)} токенов в контексте | "
              f"служебных запросов: {facts.get('requests', 0)} "
              f"(≈{summary.get('facts_service_tokens', 0)} токенов{cost_note})")
        if facts.get("error"):
            print(f"  Ошибка обновления фактов: {facts['error']}")
    if summary.get("summary_enabled"):
        if summary.get("summary_active"):
            cost_note = ""
            if summary.get("summary_cost_usd") is not None:
                cost_note = f", сжатие ≈${float(summary['summary_cost_usd']):.6f}"
            print(f"  Конспект: ≈{summary['summary_tokens']} токенов, "
                  f"покрыто {summary['summary_covered']} сообщ., "
                  f"служебных запросов: {summary['summary_requests']} "
                  f"(≈{summary['summary_service_tokens']} токенов{cost_note})")
        else:
            print("  Конспект: ещё не создан "
                  f"(служебных запросов: {summary['summary_requests']})")
        if summary.get("summary_error"):
            print(f"  Ошибка сжатия: {summary['summary_error']}")
    branch = summary.get("branch") or {}
    if branch.get("mode"):
        branch_a_id = next(
            (item.get("id") for item in (branch.get("list") or [])
             if item.get("name") == BRANCH_A_NAME),
            "—",
        )
        print(f"  Ветки: {branch.get('branches_count', 0)} "
              f"(корневая id={branch.get('root_id')}, "
              f"checkpoint «{BRANCH_A_NAME}» id={branch_a_id})")
        for item in branch.get("list") or []:
            active_note = " *" if item.get("active") else ""
            print(f"    - id={item.get('id')} «{item.get('name')}»: "
                  f"{item.get('messages')} сообщ., "
                  f"фактов: {item.get('facts_count', 0)}{active_note}")
        own = branch.get("remembers_own") or {}
        print(f"  Ветка A помнит своё: {'ДА' if own.get('a') else 'НЕТ'} | "
              f"ветка B помнит своё: {'ДА' if own.get('b') else 'НЕТ'}")
        print(f"  Изоляция веток: "
              f"{'ПОДТВЕРЖДЕНА' if branch.get('isolation_ok') else 'НАРУШЕНА'}")
        for answer_key, answer in (branch.get("answers") or {}).items():
            preview = " ".join(str(answer).split())[:120]
            print(f"    ответ ветки {answer_key.upper()}: {preview}")
        for problem in branch.get("isolation_errors") or []:
            print(f"    [!] {problem}")
    print(f"  Сессия: {'≈' if summary['estimated_any'] else ''}"
          f"{summary['session_total_tokens']} токенов | стоимость: "
          f"{_fmt_cost(summary)}")
    if summary["remembered"] is None:
        print("  Помнит факт из начала диалога: —")
    elif summary.get("branch_mode"):
        print("  Помнит факт из начала диалога: "
              f"{'ДА' if summary['remembered'] else 'НЕТ'} "
              "(каждая ветка помнит своё)")
    else:
        print("  Помнит факт из начала диалога: "
              f"{'ДА' if summary['remembered'] else 'НЕТ'}")
    if summary["probe_answer"]:
        preview = " ".join(summary["probe_answer"].split())[:220]
        print(f"  Ответ на проверочный вопрос: {preview}")
    if summary["errors"]:
        print(f"  Ошибки: {'; '.join(summary['errors'])}")

    if options.verbose:
        print("  --- по ходам ---")
        for turn in result["turns"]:
            q = " ".join(turn["question"].split())[:48]
            a = " ".join(turn["answer"].split())[:48]
            print(f"  #{turn['turn']:>2} in≈{_fmt(turn['prompt_tokens'])} "
                  f"out≈{_fmt(turn['completion_tokens'])} "
                  f"ctx≈{turn['context_tokens']:>4} | {q}… → {a}…")


def print_comparison(results: List[Dict[str, Any]]) -> None:
    """Печатает сравнительную таблицу по всем сценариям.

    Колонки: сценарий, стратегия, ходов, максимум контекста, токены сессии,
    стоимость, «помнит ли факт», число веток, служебных запросов
    (сжатие + факты) и вытесненных сообщений.
    """
    print("\n=== Сравнение ===")
    header = (f"{'Сценарий':<19}{'Стратегия':<17}{'Ходов':>6}"
              f"{'Контекст(макс)':>16}{'Сессия':>10}{'Стоимость':>14}"
              f"{'Помнит':>8}{'Веток':>7}{'Служебных запросов':>19}"
              f"{'Вытеснено':>11}")
    print(header)
    print("-" * len(header))
    for result in results:
        summary = result["summary"]
        remembered = summary["remembered"]
        remembered_text = "—" if remembered is None else ("да" if remembered else "нет")
        branch = summary.get("branch") or {}
        branches = int(branch.get("branches_count") or 0)
        branches_text = str(branches) if branches else "—"
        print(
            f"{result['key']:<19}"
            f"{_strategy_label(summary):<17}"
            f"{summary['turns_done']:>6}"
            f"{('≈' + str(summary['max_context_tokens'])):>16}"
            f"{('≈' if summary['estimated_any'] else '') + str(summary['session_total_tokens']):>10}"
            f"{_fmt_cost(summary):>14}"
            f"{remembered_text:>8}"
            f"{branches_text:>7}"
            f"{service_requests(summary):>19}"
            f"{summary['messages_dropped']:>11}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

#: Ключи CLI для сценариев стратегий и их канонические имена
STRATEGY_SCENARIOS = {
    "window": SLIDING_WINDOW,
    "facts": STICKY_FACTS,
    "branching": BRANCHING,
}


def selected_strategies(args: argparse.Namespace) -> List[str]:
    """Разбирает ``--strategies`` (какие сценарии стратегий запускать).

    Понимает имена и псевдонимы стратегий (``window``/``sliding_window``,
    ``facts``/``sticky_facts``, ``branching``/``branch``). Пустая строка,
    ``none`` или ``--no-strategies`` отключают сценарии стратегий.

    Args:
        args: разобранные аргументы командной строки.

    Returns:
        Список ключей сценариев в порядке перечисления (``window``, ``facts``,
        ``branching``), без повторов.
    """
    if getattr(args, "no_strategies", False):
        return []
    raw = str(getattr(args, "strategies", "") or "").strip()
    if not raw or raw.lower() in ("none", "off", "нет", "-"):
        return []
    selected: List[str] = []
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        try:
            canonical = normalize_name(token)
        except Exception:
            print(f"[!] Неизвестный сценарий стратегии: {token!r} — пропускаю.")
            continue
        for key, name in STRATEGY_SCENARIOS.items():
            if name == canonical and key not in selected:
                selected.append(key)
    return selected


def build_scenarios(args: argparse.Namespace) -> List[Scenario]:
    """Собирает сценарии с учётом параметров командной строки.

    Legacy-сценарии (``short``, ``long``, ``over_limit`` и
    ``long_with_summary``) создаются без стратегии — агент работает в прежнем
    режиме (summary + ``MAX_HISTORY_MESSAGES``). Сценарии стратегий
    (``window``, ``facts``, ``branching``) включаются через ``--strategies``:

    * ``long_with_summary`` показывает управление контекстом через конспект;
    * ``window`` — скользящее окно: факт из начала диалога теряется;
    * ``facts`` — липкие факты: факт сохраняется блоком фактов, видны
      служебные запросы и их токены;
    * ``branching`` — ветки: checkpoint, изоляция веток, каждая ветка
      помнит своё.
    """
    long_turns = args.turns
    scenarios = [
        Scenario(key="short", title="короткий диалог",
                 turns=3, words_per_turn=10, reply_words=20),
        Scenario(key="long", title="длинный диалог (в пределах окна)",
                 turns=long_turns, words_per_turn=args.words, reply_words=40),
        Scenario(key="over_limit", title="диалог, превышающий окно",
                 turns=long_turns, words_per_turn=args.words,
                 window_messages=args.window_messages,
                 window_tokens=args.window_tokens, reply_words=40),
    ]
    if not getattr(args, "no_summary", False):
        scenarios.append(
            Scenario(key="long_with_summary",
                     title="длинный диалог со сжатием контекста",
                     turns=long_turns, words_per_turn=args.words,
                     reply_words=40, summary=True,
                     summary_keep_last=getattr(args, "summary_keep_last", 10),
                     summary_every_n=getattr(args, "summary_every_n", 10))
        )

    strategy_turns = int(getattr(args, "strategy_turns", DEFAULT_STRATEGY_TURNS))
    strategy_window = int(getattr(args, "strategy_window",
                                 DEFAULT_STRATEGY_WINDOW))
    for key in selected_strategies(args):
        if key == "window":
            scenarios.append(
                Scenario(key="window", title="скользящее окно (факт теряется)",
                         turns=strategy_turns, words_per_turn=args.words,
                         reply_words=40, strategy=SLIDING_WINDOW,
                         strategy_window=strategy_window)
            )
        elif key == "facts":
            scenarios.append(
                Scenario(key="facts", title="липкие факты (факт сохраняется)",
                         turns=strategy_turns, words_per_turn=args.words,
                         reply_words=40, strategy=STICKY_FACTS,
                         strategy_window=strategy_window)
            )
        elif key == "branching":
            scenarios.append(
                Scenario(key="branching", title="ветки диалога (изоляция)",
                         turns=max(8, BRANCH_WARMUP_TURNS + 4),
                         words_per_turn=args.words, reply_words=40,
                         strategy=BRANCHING,
                         branch_strategy=DEFAULT_BRANCH_SUB_STRATEGY,
                         strategy_window=strategy_window, branch_mode=True)
            )
    return scenarios


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Бенчмарк диалогов: legacy-сценарии (short/long/over_limit) "
                    "и стратегии контекста (window/facts/branching).",
    )
    parser.add_argument("--mode", choices=("sim", "live", "auto"), default="sim",
                        help="sim — офлайн-имитация (по умолчанию), live — "
                             "реальные запросы, auto — live при наличии ключа")
    parser.add_argument("--yes", action="store_true",
                        help="подтвердить реальные запросы к API (для --mode live)")
    parser.add_argument("--turns", type=int, default=14,
                        help="число ходов в длинном сценарии (по умолчанию 14)")
    parser.add_argument("--words", type=int, default=25,
                        help="примерная длина «длинных» сообщений в словах")
    parser.add_argument("--window-messages", type=int, default=6,
                        help="искусственное окно в сообщениях для over_limit")
    parser.add_argument("--window-tokens", type=int, default=400,
                        help="искусственное окно в токенах для over_limit")
    parser.add_argument("--no-summary", action="store_true",
                        help="не запускать сценарий со сжатием контекста")
    parser.add_argument("--summary-keep-last", type=int, default=10,
                        help="сколько последних сообщений не сжимать")
    parser.add_argument("--summary-every-n", type=int, default=10,
                        help="размер блока сжатия (сообщений)")
    parser.add_argument("--strategies", nargs="?", const="",
                        default="window,facts,branching",
                        help="сценарии стратегий через запятую: "
                             "window, facts, branching (по умолчанию все три); "
                             "пустая строка или none — ни один")
    parser.add_argument("--no-strategies", action="store_true",
                        help="не запускать сценарии стратегий (только legacy)")
    parser.add_argument("--strategy-turns", type=int, default=DEFAULT_STRATEGY_TURNS,
                        help="число ходов в сценариях window/facts "
                             f"(по умолчанию {DEFAULT_STRATEGY_TURNS})")
    parser.add_argument("--strategy-window", type=int,
                        default=DEFAULT_STRATEGY_WINDOW,
                        help="размер окна стратегии в сообщениях "
                             f"(по умолчанию {DEFAULT_STRATEGY_WINDOW})")
    parser.add_argument("--pause", type=float, default=0.0,
                        help="пауза между запросами, секунды")
    parser.add_argument("--model", default=None, help="имя модели")
    parser.add_argument("--price-input", type=float, default=None,
                        help="цена 1M входных токенов, USD (переопределяет .env)")
    parser.add_argument("--price-output", type=float, default=None,
                        help="цена 1M выходных токенов, USD (переопределяет .env)")
    parser.add_argument("--model-limit", type=int, default=0,
                        help="реальный лимит модели в токенах (для справки)")
    parser.add_argument("--storage-db", default=None,
                        help="включить персист в указанный файл БД (по умолчанию выкл.)")
    parser.add_argument("--json", dest="json_path", default=None,
                        help="сохранить отчёт в JSON")
    parser.add_argument("--verbose", action="store_true",
                        help="печатать метрики по каждому ходу")
    parser.add_argument("--dry-run", action="store_true",
                        help="показать план диалогов и выйти (без запросов)")
    return parser.parse_args(argv)


def confirm_live(options: Options, config: Dict[str, Any]) -> bool:
    """Запрашивает подтверждение платных запросов к API."""
    if options.yes:
        return True
    print("\n[!] Режим live: будут выполнены РЕАЛЬНЫЕ запросы к DeepSeek API "
          "(расходуются токены вашего аккаунта).")
    if not sys.stdin.isatty():
        print("Подтвердите флагом --yes и запустите снова.")
        return False
    answer = input("Продолжить? [y/N]: ").strip().lower()
    return answer in ("y", "yes", "д")


def print_plan(scenarios: List[Scenario], options: Options, config: Dict[str, Any],
               counter: Any) -> None:
    """Печатает план диалогов (для --dry-run и перед запуском)."""
    print(f"Модель: {options.model or config['DEFAULT_MODEL']} | "
          f"счётчик: {counter.name} | режим: {options.mode}")
    for scenario in scenarios:
        dialog = build_dialog(scenario)
        window = []
        if scenario.window_messages:
            window.append(f"{scenario.window_messages} сообщений")
        if scenario.window_tokens:
            window.append(f"{scenario.window_tokens} токенов")
        if scenario.strategy:
            if scenario.branch_mode:
                window.append(f"стратегия branching/"
                              f"{scenario.branch_strategy or scenario.strategy}")
            else:
                window.append(f"стратегия {scenario.strategy}")
            window.append(f"окно стратегии {scenario.strategy_window}")
        messages = _branch_messages() if scenario.branch_mode else {}
        warmup = max(1, min(BRANCH_WARMUP_TURNS, int(scenario.turns) - 4))
        branch_turns = len(messages.get("a") or []) + len(messages.get("b") or [])
        planned = warmup + branch_turns if scenario.branch_mode else len(dialog)
        # В режиме веток «сырой» план — только разогрев: продолжения веток
        # задаются отдельными сообщениями (см. ниже).
        shown = dialog[:warmup] if scenario.branch_mode else dialog
        print(f"\n[{scenario.key}] {scenario.title}: ходов {planned}, "
              f"слов/ход ≈{scenario.words_per_turn}"
              + (f", окно: {', '.join(window)}" if window else ", окно: из конфигурации"))
        for number, text in enumerate(shown, start=1):
            preview = " ".join(text.split())
            if len(preview) > 90:
                preview = preview[:87] + "..."
            print(f"   {number:>2}. {preview}")
        if scenario.branch_mode:
            print(f"   ветвление: разогрев {warmup} ход(а) → "
                  f"checkpoint «{BRANCH_A_NAME}» → своё продолжение в A "
                  f"({len(messages['a'])} хода) → switch на main → своё "
                  f"продолжение в B ({len(messages['b'])} хода) → проверка изоляции")
            for label, texts in (("A", messages["a"]), ("B", messages["b"])):
                for text in texts:
                    preview = " ".join(text.split())
                    if len(preview) > 90:
                        preview = preview[:87] + "..."
                    print(f"   ветка {label}: {preview}")


def _configure_stdio() -> None:
    """Не ронять скрипт из-за кодировки консоли (Windows + перенаправление).

    В отчётах есть символы вроде «≈», которых нет в cp1251/cp866: при выводе
    в pipe Python падает с UnicodeEncodeError. Заменяем такие символы, а не
    прерываем прогон (искажение кириллицы в консоли — известное ограничение).
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError, OSError):
            pass


def main(argv: Optional[List[str]] = None) -> int:
    """Точка входа скрипта."""
    _configure_stdio()
    args = parse_args(argv)
    try:
        config = get_config()
    except Exception as exc:
        print(f"[ОШИБКА] Не удалось загрузить конфигурацию: {exc}", file=sys.stderr)
        return 1

    prices = get_prices(config)
    # CLI-цены переопределяют значения из .env (удобно для сравнения)
    if args.price_input is not None:
        prices["default"]["input"] = float(args.price_input)
    if args.price_output is not None:
        prices["default"]["output"] = float(args.price_output)
    prices_set = bool(prices["default"]["input"] or prices["default"]["output"])
    counter = create_counter(
        model=args.model or str(config["DEFAULT_MODEL"]),
        enabled=bool(config.get("TOKENIZER_ENABLED", True)),
        encoding_name=str(config.get("TOKENIZER_ENCODING", "cl100k_base")),
    )

    options = Options(
        mode=args.mode,
        yes=args.yes,
        pause=args.pause,
        verbose=args.verbose,
        json_path=args.json_path,
        storage_db=args.storage_db,
        model=args.model,
        model_limit=args.model_limit,
        api_key=str(config.get("DEEPSEEK_API_KEY") or ""),
        no_summary=bool(args.no_summary),
    )

    # Выбор режима
    if options.mode == "auto":
        options.mode = "live" if options.api_key.strip() else "sim"
    if options.mode == "live" and not options.api_key.strip():
        print("[!] Ключ DEEPSEEK_API_KEY не найден — переключаюсь в sim-режим.")
        options.mode = "sim"

    scenarios = build_scenarios(args)

    if args.dry_run:
        print_plan(scenarios, options, config, counter)
        return 0

    print_plan(scenarios, options, config, counter)
    if prices_set:
        print(f"\nЦены: вход ${prices['default']['input']}/1M, "
              f"выход ${prices['default']['output']}/1M "
              f"| хранилище: {args.storage_db or 'выключено'}")
    else:
        print("\nЦены не заданы — стоимость не считается. Укажите "
              "--price-input/--price-output или PRICE_* в .env "
              f"| хранилище: {args.storage_db or 'выключено'}")

    if options.mode == "live" and not confirm_live(options, config):
        return 2
    if options.mode == "live":
        options.pause = options.pause or 0.5

    results: List[Dict[str, Any]] = []
    for scenario in scenarios:
        print(f"\n>>> Запуск сценария «{scenario.key}»...")
        results.append(run_scenario(scenario, options, config, counter, prices))

    for result in results:
        print_scenario_report(result, options)
    print_comparison(results)

    if options.model_limit:
        print(f"\nСправка: реальный лимит модели принят за {options.model_limit} токенов.")

    if options.json_path:
        payload = {
            "mode": options.mode,
            "model": options.model or config["DEFAULT_MODEL"],
            "counter": counter.name,
            "prices_set": bool(prices["default"]["input"] or prices["default"]["output"]),
            "model_limit": options.model_limit,
            # Какие сценарии стратегий участвовали в прогоне
            "strategies": selected_strategies(args),
            "scenarios": results,
        }
        path = Path(options.json_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print(f"\nОтчёт сохранён: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
