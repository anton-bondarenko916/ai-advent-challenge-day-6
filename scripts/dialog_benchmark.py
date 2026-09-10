"""Бенчмарк диалогов с агентом: короткий / длинный / превышающий окно.

Скрипт ведёт три диалога с ``DeepSeekAgent`` и сравнивает расход токенов,
стоимость и «память» агента:

1. **short** — короткий диалог (3 хода): всё помещается в контекст;
2. **long** — длинный диалог (14 ходов): контекст растёт линейно, но в окно
   модели помещается;
3. **over_limit** — тот же длинный диалог, но окно **искусственно уменьшено**
   (по умолчанию 6 сообщений и 400 токенов): часть истории вытесняется, и агент
   «забывает» факт, сказанный в начале диалога.

В первом ходе каждого диалога сообщается факт («Меня зовут Алекс… любимый
цвет — синий»), в последнем ходе задаётся проверочный вопрос. Это позволяет
наглядно увидеть разницу между «помнит» и «не помнит».

Режимы работы:

* ``--mode sim`` (по умолчанию) — офлайн-имитация модели: детерминированные
  ответы, токены считаются локальным счётчиком, сеть не используется;
* ``--mode live`` — реальные запросы к DeepSeek API (нужен ключ и ``--yes``);
* ``--mode auto`` — live, если в конфигурации есть ключ, иначе sim.

Примеры запуска (из корня проекта)::

    python scripts/dialog_benchmark.py                  # офлайн-сравнение
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
from src.storage import ContextStore  # noqa: E402
from src.tokens import create_counter  # noqa: E402

# --- Данные «памяти» диалога -------------------------------------------------
FACT_NAME = "Алекс"
FACT_CITY = "Казани"
FACT_COLOR = "синий"
PROBE_MARKER = "ПРОВЕРКА ПАМЯТИ"

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
        key: короткий идентификатор (short/long/over_limit).
        title: человекочитаемое название.
        turns: число ходов (запросов пользователя).
        words_per_turn: примерная длина «длинных» сообщений в словах.
        window_messages: искусственное окно в сообщениях (None — из конфигурации).
        window_tokens: искусственное окно в токенах (None — не ограничивать).
    """

    key: str
    title: str
    turns: int
    words_per_turn: int = 20
    window_messages: Optional[int] = None
    window_tokens: Optional[int] = None
    reply_words: int = 30


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
        """Возвращает ответ в формате ``_handle_response``."""
        messages = self.agent.history  # копия, включая system
        last_user = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                last_user = message.get("content", "")
                break

        if PROBE_MARKER.lower() in last_user.lower():
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

def _make_agent(scenario: Scenario, options: Options, config: Dict[str, Any],
                counter: Any, prices: Dict[str, Any]) -> DeepSeekAgent:
    """Создаёт агента под конкретный сценарий (окно, модель, хранилище)."""
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
    )


def run_scenario(scenario: Scenario, options: Options, config: Dict[str, Any],
                 counter: Any, prices: Dict[str, Any]) -> Dict[str, Any]:
    """Прогоняет один сценарий диалога и собирает метрики.

    Returns:
        Словарь с полями: ключ/название сценария, список ходов (вопрос,
        ответ, токены, время), сводка (контекст, сессия, стоимость,
        вытесненные сообщения, «помнит ли факт»).
    """
    agent = _make_agent(scenario, options, config, counter, prices)
    simulated: Optional[SimulatedModel] = None
    if options.mode == "sim":
        simulated = SimulatedModel(agent, counter, reply_words=scenario.reply_words)
        agent._call_api = simulated  # подмена сетевого вызова

    turns: List[Dict[str, Any]] = []
    errors: List[str] = []
    trimmed_total = 0
    window_hits = 0
    max_context = 0

    dialog = build_dialog(scenario)
    for number, question in enumerate(dialog, start=1):
        result = agent.send_message(question)
        if result.get("error"):
            errors.append(f"ход {number}: {result['error']}")
            break

        # Искусственное ограничение окна в токенах (после обмена)
        if scenario.window_tokens:
            trimmed = enforce_token_window(agent, scenario.window_tokens)
            if trimmed:
                trimmed_total += trimmed
                window_hits += 1

        stats = agent.get_token_stats()
        tokens = result.get("tokens") or {}
        max_context = max(max_context, int(stats["context"]["tokens"]))

        turns.append({
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
        })

        if options.pause:
            time.sleep(options.pause)

    dialog_messages = max(0, len(agent.history) - 1)
    # Всего добавлено сообщений: вопрос + ответ за каждый успешный ход.
    # Разница с текущей историей — вытесненные сообщения (окно агента
    # и/или искусственное ограничение в токенах).
    dropped = max(0, 2 * len(turns) - dialog_messages)
    probe_answer = turns[-1]["answer"] if turns else ""
    remembered = check_remembered(probe_answer) if turns else None
    stats = agent.get_token_stats()

    summary = {
        "turns_done": len(turns),
        "turns_planned": len(dialog),
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
        "window_tokens_hits": window_hits,
        "token_window_trimmed": trimmed_total,
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
    """Стоимость сессии или «—», если цены не заданы."""
    if not summary.get("prices_set"):
        return "—"
    cost = float(summary.get("session_cost_usd") or 0.0) + float(
        summary.get("session_cost_estimated_usd") or 0.0
    )
    prefix = "≈" if summary.get("estimated_any") else ""
    return f"{prefix}${cost:.6f}"


def print_scenario_report(result: Dict[str, Any], options: Options) -> None:
    """Печатает отчёт по одному сценарию."""
    summary = result["summary"]
    print(f"\n=== {result['key']} — {result['title']} ===")
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
    print(f"  Сессия: {'≈' if summary['estimated_any'] else ''}"
          f"{summary['session_total_tokens']} токенов | стоимость: "
          f"{_fmt_cost(summary)}")
    if summary["remembered"] is None:
        print("  Помнит факт из начала диалога: —")
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
    """Печатает сравнительную таблицу по всем сценариям."""
    print("\n=== Сравнение ===")
    header = (f"{'Сценарий':<12}{'Ходов':>6}{'Контекст(макс)':>16}"
              f"{'Сессия':>10}{'Стоимость':>14}{'Помнит':>8}{'Вытеснено':>11}")
    print(header)
    print("-" * len(header))
    for result in results:
        summary = result["summary"]
        remembered = summary["remembered"]
        remembered_text = "—" if remembered is None else ("да" if remembered else "нет")
        print(
            f"{result['key']:<12}"
            f"{summary['turns_done']:>6}"
            f"{('≈' + str(summary['max_context_tokens'])):>16}"
            f"{('≈' if summary['estimated_any'] else '') + str(summary['session_total_tokens']):>10}"
            f"{_fmt_cost(summary):>14}"
            f"{remembered_text:>8}"
            f"{summary['messages_dropped']:>11}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_scenarios(args: argparse.Namespace) -> List[Scenario]:
    """Собирает три сценария с учётом параметров командной строки."""
    long_turns = args.turns
    return [
        Scenario(key="short", title="короткий диалог",
                 turns=3, words_per_turn=10, reply_words=20),
        Scenario(key="long", title="длинный диалог (в пределах окна)",
                 turns=long_turns, words_per_turn=args.words, reply_words=40),
        Scenario(key="over_limit", title="диалог, превышающий окно",
                 turns=long_turns, words_per_turn=args.words,
                 window_messages=args.window_messages,
                 window_tokens=args.window_tokens, reply_words=40),
    ]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Бенчмарк диалогов: короткий / длинный / превышающий окно.",
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
        print(f"\n[{scenario.key}] {scenario.title}: ходов {len(dialog)}, "
              f"слов/ход ≈{scenario.words_per_turn}"
              + (f", окно: {', '.join(window)}" if window else ", окно: из конфигурации"))
        for number, text in enumerate(dialog, start=1):
            preview = " ".join(text.split())
            if len(preview) > 90:
                preview = preview[:87] + "..."
            print(f"   {number:>2}. {preview}")


def main(argv: Optional[List[str]] = None) -> int:
    """Точка входа скрипта."""
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
