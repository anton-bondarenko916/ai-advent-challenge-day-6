"""Приёмочная проверка ТЗ ai_1_23: источники, цитаты, режим «Не знаю».

Прогоняет 10 контрольных вопросов через все 5 RAG-режимов через
``RAGChatAgent.answer_with_citations`` (реальные Ollama + DeepSeek,
JSON mode) и проверяет автоматически:

* у каждого ответа есть sources (source + section/chunk_id);
* у каждого ответа есть quotes;
* все цитаты valid: True (дословная проверка validate_quotes);
* abstain при среднем score < порога (проба посторонним вопросом —
  LLM не должен вызываться).

Результаты сохраняются в app/data/rag_mode_comparison.json в формате
вкладки «Сравнение RAG-режимов» (вкладка восстановит их при старте).

Запуск из корня проекта:
    python scripts/rag_citations_check.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent import RAGChatAgent  # noqa: E402
from app.rag.citations import ABSTAIN_THRESHOLD  # noqa: E402
from app.ui.rag_comparison_view import (  # noqa: E402
    DEFAULT_RESULTS_PATH,
    DEFAULT_TOP_K_FINAL_TEXT,
    DEFAULT_TOP_K_TEXT,
    aggregate_metrics,
    citation_checks,
    extract_keywords,
    load_questions,
    mode_metrics,
    save_run,
)
from src.config import load_config  # noqa: E402

QUESTIONS_PATH = (Path(__file__).resolve().parents[1] / "app" / "data" /
                  "test_questions.json")
# Посторонний вопрос: в документах приложения борща нет — ждём abstain
OFF_TOPIC_QUESTION = "Как приготовить борщ по классическому рецепту?"
OFF_TOPIC_MODE = "basic"

RAG_MODES = ("basic", "rewrite", "rewrite_filter", "reranker",
             "rewrite_reranker")


async def main() -> int:
    config = load_config()
    agent = RAGChatAgent.with_config(config).bind_config(config)
    chunks = agent.count_chunks()
    print(f"Чанков в базе знаний: {chunks}")
    if not chunks:
        print("База знаний пуста — проиндексируйте её на вкладке "
              "«База знаний» и повторите прогон.")
        return 2

    questions = load_questions(QUESTIONS_PATH)
    if not questions:
        print(f"Нет контрольных вопросов: {QUESTIONS_PATH}")
        return 2
    params = {"top_k": int(DEFAULT_TOP_K_TEXT),
              "top_k_final": int(DEFAULT_TOP_K_FINAL_TEXT),
              "threshold": ABSTAIN_THRESHOLD}
    print(f"Порог «Не знаю»: {ABSTAIN_THRESHOLD} · "
          f"top_k={params['top_k']} · top_k_final={params['top_k_final']}")
    print(f"Вопросов: {len(questions)} × режимов: {len(RAG_MODES)}\n")

    runs = {mode: [] for mode in RAG_MODES}
    saved_results: dict = {}
    failures: list = []
    abstained_count = 0
    for number, question in enumerate(questions, start=1):
        text = str(question.get("question") or "")
        keywords = extract_keywords(text, str(question.get("expected") or ""))
        print(f"[{number}/{len(questions)}] {text}")
        for mode in RAG_MODES:
            result = await agent.answer_with_citations(
                text, rag_mode=mode, retrieval_params=params)
            metrics = mode_metrics(result, keywords)
            checks = citation_checks(result, keywords)
            runs[mode].append(metrics)
            saved_results[mode] = {
                "answer": str(result.get("text") or ""),
                "quotes": list(result.get("quotes") or []),
                "checks": checks,
                "metrics": metrics,
            }
            abstained = checks["abstained"]
            abstained_count += int(abstained)
            status = "🤷 не знаю" if abstained else "✅" if (
                checks["sources_ok"] and checks["quotes_ok"]
                and checks["quotes_valid"]) else "❌"
            quotes_info = (f"{checks['valid_count']}/"
                           f"{checks['quotes_count']} valid")
            print(f"    {mode:<18} score="
                  f"{metrics['avg_score'] if metrics['avg_score'] is not None else '—':>5} "
                  f"источн.={metrics['chunks']} цитаты={quotes_info} {status}")
            if not abstained and not (checks["sources_ok"]
                                      and checks["quotes_ok"]
                                      and checks["quotes_valid"]):
                failures.append(
                    f"{text[:40]}… [{mode}]: "
                    f"источн.={checks['sources_ok']} "
                    f"цитаты={checks['quotes_ok']} "
                    f"valid={checks['quotes_valid']}")
        print()

    print("Агрегат по режимам:")
    aggregate = {mode: aggregate_metrics(runs[mode]) for mode in RAG_MODES}
    for mode in RAG_MODES:
        stats = aggregate[mode]
        print(f"    {mode:<18} ист.={stats['sources_rate']}% "
              f"цит.={stats['quotes_rate']}% "
              f"valid={stats['valid_rate']}% "
              f"score={stats['avg_score']}")

    # Проба режима «Не знаю»: посторонний вопрос, LLM не вызывается
    print(f"\nПроба «Не знаю» ({OFF_TOPIC_MODE}): "
          f"«{OFF_TOPIC_QUESTION}»")
    off_topic = await agent.answer_with_citations(
        OFF_TOPIC_QUESTION, rag_mode=OFF_TOPIC_MODE, retrieval_params=params)
    off_metrics = mode_metrics(off_topic, extract_keywords(
        OFF_TOPIC_QUESTION))
    print(f"    score={off_metrics['avg_score']} "
          f"abstained={off_topic.get('abstained')} "
          f"причина={off_topic.get('abstain_reason')}")
    if not off_topic.get("abstained"):
        failures.append("Посторонний вопрос НЕ привёл к «Не знаю» "
                        f"(score={off_metrics['avg_score']})")

    save_run({
        "question": str(questions[0].get("question") or ""),
        "params": params,
        "results": saved_results,
        "aggregate": aggregate,
    }, DEFAULT_RESULTS_PATH)
    print(f"\nПрогон сохранён: {DEFAULT_RESULTS_PATH}")

    if failures:
        print(f"\nПРОВАЛЕНО: {len(failures)} проблем(ы):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"\nПРИЁМОЧНАЯ ПРОВЕРКА ПРОЙДЕНА: у всех ответов есть источники "
          f"и дословные цитаты; abstain-проба: "
          f"{'сработала' if off_topic.get('abstained') else 'НЕ сработала'}; "
          f"воздержаний на контрольных вопросах: {abstained_count}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
