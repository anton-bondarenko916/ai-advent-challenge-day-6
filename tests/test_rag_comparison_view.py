"""Тесты вкладки «Сравнение RAG-режимов» (app/ui/rag_comparison_view.py).

Чистые функции (ключевые слова, метрики, агрегация, сохранение) и
сборка вкладки офлайн — с фейковым агентом, без Flet-рантайма.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import flet as ft

from app.rag.pipeline import RAG_MODES
from app.ui.rag_comparison_view import (
    RAGModesView,
    aggregate_metrics,
    extract_keywords,
    format_aggregate_line,
    format_metrics_line,
    keyword_coverage,
    load_questions,
    load_run,
    mode_metrics,
    save_run,
)
from tests.test_flet_views import FakePage, make_app

QUESTION_FILE = (Path(__file__).resolve().parents[1] / "app" / "data" /
                 "test_questions.json")


class FakeModesAgent:
    """Агент-заглушка: answer_once отдаёт результат по режиму."""

    def __init__(self, rag_mode="basic") -> None:
        self.rag_mode = rag_mode
        self.top_k = 3
        self.top_k_search = 20
        self.similarity_threshold = 0.7
        self.calls: list = []

    def set_rag_mode(self, mode: str) -> str:
        self.rag_mode = mode
        return mode

    async def answer_once(self, question, rag_enabled, rag_mode=None,
                          retrieval_params=None):
        self.calls.append((question, rag_mode, retrieval_params))
        scores = {"basic": [0.82], "rewrite": [0.85],
                  "rewrite_filter": [0.9], "reranker": [0.91],
                  "rewrite_reranker": [0.93]}
        chosen = rag_mode or self.rag_mode
        mode_scores = scores.get(chosen, [0.5])
        return {
            "text": f"Ответ {chosen} про установку приложения Windows",
            "sources": [{"source": "installation.md",
                         "section": "## Windows",
                         "strategy": "structure",
                         "distance": round(1.0 - mode_scores[0], 4),
                         "preview": "Установка приложения на Windows"}],
            "rag_used": True,
            "retrieval": {
                "search_query": f"поиск: {question}",
                "rewrite_used": chosen.startswith("rewrite"),
                "rewrite_error": None,
                "rerank_backend": ("FakeReranker"
                                   if "reranker" in chosen else None),
                "rerank_error": None,
                "scores": mode_scores,
            },
        }


class KeywordsTest(unittest.TestCase):
    """Извлечение ключевых слов и покрытие."""

    def test_extract_from_question_and_expected(self) -> None:
        keywords = extract_keywords(
            "Как установить приложение?",
            "Скачать Python, отметить Add Python to PATH")
        self.assertIn("установить", keywords)
        self.assertIn("приложение", keywords)
        self.assertIn("python", keywords)

    def test_short_words_skipped(self) -> None:
        keywords = extract_keywords("Как и что где", "это то")
        self.assertEqual(keywords, [])

    def test_capped_at_eight(self) -> None:
        keywords = extract_keywords(
            "первый второй третий четвёртый пятый шестой седьмой "
            "восьмой девятый десятый", "")
        self.assertLessEqual(len(keywords), 8)
        self.assertEqual(len(keywords), 8)

    def test_no_duplicates(self) -> None:
        keywords = extract_keywords("установка установка Windows", "")
        self.assertEqual(keywords.count("установка"), 1)

    def test_coverage(self) -> None:
        found, total = keyword_coverage(
            ["python", "установка"], "Установка через pip и python.org")
        self.assertEqual((found, total), (2, 2))

    def test_coverage_empty(self) -> None:
        self.assertEqual(keyword_coverage([], "любой текст"), (0, 0))


class MetricsTest(unittest.TestCase):
    """Автоматические метрики режима."""

    RESULT = {
        "text": "Установка приложения: скачайте Python.",
        "sources": [{"source": "installation.md", "preview": "Шаги установки",
                     "distance": 0.1}],
        "retrieval": {"scores": [0.9, 0.8], "rewrite_used": True,
                      "rerank_backend": "OllamaReranker",
                      "search_query": "установка"},
    }

    def test_chunks_and_avg_score(self) -> None:
        metrics = mode_metrics(self.RESULT, ["python"])
        self.assertEqual(metrics["chunks"], 1)
        self.assertAlmostEqual(metrics["avg_score"], 0.85, places=4)

    def test_keyword_hit_counts_sources(self) -> None:
        # «установки» из preview источника попадает в покрытие
        metrics = mode_metrics(self.RESULT, ["установки"])
        self.assertEqual(metrics["kw_found"], 1)
        self.assertEqual(metrics["kw_total"], 1)

    def test_rewrite_and_rerank_flags(self) -> None:
        metrics = mode_metrics(self.RESULT, [])
        self.assertTrue(metrics["rewrite_used"])
        self.assertEqual(metrics["rerank_backend"], "OllamaReranker")

    def test_no_chunks_gives_none_score(self) -> None:
        metrics = mode_metrics({"text": "нет базы", "sources": [],
                                "retrieval": {"scores": []}}, ["слово"])
        self.assertEqual(metrics["chunks"], 0)
        self.assertIsNone(metrics["avg_score"])
        self.assertEqual((metrics["kw_found"], metrics["kw_total"]), (0, 1))

    def test_format_metrics_line(self) -> None:
        line = format_metrics_line({"chunks": 3, "avg_score": 0.8234,
                                    "kw_found": 4, "kw_total": 5})
        self.assertEqual(line, "Чанков: 3 · Ср. score: 0.82 · Ключ.: 4/5")

    def test_format_metrics_line_no_score(self) -> None:
        line = format_metrics_line({"chunks": 0, "avg_score": None,
                                    "kw_found": 0, "kw_total": 3})
        self.assertIn("Ср. score: —", line)


class AggregateTest(unittest.TestCase):
    """Средние метрики прогона."""

    def test_average_over_runs(self) -> None:
        runs = [
            {"chunks": 3, "avg_score": 0.8, "kw_found": 4, "kw_total": 5},
            {"chunks": 2, "avg_score": 0.9, "kw_found": 3, "kw_total": 5},
        ]
        stats = aggregate_metrics(runs)
        self.assertEqual(stats["questions"], 2)
        self.assertAlmostEqual(stats["chunks"], 2.5, places=6)
        self.assertAlmostEqual(stats["avg_score"], 0.85, places=6)
        self.assertAlmostEqual(stats["kw_rate"], 70.0, places=6)

    def test_empty_runs(self) -> None:
        stats = aggregate_metrics([])
        self.assertEqual(stats["questions"], 0)
        self.assertIsNone(stats["avg_score"])
        self.assertIsNone(stats["kw_rate"])

    def test_format_aggregate_line(self) -> None:
        line = format_aggregate_line("basic", {
            "questions": 10, "chunks": 3.0, "avg_score": 0.82, "kw_rate": 78.0})
        self.assertIn("Basic RAG: 10 вопр.", line)
        self.assertIn("3.0 чанк.", line)
        self.assertIn("0.82", line)
        self.assertIn("78%", line)

    def test_format_aggregate_line_empty(self) -> None:
        self.assertIn("нет данных",
                      format_aggregate_line("basic", {}))


class SaveLoadRunTest(unittest.TestCase):
    """Сохранение/загрузка прогона (JSON переживает перезапуск)."""

    def test_roundtrip(self) -> None:
        path = Path(tempfile.mkdtemp()) / "run.json"
        payload = {"question": "Вопрос?", "results": {"basic": {
            "answer": "Ответ", "metrics": {"chunks": 3}}}}
        save_run(payload, path)
        loaded = load_run(path)
        self.assertEqual(loaded["question"], "Вопрос?")
        self.assertEqual(loaded["results"]["basic"]["metrics"]["chunks"], 3)

    def test_load_missing_gives_empty(self) -> None:
        self.assertEqual(
            load_run(Path(tempfile.mkdtemp()) / "нет.json"), {})

    def test_load_broken_json_gives_empty(self) -> None:
        path = Path(tempfile.mkdtemp()) / "run.json"
        path.write_text("{битый", encoding="utf-8")
        self.assertEqual(load_run(path), {})


class LoadQuestionsTest(unittest.TestCase):
    """Чтение контрольных вопросов (необязательных для вкладки)."""

    def test_real_file_loads(self) -> None:
        questions = load_questions(QUESTION_FILE)
        self.assertGreater(len(questions), 0)
        self.assertIn("question", questions[0])

    def test_missing_file_gives_empty(self) -> None:
        self.assertEqual(
            load_questions(Path(tempfile.mkdtemp()) / "нет.json"), [])


class RAGModesViewTest(unittest.TestCase):
    """Вкладка: сборка, прогон, метрики в колонках (офлайн)."""

    def make_view(self, agent=None):
        agent = agent or FakeModesAgent()
        app = make_app(agent)
        view = RAGModesView(
            app, questions_path=QUESTION_FILE,
            results_path=Path(tempfile.mkdtemp()) / "run.json")
        return view, agent

    def test_layout_regression_wrap_expand(self) -> None:
        """Регрессия серой области (flet 1.0): expand-ребёнок внутри
        wrap-Row ломает лейаут (Wrap — не Flex) — вместо строки вопроса
        и карточек Flet рисует серую ErrorWidget-заглушку. Воспроизведено
        скриншотом окна; тест держит контракт: у строки вопроса нет wrap,
        у строки карточек expand=True (bounded-высота для ListView)."""
        view, _ = self.make_view()
        layout = view.build()
        question_row = layout.controls[2]
        self.assertFalse(question_row.wrap)
        cards_row = layout.controls[4]
        self.assertTrue(cards_row.expand)
        self.assertEqual(cards_row.scroll, ft.ScrollMode.AUTO)
        # Карточки фиксированного размера: без явной высоты карточка даёт
        # intrinsic (~550px) и режется строкой ограниченной высоты
        for mode in RAG_MODES:
            container = view.mode_columns[mode]["card"].content
            self.assertEqual(container.width, 300)
            self.assertIsNotNone(container.height)
            self.assertEqual(container.content.scroll, ft.ScrollMode.AUTO)

    def test_build_has_five_columns(self) -> None:
        view, _ = self.make_view()
        layout = view.build()
        texts = [str(control) for control in layout.controls]
        # Колонки режимов созданы для каждого из 5 режимов
        self.assertEqual(len(view.mode_columns), 5)
        self.assertEqual(set(view.mode_columns), set(RAG_MODES))
        self.assertTrue(texts)

    def test_run_modes_fills_columns_and_saves(self) -> None:
        path = Path(tempfile.mkdtemp()) / "run.json"
        agent = FakeModesAgent()
        app = make_app(agent)
        view = RAGModesView(app, questions_path=QUESTION_FILE,
                            results_path=path)
        question = {"id": 1, "question": "Как установить приложение?",
                    "expected": "Скачать Python", "sources": []}

        async def run() -> None:
            await view._run_modes(question)

        asyncio.run(run())
        # Каждый режим вызван со своими параметрами
        self.assertEqual({mode for _, mode, _ in agent.calls}, set(RAG_MODES))
        # Метрики в колонках: у basic средний score 0.82
        self.assertIn("0.82", view.mode_columns["basic"]["metrics"].value)
        self.assertIn("Чанков: 1",
                      view.mode_columns["basic"]["metrics"].value)
        # Источники заполнены
        self.assertIn("installation.md",
                      view.mode_columns["basic"]["sources"].value)
        # Прогон сохранён
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["question"], "Как установить приложение?")
        self.assertEqual(set(saved["results"]), set(RAG_MODES))

    def test_run_all_aggregates_and_schedules_function(self) -> None:
        page = FakePage()
        agent = FakeModesAgent()
        app = make_app(agent, page=page)
        view = RAGModesView(app, questions_path=QUESTION_FILE,
                            results_path=Path(tempfile.mkdtemp()) / "r.json")
        view._on_run_all()
        handler, _args = page.tasks[0]
        # В run_task уходит КОРУТИННАЯ ФУНКЦИЯ (не корутина)
        self.assertTrue(inspect.iscoroutinefunction(handler))
        self.assertFalse(inspect.iscoroutine(handler))

        async def run() -> None:
            await view._run_all_async()

        asyncio.run(run())
        self.assertEqual(len(agent.calls), 5 * len(view.questions))
        self.assertIsNotNone(view.aggregate.get("basic"))
        self.assertEqual(view.aggregate["basic"]["questions"],
                         len(view.questions))
        # Сводка заполнена
        self.assertIn("вопр.", view.summary_lines["basic"].value)

    def test_search_async_runs_all_modes(self) -> None:
        view, agent = self.make_view()
        view.question_field.value = "Ручной вопрос про Ollama?"

        async def run() -> None:
            await view._search_async()

        asyncio.run(run())
        self.assertEqual({mode for _, mode, _ in agent.calls}, set(RAG_MODES))
        self.assertEqual(view.current_question, "Ручной вопрос про Ollama?")

    def test_retrieval_params_from_fields(self) -> None:
        view, agent = self.make_view()
        view.top_k_field.value = "15"
        view.top_k_final_field.value = "2"
        view.threshold_field.value = "0,55"  # запятая как разделитель
        params = view.retrieval_params()
        self.assertEqual(params, {"top_k": 15, "top_k_final": 2,
                                  "threshold": 0.55})

    def test_retrieval_params_invalid_falls_back(self) -> None:
        view, agent = self.make_view()
        view.top_k_field.value = "abc"
        view.top_k_final_field.value = ""
        view.threshold_field.value = "мусор"
        params = view.retrieval_params()
        self.assertEqual(params["top_k"], agent.top_k_search)
        self.assertEqual(params["top_k_final"], agent.top_k)
        self.assertAlmostEqual(params["threshold"],
                               agent.similarity_threshold)

    def test_mode_dropdown_sets_agent_mode(self) -> None:
        view, agent = self.make_view()
        view.mode_dropdown.value = "reranker"
        view._on_mode_change()
        self.assertEqual(agent.rag_mode, "reranker")

    def test_saved_run_restored_on_init(self) -> None:
        path = Path(tempfile.mkdtemp()) / "run.json"
        save_run({
            "question": "Сохранённый вопрос?",
            "results": {"basic": {
                "answer": "Ответ basic",
                "metrics": {"chunks": 3, "avg_score": 0.82,
                            "kw_found": 4, "kw_total": 5}}},
            "aggregate": {"basic": {"questions": 10, "chunks": 3.0,
                                    "avg_score": 0.82, "kw_rate": 78.0}},
        }, path)
        view = RAGModesView(make_app(FakeModesAgent()),
                            questions_path=QUESTION_FILE, results_path=path)
        self.assertEqual(view.question_field.value, "Сохранённый вопрос?")
        self.assertIn("0.82", view.mode_columns["basic"]["metrics"].value)
        self.assertIn("Ответ basic", view.mode_columns["basic"]["answer"].value)
        self.assertIn("78%", view.summary_lines["basic"].value)


if __name__ == "__main__":
    unittest.main()
