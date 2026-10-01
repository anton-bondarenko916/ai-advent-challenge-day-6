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
    citation_checks,
    extract_keywords,
    format_aggregate_line,
    format_checks_line,
    format_metrics_line,
    format_quotes_text,
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
    """Агент-заглушка: answer_with_citations отдаёт результат по режиму."""

    def __init__(self, rag_mode="basic") -> None:
        self.rag_mode = rag_mode
        self.top_k = 3
        self.top_k_search = 20
        self.similarity_threshold = 0.7
        self.calls: list = []

    def set_rag_mode(self, mode: str) -> str:
        self.rag_mode = mode
        return mode

    async def answer_with_citations(self, question, rag_mode=None,
                                    retrieval_params=None):
        self.calls.append((question, rag_mode, retrieval_params))
        scores = {"basic": [0.82], "rewrite": [0.85],
                  "rewrite_filter": [0.9], "reranker": [0.91],
                  "rewrite_reranker": [0.93]}
        chosen = rag_mode or self.rag_mode
        mode_scores = scores.get(chosen, [0.5])
        return {
            "answer": f"Ответ {chosen} про установку приложения Windows",
            "text": f"Ответ {chosen} про установку приложения Windows",
            "sources": [{"source": "installation.md",
                         "section": "## Windows",
                         "strategy": "structure",
                         "distance": round(1.0 - mode_scores[0], 4),
                         "chunk_id": 12,
                         "cited": True,
                         "preview": "Установка приложения на Windows"}],
            "citations": [{"chunk_id": 12, "source": "installation.md",
                           "section": "## Windows"}],
            "quotes": [{"chunk_id": 12,
                        "text": "Установка приложения на Windows",
                        "valid": True}],
            "abstained": False,
            "rag_used": True,
            "retrieval": {
                "search_query": f"поиск: {question}",
                "rewrite_used": chosen.startswith("rewrite"),
                "rewrite_error": None,
                "rerank_backend": ("FakeReranker"
                                   if "reranker" in chosen else None),
                "rerank_error": None,
                "scores": mode_scores,
                "avg_score": mode_scores[0],
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

    def test_quotes_counted(self) -> None:
        result = dict(self.RESULT, quotes=[
            {"chunk_id": 1, "text": "а", "valid": True},
            {"chunk_id": 2, "text": "б", "valid": False},
        ])
        metrics = mode_metrics(result, [])
        self.assertEqual(metrics["quotes"], 2)
        self.assertEqual(metrics["quotes_valid"], 1)
        self.assertFalse(metrics["abstained"])

    def test_abstained_flag(self) -> None:
        metrics = mode_metrics({"text": "Я не знаю.", "sources": [],
                                "abstained": True,
                                "retrieval": {"scores": [0.4]}}, [])
        self.assertTrue(metrics["abstained"])

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
            {"chunks": 3, "avg_score": 0.8, "kw_found": 4, "kw_total": 5,
             "quotes": 2, "quotes_valid": 2},
            {"chunks": 2, "avg_score": 0.9, "kw_found": 3, "kw_total": 5,
             "quotes": 1, "quotes_valid": 0},
        ]
        stats = aggregate_metrics(runs)
        self.assertEqual(stats["questions"], 2)
        self.assertAlmostEqual(stats["chunks"], 2.5, places=6)
        self.assertAlmostEqual(stats["avg_score"], 0.85, places=6)
        self.assertAlmostEqual(stats["kw_rate"], 70.0, places=6)
        # Источники/цитаты: доли по вопросам (ТЗ ai_1_23)
        self.assertAlmostEqual(stats["sources_rate"], 100.0, places=6)
        self.assertAlmostEqual(stats["quotes_rate"], 100.0, places=6)
        self.assertAlmostEqual(stats["valid_rate"], 50.0, places=6)

    def test_quotes_rates(self) -> None:
        runs = [
            {"chunks": 3, "avg_score": 0.8, "kw_found": 1, "kw_total": 2,
             "quotes": 2, "quotes_valid": 2},
            {"chunks": 2, "avg_score": 0.9, "kw_found": 1, "kw_total": 2,
             "quotes": 1, "quotes_valid": 0},
            {"chunks": 0, "avg_score": None, "kw_found": 0, "kw_total": 2,
             "quotes": 0, "quotes_valid": 0},
        ]
        stats = aggregate_metrics(runs)
        self.assertAlmostEqual(stats["sources_rate"], 66.7, places=6)
        self.assertAlmostEqual(stats["quotes_rate"], 66.7, places=6)
        # Валидность — доля среди вопросов с цитатами
        self.assertAlmostEqual(stats["valid_rate"], 50.0, places=6)

    def test_no_quotes_anywhere_gives_zero_rate(self) -> None:
        runs = [{"chunks": 2, "avg_score": 0.8, "kw_found": 1, "kw_total": 2,
                 "quotes": 0, "quotes_valid": 0}]
        stats = aggregate_metrics(runs)
        self.assertAlmostEqual(stats["quotes_rate"], 0.0, places=6)
        self.assertIsNone(stats["valid_rate"])

    def test_empty_runs(self) -> None:
        stats = aggregate_metrics([])
        self.assertEqual(stats["questions"], 0)
        self.assertIsNone(stats["avg_score"])
        self.assertIsNone(stats["kw_rate"])
        self.assertIsNone(stats["sources_rate"])
        self.assertIsNone(stats["quotes_rate"])

    def test_format_aggregate_line(self) -> None:
        line = format_aggregate_line("basic", {
            "questions": 10, "chunks": 3.0, "avg_score": 0.82, "kw_rate": 78.0})
        self.assertIn("Basic RAG: 10 вопр.", line)
        self.assertIn("3.0 чанк.", line)
        self.assertIn("0.82", line)
        self.assertIn("78%", line)

    def test_format_aggregate_line_with_citation_rates(self) -> None:
        line = format_aggregate_line("basic", {
            "questions": 10, "chunks": 3.0, "avg_score": 0.82,
            "kw_rate": 78.0, "sources_rate": 100.0, "quotes_rate": 90.0})
        self.assertIn("ист. 100%", line)
        self.assertIn("цит. 90%", line)

    def test_format_aggregate_line_empty(self) -> None:
        self.assertIn("нет данных",
                      format_aggregate_line("basic", {}))


class CitationChecksTest(unittest.TestCase):
    """Автопроверка ответа: источники, цитаты, валидность, смысл."""

    RESULT = {
        "text": "Установка приложения: скачайте Python.",
        "sources": [{"source": "installation.md", "chunk_id": 12}],
        "quotes": [
            {"chunk_id": 12, "text": "скачайте Python", "valid": True},
        ],
        "abstained": False,
    }

    def test_all_ok(self) -> None:
        checks = citation_checks(self.RESULT, ["установка", "python"])
        self.assertTrue(checks["sources_ok"])
        self.assertTrue(checks["quotes_ok"])
        self.assertTrue(checks["quotes_valid"])
        self.assertTrue(checks["meaning_ok"])
        self.assertEqual(checks["valid_count"], 1)
        self.assertFalse(checks["abstained"])

    def test_missing_sources_fails(self) -> None:
        checks = citation_checks(dict(self.RESULT, sources=[]), [])
        self.assertFalse(checks["sources_ok"])

    def test_missing_quotes_fails(self) -> None:
        checks = citation_checks(dict(self.RESULT, quotes=[]), [])
        self.assertFalse(checks["quotes_ok"])
        self.assertFalse(checks["quotes_valid"])

    def test_invalid_quote_fails_validation(self) -> None:
        result = dict(self.RESULT, quotes=[
            {"chunk_id": 12, "text": "цитата", "valid": False}])
        checks = citation_checks(result, [])
        self.assertFalse(checks["quotes_valid"])
        self.assertEqual(checks["valid_count"], 0)

    def test_meaning_fails_without_keywords(self) -> None:
        checks = citation_checks(self.RESULT, ["несуществующее"])
        self.assertFalse(checks["meaning_ok"])

    def test_abstained_is_not_failure(self) -> None:
        result = {"text": "Я не знаю.", "sources": [], "quotes": [],
                  "abstained": True}
        checks = citation_checks(result, ["установка"])
        self.assertTrue(checks["abstained"])
        self.assertTrue(checks["meaning_ok"])
        self.assertIn("Не знаю", format_checks_line(checks))

    def test_format_checks_line_ok(self) -> None:
        checks = citation_checks(self.RESULT, ["python"])
        line = format_checks_line(checks)
        self.assertIn("Источн.: ✅", line)
        self.assertIn("Цитаты: ✅ 1/1 valid", line)
        self.assertIn("Смысл: ✅", line)

    def test_format_checks_line_failures(self) -> None:
        checks = {"abstained": False, "sources_ok": False, "quotes_ok": True,
                  "quotes_valid": False, "quotes_count": 2, "valid_count": 1,
                  "kw_found": 0, "kw_total": 3, "meaning_ok": False}
        line = format_checks_line(checks)
        self.assertIn("Источн.: ❌", line)
        self.assertIn("Цитаты: ❌ 1/2 valid", line)
        self.assertIn("Смысл: ❌", line)

    def test_format_checks_line_empty(self) -> None:
        self.assertEqual(format_checks_line({}), "—")


class FormatQuotesTextTest(unittest.TestCase):
    """Цитаты в колонке: ✅ валидная, ⚠️ не найдена, обрезка длинных."""

    def test_valid_and_invalid_marks(self) -> None:
        text = format_quotes_text([
            {"chunk_id": 1, "text": "Первая цитата", "valid": True},
            {"chunk_id": 2, "text": "Вторая цитата", "valid": False},
        ])
        self.assertIn("«Первая цитата» ✅", text)
        self.assertIn("«Вторая цитата» ⚠️ не найдена", text)

    def test_empty_gives_dash(self) -> None:
        self.assertEqual(format_quotes_text([]), "—")
        self.assertEqual(format_quotes_text(None), "—")

    def test_long_quote_truncated(self) -> None:
        text = format_quotes_text([
            {"chunk_id": 1, "text": "д" * 300, "valid": True}])
        self.assertIn("…", text)
        self.assertLess(len(text), 140)


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
        # Источники заполнены: файл, секция, chunk_id, флаг cited
        sources_text = view.mode_columns["basic"]["sources"].value
        self.assertIn("installation.md", sources_text)
        self.assertIn("## Windows", sources_text)
        self.assertIn("chunk:12", sources_text)
        self.assertIn("✓", sources_text)
        # Цитаты с отметкой валидности
        self.assertIn("«Установка приложения на Windows» ✅",
                      view.mode_columns["basic"]["quotes"].value)
        # Блок «Проверка»: источники, цитаты, смысл
        checks_text = view.mode_columns["basic"]["checks"].value
        self.assertIn("Источн.: ✅", checks_text)
        self.assertIn("Цитаты: ✅ 1/1 valid", checks_text)
        # Прогон сохранён вместе с цитатами и проверками
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["question"], "Как установить приложение?")
        self.assertEqual(set(saved["results"]), set(RAG_MODES))
        saved_basic = saved["results"]["basic"]
        self.assertEqual(saved_basic["quotes"][0]["text"],
                         "Установка приложения на Windows")
        self.assertTrue(saved_basic["checks"]["sources_ok"])
        self.assertTrue(saved_basic["checks"]["quotes_valid"])

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

    def test_threshold_defaults_to_06(self) -> None:
        # ТЗ ai_1_23: порог «Не знаю» 0.6 (nomic-embed-text), не 0.7
        view, _ = self.make_view()
        self.assertEqual(view.threshold_field.value, "0.6")

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
                "quotes": [{"chunk_id": 12, "text": "Цитата из чанка",
                            "valid": True}],
                "checks": {"abstained": False, "sources_ok": True,
                           "quotes_ok": True, "quotes_valid": True,
                           "quotes_count": 1, "valid_count": 1,
                           "kw_found": 2, "kw_total": 3, "meaning_ok": True},
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
        self.assertIn("«Цитата из чанка» ✅",
                      view.mode_columns["basic"]["quotes"].value)
        self.assertIn("Цитаты: ✅ 1/1 valid",
                      view.mode_columns["basic"]["checks"].value)
        self.assertIn("78%", view.summary_lines["basic"].value)

    def test_saved_run_without_quotes_restores_dash(self) -> None:
        # Старый формат прогона (до цитирования) не ломает вкладку
        path = Path(tempfile.mkdtemp()) / "run.json"
        save_run({
            "question": "Старый прогон?",
            "results": {"basic": {
                "answer": "Старый ответ",
                "metrics": {"chunks": 2, "avg_score": 0.7,
                            "kw_found": 1, "kw_total": 2}}},
        }, path)
        view = RAGModesView(make_app(FakeModesAgent()),
                            questions_path=QUESTION_FILE, results_path=path)
        self.assertEqual(view.mode_columns["basic"]["quotes"].value, "—")
        self.assertEqual(view.mode_columns["basic"]["checks"].value, "—")

    def test_abstained_mode_shown_in_column(self) -> None:
        # Режим «Не знаю»: ℹ️ в проверке, 🤷 в диагностике
        path = Path(tempfile.mkdtemp()) / "run.json"

        class AbstainAgent(FakeModesAgent):
            async def answer_with_citations(self, question, rag_mode=None,
                                            retrieval_params=None):
                await super().answer_with_citations(
                    question, rag_mode, retrieval_params)
                return {
                    "answer": "Я не знаю.", "text": "Я не знаю.",
                    "sources": [], "citations": [], "quotes": [],
                    "abstained": True,
                    "abstain_reason": "средний score 0.52 ниже порога 0.60",
                    "rag_used": True,
                    "retrieval": {"search_query": question,
                                  "rewrite_used": False,
                                  "rewrite_error": None,
                                  "rerank_backend": None,
                                  "rerank_error": None,
                                  "scores": [0.52], "avg_score": 0.52},
                }

        app = make_app(AbstainAgent())
        view = RAGModesView(app, questions_path=QUESTION_FILE,
                            results_path=path)
        question = {"id": 1, "question": "Вопрос ни о чём?", "expected": "",
                    "sources": []}

        async def run() -> None:
            await view._run_modes(question)

        asyncio.run(run())
        column = view.mode_columns["basic"]
        self.assertIn("🤷 «Не знаю»", column["diag"].value)
        self.assertEqual(column["quotes"].value, "—")
        self.assertEqual(column["sources"].value, "—")
        self.assertIn("режим «Не знаю»", column["checks"].value)


if __name__ == "__main__":
    unittest.main()
