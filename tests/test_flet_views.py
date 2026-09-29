"""Тесты Flet-вкладок (app/ui/*) — офлайн, без запуска Flet.

Контролы Flet 1.0 — обычные объекты: вкладки собираются с заглушкой
app (без page), async-обработчики вызываются напрямую через asyncio.run.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.rag.embedder import EmbedderError
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from app.ui import components
from app.ui.chat_view import ChatView
from app.ui.comparison_view import (
    ComparisonView,
    load_results,
    load_test_questions,
    normalize_assessment,
    save_results,
)
from app.ui.knowledge_view import KnowledgeView, strategies_for_choice
from app.ui.settings_view import SettingsView

from tests.test_app_agent import (
    COMPARISON_CONFIG,
    MODEL,
    FakeEmbedder,
    FakeStore,
    make_agent,
)

QUESTION_FILE = Path(__file__).resolve().parents[1] / "app" / "data" / "test_questions.json"

RESULT_WITH_SOURCES = {
    "ids": [["chunk_1"]],
    "documents": [["Чанк про установку Windows: скачайте Python."]],
    "metadatas": [[{"source": "installation.md", "title": "Установка",
                    "section": "## Windows", "strategy": "structure",
                    "chunk_id": 1}]],
    "distances": [[0.12]],
}


def make_app(agent, page=None, file_picker=None):
    """Заглушка владельца вкладок (как ChatApp в app/main.py)."""
    return SimpleNamespace(
        agent=agent, page=page, file_picker=file_picker,
        notify=lambda message: None,
    )


class ComponentsTest(unittest.TestCase):
    """Общие компоненты."""

    def test_message_bubble_user_and_assistant(self) -> None:
        user = components.message_bubble("Вопрос", "user")
        assistant = components.message_bubble("Ответ", "assistant")
        self.assertEqual(len(user.content.controls), 2)
        self.assertIn("Вы", user.content.controls[0].value)
        self.assertIn("Ассистент", assistant.content.controls[0].value)

    def test_sources_panel_empty_gives_none(self) -> None:
        self.assertIsNone(components.sources_panel([]))

    def test_sources_panel_renders_rows(self) -> None:
        from app.rag.retriever import RetrievedChunk
        sources = Retriever.format_sources([
            RetrievedChunk(text="Текст чанка", source="installation.md",
                           section="## Windows", strategy="structure",
                           distance=0.1, chunk_id=1)])
        panel = components.sources_panel(sources)
        self.assertIn("Источники (1)", str(panel.title.value))
        self.assertEqual(len(panel.controls), 2)  # строка + фрагмент

    def test_truncate(self) -> None:
        self.assertEqual(components.truncate("короткий"), "короткий")
        long_text = "х" * 500
        self.assertEqual(len(components.truncate(long_text)), 220)

    def test_aggregate_assessments(self) -> None:
        values = ["Хорошо"] * 7 + ["Частично"] * 2 + ["Плохо"]
        stats = components.aggregate_assessments(values)
        self.assertEqual(stats["Хорошо"], 70.0)
        self.assertEqual(stats["Частично"], 20.0)
        self.assertEqual(stats["Плохо"], 10.0)
        self.assertEqual(stats["Оценено"], 10)

    def test_aggregate_partial_assessments(self) -> None:
        stats = components.aggregate_assessments(["Хорошо", None, "Хорошо"])
        self.assertEqual(stats["Хорошо"], 100.0)
        self.assertEqual(stats["Частично"], 0.0)
        self.assertEqual(stats["Оценено"], 2)

    def test_aggregate_empty(self) -> None:
        stats = components.aggregate_assessments([None, None])
        self.assertIsNone(stats["Хорошо"])
        self.assertEqual(stats["Оценено"], 0)

    def test_format_summary(self) -> None:
        stats = {"Хорошо": 70.0, "Частично": 20.0, "Плохо": 10.0,
                 "Оценено": 10}
        line = components.format_summary("Без RAG", stats)
        self.assertEqual(line,
                         "Без RAG: 70% хорошо, 20% частично, 10% плохо")

    def test_format_summary_empty(self) -> None:
        self.assertEqual(components.format_summary("С RAG", {"Оценено": 0}),
                         "С RAG: нет оценок")

    def test_assessment_options(self) -> None:
        self.assertEqual(components.ASSESSMENT_OPTIONS,
                         ("Хорошо", "Частично", "Плохо"))


class ChatViewTest(unittest.TestCase):
    """Вкладка «Чат»."""

    def setUp(self) -> None:
        self.agent = make_agent()
        self.agent.bind_config(dict(COMPARISON_CONFIG))
        self.chat = ChatView(make_app(self.agent))

    def test_build_returns_column(self) -> None:
        self.assertEqual(type(self.chat.build()).__name__, "Column")

    def test_restore_history(self) -> None:
        self.chat.restore_history([
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "Привет"},
            {"role": "assistant", "content": "Здравствуй!"},
        ])
        self.assertEqual(len(self.chat.messages_list.controls), 2)

    def test_send_without_rag(self) -> None:
        self.chat.input_field.value = "Вопрос без RAG"
        asyncio.run(self.chat._send_async())
        # пузырь пользователя + ответ
        self.assertEqual(len(self.chat.messages_list.controls), 2)
        self.assertEqual(self.chat.input_field.value, "")

    def test_send_with_rag_shows_sources_panel(self) -> None:
        self.agent.set_rag_enabled(True)
        self.agent.vector_store = FakeStore(RESULT_WITH_SOURCES)
        self.chat.agent = self.agent
        self.chat.input_field.value = "Как установить?"
        asyncio.run(self.chat._send_async())
        bubble = self.chat.messages_list.controls[-1]
        control_types = [type(c).__name__ for c in bubble.content.controls]
        self.assertIn("ExpansionTile", control_types)

    def test_send_with_rag_error_shows_warning(self) -> None:
        embedder = FakeEmbedder()
        embedder.fail = True
        # Retriever держит ссылку на эмбеддер — подменяем у него
        self.agent.retriever.embedder = embedder
        self.agent.set_rag_enabled(True)
        self.chat.input_field.value = "Вопрос"
        asyncio.run(self.chat._send_async())
        bubble = self.chat.messages_list.controls[-1]
        texts = " | ".join(
            getattr(c, "value", "") or "" for c in bubble.content.controls)
        self.assertIn("RAG", texts)

    def test_empty_input_ignored(self) -> None:
        self.chat.input_field.value = "   "
        asyncio.run(self.chat._send_async())
        self.assertEqual(len(self.chat.messages_list.controls), 0)

    def test_rag_toggle_updates_agent(self) -> None:
        self.chat.rag_switch.value = True
        self.chat._on_rag_toggle()
        self.assertTrue(self.agent.rag_enabled)
        self.chat.rag_switch.value = False
        self.chat._on_rag_toggle()
        self.assertFalse(self.agent.rag_enabled)

    def test_top_k_field(self) -> None:
        self.chat.top_k_field.value = "5"
        self.chat._on_top_k_change()
        self.assertEqual(self.agent.top_k, 5)
        # Некорректное значение откатывается
        self.chat.top_k_field.value = "abc"
        self.chat._on_top_k_change()
        self.assertEqual(self.chat.top_k_field.value, "5")

    def test_error_result_rendered(self) -> None:
        self.chat.show_thinking()
        self.chat.add_assistant_result({"error": "Нет сети"})
        bubble = self.chat.messages_list.controls[-1]
        texts = " | ".join(
            getattr(c, "value", "") or "" for c in bubble.content.controls)
        self.assertIn("Ошибка", texts)

    def test_tool_calls_panel_rendered(self) -> None:
        self.chat.show_thinking()
        self.chat.add_assistant_result({
            "text": "Ответ",
            "tool_calls": [{"name": "create_reminder",
                            "arguments": {"text": "позвонить"},
                            "result": "Напоминание создано"}],
        })
        bubble = self.chat.messages_list.controls[-1]
        control_types = [type(c).__name__ for c in bubble.content.controls]
        self.assertIn("ExpansionTile", control_types)

    def test_clear_history(self) -> None:
        self.chat.add_user_message("тест")
        asyncio.run(self.chat._clear_async())
        self.assertEqual(len(self.chat.messages_list.controls), 0)

    def test_rag_status_shows_chunks(self) -> None:
        self.chat.refresh_rag_status()
        self.assertIn("чанков", self.chat.rag_status_text.value)


class KnowledgeViewTest(unittest.TestCase):
    """Вкладка «База знаний»."""

    def setUp(self) -> None:
        self.agent = make_agent()
        self.tmp = Path(tempfile.mkdtemp(prefix="kb_test_"))
        self.agent.vector_store = VectorStore(persist_dir=self.tmp / "db")
        from app.rag.indexer import KnowledgeIndexer
        self.agent.embedder = FakeEmbedder()
        self.kv = KnowledgeView(make_app(self.agent))
        self.kv.indexer = KnowledgeIndexer(self.agent.embedder,
                                           self.agent.vector_store)
        self.docs = Path(tempfile.mkdtemp(prefix="docs_test_"))
        (self.docs / "a.md").write_text(
            "# Документ\n\nТекст.\n\n## Раздел\n\nСодержимое раздела.",
            encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)
        shutil.rmtree(self.docs, ignore_errors=True)

    def test_build_returns_column(self) -> None:
        self.assertEqual(type(self.kv.build()).__name__, "Column")

    def test_strategies_for_choice(self) -> None:
        self.assertEqual(strategies_for_choice("fixed"), ("fixed",))
        self.assertEqual(strategies_for_choice("structure"), ("structure",))
        self.assertEqual(strategies_for_choice("both"), ("fixed", "structure"))
        self.assertEqual(strategies_for_choice(None), ("fixed",))

    def test_indexing_updates_store_and_status(self) -> None:
        self.kv.folder_field.value = str(self.docs)
        self.kv.chunk_size_field.value = "100"
        self.kv.overlap_field.value = "10"
        self.kv.strategy_dropdown.value = "both"
        asyncio.run(self.kv._index_async())
        self.assertIn("Проиндексировано", self.kv.status_text.value)
        self.assertGreater(self.agent.vector_store.count(), 0)

    def test_indexing_bad_folder(self) -> None:
        self.kv.folder_field.value = str(self.docs / "missing")
        asyncio.run(self.kv._index_async())
        self.assertIn("Ошибка", self.kv.status_text.value)

    def test_indexing_invalid_overlap(self) -> None:
        self.kv.folder_field.value = str(self.docs)
        self.kv.chunk_size_field.value = "100"
        self.kv.overlap_field.value = "100"
        asyncio.run(self.kv._index_async())
        self.assertIn("overlap", self.kv.status_text.value)

    def test_indexing_empty_folder(self) -> None:
        empty = Path(tempfile.mkdtemp(prefix="empty_docs_"))
        try:
            self.kv.folder_field.value = str(empty)
            asyncio.run(self.kv._index_async())
            self.assertIn("нет MD-файлов", self.kv.status_text.value)
        finally:
            shutil.rmtree(empty, ignore_errors=True)

    def test_clear_button(self) -> None:
        self.kv.folder_field.value = str(self.docs)
        self.kv.chunk_size_field.value = "100"
        self.kv.overlap_field.value = "10"
        asyncio.run(self.kv._index_async())
        asyncio.run(self.kv._clear_async())
        self.assertEqual(self.agent.vector_store.count(), 0)

    def test_schedule_without_page_is_safe(self) -> None:
        # _schedule без page закрывает корутину — не падает
        self.kv._on_clear()
        self.chat_like_noop = True


class ComparisonLogicTest(unittest.TestCase):
    """Чистая логика вкладки «Сравнение»."""

    def test_load_test_questions_validates(self) -> None:
        questions = load_test_questions(QUESTION_FILE)
        self.assertEqual(len(questions), 10)
        for question in questions:
            self.assertIn("id", question)
            self.assertIn("question", question)
            self.assertIn("expected", question)
            self.assertTrue(question["sources"])

    def test_load_rejects_missing_field(self) -> None:
        import json as json_module
        bad = Path(tempfile.mkdtemp()) / "bad.json"
        bad.write_text(json_module.dumps([
            {"id": 1, "question": "в", "expected": "о"}  # нет sources
        ]), encoding="utf-8")
        try:
            with self.assertRaises(ValueError):
                load_test_questions(bad)
        finally:
            bad.unlink()

    def test_load_rejects_duplicate_ids(self) -> None:
        import json as json_module
        bad = Path(tempfile.mkdtemp()) / "dup.json"
        item = {"id": 1, "question": "в", "expected": "о", "sources": ["a.md"]}
        bad.write_text(json_module.dumps([item, dict(item)]), encoding="utf-8")
        try:
            with self.assertRaises(ValueError):
                load_test_questions(bad)
        finally:
            bad.unlink()

    def test_normalize_assessment(self) -> None:
        self.assertEqual(normalize_assessment("Хорошо"), "Хорошо")
        self.assertIsNone(normalize_assessment("чушь"))
        self.assertIsNone(normalize_assessment(None))

    def test_save_load_results_roundtrip(self) -> None:
        path = Path(tempfile.mkdtemp()) / "results.json"
        try:
            save_results({1: {"assessment_no_rag": "Хорошо",
                              "answer_rag": "ответ"}}, path)
            loaded = load_results(path)
            self.assertEqual(loaded[1]["assessment_no_rag"], "Хорошо")
        finally:
            shutil.rmtree(path.parent, ignore_errors=True)

    def test_load_results_missing_file(self) -> None:
        self.assertEqual(load_results(Path("no/such/file.json")), {})


class ComparisonViewTest(unittest.TestCase):
    """Вкладка «Сравнение» (10 вопросов из data)."""

    def setUp(self) -> None:
        self.agent = make_agent()
        self.agent.bind_config(dict(COMPARISON_CONFIG))
        self.results_path = Path(tempfile.mkdtemp()) / "results.json"
        self.view = ComparisonView(make_app(self.agent),
                                   questions_path=QUESTION_FILE,
                                   results_path=self.results_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.results_path.parent, ignore_errors=True)

    def test_build_returns_column(self) -> None:
        self.assertEqual(type(self.view.build()).__name__, "Column")

    def test_ten_question_rows(self) -> None:
        self.assertEqual(len(self.view.rows), 10)

    def test_run_all_fills_answers(self) -> None:
        asyncio.run(self.view._run_async())
        first = self.view.rows[0]
        self.assertNotEqual(first["answer_no_rag"].value, "—")
        self.assertNotEqual(first["answer_rag"].value, "—")
        # Результаты сохранены в JSON
        loaded = load_results(self.results_path)
        self.assertEqual(len(loaded), 10)

    def test_assessments_aggregated_in_summary(self) -> None:
        self.view.rows[0]["dropdown_no_rag"].value = "Хорошо"
        self.view.rows[0]["dropdown_rag"].value = "Плохо"
        self.view.refresh_summaries()
        self.assertIn("100% хорошо", self.view.summary_no_rag.value)
        self.assertIn("100% плохо", self.view.summary_rag.value)

    def test_assessment_change_persists(self) -> None:
        self.view.rows[2]["dropdown_rag"].value = "Частично"
        self.view._on_assessment_change()
        loaded = load_results(self.results_path)
        self.assertEqual(loaded[3]["assessment_rag"], "Частично")

    def test_answers_recorded_for_both_modes(self) -> None:
        asyncio.run(self.view._run_async())
        no_rag = self.view.rows[4]["answer_no_rag"].value
        rag = self.view.rows[4]["answer_rag"].value
        self.assertTrue(no_rag)
        self.assertTrue(rag)


class SettingsViewTest(unittest.TestCase):
    """Вкладка «Настройки»."""

    def setUp(self) -> None:
        self.agent = make_agent()
        self.view = SettingsView(make_app(self.agent))

    def test_build_returns_column(self) -> None:
        self.assertEqual(type(self.view.build()).__name__, "Column")

    def test_save_valid_values(self) -> None:
        self.view.temperature_field.value = "0.3"
        self.view.max_tokens_field.value = "2048"
        asyncio.run(self.view._save_async())
        self.assertEqual(self.agent.temperature, 0.3)
        self.assertEqual(self.agent.max_tokens, 2048)
        self.assertIn("Сохранено", self.view.status_text.value)

    def test_save_invalid_temperature(self) -> None:
        self.view.temperature_field.value = "5.0"
        asyncio.run(self.view._save_async())
        self.assertIn("Ошибка", self.view.status_text.value)

    def test_save_invalid_max_tokens(self) -> None:
        self.view.max_tokens_field.value = "abc"
        asyncio.run(self.view._save_async())
        self.assertIn("Ошибка", self.view.status_text.value)

    def test_model_choices_displayed(self) -> None:
        keys = [option.key for option in self.view.model_dropdown.options]
        self.assertIn("deepseek-v4-flash", keys)
        self.assertIn("deepseek-v4-pro", keys)


if __name__ == "__main__":
    unittest.main()
