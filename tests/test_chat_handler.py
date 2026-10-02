"""Тесты обработчика хода чата (ТЗ ai_1_24, модули 3-4).

Проверяются: схема хода (память задачи → RAG → гейт → ответ → обновление),
порог 0.6 и стратегия C («без источников»), блоки промпта (память задачи
в копии запроса, история чистая), JSON-экстрактор (response_format,
бюджет-ретрай, нормализация), mock-эвристики памяти, устойчивость хода
при сбое экстрактора и сброс сессии.

Запуск из корня проекта:
    python -m unittest tests.test_chat_handler -v
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.chat_handler import (
    NO_RELEVANT_SOURCES,
    STRATEGY_C_NOTE,
    ChatHandler,
    ChatTurnExtractor,
    TaskStateError,
    build_no_context_block,
    build_rag_turn_block,
    mock_task_extraction,
)
from app.rag.citations import ABSTAIN_THRESHOLD
from app.rag.retriever import RetrievedChunk
from app.task_state import TaskState, TaskStateRepository
from tests.test_app_agent import FakeStore, make_agent

CHUNKS = [
    RetrievedChunk(text="Скачайте установщик с официального сайта.",
                   source="installation.md", section="## Windows",
                   strategy="structure", distance=0.12, chunk_id=12),
    RetrievedChunk(text="Установите Ollama и выполните pull модели.",
                   source="setup.md", section="## Ollama",
                   strategy="fixed", distance=0.34, chunk_id=45),
]

METADATAS = [{"chunk_id": 12, "source": "installation.md",
              "section": "## Windows", "strategy": "structure"},
             {"chunk_id": 45, "source": "setup.md",
              "section": "## Ollama", "strategy": "fixed"}]


def make_handler(store: FakeStore = None,
                 extractor: ChatTurnExtractor = None) -> tuple:
    """ChatHandler над mock-ядром + временная БД памяти задачи."""
    agent = make_agent(rag_enabled=True)
    if store is not None:
        agent.pipeline.vector_store = store
    repo = TaskStateRepository(Path(tempfile.mkdtemp()) / "chat.db")
    if extractor is None:
        extractor = ChatTurnExtractor(api_key="test", mock_mode=True)
    return ChatHandler(agent, task_state_repo=repo,
                       extractor=extractor), agent, repo


def run(coroutine):
    return asyncio.run(coroutine)


def make_extractor(contents=None, fail=False):
    """Экстрактор с фейковым клиентом (очередь ответов, как в ТЗ-тестах)."""
    captured = []

    def create(**kwargs):
        captured.append(kwargs)
        if fail:
            raise RuntimeError("соединение оборвано")
        content, finish = contents[min(len(captured) - 1,
                                       len(contents) - 1)]
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content),
            finish_reason=finish)])

    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)))
    extractor = ChatTurnExtractor(api_key="test", client=client)
    extractor.captured = captured
    return extractor


class PromptBlocksTest(unittest.TestCase):
    """Блоки промпта: память задачи + RAG-контекст (или пометка)."""

    def test_rag_block_contains_memory_context_question(self) -> None:
        state = TaskState(session_id="s", profile_id=1, goal="Цель",
                          constraints=["без Docker"])
        block = build_rag_turn_block(state, "Какой чанк выбрать?", CHUNKS)
        self.assertIn("[ПАМЯТЬ ЗАДАЧИ]", block)
        self.assertIn("🎯 Цель диалога: Цель", block)
        self.assertIn("⚠️ Ограничения:", block)
        self.assertIn("[RAG-КОНТЕКСТ]", block)
        self.assertIn("Контекст:", block)
        self.assertIn("Скачайте установщик", block)
        self.assertIn("Вопрос: Какой чанк выбрать?", block)
        self.assertIn("[ИНСТРУКЦИЯ]", block)

    def test_no_context_block_strategy_c(self) -> None:
        block = build_no_context_block(TaskState(session_id="s",
                                                 profile_id=1))
        self.assertIn(NO_RELEVANT_SOURCES, block)
        self.assertIn("из общих знаний", block)

    def test_no_context_block_with_rag_error(self) -> None:
        block = build_no_context_block(TaskState(session_id="s",
                                                 profile_id=1),
                                       rag_error="Ollama недоступна")
        self.assertIn("ПОИСК НЕДОСТУПЕН", block)
        self.assertIn("Ollama недоступна", block)


class GateAndStrategyCTest(unittest.TestCase):
    """Гейт порога 0.6 и стратегия C («Не знаю» + ответ без RAG)."""

    def test_high_scores_rag_turn(self) -> None:
        handler, agent, repo = make_handler()
        result = run(handler.handle_message("s", 0, "Как установить?"))
        self.assertTrue(result["has_rag"])
        self.assertTrue(result["rag_used"])
        self.assertEqual(len(result["sources"]), 2)
        self.assertTrue(result["quotes"])
        self.assertTrue(all(q["valid"] for q in result["quotes"]))
        # Память задачи в копии запроса, история хранит вопрос как есть
        last_user = [m for m in agent.core._message_history
                     if m.get("role") == "user"][-1]
        self.assertNotIn("[ПАМЯТЬ ЗАДАЧИ]", last_user["content"])

    def test_low_scores_strategy_c(self) -> None:
        low = FakeStore({"ids": [["c1"]], "documents": [["Чанк."]],
                         "metadatas": [[{"chunk_id": 9, "source": "a.md",
                                         "section": "## S",
                                         "strategy": "fixed"}]],
                         "distances": [[0.5]]})
        handler, agent, repo = make_handler(store=low)
        result = run(handler.handle_message("s", 0, "Вопрос мимо базы?"))
        self.assertIsNone(result.get("error"))
        self.assertFalse(result["has_rag"])
        self.assertFalse(result["rag_used"])
        # Стратегия C: источников и цитат нет, ответ из общих знаний
        self.assertEqual(result.get("sources"), [])
        self.assertEqual(result.get("quotes"), [])
        self.assertTrue(result["answer"])
        # Память задачи всё равно обновляется (после каждого сообщения)
        self.assertTrue(repo.get_or_create("s", 0).clarifications
                        or result["task_state"] is not None)

    def test_threshold_boundary_included(self) -> None:
        # Средний score ровно 0.6 НЕ воздерживается (граница включена)
        boundary = FakeStore({"ids": [["c1"]], "documents": [["Чанк."]],
                              "metadatas": [[{"chunk_id": 9,
                                              "source": "a.md",
                                              "section": "## S",
                                              "strategy": "fixed"}]],
                              "distances": [[0.4]]})
        handler, _agent, _repo = make_handler(store=boundary)
        result = run(handler.handle_message("s", 0, "Вопрос"))
        self.assertTrue(result["has_rag"])

    def test_threshold_is_06_calibrated(self) -> None:
        handler, _agent, _repo = make_handler()
        self.assertEqual(handler.abstain_threshold, ABSTAIN_THRESHOLD)
        self.assertEqual(handler.abstain_threshold, 0.6)

    def test_embedder_error_fallback(self) -> None:
        from app.rag.embedder import EmbedderError

        class FailingEmbedder:
            async def embed_query_async(self, text: str) -> list:
                raise EmbedderError("Ollama недоступна")

        handler, agent, _repo = make_handler()
        # Поиск идёт через RAGPipeline — подменяем эмбеддер у него
        agent.pipeline.embedder = FailingEmbedder()
        result = run(handler.handle_message("s", 0, "Вопрос"))
        self.assertIsNone(result["error"])
        self.assertFalse(result["has_rag"])
        self.assertIn("Ollama недоступна", result.get("rag_error", ""))

    def test_rag_disabled_plain_turn(self) -> None:
        handler, agent, _repo = make_handler()
        agent.rag_enabled = False
        result = run(handler.handle_message("s", 0, "Вопрос"))
        self.assertNotIn("has_rag", result)
        self.assertNotIn("sources", result)
        # Память задачи при выключенном RAG не теряется
        self.assertIn("task_state", result)


class MemoryUpdateTest(unittest.TestCase):
    """Обновление памяти задачи после каждого сообщения (JSON Mode)."""

    def test_state_accumulates_across_turns(self) -> None:
        handler, _agent, repo = make_handler()
        run(handler.handle_message("s", 0,
                                   "Хочу сделать RAG для своей документации"))
        run(handler.handle_message(
            "s", 0, "Документы на русском, подойдёт ли nomic-embed-text?"))
        result = run(handler.handle_message("s", 0, "Использую Ollama"))
        state = result["task_state"]
        self.assertIn("используется Ollama", state["clarifications"])
        self.assertIn("документы на русском", state["clarifications"])
        self.assertIn("RAG", state["terms"])
        # В SQLite сохранено то же состояние (персистентность)
        persisted = TaskStateRepository(repo.db_path).get_or_create("s", 0)
        self.assertEqual(persisted.clarifications, state["clarifications"])

    def test_state_scoped_by_session(self) -> None:
        handler, _agent, _repo = make_handler()
        run(handler.handle_message("s1", 0, "Использую Ollama"))
        other = run(handler.handle_message("s2", 0, "Привет"))
        self.assertEqual(other["task_state"]["clarifications"], [])

    def test_extractor_failure_does_not_break_turn(self) -> None:
        extractor = make_extractor(fail=True)
        handler, _agent, _repo = make_handler(extractor=extractor)
        result = run(handler.handle_message("s", 0, "Как установить?"))
        self.assertIsNone(result["error"])
        self.assertTrue(result["answer"])
        self.assertIn("task_state_error", result)
        self.assertEqual(result.get("quotes"), [])

    def test_no_extractor_still_answers(self) -> None:
        agent = make_agent(rag_enabled=True)
        handler = ChatHandler(agent, task_state_repo=None, extractor=None)
        result = run(handler.handle_message("s", 0, "Как установить?"))
        self.assertIsNone(result["error"])
        self.assertEqual(result["task_state"]["goal"], "")

    def test_reset_session(self) -> None:
        handler, _agent, repo = make_handler()
        run(handler.handle_message("s", 0, "Использую Ollama"))
        self.assertTrue(handler.reset_session("s"))
        self.assertTrue(handler.load_state("s", 0).is_empty())

    def test_load_state_without_repo(self) -> None:
        handler = ChatHandler(make_agent(), task_state_repo=None)
        state = handler.load_state("s", 3)
        self.assertEqual(state.profile_id, 3)
        self.assertFalse(handler.reset_session("s"))


class ExtractionPromptTest(unittest.TestCase):
    """JSON-вызов экстрактора: контракт ТЗ (response_format и бюджет)."""

    PAYLOAD = ('{"sources": [{"chunk_id": 12, "source": "installation.md",'
               ' "section": "## Windows"}],'
               ' "quotes": [{"chunk_id": 12,'
               ' "text": "Скачайте установщик с официального сайта."}],'
               ' "task_state": {"goal": "Цель",'
               ' "clarifications": ["используется Ollama"],'
               ' "constraints": ["без Docker"],'
               ' "terms": {"chunk": "фрагмент документа"}}}')

    def test_request_contract(self) -> None:
        extractor = make_extractor(contents=[(self.PAYLOAD, "stop")])
        state = TaskState(session_id="s", profile_id=1, goal="Прошлая цель")
        parsed = extractor.extract("Вопрос", "Ответ", state, CHUNKS)
        captured = extractor.captured[0]
        self.assertEqual(captured["response_format"],
                         {"type": "json_object"})
        self.assertGreaterEqual(captured["max_tokens"], 8000)
        self.assertIn("JSON", captured["messages"][0]["content"])
        self.assertIn("Прошлая цель", captured["messages"][0]["content"])
        self.assertIn("Скачайте установщик", captured["messages"][0]["content"])
        # Нормализация: task_state с чистыми типами
        self.assertEqual(parsed["task_state"]["goal"], "Цель")
        self.assertEqual(parsed["task_state"]["clarifications"],
                         ["используется Ollama"])
        self.assertEqual(parsed["quotes"][0]["chunk_id"], 12)

    def test_empty_retried_with_double_budget(self) -> None:
        extractor = make_extractor(contents=[
            ("", "length"), (self.PAYLOAD, "stop")])
        parsed = extractor.extract("Вопрос", "Ответ",
                                   TaskState(session_id="s", profile_id=1),
                                   CHUNKS)
        self.assertEqual(parsed["task_state"]["goal"], "Цель")
        self.assertEqual(extractor.captured[0]["max_tokens"], 8000)
        self.assertEqual(extractor.captured[1]["max_tokens"], 16000)

    def test_api_error_raises_task_state_error(self) -> None:
        extractor = make_extractor(fail=True)
        with self.assertRaises(TaskStateError):
            extractor.extract("Вопрос", "Ответ",
                              TaskState(session_id="s", profile_id=1), CHUNKS)

    def test_broken_json_raises(self) -> None:
        extractor = make_extractor(contents=[("это не json", "stop")])
        with self.assertRaises(TaskStateError):
            extractor.extract("Вопрос", "Ответ",
                              TaskState(session_id="s", profile_id=1), CHUNKS)

    def test_chunk_id_string_coerced(self) -> None:
        payload = ('{"sources": [{"chunk_id": "12"}],'
                   ' "quotes": [{"chunk_id": "12", "text": "цитата"}],'
                   ' "task_state": {}}')
        extractor = make_extractor(contents=[(payload, "stop")])
        parsed = extractor.extract("Вопрос", "Ответ",
                                   TaskState(session_id="s", profile_id=1),
                                   CHUNKS)
        self.assertEqual(parsed["quotes"][0]["chunk_id"], 12)


class MockExtractionTest(unittest.TestCase):
    """Mock-эвристики памяти задачи (офлайн-сценарии ТЗ)."""

    def test_goal_from_want(self) -> None:
        parsed = mock_task_extraction(
            "Хочу сделать RAG для своей документации", "",
            TaskState(session_id="s", profile_id=1), [])
        self.assertEqual(parsed["task_state"]["goal"],
                         "Хочу сделать RAG для своей документации")
        self.assertEqual(parsed["task_state"]["terms"]["RAG"],
                         "Retrieval-Augmented Generation")

    def test_goal_not_overwritten_by_clarification(self) -> None:
        state = TaskState(session_id="s", profile_id=1,
                          goal="Хочу сделать RAG")
        parsed = mock_task_extraction("А если хочу без Docker?", "", state, [])
        self.assertEqual(parsed["task_state"]["goal"], "")
        self.assertEqual(parsed["task_state"]["constraints"], ["без Docker"])

    def test_goal_from_debugging(self) -> None:
        parsed = mock_task_extraction(
            "RAG не находит ответы на мои вопросы", "",
            TaskState(session_id="s", profile_id=1), [])
        self.assertTrue(parsed["task_state"]["goal"].startswith("Отладка:"))

    def test_clarifications_and_threshold(self) -> None:
        parsed = mock_task_extraction("Порог стоит 0.7", "",
                                      TaskState(session_id="s",
                                                profile_id=1), [])
        self.assertIn("порог similarity 0.7",
                      parsed["task_state"]["clarifications"])

    def test_uses_and_terms(self) -> None:
        parsed = mock_task_extraction("Использую nomic-embed-text", "",
                                      TaskState(session_id="s",
                                                profile_id=1), [])
        clar = parsed["task_state"]["clarifications"]
        self.assertIn("используется nomic-embed-text", clar)
        self.assertIn("nomic-embed-text как эмбеддер", clar)
        self.assertIn("nomic-embed-text", parsed["task_state"]["terms"])

    def test_quotes_from_chunks_are_verbatim(self) -> None:
        from app.rag.citations import validate_quotes
        parsed = mock_task_extraction("Как установить?", "",
                                      TaskState(session_id="s",
                                                profile_id=1), CHUNKS)
        validated = validate_quotes(
            {"quotes": parsed["quotes"]},
            [chunk.text for chunk in CHUNKS], METADATAS)["quotes"]
        self.assertTrue(all(q["valid"] for q in validated))


if __name__ == "__main__":
    unittest.main()
