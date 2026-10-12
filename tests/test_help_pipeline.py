"""Тесты команды /help (ТЗ: HelpPipeline + parse_command + диспетчер).

Проверяются: парсер команд, полный пайплайн (RAG по project_docs →
git-контекст через MCP → промпт → ответ), fallback при недоступном
git MCP и недоступном эмбеддере, изоляция режима (память задачи НЕ
обновляется, история основного диалога не расходуется), деградация
без help_pipeline, разбор имени ветки из ответа MCP.

Запуск из корня проекта:
    python -m unittest tests.test_help_pipeline -v
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from app.chat_handler import ChatHandler, ChatTurnExtractor, parse_command
from app.help_pipeline import (
    GIT_UNAVAILABLE_MARKER,
    NO_DOCS_MARKER,
    USAGE_HINT,
    HelpPipeline,
    parse_branch_text,
)
from app.rag.embedder import EmbedderError
from app.rag.retriever import RetrievedChunk
from app.task_state import TaskStateRepository
from src.mcp_client import MCPConnectionError
from tests.test_app_agent import COMPARISON_CONFIG, MODEL, make_agent


def run(coroutine):
    return asyncio.run(coroutine)


# ----------------------------------------------------------------------
# Заглушки зависимостей пайплайна
# ----------------------------------------------------------------------
class FakeProjectRAG:
    """ProjectRAG-заглушка: чанки или ошибка эмбеддера."""

    def __init__(self, chunks=None, error=None) -> None:
        self.chunks = chunks or []
        self.error = error
        self.queries: list = []

    async def search_chunks_async(self, query, top_k=3):
        self.queries.append((query, top_k))
        if self.error is not None:
            raise self.error
        return list(self.chunks)


class FakeGitMCP:
    """MCP-клиент-заглушка: имя ветки или ошибка подключения."""

    def __init__(self, result='{"result": "main"}', error=None) -> None:
        self.result = result
        self.error = error
        self.calls: list = []

    async def call_tool(self, tool_name, arguments=None):
        self.calls.append((tool_name, dict(arguments or {})))
        if self.error is not None:
            raise self.error
        return self.result


HELP_CHUNKS = [
    RetrievedChunk(text="Архитектура: ядро агент + RAG-пайплайн.",
                   source="architecture.md", title="Архитектура",
                   section="## Слои", strategy="structure",
                   distance=0.15, chunk_id=0),
    RetrievedChunk(text="Тесты запускаются через unittest discover.",
                   source="contributing.md", title="Вклад",
                   section="## Тесты", strategy="structure",
                   distance=0.35, chunk_id=1),
]


def make_pipeline(chunks=None, rag_error=None, mcp_result=None,
                  mcp_error=None, agent=None) -> tuple:
    """HelpPipeline над mock-ядром с заглушками RAG и MCP."""
    agent = agent or make_agent(rag_enabled=True)
    git = FakeGitMCP(
        result=('{"result": "main"}' if mcp_result is None
                else mcp_result),
        error=mcp_error,
    )
    pipeline = HelpPipeline(
        FakeProjectRAG(chunks=chunks, error=rag_error),
        git,
        agent,
        project_path="C:/projects/demo",
    )
    return pipeline, agent


def capture_rag_block(agent) -> dict:
    """Шпион на core.set_rag_context: записывает блок и источники."""
    captured: dict = {}
    core = agent.core
    original = core.set_rag_context

    def spy(block, sources):
        captured["block"] = block
        captured["sources"] = sources
        original(block, sources)

    core.set_rag_context = spy
    return captured


class ParseCommandTest(unittest.TestCase):
    """Парсер команд (Модуль 3 ТЗ)."""

    def test_help_with_question(self) -> None:
        self.assertEqual(
            parse_command("/help как устроена архитектура?"),
            ("help", "как устроена архитектура?"))

    def test_help_bare_command(self) -> None:
        self.assertEqual(parse_command("/help"), ("help", ""))
        self.assertEqual(parse_command("/help   "), ("help", ""))

    def test_similar_slash_text_is_chat(self) -> None:
        self.assertEqual(parse_command("/hello world"), ("chat", "/hello world"))
        self.assertEqual(parse_command("обычный вопрос"), ("chat", "обычный вопрос"))
        self.assertEqual(parse_command(""), ("chat", ""))


class ParseBranchTextTest(unittest.TestCase):
    """Имя ветки из ответа MCP (JSON-обёртка FastMCP или сырая строка)."""

    def test_unwraps_structured_result(self) -> None:
        self.assertEqual(parse_branch_text('{"result": "main"}'), "main")

    def test_plain_text(self) -> None:
        self.assertEqual(parse_branch_text("feature_x"), "feature_x")

    def test_detached_head_string(self) -> None:
        self.assertEqual(
            parse_branch_text('{"result": "detached HEAD (abc1234)"}'),
            "detached HEAD (abc1234)")

    def test_empty(self) -> None:
        self.assertEqual(parse_branch_text(""), "")
        self.assertEqual(parse_branch_text(None), "")


class HelpPipelineTest(unittest.TestCase):
    """Полный пайплайн /help над mock-ядром."""

    def test_full_pipeline_prompt_and_result(self) -> None:
        pipeline, agent = make_pipeline(chunks=HELP_CHUNKS)
        captured = capture_rag_block(agent)
        result = run(pipeline.handle_help("как устроена архитектура?"))

        # Источники и флаги режима
        self.assertTrue(result["is_help"])
        self.assertTrue(result["text"], "ответ не пустой")
        self.assertEqual(result["answer"], result["text"])
        self.assertTrue(result["rag_used"])
        self.assertEqual(len(result["sources"]), 2)
        self.assertEqual(result["git_context"], "Текущая git-ветка: main")

        # MCP-вызов: правильный инструмент и путь проекта
        rag = pipeline.project_rag
        self.assertEqual(rag.queries, [("как устроена архитектура?", 3)])
        git = pipeline.mcp_client
        self.assertEqual(
            git.calls,
            [("get_current_branch", {"repo_path": "C:/projects/demo"})],
        )

        # Промпт: инструкция + документация + git + вопрос
        block = captured["block"]
        self.assertIn("ассистент разработчика", block)
        self.assertIn("[Документация проекта]", block)
        self.assertIn("Архитектура: ядро агент + RAG-пайплайн.", block)
        self.assertIn("[Git-контекст]", block)
        self.assertIn("Текущая git-ветка: main", block)
        self.assertIn("[Вопрос]", block)
        self.assertIn("как устроена архитектура?", block)
        # Источники уходят панели UI
        self.assertEqual(len(captured["sources"]), 2)

    def test_git_unavailable_fallback(self) -> None:
        pipeline, agent = make_pipeline(
            chunks=HELP_CHUNKS,
            mcp_error=MCPConnectionError("сервер не запустился"))
        result = run(pipeline.handle_help("вопрос?"))
        # /help работает и без git (ТЗ: fallback обязателен)
        self.assertTrue(result["text"])
        self.assertIn(
            GIT_UNAVAILABLE_MARKER.format(reason="сервер не запустился"),
            result["git_context"])
        self.assertEqual(result["git_error"], "сервер не запустился")

    def test_rag_unavailable_fallback(self) -> None:
        pipeline, agent = make_pipeline(
            rag_error=EmbedderError("Ollama недоступна"))
        captured = capture_rag_block(agent)
        result = run(pipeline.handle_help("вопрос?"))
        self.assertTrue(result["text"])
        self.assertFalse(result["rag_used"])
        self.assertEqual(result["rag_error"], "Ollama недоступна")
        self.assertIn(NO_DOCS_MARKER, captured["block"])
        # Git-контекст при этом всё равно собирается
        self.assertEqual(result["git_context"], "Текущая git-ветка: main")

    def test_empty_question_usage_hint(self) -> None:
        pipeline, agent = make_pipeline()
        history_before = len(agent.core._message_history)
        result = run(pipeline.handle_help("   "))
        self.assertEqual(result["text"], USAGE_HINT)
        self.assertTrue(result["is_help"])
        # Ни LLM, ни MCP, ни RAG не вызывались
        self.assertEqual(pipeline.mcp_client.calls, [])
        self.assertEqual(pipeline.project_rag.queries, [])
        self.assertEqual(len(agent.core._message_history), history_before)

    def test_isolated_core_history_not_polluted(self) -> None:
        """Ядро агента (не /help) не получает вопрос: изолированный режим."""
        agent = make_agent(rag_enabled=True)
        agent.bind_config(dict(COMPARISON_CONFIG))
        pipeline = HelpPipeline(
            FakeProjectRAG(chunks=HELP_CHUNKS), FakeGitMCP(),
            agent, project_path="C:/projects/demo")
        run(pipeline.handle_help("как устроена архитектура?"))
        core = agent.core
        roles = [m.get("role") for m in core._message_history
                 if m.get("role") in ("user", "assistant")]
        self.assertEqual(roles, [])


class ChatHandlerHelpDispatchTest(unittest.TestCase):
    """Диспетчер ChatHandler: /help — отдельный режим без памяти задачи."""

    class FakeHelpPipeline:
        """Запоминает payload, возвращает готовый результат."""

        def __init__(self) -> None:
            self.payloads: list = []
            self.result = {
                "text": "Ответ помощи", "answer": "Ответ помощи",
                "sources": [], "git_context": "Текущая git-ветка: main",
                "rag_used": True, "has_rag": True, "is_help": True,
            }

        async def handle_help(self, question: str) -> dict:
            self.payloads.append(question)
            return dict(self.result)

    def setUp(self) -> None:
        self.agent = make_agent(rag_enabled=True)
        self.repo = TaskStateRepository(
            Path(tempfile.mkdtemp()) / "help_dispatch.db")
        self.pipeline = self.FakeHelpPipeline()
        self.handler = ChatHandler(
            self.agent,
            task_state_repo=self.repo,
            extractor=ChatTurnExtractor(api_key="test", mock_mode=True),
            help_pipeline=self.pipeline,
        )

    def test_help_command_goes_to_pipeline(self) -> None:
        result = run(self.handler.handle_message(
            "session-1", 1, "/help как устроена архитектура?"))
        self.assertEqual(self.pipeline.payloads,
                         ["как устроена архитектура?"])
        self.assertTrue(result["is_help"])
        self.assertEqual(result["text"], "Ответ помощи")

    def test_help_does_not_touch_task_state(self) -> None:
        result = run(self.handler.handle_message("session-1", 1, "/help вопрос?"))
        # Память задачи не создана и не обновлена (ТЗ: отдельный режим)
        state = self.handler.load_state("session-1", 1)
        self.assertEqual(state.to_dict()["goal"], "")
        self.assertEqual(state.to_dict()["clarifications"], [])
        # Ход не содержит следов обычного пайплайна (память/цитаты)
        self.assertNotIn("task_state", result)
        self.assertNotIn("quotes", result)

    def test_normal_message_still_updates_task_state(self) -> None:
        run(self.handler.handle_message(
            "session-1", 1, "Хочу настроить RAG поиск"))
        self.assertEqual(self.pipeline.payloads, [])
        state = self.handler.load_state("session-1", 1)
        self.assertNotEqual(state.to_dict()["goal"], "")

    def test_help_without_pipeline_degrades_to_chat(self) -> None:
        handler = ChatHandler(
            self.agent,
            task_state_repo=self.repo,
            extractor=ChatTurnExtractor(api_key="test", mock_mode=True),
            help_pipeline=None,
        )
        result = run(handler.handle_message("session-2", 1, "/help вопрос?"))
        # Деградация: обычный ход чата (ответ есть, is_help нет)
        self.assertNotIn("is_help", result)
        self.assertIn("task_state", result)


class IsolatedCoreTest(unittest.TestCase):
    """isolated_core RAGChatAgent: отдельное ядро без хранилища."""

    def test_isolated_core_cached_and_separate(self) -> None:
        agent = make_agent(rag_enabled=True)
        agent.bind_config(dict(COMPARISON_CONFIG))
        first = agent.isolated_core()
        second = agent.isolated_core()
        self.assertIs(first, second)
        self.assertIsNot(first, agent.core)
        self.assertIsNone(first._storage)

    def test_isolated_core_answer_uses_mock(self) -> None:
        agent = make_agent(rag_enabled=True)
        agent.bind_config(dict(COMPARISON_CONFIG))
        core = agent.isolated_core()
        self.assertEqual(core.model, MODEL)
        result = run(asyncio.to_thread(core.send_message, "привет"))
        self.assertTrue(str(result.get("text") or ""))


if __name__ == "__main__":
    unittest.main()
