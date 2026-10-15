"""Тесты команды /support (ТЗ ai_1_34: SupportPipeline + диспетчер).

Проверяются: разбор ответа CRM, форматирование контекста пользователя,
полный пайплайн (RAG по support_faq → CRM → промпт → ответ),
фолбэки (CRM недоступен, пользователь не выбран/не найден, эмбеддер
недоступен), изоляция режима (память задачи НЕ обновляется, история
основного диалога не расходуется), деградация без пайплайна, передача
user_id из UI через get_support_user_id.

Запуск из корня проекта:
    python -m unittest tests.test_support_pipeline -v
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from app.chat_handler import ChatHandler, ChatTurnExtractor, parse_command
from app.rag.embedder import EmbedderError
from app.rag.retriever import RetrievedChunk
from app.support_pipeline import (
    CRM_UNAVAILABLE_MARKER,
    NO_FAQ_MARKER,
    NO_USER_MARKER,
    SUPPORT_SYSTEM_INSTRUCTION,
    USAGE_HINT,
    SupportPipeline,
    parse_user_context,
)
from app.task_state import TaskStateRepository
from src.mcp_client import MCPConnectionError
from tests.test_app_agent import COMPARISON_CONFIG, MODEL, make_agent


def run(coroutine):
    return asyncio.run(coroutine)


# ----------------------------------------------------------------------
# Заглушки зависимостей пайплайна
# ----------------------------------------------------------------------
class FakeSupportRAG:
    """SupportRAG-заглушка: чанки FAQ или ошибка эмбеддера."""

    def __init__(self, chunks=None, error=None) -> None:
        self.chunks = chunks or []
        self.error = error
        self.queries: list = []

    async def search_chunks_async(self, query, top_k=3):
        self.queries.append((query, top_k))
        if self.error is not None:
            raise self.error
        return list(self.chunks)


class FakeCrmClient:
    """HTTP-клиент CRM-заглушка: контекст или ошибка подключения."""

    def __init__(self, context=None, error=None) -> None:
        self.context = context if context is not None else {
            "user": {
                "id": 42, "name": "Иван Петров",
                "email": "ivan@example.com", "status": "active",
            },
            "tickets": [{
                "id": 101, "subject": "Не работает авторизация",
                "status": "open", "created_at": "2026-10-14",
                "priority": "high",
            }],
        }
        self.error = error
        self.calls: list = []

    async def call_tool(self, tool_name, arguments=None):
        self.calls.append((tool_name, dict(arguments or {})))
        if self.error is not None:
            raise self.error
        return json.dumps({"result": self.context}, ensure_ascii=False)


FAQ_CHUNKS = [
    RetrievedChunk(
        text="Вход блокируется для статуса pending_verification: "
             "аккаунт активируется после перехода по ссылке из письма.",
        source="faq_authorization.md", title="FAQ: Авторизация",
        section="## Не работает авторизация", strategy="structure",
        distance=0.15, chunk_id=0),
    RetrievedChunk(
        text="Сброс пароля доступен по ссылке «Забыли пароль?»; "
             "ссылка одноразовая и действует 1 час.",
        source="faq_authorization.md", title="FAQ: Авторизация",
        section="## Сброс пароля", strategy="structure",
        distance=0.35, chunk_id=1),
]


def make_pipeline(chunks=None, rag_error=None, crm_context=None,
                  crm_error=None, agent=None) -> tuple:
    """SupportPipeline над mock-ядром с заглушками RAG и CRM."""
    agent = agent or make_agent(rag_enabled=True)
    crm = FakeCrmClient(context=crm_context, error=crm_error)
    pipeline = SupportPipeline(
        FakeSupportRAG(chunks=chunks, error=rag_error),
        crm,
        agent,
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


# ----------------------------------------------------------------------
# Парсер ответа CRM и форматирование контекста
# ----------------------------------------------------------------------
class ParseUserContextTest(unittest.TestCase):
    """Контекст пользователя из ответа MCP (JSON-обёртка FastMCP)."""

    def test_unwraps_structured_result(self) -> None:
        raw = json.dumps({"result": {"user": {"id": 42}}})
        self.assertEqual(parse_user_context(raw), {"user": {"id": 42}})

    def test_error_passthrough(self) -> None:
        raw = json.dumps({"result": {"error": "Пользователь 999 не найден"}})
        self.assertEqual(
            parse_user_context(raw), {"error": "Пользователь 999 не найден"})

    def test_garbage_is_empty(self) -> None:
        self.assertEqual(parse_user_context("не json"), {})
        self.assertEqual(parse_user_context(""), {})
        self.assertEqual(parse_user_context(None), {})


class FormatUserContextTest(unittest.TestCase):
    """Форматирование контекста для промпта (ТЗ)."""

    def test_full_context_with_tickets(self) -> None:
        text = SupportPipeline._format_user_context({
            "user": {"name": "Иван Петров", "email": "ivan@example.com",
                     "status": "active"},
            "tickets": [{"id": 101, "subject": "Не работает авторизация",
                         "status": "open", "priority": "high"}],
        })
        self.assertIn("Иван Петров (ivan@example.com)", text)
        self.assertIn("Статус: active", text)
        self.assertIn("#101: Не работает авторизация (open, high)", text)

    def test_no_tickets(self) -> None:
        text = SupportPipeline._format_user_context({
            "user": {"name": "Мария", "email": "m@e.com",
                     "status": "pending_verification"},
            "tickets": [],
        })
        self.assertIn("Тикеты: нет", text)

    def test_error_context(self) -> None:
        text = SupportPipeline._format_user_context(
            {"error": "Пользователь 999 не найден"})
        self.assertEqual(text, "Пользователь 999 не найден")


# ----------------------------------------------------------------------
# Полный пайплайн /support
# ----------------------------------------------------------------------
class SupportPipelineTest(unittest.TestCase):
    """Пайплайн /support над mock-ядром."""

    def test_full_pipeline_prompt_and_result(self) -> None:
        """ТЗ: контекст пользователя (тикет #101) уходит в промпт."""
        pipeline, agent = make_pipeline(chunks=FAQ_CHUNKS)
        captured = capture_rag_block(agent)
        result = run(pipeline.handle_support(
            "Почему не работает авторизация?", 42))

        self.assertTrue(result["is_support"])
        self.assertTrue(result["text"], "ответ не пустой")
        self.assertEqual(result["answer"], result["text"])
        self.assertTrue(result["rag_used"])
        self.assertTrue(result["crm_available"])
        self.assertEqual(result["user_id"], 42)
        self.assertEqual(len(result["sources"]), 2)

        # RAG-вызов: правильный вопрос и top_k
        self.assertEqual(pipeline.support_rag.queries,
                         [("Почему не работает авторизация?", 3)])
        # CRM-вызов: get_user_context с user_id из UI
        self.assertEqual(pipeline.mcp_client.calls,
                         [("get_user_context", {"user_id": 42})])

        # Промпт: инструкция + контекст пользователя + FAQ + вопрос
        block = captured["block"]
        self.assertIn(SUPPORT_SYSTEM_INSTRUCTION, block)
        self.assertIn("[Контекст пользователя]", block)
        self.assertIn("Иван Петров (ivan@example.com)", block)
        self.assertIn("#101: Не работает авторизация", block)
        self.assertIn("[FAQ и документация]", block)
        self.assertIn("pending_verification", block)
        self.assertIn("[Вопрос]", block)
        self.assertIn("Почему не работает авторизация?", block)
        # Источники уходят панели UI
        self.assertEqual(len(captured["sources"]), 2)

    def test_crm_unavailable_fallback(self) -> None:
        """ТЗ: CRM недоступен — ответ через RAG без контекста."""
        pipeline, agent = make_pipeline(
            chunks=FAQ_CHUNKS,
            crm_error=MCPConnectionError("CRM-сервер не запущен"))
        captured = capture_rag_block(agent)
        result = run(pipeline.handle_support("вопрос?", 42))
        self.assertTrue(result["text"])
        self.assertFalse(result["crm_available"])
        self.assertEqual(result["crm_error"], "CRM-сервер не запущен")
        self.assertIn(
            CRM_UNAVAILABLE_MARKER.format(reason="CRM-сервер не запущен"),
            captured["block"])
        # FAQ при этом в промпте есть
        self.assertIn("[FAQ и документация]", captured["block"])

    def test_no_user_selected_fallback(self) -> None:
        """user_id не выбран (0) — CRM не вызывается, фолбэк."""
        pipeline, agent = make_pipeline(chunks=FAQ_CHUNKS)
        captured = capture_rag_block(agent)
        result = run(pipeline.handle_support("вопрос?", 0))
        self.assertTrue(result["text"])
        self.assertFalse(result["crm_available"])
        self.assertIn(NO_USER_MARKER, captured["block"])
        self.assertEqual(pipeline.mcp_client.calls, [])

    def test_unknown_user_in_crm(self) -> None:
        """CRM ответил {"error": …} — контекста нет, фолбэк."""
        pipeline, agent = make_pipeline(
            chunks=FAQ_CHUNKS,
            crm_context={"error": "Пользователь 999 не найден"})
        captured = capture_rag_block(agent)
        result = run(pipeline.handle_support("вопрос?", 999))
        self.assertTrue(result["text"])
        self.assertFalse(result["crm_available"])
        self.assertIn("Пользователь 999 не найден", captured["block"])

    def test_rag_unavailable_fallback(self) -> None:
        """Эмбеддер недоступен — ответ без FAQ, CRM работает."""
        pipeline, agent = make_pipeline(
            rag_error=EmbedderError("Ollama недоступна"))
        captured = capture_rag_block(agent)
        result = run(pipeline.handle_support("вопрос?", 42))
        self.assertTrue(result["text"])
        self.assertFalse(result["rag_used"])
        self.assertEqual(result["rag_error"], "Ollama недоступна")
        self.assertIn(NO_FAQ_MARKER, captured["block"])
        self.assertTrue(result["crm_available"])

    def test_empty_question_usage_hint(self) -> None:
        pipeline, agent = make_pipeline()
        history_before = len(agent.core._message_history)
        result = run(pipeline.handle_support("   ", 42))
        self.assertEqual(result["text"], USAGE_HINT)
        self.assertTrue(result["is_support"])
        # Ни LLM, ни CRM, ни RAG не вызывались
        self.assertEqual(pipeline.mcp_client.calls, [])
        self.assertEqual(pipeline.support_rag.queries, [])
        self.assertEqual(len(agent.core._message_history), history_before)

    def test_isolated_core_history_not_polluted(self) -> None:
        """Ядро агента (не /support) не получает вопрос: изоляция."""
        agent = make_agent(rag_enabled=True)
        agent.bind_config(dict(COMPARISON_CONFIG))
        pipeline = SupportPipeline(
            FakeSupportRAG(chunks=FAQ_CHUNKS), FakeCrmClient(), agent)
        run(pipeline.handle_support("Почему не работает авторизация?", 42))
        roles = [m.get("role") for m in agent.core._message_history
                 if m.get("role") in ("user", "assistant")]
        self.assertEqual(roles, [])


class ParseCommandSupportTest(unittest.TestCase):
    """Парсер команд: /support перехватывается (ТЗ ai_1_34)."""

    def test_support_with_question(self) -> None:
        self.assertEqual(
            parse_command("/support Почему не работает авторизация?"),
            ("support", "Почему не работает авторизация?"))

    def test_support_bare_command(self) -> None:
        self.assertEqual(parse_command("/support"), ("support", ""))
        self.assertEqual(parse_command("/support   "), ("support", ""))

    def test_help_still_parsed(self) -> None:
        self.assertEqual(parse_command("/help вопрос"), ("help", "вопрос"))

    def test_plain_text_is_chat(self) -> None:
        self.assertEqual(parse_command("/supported"), ("chat", "/supported"))
        self.assertEqual(parse_command("обычный вопрос"),
                         ("chat", "обычный вопрос"))


class ChatHandlerSupportDispatchTest(unittest.TestCase):
    """Диспетчер ChatHandler: /support — отдельный режим, user_id из UI."""

    class FakeSupportPipeline:
        """Запоминает payload и user_id, возвращает готовый результат."""

        def __init__(self) -> None:
            self.calls: list = []
            self.result = {
                "text": "Ответ поддержки", "answer": "Ответ поддержки",
                "sources": [], "user_context": "Пользователь: Иван",
                "crm_available": True, "user_id": 42,
                "rag_used": True, "has_rag": True, "is_support": True,
            }

        async def handle_support(self, question: str, user_id: int) -> dict:
            self.calls.append((question, user_id))
            return dict(self.result)

    def setUp(self) -> None:
        self.agent = make_agent(rag_enabled=True)
        self.repo = TaskStateRepository(
            Path(tempfile.mkdtemp()) / "support_dispatch.db")
        self.pipeline = self.FakeSupportPipeline()
        self.handler = ChatHandler(
            self.agent,
            task_state_repo=self.repo,
            extractor=ChatTurnExtractor(api_key="test", mock_mode=True),
            support_pipeline=self.pipeline,
            get_support_user_id=lambda: 42,
        )

    def test_support_command_goes_to_pipeline_with_ui_user(self) -> None:
        """ТЗ: user_id берётся из UI (get_support_user_id), не хардкод."""
        result = run(self.handler.handle_message(
            "session-1", 1, "/support Почему не работает авторизация?"))
        self.assertEqual(self.pipeline.calls,
                         [("Почему не работает авторизация?", 42)])
        self.assertTrue(result["is_support"])
        self.assertEqual(result["text"], "Ответ поддержки")

    def test_support_does_not_touch_task_state(self) -> None:
        result = run(self.handler.handle_message("session-1", 1,
                                                 "/support вопрос?"))
        # Память задачи не создана и не обновлена (ТЗ: отдельный режим)
        state = self.handler.load_state("session-1", 1)
        self.assertEqual(state.to_dict()["goal"], "")
        self.assertEqual(state.to_dict()["clarifications"], [])
        self.assertNotIn("task_state", result)
        self.assertNotIn("quotes", result)

    def test_invalid_user_id_degrades_to_zero(self) -> None:
        """get_support_user_id вернул мусор — ход не падает (user_id=0)."""
        self.handler.get_support_user_id = lambda: "не число"
        run(self.handler.handle_message("session-1", 1, "/support вопрос?"))
        self.assertEqual(self.pipeline.calls, [("вопрос?", 0)])

    def test_normal_message_still_updates_task_state(self) -> None:
        run(self.handler.handle_message(
            "session-1", 1, "Хочу настроить RAG поиск"))
        self.assertEqual(self.pipeline.calls, [])
        state = self.handler.load_state("session-1", 1)
        self.assertNotEqual(state.to_dict()["goal"], "")

    def test_support_without_pipeline_degrades_to_chat(self) -> None:
        handler = ChatHandler(
            self.agent,
            task_state_repo=self.repo,
            extractor=ChatTurnExtractor(api_key="test", mock_mode=True),
            support_pipeline=None,
        )
        result = run(handler.handle_message("session-2", 1, "/support х"))
        self.assertNotIn("is_support", result)
        self.assertIn("task_state", result)


if __name__ == "__main__":
    unittest.main()
