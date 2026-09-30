"""Тесты RAG-интеграции: хук ядра (src/agent.py) и обёртка app/agent.py.

Проверяются: set_rag_context меняет только копию запроса (история
чистая), одноразовость контекста, sources в результате, режимы с/без
RAG «на лету», фолбэк при недоступном Ollama, изолированный answer_once.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import unittest

from app.agent import DEFAULT_TOP_K, RAGChatAgent
from app.rag.embedder import EmbedderError
from app.rag.retriever import RetrievedChunk
from src.agent import DeepSeekAgent

MODEL = "deepseek-v4-flash"

COMPARISON_CONFIG = {
    "DEEPSEEK_API_KEY": "test",
    "DEFAULT_MODEL": MODEL,
    "DEFAULT_SYSTEM_PROMPT": "Ты помощник.",
    "DEFAULT_TEMPERATURE": 0.5,
    "DEFAULT_MAX_TOKENS": 100,
    "MOCK_MODE": True,
    "MAX_HISTORY_MESSAGES": 10,
}


class FakeEmbedder:
    """Эмбеддер-заглушка (без сети)."""

    def __init__(self) -> None:
        self.queries: list = []
        self.fail = False

    def embed_query(self, text: str) -> list:
        return [0.1, 0.2]

    async def embed_query_async(self, text: str) -> list:
        if self.fail:
            raise EmbedderError("Ollama недоступна по адресу "
                                "http://localhost:11434")
        self.queries.append(text)
        return [0.1, 0.2]

    def embed_documents(self, texts) -> list:
        return [[0.1, 0.2] for _ in texts]


STORE_RESULT = {
    "ids": [["chunk_1", "chunk_2"]],
    "documents": [["Чанк про установку Windows.", "Чанк про Ollama."]],
    "metadatas": [[
        {"source": "installation.md", "title": "Установка",
         "section": "## Windows", "strategy": "structure", "chunk_id": 1},
        {"source": "troubleshooting.md", "title": "Неполадки",
         "section": "## Ollama", "strategy": "fixed", "chunk_id": 2},
    ]],
    "distances": [[0.12, 0.34]],
}


class FakeStore:
    """Хранилище-заглушка."""

    def __init__(self, result=None) -> None:
        self.result = result if result is not None else STORE_RESULT
        self.last_k: int = -1

    def count(self, collection_name="knowledge_base") -> int:
        return 2

    def search(self, query_embedding, k=3, collection_name="knowledge_base"):
        self.last_k = k
        return self.result


def make_agent(rag_enabled: bool = False,
               embedder: FakeEmbedder = None,
               store: FakeStore = None,
               top_k: int = DEFAULT_TOP_K) -> RAGChatAgent:
    """Собирает RAGChatAgent над mock-ядром без хранилища."""
    core = DeepSeekAgent(api_key="test", model=MODEL, mock_mode=True,
                         storage=None)
    return RAGChatAgent(
        core,
        embedder=embedder or FakeEmbedder(),
        vector_store=store or FakeStore(),
        rag_enabled=rag_enabled,
        top_k=top_k,
    )


def last_user_content(messages: list) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def run(coroutine):
    return asyncio.run(coroutine)


class RagHookCoreTest(unittest.TestCase):
    """Хук set_rag_context в ядре DeepSeekAgent."""

    def setUp(self) -> None:
        self.core = DeepSeekAgent(api_key="test", model=MODEL,
                                  mock_mode=True, storage=None)
        # В истории есть вопрос пользователя (как перед send_message)
        self.core._append_message({"role": "user",
                                   "content": "Как установить приложение?"})

    def test_build_request_augmented_only_in_copy(self) -> None:
        prompt = "Контекст:\nЧанк.\n\nВопрос: Как установить приложение?"
        self.core.set_rag_context(prompt, [{"source": "a.md"}])
        request = self.core._build_request()
        self.assertIn("Контекст:", last_user_content(request["messages"]))
        # История хранит вопрос как есть
        self.assertNotIn("Контекст:",
                         last_user_content(self.core._message_history))

    def test_apply_rag_block_does_not_mutate_history(self) -> None:
        prompt = "Контекст:\nЧанк."
        self.core.set_rag_context(prompt, [])
        messages = self.core._build_context_messages()
        augmented = self.core._apply_rag_block(messages)
        self.assertIn("Контекст:", last_user_content(augmented))
        self.assertNotIn("Контекст:", last_user_content(messages))
        self.assertIsNot(augmented, messages)

    def test_clear_rag_context(self) -> None:
        self.core.set_rag_context("БЛОК", [])
        self.core.clear_rag_context()
        request = self.core._build_request()
        self.assertNotIn("БЛОК", last_user_content(request["messages"]))

    def test_send_message_returns_sources_and_resets(self) -> None:
        agent = make_agent(rag_enabled=True)
        result = run(agent.send_message("Как установить приложение?"))
        self.assertTrue(result.get("rag_used"))
        self.assertEqual(len(result.get("sources") or []), 2)
        self.assertEqual(result["sources"][0]["source"], "installation.md")
        # Поиск идёт через RAGPipeline: Chroma запрашивается top_k_search
        # (кандидатов до фильтрации/реранка), контекст — top_k (после)
        store = agent.vector_store
        self.assertEqual(store.last_k, agent.top_k_search)

    def test_history_keeps_raw_question_after_rag_turn(self) -> None:
        agent = make_agent(rag_enabled=True)
        run(agent.send_message("Как установить приложение?"))
        for message in agent.core._message_history:
            self.assertNotIn("Контекст:", str(message.get("content")))

    def test_token_estimate_includes_context(self) -> None:
        # Один и тот же вопрос: с блоком контекста оценка токенов выше
        plain = make_agent()
        with_context = make_agent()
        with_context.core.set_rag_context(
            "Контекст:\n" + "очень длинный контекст " * 100 +
            "\n\nВопрос: Как установить?", [])
        r_plain = run(plain.send_message("Как установить?"))
        r_context = run(with_context.send_message("Как установить?"))
        tokens_plain = r_plain["tokens"]["prompt_tokens"]
        tokens_context = r_context["tokens"]["prompt_tokens"]
        self.assertGreater(tokens_context, tokens_plain)


class RAGChatAgentModesTest(unittest.TestCase):
    """Два режима и переключение «на лету»."""

    def test_mode_without_rag_no_sources(self) -> None:
        agent = make_agent(rag_enabled=False)
        result = run(agent.send_message("Привет"))
        self.assertIsNone(result.get("error"))
        self.assertNotIn("sources", result)
        self.assertNotIn("rag_used", result)

    def test_mode_with_rag_returns_sources(self) -> None:
        agent = make_agent(rag_enabled=True)
        result = run(agent.send_message("Как установить приложение?"))
        self.assertTrue(result["rag_used"])
        sources = result["sources"]
        self.assertEqual({s["source"] for s in sources},
                         {"installation.md", "troubleshooting.md"})

    def test_switch_on_the_fly(self) -> None:
        agent = make_agent(rag_enabled=False)
        run(agent.send_message("Вопрос 1"))
        agent.set_rag_enabled(True)
        result = run(agent.send_message("Вопрос 2"))
        self.assertTrue(result["rag_used"])
        agent.set_rag_enabled(False)
        result = run(agent.send_message("Вопрос 3"))
        self.assertNotIn("sources", result)

    def test_top_k_search_applied_to_search(self) -> None:
        # Поиск (top_k до фильтрации/реранка) настраивается отдельно
        agent = make_agent(rag_enabled=True, top_k=2)
        agent.set_top_k_search(15)
        run(agent.send_message("Вопрос"))
        self.assertEqual(agent.vector_store.last_k, 15)
        agent.set_top_k_search(5)
        run(agent.send_message("Вопрос"))
        self.assertEqual(agent.vector_store.last_k, 5)

    def test_context_top_k_limits_final_chunks(self) -> None:
        # top_k (контекст после фильтрации/реранка) — прежняя семантика:
        # из двух найденных кандидатов в контекст попадает один
        agent = make_agent(rag_enabled=True, top_k=1)
        result = run(agent.send_message("Вопрос"))
        self.assertEqual(len(result.get("sources") or []), 1)
        self.assertEqual(agent.top_k_final, 1)

    def test_set_top_k_validation(self) -> None:
        agent = make_agent()
        with self.assertRaises(ValueError):
            agent.set_top_k(0)
        with self.assertRaises(ValueError):
            agent.set_top_k(99)

    def test_retriever_shares_top_k_on_init(self) -> None:
        agent = make_agent(top_k=4)
        self.assertEqual(agent.retriever.top_k, 4)


class RAGFallbackTest(unittest.TestCase):
    """Недоступный Ollama / пустая база не ломают диалог."""

    def test_embedder_error_fallback(self) -> None:
        embedder = FakeEmbedder()
        embedder.fail = True
        agent = make_agent(rag_enabled=True, embedder=embedder)
        result = run(agent.send_message("Вопрос"))
        self.assertIsNone(result["error"])  # ответ получен
        self.assertIn("Ollama недоступна", result.get("rag_error", ""))
        self.assertNotIn("sources", result)

    def test_empty_base_fallback(self) -> None:
        empty_store = FakeStore(result={"ids": [[]], "documents": [[]],
                                        "metadatas": [[]], "distances": [[]]})
        agent = make_agent(rag_enabled=True, store=empty_store)
        result = run(agent.send_message("Вопрос"))
        self.assertIsNone(result["error"])
        self.assertIn("База знаний пуста", result.get("rag_error", ""))


class AnswerOnceTest(unittest.TestCase):
    """Изолированный ответ для вкладки «Сравнение»."""

    def test_answer_once_does_not_touch_main_history(self) -> None:
        agent = make_agent(rag_enabled=True)
        agent.bind_config(dict(COMPARISON_CONFIG))
        run(agent.send_message("Сообщение в чат"))
        before = len(agent.core._message_history)
        result = run(agent.answer_once("Контрольный вопрос",
                                       rag_enabled=True))
        self.assertIsNone(result["error"])
        self.assertTrue(result["rag_used"])
        self.assertEqual(len(agent.core._message_history), before)

    def test_answer_once_rag_off(self) -> None:
        agent = make_agent()
        agent.bind_config(dict(COMPARISON_CONFIG))
        result = run(agent.answer_once("Вопрос", rag_enabled=False))
        self.assertNotIn("sources", result)
        self.assertNotIn("rag_error", result)

    def test_answer_once_without_config_raises(self) -> None:
        agent = make_agent()
        with self.assertRaises(RuntimeError):
            run(agent.answer_once("Вопрос", rag_enabled=False))

    def test_comparison_core_reused(self) -> None:
        agent = make_agent()
        agent.bind_config(dict(COMPARISON_CONFIG))
        run(agent.answer_once("Вопрос 1", rag_enabled=False))
        first = agent._comparison_core
        run(agent.answer_once("Вопрос 2", rag_enabled=False))
        self.assertIs(agent._comparison_core, first)


class DelegationTest(unittest.TestCase):
    """Делегирование настроек к ядру."""

    def test_update_config(self) -> None:
        agent = make_agent()
        agent.update_config(temperature=0.9)
        self.assertEqual(agent.temperature, 0.9)

    def test_count_chunks(self) -> None:
        agent = make_agent()
        self.assertEqual(agent.count_chunks(), 2)


if __name__ == "__main__":
    unittest.main()
