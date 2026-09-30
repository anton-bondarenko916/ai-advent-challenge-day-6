"""Тесты RAG-режимов в RAGChatAgent (app/agent.py): настройки пайплайна.

Проверяются: set_rag_mode / set_similarity_threshold / set_top_k_search
(валидация), поиск через RAGPipeline в send_message (search_query,
диагностика retrieval), answer_once с режимом для вкладки «Сравнение
RAG-режимов», top_k_final как алиас top_k.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import unittest

from app.agent import RAGChatAgent
from app.rag.pipeline import RAG_MODES, RAG_MODE_REWRITE
from app.rag.rewriter import RewriteError
from tests.test_app_agent import (
    COMPARISON_CONFIG,
    MODEL,
    FakeEmbedder,
    FakeStore,
    make_agent,
)


class FakeRewriter:
    """Переформулировщик-заглушка."""

    def __init__(self, rewritten="установка Windows инструкция",
                 fail=False) -> None:
        self.rewritten = rewritten
        self.fail = fail
        self.calls: list = []

    def rewrite(self, question: str) -> str:
        self.calls.append(question)
        if self.fail:
            raise RewriteError("DeepSeek недоступен")
        return self.rewritten

    async def rewrite_async(self, question: str) -> str:
        return self.rewrite(question)


class SettingsTest(unittest.TestCase):
    """Настройки пайплайна: режим, порог, top_k (до/после)."""

    def test_defaults(self) -> None:
        agent = make_agent()
        self.assertEqual(agent.rag_mode, "basic")
        self.assertEqual(agent.top_k_search, 20)
        self.assertEqual(agent.top_k_final, 3)
        self.assertAlmostEqual(agent.similarity_threshold, 0.7)

    def test_set_rag_mode(self) -> None:
        agent = make_agent()
        for mode in RAG_MODES:
            self.assertEqual(agent.set_rag_mode(mode), mode)
        self.assertEqual(agent.rag_mode, "rewrite_reranker")

    def test_set_rag_mode_validation(self) -> None:
        agent = make_agent()
        with self.assertRaises(ValueError):
            agent.set_rag_mode("hybrid")

    def test_set_similarity_threshold(self) -> None:
        agent = make_agent()
        self.assertAlmostEqual(agent.set_similarity_threshold(0.55), 0.55)
        self.assertAlmostEqual(agent.similarity_threshold, 0.55)

    def test_set_similarity_threshold_validation(self) -> None:
        agent = make_agent()
        with self.assertRaises(ValueError):
            agent.set_similarity_threshold(1.5)
        with self.assertRaises(ValueError):
            agent.set_similarity_threshold(-0.1)

    def test_set_top_k_search(self) -> None:
        agent = make_agent()
        agent.set_top_k_search(20)
        self.assertEqual(agent.top_k_search, 20)

    def test_set_top_k_search_validation(self) -> None:
        agent = make_agent()
        with self.assertRaises(ValueError):
            agent.set_top_k_search(0)
        with self.assertRaises(ValueError):
            agent.set_top_k_search(999)

    def test_top_k_final_is_alias_of_top_k(self) -> None:
        agent = make_agent(top_k=5)
        self.assertEqual(agent.top_k_final, 5)
        agent.set_top_k_final(2)
        self.assertEqual(agent.top_k, 2)
        self.assertEqual(agent.top_k_final, 2)


class SendViaPipelineTest(unittest.TestCase):
    """send_message идёт через RAGPipeline (поиск + диагностика)."""

    def test_search_uses_top_k_search(self) -> None:
        store = FakeStore()
        agent = make_agent(rag_enabled=True, store=store)
        agent.set_top_k_search(20)
        run(agent.send_message("Как установить приложение?"))
        # Chroma запрашивается top_k_search (20), а не top_k_final
        self.assertEqual(store.last_k, 20)

    def test_result_contains_retrieval_info(self) -> None:
        agent = make_agent(rag_enabled=True)
        result = run(agent.send_message("Как установить приложение?"))
        retrieval = result.get("retrieval")
        self.assertIsNotNone(retrieval)
        self.assertEqual(retrieval["search_query"],
                         "Как установить приложение?")
        self.assertFalse(retrieval["rewrite_used"])
        # scores = 1 - distance: 0.12 → 0.88
        self.assertAlmostEqual(retrieval["scores"][0], 0.88, places=4)

    def test_rewrite_mode_uses_rewriter(self) -> None:
        agent = make_agent(rag_enabled=True)
        rewriter = FakeRewriter()
        agent.rewriter = rewriter
        agent.pipeline.rewriter = rewriter
        agent.set_rag_mode(RAG_MODE_REWRITE)
        result = run(agent.send_message("Как установить?"))
        self.assertEqual(rewriter.calls, ["Как установить?"])
        self.assertTrue(result["retrieval"]["rewrite_used"])
        self.assertEqual(result["retrieval"]["search_query"],
                         "установка Windows инструкция")

    def test_rewrite_error_reported_not_fatal(self) -> None:
        agent = make_agent(rag_enabled=True)
        rewriter = FakeRewriter(fail=True)
        agent.rewriter = rewriter
        agent.pipeline.rewriter = rewriter
        agent.set_rag_mode(RAG_MODE_REWRITE)
        result = run(agent.send_message("Как установить?"))
        self.assertIsNone(result.get("error"))
        self.assertIn("DeepSeek недоступен",
                      result["retrieval"]["rewrite_error"])
        # Чанки найдены по исходному вопросу — RAG отработал
        self.assertTrue(result.get("rag_used"))

    def test_mode_without_rag_no_retrieval(self) -> None:
        agent = make_agent(rag_enabled=False)
        result = run(agent.send_message("Привет"))
        self.assertNotIn("retrieval", result)


class AnswerOnceModesTest(unittest.TestCase):
    """answer_once с режимом (вкладка «Сравнение RAG-режимов»)."""

    def test_answer_once_with_mode(self) -> None:
        store = FakeStore()
        agent = make_agent(rag_enabled=True, store=store)
        agent.bind_config(dict(COMPARISON_CONFIG))
        agent.set_top_k_search(15)
        result = run(agent.answer_once(
            "Вопрос", rag_enabled=True,
            rag_mode="rewrite_filter",
            retrieval_params={"top_k": 15, "top_k_final": 2,
                              "threshold": 0.5}))
        self.assertTrue(result["rag_used"])
        self.assertEqual(store.last_k, 15)
        self.assertEqual(result["retrieval"]["search_query"], "Вопрос")
        # distances 0.12/0.34 → scores 0.88/0.66, порог 0.5 не режет
        self.assertEqual(len(result["sources"]), 2)

    def test_answer_once_mode_defaults_to_current(self) -> None:
        agent = make_agent(rag_enabled=True)
        agent.bind_config(dict(COMPARISON_CONFIG))
        rewriter = FakeRewriter()
        agent.rewriter = rewriter
        agent.pipeline.rewriter = rewriter
        agent.set_rag_mode(RAG_MODE_REWRITE)
        result = run(agent.answer_once("Вопрос", rag_enabled=True))
        self.assertEqual(result["retrieval"]["search_query"],
                         "установка Windows инструкция")

    def test_answer_once_does_not_touch_main_history(self) -> None:
        agent = make_agent(rag_enabled=True)
        agent.bind_config(dict(COMPARISON_CONFIG))
        run(agent.send_message("Сообщение в чат"))
        before = len(agent.core._message_history)
        run(agent.answer_once("Вопрос", rag_enabled=True,
                              rag_mode="reranker"))
        self.assertEqual(len(agent.core._message_history), before)


class MockConfigIntegrationTest(unittest.TestCase):
    """with_config: rewriter настроен (mock-режим — без сети)."""

    def test_with_config_creates_rewriter(self) -> None:
        config = dict(COMPARISON_CONFIG)
        config["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com"
        agent = RAGChatAgent.with_config(config)
        self.assertIsNotNone(agent.rewriter)
        self.assertTrue(agent.rewriter.mock_mode)
        self.assertIsNotNone(agent.reranker)
        self.assertIsNotNone(agent.pipeline.rewriter)
        self.assertIsNotNone(agent.pipeline.reranker)


def run(coroutine):
    return asyncio.run(coroutine)


if __name__ == "__main__":
    unittest.main()
