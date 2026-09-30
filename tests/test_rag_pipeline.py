"""Тесты RAGPipeline (app/rag/pipeline.py): 5 режимов, порог, rerank.

Все тесты офлайн: embedder/vector_store/rewriter/reranker — фейки.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import unittest

from app.rag.pipeline import (
    DEFAULT_THRESHOLD,
    RAG_MODES,
    RAG_MODE_BASIC,
    RAG_MODE_RERANKER,
    RAG_MODE_REWRITE,
    RAG_MODE_REWRITE_FILTER,
    RAG_MODE_REWRITE_RERANKER,
    RAGPipeline,
    filter_by_threshold,
    normalize_mode,
)
from app.rag.rewriter import RewriteError
from app.rag.reranker import RerankerError

# Chroma отдаёт расстояния (cosine distance = 1 - similarity)
STORE_RESULT = {
    "ids": [["chunk_1", "chunk_2", "chunk_3"]],
    "documents": [
        ["Установка на Windows.", "Неполадки Ollama.", "Лицензия MIT."]
    ],
    "metadatas": [[
        {"source": "installation.md", "section": "## Windows",
         "strategy": "structure", "chunk_id": 1},
        {"source": "troubleshooting.md", "section": "## Ollama",
         "strategy": "fixed", "chunk_id": 2},
        {"source": "license.md", "section": "", "strategy": "fixed",
         "chunk_id": 3},
    ]],
    # similarity: 0.9, 0.8, 0.5 → порог 0.7 отсекает третий
    "distances": [[0.1, 0.2, 0.5]],
}


class FakeEmbedder:
    """Отдаёт фиксированный вектор, запоминает запрос."""

    def __init__(self) -> None:
        self.queries: list = []

    def embed_query(self, text: str) -> list:
        self.queries.append(text)
        return [0.1, 0.2]

    async def embed_query_async(self, text: str) -> list:
        self.queries.append(text)
        return [0.1, 0.2]


class FakeStore:
    """Хранилище-заглушка с ответом collection.query."""

    def __init__(self, result=None) -> None:
        self.result = result if result is not None else STORE_RESULT
        self.last_k: int = -1

    def search(self, query_embedding, k=3, collection_name="kb"):
        self.last_k = k
        return self.result


class FakeRewriter:
    """Переформулировщик-заглушка (или падает)."""

    def __init__(self, rewritten="установка приложения Windows",
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


class FakeReranker:
    """Реранкер-заглушка: всегда ставит третий чанк первым."""

    def __init__(self, indices=None, fail=False) -> None:
        self.indices = indices if indices is not None else [2, 0, 1]
        self.fail = fail
        self.calls: list = []

    def rerank(self, query, chunks, top_k):
        self.calls.append((query, list(chunks), top_k))
        if self.fail:
            raise RerankerError("реранкер недоступен")
        return self.indices[:top_k]

    async def rerank_async(self, query, chunks, top_k):
        return self.rerank(query, chunks, top_k)


def make_pipeline(rewriter=None, reranker=None, store=None) -> RAGPipeline:
    return RAGPipeline(
        FakeEmbedder(), store or FakeStore(),
        rewriter=rewriter, reranker=reranker,
    )


class FilterByThresholdTest(unittest.TestCase):
    """Фильтр similarity (функция из ТЗ)."""

    def test_cuts_below_threshold(self) -> None:
        chunks, metas, scores = filter_by_threshold(
            ["a", "b", "c"], [{"i": 1}, {"i": 2}, {"i": 3}],
            [0.9, 0.7, 0.5], threshold=0.7)
        self.assertEqual(chunks, ["a", "b"])       # граница включается
        self.assertEqual(metas, [{"i": 1}, {"i": 2}])
        self.assertEqual(scores, [0.9, 0.7])

    def test_all_below_gives_empty(self) -> None:
        self.assertEqual(
            filter_by_threshold(["a"], [{}], [0.1], threshold=0.7),
            ([], [], []))

    def test_empty_input(self) -> None:
        self.assertEqual(filter_by_threshold([], [], [], 0.7),
                         ([], [], []))


class BasicModeTest(unittest.TestCase):
    """basic: без rewrite, без фильтра, усечение до top_k_final."""

    def test_no_rewrite_called(self) -> None:
        rewriter = FakeRewriter()
        pipeline = make_pipeline(rewriter=rewriter)
        result = pipeline.retrieve("Как установить?", mode=RAG_MODE_BASIC)
        self.assertEqual(rewriter.calls, [])
        self.assertEqual(result["search_query"], "Как установить?")
        self.assertFalse(result["rewrite_used"])

    def test_scores_converted_from_distance(self) -> None:
        # distance 0.1 → similarity 0.9 (1 - distance, требование ТЗ)
        result = make_pipeline().retrieve("вопрос", mode=RAG_MODE_BASIC)
        self.assertAlmostEqual(result["scores"][0], 0.9, places=6)
        self.assertAlmostEqual(result["scores"][1], 0.8, places=6)

    def test_search_uses_top_k(self) -> None:
        store = FakeStore()
        pipeline = make_pipeline(store=store)
        pipeline.retrieve("вопрос", mode=RAG_MODE_BASIC, top_k=20)
        self.assertEqual(store.last_k, 20)

    def test_truncates_to_top_k_final(self) -> None:
        result = make_pipeline().retrieve(
            "вопрос", mode=RAG_MODE_BASIC, top_k_final=2)
        self.assertEqual(len(result["chunks"]), 2)
        self.assertEqual(len(result["scores"]), 2)

    def test_chunks_keep_metadata(self) -> None:
        result = make_pipeline().retrieve("вопрос", mode=RAG_MODE_BASIC)
        self.assertEqual(result["chunks"][0].source, "installation.md")
        self.assertEqual(result["metadatas"][0]["source"],
                         "installation.md")
        self.assertEqual(result["metadatas"][0]["chunk_id"], 1)


class RewriteModeTest(unittest.TestCase):
    """rewrite: вопрос переформулируется перед поиском."""

    def test_search_query_is_rewritten(self) -> None:
        rewriter = FakeRewriter("установка приложения Windows инструкция")
        result = make_pipeline(rewriter=rewriter).retrieve(
            "Как поставить?", mode=RAG_MODE_REWRITE)
        self.assertEqual(result["search_query"],
                         "установка приложения Windows инструкция")
        self.assertTrue(result["rewrite_used"])

    def test_embedder_gets_rewritten_query(self) -> None:
        embedder = FakeEmbedder()
        pipeline = RAGPipeline(embedder, FakeStore(),
                               rewriter=FakeRewriter("переформулировка"))
        pipeline.retrieve("Как поставить?", mode=RAG_MODE_REWRITE)
        self.assertEqual(embedder.queries, ["переформулировка"])

    def test_rewrite_failure_degrades_to_question(self) -> None:
        rewriter = FakeRewriter(fail=True)
        result = make_pipeline(rewriter=rewriter).retrieve(
            "Исходный вопрос?", mode=RAG_MODE_REWRITE)
        self.assertEqual(result["search_query"], "Исходный вопрос?")
        self.assertFalse(result["rewrite_used"])
        self.assertIn("DeepSeek недоступен", result["rewrite_error"])
        # Чанки всё равно найдены (режим деградировал, не упал)
        self.assertEqual(len(result["chunks"]), 3)


class RewriteFilterModeTest(unittest.TestCase):
    """rewrite_filter: rewrite + отсечение по порогу."""

    def test_threshold_cuts_weak_chunks(self) -> None:
        rewriter = FakeRewriter("поиск")
        result = make_pipeline(rewriter=rewriter).retrieve(
            "вопрос", mode=RAG_MODE_REWRITE_FILTER,
            top_k=3, threshold=0.7)
        # similarity 0.9/0.8/0.5 → остаются первые два
        self.assertEqual(len(result["chunks"]), 2)
        self.assertTrue(all(score >= DEFAULT_THRESHOLD
                            for score in result["scores"]))

    def test_custom_threshold(self) -> None:
        result = make_pipeline().retrieve(
            "вопрос", mode=RAG_MODE_REWRITE_FILTER,
            top_k=3, threshold=0.85)
        self.assertEqual(len(result["chunks"]), 1)

    def test_all_cut_gives_empty(self) -> None:
        result = make_pipeline().retrieve(
            "вопрос", mode=RAG_MODE_REWRITE_FILTER,
            top_k=3, threshold=0.99)
        self.assertEqual(result["chunks"], [])
        self.assertEqual(result["scores"], [])


class RerankerModeTest(unittest.TestCase):
    """reranker / rewrite_reranker: переранжирование кандидатов."""

    def test_rerank_reorders_chunks(self) -> None:
        reranker = FakeReranker(indices=[2, 0, 1])
        result = make_pipeline(reranker=reranker).retrieve(
            "вопрос", mode=RAG_MODE_RERANKER,
            top_k=3, top_k_final=3)
        self.assertEqual(result["chunks"][0].text, "Лицензия MIT.")
        self.assertEqual(result["rerank_backend"], "FakeReranker")

    def test_rerank_gets_question_and_top_k_final(self) -> None:
        reranker = FakeReranker()
        make_pipeline(reranker=reranker).retrieve(
            "Как установить?", mode=RAG_MODE_RERANKER,
            top_k=20, top_k_final=2)
        query, chunks, top_k = reranker.calls[0]
        self.assertEqual(query, "Как установить?")
        self.assertEqual(len(chunks), 3)   # кандидатов — top_k из поиска
        self.assertEqual(top_k, 2)

    def test_rewrite_reranker_combines_both(self) -> None:
        rewriter = FakeRewriter("переформулировка")
        reranker = FakeReranker(indices=[1, 0])
        result = make_pipeline(rewriter=rewriter, reranker=reranker).retrieve(
            "Как поставить?", mode=RAG_MODE_REWRITE_RERANKER,
            top_k=3, top_k_final=2)
        # Поиск по переформулировке…
        self.assertEqual(result["search_query"], "переформулировка")
        self.assertTrue(result["rewrite_used"])
        # …реранкер ставит вторую кандидату первой, top_k_final=2
        self.assertEqual(result["chunks"][0].text, "Неполадки Ollama.")
        self.assertEqual(len(result["chunks"]), 2)
        self.assertEqual(result["rerank_backend"], "FakeReranker")

    def test_rerank_failure_degrades_to_embedding_order(self) -> None:
        reranker = FakeReranker(fail=True)
        result = make_pipeline(reranker=reranker).retrieve(
            "вопрос", mode=RAG_MODE_RERANKER, top_k=3, top_k_final=2)
        self.assertEqual(len(result["chunks"]), 2)
        self.assertEqual(result["chunks"][0].text, "Установка на Windows.")
        self.assertIsNone(result["rerank_backend"])
        self.assertIn("недоступен", result["rerank_error"])

    def test_no_reranker_configured_degrades(self) -> None:
        result = make_pipeline().retrieve(
            "вопрос", mode=RAG_MODE_RERANKER, top_k=3, top_k_final=2)
        self.assertEqual(len(result["chunks"]), 2)
        self.assertIn("не настроен", result["rerank_error"])


class AllModesTest(unittest.TestCase):
    """Каждый из 5 режимов возвращает корректный результат."""

    def test_each_mode_returns_result(self) -> None:
        pipeline = make_pipeline(rewriter=FakeRewriter(),
                                 reranker=FakeReranker())
        for mode in RAG_MODES:
            result = pipeline.retrieve("вопрос", mode=mode,
                                       top_k=3, top_k_final=2)
            self.assertEqual(result["mode"], mode, mode)
            self.assertIn("chunks", result)
            self.assertIn("metadatas", result)
            self.assertIn("scores", result)
            self.assertLessEqual(len(result["chunks"]), 2, mode)

    def test_each_mode_async(self) -> None:
        pipeline = make_pipeline(rewriter=FakeRewriter(),
                                 reranker=FakeReranker())

        async def run() -> None:
            for mode in RAG_MODES:
                result = await pipeline.retrieve_async(
                    "вопрос", mode=mode, top_k=3, top_k_final=2)
                self.assertEqual(result["mode"], mode, mode)
                self.assertLessEqual(len(result["chunks"]), 2, mode)

        asyncio.run(run())

    def test_empty_base_gives_empty(self) -> None:
        empty = {"ids": [[]], "documents": [[]], "metadatas": [[]],
                 "distances": [[]]}
        pipeline = make_pipeline(store=FakeStore(empty))
        for mode in RAG_MODES:
            result = pipeline.retrieve("вопрос", mode=mode)
            self.assertEqual(result["chunks"], [], mode)


class NormalizeModeTest(unittest.TestCase):
    """Валидация имени режима."""

    def test_valid_modes_pass(self) -> None:
        for mode in RAG_MODES:
            self.assertEqual(normalize_mode(mode), mode)

    def test_unknown_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_mode("hybrid")
        with self.assertRaises(ValueError):
            normalize_mode("")


if __name__ == "__main__":
    unittest.main()
