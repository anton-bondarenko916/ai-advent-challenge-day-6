"""Тесты retriever (app/rag/retriever.py): поиск, промпт, источники.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import unittest

from app.rag.embedder import EmbedderError
from app.rag.retriever import (
    RAG_INSTRUCTION,
    RetrievedChunk,
    Retriever,
)
from app.rag.vector_store import VectorStore


class FakeEmbedder:
    """Запоминает запросы, отдаёт фиксированный вектор."""

    def __init__(self) -> None:
        self.queries: list = []

    def embed_query(self, text: str) -> list:
        self.queries.append(text)
        return [0.1, 0.2, 0.3]

    async def embed_query_async(self, text: str) -> list:
        self.queries.append(text)
        return [0.1, 0.2, 0.3]


class FakeStore:
    """Отдаёт заготовленный ответ collection.query."""

    def __init__(self, result=None) -> None:
        self.result = result or {"ids": [[]], "documents": [[]],
                                 "metadatas": [[]], "distances": [[]]}
        self.last_k: int = -1
        self.last_embedding: list = []

    def search(self, query_embedding, k=3, collection_name="knowledge_base"):
        self.last_k = k
        self.last_embedding = query_embedding
        return self.result


RESULT_TWO_CHUNKS = {
    "ids": [["chunk_1", "chunk_2"]],
    "documents": [["Чанк про установку Windows.", "Чанк про Ollama."]],
    "metadatas": [[
        {"source": "installation.md", "title": "Установка",
         "section": "## Windows", "strategy": "structure",
         "chunk_id": 1, "chunk_size": 500, "overlap": 50},
        {"source": "troubleshooting.md", "title": "Неполадки",
         "section": "## Ollama", "strategy": "fixed",
         "chunk_id": 2, "chunk_size": 500, "overlap": 50},
    ]],
    "distances": [[0.1234, 0.5678]],
}


class RetrieveTest(unittest.TestCase):
    """Поиск топ-K."""

    def test_retrieve_maps_metadata(self) -> None:
        retriever = Retriever(FakeEmbedder(), FakeStore(RESULT_TWO_CHUNKS))
        chunks = retriever.retrieve("Как установить?")
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].source, "installation.md")
        self.assertEqual(chunks[0].section, "## Windows")
        self.assertEqual(chunks[0].strategy, "structure")
        self.assertEqual(chunks[0].chunk_id, 1)
        self.assertAlmostEqual(chunks[0].distance, 0.1234, places=4)

    def test_top_k_passed_to_store(self) -> None:
        store = FakeStore(RESULT_TWO_CHUNKS)
        retriever = Retriever(FakeEmbedder(), store, top_k=5)
        retriever.retrieve("вопрос")
        self.assertEqual(store.last_k, 5)

    def test_top_k_minimum_one(self) -> None:
        store = FakeStore(RESULT_TWO_CHUNKS)
        Retriever(FakeEmbedder(), store, top_k=0).retrieve("вопрос")
        self.assertEqual(store.last_k, 1)

    def test_query_embedding_uses_embedder(self) -> None:
        embedder = FakeEmbedder()
        store = FakeStore(RESULT_TWO_CHUNKS)
        Retriever(embedder, store).retrieve("Как установить приложение?")
        self.assertEqual(embedder.queries, ["Как установить приложение?"])
        self.assertEqual(store.last_embedding, [0.1, 0.2, 0.3])

    def test_retrieve_async(self) -> None:
        retriever = Retriever(FakeEmbedder(), FakeStore(RESULT_TWO_CHUNKS))

        async def run() -> None:
            chunks = await retriever.retrieve_async("вопрос")
            self.assertEqual(len(chunks), 2)

        asyncio.run(run())

    def test_empty_base_gives_no_chunks(self) -> None:
        retriever = Retriever(FakeEmbedder(), FakeStore())
        self.assertEqual(retriever.retrieve("вопрос"), [])

    def test_embedder_error_propagates(self) -> None:
        class Boom:
            def embed_query(self, text):
                raise EmbedderError("Ollama недоступна")

        retriever = Retriever(Boom(), FakeStore(RESULT_TWO_CHUNKS))
        with self.assertRaises(EmbedderError):
            retriever.retrieve("вопрос")


class RetrievedChunkPreviewTest(unittest.TestCase):
    """Фрагмент текста для панели источников."""

    def test_short_text_unchanged(self) -> None:
        chunk = RetrievedChunk(text="Короткий текст")
        self.assertEqual(chunk.preview, "Короткий текст")

    def test_long_text_truncated_with_ellipsis(self) -> None:
        chunk = RetrievedChunk(text="длинно" * 100)
        self.assertLessEqual(len(chunk.preview), 160)
        self.assertTrue(chunk.preview.endswith("…"))

    def test_newlines_collapsed(self) -> None:
        chunk = RetrievedChunk(text="строка 1\n\nстрока 2")
        self.assertEqual(chunk.preview, "строка 1 строка 2")


class BuildRagPromptTest(unittest.TestCase):
    """Промпт с контекстом (формат ТЗ)."""

    def setUp(self) -> None:
        self.chunks = [
            RetrievedChunk(text="Первый чанк."),
            RetrievedChunk(text="Второй чанк."),
        ]

    def test_prompt_contains_context_and_question(self) -> None:
        prompt = Retriever.build_rag_prompt("Как установить?", self.chunks)
        self.assertIn("Контекст:", prompt)
        self.assertIn("Первый чанк.", prompt)
        self.assertIn("Второй чанк.", prompt)
        self.assertIn("Вопрос: Как установить?", prompt)

    def test_prompt_contains_instruction(self) -> None:
        prompt = Retriever.build_rag_prompt("вопрос", self.chunks)
        self.assertIn(RAG_INSTRUCTION, prompt)

    def test_chunks_joined_with_blank_line(self) -> None:
        prompt = Retriever.build_rag_prompt("вопрос", self.chunks)
        self.assertIn("Первый чанк.\n\nВторой чанк.", prompt)

    def test_empty_chunks_give_empty_prompt(self) -> None:
        self.assertEqual(Retriever.build_rag_prompt("вопрос", []), "")


class FormatSourcesTest(unittest.TestCase):
    """Источники для панели под ответом."""

    def test_fields(self) -> None:
        chunks = [RetrievedChunk(
            text="Текст чанка про установку приложения на Windows.",
            source="installation.md", title="Установка",
            section="## Windows", strategy="structure",
            distance=0.12345, chunk_id=3)]
        sources = Retriever.format_sources(chunks)
        self.assertEqual(len(sources), 1)
        source = sources[0]
        self.assertEqual(source["source"], "installation.md")
        self.assertEqual(source["section"], "## Windows")
        self.assertEqual(source["strategy"], "structure")
        self.assertEqual(source["distance"], 0.1235)
        self.assertEqual(source["chunk_id"], 3)
        self.assertIn("установку", source["preview"])

    def test_empty_gives_empty(self) -> None:
        self.assertEqual(Retriever.format_sources([]), [])


if __name__ == "__main__":
    unittest.main()
