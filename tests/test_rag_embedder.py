"""Тесты эмбеддера Ollama (app/rag/embedder.py) — офлайн, httpx.MockTransport.

Проверяются: префиксы search_document:/search_query:, URL /api/embed,
батч-режим, разбор ошибок, async-варианты, is_available.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import unittest

import httpx

from app.rag.embedder import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DOCUMENT_PREFIX,
    QUERY_PREFIX,
    EmbedderError,
    OllamaEmbedder,
)


def ok_transport(captured: list, embeddings=None, status: int = 200):
    """MockTransport: запоминает запросы, возвращает заглушку ответа."""

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if status != 200:
            return httpx.Response(status, text="boom")
        return httpx.Response(200, json={"embeddings": embeddings or []})

    return httpx.MockTransport(handler)


def failing_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    return httpx.MockTransport(handler)


class SyncEmbedderTest(unittest.TestCase):
    """Sync-методы: префиксы, URL, разбор ответа."""

    def test_embed_query_uses_search_query_prefix(self) -> None:
        captured: list = []
        embedder = OllamaEmbedder(
            transport=ok_transport(captured, embeddings=[[0.5, 0.6]]))
        vector = embedder.embed_query("Как установить приложение?")
        self.assertEqual(vector, [0.5, 0.6])
        body = json.loads(captured[0].read())
        # /api/embed принимает input списком (батч)
        self.assertEqual(body["input"],
                         [QUERY_PREFIX + "Как установить приложение?"])

    def test_embed_document_uses_search_document_prefix(self) -> None:
        captured: list = []
        embedder = OllamaEmbedder(
            transport=ok_transport(captured, embeddings=[[0.1]]))
        embedder.embed_document("Текст чанка")
        body = json.loads(captured[0].read())
        self.assertEqual(body["input"], [DOCUMENT_PREFIX + "Текст чанка"])

    def test_request_url_and_model(self) -> None:
        captured: list = []
        embedder = OllamaEmbedder(
            base_url="http://127.0.0.1:11500",
            transport=ok_transport(captured, embeddings=[[0.1]]))
        embedder.embed_query("вопрос")
        self.assertTrue(str(captured[0].url).endswith("/api/embed"))
        body = json.loads(captured[0].read())
        self.assertEqual(body["model"], DEFAULT_MODEL)

    def test_embed_documents_batch(self) -> None:
        captured: list = []
        embedder = OllamaEmbedder(
            transport=ok_transport(captured,
                                   embeddings=[[0.1], [0.2], [0.3]]))
        vectors = embedder.embed_documents(["а", "б", "в"])
        self.assertEqual(len(vectors), 3)
        body = json.loads(captured[0].read())
        self.assertIsInstance(body["input"], list)
        self.assertEqual(len(body["input"]), 3)
        # Каждый элемент батча — с префиксом документа
        self.assertTrue(body["input"][0].startswith(DOCUMENT_PREFIX))

    def test_http_error_raises_embedder_error(self) -> None:
        embedder = OllamaEmbedder(transport=ok_transport([], status=500))
        with self.assertRaises(EmbedderError) as ctx:
            embedder.embed_query("вопрос")
        self.assertIn("500", str(ctx.exception))

    def test_empty_embeddings_raises(self) -> None:
        embedder = OllamaEmbedder(transport=ok_transport([], embeddings=[]))
        with self.assertRaises(EmbedderError):
            embedder.embed_query("вопрос")

    def test_connection_error_raises_with_hint(self) -> None:
        embedder = OllamaEmbedder(transport=failing_transport())
        with self.assertRaises(EmbedderError) as ctx:
            embedder.embed_query("вопрос")
        self.assertIn("ollama serve", str(ctx.exception))

    def test_is_available_true(self) -> None:
        captured: list = []
        embedder = OllamaEmbedder(
            transport=ok_transport(captured, embeddings=[]))
        # /api/tags вернёт {"embeddings": []} — для is_available важно 200
        self.assertTrue(embedder.is_available())

    def test_is_available_false_on_connection_error(self) -> None:
        embedder = OllamaEmbedder(transport=failing_transport())
        self.assertFalse(embedder.is_available())


class AsyncEmbedderTest(unittest.TestCase):
    """Async-варианты (используются async-обработчиками Flet)."""

    def test_embed_query_async(self) -> None:
        captured: list = []
        transport = ok_transport(captured, embeddings=[[0.7, 0.8]])
        embedder = OllamaEmbedder(async_transport=transport)

        async def run() -> None:
            vector = await embedder.embed_query_async("вопрос")
            self.assertEqual(vector, [0.7, 0.8])

        self._run(run())
        body = json.loads(captured[0].read())
        self.assertEqual(body["input"], [QUERY_PREFIX + "вопрос"])

    def test_embed_documents_async_batch(self) -> None:
        captured: list = []
        embedder = OllamaEmbedder(
            async_transport=ok_transport(captured, embeddings=[[0.1], [0.2]]))

        async def run() -> None:
            vectors = await embedder.embed_documents_async(["а", "б"])
            self.assertEqual(len(vectors), 2)

        self._run(run())
        body = json.loads(captured[0].read())
        self.assertEqual(len(body["input"]), 2)

    def test_async_connection_error(self) -> None:
        embedder = OllamaEmbedder(async_transport=failing_transport())

        async def run() -> None:
            with self.assertRaises(EmbedderError):
                await embedder.embed_query_async("вопрос")

        self._run(run())

    def test_is_available_async(self) -> None:
        embedder = OllamaEmbedder(
            async_transport=ok_transport([], embeddings=[]))

        async def run() -> None:
            self.assertTrue(await embedder.is_available_async())

        self._run(run())

    @staticmethod
    def _run(coroutine) -> None:
        import asyncio
        asyncio.run(coroutine)


class DefaultsTest(unittest.TestCase):
    """Значения по умолчанию (локальная Ollama, nomic-embed-text)."""

    def test_defaults(self) -> None:
        embedder = OllamaEmbedder()
        self.assertEqual(embedder.model, "nomic-embed-text")
        self.assertEqual(embedder.base_url, DEFAULT_BASE_URL)
        self.assertTrue(DEFAULT_BASE_URL.startswith("http://localhost"))

    def test_prefix_constants(self) -> None:
        self.assertEqual(DOCUMENT_PREFIX, "search_document: ")
        self.assertEqual(QUERY_PREFIX, "search_query: ")


if __name__ == "__main__":
    unittest.main()
