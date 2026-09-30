"""Тесты реранкеров (app/rag/reranker.py): Ollama, FlagEmbedding, фолбэк.

Все тесты офлайн: HTTP Ollama подменяется httpx.MockTransport,
FlagEmbedding — фейковым модулем в sys.modules.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
import unittest

import httpx

from app.rag.reranker import (
    BaseReranker,
    FallbackReranker,
    FlagEmbeddingReranker,
    LexicalReranker,
    OllamaReranker,
    RerankerError,
)

CHUNKS = [
    "Установка приложения на Windows: скачайте Python.",
    "Неполадки Ollama: запустите ollama serve.",
    "Лицензия проекта: MIT.",
]


def ollama_transport(scores):
    """MockTransport /api/generate: скор = scores[индекс пары]."""

    def handler(request: httpx.Request) -> httpx.Response:
        # Тело — JSON (кириллица закодирована \uXXXX): разбираем честно
        payload = json.loads(request.read().decode("utf-8"))
        prompt = str(payload.get("prompt") or "")
        index = next(i for i, chunk in enumerate(CHUNKS)
                     if chunk in prompt)
        return httpx.Response(200, json={
            "model": "xitao/bge-reranker-v2-m3",
            "response": str(scores[index]),
            "done": True,
        })

    return httpx.MockTransport(handler)


def failing_transport(status=404, body="model not found"):
    """MockTransport, всегда отвечающий ошибкой."""
    return httpx.MockTransport(
        lambda request: httpx.Response(status, text=body))


class OllamaRerankerTest(unittest.TestCase):
    """Скоринг пар через HTTP Ollama."""

    def test_rerank_orders_by_score_desc(self) -> None:
        # Лучший — второй чанк (про Ollama), худший — лицензия
        reranker = OllamaReranker(transport=ollama_transport([0.3, 0.9, 0.1]))
        indices = reranker.rerank("Как запустить ollama?", CHUNKS, 2)
        self.assertEqual(indices, [1, 0])

    def test_top_k_limits_result(self) -> None:
        reranker = OllamaReranker(transport=ollama_transport([0.3, 0.9, 0.1]))
        self.assertEqual(len(reranker.rerank("вопрос", CHUNKS, 1)), 1)

    def test_top_k_larger_than_chunks(self) -> None:
        reranker = OllamaReranker(transport=ollama_transport([0.3, 0.9, 0.1]))
        self.assertEqual(len(reranker.rerank("вопрос", CHUNKS, 10)), 3)

    def test_empty_chunks_gives_empty(self) -> None:
        reranker = OllamaReranker(transport=failing_transport())
        self.assertEqual(reranker.rerank("вопрос", [], 3), [])

    def test_http_error_raises_reranker_error(self) -> None:
        reranker = OllamaReranker(transport=failing_transport())
        with self.assertRaises(RerankerError):
            reranker.rerank("вопрос", CHUNKS, 2)

    def test_unparseable_score_raises(self) -> None:
        reranker = OllamaReranker(transport=failing_transport(
            200, "поток мысли без числа"))
        with self.assertRaises(RerankerError):
            reranker.rerank("вопрос", CHUNKS, 2)

    def test_rerank_async(self) -> None:
        reranker = OllamaReranker(transport=ollama_transport([0.3, 0.9, 0.1]))

        async def run() -> None:
            indices = await reranker.rerank_async("ollama", CHUNKS, 2)
            self.assertEqual(indices, [1, 0])

        asyncio.run(run())

    def test_rerank_async_error(self) -> None:
        reranker = OllamaReranker(transport=failing_transport())

        async def run() -> None:
            await reranker.rerank_async("ollama", CHUNKS, 2)

        with self.assertRaises(RerankerError):
            asyncio.run(run())


class FlagEmbeddingRerankerTest(unittest.TestCase):
    """Fallback через пакет FlagEmbedding (фейковый модуль)."""

    def setUp(self) -> None:
        # Фейковый FlagEmbedding.FlagReranker: скор = 1/(индекс+1)
        fake_flag = types.ModuleType("FlagEmbedding")

        class FakeFlagReranker:
            def __init__(self, model_name, use_fp16=False):
                self.model_name = model_name

            def compute_score(self, pairs):
                return [1.0 / (i + 1) for i in range(len(pairs))]

        fake_flag.FlagReranker = FakeFlagReranker
        self._saved = {name: sys.modules.get(name)
                       for name in ("FlagEmbedding",)}
        sys.modules["FlagEmbedding"] = fake_flag

    def tearDown(self) -> None:
        for name, module in self._saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_uses_fake_backend(self) -> None:
        reranker = FlagEmbeddingReranker()
        indices = reranker.rerank("вопрос", CHUNKS, 2)
        # Скор убывает с индексом → топ-2: [0, 1]
        self.assertEqual(indices, [0, 1])

    def test_empty_chunks_gives_empty(self) -> None:
        self.assertEqual(FlagEmbeddingReranker().rerank("в", [], 3), [])


class FlagEmbeddingMissingTest(unittest.TestCase):
    """Пакет FlagEmbedding не установлен → RerankerError (не ImportError)."""

    def setUp(self) -> None:
        self._saved = sys.modules.get("FlagEmbedding")
        sys.modules["FlagEmbedding"] = None  # import → ImportError

    def tearDown(self) -> None:
        if self._saved is None:
            sys.modules.pop("FlagEmbedding", None)
        else:
            sys.modules["FlagEmbedding"] = self._saved

    def test_missing_package_raises_reranker_error(self) -> None:
        with self.assertRaises(RerankerError):
            FlagEmbeddingReranker().rerank("вопрос", CHUNKS, 2)


class LexicalRerankerTest(unittest.TestCase):
    """Запасной реранкер: пересечение токенов."""

    def test_relevant_chunk_first(self) -> None:
        reranker = LexicalReranker()
        indices = reranker.rerank(
            "Как установить приложение на Windows?", CHUNKS, 3)
        self.assertEqual(indices[0], 0)

    def test_top_k(self) -> None:
        reranker = LexicalReranker()
        self.assertEqual(len(reranker.rerank("вопрос", CHUNKS, 1)), 1)

    def test_empty_query_gives_stable_order(self) -> None:
        reranker = LexicalReranker()
        self.assertEqual(reranker.rerank("", CHUNKS, 3), [0, 1, 2])

    def test_empty_chunks(self) -> None:
        self.assertEqual(LexicalReranker().rerank("вопрос", [], 3), [])


class FallbackRerankerTest(unittest.TestCase):
    """Цепочка: первый успешный бэкенд «залипает»."""

    def test_falls_back_to_second_backend(self) -> None:
        first = BrokenReranker()
        second = StubReranker([2, 1, 0])
        fallback = FallbackReranker([first, second])
        self.assertEqual(fallback.rerank("вопрос", CHUNKS, 2), [2, 1])
        self.assertEqual(fallback.last_backend, "StubReranker")
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 1)

    def test_sticky_backend_not_retried(self) -> None:
        first = BrokenReranker()
        second = StubReranker([0, 1, 2])
        fallback = FallbackReranker([first, second])
        fallback.rerank("вопрос", CHUNKS, 2)
        fallback.rerank("вопрос", CHUNKS, 2)
        self.assertEqual(first.calls, 1)  # упавший не дёргается повторно
        self.assertEqual(second.calls, 2)

    def test_first_working_backend_wins(self) -> None:
        fallback = FallbackReranker([
            StubReranker([0, 1, 2]), StubReranker([2, 1, 0])])
        self.assertEqual(fallback.rerank("вопрос", CHUNKS, 2), [0, 1])

    def test_all_fail_raises(self) -> None:
        fallback = FallbackReranker([BrokenReranker(), BrokenReranker()])
        with self.assertRaises(RerankerError):
            fallback.rerank("вопрос", CHUNKS, 2)

    def test_empty_chunks_no_backend_call(self) -> None:
        stub = StubReranker([])
        fallback = FallbackReranker([stub])
        self.assertEqual(fallback.rerank("вопрос", [], 3), [])
        self.assertEqual(stub.calls, 0)

    def test_empty_backends_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FallbackReranker([])

    def test_fallback_async(self) -> None:
        fallback = FallbackReranker([BrokenReranker(), StubReranker([1])])

        async def run() -> None:
            self.assertEqual(await fallback.rerank_async("в", CHUNKS, 1), [1])

        asyncio.run(run())

    def test_default_chain_order(self) -> None:
        from app.rag.reranker import create_default_reranker
        fallback = create_default_reranker()
        names = [type(b).__name__ for b in fallback.backends]
        self.assertEqual(names, ["OllamaReranker", "FlagEmbeddingReranker",
                                 "LexicalReranker"])


class BaseRerankerTest(unittest.TestCase):
    """Абстрактный интерфейс."""

    def test_cannot_instantiate_abstract(self) -> None:
        with self.assertRaises(TypeError):
            BaseReranker()  # noqa

    def test_async_default_delegates_to_sync(self) -> None:
        stub = StubReranker([0])

        async def run() -> None:
            self.assertEqual(await stub.rerank_async("в", CHUNKS, 1), [0])

        asyncio.run(run())


class StubReranker(BaseReranker):
    """Всегда возвращает заготовленные индексы (счётчик вызовов)."""

    def __init__(self, indices) -> None:
        self.indices = list(indices)
        self.calls = 0

    def rerank(self, query, chunks, top_k):
        self.calls += 1
        return self.indices[:top_k]


class BrokenReranker(BaseReranker):
    """Всегда падает RerankerError (имитация недоступного Ollama)."""

    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query, chunks, top_k):
        self.calls += 1
        raise RerankerError("Ollama reranker недоступен")


if __name__ == "__main__":
    unittest.main()
