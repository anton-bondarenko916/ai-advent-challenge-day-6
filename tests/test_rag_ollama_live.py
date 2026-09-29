"""Живой тест эмбеддера Ollama (критерий приёмки «проверить ollama serve»).

Запускается только если Ollama отвечает на http://localhost:11434
(таймаут 1 с); иначе тест пропускается — офлайн-набор не зависит от
внешних сервисов. Переопределить адрес: переменная окружения
OLLAMA_BASE_URL.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import unittest

import httpx

from app.rag.embedder import OllamaEmbedder

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def _ollama_alive() -> bool:
    try:
        response = httpx.get(f"{BASE_URL}/api/tags", timeout=1.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


@unittest.skipUnless(_ollama_alive(), f"Ollama не запущена на {BASE_URL}")
class OllamaLiveTest(unittest.TestCase):
    """Реальный вызов /api/embed локальной Ollama."""

    def setUp(self) -> None:
        self.embedder = OllamaEmbedder(model=MODEL, base_url=BASE_URL)

    def test_embed_query_returns_vector(self) -> None:
        vector = self.embedder.embed_query("Как установить приложение?")
        self.assertIsInstance(vector, list)
        self.assertGreater(len(vector), 10)
        self.assertTrue(all(isinstance(x, float) for x in vector[:5]))

    def test_document_and_query_prefixes_consistent(self) -> None:
        doc_vector = self.embedder.embed_document("Установка на Windows.")
        query_vector = self.embedder.embed_query("Установка на Windows.")
        self.assertEqual(len(doc_vector), len(query_vector))

    def test_similarity_ordering(self) -> None:
        """Похожий вопрос ближе к чанку, чем непохожий (cosine)."""
        import math

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm = math.sqrt(sum(x * x for x in a)) * \
                math.sqrt(sum(x * x for x in b))
            return dot / norm if norm else 0.0

        chunk = self.embedder.embed_document(
            "Для Windows требуется .NET 8 и установщик Python 3.11.")
        close = self.embedder.embed_query(
            "Что требуется для установки на Windows?")
        far = self.embedder.embed_query("Рецепт борща с фасолью")
        self.assertGreater(cosine(chunk, close), cosine(chunk, far))


if __name__ == "__main__":
    unittest.main()
