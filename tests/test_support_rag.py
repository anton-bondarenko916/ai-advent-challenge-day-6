"""Тесты RAG по FAQ поддержки (ТЗ ai_1_34: коллекция support_faq).

Проверяются: сбор *.md рекурсивно, индексация в ОТДЕЛЬНУЮ коллекцию
Chroma (knowledge_base/project_docs не смешиваются), метаданные чанков
(type="support", path от корня папки), батч-эмбеддинги (один POST на
всё не масштабируется — урок ai_1_33-verify), релевантный поиск,
переиндексация с перезаписью, недоступный эмбеддер (коллекция не
очищается).

Запуск из корня проекта:
    python -m unittest tests.test_support_rag -v
"""

from __future__ import annotations

import re
import shutil
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

from app.rag.embedder import EmbedderError
from app.rag.project_rag import PROJECT_DOCS_COLLECTION
from app.rag.support_rag import (
    EMBED_BATCH_SIZE,
    SUPPORT_FAQ_COLLECTION,
    SupportRAG,
    default_support_docs_path,
)
from app.rag.vector_store import VectorStore

# Слова без пунктуации (для одинаковых токенов запроса и чанка)
_WORD_RE = re.compile(r"\w+")


class KeywordEmbedder:
    """Детерминированный «эмбеддер» (crc32 — стабилен между процессами)."""

    DIM = 256

    def _vector(self, text: str) -> list:
        vector = [0.0] * self.DIM
        for word in _WORD_RE.findall(str(text).lower()):
            vector[zlib.crc32(word.encode("utf-8")) % self.DIM] += 1.0
        return vector

    def embed_documents(self, texts) -> list:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list:
        return self._vector(text)

    async def embed_query_async(self, text: str) -> list:
        return self._vector(text)


class CountingEmbedder(KeywordEmbedder):
    """Запоминает размеры батчей embed_documents (проверка нарезки)."""

    def __init__(self) -> None:
        self.batch_sizes: list = []

    def embed_documents(self, texts) -> list:
        texts = list(texts)
        self.batch_sizes.append(len(texts))
        return super().embed_documents(texts)


class FailingEmbedder(KeywordEmbedder):
    """Эмбеддер, имитирующий недоступную Ollama."""

    def embed_documents(self, texts) -> list:
        raise EmbedderError("Ollama недоступна по адресу "
                            "http://localhost:11434")

    def embed_query(self, text: str) -> list:
        raise EmbedderError("Ollama недоступна по адресу "
                            "http://localhost:11434")


def make_docs(root: Path) -> None:
    """Папка поддержки: FAQ (в т.ч. вложенный) + не-md файл."""
    (root / "faq_authorization.md").write_text(
        "# FAQ: Авторизация\n\n## Не работает авторизация\n\n"
        "Вход блокируется для статуса pending_verification.\n",
        encoding="utf-8")
    sub = root / "extra"
    sub.mkdir(exist_ok=True)
    (sub / "faq_password.md").write_text(
        "# FAQ: Пароль\n\n## Сброс пароля\n\nСсылка действует 1 час.\n",
        encoding="utf-8")
    (root / "notes.txt").write_text("заметки мимо\n", encoding="utf-8")


class SupportRAGTest(unittest.TestCase):
    """SupportRAG: сбор, индексация, поиск — на настоящем Chroma."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.docs = self.tmp / "support_docs"
        self.docs.mkdir()
        make_docs(self.docs)
        self.store = VectorStore(persist_dir=self.tmp / "db")
        self.rag = SupportRAG(self.store, KeywordEmbedder())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ------------------------------------------------------------------
    # Сбор файлов
    # ------------------------------------------------------------------
    def test_collect_files_md_recursive(self) -> None:
        files = SupportRAG.collect_files(self.docs)
        names = sorted(path.name for path, _ in files)
        self.assertEqual(names, ["faq_authorization.md", "faq_password.md"])
        types = {file_type for _, file_type in files}
        self.assertEqual(types, {"support"})

    def test_collect_files_missing_root(self) -> None:
        self.assertEqual(SupportRAG.collect_files(self.tmp / "nope"), [])

    def test_default_docs_path(self) -> None:
        self.assertTrue(default_support_docs_path().endswith("support_docs"))

    # ------------------------------------------------------------------
    # Индексация (ТЗ: НЕ смешивать support_faq с другими коллекциями)
    # ------------------------------------------------------------------
    def test_index_support_docs_separate_collection(self) -> None:
        stats = self.rag.index_support_docs(self.docs)
        self.assertEqual(stats["files"], 2)
        self.assertEqual(stats["chunks"], self.rag.count())
        self.assertGreater(self.rag.count(), 0)
        # Личная база знаний и project_docs не затронуты
        self.assertEqual(
            self.store.count("knowledge_base"), 0)
        self.assertEqual(
            self.store.count(PROJECT_DOCS_COLLECTION), 0)

    def test_index_metadata_type_and_relative_path(self) -> None:
        self.rag.index_support_docs(self.docs)
        result = self.store.search(
            KeywordEmbedder().embed_query("авторизация"), k=5,
            collection_name=SUPPORT_FAQ_COLLECTION)
        metadatas = result["metadatas"][0]
        self.assertTrue(metadatas)
        for metadata in metadatas:
            self.assertEqual(metadata["type"], "support")
            self.assertFalse(Path(metadata["path"]).is_absolute())
        sources = {m["source"] for m in metadatas}
        self.assertIn("faq_authorization.md", sources)

    def test_reindex_overwrites(self) -> None:
        self.rag.index_support_docs(self.docs)
        first_count = self.rag.count()
        (self.docs / "faq_new.md").write_text(
            "# Новое\n\nЕщё один документ FAQ про API ключи.\n",
            encoding="utf-8")
        self.rag.index_support_docs(self.docs)
        self.assertGreater(self.rag.count(), first_count)

    def test_embedder_failure_keeps_old_collection(self) -> None:
        """Недоступная Ollama: EmbedderError, старые чанки рабочие."""
        self.rag.index_support_docs(self.docs)
        count_before = self.rag.count()
        failing = SupportRAG(self.store, FailingEmbedder())
        with self.assertRaises(EmbedderError):
            failing.index_support_docs(self.docs)
        self.assertEqual(self.rag.count(), count_before)

    def test_embeddings_are_batched(self) -> None:
        """Эмбеддинги уходят батчами по EMBED_BATCH_SIZE (не один POST)."""
        counting = CountingEmbedder()
        rag = SupportRAG(self.store, counting)
        with mock.patch("app.rag.support_rag.EMBED_BATCH_SIZE", 1):
            rag.index_support_docs(self.docs)
        self.assertGreater(len(counting.batch_sizes), 1)
        self.assertTrue(
            all(size <= EMBED_BATCH_SIZE for size in counting.batch_sizes))

    def test_empty_docs_dir(self) -> None:
        empty = self.tmp / "empty"
        empty.mkdir()
        stats = self.rag.index_support_docs(empty)
        self.assertEqual(stats["files"], 0)
        self.assertEqual(stats["chunks"], 0)

    # ------------------------------------------------------------------
    # Поиск
    # ------------------------------------------------------------------
    def test_search_finds_faq_chunk(self) -> None:
        """Сценарий ТЗ: вопрос про авторизацию находит чанк FAQ.

        Текст чанка — тело секции (заголовки уходят в метаданные),
        поэтому проверяется source топ-чанка.
        """
        self.rag.index_support_docs(self.docs)
        result = self.rag.search("Не работает авторизация что делать?",
                                 top_k=3)
        metadatas = result["metadatas"][0]
        self.assertTrue(metadatas)
        self.assertEqual(metadatas[0]["source"], "faq_authorization.md")
        self.assertIn("авторизац",
                      (metadatas[0].get("section") or "").lower())

    def test_search_chunks_async(self) -> None:
        import asyncio

        self.rag.index_support_docs(self.docs)
        chunks = asyncio.run(
            self.rag.search_chunks_async("сброс пароля", top_k=2))
        self.assertTrue(chunks)
        self.assertTrue(all(hasattr(chunk, "text") for chunk in chunks))
        self.assertTrue(any(
            "парол" in f"{chunk.section} {chunk.source}".lower()
            for chunk in chunks))

    def test_clear(self) -> None:
        self.rag.index_support_docs(self.docs)
        self.assertGreater(self.rag.count(), 0)
        self.rag.clear()
        self.assertEqual(self.rag.count(), 0)


if __name__ == "__main__":
    unittest.main()
