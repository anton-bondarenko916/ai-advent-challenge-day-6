"""Тесты индексатора (app/rag/indexer.py) — реальный Chroma в tmp.

Проверяются: сбор файлов, чанкинг с метаданными, commit с батч-
эмбеддингами, переиндексация, «Обе» стратегии, пошаговое API,
обработка ошибок.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from app.rag.embedder import EmbedderError
from app.rag.indexer import (
    BOTH_STRATEGIES,
    IndexStats,
    KnowledgeIndexer,
)
from app.rag.vector_store import VectorStore

DOC_1 = """# Установка приложения

Общий текст про установку и требования.

## Windows

Скачайте установщик Python и запустите его. Отметьте Add to PATH.

## Linux

Используйте apt install python3.
"""

DOC_2 = """# Частые вопросы

## Ключи

Ключ DeepSeek API хранится в файле .env.
"""


class FakeEmbedder:
    """Детерминированный эмбеддинг по хэшу текста (офлайн)."""

    def __init__(self) -> None:
        self.batches: list = []
        self.fail = False

    def embed_documents(self, texts):
        self.batches.append(list(texts))
        if self.fail:
            raise EmbedderError("Ollama недоступна (тест)")
        vectors = []
        for text in texts:
            seed = sum(ord(ch) % 97 for ch in text) % 50 / 50
            vectors.append([seed, seed / 2 + 0.1, 1.0])
        return vectors


class IndexerBase(unittest.TestCase):
    """Fixture: временные папки документов и Chroma."""

    def setUp(self) -> None:
        self.docs_dir = Path(tempfile.mkdtemp(prefix="docs_test_"))
        self.db_dir = Path(tempfile.mkdtemp(prefix="chroma_test_"))
        (self.docs_dir / "installation.md").write_text(DOC_1, encoding="utf-8")
        (self.docs_dir / "faq.md").write_text(DOC_2, encoding="utf-8")
        (self.docs_dir / "notes.txt").write_text("не индексируется",
                                                 encoding="utf-8")
        self.embedder = FakeEmbedder()
        self.store = VectorStore(persist_dir=self.db_dir)
        self.indexer = KnowledgeIndexer(self.embedder, self.store)

    def tearDown(self) -> None:
        shutil.rmtree(self.docs_dir, ignore_errors=True)
        shutil.rmtree(self.db_dir, ignore_errors=True)


class CollectFilesTest(IndexerBase):
    """Сбор MD-файлов."""

    def test_only_md_files(self) -> None:
        files = self.indexer.collect_files(self.docs_dir)
        names = [f.name for f in files]
        self.assertEqual(names, ["faq.md", "installation.md"])

    def test_missing_folder_gives_empty(self) -> None:
        self.assertEqual(self.indexer.collect_files(self.docs_dir / "nope"), [])

    def test_markdown_extension(self) -> None:
        (self.docs_dir / "guide.markdown").write_text("# Гайд", encoding="utf-8")
        names = [f.name for f in self.indexer.collect_files(self.docs_dir)]
        self.assertIn("guide.markdown", names)


class ExtractTitleTest(IndexerBase):
    """Заголовок документа."""

    def test_first_h1_line(self) -> None:
        self.assertEqual(
            self.indexer.extract_title(DOC_1, fallback="x"),
            "Установка приложения",
        )

    def test_fallback_when_no_h1(self) -> None:
        self.assertEqual(self.indexer.extract_title("Просто текст", "file"),
                         "file")


class IndexDirectoryTest(IndexerBase):
    """Полный путь index_directory."""

    def test_stats_basic(self) -> None:
        stats = self.indexer.index_directory(
            self.docs_dir, strategies=("fixed",), chunk_size=200, overlap=30)
        self.assertIsInstance(stats, IndexStats)
        self.assertEqual(stats.files, 2)
        self.assertGreater(stats.chunks, 0)
        self.assertEqual(stats.strategies, ["fixed"])
        self.assertEqual(stats.chunk_size, 200)
        self.assertEqual(stats.overlap, 30)
        self.assertEqual(stats.errors, [])
        self.assertEqual(self.store.count(), stats.chunks)

    def test_metadata_preserved_in_store(self) -> None:
        self.indexer.index_directory(
            self.docs_dir, strategies=("structure",), chunk_size=150, overlap=20)
        result = self.store.search([1.0, 1.0, 1.0], k=50)
        metadatas = result["metadatas"][0]
        for metadata in metadatas:
            self.assertIn(metadata["source"], ("installation.md", "faq.md"))
            self.assertEqual(metadata["strategy"], "structure")
            self.assertEqual(metadata["chunk_size"], 150)
            self.assertEqual(metadata["overlap"], 20)
            self.assertGreaterEqual(metadata["chunk_id"], 0)
            self.assertIn("title", metadata)
            self.assertIn("section", metadata)

    def test_section_strings(self) -> None:
        self.indexer.index_directory(
            self.docs_dir, strategies=("structure",), chunk_size=150, overlap=20)
        result = self.store.search([1.0, 1.0, 1.0], k=50)
        sections = {m["section"] for m in result["metadatas"][0]}
        self.assertIn("## Windows", sections)
        self.assertIn("## Linux", sections)

    def test_both_strategies(self) -> None:
        stats = self.indexer.index_directory(
            self.docs_dir, strategies=BOTH_STRATEGIES,
            chunk_size=150, overlap=20)
        self.assertEqual(stats.strategies, ["fixed", "structure"])
        result = self.store.search([1.0, 1.0, 1.0], k=100)
        strategies = {m["strategy"] for m in result["metadatas"][0]}
        self.assertEqual(strategies, {"fixed", "structure"})

    def test_reindex_replaces_collection(self) -> None:
        # Мелкий chunk_size → много чанков
        self.indexer.index_directory(
            self.docs_dir, strategies=("fixed",), chunk_size=60, overlap=10)
        first_count = self.store.count()
        self.assertGreater(first_count, 0)
        # Переиндексация крупным chunk_size → чанков меньше, старых нет
        stats = self.indexer.index_directory(
            self.docs_dir, strategies=("fixed",), chunk_size=2000, overlap=50)
        self.assertLess(stats.chunks, first_count)
        self.assertEqual(self.store.count(), stats.chunks)

    def test_empty_file_reported_as_error(self) -> None:
        (self.docs_dir / "empty.md").write_text("", encoding="utf-8")
        stats = self.indexer.index_directory(
            self.docs_dir, strategies=("fixed",))
        self.assertEqual(stats.files, 2)
        self.assertTrue(any("empty.md" in e for e in stats.errors))

    def test_embedder_batch_called_once(self) -> None:
        self.indexer.index_directory(
            self.docs_dir, strategies=("fixed",), chunk_size=200, overlap=30)
        self.assertEqual(len(self.embedder.batches), 1)
        self.assertGreater(len(self.embedder.batches[0]), 0)


class StepByStepTest(IndexerBase):
    """Пошаговое API для прогресс-бара UI."""

    def test_index_file_then_commit(self) -> None:
        files = self.indexer.collect_files(self.docs_dir)
        added = [self.indexer.index_file(f, ("fixed",), 200, 30)
                 for f in files]
        self.assertGreater(sum(added), 0)
        stats = self.indexer.commit()
        self.assertEqual(stats.files, 2)
        self.assertEqual(self.store.count(), stats.chunks)

    def test_buffer_kept_on_embedder_failure(self) -> None:
        self.embedder.fail = True
        self.indexer.index_file(
            self.docs_dir / "installation.md", ("fixed",), 200, 30)
        with self.assertRaises(EmbedderError):
            self.indexer.commit()
        # Буфер не потерян: после «запуска Ollama» commit проходит
        self.embedder.fail = False
        stats = self.indexer.commit()
        self.assertGreater(stats.chunks, 0)

    def test_index_file_counts_in_buffer(self) -> None:
        self.indexer.index_file(
            self.docs_dir / "installation.md", ("fixed",), 200, 30)
        self.assertEqual(self.indexer._buffer_files, 1)  # внутренний счётчик


class ClearTest(IndexerBase):
    """Кнопка «Очистить базу»."""

    def test_clear_empties_collection_and_buffer(self) -> None:
        self.indexer.index_directory(
            self.docs_dir, strategies=("fixed",))
        self.indexer.clear()
        self.assertEqual(self.store.count(), 0)
        stats = self.indexer.commit()  # буфер пуст
        self.assertEqual(stats.chunks, 0)


class InvalidParamsTest(IndexerBase):
    """Валидация параметров до чтения файлов."""

    def test_index_file_validates_overlap(self) -> None:
        with self.assertRaises(ValueError):
            self.indexer.index_file(
                self.docs_dir / "installation.md", ("fixed",), 100, 100)

    def test_index_directory_validates_chunk_size(self) -> None:
        with self.assertRaises(ValueError):
            self.indexer.index_directory(
                self.docs_dir, strategies=("fixed",), chunk_size=10, overlap=0)


if __name__ == "__main__":
    unittest.main()
