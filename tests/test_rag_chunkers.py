"""Тесты chunking (app/rag/chunkers.py): две стратегии и метаданные.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

from langchain_core.documents import Document

from app.rag.chunkers import (
    STRATEGY_FIXED,
    STRATEGY_STRUCTURE,
    chunk_by_strategy,
    chunk_fixed,
    chunk_structure,
    enrich_metadata,
    section_from_metadata,
    validate_chunking,
)

SAMPLE_DOC = """# Руководство пользователя

Общее описание приложения и его назначения.

## Установка

Скачайте установщик с официального сайта. Запустите его от имени администратора и следуйте подсказкам мастера.

### Windows

Для Windows требуется .NET 8. Установщик предложит установить её автоматически при отсутствии.

### Linux

Используйте пакетный менеджер дистрибутива: apt, dnf или pacman.

## Использование

Нажмите кнопку «Создать» на панели инструментов.
"""


class ValidateChunkingTest(unittest.TestCase):
    """Валидация параметров chunking."""

    def test_valid_params_pass(self) -> None:
        validate_chunking(500, 50)  # не должно исключений

    def test_overlap_must_be_less_than_chunk_size(self) -> None:
        with self.assertRaises(ValueError):
            validate_chunking(100, 100)

    def test_chunk_size_too_small(self) -> None:
        with self.assertRaises(ValueError):
            validate_chunking(10, 0)

    def test_negative_overlap(self) -> None:
        with self.assertRaises(ValueError):
            validate_chunking(500, -1)

    def test_non_int_params(self) -> None:
        with self.assertRaises(ValueError):
            validate_chunking("500", 50)  # type: ignore[arg-type]


class ChunkFixedTest(unittest.TestCase):
    """Стратегия A: фиксированный размер."""

    def test_returns_documents(self) -> None:
        chunks = chunk_fixed(SAMPLE_DOC, 200, 40)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIsInstance(chunk, Document)
            self.assertTrue(chunk.page_content.strip())

    def test_chunks_not_exceed_limit(self) -> None:
        # Допуск на служебные символы: сплиттер иногда превышает лимит
        # на границах разделителей — проверяем разумный максимум
        chunks = chunk_fixed(SAMPLE_DOC, 200, 40)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.page_content), 260)

    def test_empty_text_gives_nothing(self) -> None:
        self.assertEqual(chunk_fixed("", 200, 40), [])

    def test_invalid_params_raise(self) -> None:
        with self.assertRaises(ValueError):
            chunk_fixed(SAMPLE_DOC, 100, 100)

    def test_long_text_produces_many_chunks(self) -> None:
        long_text = "слово " * 1000
        chunks = chunk_fixed(long_text, 100, 20)
        self.assertGreater(len(chunks), 10)


class ChunkStructureTest(unittest.TestCase):
    """Стратегия B: structure-aware (по заголовкам)."""

    def test_headers_in_metadata(self) -> None:
        chunks = chunk_structure(SAMPLE_DOC, 200, 40)
        h2_sections = {c.metadata.get("h2") for c in chunks}
        self.assertIn("Установка", h2_sections)
        self.assertIn("Использование", h2_sections)

    def test_h3_headers_preserved(self) -> None:
        chunks = chunk_structure(SAMPLE_DOC, 200, 40)
        h3_sections = {c.metadata.get("h3") for c in chunks}
        self.assertIn("Windows", h3_sections)
        self.assertIn("Linux", h3_sections)

    def test_content_split_by_sections(self) -> None:
        chunks = chunk_structure(SAMPLE_DOC, 200, 40)
        windows = [c for c in chunks if c.metadata.get("h3") == "Windows"]
        self.assertTrue(windows)
        self.assertIn(".NET", " ".join(c.page_content for c in windows))

    def test_invalid_params_raise(self) -> None:
        with self.assertRaises(ValueError):
            chunk_structure(SAMPLE_DOC, 100, 100)


class ChunkByStrategyTest(unittest.TestCase):
    """Диспетчер стратегий."""

    def test_fixed_strategy(self) -> None:
        chunks = chunk_by_strategy(SAMPLE_DOC, STRATEGY_FIXED, 200, 40)
        self.assertGreater(len(chunks), 0)

    def test_structure_strategy(self) -> None:
        chunks = chunk_by_strategy(SAMPLE_DOC, STRATEGY_STRUCTURE, 200, 40)
        self.assertTrue(any(c.metadata.get("h2") for c in chunks))

    def test_unknown_strategy_raises(self) -> None:
        with self.assertRaises(ValueError):
            chunk_by_strategy(SAMPLE_DOC, "semantic", 200, 40)


class SectionFromMetadataTest(unittest.TestCase):
    """Строка секции из метаданных заголовков."""

    def test_h2_wins_over_h1(self) -> None:
        self.assertEqual(
            section_from_metadata({"h1": "Руководство", "h2": "Установка"}),
            "## Установка",
        )

    def test_h3_is_deepest(self) -> None:
        self.assertEqual(
            section_from_metadata({"h2": "Установка", "h3": "Windows"}),
            "### Windows",
        )

    def test_h1_without_hash_duplication(self) -> None:
        self.assertEqual(section_from_metadata({"h1": "Введение"}),
                         "# Введение")

    def test_empty_metadata(self) -> None:
        self.assertEqual(section_from_metadata({}), "")


class EnrichMetadataTest(unittest.TestCase):
    """Обогащение метаданных чанков (формат ТЗ)."""

    def test_metadata_fields(self) -> None:
        docs = chunk_structure(SAMPLE_DOC, 200, 40)
        enriched = enrich_metadata(
            docs, source="guide.md", title="Руководство",
            strategy=STRATEGY_STRUCTURE, chunk_size=200, overlap=40,
        )
        for index, doc in enumerate(enriched):
            self.assertEqual(doc.metadata["source"], "guide.md")
            self.assertEqual(doc.metadata["title"], "Руководство")
            self.assertEqual(doc.metadata["chunk_id"], index)
            self.assertEqual(doc.metadata["strategy"], STRATEGY_STRUCTURE)
            self.assertEqual(doc.metadata["chunk_size"], 200)
            self.assertEqual(doc.metadata["overlap"], 40)

    def test_raw_header_keys_removed(self) -> None:
        docs = chunk_structure(SAMPLE_DOC, 200, 40)
        enriched = enrich_metadata(
            docs, "guide.md", "Руководство", STRATEGY_STRUCTURE, 200, 40)
        for doc in enriched:
            self.assertNotIn("h1", doc.metadata)
            self.assertNotIn("h2", doc.metadata)
            self.assertNotIn("h3", doc.metadata)

    def test_section_string_in_metadata(self) -> None:
        docs = chunk_structure(SAMPLE_DOC, 200, 40)
        enriched = enrich_metadata(
            docs, "guide.md", "Руководство", STRATEGY_STRUCTURE, 200, 40)
        sections = {doc.metadata["section"] for doc in enriched}
        self.assertIn("## Установка", sections)
        self.assertIn("### Windows", sections)

    def test_fixed_chunks_have_empty_section(self) -> None:
        docs = chunk_fixed(SAMPLE_DOC, 200, 40)
        enriched = enrich_metadata(
            docs, "guide.md", "Руководство", STRATEGY_FIXED, 200, 40)
        for doc in enriched:
            self.assertEqual(doc.metadata["section"], "")

    def test_chunk_id_offset(self) -> None:
        docs = chunk_fixed("часть " * 50, 50, 10)
        enriched = enrich_metadata(
            docs, "a.md", "A", STRATEGY_FIXED, 50, 10, first_chunk_id=7)
        self.assertEqual(enriched[0].metadata["chunk_id"], 7)
        self.assertEqual(enriched[-1].metadata["chunk_id"], 7 + len(docs) - 1)


if __name__ == "__main__":
    unittest.main()
