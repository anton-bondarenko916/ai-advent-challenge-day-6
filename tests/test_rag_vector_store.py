"""Тесты векторной БД (app/rag/vector_store.py) — реальный Chroma в tmp.

Проверяются: PersistentClient (данные между экземплярами), cosine,
index_chunks/search/clear, уникальность IDs после очистки.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from app.rag.vector_store import DEFAULT_COLLECTION, VectorStore


def fake_embedding(seed: float) -> list:
    """Детерминированный «вектор»: [seed, seed/2, 1.0]."""
    return [seed, seed / 2, 1.0]


class VectorStoreBase(unittest.TestCase):
    """Общий fixture: временный каталог хранения."""

    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="chroma_test_"))
        self.store = VectorStore(persist_dir=self.tmp_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


class CollectionTest(VectorStoreBase):
    """Создание коллекции и cosine-пространство."""

    def test_default_collection_created(self) -> None:
        collection = self.store.get_or_create_collection()
        self.assertEqual(collection.name, DEFAULT_COLLECTION)
        self.assertEqual(collection.count(), 0)

    def test_cosine_space_metadata(self) -> None:
        collection = self.store.get_or_create_collection()
        self.assertEqual(collection.metadata.get("hnsw:space"), "cosine")

    def test_persist_dir_created(self) -> None:
        self.assertTrue(self.tmp_dir.exists())


class IndexAndSearchTest(VectorStoreBase):
    """Добавление чанков и поиск."""

    def setUp(self) -> None:
        super().setUp()
        self.store.index_chunks(
            chunks=["Чанк про установку", "Чанк про погоду"],
            embeddings=[fake_embedding(0.1), fake_embedding(0.9)],
            metadatas=[
                {"source": "a.md", "section": "## Windows", "chunk_id": 0},
                {"source": "b.md", "section": "## Лето", "chunk_id": 1},
            ],
        )

    def test_index_chunks_counts(self) -> None:
        self.assertEqual(self.store.count(), 2)

    def test_index_chunks_accepts_documents(self) -> None:
        from langchain_core.documents import Document
        # Чистая коллекция: в setUp уже есть chunk_0/chunk_1
        self.store.clear()
        stored = self.store.index_chunks(
            chunks=[Document(page_content="Документ-чанк")],
            embeddings=[fake_embedding(0.5)],
            metadatas=[{"source": "c.md", "chunk_id": 2}],
        )
        self.assertEqual(stored, 1)
        self.assertEqual(self.store.count(), 1)

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.store.index_chunks(
                chunks=["один"],
                embeddings=[fake_embedding(0.1), fake_embedding(0.2)],
                metadatas=[{}],
            )

    def test_search_returns_nearest(self) -> None:
        result = self.store.search(fake_embedding(0.11), k=1)
        documents = result["documents"][0]
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0], "Чанк про установку")
        metadata = result["metadatas"][0][0]
        self.assertEqual(metadata["source"], "a.md")
        self.assertEqual(metadata["section"], "## Windows")

    def test_search_orders_by_similarity(self) -> None:
        result = self.store.search(fake_embedding(0.11), k=2)
        documents = result["documents"][0]
        self.assertEqual(documents[0], "Чанк про установку")

    def test_search_k_larger_than_collection(self) -> None:
        result = self.store.search(fake_embedding(0.1), k=10)
        self.assertEqual(len(result["documents"][0]), 2)

    def test_distances_present(self) -> None:
        result = self.store.search(fake_embedding(0.1), k=1)
        self.assertGreaterEqual(len(result["distances"][0]), 1)
        # cosine distance: 0 — точное совпадение
        self.assertGreaterEqual(result["distances"][0][0], 0.0)

    def test_metadata_not_lost(self) -> None:
        result = self.store.search(fake_embedding(0.9), k=1)
        metadata = result["metadatas"][0][0]
        # Chroma хранит ровно те ключи, что переданы при индексации
        self.assertEqual(metadata["chunk_id"], 1)
        self.assertEqual(metadata["source"], "b.md")
        self.assertEqual(metadata["section"], "## Лето")


class PersistenceTest(VectorStoreBase):
    """PersistentClient: данные живут между экземплярами/запусками."""

    def test_data_visible_from_new_instance(self) -> None:
        self.store.index_chunks(
            chunks=["персистентный чанк"],
            embeddings=[fake_embedding(0.3)],
            metadatas=[{"source": "p.md", "chunk_id": 0}],
        )
        # Новый клиент поверх того же каталога — как перезапуск приложения
        second_store = VectorStore(persist_dir=self.tmp_dir)
        self.assertEqual(second_store.count(), 1)
        result = second_store.search(fake_embedding(0.3), k=1)
        self.assertEqual(result["documents"][0][0], "персистентный чанк")


class ClearTest(VectorStoreBase):
    """Очистка коллекции и уникальность IDs при переиндексации."""

    def test_clear_empties_collection(self) -> None:
        self.store.index_chunks(
            chunks=["a", "b"],
            embeddings=[fake_embedding(0.1), fake_embedding(0.2)],
            metadatas=[{"chunk_id": 0}, {"chunk_id": 1}],
        )
        self.store.clear()
        self.assertEqual(self.store.count(), 0)

    def test_clear_on_missing_collection_is_safe(self) -> None:
        fresh = VectorStore(
            persist_dir=self.tmp_dir / "nested_fresh")
        fresh.clear()  # не должно исключений
        self.assertEqual(fresh.count(), 0)

    def test_reindex_after_clear_unique_ids(self) -> None:
        self.store.index_chunks(
            chunks=["старый чанк"],
            embeddings=[fake_embedding(0.1)],
            metadatas=[{"chunk_id": 0}],
        )
        self.store.clear()
        # Те же IDs chunk_0.. — после clear они уникальны
        self.store.index_chunks(
            chunks=["новый чанк 1", "новый чанк 2"],
            embeddings=[fake_embedding(0.1), fake_embedding(0.2)],
            metadatas=[{"chunk_id": 0}, {"chunk_id": 1}],
        )
        self.assertEqual(self.store.count(), 2)
        result = self.store.search(fake_embedding(0.1), k=2)
        self.assertNotIn("старый чанк", result["documents"][0])

    def test_search_empty_collection_returns_empty_structure(self) -> None:
        result = self.store.search(fake_embedding(0.1), k=3)
        self.assertEqual(result["documents"], [[]])
        self.assertEqual(result["metadatas"], [[]])
        self.assertEqual(result["distances"], [[]])


if __name__ == "__main__":
    unittest.main()
