"""Векторная БД на Chroma: PersistentClient, cosine similarity.

Данные сохраняются на диск между запусками приложения
(chromadb.PersistentClient). Пространство расстояний — cosine
(metadata={"hnsw:space": "cosine"}), как требует ТЗ.

Формат ответа search повторяет collection.query Chroma:
{"ids", "documents", "metadatas", "distances"} — списки по одному
элементу на каждый query-вектор.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document

from src.config import PROJECT_ROOT

DEFAULT_COLLECTION = "knowledge_base"
DEFAULT_PERSIST_DIR = PROJECT_ROOT / "app" / "data" / "chroma_db"

# Формат метаданных чанка (см. app.rag.chunkers.enrich_metadata)
ChunkMetadata = Dict[str, Any]


class VectorStore:
    """Обёртка над chromadb.PersistentClient с cosine-пространством.

    Attributes:
        persist_dir: каталог хранения коллекций на диске.
    """

    def __init__(self, persist_dir: Union[str, Path, None] = None) -> None:
        self.persist_dir = Path(persist_dir) if persist_dir else DEFAULT_PERSIST_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        # Анонимная телеметрия отключена: приложение может работать офлайн
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

    # ------------------------------------------------------------------
    # Публичное API
    # ------------------------------------------------------------------
    def get_or_create_collection(
        self,
        name: str = DEFAULT_COLLECTION,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Возвращает существующую или создаёт новую коллекцию.

        Args:
            name: имя коллекции (по умолчанию knowledge_base).
            metadata: метаданные коллекции; по умолчанию cosine.

        Returns:
            Коллекция Chroma.
        """
        if metadata is None:
            metadata = {"hnsw:space": "cosine"}
        return self.client.get_or_create_collection(
            name=name,
            metadata=metadata,
        )

    def index_chunks(
        self,
        chunks: Sequence[Union[Document, str]],
        embeddings: Sequence[List[float]],
        metadatas: Sequence[ChunkMetadata],
        collection_name: str = DEFAULT_COLLECTION,
    ) -> int:
        """Добавляет чанки с эмбеддингами и метаданными в коллекцию.

        IDs генерируются как chunk_0..chunk_N-1 (порядок совпадает с
        порядком входных списков). Вызывать после clear — иначе IDs
        повторятся: Chroma молча пропустит дубликаты.

        Args:
            chunks: список Document'ов (page_content) или готовых строк.
            embeddings: векторы той же длины, что chunks.
            metadatas: метаданные той же длины, что chunks.
            collection_name: имя коллекции.

        Returns:
            Число добавленных чанков.

        Raises:
            ValueError: длины списков не совпадают.
        """
        if not (len(chunks) == len(embeddings) == len(metadatas)):
            raise ValueError(
                f"Длины не совпадают: chunks={len(chunks)}, "
                f"embeddings={len(embeddings)}, metadatas={len(metadatas)}"
            )
        documents = [
            chunk if isinstance(chunk, str) else chunk.page_content
            for chunk in chunks
        ]
        collection = self.get_or_create_collection(collection_name)
        collection.add(
            ids=[f"chunk_{i}" for i in range(len(documents))],
            documents=documents,
            embeddings=list(embeddings),
            metadatas=list(metadatas),
        )
        return len(documents)

    def search(
        self,
        query_embedding: List[float],
        k: int = 3,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> Dict[str, List[Any]]:
        """Ищет топ-K ближайших чанков по cosine.

        Args:
            query_embedding: вектор вопроса (префикс search_query:).
            k: число результатов.
            collection_name: имя коллекции.

        Returns:
            Результат collection.query: ids/documents/metadatas/distances
            (списки по одному элементу на запрос).
        """
        collection = self.get_or_create_collection(collection_name)
        if collection.count() == 0:
            empty: List[Any] = []
            return {
                "ids": [empty],
                "documents": [empty],
                "metadatas": [empty],
                "distances": [empty],
            }
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )

    def count(self, collection_name: str = DEFAULT_COLLECTION) -> int:
        """Число чанков в коллекции."""
        return self.get_or_create_collection(collection_name).count()

    def clear(self, collection_name: str = DEFAULT_COLLECTION) -> None:
        """Полностью удаляет и пересоздаёт пустую коллекцию.

        Используется перед переиндексацией: гарантирует уникальность
        IDs (chunk_0..chunk_N-1) и отсутствие устаревших чанков.

        Args:
            collection_name: имя коллекции.
        """
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            # Коллекции может не быть — считать пустой
            pass
        self.get_or_create_collection(collection_name)
