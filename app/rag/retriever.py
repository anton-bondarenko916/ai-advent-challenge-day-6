"""Поиск по базе знаний: вопрос → топ-K чанков → промпт и источники.

Retriever связывает OllamaEmbedder (префикс search_query:) и VectorStore
(cosine, топ-K). Формирование промпта — по формату ТЗ:

    Контекст:
    {чанк 1}

    {чанк 2}

    Вопрос: {вопрос}

    Ответь на вопрос, используя только предоставленный контекст.
    Если в контексте нет ответа — скажи об этом.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from app.rag.embedder import OllamaEmbedder
from app.rag.vector_store import DEFAULT_COLLECTION, VectorStore

PREVIEW_LENGTH = 160

# Инструкция для LLM в промпте с контекстом (формат ТЗ)
RAG_INSTRUCTION = (
    "Ответь на вопрос, используя только предоставленный контекст. "
    "Если в контексте нет ответа — скажи об этом."
)


@dataclass
class RetrievedChunk:
    """Чанк, найденный в базе знаний.

    Attributes:
        text: текст чанка.
        source: имя файла-источника ("installation.md").
        title: заголовок документа.
        section: секция ("## Windows") или "".
        strategy: стратегия chunking ("fixed" | "structure").
        distance: косинусное расстояние (0 — точное совпадение).
        chunk_id: сквозной номер чанка при индексации.
    """

    text: str
    source: str = ""
    title: str = ""
    section: str = ""
    strategy: str = ""
    distance: float = 0.0
    chunk_id: int = -1

    @property
    def preview(self) -> str:
        """Короткий фрагмент для панели «Источники»."""
        collapsed = " ".join(self.text.split())
        if len(collapsed) > PREVIEW_LENGTH:
            return collapsed[: PREVIEW_LENGTH - 1] + "…"
        return collapsed


class Retriever:
    """Поиск топ-K чанков и сборка промпта/источников.

    Attributes:
        top_k: сколько чанков подавать в контекст (по умолчанию 3).
    """

    def __init__(
        self,
        embedder: OllamaEmbedder,
        vector_store: VectorStore,
        top_k: int = 3,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = max(1, int(top_k))
        self.collection_name = collection_name

    # ------------------------------------------------------------------
    # Поиск
    # ------------------------------------------------------------------
    def retrieve(self, question: str) -> List[RetrievedChunk]:
        """Ищет чанки по вопросу (sync-вариант).

        Args:
            question: вопрос пользователя.

        Returns:
            До top_k чанков, отсортированных по релевантности (порядок
            Chroma). Пустая база → пустой список.

        Raises:
            EmbedderError: Ollama недоступна (не гасится — решение о
                фолбэке принимает вызывающий код).
        """
        query_embedding = self.embedder.embed_query(question)
        return self._chunks_from_result(
            self.vector_store.search(query_embedding, k=self.top_k,
                                     collection_name=self.collection_name)
        )

    async def retrieve_async(self, question: str) -> List[RetrievedChunk]:
        """Async-вариант retrieve (сетевой вызов — не блокирует UI-луп)."""
        query_embedding = await self.embedder.embed_query_async(question)
        result = self.vector_store.search(
            query_embedding, k=self.top_k, collection_name=self.collection_name
        )
        return self._chunks_from_result(result)

    def _chunks_from_result(self, result: Dict[str, List[Any]]) -> List[RetrievedChunk]:
        """Собирает RetrievedChunk из ответа collection.query."""
        chunks: List[RetrievedChunk] = []
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for index, text in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            metadata = metadata or {}
            distance = distances[index] if index < len(distances) else 0.0
            chunks.append(RetrievedChunk(
                text=text,
                source=str(metadata.get("source", "")),
                title=str(metadata.get("title", "")),
                section=str(metadata.get("section", "")),
                strategy=str(metadata.get("strategy", "")),
                distance=float(distance) if distance is not None else 0.0,
                chunk_id=int(metadata.get("chunk_id", -1)),
            ))
        return chunks

    # ------------------------------------------------------------------
    # Промпт и источники
    # ------------------------------------------------------------------
    @staticmethod
    def build_rag_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
        """Собирает промпт с контекстом (формат ТЗ).

        Args:
            question: вопрос пользователя.
            chunks: найденные чанки (может быть пусто).

        Returns:
            Полный текст последнего user-сообщения запроса; пустая
            строка, если чанков нет (режим без RAG / фолбэк).
        """
        if not chunks:
            return ""
        context = "\n\n".join(chunk.text for chunk in chunks)
        return (
            f"Контекст:\n{context}\n\n"
            f"Вопрос: {question}\n\n"
            f"{RAG_INSTRUCTION}"
        )

    @staticmethod
    def format_sources(chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
        """Готовит источники для панели под ответом LLM.

        Args:
            chunks: найденные чанки.

        Returns:
            Список словарей {source, title, section, strategy, distance,
            chunk_id, preview} — метаданные из ТЗ + фрагмент текста.
        """
        return [
            {
                "source": chunk.source,
                "title": chunk.title,
                "section": chunk.section,
                "strategy": chunk.strategy,
                "distance": round(chunk.distance, 4),
                "chunk_id": chunk.chunk_id,
                "preview": chunk.preview,
            }
            for chunk in chunks
        ]
