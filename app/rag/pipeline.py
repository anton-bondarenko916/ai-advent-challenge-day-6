"""RAG-пайплайн с пятью режимами: rewrite / threshold / reranker.

Режимы (константы RAG_MODE_*):

    basic            question → embed → search(top_k) → LLM
    rewrite          question → LLM rewrite → embed → search(top_k) → LLM
    rewrite_filter   rewrite → search(top_k) → filter(threshold) → LLM
    reranker         search(top_k) → rerank → top_k_final → LLM
    rewrite_reranker rewrite → search(top_k) → rerank → top_k_final → LLM

Косинусная мера Chroma: distance = 1 - similarity, поэтому скор
релевантности всегда восстанавливается как ``1.0 - distance``.

Отказы отдельных ступеней не ломают режим (философия приложения —
диалог продолжается): ошибка rewrite → поиск по исходному вопросу,
ошибка реранка → первые top_k_final кандидатов; причина попадает в
``rewrite_error`` / ``rerank_error`` результата.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.rag.embedder import OllamaEmbedder
from app.rag.retriever import RetrievedChunk
from app.rag.rewriter import QueryRewriter, RewriteError
from app.rag.reranker import BaseReranker, RerankerError
from app.rag.vector_store import DEFAULT_COLLECTION, VectorStore

# ----------------------------------------------------------------------
# Режимы
# ----------------------------------------------------------------------
RAG_MODE_BASIC = "basic"
RAG_MODE_REWRITE = "rewrite"
RAG_MODE_REWRITE_FILTER = "rewrite_filter"
RAG_MODE_RERANKER = "reranker"
RAG_MODE_REWRITE_RERANKER = "rewrite_reranker"

RAG_MODES: Tuple[str, ...] = (
    RAG_MODE_BASIC,
    RAG_MODE_REWRITE,
    RAG_MODE_REWRITE_FILTER,
    RAG_MODE_RERANKER,
    RAG_MODE_REWRITE_RERANKER,
)

# Подписи для UI (Dropdown «Режим»)
RAG_MODE_LABELS: Dict[str, str] = {
    RAG_MODE_BASIC: "Basic RAG",
    RAG_MODE_REWRITE: "Rewrite",
    RAG_MODE_REWRITE_FILTER: "Rewrite + Filter",
    RAG_MODE_RERANKER: "Reranker",
    RAG_MODE_REWRITE_RERANKER: "Rewrite + Reranker",
}

REWRITE_MODES = frozenset({
    RAG_MODE_REWRITE, RAG_MODE_REWRITE_FILTER, RAG_MODE_REWRITE_RERANKER,
})
FILTER_MODES = frozenset({RAG_MODE_REWRITE_FILTER})
RERANK_MODES = frozenset({RAG_MODE_RERANKER, RAG_MODE_REWRITE_RERANKER})

# Значения параметров поиска по умолчанию (ТЗ)
DEFAULT_TOP_K = 20          # кандидатов до фильтрации/реранка
DEFAULT_TOP_K_FINAL = 3     # чанков в контексте после
DEFAULT_THRESHOLD = 0.7     # порог similarity (cosine)
TOP_K_SEARCH_MAX = 50


def normalize_mode(mode: str) -> str:
    """Проверяет имя режима; неизвестное — ValueError."""
    value = str(mode or "").strip()
    if value not in RAG_MODES:
        raise ValueError(
            f"Неизвестный RAG-режим {mode!r}. Доступны: "
            f"{', '.join(RAG_MODES)}"
        )
    return value


def filter_by_threshold(
    chunks: List[str],
    metadatas: List[dict],
    scores: List[float],
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[List[str], List[dict], List[float]]:
    """Отсекает чанки с similarity ниже порога.

    Для cosine similarity: scores = 1 - distance (конвертация сделана
    в RAGPipeline.retrieve, сюда приходят уже similarity).

    Args:
        chunks: тексты чанков.
        metadatas: метаданные той же длины.
        scores: similarity той же длины.
        threshold: порог (граница включается).

    Returns:
        Три параллельных списка после фильтрации; всё отсечено →
        три пустых списка.
    """
    filtered = [
        (chunk, metadata, score)
        for chunk, metadata, score in zip(chunks, metadatas, scores)
        if score >= threshold
    ]
    if not filtered:
        return [], [], []
    chunks_f, metas_f, scores_f = zip(*filtered)
    return list(chunks_f), list(metas_f), list(scores_f)


class RAGPipeline:
    """Поиск с настраиваемыми ступенями (режим задаётся при вызове).

    Attributes:
        embedder: OllamaEmbedder (префикс search_query:).
        vector_store: VectorStore (Chroma, cosine).
        rewriter: QueryRewriter (None — rewrite-режимы работают без
            переформулировки, как basic);
        reranker: реранкер (None — rerank-режимы берут топ-K эмбеддинга).
        collection_name: имя коллекции Chroma.
    """

    def __init__(
        self,
        embedder: OllamaEmbedder,
        vector_store: VectorStore,
        rewriter: Optional[QueryRewriter] = None,
        reranker: Optional[BaseReranker] = None,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.rewriter = rewriter
        self.reranker = reranker
        self.collection_name = collection_name

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------
    def retrieve(
        self,
        question: str,
        mode: str = RAG_MODE_BASIC,
        top_k: int = DEFAULT_TOP_K,
        top_k_final: int = DEFAULT_TOP_K_FINAL,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> Dict[str, Any]:
        """Выполняет поиск в зависимости от режима (sync-вариант).

        Args:
            question: вопрос пользователя.
            mode: один из RAG_MODES.
            top_k: сколько кандидатов запрашивать у Chroma (до
                фильтрации/реранка).
            top_k_final: сколько чанков оставить для контекста.
            threshold: порог similarity для rewrite_filter.

        Returns:
            {"chunks": List[RetrievedChunk], "metadatas": List[dict],
            "scores": List[float] (similarity = 1 - distance),
            "search_query": str, "mode": str, "rewrite_used": bool,
            "rewrite_error": Optional[str],
            "rerank_backend": Optional[str],
            "rerank_error": Optional[str]}
        """
        mode = normalize_mode(mode)
        search_query, rewrite_used, rewrite_error = self._rewrite(question, mode)
        query_embedding = self.embedder.embed_query(search_query)
        result = self.vector_store.search(
            query_embedding, k=max(1, int(top_k)),
            collection_name=self.collection_name,
        )
        chunks = self._chunks_from_result(result)
        scores = [1.0 - chunk.distance for chunk in chunks]
        metadatas = [self._metadata_of(chunk) for chunk in chunks]

        if mode in FILTER_MODES:
            texts, metadatas, scores = filter_by_threshold(
                [chunk.text for chunk in chunks], metadatas, scores, threshold
            )
            chunks = self._rebuild_chunks(texts, metadatas, scores)

        rerank_error: Optional[str] = None
        rerank_backend: Optional[str] = None
        if mode in RERANK_MODES:
            chunks, scores, rerank_backend, rerank_error = self._rerank(
                question, chunks, scores, top_k_final
            )
        else:
            chunks, scores = chunks[:top_k_final], scores[:top_k_final]

        return {
            "chunks": chunks,
            "metadatas": [self._metadata_of(chunk) for chunk in chunks],
            "scores": scores,
            "search_query": search_query,
            "mode": mode,
            "rewrite_used": rewrite_used,
            "rewrite_error": rewrite_error,
            "rerank_backend": rerank_backend,
            "rerank_error": rerank_error,
        }

    # ------------------------------------------------------------------
    # Async (сетевые ступени — native/to_thread, Chroma локальная)
    # ------------------------------------------------------------------
    async def retrieve_async(
        self,
        question: str,
        mode: str = RAG_MODE_BASIC,
        top_k: int = DEFAULT_TOP_K,
        top_k_final: int = DEFAULT_TOP_K_FINAL,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> Dict[str, Any]:
        """Async-вариант retrieve (для Flet-обработчиков)."""
        mode = normalize_mode(mode)
        search_query, rewrite_used, rewrite_error = await self._rewrite_async(
            question, mode
        )
        query_embedding = await self.embedder.embed_query_async(search_query)
        result = self.vector_store.search(
            query_embedding, k=max(1, int(top_k)),
            collection_name=self.collection_name,
        )
        chunks = self._chunks_from_result(result)
        scores = [1.0 - chunk.distance for chunk in chunks]
        metadatas = [self._metadata_of(chunk) for chunk in chunks]

        if mode in FILTER_MODES:
            texts, metadatas, scores = filter_by_threshold(
                [chunk.text for chunk in chunks], metadatas, scores, threshold
            )
            chunks = self._rebuild_chunks(texts, metadatas, scores)

        rerank_error: Optional[str] = None
        rerank_backend: Optional[str] = None
        if mode in RERANK_MODES:
            chunks, scores, rerank_backend, rerank_error = (
                await self._rerank_async(question, chunks, scores, top_k_final)
            )
        else:
            chunks, scores = chunks[:top_k_final], scores[:top_k_final]

        return {
            "chunks": chunks,
            "metadatas": [self._metadata_of(chunk) for chunk in chunks],
            "scores": scores,
            "search_query": search_query,
            "mode": mode,
            "rewrite_used": rewrite_used,
            "rewrite_error": rewrite_error,
            "rerank_backend": rerank_backend,
            "rerank_error": rerank_error,
        }

    # ------------------------------------------------------------------
    # Ступени
    # ------------------------------------------------------------------
    def _rewrite(self, question: str, mode: str):
        """Шаг переформулировки (sync): (search_query, used, error)."""
        if mode not in REWRITE_MODES or self.rewriter is None:
            return question, False, None
        try:
            return self.rewriter.rewrite(question), True, None
        except RewriteError as exc:
            # Режим деградирует до basic-поиска, ход не падает
            return question, False, str(exc)

    async def _rewrite_async(self, question: str, mode: str):
        if mode not in REWRITE_MODES or self.rewriter is None:
            return question, False, None
        try:
            return await self.rewriter.rewrite_async(question), True, None
        except RewriteError as exc:
            return question, False, str(exc)

    def _rerank(self, question, chunks, scores, top_k_final):
        """Шаг реранка (sync): (чанки, скоры, бэкенд, ошибка)."""
        if not chunks:
            return chunks, scores, None, None
        try:
            if self.reranker is None:
                raise RerankerError("реранкер не настроен")
            indices = self.reranker.rerank(
                question, [chunk.text for chunk in chunks], top_k_final
            )
            backend = getattr(self.reranker, "last_backend", None) or (
                type(self.reranker).__name__
            )
            picked_chunks, picked_scores = self._pick(
                chunks, scores, indices)
            return picked_chunks, picked_scores, backend, None
        except RerankerError as exc:
            # Деградация: топ-K по порядку эмбеддинга
            return (chunks[:top_k_final], scores[:top_k_final],
                    None, str(exc))

    async def _rerank_async(self, question, chunks, scores, top_k_final):
        if not chunks:
            return chunks, scores, None, None
        try:
            if self.reranker is None:
                raise RerankerError("реранкер не настроен")
            indices = await self.reranker.rerank_async(
                question, [chunk.text for chunk in chunks], top_k_final
            )
            backend = getattr(self.reranker, "last_backend", None) or (
                type(self.reranker).__name__
            )
            picked_chunks, picked_scores = self._pick(
                chunks, scores, indices)
            return picked_chunks, picked_scores, backend, None
        except RerankerError as exc:
            return (chunks[:top_k_final], scores[:top_k_final],
                    None, str(exc))

    @staticmethod
    def _pick(chunks: List[RetrievedChunk], scores: List[float],
              indices: List[int]):
        """Переставляет чанки и скоры по индексам реранка."""
        return (
            [chunks[i] for i in indices],
            [scores[i] for i in indices],
        )

    # ------------------------------------------------------------------
    # Разбор ответа Chroma
    # ------------------------------------------------------------------
    def _chunks_from_result(self, result: Dict[str, List[Any]]
                            ) -> List[RetrievedChunk]:
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

    @staticmethod
    def _metadata_of(chunk: RetrievedChunk) -> Dict[str, Any]:
        """Метаданные чанка (для фильтрации и результата)."""
        return {
            "source": chunk.source,
            "title": chunk.title,
            "section": chunk.section,
            "strategy": chunk.strategy,
            "chunk_id": chunk.chunk_id,
        }

    @staticmethod
    def _rebuild_chunks(texts: List[str], metadatas: List[dict],
                        scores: List[float]) -> List[RetrievedChunk]:
        """Восстанавливает чанки после фильтрации параллельных списков.

        distance = 1 - score (обратная конвертация косинусной меры).
        """
        chunks: List[RetrievedChunk] = []
        for text, metadata, score in zip(texts, metadatas, scores):
            chunks.append(RetrievedChunk(
                text=text,
                source=str(metadata.get("source", "")),
                title=str(metadata.get("title", "")),
                section=str(metadata.get("section", "")),
                strategy=str(metadata.get("strategy", "")),
                distance=1.0 - score,
                chunk_id=int(metadata.get("chunk_id", -1)),
            ))
        return chunks
