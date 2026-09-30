"""RAG-пайплайн: chunking → Ollama-эмбеддинги → Chroma → поиск.

Модули:
    chunkers     — две стратегии разбиения MD-документов (fixed / structure);
    embedder     — OllamaEmbedder (префиксы search_document:/search_query:);
    vector_store — VectorStore поверх chromadb.PersistentClient (cosine);
    retriever    — поиск топ-K чанков + формирование промпта и источников;
    indexer      — индексация папки MD-файлов (оркестрация);
    rewriter     — QueryRewriter: переформулировка вопроса (DeepSeek);
    reranker     — реранкеры (Ollama → FlagEmbedding → Lexical) + fallback;
    pipeline     — RAGPipeline: 5 режимов (basic/rewrite/filter/rerank).
"""

from __future__ import annotations

from app.rag.chunkers import chunk_fixed, chunk_structure, enrich_metadata
from app.rag.embedder import EmbedderError, OllamaEmbedder
from app.rag.indexer import IndexStats, KnowledgeIndexer
from app.rag.pipeline import (
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    DEFAULT_TOP_K_FINAL,
    RAG_MODES,
    RAG_MODE_BASIC,
    RAG_MODE_LABELS,
    RAG_MODE_RERANKER,
    RAG_MODE_REWRITE,
    RAG_MODE_REWRITE_FILTER,
    RAG_MODE_REWRITE_RERANKER,
    RAGPipeline,
    filter_by_threshold,
    normalize_mode,
)
from app.rag.reranker import (
    BaseReranker,
    FallbackReranker,
    FlagEmbeddingReranker,
    LexicalReranker,
    OllamaReranker,
    RerankerError,
    create_default_reranker,
)
from app.rag.retriever import RetrievedChunk, Retriever
from app.rag.rewriter import QueryRewriter, RewriteError
from app.rag.vector_store import VectorStore

__all__ = [
    "chunk_fixed",
    "chunk_structure",
    "enrich_metadata",
    "EmbedderError",
    "OllamaEmbedder",
    "IndexStats",
    "KnowledgeIndexer",
    "RetrievedChunk",
    "Retriever",
    "VectorStore",
    "QueryRewriter",
    "RewriteError",
    "BaseReranker",
    "OllamaReranker",
    "FlagEmbeddingReranker",
    "LexicalReranker",
    "FallbackReranker",
    "RerankerError",
    "create_default_reranker",
    "RAGPipeline",
    "filter_by_threshold",
    "normalize_mode",
    "RAG_MODES",
    "RAG_MODE_BASIC",
    "RAG_MODE_REWRITE",
    "RAG_MODE_REWRITE_FILTER",
    "RAG_MODE_RERANKER",
    "RAG_MODE_REWRITE_RERANKER",
    "RAG_MODE_LABELS",
    "DEFAULT_TOP_K",
    "DEFAULT_TOP_K_FINAL",
    "DEFAULT_THRESHOLD",
]
