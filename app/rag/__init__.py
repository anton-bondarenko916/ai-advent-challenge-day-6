"""RAG-пайплайн: chunking → Ollama-эмбеддинги → Chroma → поиск.

Модули:
    chunkers     — две стратегии разбиения MD-документов (fixed / structure);
    embedder     — OllamaEmbedder (префиксы search_document:/search_query:);
    vector_store — VectorStore поверх chromadb.PersistentClient (cosine);
    retriever    — поиск топ-K чанков + формирование промпта и источников;
    indexer      — индексация папки MD-файлов (оркестрация).
"""

from __future__ import annotations

from app.rag.chunkers import chunk_fixed, chunk_structure, enrich_metadata
from app.rag.embedder import EmbedderError, OllamaEmbedder
from app.rag.indexer import IndexStats, KnowledgeIndexer
from app.rag.retriever import RetrievedChunk, Retriever
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
]
