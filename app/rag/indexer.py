"""Индексация папки MD-файлов: чтение → chunking → эмбеддинги → Chroma.

Оркестрация разнесена на два уровня:

* пошаговое API для UI (реальный прогресс-бар): collect_files →
  index_file (на каждый файл) → commit. ``index_file`` читает файл и
  чанкует его в буфер; тяжёлый сетевой батч (эмбеддинги) и запись в
  Chroma выполняются один раз в ``commit``;
* ``index_directory`` — sync- convenience для тестов и консольного
  использования (тот же путь целиком).

Перед commit коллекция очищается: IDs chunk_0..chunk_N-1 остаются
уникальными при повторной индексации, устаревшие чанки не копятся.
Метаданные чанков (source, title, section, chunk_id, strategy,
chunk_size, overlap) не теряются — их собирает enrich_metadata.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

from langchain_core.documents import Document

from app.rag.chunkers import (
    STRATEGY_FIXED,
    STRATEGY_STRUCTURE,
    chunk_by_strategy,
    enrich_metadata,
    validate_chunking,
)
from app.rag.embedder import OllamaEmbedder
from app.rag.vector_store import DEFAULT_COLLECTION, VectorStore

DOC_EXTENSIONS = (".md", ".markdown")

# Псевдоним для UI: «Обе» = (fixed, structure)
BOTH_STRATEGIES: Tuple[str, ...] = (STRATEGY_FIXED, STRATEGY_STRUCTURE)


@dataclass
class IndexStats:
    """Итог индексации для статус-панели «Базы знаний».

    Attributes:
        files: число проиндексированных файлов.
        chunks: общее число чанков в коллекции после commit.
        strategies: использованные стратегии ("fixed", "structure").
        chunk_size: размер чанка.
        overlap: перекрытие.
        duration: длительность в секундах.
        errors: сообщения о пропущенных файлах (не читаются/ошибка
            chunking) — индексация не прерывается.
    """

    files: int = 0
    chunks: int = 0
    strategies: List[str] = field(default_factory=list)
    chunk_size: int = 500
    overlap: int = 50
    duration: float = 0.0
    errors: List[str] = field(default_factory=list)


class KnowledgeIndexer:
    """Индексатор MD-документов в векторную базу Chroma.

    Attributes:
        embedder: OllamaEmbedder (префиксы search_document:).
        vector_store: VectorStore (Chroma, cosine).
        collection_name: имя коллекции (по умолчанию knowledge_base).
    """

    def __init__(
        self,
        embedder: OllamaEmbedder,
        vector_store: VectorStore,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.collection_name = collection_name
        # Буфер (text, metadata) между index_file и commit
        self._buffer: List[Tuple[str, Dict[str, Any]]] = []
        self._buffer_files: int = 0
        self._buffer_errors: List[str] = []

    # ------------------------------------------------------------------
    # Пошаговое API (для прогресс-бара UI)
    # ------------------------------------------------------------------
    @staticmethod
    def collect_files(folder: Union[str, Path]) -> List[Path]:
        """Собирает MD-файлы папки (без рекурсии, по расширению).

        Args:
            folder: каталог с документами.

        Returns:
            Отсортированный список путей *.md / *.markdown.
        """
        root = Path(folder)
        if not root.is_dir():
            return []
        files = [
            path for path in sorted(root.iterdir())
            if path.is_file() and path.suffix.lower() in DOC_EXTENSIONS
        ]
        return files

    @staticmethod
    def extract_title(text: str, fallback: str) -> str:
        """Заголовок документа: первая строка "# ...", иначе имя файла."""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# ") and len(stripped) > 2:
                return stripped[2:].strip()
            if stripped:
                # Первая непустая строка без "# " — заголовка нет
                break
        return fallback

    def index_file(
        self,
        path: Union[str, Path],
        strategies: Sequence[str] = (STRATEGY_FIXED,),
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> int:
        """Читает файл, чанкует выбранными стратегиями, кладёт в буфер.

        Args:
            path: путь к MD-файлу.
            strategies: последовательность стратегий из {"fixed",
                "structure"} (для «Обе» — обе).
            chunk_size: максимальный размер чанка.
            overlap: перекрытие чанков.

        Returns:
            Число добавленных в буфер чанков (0 при ошибке — она
            записывается в errors и попадёт в IndexStats).

        Raises:
            ValueError: некорректные chunking-параметры (проверяются до
                чтения файлов).
        """
        validate_chunking(chunk_size, overlap)
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            self._buffer_errors.append(f"{path.name}: не читается ({exc})")
            return 0
        if not text.strip():
            self._buffer_errors.append(f"{path.name}: пустой файл пропущен")
            return 0
        title = self.extract_title(text, fallback=path.stem)
        added = 0
        for strategy in strategies:
            try:
                documents = chunk_by_strategy(text, strategy, chunk_size, overlap)
            except Exception as exc:
                self._buffer_errors.append(
                    f"{path.name}: ошибка chunking {strategy!r} ({exc})"
                )
                continue
            enrich_metadata(
                documents,
                source=path.name,
                title=title,
                strategy=strategy,
                chunk_size=chunk_size,
                overlap=overlap,
                first_chunk_id=len(self._buffer),
            )
            for doc in documents:
                self._buffer.append((doc.page_content, dict(doc.metadata)))
                added += 1
        if added:
            self._buffer_files += 1
        return added

    def commit(self) -> IndexStats:
        """Эмбеддирует буфер одним батчем и перезаписывает коллекцию.

        Returns:
            IndexStats: файлы, чанки, стратегии, длительность, ошибки.

        Raises:
            EmbedderError: Ollama недоступна (буфер сохраняется — commit
                можно повторить после запуска Ollama).
        """
        started = time.perf_counter()
        texts = [text for text, _ in self._buffer]
        metadatas = [metadata for _, metadata in self._buffer]
        embeddings = self.embedder.embed_documents(texts) if texts else []
        self.vector_store.clear(self.collection_name)
        stored = 0
        if texts:
            stored = self.vector_store.index_chunks(
                chunks=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                collection_name=self.collection_name,
            )
        strategies = sorted({
            metadata.get("strategy", "")
            for metadata in metadatas
        } - {""})
        stats = IndexStats(
            files=self._buffer_files,
            chunks=stored,
            strategies=strategies,
            chunk_size=metadatas[0]["chunk_size"] if metadatas else 500,
            overlap=metadatas[0]["overlap"] if metadatas else 50,
            duration=round(time.perf_counter() - started, 2),
            errors=list(self._buffer_errors),
        )
        self._reset_buffer()
        return stats

    # ------------------------------------------------------------------
    # Полный путь одним вызовом (тесты, консоль)
    # ------------------------------------------------------------------
    def index_directory(
        self,
        folder: Union[str, Path],
        strategies: Sequence[str] = (STRATEGY_FIXED,),
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> IndexStats:
        """Индексирует папку целиком (sync-вариант).

        Args:
            folder: каталог с MD-файлами.
            strategies: стратегии chunking.
            chunk_size: максимальный размер чанка.
            overlap: перекрытие чанков.

        Returns:
            IndexStats с итогами.
        """
        validate_chunking(chunk_size, overlap)
        self._reset_buffer()
        for path in self.collect_files(folder):
            self.index_file(path, strategies, chunk_size, overlap)
        return self.commit()

    def clear(self) -> None:
        """Очищает коллекцию и буфер (кнопка «Очистить базу»)."""
        self._reset_buffer()
        self.vector_store.clear(self.collection_name)

    def _reset_buffer(self) -> None:
        self._buffer = []
        self._buffer_files = 0
        self._buffer_errors = []
