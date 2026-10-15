"""RAG по FAQ и документации поддержки: отдельная коллекция Chroma.

Личная база знаний (knowledge_base), документация проекта (project_docs)
и код (project_code) НЕ смешиваются с поддержкой: FAQ индексируется в
отдельную коллекцию ``support_faq`` того же VectorStore (Chroma, cosine)
— ТЗ ai_1_34 «ЧЕГО НЕ ДЕЛАТЬ: не смешивать support_faq с другими
коллекциями».

Что индексируется: все ``*.md`` рекурсивно из папки поддержки
(по умолчанию support_docs/ в корне проекта; путь хранится в app_state,
ключ support_docs_path). Код и конфиги поддержкой не индексируются.

Chunking — structure-aware (chunk_structure: по заголовкам #/##/###),
chunk_size=500, overlap=50 (ТЗ). Метаданные чанка: source (имя файла),
path (путь ОТ КОРНЯ папки поддержки), title/section (заголовки),
chunk_id (сквозной), strategy/chunk_size/overlap, type (``support``).

Переиндексация перезаписывает коллекцию целиком; эмбеддинги генерируются
БАТЧАМИ по EMBED_BATCH_SIZE ДО очистки (один POST /api/embed на всё не
масштабируется — урок ai_1_33-verify; при недоступной Ollama старая
коллекция остаётся рабочей).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from app.rag.chunkers import chunk_structure, enrich_metadata, validate_chunking
from app.rag.embedder import EmbedderError
from app.rag.indexer import KnowledgeIndexer
from app.rag.project_rag import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    PROJECT_DOCS_COLLECTION,
    ProjectRAG,
)

# Имя коллекции поддержки (отдельно от knowledge_base/project_docs)
SUPPORT_FAQ_COLLECTION = "support_faq"

# Ключ app_state (SQLite) с папкой поддержки, выбранной в UI
SUPPORT_DOCS_PATH_KEY = "support_docs_path"

TYPE_SUPPORT = "support"

# Размер батча эмбеддингов (см. CodeRAG: один POST на всё не масштабируется)
EMBED_BATCH_SIZE = 64


def default_support_docs_path() -> str:
    """Папка поддержки по умолчанию — support_docs/ в корне проекта.

    Папка поставляется с приложением (FAQ по частым вопросам), поэтому
    индексация «из коробки» работает сразу после установки.
    """
    from src.config import PROJECT_ROOT

    return str(PROJECT_ROOT / "support_docs")


class SupportRAG(ProjectRAG):
    """RAG по FAQ и документации поддержки (коллекция support_faq).

    Наследует у ProjectRAG поиск (search/search_async/search_chunks_
    async/count/clear) и разбор чанков; переопределяются коллекция и
    сбор файлов (все *.md рекурсивно, метаданные type="support").

    Attributes:
        vector_store: VectorStore (Chroma, cosine) — тот же экземпляр,
            что у личной базы знаний, другая коллекция.
        embedder: OllamaEmbedder (префиксы search_document/search_query).
        collection_name: имя коллекции (по умолчанию support_faq).
    """

    COLLECTION_NAME = SUPPORT_FAQ_COLLECTION

    def __init__(
        self,
        vector_store: Any,
        embedder: Any,
        collection_name: str = SUPPORT_FAQ_COLLECTION,
    ) -> None:
        super().__init__(vector_store, embedder,
                         collection_name=collection_name)

    # ------------------------------------------------------------------
    # Сбор файлов (все *.md рекурсивно — ТЗ)
    # ------------------------------------------------------------------
    @staticmethod
    def collect_files(docs_path: Union[str, Path]) -> List[Tuple[Path, str]]:
        """Собирает ``*.md`` папки поддержки рекурсивно.

        Args:
            docs_path: корень папки поддержки.

        Returns:
            Список (путь, тип ``support``) в детерминированном порядке
            (sorted); несуществующий путь — пустой список.
        """
        root = Path(docs_path)
        if not root.is_dir():
            return []
        return [
            (path, TYPE_SUPPORT)
            for path in sorted(root.rglob("*.md"))
            if path.is_file()
        ]

    # ------------------------------------------------------------------
    # Индексация
    # ------------------------------------------------------------------
    def index_support_docs(
        self,
        docs_path: Union[str, Path],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ) -> Dict[str, Any]:
        """Индексирует FAQ + документацию в коллекцию support_faq.

        Файлы собираются collect_files, режутся chunk_structure
        (заголовки → секции), эмбеддинги генерируются батчами ДО
        очистки, коллекция перезаписывается целиком.

        Args:
            docs_path: корень папки поддержки.
            chunk_size: максимальный размер чанка в символах.
            overlap: перекрытие чанков в символах.

        Returns:
            Статистика: {"files", "chunks", "errors", "duration"}
            (errors — пропущенные файлы; индексация не прерывается).

        Raises:
            ValueError: некорректные chunk_size/overlap.
            app.rag.embedder.EmbedderError: Ollama недоступна (коллекция
                НЕ очищается — старые чанки остаются рабочими).
        """
        validate_chunking(chunk_size, overlap)
        started = time.perf_counter()
        root = Path(docs_path)
        files = self.collect_files(root)

        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        errors: List[str] = []
        indexed_files = 0
        for path, file_type in files:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                errors.append(f"{path.name}: не читается ({exc})")
                continue
            if not text.strip():
                errors.append(f"{path.name}: пустой файл пропущен")
                continue
            try:
                relative = str(path.relative_to(root))
            except ValueError:  # путь вне корня (симлинк) — как есть
                relative = str(path)
            title = KnowledgeIndexer.extract_title(text, fallback=path.stem)
            chunks = chunk_structure(text, chunk_size=chunk_size,
                                     overlap=overlap)
            enrich_metadata(
                chunks,
                source=path.name,
                title=title,
                strategy="structure",
                chunk_size=chunk_size,
                overlap=overlap,
                first_chunk_id=len(texts),
            )
            for chunk in chunks:
                chunk.metadata["path"] = relative
                chunk.metadata["type"] = file_type
                texts.append(chunk.page_content)
                metadatas.append(dict(chunk.metadata))
            indexed_files += 1

        stored = 0
        if texts:
            # Батчами по EMBED_BATCH_SIZE и ДО clear: недоступный эмбеддер
            # оставляет старую коллекцию рабочей (EmbedderError уходит
            # вызывающему)
            embeddings: List[List[float]] = []
            for start in range(0, len(texts), EMBED_BATCH_SIZE):
                embeddings.extend(
                    self.embedder.embed_documents(
                        texts[start:start + EMBED_BATCH_SIZE]))
            self.vector_store.clear(self.collection_name)
            stored = self.vector_store.index_chunks(
                chunks=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                collection_name=self.collection_name,
            )
        return {
            "files": indexed_files,
            "chunks": stored,
            "errors": errors,
            "duration": round(time.perf_counter() - started, 2),
        }
