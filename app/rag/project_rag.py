"""RAG по документации проекта: отдельная коллекция Chroma.

Личная база знаний (knowledge_base) и документация проекта НЕ
смешиваются: проект индексируется в отдельную коллекцию
``project_docs`` того же VectorStore (Chroma, cosine).

Что индексируется (код *.py НЕ индексируется):

* README в корне проекта (readme.md / README.md / README.MD);
* ``docs/**/*.md`` — рекурсивно, включая подкаталоги;
* конфиги в корне: requirements.txt, pyproject.toml, setup.py,
  docker-compose.yml.

Chunking — structure-aware (chunk_structure: по заголовкам #/##/###,
метаданные h1/h2/h3), как в ТЗ; конфиги без заголовков превращаются
сплиттером в одну секцию и дальше режутся рекурсивно.

Метаданные чанка (формат панели «Источники» + тип файла): source
(имя файла), path (полный путь), title (первый H1), section (самый
глубокий заголовок), chunk_id (сквозной), strategy, chunk_size,
overlap, type (readme | docs | config).

Переиндексация перезаписывает коллекцию целиком (clear перед
index_chunks — IDs chunk_0..chunk_N-1 остаются уникальными).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from app.rag.chunkers import chunk_structure, enrich_metadata
from app.rag.embedder import OllamaEmbedder
from app.rag.indexer import KnowledgeIndexer
from app.rag.retriever import RetrievedChunk
from app.rag.vector_store import VectorStore

# Имя коллекции документации проекта (отдельно от knowledge_base)
PROJECT_DOCS_COLLECTION = "project_docs"

# Ключ app_state (SQLite) с путём проекта, выбранным в UI
PROJECT_DOCS_PATH_KEY = "project_docs_path"

# README-варианты в порядке приоритета (ТЗ)
README_CANDIDATES: Tuple[str, ...] = ("README.md", "readme.md", "README.MD")

# Конфиги в корне проекта (код *.py не индексируется — setup.py
# исключение из ТЗ: это декларативный конфиг пакета)
CONFIG_FILES: Tuple[str, ...] = (
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "docker-compose.yml",
)

TYPE_README = "readme"
TYPE_DOCS = "docs"
TYPE_CONFIG = "config"

# Параметры chunking по умолчанию (ТЗ)
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50


def default_project_path() -> str:
    """Путь проекта по умолчанию — корень этого приложения.

    У приложения есть README.md и docs/*.md, поэтому индексация «себя
    же» работает сразу после установки.
    """
    from src.config import PROJECT_ROOT

    return str(PROJECT_ROOT)


class ProjectRAG:
    """RAG по документации проекта (коллекция project_docs).

    Attributes:
        vector_store: VectorStore (Chroma, cosine) — тот же экземпляр,
            что у личной базы знаний, другая коллекция.
        embedder: OllamaEmbedder (префиксы search_document/search_query).
        collection_name: имя коллекции (по умолчанию project_docs).
    """

    COLLECTION_NAME = PROJECT_DOCS_COLLECTION

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: OllamaEmbedder,
        collection_name: str = PROJECT_DOCS_COLLECTION,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.collection_name = collection_name

    # ------------------------------------------------------------------
    # Сбор файлов
    # ------------------------------------------------------------------
    @staticmethod
    def collect_files(project_path: Union[str, Path]) -> List[Tuple[Path, str]]:
        """Собирает README + docs/*.md + конфиги (код *.py не трогаем).

        Args:
            project_path: корень проекта.

        Returns:
            Список (путь, тип) в детерминированном порядке: README →
            docs (sorted, рекурсивно) → конфиги. Несуществующий путь —
            пустой список.
        """
        root = Path(project_path)
        result: List[Tuple[Path, str]] = []
        if not root.is_dir():
            return result

        for name in README_CANDIDATES:
            readme = root / name
            if readme.is_file():
                result.append((readme, TYPE_README))
                break

        docs_dir = root / "docs"
        if docs_dir.is_dir():
            for md in sorted(docs_dir.rglob("*.md")):
                if md.is_file():
                    result.append((md, TYPE_DOCS))

        for config_name in CONFIG_FILES:
            config = root / config_name
            if config.is_file():
                result.append((config, TYPE_CONFIG))
        return result

    # ------------------------------------------------------------------
    # Индексация
    # ------------------------------------------------------------------
    def index_project(
        self,
        project_path: Union[str, Path],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ) -> Dict[str, Any]:
        """Индексирует документацию проекта в коллекцию project_docs.

        Файлы собираются collect_files, режутся chunk_structure
        (заголовки → секции), эмбеддинги генерируются одним батчем
        (embedder.embed_documents), коллекция перезаписывается целиком.

        Args:
            project_path: корень проекта (папка с README и docs/).
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
        started = time.perf_counter()
        files = self.collect_files(project_path)

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
                chunk.metadata["path"] = str(path)
                chunk.metadata["type"] = file_type
                texts.append(chunk.page_content)
                metadatas.append(dict(chunk.metadata))
            indexed_files += 1

        stored = 0
        if texts:
            embeddings = self.embedder.embed_documents(texts)
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

    # ------------------------------------------------------------------
    # Поиск
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 3) -> Dict[str, List[Any]]:
        """Ищет по документации проекта (формат ответа Chroma).

        Args:
            query: вопрос пользователя.
            top_k: число результатов.

        Returns:
            {"ids", "documents", "metadatas", "distances"} — списки по
            одному элементу на запрос (как collection.query); пустая
            коллекция — пустые списки.
        """
        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.search(
            query_embedding,
            k=max(1, int(top_k)),
            collection_name=self.collection_name,
        )

    async def search_async(self, query: str,
                           top_k: int = 3) -> Dict[str, List[Any]]:
        """Async-вариант search (сетевой вызов эмбеддера — не в UI-лупе)."""
        query_embedding = await self.embedder.embed_query_async(query)
        return self.vector_store.search(
            query_embedding,
            k=max(1, int(top_k)),
            collection_name=self.collection_name,
        )

    def search_chunks(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        """Поиск с разбором в RetrievedChunk (для промпта и источников)."""
        return self._chunks_from_result(self.search(query, top_k))

    async def search_chunks_async(self, query: str,
                                  top_k: int = 3) -> List[RetrievedChunk]:
        """Async-вариант search_chunks (используется HelpPipeline)."""
        result = await self.search_async(query, top_k)
        return self._chunks_from_result(result)

    def count(self) -> int:
        """Число чанков в коллекции project_docs."""
        return self.vector_store.count(self.collection_name)

    def clear(self) -> None:
        """Удаляет коллекцию документации проекта."""
        self.vector_store.clear(self.collection_name)

    # ------------------------------------------------------------------
    # Внутреннее
    # ------------------------------------------------------------------
    @staticmethod
    def _chunks_from_result(result: Dict[str, List[Any]]
                            ) -> List[RetrievedChunk]:
        """RetrievedChunk из ответа collection.query (как в RAGPipeline)."""
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
