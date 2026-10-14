"""RAG по исходному коду проекта: отдельная коллекция Chroma.

Документация проекта (project_docs, см. app/rag/project_rag.py) и код
НЕ смешиваются: *.py индексируются в отдельную коллекцию ``project_code``
того же VectorStore (Chroma, cosine).

Что индексируется: все ``*.py`` рекурсивно, КРОМЕ служебных каталогов
(venv, .venv, __pycache__, .git). Тесты проекта индексируются тоже —
для ревью PR контекст тестов полезен (что покрыто, что нет).

Chunking — код-ориентированный RecursiveCharacterTextSplitter: сначала
режем по границам классов и функций (``\\n\\nclass `` / ``\\n\\ndef ``),
потом по абзацам/строкам; chunk_size=800, overlap=100 (ТЗ).

Метаданные чанка: source (имя файла), path (путь от корня проекта),
chunk_id (сквозной), type (``code``), strategy/chunk_size/overlap.

Переиндексация перезаписывает коллекцию целиком (clear перед
index_chunks — IDs chunk_0..chunk_N-1 остаются уникальными). Эмбеддинги
генерируются ОДНИМ батчем ДО очистки: если Ollama недоступна, старая
коллекция остаётся рабочей (тот же принцип, что у ProjectRAG).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from app.rag.chunkers import _get_splitters, validate_chunking
from app.rag.embedder import OllamaEmbedder
from app.rag.vector_store import VectorStore
from src.utils import get_logger

logger = get_logger("deepseek-chat.code-rag")

# Имя коллекции кода проекта (отдельно от knowledge_base/project_docs)
PROJECT_CODE_COLLECTION = "project_code"

TYPE_CODE = "code"
STRATEGY_CODE = "code"

# Параметры chunking по умолчанию (ТЗ)
DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 100

# Разделители код-ориентированного сплиттера (ТЗ): классы и функции —
# первичные границы, затем абзацы, строки, слова
CODE_SEPARATORS: Tuple[str, ...] = (
    "\n\nclass ", "\n\ndef ", "\n\n", "\n", " ", "",
)

# Служебные каталоги, которые НЕ индексируются (ТЗ «ЧЕГО НЕ ДЕЛАТЬ»)
EXCLUDED_PARTS: Tuple[str, ...] = ("venv", ".venv", "__pycache__", ".git")

# Обрезка чанка в RAG-контексте промпта (символов)
CONTEXT_CHUNK_LIMIT = 500

# Размер батча эмбеддингов. ВСЕ чанки одним POST не отправлять: на
# реальном проекте (сотни чанков) Ollama не укладывается в таймаут
# эмбеддера → EmbedderError на старте watcher'а (найдено живым тестом)
EMBED_BATCH_SIZE = 64

NO_CONTEXT_MESSAGE = "RAG-контекст: нет релевантных файлов."


class CodeRAG:
    """RAG по коду проекта (коллекция project_code).

    Attributes:
        vector_store: VectorStore (Chroma, cosine) — тот же экземпляр,
            что у личной базы знаний и project_docs, другая коллекция.
        embedder: OllamaEmbedder (префиксы search_document/search_query).
        collection_name: имя коллекции (по умолчанию project_code).
    """

    COLLECTION_NAME = PROJECT_CODE_COLLECTION

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: OllamaEmbedder,
        collection_name: str = PROJECT_CODE_COLLECTION,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.collection_name = collection_name

    # ------------------------------------------------------------------
    # Сбор файлов
    # ------------------------------------------------------------------
    @staticmethod
    def collect_py_files(project_path: Union[str, Path]) -> List[Path]:
        """Собирает ``*.py`` проекта без служебных каталогов.

        Args:
            project_path: корень проекта.

        Returns:
            Список путей в детерминированном порядке (sorted);
            несуществующий путь — пустой список.
        """
        root = Path(project_path)
        if not root.is_dir():
            return []
        files = [
            path for path in root.rglob("*.py")
            if path.is_file()
            and not any(part in path.parts for part in EXCLUDED_PARTS)
        ]
        return sorted(files)

    # ------------------------------------------------------------------
    # Индексация
    # ------------------------------------------------------------------
    def index_code(
        self,
        project_path: Union[str, Path],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ) -> Dict[str, Any]:
        """Индексирует все ``*.py`` проекта в коллекцию project_code.

        Файлы собираются collect_py_files, режутся код-ориентированным
        сплиттером (границы классов/функций), эмбеддинги генерируются
        одним батчем, коллекция перезаписывается целиком.

        Args:
            project_path: корень проекта.
            chunk_size: максимальный размер чанка в символах (>= 50).
            overlap: перекрытие чанков в символах.

        Returns:
            Статистика: {"files", "chunks", "errors"} (errors —
            пропущенные файлы; индексация не прерывается).

        Raises:
            ValueError: некорректные chunk_size/overlap.
            app.rag.embedder.EmbedderError: Ollama недоступна (коллекция
                НЕ очищается — старые чанки остаются рабочими).
        """
        validate_chunking(chunk_size, overlap)
        started = time.perf_counter()

        _, recursive_splitter_cls = _get_splitters()
        splitter = recursive_splitter_cls(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=list(CODE_SEPARATORS),
        )

        root = Path(project_path)
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        errors: List[str] = []
        indexed_files = 0
        for path in self.collect_py_files(project_path):
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
            except ValueError:  # путь вне корня (симлинк) — храним как есть
                relative = str(path)

            for chunk in splitter.create_documents([text]):
                metadatas.append({
                    "source": path.name,
                    "path": relative,
                    "chunk_id": len(texts),
                    "type": TYPE_CODE,
                    "strategy": STRATEGY_CODE,
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                })
                texts.append(chunk.page_content)
            indexed_files += 1

        stored = 0
        if texts:
            # Батчами по EMBED_BATCH_SIZE: один POST на всё не масштабируется
            embeddings: List[List[float]] = []
            for start in range(0, len(texts), EMBED_BATCH_SIZE):
                embeddings.extend(
                    self.embedder.embed_documents(
                        texts[start:start + EMBED_BATCH_SIZE]))
                logger.info(
                    "Эмбеддинг кода: %d/%d",
                    min(start + EMBED_BATCH_SIZE, len(texts)), len(texts))
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
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Ищет релевантные чанки кода.

        Args:
            query: поисковый запрос (например, заголовок PR).
            top_k: число результатов.

        Returns:
            Список {"text", "metadata", "distance"} — удобно для
            сборки RAG-контекста; пустая/отсутствующая коллекция —
            пустой список.
        """
        query_embedding = self.embedder.embed_query(query)
        result = self.vector_store.search(
            query_embedding,
            k=max(1, int(top_k)),
            collection_name=self.collection_name,
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        chunks: List[Dict[str, Any]] = []
        for index, text in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = distances[index] if index < len(distances) else 0.0
            chunks.append({
                "text": text,
                "metadata": metadata or {},
                "distance": float(distance) if distance is not None else 0.0,
            })
        return chunks

    def count(self) -> int:
        """Число чанков в коллекции project_code."""
        return self.vector_store.count(self.collection_name)

    def clear(self) -> None:
        """Удаляет коллекцию кода проекта."""
        self.vector_store.clear(self.collection_name)
