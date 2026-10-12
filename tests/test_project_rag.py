"""Тесты RAG по документации проекта (ТЗ: коллекция project_docs).

Проверяются: сбор файлов (README + docs/*.md рекурсивно + конфиги,
код *.py НЕ индексируется), индексация в ОТДЕЛЬНУЮ коллекцию Chroma
(личная knowledge_base не смешивается), метаданные чанков (source/
path/title/section/chunk_id/type), релевантный поиск, переиндексация
с перезаписью, ошибки (пустые/нечитаемые файлы), недоступный эмбеддер
(коллекция не очищается, EmbedderError пробрасывается).

Запуск из корня проекта:
    python -m unittest tests.test_project_rag -v
"""

from __future__ import annotations

import re
import shutil
import tempfile
import unittest
import zlib
from pathlib import Path

from app.rag.embedder import EmbedderError
from app.rag.project_rag import (
    PROJECT_DOCS_COLLECTION,
    ProjectRAG,
    default_project_path,
)
from app.rag.retriever import RetrievedChunk
from app.rag.vector_store import VectorStore

# Слова без пунктуации (для одинаковых токенов запроса и чанка)
_WORD_RE = re.compile(r"\w+")


class KeywordEmbedder:
    """Детерминированный «эмбеддер» (без сети): вектор по хешам слов.

    Хеш слова — crc32 (СТАБИЛЕН между процессами, в отличие от builtin
    hash(): у того случайная соль и тест «плавает» от запуска к запуску).
    256 измерений — коллизий между словами фикстур нет.
    """

    DIM = 256

    def _vector(self, text: str) -> list:
        vector = [0.0] * self.DIM
        for word in _WORD_RE.findall(str(text).lower()):
            vector[zlib.crc32(word.encode("utf-8")) % self.DIM] += 1.0
        return vector

    def embed_documents(self, texts) -> list:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list:
        return self._vector(text)

    async def embed_query_async(self, text: str) -> list:
        return self._vector(text)

    async def embed_documents_async(self, texts) -> list:
        return [self._vector(t) for t in texts]


class FailingEmbedder(KeywordEmbedder):
    """Эмбеддер, имитирующий недоступный Ollama."""

    def embed_documents(self, texts) -> list:
        raise EmbedderError("Ollama недоступна по адресу "
                            "http://localhost:11434")

    def embed_query(self, text: str) -> list:
        raise EmbedderError("Ollama недоступна по адресу "
                            "http://localhost:11434")


def make_project(root: Path) -> None:
    """Тестовый проект: README, docs (в т.ч. вложенные), конфиги, код."""
    (root / "README.md").write_text(
        "# Демо проект\n\n## Установка\n\nУстановка через pip install "
        "демо.\n", encoding="utf-8")
    docs = root / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "architecture.md").write_text(
        "# Архитектура\n\n## Слои\n\nАрхитектура: ядро и UI слои "
        "приложения.\n", encoding="utf-8")
    (docs / "sub").mkdir(exist_ok=True)
    (docs / "sub" / "nested.md").write_text(
        "# Вложенный документ\n\nВложенный документ про RAG "
        "поиск.\n", encoding="utf-8")
    (root / "requirements.txt").write_text(
        "flet==1.0\nchromadb\n", encoding="utf-8")
    (root / "main.py").write_text(
        "print('код не должен индексироваться')\n", encoding="utf-8")
    (root / "notes.txt").write_text("заметки мимо\n", encoding="utf-8")


class ProjectRAGTest(unittest.TestCase):
    """ProjectRAG: сбор, индексация, поиск — на настоящем Chroma."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.project = self.tmp / "project"
        self.project.mkdir()
        make_project(self.project)
        self.store = VectorStore(persist_dir=self.tmp / "db")
        self.rag = ProjectRAG(self.store, KeywordEmbedder())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ------------------------------------------------------------------
    # Сбор файлов
    # ------------------------------------------------------------------
    def test_collect_files_readme_docs_configs(self) -> None:
        files = ProjectRAG.collect_files(self.project)
        pairs = [(path.name, file_type) for path, file_type in files]
        self.assertIn(("README.md", "readme"), pairs)
        self.assertIn(("architecture.md", "docs"), pairs)
        self.assertIn(("nested.md", "docs"), pairs)
        self.assertIn(("requirements.txt", "config"), pairs)
        names = [name for name, _ in pairs]
        self.assertNotIn("main.py", names)   # код не индексируется
        self.assertNotIn("notes.txt", names)

    def test_collect_files_docs_recursive(self) -> None:
        docs = [(path.relative_to(self.project), file_type)
                for path, file_type in ProjectRAG.collect_files(self.project)
                if file_type == "docs"]
        self.assertEqual(len(docs), 2)  # architecture.md + sub/nested.md

    def test_collect_files_lowercase_readme(self) -> None:
        (self.project / "README.md").unlink()
        (self.project / "readme.md").write_text("# Другой\n",
                                                encoding="utf-8")
        files = ProjectRAG.collect_files(self.project)
        readmes = [path for path, file_type in files
                   if file_type == "readme"]
        # Ровно один README; на Windows имя не различает регистр
        self.assertEqual(len(readmes), 1)

    def test_collect_files_missing_root(self) -> None:
        self.assertEqual(
            ProjectRAG.collect_files(self.tmp / "no_such_dir"), [])

    # ------------------------------------------------------------------
    # Индексация
    # ------------------------------------------------------------------
    def test_index_project_separate_collection(self) -> None:
        stats = self.rag.index_project(self.project)
        self.assertEqual(stats["files"], 4)
        self.assertEqual(self.store.count(PROJECT_DOCS_COLLECTION),
                         stats["chunks"])
        # Личная база знаний не тронута: коллекции не смешиваются
        self.assertEqual(self.store.count("knowledge_base"), 0)

    def test_index_project_metadata(self) -> None:
        self.rag.index_project(self.project)
        result = self.store.search(
            KeywordEmbedder().embed_query("архитектура"),
            k=10, collection_name=PROJECT_DOCS_COLLECTION,
        )
        metadatas = result["metadatas"][0]
        self.assertTrue(metadatas)
        for metadata in metadatas:
            self.assertIn("source", metadata)
            self.assertIn("path", metadata)
            self.assertIn("title", metadata)
            self.assertIn("section", metadata)
            self.assertIn("chunk_id", metadata)
            self.assertIn(metadata["type"], ("readme", "docs", "config"))
        architecture = [m for m in metadatas
                        if m["source"] == "architecture.md"]
        self.assertTrue(architecture)
        self.assertEqual(architecture[0]["title"], "Архитектура")
        self.assertTrue(str(architecture[0]["path"]).endswith(
            str(Path("docs") / "architecture.md")))

    def test_reindex_replaces_collection(self) -> None:
        first = self.rag.index_project(self.project)
        second = self.rag.index_project(self.project)
        self.assertEqual(first["chunks"], second["chunks"])
        self.assertEqual(self.store.count(PROJECT_DOCS_COLLECTION),
                         second["chunks"])

    def test_empty_file_is_skipped_with_error(self) -> None:
        (self.project / "docs" / "empty.md").write_text("   \n",
                                                        encoding="utf-8")
        stats = self.rag.index_project(self.project)
        self.assertEqual(stats["files"], 4)  # empty.md не посчитан
        self.assertTrue(any("empty.md" in error
                            for error in stats["errors"]))

    def test_embedder_error_keeps_collection(self) -> None:
        self.rag.index_project(self.project)
        before = self.store.count(PROJECT_DOCS_COLLECTION)
        rag = ProjectRAG(self.store, FailingEmbedder())
        with self.assertRaises(EmbedderError):
            rag.index_project(self.project)
        self.assertEqual(self.store.count(PROJECT_DOCS_COLLECTION), before)

    # ------------------------------------------------------------------
    # Поиск
    # ------------------------------------------------------------------
    def test_search_finds_relevant_chunk(self) -> None:
        self.rag.index_project(self.project)
        chunks = self.rag.search_chunks("архитектура", top_k=2)
        self.assertTrue(chunks)
        self.assertIsInstance(chunks[0], RetrievedChunk)
        self.assertEqual(chunks[0].source, "architecture.md")

    def test_search_empty_collection(self) -> None:
        result = self.rag.search("архитектура", top_k=3)
        self.assertEqual(result["documents"], [[]])
        self.assertEqual(self.rag.search_chunks("архитектура"), [])

    def test_count_and_clear(self) -> None:
        self.rag.index_project(self.project)
        self.assertGreater(self.rag.count(), 0)
        self.rag.clear()
        self.assertEqual(self.rag.count(), 0)

    def test_search_async_variant(self) -> None:
        import asyncio

        self.rag.index_project(self.project)
        chunks = asyncio.run(self.rag.search_chunks_async("архитектура",
                                                          top_k=2))
        self.assertEqual(chunks[0].source, "architecture.md")


class DefaultPathTest(unittest.TestCase):
    """Путь проекта по умолчанию — корень этого приложения."""

    def test_default_path_is_project_root_with_readme(self) -> None:
        path = Path(default_project_path())
        self.assertTrue((path / "README.md").is_file())


if __name__ == "__main__":
    unittest.main()
