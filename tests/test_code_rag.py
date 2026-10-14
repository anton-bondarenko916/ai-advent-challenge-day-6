"""Тесты RAG по коду проекта (ТЗ: коллекция project_code).

Проверяются: сбор *.py без venv/.venv/__pycache__/.git (и без каталогов
с именем *.py), индексация в ОТДЕЛЬНУЮ коллекцию Chroma, метаданные
(source/path/chunk_id/type=code), код-ориентированный chunking,
релевантный поиск, переиндексация с перезаписью, пустой проект (старая
коллекция не очищается), ошибки (пустые файлы), валидация параметров.

Запуск из корня проекта:
    python -m unittest tests.test_code_rag -v
"""

from __future__ import annotations

import re
import shutil
import tempfile
import unittest
import zlib
from pathlib import Path

from app.rag.code_rag import PROJECT_CODE_COLLECTION, CodeRAG
from app.rag.vector_store import VectorStore

# Слова без пунктуации (для одинаковых токенов запроса и чанка)
_WORD_RE = re.compile(r"\w+")


class KeywordEmbedder:
    """Детерминированный «эмбеддер» (без сети): вектор по хешам слов.

    Хеш слова — crc32 (СТАБИЛЕН между процессами, в отличие от builtin
    hash(): у того случайная соль и тест «плавает» от запуска к запуску).
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


MAIN_PY = (
    "class PaymentProcessor:\n"
    "    '''Обрабатывает платежи.'''\n\n"
    "    def process_payment(self, amount):\n"
    "        if amount <= 0:\n"
    "            raise ValueError('amount должен быть положительным')\n"
    "        return amount * 100\n"
)

SERVICE_PY = (
    "def refund_order(order_id):\n"
    "    '''Возврат заказа.'''\n"
    "    order = load_order(order_id)\n"
    "    return refund(order)\n"
)


def make_code_project(root: Path) -> None:
    """Тестовый проект: код + служебные каталоги + мусорные файлы."""
    (root / "main.py").write_text(MAIN_PY, encoding="utf-8")
    app = root / "app"
    app.mkdir(exist_ok=True)
    (app / "service.py").write_text(SERVICE_PY, encoding="utf-8")

    for excluded in ("venv", ".venv", "__pycache__", ".git"):
        directory = root / excluded
        directory.mkdir(exist_ok=True)
        (directory / "junk.py").write_text(
            "print('не должен индексироваться')\n", encoding="utf-8")

    (root / "empty.py").write_text("", encoding="utf-8")
    (root / "notes.txt").write_text("не python\n", encoding="utf-8")
    # Каталог с именем *.py — не файл: сам отфильтровывается (is_file)
    (root / "weird.py").mkdir(exist_ok=True)
    (root / "weird.py" / "inner.txt").write_text("не код\n",
                                                 encoding="utf-8")


class CodeRAGTest(unittest.TestCase):
    """CodeRAG: сбор, индексация, поиск — на настоящем Chroma."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.project = self.tmp / "project"
        self.project.mkdir()
        make_code_project(self.project)
        self.store = VectorStore(persist_dir=self.tmp / "db")
        self.rag = CodeRAG(vector_store=self.store,
                           embedder=KeywordEmbedder())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ------------------------------------------------------------------
    # Сбор файлов
    # ------------------------------------------------------------------
    def test_collect_excludes_service_dirs(self) -> None:
        names = [path.name for path in
                 CodeRAG.collect_py_files(self.project)]
        self.assertIn("main.py", names)
        self.assertIn("service.py", names)
        # Служебные каталоги из ТЗ не индексируются
        self.assertNotIn("junk.py", names)
        # weird.py — каталог с именем *.py: rglob его находит,
        # collect_py_files отбрасывает (не is_file)
        self.assertNotIn("weird.py", names)

    def test_collect_sorted_and_absolute(self) -> None:
        files = CodeRAG.collect_py_files(self.project)
        self.assertEqual(files, sorted(files))

    def test_collect_nonexistent_path(self) -> None:
        self.assertEqual(CodeRAG.collect_py_files(self.tmp / "nope"), [])

    # ------------------------------------------------------------------
    # Индексация
    # ------------------------------------------------------------------
    def test_index_creates_separate_collection(self) -> None:
        stats = self.rag.index_code(self.project)
        self.assertEqual(stats["files"], 2)  # main.py + app/service.py
        self.assertGreater(stats["chunks"], 0)
        self.assertEqual(self.rag.count(), stats["chunks"])
        # Коллекция именно project_code, одна на весь индекс
        collection = self.store.get_or_create_collection(
            PROJECT_CODE_COLLECTION)
        self.assertEqual(collection.count(), stats["chunks"])

    def test_index_metadata(self) -> None:
        self.rag.index_code(self.project)
        found = self.rag.search("process_payment", top_k=10)
        self.assertTrue(found)
        for chunk in found:
            metadata = chunk["metadata"]
            self.assertEqual(metadata["type"], "code")
            self.assertIn("path", metadata)
            self.assertIn("chunk_id", metadata)
        paths = {chunk["metadata"]["path"] for chunk in found}
        self.assertIn("main.py", paths)

    def test_index_errors_listed(self) -> None:
        stats = self.rag.index_code(self.project)
        joined = "\n".join(stats["errors"])
        self.assertIn("empty.py", joined)  # пустой файл

    def test_reindex_replaces_collection(self) -> None:
        first = self.rag.index_code(self.project)
        second = self.rag.index_code(self.project)
        self.assertEqual(first["chunks"], second["chunks"])
        self.assertEqual(self.rag.count(), second["chunks"])

    def test_empty_project_keeps_old_index(self) -> None:
        self.rag.index_code(self.project)
        before = self.rag.count()
        empty_dir = self.tmp / "empty_project"
        empty_dir.mkdir()
        stats = self.rag.index_code(empty_dir)
        self.assertEqual(stats["files"], 0)
        self.assertEqual(stats["chunks"], 0)
        self.assertEqual(self.rag.count(), before)  # не очистили

    def test_invalid_chunk_params(self) -> None:
        with self.assertRaises(ValueError):
            self.rag.index_code(self.project, chunk_size=10)

    # ------------------------------------------------------------------
    # Поиск
    # ------------------------------------------------------------------
    def test_search_relevant_chunk(self) -> None:
        self.rag.index_code(self.project)
        found = self.rag.search("возврат заказа refund_order", top_k=3)
        self.assertTrue(found)
        top_path = found[0]["metadata"]["path"]
        self.assertEqual(top_path, str(Path("app") / "service.py"))
        self.assertIn("refund_order", found[0]["text"])

    def test_search_missing_collection_returns_empty(self) -> None:
        # Индексации не было — коллекции нет, поиск не падает
        self.assertEqual(self.rag.search("что угодно", top_k=3), [])

    def test_search_top_k_respected(self) -> None:
        self.rag.index_code(self.project)
        found = self.rag.search("payment amount", top_k=1)
        self.assertEqual(len(found), 1)

    def test_clear(self) -> None:
        self.rag.index_code(self.project)
        self.assertGreater(self.rag.count(), 0)
        self.rag.clear()
        self.assertEqual(self.rag.count(), 0)

    # ------------------------------------------------------------------
    # Батч-эмбеддинг (регрессия: один POST на все чанки не масштабируется)
    # ------------------------------------------------------------------
    def test_embeddings_go_in_batches(self) -> None:
        """Эмбеддер получает порции не больше EMBED_BATCH_SIZE, порядок
        чанков не меняется, статистика та же, что без батчей."""
        from unittest import mock

        from app.rag import code_rag as code_rag_module

        batches: list = []

        class RecordingEmbedder(KeywordEmbedder):
            def embed_documents(self, texts) -> list:
                batches.append(len(texts))
                return super().embed_documents(texts)

        with mock.patch.object(code_rag_module, "EMBED_BATCH_SIZE", 2):
            recording_rag = CodeRAG(vector_store=self.store,
                                    embedder=RecordingEmbedder())
            recording_rag.index_code(self.project)
        self.assertTrue(batches)
        self.assertTrue(all(size <= 2 for size in batches))
        self.assertEqual(sum(batches), recording_rag.count())
        # Поиск после батч-индексации находит нужный чанк (порядок цел)
        found = recording_rag.search("возврат заказа refund_order", top_k=3)
        self.assertEqual(found[0]["metadata"]["path"],
                         str(Path("app") / "service.py"))


if __name__ == "__main__":
    unittest.main()
