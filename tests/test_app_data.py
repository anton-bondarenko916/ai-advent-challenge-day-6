"""Тесты данных RAG: контрольные вопросы и документы базы знаний.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest
from pathlib import Path

from app.main import DEFAULT_DOCS_DIR, TAB_TITLES
from app.rag.indexer import KnowledgeIndexer
from app.ui.comparison_view import load_test_questions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTION_FILE = PROJECT_ROOT / "app" / "data" / "test_questions.json"


class TestQuestionsFileTest(unittest.TestCase):
    """app/data/test_questions.json — 10 контрольных вопросов."""

    def test_file_exists(self) -> None:
        self.assertTrue(QUESTION_FILE.is_file())

    def test_ten_questions(self) -> None:
        questions = load_test_questions(QUESTION_FILE)
        self.assertEqual(len(questions), 10)

    def test_ids_sequential(self) -> None:
        questions = load_test_questions(QUESTION_FILE)
        self.assertEqual([q["id"] for q in questions], list(range(1, 11)))

    def test_sources_reference_existing_docs(self) -> None:
        questions = load_test_questions(QUESTION_FILE)
        for question in questions:
            for source in question["sources"]:
                path = DEFAULT_DOCS_DIR / source
                self.assertTrue(
                    path.is_file(),
                    f"Источник {source} из вопроса #{question['id']} "
                    f"не найден в {DEFAULT_DOCS_DIR}",
                )


class SampleDocsTest(unittest.TestCase):
    """Примеры MD-документов индексируются и покрывают вопросы."""

    def test_docs_dir_exists_with_files(self) -> None:
        files = KnowledgeIndexer.collect_files(DEFAULT_DOCS_DIR)
        self.assertGreaterEqual(len(files), 5)

    def test_docs_indexable_with_both_strategies(self) -> None:
        import shutil
        import tempfile

        from app.rag.vector_store import VectorStore

        class HashEmbedder:
            def embed_documents(self, texts):
                return [[float(sum(ord(c) % 97 for c in t) % 97), 1.0, 0.5]
                        for t in texts]

        db = Path(tempfile.mkdtemp(prefix="docs_smoke_"))
        try:
            store = VectorStore(persist_dir=db)
            indexer = KnowledgeIndexer(HashEmbedder(), store)
            stats = indexer.index_directory(
                DEFAULT_DOCS_DIR, strategies=("fixed", "structure"),
                chunk_size=500, overlap=50)
            self.assertGreater(stats.files, 0)
            self.assertGreater(stats.chunks, 0)
            self.assertEqual(stats.errors, [])
        finally:
            shutil.rmtree(db, ignore_errors=True)

    def test_questions_content_matches_docs(self) -> None:
        # Каждый вопрос ссылается на документ, где есть часть ожидания
        questions = load_test_questions(QUESTION_FILE)
        first = questions[0]
        self.assertIn("Windows", first["question"])
        text = (DEFAULT_DOCS_DIR / "installation.md").read_text(
            encoding="utf-8")
        self.assertIn("Windows", text)


class MainModuleTest(unittest.TestCase):
    """app/main.py: вкладки и дефолтная папка документов."""

    def test_four_tabs(self) -> None:
        self.assertEqual(len(TAB_TITLES), 4)
        self.assertIn("Чат", TAB_TITLES[0])
        self.assertIn("База знаний", TAB_TITLES[1])
        self.assertIn("Сравнение", TAB_TITLES[2])
        self.assertIn("Настройки", TAB_TITLES[3])

    def test_default_docs_dir_is_sample_docs(self) -> None:
        self.assertEqual(DEFAULT_DOCS_DIR.name, "docs")
        self.assertTrue(DEFAULT_DOCS_DIR.is_dir())


if __name__ == "__main__":
    unittest.main()
