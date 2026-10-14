"""Тесты обёртки Inspectra (src/inspectra_runner.py) — офлайн.

subprocess.run подменяется (реальный CLI не зовётся), CodeRAG — заглушка
с фиксированными результатами поиска. Проверяются: команда запуска
(--pr/--provider/--model/--post-comment/--pr-summary/--config, cwd),
содержимое промпта .inspectra.yml (RAG-чанки + фокус «баги и edge
cases» + severity), удаление временного конфига, восстановление
пользовательского .inspectra.yml, деградация при пустом RAG и ошибке
RAG, обработка ошибок запуска (нет CLI, таймаут, ненулевой код
возврата) — ни одна не пробрасывается.

Запуск из корня проекта:
    python -m unittest tests.test_inspectra_runner -v
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from src.inspectra_runner import (
    CONFIG_NAME,
    NO_CONTEXT_MESSAGE,
    InspectraRunner,
)
from app.rag.embedder import EmbedderError

FAKE_CHUNK = {
    "text": "def process_payment(amount):\n    if amount <= 0:\n"
            "        raise ValueError('positive required')\n",
    "metadata": {"path": "app/payments.py", "source": "payments.py"},
    "distance": 0.2,
}


class FakeCodeRAG:
    """Заглушка CodeRAG: фиксированный результат / ошибка поиска."""

    def __init__(self, results=None, error: Exception = None) -> None:
        self.results = [dict(FAKE_CHUNK)] if results is None else results
        self.error = error
        self.calls: list = []

    def search(self, query: str, top_k: int = 5) -> list:
        self.calls.append((query, top_k))
        if self.error is not None:
            raise self.error
        return self.results[:top_k]


class InspectraRunnerTestCase(unittest.TestCase):
    """Общая база: временный проект-ревью + подмена subprocess.run."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.project = self.tmp / "reviewed_project"
        self.project.mkdir()
        self.record: dict = {}

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_runner(self, **kwargs) -> InspectraRunner:
        defaults = dict(
            project_path=self.project,
            code_rag=kwargs.pop("code_rag", FakeCodeRAG()),
        )
        defaults.update(kwargs)
        return InspectraRunner(**defaults)

    def patch_run(self, returncode: int = 0, stdout: str = "ok",
                  stderr: str = "", raise_exc: Exception = None):
        """Подмена subprocess.run: запоминает вызов, читает конфиг ДО
        его удаления, возвращает CompletedProcess / бросает исключение."""

        def fake_run(cmd, **kwargs):
            self.record["cmd"] = list(cmd)
            self.record["kwargs"] = kwargs
            if raise_exc is not None:
                raise raise_exc
            config_path = Path(cmd[cmd.index("--config") + 1])
            self.record["config_yaml"] = config_path.read_text(
                encoding="utf-8")
            return subprocess.CompletedProcess(
                cmd, returncode, stdout, stderr)

        return mock.patch("src.inspectra_runner.subprocess.run",
                          side_effect=fake_run)

    # ------------------------------------------------------------------
    # Успешное ревью
    # ------------------------------------------------------------------
    def test_review_success_command_and_cleanup(self) -> None:
        runner = self.make_runner()
        with self.patch_run():
            result = runner.review_pr(42, "Fix payment edge case")

        self.assertTrue(result["success"])
        self.assertEqual(result["returncode"], 0)
        cmd = self.record["cmd"]
        self.assertEqual(cmd[0], "inspectra")
        self.assertEqual(cmd[1], "review")
        self.assertIn("--pr", cmd)
        self.assertEqual(cmd[cmd.index("--pr") + 1], "42")
        self.assertEqual(cmd[cmd.index("--provider") + 1], "ollama")
        self.assertEqual(cmd[cmd.index("--model") + 1],
                         "qwen2.5-coder-14b-32k")
        # ТЗ: --post-comment по умолчанию НЕТ — обязан быть явным
        self.assertIn("--post-comment", cmd)
        self.assertIn("--pr-summary", cmd)
        self.assertIn("--config", cmd)
        # cwd — проект-ревью; текст/ошибки проходят насквозь
        self.assertEqual(self.record["kwargs"]["cwd"], str(self.project))
        self.assertEqual(result["stdout"], "ok")
        # Временный конфиг удалён
        self.assertFalse((self.project / CONFIG_NAME).exists())

    def test_prompt_contains_rag_chunks_and_focus(self) -> None:
        runner = self.make_runner()
        with self.patch_run():
            runner.review_pr(42, "Fix payment edge case")

        config = yaml.safe_load(self.record["config_yaml"])
        self.assertIn("prompt", config)
        prompt = config["prompt"]
        # RAG-чанк с путём попал в промпт
        self.assertIn("RAG-контекст", prompt)
        self.assertIn("app/payments.py", prompt)
        self.assertIn("process_payment", prompt)
        # Фокус: баги и edge cases, severity + рекомендация
        self.assertIn("баги", prompt.lower())
        self.assertIn("edge cases", prompt.lower())
        self.assertIn("severity", prompt.lower())
        self.assertIn("рекомендацию", prompt.lower())
        self.assertIn("#42", prompt)

    def test_rag_query_uses_pr_title(self) -> None:
        fake_rag = FakeCodeRAG()
        runner = self.make_runner(code_rag=fake_rag)
        with self.patch_run():
            runner.review_pr(7, "Add refund flow")
        self.assertEqual(fake_rag.calls[0][0], "Add refund flow")
        self.assertEqual(fake_rag.calls[0][1], runner.top_k)

    # ------------------------------------------------------------------
    # RAG-контекст: пустой / ошибочный
    # ------------------------------------------------------------------
    def test_empty_rag_results_degrade_prompt(self) -> None:
        runner = self.make_runner(code_rag=FakeCodeRAG(results=[]))
        with self.patch_run():
            result = runner.review_pr(1, "Anything")
        self.assertTrue(result["success"])
        prompt = yaml.safe_load(self.record["config_yaml"])["prompt"]
        self.assertIn(NO_CONTEXT_MESSAGE, prompt)

    def test_rag_error_does_not_break_review(self) -> None:
        runner = self.make_runner(
            code_rag=FakeCodeRAG(error=EmbedderError("Ollama недоступна")))
        with self.patch_run():
            result = runner.review_pr(1, "Anything")
        self.assertTrue(result["success"])
        prompt = yaml.safe_load(self.record["config_yaml"])["prompt"]
        self.assertIn(NO_CONTEXT_MESSAGE, prompt)

    def test_build_rag_context_formats_chunks(self) -> None:
        runner = self.make_runner()
        context = runner.build_rag_context("payment", pr_number=5)
        self.assertIn("app/payments.py", context)
        self.assertIn("process_payment", context)
        self.assertIn("(к PR #5)", context)

    def test_build_rag_context_truncates_chunk_text(self) -> None:
        long_text = "x" * 5000
        runner = self.make_runner(code_rag=FakeCodeRAG(
            results=[{"text": long_text, "metadata": {"path": "a.py"},
                      "distance": 0.5}]))
        context = runner.build_rag_context("query")
        chunk_line = context.split("--- a.py ---")[1]
        self.assertLessEqual(len(chunk_line.strip()), 501)

    # ------------------------------------------------------------------
    # Пользовательский .inspectra.yml не теряется
    # ------------------------------------------------------------------
    def test_existing_user_config_restored(self) -> None:
        user_config = self.project / CONFIG_NAME
        user_config.write_text("# my own config\nprompt: custom\n",
                               encoding="utf-8")
        runner = self.make_runner()
        with self.patch_run():
            runner.review_pr(1, "Anything")
        self.assertTrue(user_config.exists())
        self.assertEqual(
            user_config.read_text(encoding="utf-8"),
            "# my own config\nprompt: custom\n")

    def test_config_removed_when_no_user_config(self) -> None:
        runner = self.make_runner()
        with self.patch_run():
            runner.review_pr(1, "Anything")
        self.assertFalse((self.project / CONFIG_NAME).exists())

    # ------------------------------------------------------------------
    # Ошибки запуска — не пробрасываются
    # ------------------------------------------------------------------
    def test_nonzero_returncode_is_failure(self) -> None:
        runner = self.make_runner()
        with self.patch_run(returncode=1, stdout="",
                            stderr="model not found"):
            result = runner.review_pr(3, "Anything")
        self.assertFalse(result["success"])
        self.assertEqual(result["stderr"], "model not found")
        self.assertEqual(result["returncode"], 1)

    def test_timeout_is_failure(self) -> None:
        runner = self.make_runner(timeout=600)
        exc = subprocess.TimeoutExpired(cmd="inspectra review", timeout=600)
        with self.patch_run(raise_exc=exc):
            result = runner.review_pr(3, "Anything")
        self.assertFalse(result["success"])
        self.assertEqual(result["returncode"], -1)
        self.assertIn("таймаут", result["stderr"].lower())

    def test_missing_binary_is_failure(self) -> None:
        runner = self.make_runner()
        with self.patch_run(raise_exc=FileNotFoundError("inspectra")):
            result = runner.review_pr(3, "Anything")
        self.assertFalse(result["success"])
        self.assertEqual(result["returncode"], -1)
        self.assertIn("не найден", result["stderr"])

    def test_custom_binary_and_model_in_command(self) -> None:
        runner = self.make_runner(binary="C:/tools/inspectra.exe",
                                  model="custom-coder", provider="ollama")
        with self.patch_run():
            result = runner.review_pr(9, "Anything")
        self.assertTrue(result["success"])
        self.assertEqual(self.record["cmd"][0], "C:/tools/inspectra.exe")
        self.assertEqual(self.record["cmd"][
            self.record["cmd"].index("--model") + 1], "custom-coder")

    # ------------------------------------------------------------------
    # Окружение подпроцесса: Inspectra читает репозиторий из
    # GITHUB_REPOSITORY (конвенция Actions), а не из нашего GITHUB_REPO
    # ------------------------------------------------------------------
    def test_github_repository_env_set_for_pr_mode(self) -> None:
        runner = self.make_runner(repo="octocat/hello-world")
        with self.patch_run():
            runner.review_pr(5, "Anything")
        env = self.record["kwargs"]["env"]
        self.assertEqual(env["GITHUB_REPOSITORY"], "octocat/hello-world")
        # Остальное окружение не теряется
        self.assertIn("PATH", env)

    def test_no_repo_env_untouched(self) -> None:
        runner = self.make_runner()
        with self.patch_run():
            runner.review_pr(5, "Anything")
        self.assertNotIn("GITHUB_REPOSITORY", self.record["kwargs"]["env"])

    def test_subprocess_env_utf8_mode(self) -> None:
        """PYTHONUTF8=1: inspectra читает конфиг/diff не в кодировке
        локали Windows (cp1251), а в UTF-8 (регрессия «charmap codec»)."""
        runner = self.make_runner(repo="octocat/hello-world")
        with self.patch_run():
            runner.review_pr(5, "Anything")
        self.assertEqual(self.record["kwargs"]["env"]["PYTHONUTF8"], "1")

    def test_config_file_is_pure_ascii(self) -> None:
        r"""Конфиг читается ЛЮБОЙ кодировкой локали (регрессия «Config
        error: 'charmap' codec can't decode»): inspectra открывает
        .inspectra.yml без encoding= — кириллица должна быть
        \uXXXX-экранирована, а safe_load возвращает точный текст."""
        runner = self.make_runner()
        raw_bytes = None

        def fake_run(cmd, **kwargs):
            nonlocal raw_bytes
            config_path = Path(cmd[cmd.index("--config") + 1])
            raw_bytes = config_path.read_bytes()
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        with mock.patch("src.inspectra_runner.subprocess.run",
                        side_effect=fake_run):
            result = runner.review_pr(42, "Фикс платежей")  # кириллица в промпт не идёт, но RAG-чанк с ней

        self.assertTrue(result["success"])
        self.assertIsNotNone(raw_bytes)
        raw_bytes.decode("ascii")  # не бросает — файл в ASCII
        prompt = yaml.safe_load(raw_bytes.decode("ascii"))["prompt"]
        # Точный round-trip: русский текст восстановлен из экранирования
        self.assertIn("баги", prompt.lower())
        self.assertIn("process_payment", prompt)


if __name__ == "__main__":
    unittest.main()
