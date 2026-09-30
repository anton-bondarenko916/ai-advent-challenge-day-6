"""Тесты QueryRewriter (app/rag/rewriter.py): DeepSeek-переформулировка.

Все тесты офлайн: SDK-клиент подменяется фейковым объектом.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import unittest

from app.rag.rewriter import (
    REWRITE_MAX_TOKENS,
    REWRITE_PROMPT,
    REWRITE_TEMPERATURE,
    QueryRewriter,
    RewriteError,
)


class FakeCompletions:
    """Подмена client.chat.completions.create (запоминает запрос)."""

    def __init__(self, text: str = "", error: Exception = None) -> None:
        self.text = text
        self.error = error
        self.calls: list = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        message = type("Message", (), {"content": self.text})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class FakeClient:
    """Объект с интерфейсом openai.OpenAI (chat.completions)."""

    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = type("Chat", (), {})()
        self.chat.completions = completions


def make_rewriter(text="", error=None, **kwargs) -> tuple:
    """QueryRewriter с фейковым клиентом (клиент возвращается тоже)."""
    completions = FakeCompletions(text=text, error=error)
    rewriter = QueryRewriter(
        api_key="test-key", client=FakeClient(completions), **kwargs)
    return rewriter, completions


class RewriteTest(unittest.TestCase):
    """Переформулировка вопроса."""

    def test_returns_stripped_rewrite(self) -> None:
        rewriter, completions = make_rewriter(
            "  установка приложения Windows пошагово  \n")
        self.assertEqual(rewriter.rewrite("Как установить приложение?"),
                         "установка приложения Windows пошагово")

    def test_prompt_contains_question(self) -> None:
        rewriter, completions = make_rewriter("переформулировка")
        rewriter.rewrite("Как установить приложение?")
        messages = completions.calls[0]["messages"]
        self.assertEqual(messages[0]["content"], REWRITE_PROMPT.format(
            question="Как установить приложение?"))
        self.assertIn("Как установить приложение?",
                      messages[0]["content"])

    def test_model_temperature_max_tokens(self) -> None:
        rewriter, completions = make_rewriter("ок")
        rewriter.rewrite("вопрос")
        call = completions.calls[0]
        self.assertEqual(call["model"], "deepseek-v4-flash")
        self.assertEqual(call["temperature"], REWRITE_TEMPERATURE)
        self.assertEqual(call["max_tokens"], REWRITE_MAX_TOKENS)

    def test_multiline_answer_collapsed(self) -> None:
        rewriter, _ = make_rewriter("первая строка\n\nвторая строка")
        self.assertEqual(rewriter.rewrite("вопрос"),
                         "первая строка вторая строка")

    def test_empty_answer_falls_back_to_question(self) -> None:
        # Пустой ответ модели (бюджет ушёл в reasoning) — не ошибка:
        # поиск по исходному вопросу лучше падения режима
        rewriter, _ = make_rewriter("")
        self.assertEqual(rewriter.rewrite("Исходный вопрос?"),
                         "Исходный вопрос?")

    def test_network_error_raises(self) -> None:
        rewriter, _ = make_rewriter(error=RuntimeError("connection refused"))
        with self.assertRaises(RewriteError):
            rewriter.rewrite("вопрос")

    def test_empty_question_short_circuits(self) -> None:
        rewriter, completions = make_rewriter("что-то")
        self.assertEqual(rewriter.rewrite("   "), "")
        self.assertEqual(completions.calls, [])


class MockModeTest(unittest.TestCase):
    """Mock-режим: сеть не используется."""

    def test_mock_returns_original(self) -> None:
        rewriter, completions = make_rewriter(
            "подделка", mock_mode=True)
        self.assertEqual(rewriter.rewrite("Как установить приложение?"),
                         "Как установить приложение?")
        self.assertEqual(completions.calls, [])

    def test_mock_async(self) -> None:
        rewriter, completions = make_rewriter(mock_mode=True)

        async def run() -> None:
            result = await rewriter.rewrite_async("Вопрос про Ollama?")
            self.assertEqual(result, "Вопрос про Ollama?")

        asyncio.run(run())
        self.assertEqual(completions.calls, [])


class RewriteAsyncTest(unittest.TestCase):
    """Async-обёртка."""

    def test_rewrite_async(self) -> None:
        rewriter, _ = make_rewriter("поисковый вопрос")

        async def run() -> None:
            result = await rewriter.rewrite_async("Какой вопрос?")
            self.assertEqual(result, "поисковый вопрос")

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
