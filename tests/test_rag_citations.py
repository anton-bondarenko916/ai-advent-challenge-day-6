"""Тесты цитирования (app/rag/citations.py) и answer_with_citations.

ТЗ ai_1_23: JSON mode (answer/sources/quotes), дословная валидация
цитат, режим «Не знаю» при среднем score < 0.6 (LLM не вызывается),
деградация при ошибке JSON-вызова.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from app.agent import RAGChatAgent
from app.rag.citations import (
    ABSTAIN_ANSWER,
    ABSTAIN_THRESHOLD,
    CITATION_MAX_TOKENS,
    CITATION_TEMPERATURE,
    CitationAnswerer,
    CitationError,
    LLM_DONT_KNOW,
    abstain_result,
    build_citation_prompt,
    mock_citation,
    normalize_citation,
    parse_citation_json,
    should_abstain,
    validate_quotes,
)
from app.rag.retriever import RetrievedChunk
from tests.test_app_agent import (
    COMPARISON_CONFIG,
    MODEL,
    FakeEmbedder,
    FakeStore,
    make_agent,
)


def run(coroutine):
    return asyncio.run(coroutine)


CHUNKS = [
    RetrievedChunk(
        text="Скачайте установщик с официального сайта.",
        source="installation.md", title="Установка",
        section="## Windows", strategy="structure",
        distance=0.12, chunk_id=12,
    ),
    RetrievedChunk(
        text="Запустите ollama serve перед индексацией.",
        source="troubleshooting.md", title="Неполадки",
        section="## Ollama", strategy="fixed",
        distance=0.34, chunk_id=13,
    ),
]

METADATAS = [
    {"source": "installation.md", "chunk_id": 12},
    {"source": "troubleshooting.md", "chunk_id": 13},
]


class ShouldAbstainTest(unittest.TestCase):
    """Гейт «Не знаю»: порог 0.6 для nomic-embed-text (не 0.7)."""

    def test_threshold_calibrated_for_nomic(self) -> None:
        self.assertAlmostEqual(ABSTAIN_THRESHOLD, 0.6)

    def test_empty_scores_abstain(self) -> None:
        self.assertTrue(should_abstain([]))

    def test_below_threshold_abstain(self) -> None:
        self.assertTrue(should_abstain([0.5, 0.55]))

    def test_above_threshold_answer(self) -> None:
        self.assertFalse(should_abstain([0.88, 0.66]))

    def test_boundary_included(self) -> None:
        # 0.6 не меньше порога 0.6 — воздержания нет
        self.assertFalse(should_abstain([0.6]))
        self.assertTrue(should_abstain([0.5999]))

    def test_custom_threshold(self) -> None:
        self.assertTrue(should_abstain([0.8], threshold=0.9))

    def test_zero_scores_abstain(self) -> None:
        self.assertTrue(should_abstain([0.0, 0.0]))


class ValidateQuotesTest(unittest.TestCase):
    """Дословная проверка цитат по чанкам."""

    def test_valid_exact_quote_marked_true(self) -> None:
        result = validate_quotes(
            {"quotes": [{"chunk_id": 12,
                         "text": "Скачайте установщик с официального сайта."}]},
            [chunk.text for chunk in CHUNKS], METADATAS)
        self.assertTrue(result["quotes"][0]["valid"])

    def test_invalid_quote_marked_false_and_kept(self) -> None:
        result = validate_quotes(
            {"quotes": [{"chunk_id": 12, "text": "Придуманная цитата."}]},
            [chunk.text for chunk in CHUNKS], METADATAS)
        quote = result["quotes"][0]
        self.assertFalse(quote["valid"])
        self.assertEqual(quote["text"], "Придуманная цитата.")

    def test_unknown_chunk_id_invalid(self) -> None:
        result = validate_quotes(
            {"quotes": [{"chunk_id": 999, "text": "Скачайте установщик."}]},
            [chunk.text for chunk in CHUNKS], METADATAS)
        self.assertFalse(result["quotes"][0]["valid"])

    def test_whitespace_differences_still_verbatim(self) -> None:
        # LLM перенёс строку — цитата остаётся дословной
        result = validate_quotes(
            {"quotes": [{"chunk_id": 12,
                         "text": "Скачайте установщик\nс официального сайта."}]},
            ["Скачайте установщик\nс официального сайта."],
            [{"chunk_id": 12}])
        self.assertTrue(result["quotes"][0]["valid"])

    def test_string_chunk_id_coerced(self) -> None:
        result = validate_quotes(
            {"quotes": [{"chunk_id": "12", "text": "Скачайте установщик"}]},
            [chunk.text for chunk in CHUNKS], METADATAS)
        self.assertTrue(result["quotes"][0]["valid"])
        self.assertEqual(result["quotes"][0]["chunk_id"], 12)

    def test_empty_quote_invalid(self) -> None:
        result = validate_quotes(
            {"quotes": [{"chunk_id": 12, "text": "   "}]},
            [chunk.text for chunk in CHUNKS], METADATAS)
        self.assertFalse(result["quotes"][0]["valid"])

    def test_missing_quotes_gives_empty_list(self) -> None:
        result = validate_quotes({}, [chunk.text for chunk in CHUNKS],
                                 METADATAS)
        self.assertEqual(result["quotes"], [])

    def test_non_dict_quotes_skipped(self) -> None:
        result = validate_quotes(
            {"quotes": ["мусор", {"chunk_id": 12, "text": "Скачайте"}]},
            [chunk.text for chunk in CHUNKS], METADATAS)
        self.assertEqual(len(result["quotes"]), 1)
        self.assertTrue(result["quotes"][0]["valid"])

    def test_mixed_quotes(self) -> None:
        result = validate_quotes(
            {"quotes": [
                {"chunk_id": 12, "text": "Скачайте установщик"},
                {"chunk_id": 13, "text": "нет такой фразы в чанке"},
            ]},
            [chunk.text for chunk in CHUNKS], METADATAS)
        self.assertTrue(result["quotes"][0]["valid"])
        self.assertFalse(result["quotes"][1]["valid"])


class ParseCitationJsonTest(unittest.TestCase):
    """Разбор JSON-ответа модели (прямой, в ограждении, с прозой)."""

    def test_plain_json(self) -> None:
        parsed = parse_citation_json(
            '{"answer": "Ответ", "sources": [], "quotes": []}')
        self.assertEqual(parsed["answer"], "Ответ")

    def test_code_fence_stripped(self) -> None:
        raw = '```json\n{"answer": "Ответ", "sources": [], "quotes": []}\n```'
        self.assertEqual(parse_citation_json(raw)["answer"], "Ответ")

    def test_prose_around_json(self) -> None:
        raw = 'Вот ответ: {"answer": "Ответ", "sources": [], "quotes": []} — готово'
        self.assertEqual(parse_citation_json(raw)["answer"], "Ответ")

    def test_broken_json_raises(self) -> None:
        with self.assertRaises(CitationError):
            parse_citation_json("совсем не JSON")

    def test_non_object_raises(self) -> None:
        with self.assertRaises(CitationError):
            parse_citation_json('["массив"]')

    def test_empty_raises(self) -> None:
        with self.assertRaises(CitationError):
            parse_citation_json("")


class NormalizeCitationTest(unittest.TestCase):
    """Приведение JSON к контракту answer/sources/quotes."""

    def test_fields_coerced(self) -> None:
        parsed = normalize_citation({
            "answer": " Ответ ",
            "sources": [{"chunk_id": "12", "source": "a.md"}],
            "quotes": [{"chunk_id": 12, "text": "цитата"},
                       "мусор"],
        })
        self.assertEqual(parsed["answer"], "Ответ")
        self.assertEqual(parsed["sources"], [{"chunk_id": "12",
                                              "source": "a.md"}])
        self.assertEqual(parsed["quotes"], [{"chunk_id": 12,
                                             "text": "цитата"}])

    def test_missing_fields(self) -> None:
        parsed = normalize_citation({})
        self.assertEqual(parsed, {"answer": "", "sources": [], "quotes": []})


class BuildCitationPromptTest(unittest.TestCase):
    """Промпт JSON-режима: слово «json», контекст, вопрос, шапки чанков."""

    def test_prompt_contains_contract(self) -> None:
        prompt = build_citation_prompt("Как установить?", CHUNKS)
        self.assertIn("JSON", prompt)
        self.assertIn('"answer"', prompt)
        self.assertIn('"sources"', prompt)
        self.assertIn('"quotes"', prompt)
        self.assertIn('answer: "Не знаю"', prompt)
        self.assertIn("ДОСЛОВНЫЕ", prompt)
        self.assertIn("Как установить?", prompt)

    def test_prompt_contains_chunk_headers(self) -> None:
        prompt = build_citation_prompt("Как установить?", CHUNKS)
        self.assertIn("[chunk_id=12 | installation.md | ## Windows]", prompt)
        self.assertIn("[chunk_id=13 | troubleshooting.md | ## Ollama]", prompt)
        self.assertIn("Скачайте установщик с официального сайта.", prompt)


class CitationAnswererTest(unittest.TestCase):
    """JSON-mode вызов: response_format, температура, парсинг, ошибки."""

    def make_answerer(self, contents=None, fail=False):
        """Фейковый клиент с очередью ответов и finish_reason."""
        captured = []

        def create(**kwargs):
            captured.append(kwargs)
            if fail:
                raise RuntimeError("соединение оборвано")
            content, finish = contents[min(len(captured) - 1,
                                           len(contents) - 1)]
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish)])

        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)))
        answerer = CitationAnswerer(api_key="test", client=client)
        answerer.captured = captured
        return answerer

    def test_request_uses_json_object_format(self) -> None:
        answerer = self.make_answerer(
            contents=[('{"answer": "Ответ", "sources": [], "quotes": []}',
                       "stop")])
        parsed = answerer.answer("Как установить?", CHUNKS)
        self.assertEqual(parsed["answer"], "Ответ")
        captured = answerer.captured[0]
        # ТЗ: response_format гарантирует валидный JSON
        self.assertEqual(captured["response_format"],
                         {"type": "json_object"})
        self.assertAlmostEqual(captured["temperature"], 0.2)
        self.assertEqual(captured["temperature"], CITATION_TEMPERATURE)
        # 2000 мало: thinking-модель тратит бюджет на reasoning и
        # JSON обрезался до пустого ответа (урок NEWS_SUMMARY_MAX_TOKENS)
        self.assertGreaterEqual(captured["max_tokens"], 8000)
        self.assertIn("JSON", captured["messages"][0]["content"])
        self.assertIn("Как установить?", captured["messages"][0]["content"])

    def test_fenced_json_parsed(self) -> None:
        answerer = self.make_answerer(contents=[(
            '```json\n{"answer": "Ответ", "sources": [], "quotes": []}\n```',
            "stop")])
        self.assertEqual(answerer.answer("Вопрос", CHUNKS)["answer"], "Ответ")

    def test_api_error_raises_citation_error(self) -> None:
        answerer = self.make_answerer(fail=True)
        with self.assertRaises(CitationError):
            answerer.answer("Вопрос", CHUNKS)

    def test_empty_content_retried_with_double_budget(self) -> None:
        answerer = self.make_answerer(contents=[
            ("", "length"),
            ('{"answer": "Ответ", "sources": [], "quotes": []}', "stop"),
        ])
        parsed = answerer.answer("Вопрос", CHUNKS)
        self.assertEqual(parsed["answer"], "Ответ")
        self.assertEqual(len(answerer.captured), 2)
        self.assertEqual(answerer.captured[0]["max_tokens"], 8000)
        self.assertEqual(answerer.captured[1]["max_tokens"], 16000)

    def test_truncated_json_retried(self) -> None:
        # finish_reason=length с недокрученным JSON — тоже повод повторить
        answerer = self.make_answerer(contents=[
            ('{"answer": "Об', "length"),
            ('{"answer": "Ответ", "sources": [], "quotes": []}', "stop"),
        ])
        self.assertEqual(answerer.answer("Вопрос", CHUNKS)["answer"], "Ответ")
        self.assertEqual(len(answerer.captured), 2)

    def test_double_empty_raises(self) -> None:
        answerer = self.make_answerer(contents=[("", "length"), ("", "length")])
        with self.assertRaises(CitationError):
            answerer.answer("Вопрос", CHUNKS)

    def test_broken_json_raises(self) -> None:
        answerer = self.make_answerer(
            contents=[("это не json", "stop")])
        with self.assertRaises(CitationError):
            answerer.answer("Вопрос", CHUNKS)

    def test_mock_mode_builds_verbatim_quotes(self) -> None:
        answerer = CitationAnswerer(api_key="test", mock_mode=True)
        parsed = run(answerer.answer_async("Как установить?", CHUNKS))
        self.assertEqual(len(parsed["sources"]), 2)
        self.assertEqual(parsed["sources"][0]["chunk_id"], 12)
        self.assertEqual(parsed["sources"][0]["source"], "installation.md")
        # Цитаты — точные подстроки чанков (с точностью до пробелов)
        validated = validate_quotes(
            {"quotes": parsed["quotes"]},
            [chunk.text for chunk in CHUNKS], METADATAS)
        self.assertTrue(all(q["valid"] for q in validated["quotes"]))

    def test_mock_no_chunks_dont_know(self) -> None:
        parsed = mock_citation("Вопрос", [])
        self.assertEqual(parsed["answer"], LLM_DONT_KNOW)
        self.assertEqual(parsed["sources"], [])


class MockCitationTest(unittest.TestCase):
    """mock_citation: валидный JSON из чанков."""

    def test_builds_sources_and_quotes(self) -> None:
        parsed = mock_citation("Как установить?", CHUNKS)
        self.assertEqual(parsed["sources"][0],
                         {"chunk_id": 12, "source": "installation.md",
                          "section": "## Windows"})
        self.assertTrue(parsed["quotes"])
        for quote in parsed["quotes"]:
            self.assertIn("chunk_id", quote)
            self.assertTrue(quote["text"])


class AbstainResultTest(unittest.TestCase):
    """Заготовка «Не знаю»: пустые sources/quotes, abstained=True."""

    def test_fields(self) -> None:
        result = abstain_result("низкий score")
        self.assertEqual(result["answer"], ABSTAIN_ANSWER)
        self.assertEqual(result["text"], ABSTAIN_ANSWER)
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["quotes"], [])
        self.assertTrue(result["abstained"])
        self.assertEqual(result["abstain_reason"], "низкий score")
        self.assertIsNone(result["error"])


class FakeCitationAnswerer:
    """Ответчик-заглушка для answer_with_citations."""

    def __init__(self, parsed=None, fail=False) -> None:
        self.parsed = parsed if parsed is not None else {
            "answer": "Чанк про установку Windows.",
            "sources": [{"chunk_id": 1, "source": "installation.md",
                         "section": "## Windows"}],
            "quotes": [{"chunk_id": 1,
                        "text": "Чанк про установку Windows."}],
        }
        self.fail = fail
        self.calls: list = []

    async def answer_async(self, question, chunks):
        self.calls.append((question, list(chunks)))
        if self.fail:
            raise CitationError("DeepSeek недоступен")
        return dict(self.parsed)


class AnswerWithCitationsTest(unittest.TestCase):
    """answer_with_citations: гейт, JSON mode, валидация, деградация."""

    def make_agent(self, store=None, embedder=None, answerer=None):
        agent = make_agent(rag_enabled=True, store=store,
                           embedder=embedder)
        agent.citation_answerer = answerer or FakeCitationAnswerer()
        return agent

    def test_happy_path_sources_quotes_validated(self) -> None:
        # scores 0.88/0.66, среднее 0.77 ≥ 0.6 — LLM вызывается
        answerer = FakeCitationAnswerer(parsed={
            "answer": "Установка на Windows: Чанк про установку Windows.",
            "sources": [{"chunk_id": 1, "source": "installation.md",
                         "section": "## Windows"}],
            "quotes": [
                {"chunk_id": 1, "text": "Чанк про установку Windows."},
                {"chunk_id": 2, "text": "выдуманная цитата"},
            ],
        })
        agent = self.make_agent(answerer=answerer)
        result = run(agent.answer_with_citations(
            "Как установить приложение на Windows?",
            rag_mode="basic"))
        self.assertFalse(result["abstained"])
        self.assertEqual(len(answerer.calls), 1)
        self.assertEqual(result["answer"],
                         "Установка на Windows: Чанк про установку Windows.")
        self.assertEqual(result["text"], result["answer"])
        # Цитаты валидированы: дословная ✅, выдуманная ❌
        self.assertTrue(result["quotes"][0]["valid"])
        self.assertFalse(result["quotes"][1]["valid"])
        # Источники: метаданные поиска + флаг cited
        self.assertEqual(len(result["sources"]), 2)
        self.assertEqual(result["sources"][0]["chunk_id"], 1)
        self.assertTrue(result["sources"][0]["cited"])
        self.assertFalse(result["sources"][1]["cited"])
        self.assertEqual(result["citations"][0]["chunk_id"], 1)
        self.assertTrue(result["rag_used"])
        self.assertIsNone(result["error"])
        self.assertAlmostEqual(result["retrieval"]["avg_score"], 0.77,
                               places=4)

    def test_low_score_abstains_without_llm_call(self) -> None:
        # distances 0.45/0.5 → scores 0.55/0.5, среднее 0.525 < 0.6
        store = FakeStore(result={
            "ids": [["chunk_1", "chunk_2"]],
            "documents": [["Чанк про установку Windows.",
                           "Чанк про Ollama."]],
            "metadatas": [[{"source": "installation.md", "chunk_id": 1},
                           {"source": "troubleshooting.md",
                            "chunk_id": 2}]],
            "distances": [[0.45, 0.5]],
        })
        answerer = FakeCitationAnswerer()
        agent = self.make_agent(store=store, answerer=answerer)
        result = run(agent.answer_with_citations("Вопрос?", rag_mode="basic"))
        self.assertTrue(result["abstained"])
        # Гейт ДО вызова LLM: ответчик не дёргался
        self.assertEqual(answerer.calls, [])
        self.assertEqual(result["answer"], ABSTAIN_ANSWER)
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["quotes"], [])
        self.assertIn("ниже порога", result["abstain_reason"])
        self.assertTrue(result["rag_used"])

    def test_embedder_fail_abstains_with_reason(self) -> None:
        embedder = FakeEmbedder()
        embedder.fail = True
        answerer = FakeCitationAnswerer()
        agent = self.make_agent(embedder=embedder, answerer=answerer)
        result = run(agent.answer_with_citations("Вопрос?", rag_mode="basic"))
        self.assertTrue(result["abstained"])
        self.assertEqual(answerer.calls, [])
        self.assertIn("Ollama недоступна", result["rag_error"])
        self.assertIn("Ollama недоступна", result["abstain_reason"])

    def test_empty_retrieval_abstains(self) -> None:
        store = FakeStore(result={"ids": [[]], "documents": [[]],
                                  "metadatas": [[]], "distances": [[]]})
        answerer = FakeCitationAnswerer()
        agent = self.make_agent(store=store, answerer=answerer)
        result = run(agent.answer_with_citations("Вопрос?", rag_mode="basic"))
        self.assertTrue(result["abstained"])
        self.assertEqual(answerer.calls, [])
        self.assertEqual(result["abstain_reason"], "чанки не найдены")

    def test_citation_error_degrades_not_crashes(self) -> None:
        answerer = FakeCitationAnswerer(fail=True)
        agent = self.make_agent(answerer=answerer)
        result = run(agent.answer_with_citations(
            "Как установить приложение на Windows?", rag_mode="basic"))
        self.assertFalse(result["abstained"])
        self.assertIn("DeepSeek недоступен", result["citation_error"])
        # Источники поиска сохранились, цитат нет
        self.assertEqual(len(result["sources"]), 2)
        self.assertEqual(result["quotes"], [])

    def test_llm_dont_know_maps_to_abstain(self) -> None:
        answerer = FakeCitationAnswerer(parsed={
            "answer": LLM_DONT_KNOW, "sources": [], "quotes": []})
        agent = self.make_agent(answerer=answerer)
        result = run(agent.answer_with_citations(
            "Как установить приложение на Windows?", rag_mode="basic"))
        self.assertTrue(result["abstained"])
        self.assertEqual(result["abstain_reason"],
                         "модель не нашла ответа в контексте")
        self.assertTrue(result["rag_used"])

    def test_llm_empty_payload_maps_to_abstain(self) -> None:
        answerer = FakeCitationAnswerer(parsed={"answer": "", "sources": [],
                                                "quotes": []})
        agent = self.make_agent(answerer=answerer)
        result = run(agent.answer_with_citations(
            "Как установить приложение на Windows?", rag_mode="basic"))
        self.assertTrue(result["abstained"])

    def test_mock_answerer_end_to_end(self) -> None:
        # Полный офлайн-путь: mock-ответчик → валидные цитаты
        agent = self.make_agent(
            answerer=CitationAnswerer(api_key="test", mock_mode=True))
        result = run(agent.answer_with_citations(
            "Как установить приложение на Windows?",
            rag_mode="rewrite_filter",
            retrieval_params={"top_k": 20, "top_k_final": 2,
                              "threshold": 0.6}))
        self.assertFalse(result["abstained"])
        self.assertTrue(result["quotes"])
        self.assertTrue(all(q["valid"] for q in result["quotes"]))
        self.assertEqual(result["mode"], "rewrite_filter")

    def test_answer_once_still_plain_text(self) -> None:
        # Старый путь сравнения (вкладка «Сравнение») не задет
        agent = self.make_agent()
        agent.bind_config(dict(COMPARISON_CONFIG))
        result = run(agent.answer_once("Вопрос", rag_enabled=True,
                                       rag_mode="basic"))
        self.assertNotIn("quotes", result)
        self.assertNotIn("abstained", result)
        self.assertTrue(result["text"])


class FlakyAnswerer:
    """Ответчик с очередью ответов: флап модели (ответ без цитат)."""

    def __init__(self, payloads) -> None:
        self.payloads = list(payloads)
        self.calls = 0

    async def answer_async(self, question, chunks):
        index = min(self.calls, len(self.payloads) - 1)
        self.calls += 1
        return dict(self.payloads[index])


class AnswerRetryTest(unittest.TestCase):
    """Повторная попытка, когда модель прислала ответ без цитат."""

    WITH_QUOTES = {
        "answer": "Чанк про установку Windows.",
        "sources": [{"chunk_id": 1, "source": "installation.md"}],
        "quotes": [{"chunk_id": 1, "text": "Чанк про установку Windows."}],
    }
    WITHOUT_QUOTES = {
        "answer": "Чанк про установку Windows.",
        "sources": [],
        "quotes": [],
    }

    def make_agent(self, answerer):
        agent = make_agent(rag_enabled=True)
        agent.citation_answerer = answerer
        return agent

    def test_retry_recovers_quotes(self) -> None:
        answerer = FlakyAnswerer([self.WITHOUT_QUOTES, self.WITH_QUOTES])
        agent = self.make_agent(answerer)
        result = run(agent.answer_with_citations(
            "Как установить приложение на Windows?", rag_mode="basic"))
        self.assertEqual(answerer.calls, 2)
        self.assertFalse(result["abstained"])
        self.assertEqual(len(result["quotes"]), 1)
        self.assertTrue(result["quotes"][0]["valid"])

    def test_second_empty_try_is_kept(self) -> None:
        answerer = FlakyAnswerer([self.WITHOUT_QUOTES])
        agent = self.make_agent(answerer)
        result = run(agent.answer_with_citations(
            "Как установить приложение на Windows?", rag_mode="basic"))
        # Ровно одна повторная попытка, дальше — как есть (проверка ❌)
        self.assertEqual(answerer.calls, 2)
        self.assertFalse(result["abstained"])
        self.assertEqual(result["quotes"], [])
        self.assertTrue(result["answer"])

    def test_good_answer_not_retried(self) -> None:
        answerer = FlakyAnswerer([self.WITH_QUOTES])
        agent = self.make_agent(answerer)
        run(agent.answer_with_citations(
            "Как установить приложение на Windows?", rag_mode="basic"))
        self.assertEqual(answerer.calls, 1)

    def test_dont_know_not_retried(self) -> None:
        # Честное «Не знаю» — не повод для повторного вызова
        answerer = FlakyAnswerer([{"answer": LLM_DONT_KNOW, "sources": [],
                                   "quotes": []}])
        agent = self.make_agent(answerer)
        result = run(agent.answer_with_citations(
            "Как установить приложение на Windows?", rag_mode="basic"))
        self.assertEqual(answerer.calls, 1)
        self.assertTrue(result["abstained"])


class WithConfigCitationAnswererTest(unittest.TestCase):
    """with_config: CitationAnswerer настроен из конфига (mock offline)."""

    def test_with_config_creates_citation_answerer(self) -> None:
        config = dict(COMPARISON_CONFIG)
        agent = RAGChatAgent.with_config(config)
        self.assertIsNotNone(agent.citation_answerer)
        self.assertTrue(agent.citation_answerer.mock_mode)
        self.assertEqual(agent.citation_answerer.model, MODEL)
        self.assertAlmostEqual(
            agent.citation_answerer.temperature, CITATION_TEMPERATURE)
        self.assertEqual(
            agent.citation_answerer.max_tokens, CITATION_MAX_TOKENS)

    def test_default_threshold_not_seventy(self) -> None:
        # ТЗ: порог воздержания 0.6, а не 0.7
        self.assertAlmostEqual(ABSTAIN_THRESHOLD, 0.6)
        self.assertLess(ABSTAIN_THRESHOLD, 0.7)


if __name__ == "__main__":
    unittest.main()
