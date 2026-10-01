"""Обязательные источники, цитаты и режим «Не знаю» (ТЗ ai_1_23).

Три модуля ТЗ в одном месте:

1. Структурированный вывод LLM (JSON mode): промпт ``CITATION_PROMPT``
   и ``CitationAnswerer`` — отдельный вызов DeepSeek с
   ``response_format={"type": "json_object"}``; модель возвращает
   ``{"answer", "sources": [{"chunk_id", "source", "section"}],
   "quotes": [{"chunk_id", "text"}]}``.
2. Валидация цитат: ``validate_quotes`` проверяет дословно, что каждая
   цитата есть в своём чанке; неверная не выбрасывается, а помечается
   ``valid: False`` (цифры валидности попадают в UI и метрики).
3. Режим «Не знаю» (gating): ``should_abstain`` — средний similarity
   ниже порога → LLM НЕ вызывается вовсе, возвращается заготовка
   ``abstain_result``. Порог по умолчанию 0.6 — откалиброван для
   nomic-embed-text (0.7 слишком высокий: валидные ответы отсекаются).

Надёжность — в духе остальных ступеней RAG: ошибка JSON-вызова
(``CitationError``) не роняет ход, вызывающий код (RAGChatAgent.
answer_with_citations) деградирует до ответа с источниками без цитат.
В mock-режиме сеть не трогается: ``mock_citation`` собираает валидный
JSON из чанков (офлайн-тесты и запуск с --mock).
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from app.rag.retriever import RetrievedChunk
from src.utils import get_logger

logger = get_logger("deepseek-chat.rag.citations")

# ----------------------------------------------------------------------
# Константы (значения из ТЗ)
# ----------------------------------------------------------------------
# Порог режима «Не знаю»: средний similarity ниже — LLM не вызывается.
# 0.6 откалиброван для nomic-embed-text; 0.7 завышен.
ABSTAIN_THRESHOLD = 0.6

# Ответ-заготовка при воздержании (формулировка из ТЗ)
ABSTAIN_ANSWER = (
    "Я не знаю. В предоставленных документах нет достаточно "
    "релевантной информации. Пожалуйста, уточните вопрос."
)

# Ответ модели, когда в контексте нет ответа (инструкция в промпте)
LLM_DONT_KNOW = "Не знаю"

CITATION_TEMPERATURE = 0.2    # низкая температура — стабильный JSON
# deepseek-v4-flash — thinking-модель: часть бюджета уходит в reasoning,
# при 2000 JSON регулярно не помещался (пустой content/finish=length).
# Урок NEWS_SUMMARY_MAX_TOKENS: дефолт с запасом + повтор с удвоением.
CITATION_MAX_TOKENS = 8000
CITATION_TIMEOUT = 60.0

# Модель/базовый URL по умолчанию — как у ядра диалога
DEFAULT_CITATION_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"

# Промпт из ТЗ: слово «JSON» в тексте обязательно (иначе некоторые
# модели игнорируют response_format и отвечают прозой)
CITATION_PROMPT = """Ты — технический ассистент. Отвечай на вопрос, используя ТОЛЬКО предоставленный контекст.

Твой ответ ДОЛЖЕН быть валидным JSON-объектом со следующей структурой:

{{
  "answer": "Твой ответ здесь, основанный на контексте.",
  "sources": [
    {{"chunk_id": 12, "source": "installation.md", "section": "## Windows"}}
  ],
  "quotes": [
    {{"chunk_id": 12, "text": "Скачайте установщик с официального сайта."}}
  ]
}}

ВАЖНО:
- В поле "quotes" указывай ДОСЛОВНЫЕ фрагменты из контекста (точная копия текста).
- Каждая цитата должна быть привязана к chunk_id, из которого она взята.
- Не перефразируй цитаты.
- Если в контексте нет ответа — верни answer: "Не знаю", sources: [], quotes: [].

Контекст:
{context}

Вопрос: {question}"""


class CitationError(RuntimeError):
    """Ошибка JSON-вызова: API недоступен или ответ не разобрался."""


# ----------------------------------------------------------------------
# Модуль 3: режим «Не знаю» (gating ДО вызова LLM)
# ----------------------------------------------------------------------
def should_abstain(scores: List[float],
                   threshold: float = ABSTAIN_THRESHOLD) -> bool:
    """True, если средний score ниже порога (или скор нет вовсе).

    Порог по умолчанию 0.6 — откалиброван для nomic-embed-text.
    Граница включается: средний ровно 0.6 НЕ воздерживается.
    """
    if not scores:
        return True
    avg_score = sum(scores) / len(scores)
    return avg_score < threshold


def abstain_result(reason: str = "") -> Dict[str, Any]:
    """Заготовка ответа «Не знаю»: без источников и цитат.

    Args:
        reason: причина воздержания (диагностика для UI/логов).

    Returns:
        Словарь с answer/text (формулировка ТЗ), пустыми sources/
        citations/quotes и abstained=True.
    """
    return {
        "answer": ABSTAIN_ANSWER,
        "text": ABSTAIN_ANSWER,
        "sources": [],
        "citations": [],
        "quotes": [],
        "abstained": True,
        "abstain_reason": reason,
        "rag_used": False,
        "error": None,
    }


# ----------------------------------------------------------------------
# Модуль 1: промпт и JSON-вызов
# ----------------------------------------------------------------------
def build_citation_prompt(question: str,
                          chunks: List[RetrievedChunk]) -> str:
    """Промпт CITATION_PROMPT с контекстом из чанков.

    Каждый чанк предваряется шапкой ``[chunk_id=… | source | section]``:
    модель копирует оттуда значения для поля "sources", а не выдумывает.
    """
    blocks = []
    for chunk in chunks:
        header = (f"[chunk_id={chunk.chunk_id} | "
                  f"{chunk.source or '—'} | {chunk.section or '—'}]")
        blocks.append(f"{header}\n{chunk.text}")
    context = "\n\n".join(blocks)
    return CITATION_PROMPT.format(context=context, question=question)


def parse_citation_json(raw: str) -> Dict[str, Any]:
    """Разбирает ответ модели в JSON-объект (устойчиво к обёрткам).

    Убирает ```-ограждения; если вокруг JSON есть проза — берёт текст
    от первой ``{`` до последней ``}``.

    Raises:
        CitationError: валидного JSON-объекта в ответе нет.
    """
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    parsed = _loads_or_none(text)
    if parsed is None:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            parsed = _loads_or_none(text[start:end + 1])
    if not isinstance(parsed, dict):
        raise CitationError(
            f"Ответ модели не содержит валидного JSON: {text[:200]!r}"
        )
    return parsed


def _loads_or_none(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def normalize_citation(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Приводит разобранный JSON к контракту: answer/sources/quotes."""
    answer = str(parsed.get("answer") or "").strip()
    sources = [dict(item) for item in (parsed.get("sources") or [])
               if isinstance(item, dict)]
    quotes = []
    for item in parsed.get("quotes") or []:
        if not isinstance(item, dict):
            continue
        quotes.append({
            "chunk_id": _coerce_chunk_id(item.get("chunk_id")),
            "text": str(item.get("text") or ""),
        })
    return {"answer": answer, "sources": sources, "quotes": quotes}


def _coerce_chunk_id(raw: Any) -> Any:
    """chunk_id к int, если возможно (модель может прислать строку)."""
    try:
        return int(raw)
    except (TypeError, ValueError):
        return raw


def mock_citation(question: str,
                  chunks: List[RetrievedChunk]) -> Dict[str, Any]:
    """Офлайн-ответ CitationAnswerer (mock-режим, без сети).

    Ответ и цитаты собираются из текстов чанков: цитаты — точные
    подстроки (с точностью до пробелов), поэтому validate_quotes
    помечает их валидными. Нет чанков → «Не знаю».
    """
    if not chunks:
        return {"answer": LLM_DONT_KNOW, "sources": [], "quotes": []}
    collapsed = " ".join(chunks[0].text.split())
    answer = collapsed[:200] or LLM_DONT_KNOW
    sources = [{
        "chunk_id": chunk.chunk_id,
        "source": chunk.source,
        "section": chunk.section,
    } for chunk in chunks]
    quotes = []
    for chunk in chunks:
        text = " ".join(chunk.text.split())
        if text:
            quotes.append({"chunk_id": chunk.chunk_id,
                           "text": text[:80]})
    return {"answer": answer, "sources": sources, "quotes": quotes}


class CitationAnswerer:
    """JSON-mode вызов DeepSeek за ответом с источниками и цитатами.

    Attributes:
        model: модель служебного запроса (deepseek-v4-flash).
        base_url: базовый URL API (DeepSeek совместим с OpenAI SDK).
        temperature: 0.2 — стабильный JSON (значение из ТЗ).
        max_tokens: 2000 — с запасом, чтобы JSON не обрезался.
        mock_mode: True — сеть не используется, ответ собирается из
            чанков локально (запуск --mock и офлайн-тесты).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_CITATION_MODEL,
        temperature: float = CITATION_TEMPERATURE,
        max_tokens: int = CITATION_MAX_TOKENS,
        mock_mode: bool = False,
        timeout: float = CITATION_TIMEOUT,
        client: Optional[OpenAI] = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.mock_mode = bool(mock_mode)
        self._timeout = timeout
        # Клиент можно передать готовым (тесты — фейковый объект)
        self._client = client

    @property
    def client(self) -> OpenAI:
        """Ленивый SDK-клиент (создаётся при первом обращении)."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key or "missing-api-key",
                base_url=self.base_url,
                timeout=self._timeout,
            )
        return self._client

    # ------------------------------------------------------------------
    # Публичное API
    # ------------------------------------------------------------------
    def answer(self, question: str,
               chunks: List[RetrievedChunk]) -> Dict[str, Any]:
        """Отвечает на вопрос в JSON mode (answer/sources/quotes).

        Args:
            question: вопрос пользователя.
            chunks: чанки контекста (для шапок chunk_id/source/section).

        Returns:
            Нормализованный словарь normalize_citation.

        Raises:
            CitationError: сеть/API/дважды пустой ответ/невалидный JSON.
        """
        if self.mock_mode:
            return mock_citation(question, chunks)
        prompt = build_citation_prompt(question, chunks)
        raw, finish_reason = self._call(prompt, self.max_tokens)
        if not raw.strip() or finish_reason == "length":
            # Бюджет ушёл в reasoning либо JSON обрезан: одна повторная
            # попытка с удвоенным запасом (как у суммаризатора новостей)
            logger.warning(
                "JSON-ответ пуст или обрезан (finish_reason=%s) — "
                "повтор с удвоенным max_tokens", finish_reason)
            raw, finish_reason = self._call(prompt, self.max_tokens * 2)
        if not raw.strip():
            raise CitationError("Пустой ответ модели в JSON-режиме")
        return normalize_citation(parse_citation_json(raw))

    def _call(self, prompt: str, max_tokens: int
              ) -> tuple:
        """Один вызов API: (content, finish_reason).

        Raises:
            CitationError: ошибка сети/API.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                # Гарантирует валидный JSON на стороне API
                response_format={"type": "json_object"},
                temperature=self.temperature,
                max_tokens=int(max_tokens),
            )
        except Exception as exc:  # noqa: BLE001 — SDK бросает широкий набор
            raise CitationError(
                f"JSON-режим недоступен ({self.model}): {exc}"
            ) from exc
        choice = response.choices[0]
        return (str(choice.message.content or ""),
                getattr(choice, "finish_reason", None))

    async def answer_async(self, question: str,
                           chunks: List[RetrievedChunk]) -> Dict[str, Any]:
        """Async-вариант answer (SDK синхронный — уход в поток)."""
        return await asyncio.to_thread(self.answer, question, chunks)


# ----------------------------------------------------------------------
# Модуль 2: валидация цитат
# ----------------------------------------------------------------------
def validate_quotes(result: dict, chunks: List[str],
                    metadatas: List[dict]) -> dict:
    """Проверяет, что каждая цитата действительно есть в своём чанке.

    Цитата ищется в чанке с chunk_id из метаданных. Совпадение — точная
    подстрока; как запас — подстрока после схлопывания пробелов/переносов
    (LLM иногда переносит строки, цитата при этом остаётся дословной).
    Неверная цитата не выбрасывается: помечается ``valid: False``, чтобы
    UI и метрики показывали пропуск, а не молча теряли его.

    Args:
        result: разобранный ответ LLM (мутируется: ключ "quotes").
        chunks: тексты чанков контекста.
        metadatas: метаданные той же длины (поле chunk_id).

    Returns:
        Тот же result с валидированными quotes
        ({"chunk_id", "text", "valid"}).
    """
    chunk_map: Dict[Any, str] = {}
    for index, metadata in enumerate(metadatas or []):
        if index >= len(chunks):
            break
        chunk_id = _coerce_chunk_id((metadata or {}).get("chunk_id"))
        if chunk_id is not None:
            chunk_map[chunk_id] = chunks[index]

    validated: List[Dict[str, Any]] = []
    for quote in result.get("quotes") or []:
        if not isinstance(quote, dict):
            continue
        chunk_id = _coerce_chunk_id(quote.get("chunk_id"))
        text = str(quote.get("text") or "")
        chunk_text = chunk_map.get(chunk_id) if chunk_id is not None else None
        validated.append({
            **quote,
            "chunk_id": chunk_id,
            "text": text,
            "valid": _contains_quote(chunk_text or "", text),
        })
    result["quotes"] = validated
    return result


def _contains_quote(chunk_text: str, quote_text: str) -> bool:
    """Дословная проверка цитаты по чанку (с запасом на пробелы)."""
    quote_text = quote_text.strip()
    if not quote_text or not chunk_text:
        return False
    if quote_text in chunk_text:
        return True
    collapsed_chunk = " ".join(chunk_text.split())
    return " ".join(quote_text.split()) in collapsed_chunk
