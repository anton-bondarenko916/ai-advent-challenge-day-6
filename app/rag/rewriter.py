"""Query Rewrite: переформулировка вопроса для лучшего векторного поиска.

Отдельный служебный вызов DeepSeek (тот же OpenAI-совместимый SDK, что
и у ядра): вопрос пользователя → «поисковая» формулировка с ключевыми
словами технической документации. Используется режимами rewrite /
rewrite_filter / rewrite_reranker пайплайна (app.rag.pipeline).

Надёжность: ошибка сети/API не должна ломать RAG-ход — вызывающий код
(RAGPipeline) ловит RewriteError и подставляет исходный вопрос
(режим деградирует до basic-поиска, диагностика — в результате хода).

В mock-режиме сеть не трогается вовсе: rewrite возвращает вопрос как
есть (офлайн-тесты и запуск с --mock).
"""

from __future__ import annotations

import asyncio
from typing import Optional

from openai import OpenAI

from src.utils import get_logger

logger = get_logger("deepseek-chat.rag.rewriter")

# Значения по умолчанию (модель/базовый URL — как у ядра диалога)
DEFAULT_REWRITE_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
REWRITE_TEMPERATURE = 0.3
REWRITE_MAX_TOKENS = 200
REWRITE_TIMEOUT = 30.0

REWRITE_PROMPT = """Переформулируй следующий вопрос для поиска по технической документации.
Сделай его более конкретным и «поисковым». Сохрани смысл.

Вопрос: {question}

Верни только переформулированный вопрос, без пояснений."""


class RewriteError(RuntimeError):
    """Ошибка переформулировки: DeepSeek недоступен или вернул ошибку."""


class QueryRewriter:
    """Переформулировщик вопросов через DeepSeek API.

    Attributes:
        model: модель служебного запроса (deepseek-v4-flash).
        base_url: базовый URL API (DeepSeek совместим с OpenAI SDK).
        mock_mode: True — сеть не используется, rewrite возвращает
            исходный вопрос (запуск --mock и офлайн-тесты).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_REWRITE_MODEL,
        temperature: float = REWRITE_TEMPERATURE,
        max_tokens: int = REWRITE_MAX_TOKENS,
        mock_mode: bool = False,
        timeout: float = REWRITE_TIMEOUT,
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
                api_key=self._api_key_or_stub(),
                base_url=self.base_url,
                timeout=self._timeout,
            )
        return self._client

    def _api_key_or_stub(self) -> str:
        """Ключ API; пустой заменяется заглушкой, чтобы конструктор SDK
        не упал (запрос всё равно вернёт 401 → RewriteError → фолбэк)."""
        return self.api_key or "missing-api-key"

    # ------------------------------------------------------------------
    # Публичное API
    # ------------------------------------------------------------------
    def rewrite(self, question: str) -> str:
        """Переформулирует вопрос для векторного поиска.

        Args:
            question: исходный вопрос пользователя.

        Returns:
            Переформулированный вопрос (обрезан, переносы схлопнуты).
            Пустой ответ модели / mock-режим / ошибка → исходный вопрос.

        Raises:
            RewriteError: только при сетевой ошибке/ошибке API в
                не-mock-режиме (пустой ответ ошибкой не считается —
                возвращается исходный вопрос).
        """
        question = (question or "").strip()
        if not question:
            return ""
        if self.mock_mode:
            return question
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": REWRITE_PROMPT.format(question=question),
                }],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 — SDK бросает широкий набор
            raise RewriteError(
                f"Query rewrite не выполнен ({self.model}): {exc}"
            ) from exc
        rewritten = str(response.choices[0].message.content or "").strip()
        if not rewritten:
            # Пустой ответ (например, весь бюджет ушёл в reasoning) —
            # поиск по исходному вопросу лучше падения режима
            logger.warning("Query rewrite вернул пустой ответ — "
                           "ищу по исходному вопросу")
            return question
        return " ".join(rewritten.split())

    async def rewrite_async(self, question: str) -> str:
        """Async-вариант rewrite (SDK синхронный — уход в поток)."""
        return await asyncio.to_thread(self.rewrite, question)
