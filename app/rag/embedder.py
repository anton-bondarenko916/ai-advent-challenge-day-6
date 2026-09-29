"""Эмбеддер на локальной Ollama (модель nomic-embed-text).

Используется HTTP-эндпоинт ``POST /api/embed`` (вход "input" принимает
и строку, и список строк — батч). Обязательны префиксы задач для
nomic-embed-text:

    search_document: — для чанков базы знаний (индексация);
    search_query:    — для вопросов пользователя (поиск).

Облачные эмбеддеры не используются — только локальная Ollama.

Все методы имеют sync- и async-варианты: sync используется индексатором
в фоне (page.run_thread/to_thread), async — в async-обработчиках Flet.
Для тестов в конструктор можно передать httpx-transport (MockTransport).
"""

from __future__ import annotations

from typing import List, Optional

import httpx

DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_BASE_URL = "http://localhost:11434"


class EmbedderError(RuntimeError):
    """Ошибка эмбеддера: Ollama недоступна или вернула ошибку."""


class OllamaEmbedder:
    """Клиент эмбеддингов Ollama (/api/embed) с префиксами задач.

    Attributes:
        model: имя модели эмбеддингов (по умолчанию nomic-embed-text).
        base_url: базовый URL Ollama (по умолчанию localhost:11434).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        transport: Optional[httpx.BaseTransport] = None,
        async_transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport
        self._async_transport = async_transport

    # ------------------------------------------------------------------
    # Публичное API
    # ------------------------------------------------------------------
    def embed_document(self, text: str) -> List[float]:
        """Эмбеддинг чанка базы знаний (префикс search_document:)."""
        return self._embed(DOCUMENT_PREFIX + text)

    def embed_query(self, text: str) -> List[float]:
        """Эмбеддинг вопроса пользователя (префикс search_query:)."""
        return self._embed(QUERY_PREFIX + text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Батч-эмбеддинг чанков (один POST со списком input)."""
        return self._embed_many([DOCUMENT_PREFIX + t for t in texts])

    async def embed_document_async(self, text: str) -> List[float]:
        """Async-вариант embed_document."""
        return await self._embed_async(DOCUMENT_PREFIX + text)

    async def embed_query_async(self, text: str) -> List[float]:
        """Async-вариант embed_query."""
        return await self._embed_async(QUERY_PREFIX + text)

    async def embed_documents_async(self, texts: List[str]) -> List[List[float]]:
        """Async-вариант embed_documents (батч)."""
        return await self._embed_many_async([DOCUMENT_PREFIX + t for t in texts])

    def is_available(self) -> bool:
        """Проверяет, отвечает ли Ollama (GET /api/tags, таймаут 2 с)."""
        try:
            with self._sync_client() as client:
                response = client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def is_available_async(self) -> bool:
        """Async-вариант is_available."""
        try:
            async with self._async_client() as client:
                response = await client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    # ------------------------------------------------------------------
    # Внутреннее
    # ------------------------------------------------------------------
    def _embed(self, text: str) -> List[float]:
        """Один текст → один вектор.

        Args:
            text: текст с префиксом задачи.

        Returns:
            Вектор эмбеддинга.

        Raises:
            EmbedderError: Ollama недоступна или вернула ошибку.
        """
        return self._embed_many([text])[0]

    def _embed_many(self, texts: List[str]) -> List[List[float]]:
        try:
            with self._sync_client() as client:
                response = client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": texts},
                )
        except httpx.HTTPError as exc:
            raise EmbedderError(self._connection_message(exc)) from exc
        return self._parse_embeddings(response)

    async def _embed_async(self, text: str) -> List[float]:
        return (await self._embed_many_async([text]))[0]

    async def _embed_many_async(self, texts: List[str]) -> List[List[float]]:
        try:
            async with self._async_client() as client:
                response = await client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": texts},
                )
        except httpx.HTTPError as exc:
            raise EmbedderError(self._connection_message(exc)) from exc
        return self._parse_embeddings(response)

    def _sync_client(self) -> httpx.Client:
        return httpx.Client(
            timeout=self._timeout, transport=self._transport
        )

    def _async_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self._timeout, transport=self._async_transport
        )

    def _parse_embeddings(self, response: httpx.Response) -> List[List[float]]:
        """Разбирает ответ /api/embed с человеческой диагностикой ошибок."""
        if response.status_code != 200:
            raise EmbedderError(
                f"Ollama /api/embed вернула {response.status_code}: "
                f"{response.text[:200]}"
            )
        payload = response.json()
        embeddings = payload.get("embeddings")
        if not embeddings:
            raise EmbedderError(
                f"Ollama не вернула эмбеддинги (модель {self.model!r}): "
                f"{str(payload)[:200]}"
            )
        return embeddings

    def _connection_message(self, exc: httpx.HTTPError) -> str:
        return (
            f"Ollama недоступна по адресу {self.base_url} "
            f"(модель {self.model!r}). Запустите 'ollama serve' и проверьте, "
            f"что модель загружена. Детали: {exc}"
        )
