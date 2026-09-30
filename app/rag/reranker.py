"""Reranker: переранжирование кандидатов cross-encoder-моделью.

Проблема: Ollama не поддерживает reranker-модели официально — кастомные
сборки (xitao/bge-reranker-v2-m3 и т.п.) могут отсутствовать/не
отвечать. Поэтому:

* ``OllamaReranker`` — попытка через HTTP Ollama (модель по умолчанию
  ``xitao/bge-reranker-v2-m3``); любая ошибка → ``RerankerError``;
* ``FlagEmbeddingReranker`` — официальный путь через пакет
  FlagEmbedding (``BAAI/bge-reranker-v2-m3``, use_fp16); пакет
  опционален — нет установки → ``RerankerError``;
* ``LexicalReranker`` — детерминированный запасной вариант без
  зависимостей: скор лексического пересечения токенов запроса и чанка;
* ``FallbackReranker`` — цепочка из вышеперечисленных: используется
  первый заработавший бэкенд и «залипает» на нём (упавший не
  дёргается повторно), имя активного — в ``last_backend`` для UI.

Все реализации возвращают ИНДЕКСЫ топ-K чанков (порядок убывания
релевантности), сами списки чанков не меняют — перестановку делает
RAGPipeline. Сетевые вызовы (Ollama) имеют sync- и async-варианты;
тяжёлый FlagEmbedding в async уходит в поток (asyncio.to_thread).
"""

from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from typing import List, Optional

import httpx

from src.utils import get_logger

logger = get_logger("deepseek-chat.rag.reranker")

DEFAULT_RERANK_MODEL = "xitao/bge-reranker-v2-m3"
DEFAULT_FLAG_EMBEDDING_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_BASE_URL = "http://localhost:11434"
RERANK_TIMEOUT = 120.0

# Минимальная длина токена для лексического реранка (короткие слова —
# предлоги/союзы, релевантности не добавляют)
LEXICAL_MIN_TOKEN_LEN = 4


class RerankerError(RuntimeError):
    """Ошибка реранка: бэкенд недоступен или вернул некорректный ответ."""


class BaseReranker(ABC):
    """Интерфейс реранкера: запрос + чанки → индексы топ-K."""

    @abstractmethod
    def rerank(self, query: str, chunks: List[str], top_k: int) -> List[int]:
        """Возвращает индексы топ-K чанков после реранка.

        Args:
            query: поисковый запрос.
            chunks: тексты чанков-кандидатов.
            top_k: сколько индексов вернуть.

        Returns:
            Индексы (в порядке убывания релевантности), каждые 0..N-1.

        Raises:
            RerankerError: бэкенд недоступен.
        """

    async def rerank_async(self, query: str, chunks: List[str],
                           top_k: int) -> List[int]:
        """Async-вариант (по умолчанию — sync-вызов в отдельном потоке)."""
        return await asyncio.to_thread(self.rerank, query, chunks, top_k)


def _clamp_top_k(chunks: List[str], top_k: int) -> int:
    """top_k в границах [0, len(chunks)] (пустой список → 0)."""
    return max(0, min(int(top_k), len(chunks)))


class OllamaReranker(BaseReranker):
    """Попытка использовать bge-reranker-v2-m3 через Ollama.

    ВНИМАНИЕ: стандартная Ollama reranker-модели не поддерживает —
    метод бросает RerankerError при любой ошибке (нет модели, нет
    ответа, неразбираемый скор); решение о фолбэке принимает
    FallbackReranker.

    Attributes:
        model: имя кастомной сборки reranker'а в Ollama.
        base_url: базовый URL Ollama.
    """

    def __init__(
        self,
        model: str = DEFAULT_RERANK_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = RERANK_TIMEOUT,
        transport: Optional[httpx.BaseTransport] = None,
        async_transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport
        self._async_transport = async_transport

    # ------------------------------------------------------------------
    def _score_prompt(self, query: str, chunk: str) -> str:
        """Промпт пары «запрос — документ» для reranker-модели."""
        return f"query: {query}</s></s>document: {chunk}"

    def _parse_score(self, payload: dict) -> float:
        """Достаёт скор релевантности из ответа /api/generate."""
        raw = str(payload.get("response") or "").strip()
        if not raw:
            raise RerankerError(
                f"Ollama reranker {self.model!r} вернул пустой ответ"
            )
        match = re.search(r"-?\d+(?:[.,]\d+)?", raw)
        if match is None:
            raise RerankerError(
                f"Ollama reranker {self.model!r} вернул не число: {raw[:100]}"
            )
        return float(match.group(0).replace(",", "."))

    def _parse_response(self, response: httpx.Response) -> float:
        """Скор из HTTP-ответа: не-200/не-JSON/пустой → RerankerError."""
        if response.status_code != 200:
            raise RerankerError(
                f"Ollama reranker {self.model!r}: HTTP {response.status_code}"
                f" — {response.text[:200]}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RerankerError(
                f"Ollama reranker {self.model!r} вернул не-JSON: "
                f"{response.text[:100]}"
            ) from exc
        return self._parse_score(payload)

    def _score_pair(self, client: httpx.Client, query: str,
                    chunk: str) -> float:
        """Скор одной пары (POST /api/generate, без стриминга)."""
        try:
            response = client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": self._score_prompt(query, chunk),
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
            )
        except httpx.HTTPError as exc:
            raise RerankerError(
                f"Ollama reranker {self.model!r} недоступен "
                f"({self.base_url}): {exc}"
            ) from exc
        return self._parse_response(response)

    # ------------------------------------------------------------------
    def rerank(self, query: str, chunks: List[str], top_k: int) -> List[int]:
        """Скорит все пары синхронно, возвращает индексы топ-K."""
        k = _clamp_top_k(chunks, top_k)
        if k == 0:
            return []
        with httpx.Client(timeout=self._timeout,
                          transport=self._transport) as client:
            scores = [self._score_pair(client, query, chunk)
                      for chunk in chunks]
        return self._indices_by_score(scores, k)

    async def rerank_async(self, query: str, chunks: List[str],
                           top_k: int) -> List[int]:
        """Native-async вариант (один AsyncClient на все пары)."""
        k = _clamp_top_k(chunks, top_k)
        if k == 0:
            return []
        # MockTransport в тестах передаётся как sync-транспорт — он
        # работает и в AsyncClient, поэтому используем его как запаску
        transport = self._async_transport or self._transport
        async with httpx.AsyncClient(
            timeout=self._timeout, transport=transport
        ) as client:
            scores = [await self._score_pair_async(client, query, chunk)
                      for chunk in chunks]
        return self._indices_by_score(scores, k)

    async def _score_pair_async(self, client: httpx.AsyncClient, query: str,
                                chunk: str) -> float:
        """Async-скор одной пары."""
        try:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": self._score_prompt(query, chunk),
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
            )
        except httpx.HTTPError as exc:
            raise RerankerError(
                f"Ollama reranker {self.model!r} недоступен "
                f"({self.base_url}): {exc}"
            ) from exc
        return self._parse_response(response)

    @staticmethod
    def _indices_by_score(scores: List[float], k: int) -> List[int]:
        """Индексы k наибольших скоров (при равенстве — меньший индекс)."""
        ranked = sorted(range(len(scores)),
                        key=lambda i: scores[i], reverse=True)
        return ranked[:k]


class FlagEmbeddingReranker(BaseReranker):
    """Реранкер через пакет FlagEmbedding (официальный путь BAAI).

    Пакет опционален (тянет torch): отсутствует → RerankerError при
    первом вызове (импорт ленивый, результат кэшируется).
    """

    def __init__(self, model_name: str = DEFAULT_FLAG_EMBEDDING_MODEL,
                 use_fp16: bool = True) -> None:
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self._reranker = None

    def _ensure_model(self):
        """Ленивая загрузка FlagReranker (первый вызов — тяжёлый)."""
        if self._reranker is None:
            try:
                from FlagEmbedding import FlagReranker
            except ImportError as exc:
                raise RerankerError(
                    "Пакет FlagEmbedding не установлен "
                    "(pip install FlagEmbedding) — реранкер недоступен"
                ) from exc
            self._reranker = FlagReranker(self.model_name,
                                          use_fp16=self.use_fp16)
        return self._reranker

    def rerank(self, query: str, chunks: List[str], top_k: int) -> List[int]:
        """compute_score по парам → индексы топ-K по убыванию скора."""
        k = _clamp_top_k(chunks, top_k)
        if k == 0:
            return []
        reranker = self._ensure_model()
        pairs = [[query, chunk] for chunk in chunks]
        try:
            scores = reranker.compute_score(pairs)
        except Exception as exc:  # noqa: BLE001 — ошибки модели широкие
            raise RerankerError(
                f"FlagEmbedding {self.model_name!r} упал: {exc}"
            ) from exc
        scores = [float(score) for score in
                  (scores if isinstance(scores, list) else [scores])]
        return OllamaReranker._indices_by_score(scores, k)


class LexicalReranker(BaseReranker):
    """Запасной реранкер без зависимостей: пересечение токенов.

    Скор — доля токенов запроса (длина ≥ 4), найденных в чанке.
    Работает всегда, качество ниже cross-encoder'а: это последняя
    ступень fallback-цепочки, чтобы режимы reranker не падали совсем.
    """

    def rerank(self, query: str, chunks: List[str], top_k: int) -> List[int]:
        k = _clamp_top_k(chunks, top_k)
        if k == 0:
            return []
        query_tokens = self._tokens(query)
        scores = []
        for chunk in chunks:
            if not query_tokens:
                scores.append(0.0)
                continue
            text = " " + " ".join(self._tokens(chunk)) + " "
            hits = sum(1 for token in query_tokens if token in text)
            scores.append(hits / len(query_tokens))
        return OllamaReranker._indices_by_score(scores, k)

    @staticmethod
    def _tokens(text: str) -> List[str]:
        """Токены нижнего регистра (буквы/цифры), длина ≥ 4, без дублей."""
        words = re.findall(r"[а-яёa-z0-9]+", (text or "").lower())
        seen: "set[str]" = set()
        tokens: List[str] = []
        for word in words:
            if len(word) >= LEXICAL_MIN_TOKEN_LEN and word not in seen:
                seen.add(word)
                tokens.append(word)
        return tokens


class FallbackReranker(BaseReranker):
    """Цепочка реранкеров: первый успешный бэкенд «залипает».

    Attributes:
        backends: кандидаты по порядку приоритета.
        last_backend: имя бэкенда последнего успешного вызова
            (класс реранкера — для метрик UI); None — ещё не вызывался.
    """

    def __init__(self, backends: List[BaseReranker]) -> None:
        if not backends:
            raise ValueError("FallbackReranker: список бэкендов пуст")
        self.backends = list(backends)
        self.last_backend: Optional[str] = None
        self._active_index: Optional[int] = None

    def rerank(self, query: str, chunks: List[str], top_k: int) -> List[int]:
        if _clamp_top_k(chunks, top_k) == 0:
            return []
        last_error: Optional[Exception] = None
        order = self._backend_order()
        for index in order:
            backend = self.backends[index]
            try:
                indices = backend.rerank(query, chunks, top_k)
            except RerankerError as exc:
                logger.warning("Reranker %s не сработал: %s — пробую следующий",
                               type(backend).__name__, exc)
                last_error = exc
                continue
            self._active_index = index
            self.last_backend = type(backend).__name__
            return indices
        raise RerankerError(
            f"Ни один реранкер не сработал "
            f"({', '.join(type(b).__name__ for b in self.backends)}): "
            f"{last_error}"
        )

    async def rerank_async(self, query: str, chunks: List[str],
                           top_k: int) -> List[int]:
        if _clamp_top_k(chunks, top_k) == 0:
            return []
        last_error: Optional[Exception] = None
        for index in self._backend_order():
            backend = self.backends[index]
            try:
                indices = await backend.rerank_async(query, chunks, top_k)
            except RerankerError as exc:
                logger.warning("Reranker %s не сработал: %s — пробую следующий",
                               type(backend).__name__, exc)
                last_error = exc
                continue
            self._active_index = index
            self.last_backend = type(backend).__name__
            return indices
        raise RerankerError(
            f"Ни один реранкер не сработал "
            f"({', '.join(type(b).__name__ for b in self.backends)}): "
            f"{last_error}"
        )

    def _backend_order(self) -> List[int]:
        """Активный бэкенд (если залеп) → остальные по порядку."""
        if self._active_index is not None:
            rest = [i for i in range(len(self.backends))
                    if i != self._active_index]
            return [self._active_index] + rest
        return list(range(len(self.backends)))


def create_default_reranker(
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_RERANK_MODEL,
) -> FallbackReranker:
    """Fallback-цепочка приложения: Ollama → FlagEmbedding → Lexical."""
    return FallbackReranker([
        OllamaReranker(model=model, base_url=base_url),
        FlagEmbeddingReranker(),
        LexicalReranker(),
    ])
