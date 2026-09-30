"""RAGChatAgent: DeepSeekAgent + RAG-пайплайн (пять RAG-режимов).

Обёртка над ядром ``src.agent.DeepSeekAgent`` (профили, память, FSM
задач, инварианты, MCP-инструменты — без изменений). Добавляет режимы
из ТЗ:

* без RAG: ``Вопрос → LLM → Ответ``;
* с RAG: ``Вопрос → RAGPipeline(mode) → контекст → LLM → Ответ +
  источники``, где RAGPipeline (app.rag.pipeline) поддерживает пять
  режимов: basic / rewrite / rewrite_filter / reranker /
  rewrite_reranker (Query Rewrite через DeepSeek, порог similarity,
  реранкер с fallback-цепочкой).

Ядро агента остаётся синхронным (SDK OpenAI); все сетевые вызовы
выполняются через ``asyncio.to_thread`` / native-async клиенты, поэтому
Flet-обработчики (async, page.run_task) не блокируют UI-луп, а
обновления UI делает только событийный поток.

RAG не ломает диалог: если Ollama недоступна или база пуста, вопрос
уходит обычным режимом, а причина попадает в ``result["rag_error"]``;
сбой отдельной ступени пайплайна (rewrite/rerank) деградирует режим,
не падая (``result["retrieval"]`` хранит диагностику).

``answer_once`` — изолированный однократный ответ на отдельном «чистом»
агенте (без хранилища): используется вкладками «Сравнение» и
«Сравнение RAG-режимов», чтобы контрольные вопросы не засоряли историю
основного чата.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

from app.rag.embedder import EmbedderError, OllamaEmbedder
from app.rag.pipeline import (
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K_FINAL,
    RAG_MODE_BASIC,
    TOP_K_SEARCH_MAX,
    RAGPipeline,
    normalize_mode,
)
from app.rag.pipeline import DEFAULT_TOP_K as DEFAULT_TOP_K_SEARCH
from app.rag.reranker import create_default_reranker
from app.rag.retriever import RetrievedChunk, Retriever
from app.rag.rewriter import QueryRewriter
from app.rag.vector_store import VectorStore
from src.agent import DeepSeekAgent
from src.agent_factory import create_agent

# Размер контекста (после фильтрации/реранка) — прежнее имя top_k
DEFAULT_TOP_K = DEFAULT_TOP_K_FINAL
TOP_K_MIN = 1
TOP_K_MAX = 10
THRESHOLD_MIN = 0.0
THRESHOLD_MAX = 1.0


class RAGChatAgent:
    """Агент чата с переключаемым RAG-режимом (5 режимов пайплайна).

    Attributes:
        core: ядро DeepSeekAgent (вся функциональность src).
        embedder: OllamaEmbedder.
        vector_store: VectorStore (Chroma, PersistentClient).
        retriever: Retriever (простой топ-K, остался для совместимости).
        pipeline: RAGPipeline (rewrite/threshold/rerank, 5 режимов).
        rag_enabled: включён ли режим с RAG (переключатель в чате).
        rag_mode: активный режим пайплайна (basic/rewrite/...).
        top_k: чанков в контексте после фильтрации/реранка (top_k_final).
        top_k_search: кандидатов из Chroma до фильтрации/реранка.
        similarity_threshold: порог отсечения rewrite_filter.
    """

    def __init__(
        self,
        core: DeepSeekAgent,
        embedder: Optional[OllamaEmbedder] = None,
        vector_store: Optional[VectorStore] = None,
        retriever: Optional[Retriever] = None,
        rewriter: Optional[QueryRewriter] = None,
        reranker: Optional[Any] = None,
        rag_enabled: bool = False,
        top_k: int = DEFAULT_TOP_K,
        top_k_search: int = DEFAULT_TOP_K_SEARCH,
        rag_mode: str = RAG_MODE_BASIC,
        similarity_threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.core = core
        self.embedder = embedder or OllamaEmbedder()
        self.vector_store = vector_store or VectorStore()
        self.retriever = retriever or Retriever(
            self.embedder, self.vector_store, top_k=top_k
        )
        self.rewriter = rewriter
        self.reranker = reranker
        self.rag_enabled = bool(rag_enabled)
        self.top_k = int(top_k)
        self.top_k_search = int(top_k_search)
        self.rag_mode = normalize_mode(rag_mode)
        self.similarity_threshold = float(similarity_threshold)
        # Пайплайн работает с теми же embedder/vector_store, что и
        # старый retriever (переключение настроек действует на оба пути)
        self.pipeline = RAGPipeline(
            self.embedder, self.vector_store,
            rewriter=self.rewriter, reranker=self.reranker,
        )
        # Конфиг для ленивого создания изолированного агента сравнения
        self._config: Optional[Dict[str, Any]] = None
        self._comparison_core: Optional[DeepSeekAgent] = None
        self.retriever.top_k = self.top_k

    @classmethod
    def with_config(cls, config: Dict[str, Any]) -> "RAGChatAgent":
        """Создаёт ядро через фабрику и RAG-компоненты (настройки из env).

        Переменные окружения: OLLAMA_BASE_URL (по умолчанию
        http://localhost:11434), OLLAMA_EMBED_MODEL (nomic-embed-text),
        OLLAMA_RERANK_MODEL (xitao/bge-reranker-v2-m3).

        Args:
            config: конфигурация load_config.

        Returns:
            Готовый RAGChatAgent (rewriter и fallback-реранкер настроены).
        """
        core = create_agent(config)
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        embedder = OllamaEmbedder(
            model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            base_url=ollama_url,
        )
        vector_store = VectorStore()
        rewriter = QueryRewriter(
            api_key=str(config.get("DEEPSEEK_API_KEY") or ""),
            base_url=str(config.get("DEEPSEEK_BASE_URL")
                         or "https://api.deepseek.com"),
            model=str(config.get("DEFAULT_MODEL") or "deepseek-v4-flash"),
            mock_mode=bool(config.get("MOCK_MODE")),
        )
        reranker = create_default_reranker(
            base_url=ollama_url,
            model=os.getenv("OLLAMA_RERANK_MODEL",
                            "xitao/bge-reranker-v2-m3"),
        )
        return cls(core, embedder=embedder, vector_store=vector_store,
                   rewriter=rewriter, reranker=reranker)

    # ------------------------------------------------------------------
    # Настройки режима (UI: переключатель + параметры, «на лету»)
    # ------------------------------------------------------------------
    def set_rag_enabled(self, enabled: bool) -> None:
        """Включает/выключает RAG-режим (действует со следующего хода)."""
        self.rag_enabled = bool(enabled)

    def set_rag_mode(self, mode: str) -> str:
        """Активный режим пайплайна (basic/rewrite/...).

        Raises:
            ValueError: неизвестное имя режима.
        """
        self.rag_mode = normalize_mode(mode)
        return self.rag_mode

    def set_top_k(self, top_k: int) -> None:
        """Чанков в контексте (top_k_final, 1..TOP_K_MAX), сразу.

        Raises:
            ValueError: top_k вне допустимых границ.
        """
        value = int(top_k)
        if not TOP_K_MIN <= value <= TOP_K_MAX:
            raise ValueError(
                f"top_k должен быть от {TOP_K_MIN} до {TOP_K_MAX}, "
                f"получено {top_k!r}"
            )
        self.top_k = value
        self.retriever.top_k = value

    def set_top_k_final(self, top_k_final: int) -> None:
        """Алиас set_top_k (имя из ТЗ: top_k после фильтрации/реранка)."""
        self.set_top_k(top_k_final)

    @property
    def top_k_final(self) -> int:
        """Чанков в контексте (имя из ТЗ) — то же, что top_k."""
        return self.top_k

    def set_top_k_search(self, top_k: int) -> None:
        """Кандидатов из Chroma до фильтрации/реранка (1..MAX).

        Raises:
            ValueError: значение вне допустимых границ.
        """
        value = int(top_k)
        if not TOP_K_MIN <= value <= TOP_K_SEARCH_MAX:
            raise ValueError(
                f"top_k (до фильтрации) должен быть от {TOP_K_MIN} "
                f"до {TOP_K_SEARCH_MAX}, получено {top_k!r}"
            )
        self.top_k_search = value

    def set_similarity_threshold(self, threshold: float) -> float:
        """Порог similarity для rewrite_filter (0.0..1.0).

        Raises:
            ValueError: порог вне допустимых границ.
        """
        value = float(threshold)
        if not THRESHOLD_MIN <= value <= THRESHOLD_MAX:
            raise ValueError(
                f"Порог similarity должен быть от {THRESHOLD_MIN} до "
                f"{THRESHOLD_MAX}, получено {threshold!r}"
            )
        self.similarity_threshold = value
        return value

    def count_chunks(self) -> int:
        """Число чанков в базе знаний (статус в чате и на вкладке БЗ)."""
        return self.vector_store.count()

    async def is_rag_available(self) -> bool:
        """Отвечает ли Ollama (быстрая проверка для статус-строки)."""
        return await self.embedder.is_available_async()

    # ------------------------------------------------------------------
    # Диалог
    # ------------------------------------------------------------------
    def _retrieval_params(self) -> Dict[str, int]:
        """Текущие параметры поиска (режим берётся из self.rag_mode)."""
        return {
            "top_k": self.top_k_search,
            "top_k_final": self.top_k,
            "threshold": self.similarity_threshold,
        }

    async def send_message(self, user_input: str) -> Dict[str, Any]:
        """Отправляет сообщение в активном режиме (с RAG или без).

        С RAG: вопрос проходит RAGPipeline (rewrite/threshold/rerank по
        self.rag_mode), собирается промпт по формату ТЗ и передаётся
        ядру через set_rag_context (история хранит вопрос как есть).
        Источники — в ``result["sources"]``, диагностика поиска — в
        ``result["retrieval"]``.

        Недоступный Ollama / пустая база не ломают диалог: вопрос уходит
        без контекста, причина — в ``result["rag_error"]``.

        Args:
            user_input: вопрос пользователя.

        Returns:
            Результат core.send_message + опционально "sources",
            "rag_used", "rag_error", "retrieval" (search_query, scores,
            rewrite/rerank диагностика).
        """
        rag_error: Optional[str] = None
        retrieval: Optional[Dict[str, Any]] = None
        chunks: List[RetrievedChunk] = []
        if self.rag_enabled:
            try:
                retrieval = await self.pipeline.retrieve_async(
                    user_input, mode=self.rag_mode,
                    **self._retrieval_params(),
                )
                chunks = retrieval["chunks"]
            except EmbedderError as exc:
                rag_error = str(exc)
            if chunks:
                prompt = Retriever.build_rag_prompt(user_input, chunks)
                self.core.set_rag_context(
                    prompt, Retriever.format_sources(chunks)
                )
            elif rag_error is None:
                rag_error = (
                    "База знаний пуста или ничего не найдено — "
                    "ответ дан без RAG."
                )
        result = await asyncio.to_thread(self.core.send_message, user_input)
        if rag_error is not None:
            result["rag_error"] = rag_error
        if retrieval is not None:
            result["retrieval"] = {
                "search_query": retrieval["search_query"],
                "rewrite_used": retrieval["rewrite_used"],
                "rewrite_error": retrieval["rewrite_error"],
                "rerank_backend": retrieval["rerank_backend"],
                "rerank_error": retrieval["rerank_error"],
                "scores": [round(score, 4) for score in retrieval["scores"]],
            }
        return result

    # ------------------------------------------------------------------
    # Изолированный ответ (вкладки «Сравнение» и «Сравнение RAG-режимов»)
    # ------------------------------------------------------------------
    def _comparison_agent(self) -> DeepSeekAgent:
        """Отдельный агент без хранилища для контрольных вопросов.

        Создаётся лениво из копии конфигурации с STORAGE_ENABLED=False:
        прогон сравнения не пишет в SQLite и не трогает историю чата.
        """
        if self._comparison_core is None:
            if self._config is None:
                raise RuntimeError(
                    "answer_once недоступен: агент создан без конфигурации "
                    "(передайте config в bind_config)"
                )
            comparison_config = dict(self._config)
            comparison_config["STORAGE_ENABLED"] = False
            self._comparison_core = create_agent(comparison_config)
        return self._comparison_core

    def bind_config(self, config: Dict[str, Any]) -> "RAGChatAgent":
        """Сохраняет конфиг для изолированного агента сравнения."""
        self._config = config
        return self

    async def answer_once(
        self,
        question: str,
        rag_enabled: bool,
        rag_mode: Optional[str] = None,
        retrieval_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Однократный ответ на чистом агенте (не меняет историю чата).

        Args:
            question: контрольный вопрос.
            rag_enabled: выполнять ли поиск по базе знаний.
            rag_mode: режим пайплайна для этого запроса (None — текущий
                self.rag_mode); служит вкладке «Сравнение RAG-режимов».
            retrieval_params: {"top_k", "top_k_final", "threshold"} —
                переопределение параметров поиска (None — текущие).

        Returns:
            Тот же формат, что send_message (+ "retrieval" при RAG).
        """
        core = self._comparison_agent()
        rag_error: Optional[str] = None
        retrieval: Optional[Dict[str, Any]] = None
        core.clear_history()
        if rag_enabled:
            params = dict(self._retrieval_params())
            if retrieval_params:
                params.update(retrieval_params)
            mode = self.rag_mode if rag_mode is None else rag_mode
            chunks: List[RetrievedChunk] = []
            try:
                retrieval = await self.pipeline.retrieve_async(
                    question, mode=mode, **params
                )
                chunks = retrieval["chunks"]
            except EmbedderError as exc:
                chunks, rag_error = [], str(exc)
            if chunks:
                core.set_rag_context(
                    Retriever.build_rag_prompt(question, chunks),
                    Retriever.format_sources(chunks),
                )
            elif rag_error is None:
                rag_error = "База знаний пуста или ничего не найдено."
        result = await asyncio.to_thread(core.send_message, question)
        if rag_error is not None:
            result["rag_error"] = rag_error
        if retrieval is not None:
            result["retrieval"] = {
                "search_query": retrieval["search_query"],
                "rewrite_used": retrieval["rewrite_used"],
                "rewrite_error": retrieval["rewrite_error"],
                "rerank_backend": retrieval["rerank_backend"],
                "rerank_error": retrieval["rerank_error"],
                "scores": [round(score, 4) for score in retrieval["scores"]],
            }
        return result

    # ------------------------------------------------------------------
    # Делегирование к ядру (для UI)
    # ------------------------------------------------------------------
    def clear_history(self) -> None:
        """Очищает историю чата (ядро)."""
        self.core.clear_history()

    def close(self) -> None:
        """Финальный flush контекста ядра при закрытии приложения."""
        self.core.close()

    def update_config(self, **kwargs: Any) -> None:
        """Обновляет параметры ядра (модель/температура/max_tokens)."""
        self.core.update_config(**kwargs)

    def set_profile(self, profile_id: int) -> Dict[str, Any]:
        """Переключает активный профиль (ядро)."""
        return self.core.set_profile(profile_id)

    @property
    def model(self) -> str:
        return self.core.model

    @property
    def temperature(self) -> float:
        return self.core.temperature

    @property
    def max_tokens(self) -> int:
        return self.core.max_tokens

    @property
    def profile_repository(self):
        return self.core.profile_repository
