"""RAGChatAgent: DeepSeekAgent + RAG-пайплайн (два режима работы).

Обёртка над ядром ``src.agent.DeepSeekAgent`` (профили, память, FSM
задач, инварианты, MCP-инструменты — без изменений). Добавляет режимы
из ТЗ:

* без RAG: ``Вопрос → LLM → Ответ``;
* с RAG: ``Вопрос → embed_query → Chroma.search(top_k) → контекст →
  LLM → Ответ + источники``.

Ядро агента остаётся синхронным (SDK OpenAI); все сетевые вызовы
выполняются через ``asyncio.to_thread``, поэтому Flet-обработчики
(async, page.run_task) не блокируют UI-луп, а обновления UI делает
только событийный поток.

RAG не ломает диалог: если Ollama недоступна или база пуста, вопрос
уходит обычным режимом, а причина попадает в ``result["rag_error"]``
(UI показывает предупреждение).

``answer_once`` — изолированный однократный ответ на отдельном «чистом»
агенте (без хранилища): используется вкладкой «Сравнение», чтобы
контрольные вопросы не засоряли историю основного чата.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

from app.rag.embedder import EmbedderError, OllamaEmbedder
from app.rag.retriever import RetrievedChunk, Retriever
from app.rag.vector_store import VectorStore
from src.agent import DeepSeekAgent
from src.agent_factory import create_agent

TOP_K_MIN = 1
TOP_K_MAX = 10
DEFAULT_TOP_K = 3


class RAGChatAgent:
    """Агент чата с переключаемым RAG-режимом.

    Attributes:
        core: ядро DeepSeekAgent (вся функциональность src).
        embedder: OllamaEmbedder.
        vector_store: VectorStore (Chroma, PersistentClient).
        retriever: Retriever (топ-K + промпт + источники).
        rag_enabled: включён ли режим с RAG (переключатель в чате).
    """

    def __init__(
        self,
        core: DeepSeekAgent,
        embedder: Optional[OllamaEmbedder] = None,
        vector_store: Optional[VectorStore] = None,
        retriever: Optional[Retriever] = None,
        rag_enabled: bool = False,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.core = core
        self.embedder = embedder or OllamaEmbedder()
        self.vector_store = vector_store or VectorStore()
        self.retriever = retriever or Retriever(
            self.embedder, self.vector_store, top_k=top_k
        )
        self.rag_enabled = bool(rag_enabled)
        self.top_k = int(top_k)
        # Конфиг для ленивого создания изолированного агента сравнения
        self._config: Optional[Dict[str, Any]] = None
        self._comparison_core: Optional[DeepSeekAgent] = None
        self.retriever.top_k = self.top_k

    @classmethod
    def with_config(cls, config: Dict[str, Any]) -> "RAGChatAgent":
        """Создаёт ядро через фабрику и RAG-компоненты (настройки из env).

        Переменные окружения: OLLAMA_BASE_URL (по умолчанию
        http://localhost:11434), OLLAMA_EMBED_MODEL (nomic-embed-text).

        Args:
            config: конфигурация load_config.

        Returns:
            Готовый RAGChatAgent.
        """
        core = create_agent(config)
        embedder = OllamaEmbedder(
            model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
        vector_store = VectorStore()
        return cls(core, embedder=embedder, vector_store=vector_store)

    # ------------------------------------------------------------------
    # Настройки режима (UI: переключатель + top_k, «на лету»)
    # ------------------------------------------------------------------
    def set_rag_enabled(self, enabled: bool) -> None:
        """Включает/выключает RAG-режим (действует со следующего хода)."""
        self.rag_enabled = bool(enabled)

    def set_top_k(self, top_k: int) -> None:
        """Число чанков в контексте (1..TOP_K_MAX), применяется сразу.

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

    def count_chunks(self) -> int:
        """Число чанков в базе знаний (статус в чате и на вкладке БЗ)."""
        return self.vector_store.count()

    async def is_rag_available(self) -> bool:
        """Отвечает ли Ollama (быстрая проверка для статус-строки)."""
        return await self.embedder.is_available_async()

    # ------------------------------------------------------------------
    # Диалог
    # ------------------------------------------------------------------
    async def send_message(self, user_input: str) -> Dict[str, Any]:
        """Отправляет сообщение в активном режиме (с RAG или без).

        С RAG: вопрос эмбеддируется (search_query:), находится топ-K
        чанков, собирается промпт по формату ТЗ и передаётся ядру через
        set_rag_context (история хранит вопрос как есть). Источники
        возвращаются в ``result["sources"]``.

        Недоступный Ollama / пустая база не ломают диалог: вопрос уходит
        без контекста, причина — в ``result["rag_error"]``.

        Args:
            user_input: вопрос пользователя.

        Returns:
            Результат core.send_message + опционально "sources"
            (метаданные чанков), "rag_used" (контекст попал в промпт)
            и "rag_error" (почему RAG не применён).
        """
        rag_error: Optional[str] = None
        if self.rag_enabled:
            chunks: List[RetrievedChunk] = []
            try:
                chunks = await self.retriever.retrieve_async(user_input)
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
        return result

    # ------------------------------------------------------------------
    # Изолированный ответ (вкладка «Сравнение»)
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
        self, question: str, rag_enabled: bool
    ) -> Dict[str, Any]:
        """Однократный ответ на чистом агенте (не меняет историю чата).

        Args:
            question: контрольный вопрос.
            rag_enabled: выполнять ли поиск по базе знаний.

        Returns:
            Тот же формат, что send_message.
        """
        core = self._comparison_agent()
        rag_error: Optional[str] = None
        core.clear_history()
        if rag_enabled:
            try:
                chunks = await self.retriever.retrieve_async(question)
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
