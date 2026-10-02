"""Обработчик хода чата: RAG + память задачи + цитаты (ТЗ ai_1_24).

Схема обработки сообщения (Модуль 4 ТЗ):

    1. Загрузить память задачи сессии (task_state, SQLite);
    2. RAG-поиск по вопросу (всегда включён в чате);
    3. Гейт релевантности: средний similarity >= 0.6 (ABSTAIN_THRESHOLD,
       откалиброван для nomic-embed-text) → RAG-контекст; ниже —
       СТРАТЕГИЯ C: контекст заменяется пометкой «НЕТ РЕЛЕВАНТНЫХ
       ИСТОЧНИКОВ», ответ даётся из общих знаний (без источников);
    4. Промпт: [система ядра] + [память задачи] + [RAG-контекст или
       пометка] + [история диалога] + [вопрос] — память задачи и
       контекст уходят через core.set_rag_context (подстановка в копию
       запроса, история чистая);
    5. Ответ даёт ядро DeepSeekAgent (инструменты MCP, FSM, инварианты,
       токены и персистентность работают как раньше);
    6. Обновление памяти задачи и цитаты — JSON Mode (Вариант B ТЗ):
       ChatTurnExtractor отдельным вызовом с response_format=
       {"type": "json_object"} возвращает {sources, quotes,
       task_state}; цитаты дословно проверяются по чанкам
       (validate_quotes), память мерджится и сохраняется в SQLite.

Отказоустойчивость — в духе остальных ступеней: недоступный Ollama,
пустая база, ошибка JSON-вызова или БД не ломают ход (ответ показывается,
причина — в логах и диагностике результата).
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from app.rag.citations import (
    ABSTAIN_THRESHOLD,
    CitationError,
    parse_citation_json,
    should_abstain,
    validate_quotes,
)
from app.rag.embedder import EmbedderError
from app.rag.retriever import RetrievedChunk, Retriever
from app.task_state import (
    TaskState,
    TaskStateRepository,
    normalize_task_update,
)
from src.utils import get_logger

logger = get_logger("deepseek-chat.chat_handler")

# Пометка вместо RAG-контекста при стратегии C (формулировка из ТЗ)
NO_RELEVANT_SOURCES = "НЕТ РЕЛЕВАНТНЫХ ИСТОЧНИКОВ"

# Пометка для UI при стратегии C (Модуль 5 ТЗ)
STRATEGY_C_NOTE = ("⚠️ В документах нет информации по этому вопросу. "
                   "Ответ дан из общих знаний (без источников).")

# Инструкции для блока контекста основного хода (текст последнего
# user-сообщения запроса; история хранит вопрос как есть)
RAG_TURN_INSTRUCTION = (
    "Отвечай на вопрос, используя RAG-контекст. Учитывай память задачи: "
    "цель диалога, уточнения и ограничения."
)
NO_CONTEXT_INSTRUCTION = (
    "В базе знаний нет информации по этому вопросу — дай ответ из общих "
    "знаний. Учитывай память задачи: цель диалога, уточнения и ограничения."
)

# Температура/бюджет служебного JSON-вызова — как у CitationAnswerer:
# thinking-модель тратит бюджет на reasoning, 2000 мало (урок
# NEWS_SUMMARY_MAX_TOKENS)
EXTRACTION_TEMPERATURE = 0.2
EXTRACTION_MAX_TOKENS = 8000
EXTRACTION_TIMEOUT = 60.0

DEFAULT_EXTRACTION_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"

# Промпт обновления памяти задачи (Модуль 3 ТЗ, адаптирован под ход
# чата: ответ уже получен ядром, экстрактор обновляет память и выбирает
# цитаты-подтверждения). Слово «JSON» в тексте обязательно.
TASK_EXTRACTION_PROMPT = """Ты — аналитик диалога с памятью задачи. Твоя задача — обновить память задачи (task state) и выбрать источники с дословными цитатами.

[ПАМЯТЬ ЗАДАЧИ]
{task_state}

[RAG-КОНТЕКСТ]
{context}

Вопрос пользователя: {question}

Ответ ассистента: {answer}

Твой ответ ДОЛЖЕН быть валидным JSON-объектом:

{{
  "sources": [
    {{"chunk_id": 12, "source": "installation.md", "section": "## Windows"}}
  ],
  "quotes": [
    {{"chunk_id": 12, "text": "Дословная цитата из чанка."}}
  ],
  "task_state": {{
    "goal": "Обновлённая цель диалога (или пустая строка, если не изменилась)",
    "clarifications": ["новое уточнение, если появилось"],
    "constraints": ["новое ограничение, если появилось"],
    "terms": {{"новый_термин": "определение"}}
  }}
}}

ВАЖНО:
- Цитаты — ДОСЛОВНЫЕ фрагменты из RAG-КОНТЕКСТА (точная копия текста), подтверждающие ответ; подходящих нет — верни пустой список.
- sources — только чанки, реально использованные в ответе; нет — пустой список.
- Обновляй task_state ТОЛЬКО если появилась новая информация: новая цель, уточнение, ограничение или термин.
- Если ничего не изменилось — верни пустые поля task_state."""


class TaskStateError(RuntimeError):
    """Ошибка JSON-вызова экстрактора: API недоступен/ответ не разобрался."""


def format_context_for_extraction(chunks: List[RetrievedChunk]) -> str:
    """RAG-контекст для промпта экстрактора (шапки chunk_id как в ТЗ)."""
    if not chunks:
        return "НЕТ РЕЛЕВАНТНЫХ ИСТОЧНИКОВ"
    blocks = []
    for chunk in chunks:
        header = (f"[chunk_id={chunk.chunk_id} | "
                  f"{chunk.source or '—'} | {chunk.section or '—'}]")
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


def build_rag_turn_block(task_state: TaskState, question: str,
                         chunks: List[RetrievedChunk]) -> str:
    """Блок последнего user-сообщения: память задачи + RAG-контекст.

    Структура промпта из ТЗ: [ПАМЯТЬ ЗАДАЧИ] + [RAG-КОНТЕКСТ] +
    [ИНСТРУКЦИЯ]; история диалога и системный промпт добавляются ядром.
    """
    return (
        f"[ПАМЯТЬ ЗАДАЧИ]\n{task_state.to_prompt_block()}\n\n"
        f"[RAG-КОНТЕКСТ]\n{Retriever.build_rag_prompt(question, chunks)}\n\n"
        f"[ИНСТРУКЦИЯ]\n{RAG_TURN_INSTRUCTION}"
    )


def build_no_context_block(task_state: TaskState,
                           rag_error: Optional[str] = None) -> str:
    """Блок стратегии C: память задачи + пометка вместо контекста."""
    if rag_error:
        marker = f"ПОИСК НЕДОСТУПЕН: {rag_error}"
    else:
        marker = NO_RELEVANT_SOURCES
    return (
        f"[ПАМЯТЬ ЗАДАЧИ]\n{task_state.to_prompt_block()}\n\n"
        f"[RAG-КОНТЕКСТ]\n{marker}\n\n"
        f"[ИНСТРУКЦИЯ]\n{NO_CONTEXT_INSTRUCTION}"
    )


# ----------------------------------------------------------------------
# Mock-экстракция (офлайн: --mock и тесты, без сети)
# ----------------------------------------------------------------------
_GOAL_DEBUG_RE = re.compile(r"не находит|не работает|не отвечает|отлад",
                            re.IGNORECASE)
_GOAL_WANT_RE = re.compile(r"хочу\s+(.{5,160})", re.IGNORECASE)
_GOAL_SETUP_RE = re.compile(r"настро|\bкак\s+(?:настроить|подключить)",
                            re.IGNORECASE)
_USES_RE = re.compile(r"использую\s+([\w\-]+(?:\s+[\w\-]+)?)", re.IGNORECASE)
_THRESHOLD_RE = re.compile(r"порог\w*[^0-9\n]{0,24}?(0?\.\d+|\d+)",
                           re.IGNORECASE)

# Термины, которые mock-экстрактор узнаёт в вопросе (слово → определение)
_KNOWN_TERMS = (
    ("rag", "RAG", "Retrieval-Augmented Generation"),
    ("чанк", "chunk", "фрагмент документа"),
    ("chunk", "chunk", "фрагмент документа"),
    ("эмбедд", "embedding", "векторное представление текста"),
    ("embedding", "embedding", "векторное представление текста"),
    ("nomic-embed-text", "nomic-embed-text",
     "локальная модель эмбеддингов Ollama"),
    ("ollama", "Ollama", "локальный сервер LLM"),
    ("chroma", "Chroma", "векторная база данных"),
    ("similarity", "similarity",
     "мера близости запроса и чанка (косинус)"),
)


def mock_task_extraction(question: str, answer: str, task_state: TaskState,
                         chunks: List[RetrievedChunk]) -> Dict[str, Any]:
    """Офлайн-экстракция ChatTurnExtractor (mock-режим, без сети).

    Детерминированные эвристики по вопросу: цель (хочу/настроить/
    отладка), уточнения (русский язык, nomic-embed-text, «использую X»,
    «порог 0.7»), ограничения (без Docker, локальные модели), термины
    (известный словарь). Возвращаются ТОЛЬКО новые элементы — слияние
    всё равно дедуплицируется в TaskState.apply_update. Источники и
    цитаты собираются из чанков (цитаты — точные подстроки, поэтому
    validate_quotes помечает их валидными).
    """
    text = str(question or "")
    low = text.lower()
    update: Dict[str, Any] = {
        "goal": "",
        "clarifications": [],
        "constraints": [],
        "terms": {},
    }

    goal = ""
    want = _GOAL_WANT_RE.search(text)
    debug = _GOAL_DEBUG_RE.search(low)
    if debug:
        # Проблема сформулирована явно — цель (пере)определяется ею
        goal = f"Отладка: {text.strip()}"
    elif want and not task_state.goal:
        # «Хочу …» задаёт цель только пока цели нет: уточняющие вопросы
        # («А если хочу без Docker?») цель не перезаписывают
        goal = f"Хочу {want.group(1).strip()}"
    elif _GOAL_SETUP_RE.search(low) and not task_state.goal:
        goal = text.strip()
    if goal:
        update["goal"] = goal[:160]

    if "русск" in low:
        update["clarifications"].append("документы на русском")
    if "nomic-embed-text" in low:
        update["clarifications"].append("nomic-embed-text как эмбеддер")
    uses = _USES_RE.search(text)
    if uses:
        update["clarifications"].append(
            f"используется {uses.group(1).strip()}")
    threshold = _THRESHOLD_RE.search(text)
    if threshold:
        update["clarifications"].append(
            f"порог similarity {threshold.group(1)}")

    if "docker" in low and re.search(r"без\s+docker", low):
        update["constraints"].append("без Docker")
    if "локальн" in low:
        update["constraints"].append("только локальные модели")

    for needle, term, definition in _KNOWN_TERMS:
        if needle in low:
            update["terms"][term] = definition

    sources = [{
        "chunk_id": chunk.chunk_id,
        "source": chunk.source,
        "section": chunk.section,
    } for chunk in chunks]
    quotes = []
    for chunk in chunks:
        collapsed = " ".join(chunk.text.split())
        if collapsed:
            quotes.append({"chunk_id": chunk.chunk_id,
                           "text": collapsed[:80]})
    return {"sources": sources, "quotes": quotes, "task_state": update}


# ----------------------------------------------------------------------
# JSON-экстрактор (Модуль 3 ТЗ: JSON Mode, Вариант B)
# ----------------------------------------------------------------------
class ChatTurnExtractor:
    """JSON-mode вызов DeepSeek: обновление памяти задачи + цитаты.

    Attributes:
        model: модель служебного запроса (deepseek-v4-flash).
        base_url: базовый URL API (DeepSeek совместим с OpenAI SDK).
        temperature: 0.2 — стабильный JSON.
        max_tokens: 8000 — с запасом (thinking-модель, см. citations).
        mock_mode: True — сеть не используется (mock_task_extraction).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_EXTRACTION_MODEL,
        temperature: float = EXTRACTION_TEMPERATURE,
        max_tokens: int = EXTRACTION_MAX_TOKENS,
        mock_mode: bool = False,
        timeout: float = EXTRACTION_TIMEOUT,
        client: Optional[OpenAI] = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.mock_mode = bool(mock_mode)
        self._timeout = timeout
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

    def extract(self, question: str, answer: str, task_state: TaskState,
                chunks: List[RetrievedChunk]) -> Dict[str, Any]:
        """Обновление памяти + источники/цитаты для одного хода.

        Args:
            question: вопрос пользователя.
            answer: ответ ассистента (уже получен ядром).
            task_state: текущая память задачи (для контекста промпта).
            chunks: чанки RAG-контекста этого хода (может быть пусто).

        Returns:
            {"sources": [...], "quotes": [...], "task_state": {...}}
            (нормализованные типы; валидация цитат — на стороне
            ChatHandler через validate_quotes).

        Raises:
            TaskStateError: сеть/API/дважды пустой ответ/невалидный JSON.
        """
        if self.mock_mode:
            return mock_task_extraction(question, answer, task_state, chunks)
        prompt = TASK_EXTRACTION_PROMPT.format(
            task_state=task_state.to_prompt_block(),
            context=format_context_for_extraction(chunks),
            question=str(question or ""),
            answer=str(answer or ""),
        )
        raw, finish_reason = self._call(prompt, self.max_tokens)
        if not raw.strip() or finish_reason == "length":
            # Бюджет ушёл в reasoning либо JSON обрезан: одна повторная
            # попытка с удвоенным запасом (как у CitationAnswerer)
            logger.warning(
                "JSON-ответ экстрактора пуст или обрезан "
                "(finish_reason=%s) — повтор с удвоенным max_tokens",
                finish_reason)
            raw, finish_reason = self._call(prompt, self.max_tokens * 2)
        if not raw.strip():
            raise TaskStateError("Пустой ответ модели в JSON-режиме")
        try:
            parsed = parse_citation_json(raw)
        except CitationError as exc:
            raise TaskStateError(str(exc)) from exc
        return self._normalize(parsed)

    async def extract_async(self, question: str, answer: str,
                            task_state: TaskState,
                            chunks: List[RetrievedChunk]) -> Dict[str, Any]:
        """Async-вариант extract (SDK синхронный — уход в поток)."""
        return await asyncio.to_thread(
            self.extract, question, answer, task_state, chunks)

    def _call(self, prompt: str, max_tokens: int) -> tuple:
        """Один вызов API: (content, finish_reason).

        Raises:
            TaskStateError: ошибка сети/API.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                # Гарантирует валидный JSON на стороне API (ТЗ: не забыть)
                response_format={"type": "json_object"},
                temperature=self.temperature,
                max_tokens=int(max_tokens),
            )
        except Exception as exc:  # noqa: BLE001 — SDK бросает широкий набор
            raise TaskStateError(
                f"JSON-режим недоступен ({self.model}): {exc}") from exc
        choice = response.choices[0]
        return (str(choice.message.content or ""),
                getattr(choice, "finish_reason", None))

    @staticmethod
    def _normalize(parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Приводит ответ модели к контракту {sources, quotes, task_state}."""
        sources = [dict(item) for item in (parsed.get("sources") or [])
                   if isinstance(item, dict)]
        quotes = []
        for item in parsed.get("quotes") or []:
            if not isinstance(item, dict):
                continue
            try:
                chunk_id = int(item.get("chunk_id"))
            except (TypeError, ValueError):
                chunk_id = item.get("chunk_id")
            quotes.append({"chunk_id": chunk_id,
                           "text": str(item.get("text") or "")})
        return {
            "sources": sources,
            "quotes": quotes,
            "task_state": normalize_task_update(parsed.get("task_state")),
        }


def build_chat_extractor(config: Dict[str, Any]) -> ChatTurnExtractor:
    """Экстрактор из конфигурации приложения (ключ, модель, mock-режим)."""
    return ChatTurnExtractor(
        api_key=str(config.get("DEEPSEEK_API_KEY") or ""),
        base_url=str(config.get("DEEPSEEK_BASE_URL")
                     or DEFAULT_BASE_URL),
        model=str(config.get("DEFAULT_MODEL")
                  or DEFAULT_EXTRACTION_MODEL),
        mock_mode=bool(config.get("MOCK_MODE")),
    )


# ----------------------------------------------------------------------
# Обработчик хода чата (Модуль 4 ТЗ)
# ----------------------------------------------------------------------
def _retrieval_diagnostics(retrieval: Optional[Dict[str, Any]]
                           ) -> Dict[str, Any]:
    """Диагностика поиска для результата хода (+ средний score)."""
    if retrieval is None:
        return {}
    scores = [round(float(score), 4) for score in retrieval["scores"]]
    return {
        "search_query": retrieval["search_query"],
        "rewrite_used": retrieval["rewrite_used"],
        "rewrite_error": retrieval["rewrite_error"],
        "rerank_backend": retrieval["rerank_backend"],
        "rerank_error": retrieval["rerank_error"],
        "scores": scores,
        "avg_score": (round(sum(scores) / len(scores), 4)
                      if scores else None),
    }


def _cited_flags(sources: List[Dict[str, Any]],
                 citations: List[Dict[str, Any]]) -> None:
    """Помечает sources флагом cited: LLM реально сослался на чанк."""
    cited_ids = set()
    for item in citations:
        try:
            cited_ids.add(int(item.get("chunk_id")))
        except (TypeError, ValueError):
            continue
    for source in sources:
        source["cited"] = (source["chunk_id"] in cited_ids
                           if cited_ids else True)


class ChatHandler:
    """Полный цикл хода чата: память задачи → RAG → ответ → обновление.

    Attributes:
        agent: RAGChatAgent (ядро + RAG-пайплайн).
        task_state_repo: репозиторий памяти задачи (None — состояние
            только в памяти хода, например при STORAGE_ENABLED=False).
        extractor: ChatTurnExtractor (None — ход без обновления памяти
            и цитат: деградация, ответ и источники остаются).
        abstain_threshold: порог стратегии C (0.6, nomic-embed-text).
    """

    def __init__(
        self,
        agent: Any,
        task_state_repo: Optional[TaskStateRepository] = None,
        extractor: Optional[ChatTurnExtractor] = None,
        abstain_threshold: float = ABSTAIN_THRESHOLD,
    ) -> None:
        self.agent = agent
        self.task_state_repo = task_state_repo
        self.extractor = extractor
        self.abstain_threshold = float(abstain_threshold)

    # ------------------------------------------------------------------
    # Память задачи
    # ------------------------------------------------------------------
    def load_state(self, session_id: str, profile_id: int) -> TaskState:
        """Память задачи сессии: из SQLite или новая (шаг 1 схемы ТЗ)."""
        if self.task_state_repo is None:
            return TaskState(session_id=str(session_id),
                             profile_id=int(profile_id))
        return self.task_state_repo.get_or_create(
            str(session_id), int(profile_id))

    def reset_session(self, session_id: str) -> bool:
        """Сброс памяти задачи сессии («Новый диалог», «Сбросить»)."""
        if self.task_state_repo is None:
            return False
        return self.task_state_repo.reset(str(session_id))

    # ------------------------------------------------------------------
    # Ход диалога (схема из ТЗ)
    # ------------------------------------------------------------------
    async def handle_message(self, session_id: str, profile_id: int,
                             user_input: str) -> Dict[str, Any]:
        """Обрабатывает сообщение: RAG → гейт → ответ → память задачи.

        Args:
            session_id: UUID сессии диалога.
            profile_id: активный профиль.
            user_input: вопрос пользователя.

        Returns:
            Результат ядра (text/sources/tool_calls/tokens/…) плюс:
            "answer" (то же, что text), "has_rag" (стратегия C — False),
            "quotes" (валидированные), "task_state" (обновлённый словарь),
            "retrieval" (диагностика + avg_score), "rag_error"? ,
            "task_state_error"? (сбой экстрактора, ход не сломан).
        """
        user_input = str(user_input or "")
        # 1. Память задачи
        task_state = self.load_state(session_id, profile_id)

        # 2. RAG-поиск (в чате всегда включён; выключается только
        #    программно — тогда ход идёт как обычный диалог без RAG)
        rag_attempted = bool(getattr(self.agent, "rag_enabled", False))
        rag_error: Optional[str] = None
        retrieval: Optional[Dict[str, Any]] = None
        chunks: List[RetrievedChunk] = []
        if rag_attempted:
            try:
                retrieval = await self.agent.pipeline.retrieve_async(
                    user_input,
                    mode=self.agent.rag_mode,
                    **self.agent._retrieval_params(),  # noqa: SLF001
                )
                chunks = retrieval["chunks"]
            except EmbedderError as exc:
                rag_error = str(exc)

        # 3. Гейт порога (Стратегия C): средний similarity < 0.6
        scores = [float(score) for score in
                  (retrieval["scores"] if retrieval else [])]
        has_rag = bool(
            rag_attempted and rag_error is None
            and not should_abstain(scores, self.abstain_threshold)
        )

        # 4. Промпт: память задачи + RAG-контекст (или пометка) в копию
        #    запроса ядра; история и системный промпт — у ядра
        if rag_attempted:
            if has_rag:
                block = build_rag_turn_block(task_state, user_input, chunks)
                self.agent.core.set_rag_context(
                    block, Retriever.format_sources(chunks))
            else:
                self.agent.core.set_rag_context(
                    build_no_context_block(task_state, rag_error), None)

        # 5. Ответ ядра (инструменты/FSM/токены/персистентность — как раньше)
        result = await asyncio.to_thread(self.agent.core.send_message,
                                         user_input)
        if rag_error is not None:
            result["rag_error"] = rag_error
        if retrieval is not None:
            result["retrieval"] = _retrieval_diagnostics(retrieval)
        if rag_attempted:
            result["has_rag"] = has_rag
            result["rag_used"] = has_rag

        # 6-7. Обновление памяти задачи (JSON Mode) + цитаты
        if not result.get("error"):
            await self._apply_task_state(session_id, task_state, user_input,
                                         result, chunks, retrieval, has_rag)

        result["answer"] = str(result.get("text") or "")
        result["task_state"] = task_state.to_dict()
        return result

    # ------------------------------------------------------------------
    # Шаги 6-7: память задачи через JSON Mode (Вариант B)
    # ------------------------------------------------------------------
    async def _apply_task_state(self, session_id: str,
                                task_state: TaskState,
                                question: str, result: Dict[str, Any],
                                chunks: List[RetrievedChunk],
                                retrieval: Optional[Dict[str, Any]],
                                has_rag: bool) -> None:
        """Экстракция и сохранение памяти задачи, цитаты, флаги cited.

        Память задачи обновляется на КАЖДОМ ходу (ТЗ: «память обновляется
        после каждого сообщения»), источники с цитатами — только при
        RAG-ходе (стратегия C отвечает без источников). Сбой экстрактора
        не ломает ход: ответ остаётся, причина — в "task_state_error".
        """
        if self.extractor is None:
            return
        answer = str(result.get("text") or "")
        try:
            parsed = await self.extractor.extract_async(
                question, answer, task_state, chunks)
        except Exception as exc:  # noqa: BLE001 — ход не должен падать
            logger.warning("Экстрактор памяти задачи не сработал: %s", exc)
            result["task_state_error"] = str(exc)
            result["quotes"] = []
            return

        # 7. Мердж и сохранение обновлённой памяти задачи
        task_state.apply_update(parsed.get("task_state") or {})
        if self.task_state_repo is not None:
            self.task_state_repo.update(task_state)
            # messages текущего диалога помечаются session_id сессии
            self.task_state_repo.tag_session_messages(session_id)

        # Цитаты/источники — только на RAG-ходе; стратегия C отвечает
        # «без источников» (пометка в UI добавляет ChatView)
        result["quotes"] = []
        if has_rag and chunks:
            quotes = validate_quotes(
                {"quotes": parsed.get("quotes") or []},
                [chunk.text for chunk in chunks],
                retrieval["metadatas"] if retrieval else [],
            )["quotes"]
            result["quotes"] = quotes
            result["citations"] = parsed.get("sources") or []
            sources = Retriever.format_sources(chunks)
            _cited_flags(sources, result["citations"])
            result["sources"] = sources
