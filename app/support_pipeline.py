"""Пайплайн команды /support: CRM-контекст + RAG по FAQ + LLM.

Схема обработки (ТЗ ai_1_34):

    /support Почему не работает авторизация?
      → parse_command → ("support", "Почему не работает авторизация?")
      → SupportRAG.search_chunks_async (коллекция support_faq)
      → MCPClient.call_tool("get_user_context", {"user_id": …})
        через собственный HTTP-сервер crm_server.py (порт 8004)
      → промпт: [инструкция] + [Контекст пользователя] + [FAQ]
        + [Вопрос] — уходит в ядро через set_rag_context (блок заменяет
        последнее user-сообщение запроса, история чистая)
      → ответ ядра + источники (панель под ответом, как в обычном RAG)

Фолбэк (ТЗ «ЧЕГО НЕ ДЕЛАТЬ»): недоступный CRM (сервер не поднят,
таймаут), пользователь не выбран в UI или не найден в CRM НЕ ломают
ответ — в промпт уходит пометка («Контекст пользователя недоступен»),
/support отвечает только по FAQ (crm_available=False, причина — в
crm_error результата).

Режим изолированный — как /help: ход выполняется на ОТДЕЛЬНОМ ядре без
хранилища (isolated_core RAGChatAgent) — история основного диалога,
память задачи (task state) и FSM задач не затрагиваются. Перед каждым
ходом история изолированного ядра очищается.

Отказоустойчивость: недоступный эмбеддер НЕ ломает ответ — в промпт
уходит пометка вместо FAQ, причина — в rag_error результата. LLM
вызывается всегда (кроме пустого вопроса).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from app.rag.embedder import EmbedderError
from app.rag.retriever import RetrievedChunk, Retriever
from src.utils import get_logger

logger = get_logger("deepseek-chat.support")

# Системная инструкция пайплайна (ТЗ)
SUPPORT_SYSTEM_INSTRUCTION = (
    "Ты — ассистент службы поддержки. Отвечай на вопрос пользователя, "
    "используя FAQ и контекст его тикетов."
)

# Шаблон промпта (ТЗ): блок заменяет последнее user-сообщение запроса
# (вопрос включён в блок — дублирования нет)
SUPPORT_PROMPT_TEMPLATE = """{instruction}

[Контекст пользователя]
{user_context}

[FAQ и документация]
{faq_context}

[Вопрос]
{question}

Ответь на вопрос, ссылаясь на источники. Если информации нет — скажи об этом."""

# Пометки вместо контекста при деградации (фолбэк ТЗ)
NO_FAQ_MARKER = "(FAQ и документация поддержки не найдены или недоступны)"
CRM_UNAVAILABLE_MARKER = "Контекст пользователя недоступен: {reason}"
NO_USER_MARKER = "Пользователь не выбран — контекст CRM недоступен."

# Подсказка при пустом вопросе (/support без текста)
USAGE_HINT = (
    "Команда /support отвечает на вопросы пользователей по FAQ и "
    "документации поддержки с контекстом CRM (тикеты, статус). Выберите "
    "пользователя в списке над полем ввода и напишите вопрос после "
    "команды, например: /support Почему не работает авторизация?"
)


def parse_user_context(text: str) -> Dict[str, Any]:
    """Контекст пользователя из ответа MCP-клиента.

    FastMCP оборачивает результат в structuredContent
    ``{"result": {...}}`` — клиент возвращает его pretty-JSON'ом;
    принимается и сырая строка (устойчивость к версии SDK). Ошибки
    сервера (``{"error": …}``) возвращаются как есть — решение о
    деградации принимает SupportPipeline.
    """
    raw = str(text or "").strip()
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    if isinstance(payload, dict) and set(payload) == {"result"}:
        payload = payload["result"]
    return payload if isinstance(payload, dict) else {}


class SupportPipeline:
    """Обработчик команды /support: CRM + FAQ + LLM.

    Attributes:
        support_rag: SupportRAG (коллекция support_faq).
        mcp_client: HTTP-клиент CRM-сервера (crm_server.py, порт 8004).
        agent: ядро для ответа — изолированный DeepSeekAgent без
            хранилища (или RAGChatAgent, тогда берётся его .core).
        top_k: чанков FAQ в контексте.
    """

    def __init__(self, support_rag: Any, mcp_client: Any, agent: Any,
                 top_k: int = 3) -> None:
        self.support_rag = support_rag
        self.mcp_client = mcp_client
        self.agent = agent
        self.top_k = max(1, int(top_k))

    def _core(self) -> Any:
        """Ядро для хода /support (аналог /help).

        Приоритет — изолированное ядро без хранилища (isolated_core
        RAGChatAgent): история основного чата, память задачи и FSM не
        затрагиваются. Нет изолированного (агент-заглушка тестов без
        bind_config) — основной .core: деградация, ответ корректен.
        """
        isolated = getattr(self.agent, "isolated_core", None)
        if callable(isolated):
            try:
                return isolated()
            except RuntimeError as exc:
                logger.warning(
                    "/support: изолированное ядро недоступно (%s) — ход "
                    "пойдёт через основное", exc)
        return getattr(self.agent, "core", self.agent)

    async def handle_support(self, question: str,
                             user_id: int) -> Dict[str, Any]:
        """Отвечает на вопрос пользователя поддержки (полный пайплайн ТЗ).

        Args:
            question: текст вопроса ПОСЛЕ команды (payload парсера).
            user_id: ID пользователя CRM (из выпадающего списка UI;
                0 — не выбран: ход идёт без CRM-контекста).

        Returns:
            {"text"/"answer", "sources" (метаданные чанков FAQ),
            "user_context" (строка для промпта), "crm_available",
            "user_id", "rag_used", "has_rag", "question",
            "is_support": True; опционально "rag_error"/"crm_error"}.
            Пустой вопрос → подсказка по использованию (без LLM/MCP/RAG).
        """
        question = str(question or "").strip()
        if not question:
            return {
                "text": USAGE_HINT,
                "answer": USAGE_HINT,
                "sources": [],
                "user_context": "",
                "crm_available": False,
                "user_id": int(user_id or 0),
                "rag_used": False,
                "has_rag": False,
                "question": "",
                "is_support": True,
            }

        # 1. RAG по FAQ поддержки (недоступный эмбеддер не ломает ход)
        chunks: List[RetrievedChunk] = []
        rag_error: Optional[str] = None
        try:
            chunks = await self.support_rag.search_chunks_async(
                question, top_k=self.top_k)
        except EmbedderError as exc:
            rag_error = str(exc)
            logger.warning("/support: поиск по support_faq не удался: %s",
                           exc)

        # 2. Контекст пользователя через CRM (фолбэк — без него)
        user_context, crm_error, crm_available = await self._user_context(
            user_id)

        # 3. Промпт: инструкция + контекст пользователя + FAQ + вопрос
        faq_context = (
            "\n\n".join(chunk.text for chunk in chunks) or NO_FAQ_MARKER)
        prompt = SUPPORT_PROMPT_TEMPLATE.format(
            instruction=SUPPORT_SYSTEM_INSTRUCTION,
            user_context=user_context,
            faq_context=faq_context,
            question=question,
        )

        # 4. Ответ изолированного ядра (история/память задачи не трогаются)
        core = self._core()
        core.clear_history()
        core.set_rag_context(prompt, Retriever.format_sources(chunks))
        result = await asyncio.to_thread(core.send_message, question)

        # 5. Диагностика результата (панель источников рендерит ChatView)
        text = str(result.get("text") or "")
        result["answer"] = text
        result["question"] = question
        result["is_support"] = True
        result["user_context"] = user_context
        result["crm_available"] = crm_available
        result["user_id"] = int(user_id or 0)
        result["rag_used"] = bool(chunks)
        result["has_rag"] = bool(chunks)
        if rag_error is not None:
            result["rag_error"] = rag_error
        if crm_error is not None:
            result["crm_error"] = crm_error
        if not text and not result.get("error"):
            result["text"] = (
                "Не удалось получить ответ: модель не вернула текст. "
                "Попробуйте ещё раз."
            )
            result["answer"] = result["text"]
        return result

    async def _user_context(self, user_id: int) -> tuple:
        """Контекст пользователя для промпта: (строка, ошибка, доступен).

        Любая ошибка (CRM не поднят, таймаут, пользователь не выбран или
        не найден) превращается в пометку — /support отвечает и без CRM
        (ТЗ: фолбэк обязателен).
        """
        user_id = int(user_id or 0)
        if user_id <= 0:
            return (NO_USER_MARKER, "пользователь не выбран", False)
        try:
            raw = await self.mcp_client.call_tool(
                "get_user_context", {"user_id": user_id})
            context = parse_user_context(raw)
        except Exception as exc:  # noqa: BLE001 — деградация вместо падения
            logger.warning("/support: CRM недоступен: %s", exc)
            reason = str(exc)
            return (
                CRM_UNAVAILABLE_MARKER.format(reason=reason), reason, False)
        if not context:
            reason = "CRM вернул некорректный ответ"
            return (
                CRM_UNAVAILABLE_MARKER.format(reason=reason), reason, False)
        formatted = self._format_user_context(context)
        # {"error": …} — CRM ответил, но контекста нет (пользователь
        # не найден): для UI это «CRM недоступен»
        crm_available = "error" not in context
        return (formatted, (None if crm_available else formatted),
                crm_available)

    @staticmethod
    def _format_user_context(context: Dict[str, Any]) -> str:
        """Форматирует контекст пользователя для промпта (ТЗ)."""
        if "error" in context:
            return str(context["error"])

        user = context.get("user") or {}
        tickets = context.get("tickets") or []

        parts = [
            f"Пользователь: {user.get('name', 'N/A')} "
            f"({user.get('email', 'N/A')})",
            f"Статус: {user.get('status', 'N/A')}",
        ]
        if tickets:
            parts.append("Тикеты:")
            for ticket in tickets:
                parts.append(
                    f"  • #{ticket.get('id')}: {ticket.get('subject', '')} "
                    f"({ticket.get('status', '')}, "
                    f"{ticket.get('priority', '')})")
        else:
            parts.append("Тикеты: нет")
        return "\n".join(parts)
