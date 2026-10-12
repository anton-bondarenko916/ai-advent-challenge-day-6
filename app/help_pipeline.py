"""Пайплайн команды /help: RAG по документации проекта + git-контекст.

Схема обработки (ТЗ):

    /help как устроена архитектура?
      → parse_command → ("help", "как устроена архитектура?")
      → ProjectRAG.search_chunks_async (коллекция project_docs)
      → MCPClient.call_tool("get_current_branch", {"repo_path": …})
        через собственный stdio-сервер my_git_server.py
      → промпт: [инструкция] + [Документация проекта] + [Git-контекст]
        + [Вопрос] — уходит в ядро через set_rag_context (блок заменяет
        последнее user-сообщение запроса, история чистая)
      → ответ ядра + источники (панель под ответом, как в обычном RAG)

Режим изолированный: /help выполняется на ОТДЕЛЬНОМ ядре без
хранилища (isolated_core RAGChatAgent, аналог answer_once) — история
основного диалога, память задачи (task state) и FSM задач не
затрагиваются. Перед каждым ходом история изолированного ядра
очищается.

Отказоустойчивость: недоступный git MCP (сервер не поднялся, таймаут)
и недоступный эмбеддер НЕ ломают ответ — в промпт уходит пометка
(«Git-контекст недоступен: …»), причина — в rag_error/git_error
результата. LLM вызывается всегда (кроме пустого вопроса).
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, Dict, List, Optional

from app.rag.embedder import EmbedderError
from app.rag.retriever import RetrievedChunk, Retriever
from src.config import PROJECT_ROOT
from src.mcp_client import MCPClient
from src.utils import get_logger

logger = get_logger("deepseek-chat.help")

# Имя скрипта собственного git MCP-сервера (stdio, корень проекта)
GIT_SERVER_SCRIPT = "my_git_server.py"

# Таймаут вызова get_current_branch: локальный subprocess с git —
# быстро, но первый запуск python-процесса на Windows небыстрый
GIT_MCP_TIMEOUT = 30.0

# Системная инструкция пайплайна (ТЗ)
HELP_SYSTEM_INSTRUCTION = (
    "Ты — ассистент разработчика. Отвечай на вопросы о проекте, "
    "используя документацию и git-контекст."
)

# Шаблон промпта (ТЗ): блок заменяет последнее user-сообщение запроса
# (вопрос включён в блок — дублирования нет)
HELP_PROMPT_TEMPLATE = """{instruction}

[Документация проекта]
{context}

[Git-контекст]
{git_context}

[Вопрос]
{question}

Ответь на вопрос, ссылаясь на источники. Если информации нет — скажи об этом."""

# Пометки вместо контекста при деградации
NO_DOCS_MARKER = "(документация проекта не найдена или недоступна)"
GIT_UNAVAILABLE_MARKER = "Git-контекст недоступен: {reason}"

# Подсказка при пустом вопросе (/help без текста)
USAGE_HINT = (
    "Команда /help отвечает на вопросы о проекте по его документации "
    "(README, docs/, конфиги) с git-контекстом. Напишите вопрос после "
    "команды, например: /help как устроена архитектура?"
)


def build_git_mcp_client(config: Optional[Dict[str, Any]] = None,
                         timeout: float = GIT_MCP_TIMEOUT) -> MCPClient:
    """Stdio-клиент собственного git MCP-сервера.

    Команда: ``python -u my_git_server.py`` (sys.executable — тот
    интерпретатор, под которым работает приложение; ``-u`` обязателен,
    иначе stdout подпроцесса буферизуется). Каждый вызов — отдельная
    сессия (запуск → initialize → call → завершение), как в панели MCP.

    Args:
        config: конфигурация приложения (зарезервировано; клиент
            git-сервера не зависит от настроек панели MCP).
        timeout: таймаут handshake и tools/call, секунды.
    """
    return MCPClient(
        server_command=[sys.executable, "-u",
                        str(PROJECT_ROOT / GIT_SERVER_SCRIPT)],
        timeout=timeout,
    )


def parse_branch_text(text: str) -> str:
    """Имя ветки из ответа MCP-клиента.

    FastMCP оборачивает строковый результат в structuredContent
    ``{"result": "..."}`` — клиент возвращает его pretty-JSON'ом;
    принимается и сырая строка (устойчивость к версии SDK). Ошибки
    сервера («Путь не существует: …») возвращаются как есть — решение
    о деградации принимает HelpPipeline.
    """
    raw = str(text or "").strip()
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return raw
    if isinstance(payload, dict) and "result" in payload:
        return str(payload["result"]).strip()
    return raw


class HelpPipeline:
    """Обработчик команды /help: документация проекта + git + LLM.

    Attributes:
        project_rag: ProjectRAG (коллекция project_docs).
        mcp_client: MCPClient git-сервера (my_git_server.py, stdio).
        agent: ядро для ответа — изолированный DeepSeekAgent без
            хранилища (или RAGChatAgent, тогда берётся его .core).
        project_path: абсолютный путь к проекту (репозиторий).
        top_k: чанков документации в контексте.
    """

    def __init__(self, project_rag: Any, mcp_client: Any, agent: Any,
                 project_path: str, top_k: int = 3) -> None:
        self.project_rag = project_rag
        self.mcp_client = mcp_client
        self.agent = agent
        self.project_path = str(project_path)
        self.top_k = max(1, int(top_k))

    def _core(self) -> Any:
        """Ядро для хода /help.

        Приоритет — изолированное ядро без хранилища (isolated_core
        RAGChatAgent): история основного чата, память задачи и FSM не
        затрагиваются, ядро кэшируется внутри агента. Нет изолированного
        (агент-заглушка тестов без bind_config) — основной .core:
        деградация, ответ корректен, но ход виден в истории.
        """
        isolated = getattr(self.agent, "isolated_core", None)
        if callable(isolated):
            try:
                return isolated()
            except RuntimeError as exc:
                logger.warning(
                    "/help: изолированное ядро недоступно (%s) — ход "
                    "пойдёт через основное", exc)
        return getattr(self.agent, "core", self.agent)

    async def handle_help(self, question: str) -> Dict[str, Any]:
        """Отвечает на вопрос о проекте (полный пайплайн ТЗ).

        Args:
            question: текст вопроса ПОСЛЕ команды (payload парсера).

        Returns:
            {"text"/"answer", "sources" (метаданные чанков),
            "git_context" (строка для промпта), "rag_used",
            "has_rag", "question", "is_help": True;
            опционально "rag_error"/"git_error"}.
            Пустой вопрос → подсказка по использованию (без LLM и MCP).
        """
        question = str(question or "").strip()
        if not question:
            return {
                "text": USAGE_HINT,
                "answer": USAGE_HINT,
                "sources": [],
                "git_context": "",
                "rag_used": False,
                "has_rag": False,
                "question": "",
                "is_help": True,
            }

        # 1. RAG по документации проекта
        chunks: List[RetrievedChunk] = []
        rag_error: Optional[str] = None
        try:
            chunks = await self.project_rag.search_chunks_async(
                question, top_k=self.top_k)
        except EmbedderError as exc:
            rag_error = str(exc)
            logger.warning("/help: поиск по project_docs не удался: %s", exc)

        # 2. Git-контекст через MCP (недоступность не ломает ответ)
        git_context, git_error = await self._git_context()

        # 3. Промпт: инструкция + документация + git + вопрос
        context = "\n\n".join(chunk.text for chunk in chunks) or NO_DOCS_MARKER
        prompt = HELP_PROMPT_TEMPLATE.format(
            instruction=HELP_SYSTEM_INSTRUCTION,
            context=context,
            git_context=git_context,
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
        result["is_help"] = True
        result["rag_used"] = bool(chunks)
        result["has_rag"] = bool(chunks)
        result["git_context"] = git_context
        if rag_error is not None:
            result["rag_error"] = rag_error
        if git_error is not None:
            result["git_error"] = git_error
        if not text and not result.get("error"):
            result["text"] = (
                "Не удалось получить ответ: модель не вернула текст. "
                "Попробуйте ещё раз."
            )
            result["answer"] = result["text"]
        return result

    async def _git_context(self) -> tuple:
        """Git-контекст для промпта: (строка, ошибка или None).

        Вызывает get_current_branch через MCP; ЛЮБАЯ ошибка (сервер не
        поднялся, таймаут, путь не существует) превращается в пометку —
        /help отвечает и без ветки (ТЗ: fallback обязателен).
        """
        try:
            raw = await self.mcp_client.call_tool(
                "get_current_branch", {"repo_path": self.project_path})
            branch = parse_branch_text(raw)
        except Exception as exc:  # noqa: BLE001 — деградация вместо падения
            logger.warning("/help: git-контекст недоступен: %s", exc)
            reason = str(exc)
            return (GIT_UNAVAILABLE_MARKER.format(reason=reason), reason)
        if not branch:
            branch = "(не определена)"
        return (f"Текущая git-ветка: {branch}", None)
