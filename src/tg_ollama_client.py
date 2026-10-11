"""Асинхронный клиент Ollama для Telegram-бота (ТЗ ai_1_31).

Бот говорит с локальной моделью напрямую (``POST /api/chat``), минуя
ядро DeepSeekAgent: бот многопользовательский, а ядро — однохоловой
singleton с историей одной сессии. Ответы НЕ стримятся (``stream:
false``) — Telegram всё равно не умеет потоковую доставку, стриминг с
редактированием сообщения — полировка v2.

Таймауты: локальная модель с холодной загрузкой (5+ ГБ с диска) может
думать минутами, поэтому read = 300 с — generous по ТЗ; connect/пул —
обычные. Один ``httpx.AsyncClient`` на процесс (создаётся в
конструкторе, закрывается ``aclose()``).

Любая сетевая проблема (сайт не запущен, таймаут, DNS) оборачивается в
:class:`OllamaUnavailableError` — наверх превращается в дружелюбное
сообщение пользователю, а не в падение цикла polling.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx

from .utils import get_logger

logger = get_logger("deepseek-chat.tg-ollama")

DEFAULT_OLLAMA_BASE = "http://localhost:11434"
DEFAULT_READ_TIMEOUT = 300.0


class OllamaChatError(RuntimeError):
    """Ошибка на стороне сервиса Ollama (поле error, 4xx/5xx, кривой JSON)."""


class OllamaUnavailableError(OllamaChatError):
    """Ollama недоступна (не запущен сервер, сеть, таймаут)."""


class OllamaChatClient:
    """Клиент нативного API Ollama: /api/chat и /api/tags.

    Args:
        base_url: база Ollama; None — OLLAMA_BASE_URL или localhost:11434.
        model_default: модель по умолчанию (TELEGRAM_MODEL; используется
            логикой ChatTurn, клиенту не нужен — оставлен для подписи).
        read_timeout: таймаут чтения ответа (генерация может быть долгой).
        transport: httpx-транспорт для тестов (MockTransport); None —
            обычный сетевой транспорт.
    """

    def __init__(self, base_url: Optional[str] = None,
                 model_default: str = "",
                 read_timeout: float = DEFAULT_READ_TIMEOUT,
                 transport: Optional[httpx.BaseTransport] = None) -> None:
        self.base_url = (str(base_url) if base_url else os.getenv(
            "OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE)).rstrip("/")
        self.model_default = str(model_default)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=read_timeout,
                                  write=30.0, pool=30.0),
            transport=transport,
        )

    async def list_models(self) -> List[str]:
        """Установленные модели (GET /api/tags → имена).

        Raises:
            OllamaUnavailableError: сервер не отвечает.
            OllamaChatError: HTTP-ошибка или кривой JSON.
        """
        response = await self._request("GET", "/api/tags")
        try:
            models = response.json().get("models") or []
        except ValueError as exc:
            raise OllamaChatError(
                f"Кривой ответ /api/tags: {exc}") from exc
        return [str(item.get("name")) for item in models if item.get("name")]

    async def chat(self, model: str, messages: List[Dict[str, str]]) -> str:
        """Один ход диалога (POST /api/chat, без стриминга).

        Args:
            model: имя модели (``qwen2.5:7b``).
            messages: история в формате Ollama
                ([{"role", "content"}], включая новое user-сообщение).

        Returns:
            Текст ответа ассистента.

        Raises:
            OllamaUnavailableError: сервер не отвечает/таймаут.
            OllamaChatError: HTTP-ошибка, поле error в ответе, кривой
                JSON или пустой content.
        """
        payload = {
            "model": str(model),
            "messages": list(messages or []),
            "stream": False,
        }
        response = await self._request("POST", "/api/chat", json=payload)
        try:
            data: Dict[str, Any] = response.json()
        except ValueError as exc:
            raise OllamaChatError(f"Кривой ответ /api/chat: {exc}") from exc
        error = data.get("error")
        if error:
            raise OllamaChatError(str(error))
        content = (data.get("message") or {}).get("content")
        if not content:
            raise OllamaChatError(
                "Пустой ответ модели (весь бюджет ушли в reasoning?)")
        return str(content)

    async def aclose(self) -> None:
        """Закрывает httpx-клиент (при остановке бота)."""
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Внутреннее
    # ------------------------------------------------------------------
    async def _request(self, method: str, path: str,
                       **kwargs: Any) -> httpx.Response:
        """Запрос с переводом сетевых проблем в OllamaUnavailableError."""
        url = f"{self.base_url}{path}"
        try:
            response = await self._client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            logger.warning("Ollama недоступна (%s): %s", url, exc)
            raise OllamaUnavailableError(
                f"сервер не отвечает ({exc.__class__.__name__})") from exc
        if response.status_code != 200:
            raise OllamaChatError(
                f"HTTP {response.status_code}: "
                f"{response.text[:200]}")
        return response
