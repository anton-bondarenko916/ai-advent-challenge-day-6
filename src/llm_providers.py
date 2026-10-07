"""Абстракция LLM-провайдеров: облако DeepSeek и локальная Ollama (ТЗ ai_1_25).

Ключевой момент: Ollama предоставляет OpenAI-совместимый эндпоинт на
``http://localhost:11434/v1`` — тот же SDK ``openai.OpenAI`` работает с
ним без изменений, меняются только ``base_url`` и ``api_key`` (ключ
обязателен для SDK, но Ollama его игнорирует).

Провайдер — источник клиента, модели и base_url для ядра
(:meth:`src.agent.DeepSeekAgent.switch_provider`); base_url нигде не
хардкодится в запросах — он берётся из конфигурации провайдера
(переменные окружения DEEPSEEK_BASE_URL / OLLAMA_BASE_URL).

Локальные модели могут не поддерживать tool calling: OllamaProvider
проверяет capability ``tools`` через ``/api/show`` и сообщает ядру —
без поддержки инструменты MCP не отправляются в запрос.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_OLLAMA_API_BASE = "http://localhost:11434"
OLLAMA_TOOLS_TIMEOUT = 5.0
OLLAMA_TAGS_TIMEOUT = 3.0

# Семейство моделей DeepSeek (владение моделью проверяет профиль:
# настройки профиля не должны подменять модель чужого провайдера)
DEEPSEEK_MODELS = ("deepseek-v4-flash", "deepseek-v4-pro")


def _deepseek_base_url() -> str:
    """base_url DeepSeek из окружения (по умолчанию облако)."""
    return os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL).rstrip("/")


def _ollama_api_base() -> str:
    """База Ollama без /v1 (API-эндпоинты /api/tags, /api/show)."""
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_API_BASE).rstrip("/")


def _model_base_name(model: str) -> str:
    """Имя модели без тега (``qwen2.5:7b`` → ``qwen2.5``)."""
    return str(model or "").split(":", 1)[0].strip().lower()


class ProviderUnavailableError(RuntimeError):
    """Провайдер недоступен (локальный сервер не запущен / модели нет)."""


class BaseLLMProvider(ABC):
    """Базовый интерфейс LLM-провайдера.

    Attributes:
        id: машинное имя провайдера (ключ LLMProviderFactory.PROVIDERS).
        name: человекочитаемое имя (для UI и логов).
        model: модель чата по умолчанию.
        base_url: адрес API (не хардкодится в ядре — берётся отсюда).
        client: готовый клиент openai.OpenAI (ядро использует напрямую).
        local: True — локальный провайдер (MCP-инструменты не отправляются
            в запрос независимо от capability модели: небольшие локальные
            модели склонны к ложным вызовам инструментов, например
            search_news на вопрос «факт о космосе»).
    """

    id: str = ""
    name: str = ""
    local: bool = False

    def __init__(self, model: str, base_url: str, client: OpenAI) -> None:
        self.model = str(model)
        self.base_url = str(base_url)
        self.client = client

    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, Any]],
                        **kwargs: Any) -> Dict[str, Any]:
        """Выполняет запрос к LLM.

        Args:
            messages: сообщения в формате OpenAI.
            **kwargs: остальные параметры chat.completions.create.

        Returns:
            ``{"content": str, "tool_calls": list|None, "usage": ...}``.
        """

    @abstractmethod
    def supports_tool_calling(self) -> bool:
        """Поддерживает ли модель tool calling (false → MCP отключается)."""

    @abstractmethod
    def is_available(self) -> bool:
        """Доступен ли провайдер (для локальных — запущен ли сервер)."""

    @abstractmethod
    def owns_model(self, model: str) -> bool:
        """Относится ли модель к семейству этого провайдера.

        Используется ядром, чтобы настройки профиля (хранящие модель
        DeepSeek) не подменяли модель активного провайдера.
        """

    @abstractmethod
    def model_choices(self) -> tuple:
        """Модели, доступные для выбора в настройках при этом провайдере."""

    def describe(self) -> str:
        """Строка для статус-бара: «DeepSeek (облако) · deepseek-v4-flash»."""
        return f"{self.name} · {self.model}"


class DeepSeekProvider(BaseLLMProvider):
    """Провайдер DeepSeek API (облако)."""

    id = "deepseek"
    name = "DeepSeek (облако)"

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "deepseek-v4-flash",
                 base_url: Optional[str] = None) -> None:
        """Создаёт клиента OpenAI SDK поверх DeepSeek API.

        Args:
            api_key: ключ DeepSeek API; None/пусто — DEEPSEEK_API_KEY из
                окружения; нет и его — заглушка «missing-api-key» (SDK
                требует непустой ключ; ошибка аутентификации проявится
                на ходе, а не падением переключения).
            model: модель чата (deepseek-v4-flash / deepseek-v4-pro).
            base_url: адрес API; None — DEEPSEEK_BASE_URL или облако.
        """
        resolved_key = (
            str(api_key or "").strip()
            or os.getenv("DEEPSEEK_API_KEY", "")
            or "missing-api-key"
        )
        url = base_url or _deepseek_base_url()
        super().__init__(
            model=model,
            base_url=url,
            client=OpenAI(
                api_key=resolved_key,
                base_url=url,
                timeout=60.0,
                max_retries=1,
            ),
        )

    def chat_completion(self, messages: List[Dict[str, Any]],
                        **kwargs: Any) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        message = response.choices[0].message
        return {
            "content": message.content,
            "tool_calls": getattr(message, "tool_calls", None),
            "usage": response.usage,
        }

    def supports_tool_calling(self) -> bool:
        return True

    def is_available(self) -> bool:
        return True  # облако считается доступным (ошибки сети — на ходе)

    def owns_model(self, model: str) -> bool:
        return str(model) in DEEPSEEK_MODELS

    def model_choices(self) -> tuple:
        return DEEPSEEK_MODELS


class OllamaProvider(BaseLLMProvider):
    """Провайдер локальной Ollama (OpenAI-совместимый API на /v1)."""

    id = ""
    name = "Ollama"
    # Локальные модели работают без MCP-инструментов (политика ядра:
    # запрос идёт только к модели — см. DeepSeekAgent.switch_provider)
    local = True

    def __init__(self, model: str = "qwen2.5:7b",
                 base_url: Optional[str] = None,
                 api_base: Optional[str] = None) -> None:
        """Создаёт клиента поверх локального сервера Ollama.

        Args:
            model: имя модели в Ollama (например, ``qwen2.5:7b``).
            base_url: OpenAI-совместимый эндпоинт (…/v1); None —
                OLLAMA_BASE_URL + /v1.
            api_base: база нативного API (…/api/…); None — OLLAMA_BASE_URL.
        """
        self.api_base = (api_base or _ollama_api_base()).rstrip("/")
        url = (base_url or f"{self.api_base}/v1").rstrip("/")
        # api_key обязателен для SDK, но Ollama его игнорирует
        super().__init__(
            model=model,
            base_url=url,
            client=OpenAI(
                api_key="ollama",
                base_url=url,
                timeout=120.0,  # локальный инференс медленнее облака
                max_retries=1,
            ),
        )

    def chat_completion(self, messages: List[Dict[str, Any]],
                        **kwargs: Any) -> Dict[str, Any]:
        # Убираем параметры облачных API, которые Ollama может не принять
        kwargs.pop("reasoning_effort", None)
        kwargs.pop("thinking", None)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        message = response.choices[0].message
        return {
            "content": message.content,
            "tool_calls": getattr(message, "tool_calls", None),
            "usage": response.usage,
        }

    def supports_tool_calling(self) -> bool:
        """Проверяет capability ``tools`` модели через /api/show.

        Ошибка сервера/сети трактуется как «не поддерживает»: ход не
        должен падать из-за проверки, а инструменты при живом сервере
        с tool-моделью будут доступны после следующей проверки.
        """
        try:
            response = httpx.post(
                f"{self.api_base}/api/show",
                json={"model": self.model},
                timeout=OLLAMA_TOOLS_TIMEOUT,
            )
        except Exception:
            return False
        if response.status_code != 200:
            return False
        try:
            capabilities = response.json().get("capabilities") or []
        except ValueError:
            return False
        return "tools" in capabilities

    def is_available(self) -> bool:
        """Сервер запущен и модель присутствует (по /api/tags)."""
        try:
            response = httpx.get(
                f"{self.api_base}/api/tags", timeout=OLLAMA_TAGS_TIMEOUT)
        except Exception:
            return False
        if response.status_code != 200:
            return False
        try:
            models = response.json().get("models") or []
        except ValueError:
            return False
        base = _model_base_name(self.model)
        return any(base and str(item.get("name", "")).lower().startswith(base)
                   for item in models)

    def owns_model(self, model: str) -> bool:
        return _model_base_name(model) == _model_base_name(self.model)

    def model_choices(self) -> tuple:
        return (self.model,)


class LLMProviderFactory:
    """Создаёт провайдера по имени (реестр PROVIDERS — источник конфигов).

    Каждый элемент реестра: ``name`` (подпись в UI), ``model`` (модель
    чата), ``requires_ollama`` (нужен ли локальный сервер). base_url
    внутрь реестра не кладётся — он вычисляется провайдером из
    окружения (DEEPSEEK_BASE_URL / OLLAMA_BASE_URL).
    """

    PROVIDERS: Dict[str, Dict[str, Any]] = {
        "deepseek": {
            "name": DeepSeekProvider.name,
            "model": "deepseek-v4-flash",
            "requires_ollama": False,
        },
        "ollama_qwen": {
            "name": "Ollama — Qwen 2.5 7B",
            "model": "qwen2.5:7b",
            "requires_ollama": True,
        },
        "ollama_llama": {
            "name": "Ollama — Llama 3.1 8B",
            "model": "llama3.1:8b",
            "requires_ollama": True,
        },
    }

    @classmethod
    def spec(cls, provider_name: str) -> Dict[str, Any]:
        """Описание провайдера из реестра.

        Raises:
            ValueError: неизвестное имя провайдера.
        """
        spec = cls.PROVIDERS.get(str(provider_name or ""))
        if spec is None:
            known = ", ".join(sorted(cls.PROVIDERS))
            raise ValueError(
                f"Неизвестный провайдер: {provider_name!r}. "
                f"Доступные: {known}")
        return spec

    @classmethod
    def create(cls, provider_name: str,
               api_key: Optional[str] = None) -> BaseLLMProvider:
        """Создаёт провайдера по имени.

        Args:
            provider_name: ключ реестра (deepseek / ollama_qwen / …).
            api_key: ключ DeepSeek API; None — взять DEEPSEEK_API_KEY
                из окружения (для Ollama не используется).

        Returns:
            Готовый провайдер (BaseLLMProvider).

        Raises:
            ValueError: неизвестное имя провайдера.
        """
        spec = cls.spec(provider_name)
        if provider_name == "deepseek":
            # api_key=None — DeepSeekProvider сам берёт ключ из окружения
            return DeepSeekProvider(
                api_key=api_key,
                model=spec["model"],
            )
        # id/name инстансу: ключ реестра знает только фабрика
        # (ollama_qwen / ollama_llama), подпись — как в UI
        provider = OllamaProvider(model=spec["model"])
        provider.id = provider_name
        provider.name = spec["name"]
        return provider

    @classmethod
    def get_available_providers(cls) -> List[Dict[str, Any]]:
        """Список провайдеров с признаком доступности (для UI/диагностики)."""
        items: List[Dict[str, Any]] = []
        for provider_id, spec in cls.PROVIDERS.items():
            available = True
            if spec["requires_ollama"]:
                try:
                    available = OllamaProvider(
                        model=spec["model"]).is_available()
                except Exception:
                    available = False
            items.append({
                "id": provider_id,
                "name": spec["name"],
                "model": spec["model"],
                "available": available,
            })
        return items
