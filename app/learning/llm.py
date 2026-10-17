"""JSON-режим LLM для платформы обучения (ТЗ ai_1_36).

Отдельный служебный вызов DeepSeek (OpenAI-совместимый SDK) с
``response_format={"type": "json_object"}`` — тот же приём, что в
app/rag/citations.py: слово «JSON» в промпте, температура 0.2–0.3,
бюджет токенов с запасом (deepseek-v4-flash — thinking-модель: при
малом лимите JSON не помещается) и ОДИН повтор с удвоенным лимитом
при пустом/обрезанном ответе.

:func:`build_json_llm` собирает клиент из конфигурации; в mock-режиме
(--mock и офлайн-тесты) отвечает :class:`MockJsonLLM` — детерминированные
план/проверка без сети (для выбора файлов парсит дерево проекта из
промпта, поэтому мок-проверка работает на любом проекте).
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, Optional

from openai import OpenAI

from src.utils import get_logger

logger = get_logger("deepseek-chat.learning.llm")

# Модель/базовый URL — как у ядра диалога и RAG-пайплайнов
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"

JSON_TEMPERATURE = 0.2     # низкая температура — стабильный JSON
JSON_MAX_TOKENS = 8000     # thinking-модель тратит бюджет и на reasoning
JSON_TIMEOUT = 120.0

# Маркеры задач в промптах (их же используют MockJsonLLM и тесты) —
# один клиент обслуживает три разных JSON-контракта
PLAN_MARKER = "[ЗАДАЧА: ПЛАН ТЕМЫ]"
SELECT_FILES_MARKER = "[ЗАДАЧА: ВЫБОР ФАЙЛОВ]"
REVIEW_MARKER = "[ЗАДАЧА: ПРОВЕРКА ЗАДАНИЯ]"


class LLMError(RuntimeError):
    """Ошибка JSON-вызова: сеть/API/невалидный JSON."""


def parse_json_object(raw: str) -> Dict[str, Any]:
    """JSON-объект из ответа модели (устойчиво к ```-ограждениям и прозе).

    Raises:
        LLMError: валидного JSON-объекта в ответе нет.
    """
    from app.rag.citations import CitationError, parse_citation_json
    try:
        return parse_citation_json(raw)
    except CitationError as exc:
        raise LLMError(str(exc)) from exc


class BaseJsonLLM:
    """Общий контракт: generate_json(prompt) → dict (+ async-вариант)."""

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        raise NotImplementedError

    async def generate_json_async(self, prompt: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self.generate_json, prompt)


class DeepSeekJsonLLM(BaseJsonLLM):
    """Служебный JSON-вызов DeepSeek (план темы / выбор файлов / проверка).

    Attributes:
        model: модель служебного запроса.
        max_tokens: базовый бюджет ответа (ретрай — с удвоением).
        mock_mode: True — отвечает MockJsonLLM (сеть не используется).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        temperature: float = JSON_TEMPERATURE,
        max_tokens: int = JSON_MAX_TOKENS,
        timeout: float = JSON_TIMEOUT,
        mock_mode: bool = False,
        client: Optional[OpenAI] = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self._timeout = float(timeout)
        self.mock_mode = bool(mock_mode)
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

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """Промпт → JSON-объект (пустой/обрезанный ответ — один ретрай
        с удвоенным max_tokens, урок NEWS_SUMMARY_MAX_TOKENS)."""
        if self.mock_mode:
            return MockJsonLLM().generate_json(prompt)
        raw, finish = self._call(prompt, self.max_tokens)
        if not raw.strip() or finish == "length":
            logger.warning(
                "JSON-ответ пуст или обрезан (finish_reason=%s) — "
                "повтор с удвоенным max_tokens", finish)
            raw, finish = self._call(prompt, self.max_tokens * 2)
        if not raw.strip():
            raise LLMError("Пустой ответ модели в JSON-режиме")
        return parse_json_object(raw)

    def _call(self, prompt: str, max_tokens: int) -> tuple:
        """Один вызов API: (content, finish_reason)."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=self.temperature,
                max_tokens=int(max_tokens),
            )
        except Exception as exc:  # noqa: BLE001 — SDK бросает широкий набор
            raise LLMError(f"JSON-режим недоступен ({self.model}): {exc}") from exc
        choice = response.choices[0]
        return (str(choice.message.content or ""),
                getattr(choice, "finish_reason", None))


class MockJsonLLM(BaseJsonLLM):
    """Детерминированные ответы без сети (--mock и офлайн-тесты).

    Контракт определяется маркером в промпте:

    * PLAN_MARKER — план темы (заголовок/уровень читаются из промпта);
    * SELECT_FILES_MARKER — файлы для чтения: «file»-листья парсятся
      из дерева проекта в промпте (мок-проверка работает на любом
      проекте, а не только на фикстуре тестов);
    * REVIEW_MARKER — фиксированная оценка с фидбеком.
    """

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        text = str(prompt or "")
        if PLAN_MARKER in text:
            return self._plan(text)
        if SELECT_FILES_MARKER in text:
            return self._select_files(text)
        if REVIEW_MARKER in text:
            return self._review(text)
        raise LLMError(
            "MockJsonLLM: в промпте нет известного маркера задачи")

    # ------------------------------------------------------------------
    def _plan(self, prompt: str) -> Dict[str, Any]:
        title = _prompt_field(prompt, r"Тема:\s*(.+)") or "Тема курса"
        level = _prompt_field(prompt, r"Уровень пользователя:\s*(\w+)")
        suffix = {
            "beginner": "с нуля, простыми словами",
            "intermediate": "с практикой и типовыми паттернами",
            "advanced": "с продвинутыми приёмами и подводными камнями",
        }.get(str(level or "").strip(), "по вашему уровню")
        return {
            "title": title,
            "description": f"Тема «{title}» — {suffix} "
                           f"(мок-генерация, офлайн-режим).",
            "subtopics": [
                {
                    "title": f"{title}: основы",
                    "theory": f"## Основы\nТеория подтемы 1 темы "
                              f"«{title}» ({suffix}).",
                    "task": "Создайте минимальный проект по разобранной "
                            "в теории схеме.",
                    "rubric": "Проект собирается; структура соответствует "
                              "теории; код читаем.",
                },
                {
                    "title": f"{title}: практика",
                    "theory": f"## Практика\nТеория подтемы 2 темы "
                              f"«{title}» ({suffix}).",
                    "task": "Доработайте проект: добавьте функциональность "
                            "из задания.",
                    "rubric": "Функциональность работает; есть обработка "
                              "базовых ошибок.",
                },
                {
                    "title": f"{title}: закрепление",
                    "theory": f"## Закрепление\nТеория подтемы 3 темы "
                              f"«{title}» ({suffix}).",
                    "task": "Выполните итоговое упражнение и опишите "
                            "решение в README проекта.",
                    "rubric": "Упражнение выполнено; README объясняет "
                              "ключевые решения.",
                },
            ],
        }

    def _select_files(self, prompt: str) -> Dict[str, Any]:
        marker = "Дерево проекта (JSON):"
        index = prompt.find(marker)
        tree_text = prompt[index + len(marker):] if index != -1 else ""
        return {"files": _tree_file_paths(tree_text)[:5]}

    def _review(self, prompt: str) -> Dict[str, Any]:
        return {
            "score": 5,
            "feedback": "Мок-проверка (офлайн-режим): структура проекта "
                        "соответствует заданию. Что сделано хорошо — "
                        "проект собирается. Что улучшить — добавьте "
                        "тесты и обработку ошибок.",
        }


def _prompt_field(prompt: str, pattern: str) -> Optional[str]:
    """Значение поля из промпта по регулярному выражению (для мока)."""
    match = re.search(pattern, prompt)
    return match.group(1).strip() if match else None


def _tree_file_paths(tree_text: str) -> list:
    """«file»-листья дерева (fs_server.tree) — плоский список путей.

    Текст после дерева (продолжение промпта) игнорируется:
    ``raw_decode`` читает первый валидный JSON-объект.
    """
    start = tree_text.find("{")
    if start == -1:
        return []
    try:
        tree, _ = json.JSONDecoder().raw_decode(tree_text[start:])
    except ValueError:
        return []

    paths: list = []

    def _walk(node: Any, prefix: str) -> None:
        if not isinstance(node, dict):
            return
        for name, value in node.items():
            path = f"{prefix}/{name}" if prefix else name
            if value == "file":
                paths.append(path)
            elif isinstance(value, dict):
                _walk(value, path)

    _walk(tree, "")
    return paths


def build_json_llm(config: Dict[str, Any]) -> DeepSeekJsonLLM:
    """Клиент из конфигурации load_config (mock-режим уважается)."""
    return DeepSeekJsonLLM(
        api_key=str(config.get("DEEPSEEK_API_KEY") or ""),
        base_url=str(config.get("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL),
        model=str(config.get("DEFAULT_MODEL") or DEFAULT_MODEL),
        mock_mode=bool(config.get("MOCK_MODE", False)),
    )
