"""Счётчик токенов для оценки контекста и ответов.

Модуль предоставляет две реализации:
- TiktokenCounter — на базе пакета ``tiktoken`` (если он установлен и удаётся
  получить кодировку). Для моделей DeepSeek своей кодировки в реестре
  tiktoken нет, поэтому используется ``cl100k_base`` (или имя из конфигурации);
- HeuristicCounter — встроенная эвристика на стандартной библиотеке:
  приближённая оценка по количеству символов/слов (кириллица «дороже» латиницы).

Любая проблема с tiktoken (пакет не установлен, нет доступа к сети для загрузки
BPE-файла, неизвестная кодировка) перехватывается и приводит к автоматическому
переключению на эвристику — приложение продолжает работать.

Оценка сообщений учитывает служебные токены формата чата
(``CHAT_MESSAGE_OVERHEAD`` на сообщение + ``CHAT_PRIMING_OVERHEAD``), чтобы
приблизить результат к ``usage.prompt_tokens`` из API.

Важно: локальная оценка (даже tiktoken) не гарантирует точного совпадения с
токенизатором DeepSeek. Точные значения даёт только поле ``usage`` в ответе API.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from . import utils

logger = utils.get_logger("deepseek-chat.tokens")

# Кодировка по умолчанию для tiktoken (DeepSeek-моделей в реестре нет)
DEFAULT_ENCODING = "cl100k_base"

# Служебные токены формата чата (как в OpenAI cookbook)
CHAT_MESSAGE_OVERHEAD = 4
CHAT_PRIMING_OVERHEAD = 3

# Флаг: предупреждение о недоступности tiktoken выводится только один раз
_warned_about_tiktoken = False


class TokenCounter:
    """Базовый интерфейс счётчика токенов."""

    #: человекочитаемое имя движка (для UI)
    name: str = "counter"
    #: True, если счётчик основан на токенизаторе (tiktoken), иначе эвристика
    is_exact: bool = False

    def count_text(self, text: str) -> int:
        """Возвращает оценку числа токенов в тексте."""
        raise NotImplementedError

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Оценивает число токенов всего списка сообщений чата.

        Args:
            messages: список сообщений {role, content}.

        Returns:
            Оценка с учётом служебных токенов формата чата.
        """
        total = CHAT_PRIMING_OVERHEAD
        for message in messages or []:
            content = message.get("content") or ""
            total += CHAT_MESSAGE_OVERHEAD + self.count_text(str(content))
        return total


class HeuristicCounter(TokenCounter):
    """Приближённый счётчик без внешних зависимостей.

    Формула (подобрана для смешанного русского/английского текста):
    ``ceil(cyrillic / 2.5 + other_chars / 4 + 0.2 * words)``.
    """

    name = "heuristic"
    is_exact = False

    def count_text(self, text: str) -> int:
        """Оценивает токены эвристикой (пустая строка → 0)."""
        if not text or not text.strip():
            return 0
        cyrillic = sum(1 for ch in text if "\u0400" <= ch <= "\u04FF")
        other = len(text) - cyrillic
        words = len(text.split())
        estimate = cyrillic / 2.5 + other / 4.0 + 0.2 * words
        return max(1, math.ceil(estimate))


class TiktokenCounter(TokenCounter):
    """Счётчик на базе пакета ``tiktoken``.

    Кодировка выбирается так: ``encoding_for_model(model)`` → при неудаче
    ``get_encoding(encoding_name)``. Созданные кодировки кэшируются на уровне
    класса (создание кодировки — дорогая операция).

    Raises:
        Exception: если пакет недоступен или кодировку получить не удалось
            (обрабатывается в create_counter).
    """

    _encodings: Dict[str, Any] = {}  # кэш: имя кодировки → объект encoding

    def __init__(self, model: Optional[str] = None,
                 encoding_name: str = DEFAULT_ENCODING) -> None:
        """Создаёт счётчик для указанной модели.

        Args:
            model: имя модели (может быть неизвестно tiktoken).
            encoding_name: резервная кодировка.
        """
        import tiktoken  # локальный импорт: пакет опционален

        self._tiktoken = tiktoken
        encoding = self._resolve_encoding(tiktoken, model, encoding_name)
        self._encoding = encoding
        self._encoding_name = getattr(encoding, "name", encoding_name)
        self.name = f"tiktoken:{self._encoding_name}"
        self.is_exact = True

    @classmethod
    def _resolve_encoding(cls, tiktoken_module: Any, model: Optional[str],
                          encoding_name: str) -> Any:
        """Возвращает объект кодировки, используя кэш класса."""
        if model:
            try:
                encoding = tiktoken_module.encoding_for_model(model)
                cls._encodings[encoding.name] = encoding
                return encoding
            except Exception:
                # Модель неизвестна tiktoken — используем резервную кодировку
                pass
        if encoding_name not in cls._encodings:
            cls._encodings[encoding_name] = tiktoken_module.get_encoding(
                encoding_name
            )
        return cls._encodings[encoding_name]

    def count_text(self, text: str) -> int:
        """Считает токены кодировщиком tiktoken."""
        if not text:
            return 0
        # disallowed_special=() — произвольный текст не должен вызывать ошибку
        return len(self._encoding.encode(text, disallowed_special=()))


def create_counter(model: Optional[str] = None, enabled: bool = True,
                   encoding_name: str = DEFAULT_ENCODING) -> TokenCounter:
    """Создаёт счётчик токенов: tiktoken при возможности, иначе эвристику.

    Args:
        model: имя модели для выбора кодировки.
        enabled: False — принудительно использовать эвристику.
        encoding_name: резервная кодировка tiktoken.

    Returns:
        TokenCounter: TiktokenCounter либо HeuristicCounter.
    """
    global _warned_about_tiktoken
    if enabled:
        try:
            return TiktokenCounter(model=model, encoding_name=encoding_name)
        except Exception as exc:  # ImportError, ошибка загрузки BPE и т.п.
            if not _warned_about_tiktoken:
                logger.warning(
                    "tiktoken недоступен (%s) — используется встроенная "
                    "эвристика оценки токенов.", exc,
                )
                _warned_about_tiktoken = True
    return HeuristicCounter()
