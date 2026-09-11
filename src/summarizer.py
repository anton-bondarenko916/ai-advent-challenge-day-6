"""Суммаризация (сжатие) истории диалога для управления контекстом.

Модуль инкапсулирует всё, что связано с конспектом диалога:
- условие срабатывания сжатия (по числу сообщений);
- сборку служебного запроса к модели (отдельный system-промпт + маркер);
- вызов модели через переданный callable (в тестах — без сети);
- обрезку конспекта по лимиту токенов.

Маркер ``SUMMARY_MARKER`` обязательно присутствует в служебном запросе: по нему
служебный вызов отличается от обычного (тесты, sim-режим бенчмарка, логи).
В обычные запросы диалога маркер не попадает.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import utils

logger = utils.get_logger("deepseek-chat.summary")

#: Маркер служебного запроса сжатия (см. модульный docstring)
SUMMARY_MARKER = "[[СЖАТИЕ_ИСТОРИИ]]"

#: Промпт-инструкция по умолчанию
DEFAULT_SUMMARY_PROMPT = (
    "Ты — модуль сжатия контекста диалога. Объедини предыдущий конспект (если он "
    "есть) и новый фрагмент диалога в один краткий конспект.\n"
    "Требования:\n"
    "— сохрани все ключевые факты: имена, предпочтения, договорённости, числа, "
    "даты, ограничения и открытые вопросы;\n"
    "— не выдумывай факты, которых не было в диалоге, и ничего не додумывай;\n"
    "— пиши на языке диалога, кратко, разделами: «Факты», «Решения», "
    "«Открытые вопросы»;\n"
    "— не добавляй вступлений, извинений и выводов — только конспект.\n"
    f"Служебная метка запроса: {SUMMARY_MARKER}"
)

#: Максимальная длина служебного ответа по умолчанию
DEFAULT_SUMMARY_MAX_TOKENS = 400
#: Температура служебного запроса по умолчанию
DEFAULT_SUMMARY_TEMPERATURE = 0.2

# Сигнатура callable, выполняющего служебный запрос к API:
# принимает request (dict для chat.completions.create), возвращает payload
# вида {"content", "model", "tokens", "finish_reason"}.
ServiceCall = Callable[[Dict[str, Any]], Dict[str, Any]]


class ContextSummarizer:
    """Формирует и выполняет служебные запросы сжатия контекста.

    Attributes:
        prompt: промпт-инструкция для модели.
        temperature: температура служебного запроса.
        max_tokens: лимит длины конспекта.
        model: модель для сжатия ("" — использовать модель диалога).
    """

    def __init__(self, prompt: Optional[str] = None,
                 temperature: float = DEFAULT_SUMMARY_TEMPERATURE,
                 max_tokens: int = DEFAULT_SUMMARY_MAX_TOKENS,
                 model: str = "") -> None:
        self.prompt = (prompt or "").strip() or DEFAULT_SUMMARY_PROMPT
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.model = model or ""

    # ------------------------------------------------------------------
    # Условие срабатывания и границы блока
    # ------------------------------------------------------------------

    @staticmethod
    def should_summarize(dialog_length: int, covered: int,
                         keep_last: int, every_n: int) -> bool:
        """Пора ли сжимать историю (триггер — только по числу сообщений).

        Сжимаем, когда «лишних» (не покрытых конспектом и не входящих в
        последние ``keep_last``) сообщений накопилось не меньше ``every_n``.

        Args:
            dialog_length: всего сообщений диалога (без system).
            covered: сколько сообщений уже покрыто конспектом.
            keep_last: сколько последних сообщений остаются «как есть».
            every_n: размер блока сжатия (K).

        Returns:
            True, если нужно запускать сжатие.
        """
        overflow = int(dialog_length) - int(keep_last) - int(covered)
        return overflow >= max(1, int(every_n))

    @staticmethod
    def chunk_bounds(dialog_length: int, covered: int,
                     keep_last: int) -> Tuple[int, int]:
        """Возвращает границы [start, end) блока, который нужно сжать.

        Args:
            dialog_length: всего сообщений диалога (без system).
            covered: сколько сообщений уже покрыто конспектом.
            keep_last: сколько последних сообщений остаются «как есть».

        Returns:
            Кортеж (start, end); если start >= end — сжимать нечего.
        """
        end = max(0, int(dialog_length) - max(0, int(keep_last)))
        start = min(max(0, int(covered)), end)
        return start, end

    # ------------------------------------------------------------------
    # Сборка и выполнение служебного запроса
    # ------------------------------------------------------------------

    def build_request(self, previous_summary: Optional[str],
                      chunk: Sequence[Dict[str, str]],
                      model: Optional[str] = None) -> Dict[str, Any]:
        """Формирует запрос к API для получения конспекта.

        Args:
            previous_summary: предыдущий конспект (None/пусто — первый раз).
            chunk: сообщения, которые нужно добавить к конспекту.
            model: модель для запроса (по умолчанию — self.model).

        Returns:
            Словарь параметров для chat.completions.create.
        """
        parts: List[str] = []
        if previous_summary:
            parts.append("Предыдущий конспект:\n" + previous_summary.strip())
        parts.append("Новый фрагмент диалога:\n" + self._format_chunk(chunk))
        return {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": "\n\n".join(parts)},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    @staticmethod
    def _format_chunk(chunk: Sequence[Dict[str, str]]) -> str:
        """Превращает сообщения фрагмента в текст для конспекта."""
        lines: List[str] = []
        for message in chunk or []:
            role = message.get("role")
            name = "Пользователь" if role == "user" else "Ассистент"
            lines.append(f"{name}: {message.get('content', '')}")
        return "\n".join(lines)

    def summarize(self, previous_summary: Optional[str],
                  chunk: Sequence[Dict[str, str]],
                  call_api: ServiceCall) -> Dict[str, Any]:
        """Просит модель объединить конспект и новый фрагмент.

        Ошибки не пробрасываются: возвращается словарь с полем ``error``,
        чтобы вызывающий код мог сохранить историю и повторить попытку позже.

        Args:
            previous_summary: предыдущий конспект или None.
            chunk: новый фрагмент диалога.
            call_api: callable, выполняющий запрос к API.

        Returns:
            Словарь {"text", "model", "prompt_tokens", "completion_tokens",
            "estimated", "error"}.
        """
        result: Dict[str, Any] = {
            "text": "", "model": None, "prompt_tokens": None,
            "completion_tokens": None, "estimated": False, "error": None,
        }
        if not chunk:
            result["error"] = "Нечего сжимать: пустой фрагмент диалога."
            return result

        request = self.build_request(previous_summary, chunk)
        if not request.get("model"):
            result["error"] = "Не задана модель для сжатия контекста."
            return result

        try:
            payload = call_api(request) or {}
        except Exception as exc:  # ApiError и любые неожиданные ошибки
            message = getattr(exc, "message", None) or utils.humanize_api_error(exc)
            result["error"] = str(message)
            return result

        content = (payload.get("content") or "").strip()
        tokens = payload.get("tokens") or {}
        result["model"] = payload.get("model")
        result["prompt_tokens"] = tokens.get("prompt_tokens")
        result["completion_tokens"] = tokens.get("completion_tokens")
        result["estimated"] = bool(tokens.get("estimated", False))
        if not content:
            result["error"] = "Модель вернула пустой конспект."
            return result
        result["text"] = content
        return result

    # ------------------------------------------------------------------
    # Обрезка конспекта
    # ------------------------------------------------------------------

    @staticmethod
    def truncate(text: str, max_tokens: int, counter: Any,
                 suffix: str = " …") -> str:
        """Обрезает текст так, чтобы оценка токенов была не больше лимита.

        Args:
            text: исходный текст.
            max_tokens: лимит токенов.
            counter: счётчик токенов (src/tokens.py).
            suffix: что добавить в конец обрезанного текста.

        Returns:
            Текст в пределах лимита (или исходный, если он уже короткий).
        """
        if not text or max_tokens <= 0:
            return text
        try:
            if counter.count_text(text) <= max_tokens:
                return text
        except Exception:  # счётчик не должен ломать сжатие
            return text

        words = text.split()
        low, high = 0, len(words)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = " ".join(words[:middle]) + suffix
            try:
                fits = counter.count_text(candidate) <= max_tokens
            except Exception:
                return text
            if fits:
                low = middle
            else:
                high = middle - 1

        if low <= 0:
            logger.warning(
                "Конспект не удалось уложить в %d токенов — текст обрезан "
                "до служебной пометки.", max_tokens,
            )
            return suffix.strip()
        logger.info(
            "Конспект обрезан по лимиту: %d слов из %d.", low, len(words),
        )
        return " ".join(words[:low]) + suffix


def should_summarize(dialog_length: int, covered: int,
                     keep_last: int, every_n: int) -> bool:
    """Модульная обёртка над :meth:`ContextSummarizer.should_summarize`."""
    return ContextSummarizer.should_summarize(
        dialog_length, covered, keep_last, every_n
    )


def get_default_settings() -> Dict[str, Any]:
    """Настройки сжатия по умолчанию (сжатие выключено).

    Агент включает сжатие только если настройки переданы явно (из конфигурации
    приложения), поэтому прямое создание агента в тестах не выполняет
    служебных запросов.
    """
    return {
        "enabled": False,
        "keep_last": 10,
        "every_n": 10,
        "max_tokens": DEFAULT_SUMMARY_MAX_TOKENS,
        "temperature": DEFAULT_SUMMARY_TEMPERATURE,
        "model": "",
        "prompt": "",
    }
