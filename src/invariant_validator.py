"""Пост-проверка ответов агента на нарушение инвариантов.

Модуль инкапсулирует всё, что связано с контролем инвариантов ПОСЛЕ
генерации ответа:

- :class:`ValidationResult` — результат одной проверки;
- :class:`InvariantValidator` — проверка ответа служебным LLM-запросом
  (:meth:`InvariantValidator.validate`) и цикл «проверил → нарушил →
  перегенерировал» (:meth:`InvariantValidator.validate_and_regenerate`)
  с лимитом попыток;
- :meth:`InvariantValidator.find_conflicts` — проверка нового инварианта
  на противоречие существующим (используется UI при сохранении;
  не блокирует сохранение, только предупреждает);
- :class:`Conflict` — найденное противоречие.

Все LLM-вызовы выполняются через переданный callable (``call_api``) —
тот же приём, что и в суммаризаторе: в тестах подставляется заглушка,
в приложении это ``DeepSeekAgent._call_service_api``. Любая ошибка
служебного запроса (включая mock-режим, где клиента нет) не ломает
основной ответ: проверка просто пропускается.

Маркер ``VALIDATOR_MARKER`` обязательно присутствует в служебном запросе:
по нему служебный вызов отличается от обычного (логи, тесты).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import utils
from .invariant_models import SCOPE_GLOBAL, Invariant, InvariantCategory
from .invariant_prompt_builder import _format_invariant_line

logger = utils.get_logger("deepseek-chat.invariant_validator")

#: Маркер служебного запроса пост-проверки
VALIDATOR_MARKER = "[[ПРОВЕРКА_ИНВАРИАНТОВ]]"
#: Маркер служебного запроса проверки конфликтов
CONFLICT_MARKER = "[[ПРОВЕРКА_КОНФЛИКТОВ]]"

ServiceCall = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass
class ValidationResult:
    """Результат одной проверки ответа валидатором.

    Attributes:
        violates: True — ответ ПРЕДЛАГАЕТ решение, противоречащее правилу.
        invariant_id: какой инвариант нарушен (None — не определён).
        reason: объяснение нарушения (для лога и UI).
        refusal: True — ответ является отказом по инварианту (нарушения
            нет, но отказ логируется для аналитики).
        error: текст ошибки служебного запроса (None — проверки не было
            или прошла успешно).
    """

    violates: bool = False
    invariant_id: Optional[int] = None
    reason: str = ""
    refusal: bool = False
    error: Optional[str] = None


@dataclass
class Conflict:
    """Противоречие между новым и существующим инвариантом.

    Attributes:
        invariant_id: id существующего инварианта (None — не определён).
        reason: причина противоречия (текст от LLM).
    """

    invariant_id: Optional[int] = None
    reason: str = ""


def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    """Достаёт первый JSON-объект из ответа модели (устойчиво к мусору).

    Модель может обернуть JSON в markdown-код или добавить пояснения:
    срезаем код-ограждения, затем ищем подстроку от ``{`` до ``}``.
    """
    text = str(raw or "").strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _as_bool(value: Any) -> bool:
    """Приводит ответ модели к логическому значению."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "да")
    return bool(value)


def _as_optional_int(value: Any) -> Optional[int]:
    """Приводит id инварианта к int или None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class InvariantValidator:
    """Проверяет ответы агента на нарушение инвариантов (второй LLM-вызов).

    Attributes:
        enabled: общий выключатель пост-проверки (конфиг ENABLE_POST_CHECK).
        max_regenerations: лимит перегенераций (конфиг MAX_REGENERATIONS).
        min_response_length: ответы короче не проверяются — экономия
            токенов (конфиг MIN_RESPONSE_LENGTH).
        prompt: промпт-инструкция валидатора.
        model: модель проверки ("" — модель диалога).
        temperature: температура служебного запроса.
        max_tokens: лимит длины ответа валидатора.

    После :meth:`validate_and_regenerate` заполняются атрибуты
    ``last_result`` (итоговая :class:`ValidationResult`) и
    ``last_unresolved`` (True — нарушение осталось после лимита попыток):
    агент использует их для записи в лог нарушений.
    """

    #: Лимит перегенераций (по ТЗ; перекрывается конфигом)
    MAX_REGENERATIONS = 2
    #: Не проверять короткие ответы (по ТЗ; перекрывается конфигом)
    MIN_RESPONSE_LENGTH = 100
    #: Пост-проверка включена по умолчанию (по ТЗ; перекрывается конфигом)
    ENABLED = True

    DEFAULT_PROMPT = (
        "Ты — валидатор инвариантов. Проверь, нарушает ли ответ агента "
        "следующие правила.\n\n"
        "Нарушение = агент ПРЕДЛАГАЕТ, РЕКОМЕНДУЕТ или ОПИСЫВАЕТ решение, "
        "противоречащее правилу.\n\n"
        "НЕ является нарушением:\n"
        "- Упоминание правила в отрицательном контексте («не используй X»);\n"
        "- Объяснение, почему правило существует;\n"
        "- Цитирование правила как примера.\n\n"
        "Отказ = агент ОТКАЗЫВАЕТСЯ выполнять запрос со ссылкой на "
        "инвариант и предлагает альтернативы. Отказ нарушением НЕ считается.\n\n"
        "Инварианты:\n{invariants_list}\n\n"
        "Ответ агента:\n{response}\n\n"
        "Верни строго JSON без markdown:\n"
        '{"violates": true/false, "refusal": true/false, '
        '"violated_invariant_id": int|null, "reason": "..."}\n\n'
        f"Служебная метка запроса: {VALIDATOR_MARKER}"
    )

    DEFAULT_CONFLICT_PROMPT = (
        "Проверь, противоречит ли новый инвариант существующим.\n\n"
        "Существующие:\n{existing_list}\n\n"
        "Новый: {new_invariant}\n\n"
        "Верни строго JSON без markdown:\n"
        '{"conflicts": [{"id": int, "reason": "..."}]}\n'
        "Если противоречий нет — верни {\"conflicts\": []}.\n\n"
        f"Служебная метка запроса: {CONFLICT_MARKER}"
    )

    def __init__(
        self,
        enabled: bool = ENABLED,
        max_regenerations: int = MAX_REGENERATIONS,
        min_response_length: int = MIN_RESPONSE_LENGTH,
        prompt: str = "",
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> None:
        """Создаёт валидатор (настройки обычно приходят из конфига).

        Args:
            enabled: включена ли пост-проверка.
            max_regenerations: лимит перегенераций (>= 0).
            min_response_length: минимальная длина ответа для проверки.
            prompt: своя инструкция валидатора; "" — встроенная.
            model: модель проверки; "" — модель диалога.
            temperature: температура служебного запроса.
            max_tokens: лимит длины ответа валидатора.
        """
        self.enabled = bool(enabled)
        try:
            self.max_regenerations = max(0, int(max_regenerations))
        except (TypeError, ValueError):
            self.max_regenerations = self.MAX_REGENERATIONS
        try:
            self.min_response_length = max(0, int(min_response_length))
        except (TypeError, ValueError):
            self.min_response_length = self.MIN_RESPONSE_LENGTH
        self.prompt = (prompt or "").strip() or self.DEFAULT_PROMPT
        self.model = model or ""
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        #: Результат последней проверки (заполняется validate/…regenerate)
        self.last_result: Optional[ValidationResult] = None
        #: True — нарушение не устранено после лимита перегенераций
        self.last_unresolved: bool = False

    # ------------------------------------------------------------------
    # Служебные промпты
    # ------------------------------------------------------------------

    @staticmethod
    def format_invariants_list(invariants: Sequence[Invariant]) -> str:
        """Форматирует список инвариантов для промпта («id. [G] title — rule»)."""
        lines = []
        for invariant in invariants or []:
            marker = "[G]" if invariant.scope == SCOPE_GLOBAL else "[P]"
            lines.append(
                f"{invariant.id}. {marker} {invariant.title} — {invariant.rule}"
            )
        return "\n".join(lines)

    def _validate_request(self, response: str,
                          invariants: Sequence[Invariant]) -> Dict[str, Any]:
        """Собирает запрос пост-проверки для chat.completions.create."""
        prompt = self.prompt
        if "{invariants_list}" in prompt:
            prompt = prompt.replace(
                "{invariants_list}",
                self.format_invariants_list(invariants),
            )
        if "{response}" in prompt:
            prompt = prompt.replace("{response}", response)
        return {
            "model": self.model or None,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user",
                 "content": "Проверь ответ и верни JSON."},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    # ------------------------------------------------------------------
    # Одна проверка
    # ------------------------------------------------------------------

    def validate(
        self,
        response: str,
        invariants: Sequence[Invariant],
        call_api: Optional[ServiceCall] = None,
    ) -> ValidationResult:
        """Проверяет ответ агента служебным LLM-запросом.

        Ошибки (сеть, mock-режим, битый JSON) не пробрасываются:
        возвращается результат без нарушения с заполненным ``error`` —
        пост-проверка не должна ломать основной ответ.

        Args:
            response: текст ответа агента.
            invariants: действующие инварианты.
            call_api: callable запроса к API; None — проверка пропускается.

        Returns:
            :class:`ValidationResult` (атрибут ``last_result`` также
            обновляется).
        """
        result = ValidationResult()
        self.last_result = result
        if not invariants:
            return result
        if call_api is None:
            result.error = "Не задан вызов API — пост-проверка пропущена."
            return result

        request = self._validate_request(str(response or ""), invariants)
        try:
            payload = call_api(request) or {}
        except Exception as exc:  # ApiError и любые неожиданные ошибки
            message = getattr(exc, "message", None) or utils.humanize_api_error(exc)
            result.error = str(message)
            logger.warning("Пост-проверка инвариантов не удалась: %s",
                           result.error)
            return result

        parsed = _extract_json(str(payload.get("content") or ""))
        if parsed is None:
            result.error = "Валидатор вернул не JSON — пост-проверка пропущена."
            logger.warning(result.error)
            return result

        result.violates = _as_bool(parsed.get("violates"))
        result.refusal = _as_bool(parsed.get("refusal"))
        result.invariant_id = _as_optional_int(
            parsed.get("violated_invariant_id")
        )
        result.reason = str(parsed.get("reason") or "")
        return result

    # ------------------------------------------------------------------
    # Цикл «проверка → перегенерация»
    # ------------------------------------------------------------------

    def validate_and_regenerate(
        self,
        agent: Any,
        user_query: str,
        first_response: str,
        invariants: Sequence[Invariant],
        call_api: Optional[ServiceCall] = None,
    ) -> Tuple[str, int]:
        """Проверяет ответ и при нарушении перегенерирует его.

        Цикл: проверка → если нарушение, агент перегенерирует ответ с
        явным напоминанием про нарушенный инвариант → проверка снова.
        Максимум ``max_regenerations`` перегенераций; если после лимита
        нарушение осталось — возвращается последний ответ, а
        ``last_unresolved`` = True (агент пишет это в лог нарушений и
        показывает предупреждение в UI).

        Args:
            agent: объект агента с методом
                ``regenerate_last_response(user_query, violated_rule)``.
            user_query: запрос пользователя.
            first_response: первый ответ агента.
            invariants: действующие инварианты.
            call_api: callable запроса к API для валидатора.

        Returns:
            Кортеж ``(финальный_ответ, количество_перегенераций)``.
        """
        self.last_unresolved = False
        current = str(first_response or "")
        if not invariants:
            self.last_result = ValidationResult()
            return current, 0

        result = self.validate(current, invariants, call_api)
        regenerations = 0
        while result.violates and regenerations < self.max_regenerations:
            invariant = self._find_invariant(invariants, result.invariant_id)
            rule = invariant.rule if invariant is not None else ""
            try:
                regenerated = agent.regenerate_last_response(
                    user_query, rule
                )
            except Exception as exc:  # перегенерация не должна ломать ответ
                logger.warning(
                    "Перегенерация по инварианту не удалась: %s", exc
                )
                break
            if not str(regenerated or "").strip():
                break
            current = str(regenerated)
            regenerations += 1
            result = self.validate(current, invariants, call_api)

        self.last_result = result
        self.last_unresolved = bool(result.violates)
        if regenerations:
            status = "нарушение осталось" if result.violates else "устранено"
            logger.info(
                "Пост-проверка инвариантов: перегенераций %d, %s.",
                regenerations, status,
            )
        return current, regenerations

    @staticmethod
    def _find_invariant(invariants: Sequence[Invariant],
                        invariant_id: Optional[int]) -> Optional[Invariant]:
        """Инвариант по id (None — первый из списка)."""
        if invariant_id is None:
            return invariants[0] if invariants else None
        for invariant in invariants:
            if invariant.id == invariant_id:
                return invariant
        return invariants[0] if invariants else None

    # ------------------------------------------------------------------
    # Проверка конфликтов инвариантов (UI: при сохранении)
    # ------------------------------------------------------------------

    def find_conflicts(
        self,
        new_invariant: Invariant,
        existing: Sequence[Invariant],
        categories: Sequence[InvariantCategory] = (),
        call_api: Optional[ServiceCall] = None,
    ) -> List[Conflict]:
        """Ищет противоречия нового инварианта с существующими (LLM-запрос).

        Используется UI при сохранении: найденные конфликты показываются
        предупреждением, но НЕ блокируют сохранение. Ошибки не
        пробрасываются — возвращается пустой список (сохранение без
        предупреждения).

        Args:
            new_invariant: сохраняемый инвариант.
            existing: уже существующие инварианты (того же профиля/глобальные;
                сам сохраняемый — исключить заранее).
            categories: категории (для подписей в промпте).
            call_api: callable запроса к API; None — проверка пропускается.

        Returns:
            Список :class:`Conflict` (пустой — конфликтов нет или проверка
            не удалась).
        """
        if not existing or call_api is None:
            return []
        category_names = {
            int(category.id): category.display_name
            for category in categories if category.id is not None
        }
        existing_lines = []
        for invariant in existing:
            display = category_names.get(
                int(invariant.category_id), str(invariant.category_id)
            )
            existing_lines.append(
                f"{invariant.id}. ({display}) "
                f"{_format_invariant_line(invariant).lstrip('- ')}"
            )
        new_text = (
            f"«{new_invariant.title}» — {new_invariant.rule}"
        )
        prompt = self.DEFAULT_CONFLICT_PROMPT.replace(
            "{existing_list}", "\n".join(existing_lines)
        ).replace("{new_invariant}", new_text)
        request = {
            "model": self.model or None,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Проверь конфликты и верни JSON."},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        try:
            payload = call_api(request) or {}
        except Exception as exc:
            logger.warning("Проверка конфликтов инвариантов не удалась: %s", exc)
            return []

        parsed = _extract_json(str(payload.get("content") or ""))
        if parsed is None:
            return []
        raw_conflicts = parsed.get("conflicts")
        if not isinstance(raw_conflicts, list):
            return []
        conflicts: List[Conflict] = []
        for item in raw_conflicts:
            if not isinstance(item, dict):
                continue
            conflicts.append(Conflict(
                invariant_id=_as_optional_int(item.get("id")),
                reason=str(item.get("reason") or ""),
            ))
        return conflicts
