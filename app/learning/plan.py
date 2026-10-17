"""План обучения: ленивая генерация темы через LLM (ТЗ ai_1_36).

Курс — 5 фиксированных тем; БД при входе наполняется «пустыми» строками
(generated_at NULL). При ПЕРВОМ входе в тему PlanService генерирует её
структуру (описание + 3–4 подтемы с теорией/заданием/рубрикой) одним
JSON-запросом с учётом уровня пользователя (beginner/intermediate/
advanced) и сохраняет в БД — повторный вход читает из БД, LLM не
вызывается (ТЗ «не генерировать весь план сразу»).

После генерации темы создаётся напоминание («вернись к теме через
3 дня») через MCP reminder-сервера — best-effort: недоступный сервер
НЕ ломает результат генерации (в ответе — пометка reminder_error).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.utils import get_logger
from app.learning.db import LearningRepository
from app.learning.llm import PLAN_MARKER

logger = get_logger("deepseek-chat.learning.plan")

# 5 тем курса (ТЗ): заголовки фиксированы, контент — ленивая генерация
TOPIC_TITLES = (
    "Основы: первый проект на Android",
    "UI: Compose, макеты и Material",
    "Жизненный цикл и состояние приложения",
    "Списки, адаптеры и работа с сетью",
    "Данные: Room, ViewModel и архитектура",
)

# Подсказки уровня в промпт (тест «уровни» проверяет попадание в промпт)
LEVEL_GUIDANCE = {
    "beginner": "Объясняй максимально просто, без предположений о "
                "предыдущем опыте; задания небольшие и пошаговые.",
    "intermediate": "Считай, что базовый опыт есть; делай упор на "
                    "практику и типовые паттерны.",
    "advanced": "Рассчитывай на опытного разработчика; включай "
                "продвинутые приёмы, оптимизации и подводные камни.",
}

DEFAULT_LEVEL_GUIDANCE = (
    "Подстрой сложность под указанный уровень пользователя.")

# Напоминание «вернись к теме» — через 3 дня (ТЗ)
REMINDER_DELAY_MINUTES = 3 * 24 * 60

PLAN_PROMPT_TEMPLATE = """{marker}
Ты — методист курса Android-разработки. Сгенерируй тему курса обучения.

Тема: {title}
Уровень пользователя: {level}. {guidance}

Верни JSON-объект следующей структуры:
{{
  "title": "Название темы",
  "description": "Краткое описание темы (2-3 предложения)",
  "subtopics": [
    {{
      "title": "Название подтемы",
      "theory": "Теория в Markdown (500-800 слов)",
      "task": "Практическое задание",
      "rubric": "Критерии оценки (что считается хорошим выполнением)"
    }}
  ]
}}

Требования:
- Тема должна содержать 3-4 подтемы.
- Фокус на практических навыках Android-разработки.
- Теория — конкретная, с примерами кода (Kotlin).
- Задание выполняется в отдельном проекте; rubric должна позволять
  однозначно оценить выполнение по коду проекта.
"""

REMINDER_TEXT_TEMPLATE = (
    "Вернись к теме «{title}» — продолжи обучение!")


class PlanGenerationError(RuntimeError):
    """Ошибка ленивой генерации темы (тема не найдена / LLM / JSON)."""


def build_plan_prompt(title: str, level: str) -> str:
    """Промпт генерации темы: заголовок + уровень + подсказка уровня."""
    guidance = LEVEL_GUIDANCE.get(str(level or "").strip().lower(),
                                  DEFAULT_LEVEL_GUIDANCE)
    return PLAN_PROMPT_TEMPLATE.format(
        marker=PLAN_MARKER, title=title,
        level=str(level or "beginner"), guidance=guidance)


def normalize_plan(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Приводит JSON плана к контракту БД: title/description +
    3–4 подтемы (лишние отбрасываются), у каждой непустые title/theory/
    task/rubric (пустые поля заменяются пометкой).

    Raises:
        PlanGenerationError: подтем нет вовсе или ответ не словарь.
    """
    if not isinstance(data, dict):
        raise PlanGenerationError("Ответ модели не является JSON-объектом")
    raw_subtopics = data.get("subtopics")
    if not isinstance(raw_subtopics, list) or not raw_subtopics:
        raise PlanGenerationError(
            "В ответе модели нет подтем (subtopics)")
    subtopics: List[Dict[str, Any]] = []
    for item in raw_subtopics[:4]:  # ТЗ: 3-4 подтемы
        if not isinstance(item, dict):
            continue
        subtopics.append({
            "title": str(item.get("title") or "").strip()
                     or "Подтема без названия",
            "theory": str(item.get("theory") or "").strip()
                      or "(теория не сгенерирована)",
            "task": str(item.get("task") or "").strip()
                    or "(задание не сгенерировано)",
            "rubric": str(item.get("rubric") or "").strip()
                      or "(критерии оценки не сгенерированы)",
        })
    if not subtopics:
        raise PlanGenerationError(
            "В ответе модели нет корректных подтем")
    return subtopics


class PlanService:
    """Ленивая генерация тем курса + напоминания reminder-сервера.

    Attributes:
        repo: LearningRepository (темы/подтемы/пользователи).
        llm: JSON-клиент (DeepSeekJsonLLM / MockJsonLLM / фейк тестов).
        reminder_client: HTTP MCP-клиент reminder-сервера; None —
            напоминания отключены.
        reminder_delay_minutes: интервал напоминания (ТЗ: 3 дня).
    """

    def __init__(self, repo: LearningRepository, llm: Any,
                 reminder_client: Optional[Any] = None,
                 reminder_delay_minutes: int = REMINDER_DELAY_MINUTES
                 ) -> None:
        self.repo = repo
        self.llm = llm
        self.reminder_client = reminder_client
        self.reminder_delay_minutes = int(reminder_delay_minutes)

    def ensure_topics(self, user_id: int) -> List[Dict[str, Any]]:
        """5 тем пользователя (создаются при первом показе дашборда)."""
        return self.repo.ensure_topics(user_id, TOPIC_TITLES)

    async def get_or_generate_topic(self, user_id: int,
                                    topic_id: int) -> Dict[str, Any]:
        """Тема из БД; не сгенерирована — генерация одним JSON-запросом.

        Повторный вход в тему LLM не вызывает (сначала проверка БД).
        Ключ "reminder_status" результата — текст о попытке создать
        напоминание (или None, если напоминания отключены).
        """
        topic = self.repo.get_topic(user_id, topic_id)
        if topic is None:
            raise PlanGenerationError(
                f"Тема {topic_id} не найдена у пользователя {user_id}")
        if topic["is_generated"]:
            topic["reminder_status"] = None
            return topic

        user = self.repo.get_user(user_id) or {"level": "beginner"}
        prompt = build_plan_prompt(topic["title"], user.get("level"))
        data = await self.llm.generate_json_async(prompt)
        subtopics = normalize_plan(data)

        title = str(data.get("title") or "").strip() or topic["title"]
        description = str(data.get("description") or "").strip()
        self.repo.save_subtopics(topic_id, subtopics)
        self.repo.mark_topic_generated(topic_id, title, description)
        logger.info("Тема %s сгенерирована (%d подтем, уровень %s)",
                    title, len(subtopics), user.get("level"))

        topic = self.repo.get_topic(user_id, topic_id)
        topic["reminder_status"] = await self.schedule_reminder(title)
        return topic

    async def schedule_reminder(self, topic_title: str) -> Optional[str]:
        """Напоминание «вернись к теме» (best-effort, ТЗ Модуль 6).

        Returns:
            None — напоминания отключены; строка — результат (успех или
            причина, по которой напоминание не создано).
        """
        if self.reminder_client is None:
            return None
        text = REMINDER_TEXT_TEMPLATE.format(title=topic_title)
        try:
            await self.reminder_client.call_tool(
                "add_reminder",
                {
                    "text": text,
                    "interval_minutes": self.reminder_delay_minutes,
                    "repeat": False,
                },
            )
        except Exception as exc:  # noqa: BLE001 — не ломаем генерацию
            logger.warning("Напоминание не создано: %s", exc)
            return f"Напоминание не создано: {exc}"
        return (f"Напоминание создано: «{text}» "
                f"(через {self.reminder_delay_minutes // 60} ч)")
