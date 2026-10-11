"""Чистая логика Telegram-бота: доступ, разбивка, ход чата (ТЗ ai_1_31).

Слой сознательно отделён от aiogram: функции и :class:`ChatTurn`
тестируются офлайн без Telegram (httpx замокан, БД временная), а
хендлеры в telegram_bot.py остаются тонкой обёрткой (достать user_id,
отправить части, показать typing).

Политика доступа — белый список Telegram user ID (TELEGRAM_ALLOWED_IDS,
CSV из .env): чужой бот не найдёт, а если найдёт — получит отказ, его ID
попадёт в лог (владелец добавит).
"""

from __future__ import annotations

import logging
from typing import Dict, FrozenSet, List, Optional

from .tg_ollama_client import OllamaChatClient, OllamaChatError
from .tg_storage import TelegramHistoryStore

logger = logging.getLogger("deepseek-chat.tg-bot")

# Жёсткий лимит Telegram — 4096 символов на сообщение; режем с запасом
TELEGRAM_MESSAGE_LIMIT = 4096
SAFE_SPLIT_LIMIT = 4000


def parse_allowed_ids(raw: str) -> FrozenSet[int]:
    """Множество разрешённых user_id из CSV-строки (.env).

    Пробелы/пустые элементы игнорируются, мусорные значения — warning
    и пропуск (кривая конфигурация не должна валить бота).
    """
    allowed: set = set()
    for chunk in str(raw or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            allowed.add(int(chunk))
        except ValueError:
            logger.warning(
                "TELEGRAM_ALLOWED_IDS: мусорный элемент %r — пропущен",
                chunk)
    return frozenset(allowed)


def is_allowed(user_id: Optional[int], allowed: FrozenSet[int]) -> bool:
    """Проверка белого списка (None — не разрешён)."""
    if user_id is None:
        return False
    return int(user_id) in allowed


def split_message(text: str,
                  limit: int = SAFE_SPLIT_LIMIT) -> List[str]:
    """Разбивает ответ на части не длиннее ``limit`` символов.

    Резать стараемся по границам переноса строки, затем пробела
    (ближайшие к лимиту); текст без разделителей — жёсткий разрез на
    лимите. Короткий/пустой текст возвращается одним элементом.
    """
    text = str(text or "")
    if len(text) <= limit:
        return [text]
    parts: List[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 1, limit)
        if cut < limit // 2:
            space = remaining.rfind(" ", 1, limit)
            if space > cut:
                cut = space
        if cut <= 0:
            cut = limit
        parts.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    parts.append(remaining)
    # rstrip/lstrip на стыках могут дать пустые части — выбрасываем
    return [part for part in parts if part] or [text]


class ChatTurn:
    """Полный ход чата одного пользователя: история → Ollama → история.

    Схема (ТЗ): загрузить хвост истории (TELEGRAM_HISTORY_LIMIT) →
    дописать user-сообщение → запрос к Ollama → дописать
    assistant-сообщение → вернуть ответ, разбитый на части <= 4096.

    При ошибке Ollama user-сообщение В ИСТОРИИ ОСТАЁТСЯ (вопрос не
    теряется, пользователь переспросит), assistant-сообщение не
    дописывается — исключение уходит наверх (хендлер покажет
    дружелюбное сообщение).
    """

    def __init__(self, client: OllamaChatClient,
                 store: TelegramHistoryStore,
                 history_limit: int = 20,
                 default_model: str = "") -> None:
        self.client = client
        self.store = store
        self.history_limit = max(2, int(history_limit))
        self.default_model = str(default_model)

    async def resolve_model(self, user_id: int) -> str:
        """Модель пользователя: выбор из /model → дефолт из конфига →
        первая установленная в Ollama (нечего брать — исключение)."""
        model = (self.store.get_model(user_id)
                 or self.default_model).strip()
        if model:
            return model
        models = await self.client.list_models()
        if not models:
            raise OllamaChatError(
                "в Ollama нет установленных моделей "
                "(выполните: ollama pull qwen2.5:7b)")
        return models[0]

    async def answer(self, user_id: int, text: str) -> List[str]:
        """Ход диалога; возвращает части ответа (разбивка по 4096).

        Raises:
            OllamaChatError: любая ошибка сервиса/модели (хендлер
                превращает её в сообщение пользователю).
        """
        history = self.store.load(user_id, self.history_limit)
        history.append({"role": "user", "content": str(text)})
        self.store.append(user_id, "user", str(text))
        model = await self.resolve_model(user_id)
        reply = await self.client.chat(model, history)
        self.store.append(user_id, "assistant", reply)
        return split_message(reply)
