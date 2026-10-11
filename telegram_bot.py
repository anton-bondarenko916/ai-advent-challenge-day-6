"""Telegram-бот для локальной LLM (Ollama) — точка входа (ТЗ ai_1_31).

Схема: Telegram ↔ long polling (только ИСХОДЯЩИЕ соединения — порты
наружу не открываются, брандмауэр/DDNS/Nginx не нужны) → POST
/api/chat локальной Ollama. Запуск: ``python telegram_bot.py`` (токен
и белый список — в .env, см. docs/telegram_bot.md).

Слой aiogram сознательно тонкий: доступ (белый список user ID),
команды, typing-индикатор, отправка частей. Вся логика (разбивка 4096,
история в SQLite, запросы к Ollama) — в src/tg_bot_logic.py и
тестируется офлайн.

Правила хендлеров:

* гейт белого списка первым делом; незнакомец получает стандартный
  отказ (БЕЗ повтора его текста), его ID — в лог один раз;
* parse_mode НЕ задаётся: сырой вывод модели содержит <, *, ``` —
  HTML/Markdown-парсер Telegram падает;
* ответ длиннее 4096 символов уходит несколькими сообщениями с паузой
  (flood-лимиты Telegram ~1 сообщение/с на чат);
* любая ошибка — traceback в лог + сообщение пользователю; цикл
  polling не умирает.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

try:  # aiogram ставится вместе с requirements.txt; без него бот не нужен
    from aiogram import Bot, Dispatcher, F, Router
    from aiogram.filters import Command, CommandStart
    from aiogram.types import (CallbackQuery, InlineKeyboardButton,
                               InlineKeyboardMarkup, Message)
    from aiogram.utils.chat_action import ChatActionSender
except ImportError as exc:  # pragma: no cover — окружение без aiogram
    raise ImportError(
        "Не установлен aiogram — выполните: pip install -r requirements.txt"
    ) from exc

from src.config import ENV_PATH, get_telegram_settings, load_config
from src.tg_bot_logic import (
    ChatTurn,
    is_allowed,
    parse_allowed_ids,
    split_message,
)
from src.tg_ollama_client import OllamaChatClient, OllamaChatError, \
    OllamaUnavailableError
from src.tg_storage import TelegramHistoryStore

logger = logging.getLogger("deepseek-chat.telegram-bot")

LOG_PATH = Path(__file__).resolve().parent / "telegram_bot.log"
# Пауза между частями длинного ответа (flood-лимиты Telegram)
SEND_PART_DELAY = 0.4

HELP_TEXT = (
    "🤖 Бот локальной модели (Ollama на домашнем ПК).\n\n"
    "Команды:\n"
    "/model — выбрать модель (список установленных)\n"
    "/clear — очистить историю диалога\n"
    "/help — эта справка\n\n"
    "Просто пишите сообщения — отвечу из локальной модели. "
    "История хранится на ПК отдельно для каждого пользователя."
)

REFUSAL_TEXT = ("⛔ Нет доступа к этому боту. "
                "Ваш ID: {user_id} — сообщите его владельцу.")


class TelegramChatBot:
    """Сборка бота: роутер aiogram + сервисы (история, клиент Ollama).

    Args:
        settings: словарь get_telegram_settings()
            (token/allowed_ids/model/history_limit/db_path).
        client/store: подстановка для тестов; None — создать по настройкам.
    """

    def __init__(self, settings: Dict[str, Any],
                 client: Optional[OllamaChatClient] = None,
                 store: Optional[TelegramHistoryStore] = None) -> None:
        self.token = str(settings.get("token") or "")
        self.allowed = parse_allowed_ids(settings.get("allowed_ids") or "")
        self.client = client or OllamaChatClient(
            model_default=str(settings.get("model") or ""))
        self.store = store or TelegramHistoryStore(settings.get("db_path")
                                                   or "data/telegram_bot.db")
        self.turn = ChatTurn(
            self.client, self.store,
            history_limit=int(settings.get("history_limit") or 20),
            default_model=str(settings.get("model") or ""),
        )
        # Бот появляется в run(); для офлайн-тестов подставляют мок
        self.bot: Optional[Bot] = None
        self._refused_logged: set = set()

        self.router = Router()
        self._register_handlers()

    # ------------------------------------------------------------------
    # Регистрация хендлеров
    # ------------------------------------------------------------------
    def _register_handlers(self) -> None:
        self.router.message.register(self.cmd_start, CommandStart())
        self.router.message.register(self.cmd_help, Command("help"))
        self.router.message.register(self.cmd_clear, Command("clear"))
        self.router.message.register(self.cmd_model, Command("model"))
        self.router.callback_query.register(
            self.on_model_pick, F.data.startswith("model:"))
        self.router.message.register(self.on_text, F.text)

    # ------------------------------------------------------------------
    # Гейт доступа
    # ------------------------------------------------------------------
    def _check_access(self, user_id: Optional[int]) -> bool:
        """Белый список; отказ чужому — в чат и в лог (один раз на ID)."""
        if is_allowed(user_id, self.allowed):
            return True
        if user_id not in self._refused_logged:
            self._refused_logged.add(user_id)
            logger.warning(
                "Попытка доступа постороннего: user_id=%s "
                "(добавьте в TELEGRAM_ALLOWED_IDS при необходимости)",
                user_id)
        return False

    # ------------------------------------------------------------------
    # Команды
    # ------------------------------------------------------------------
    async def cmd_start(self, message: Message) -> None:
        user_id = getattr(message.from_user, "id", None)
        if not self._check_access(user_id):
            await message.answer(REFUSAL_TEXT.format(user_id=user_id))
            return
        current = self.store.get_model(user_id) or self.turn.default_model
        model_line = f"\nТекущая модель: {current}" if current else ""
        await message.answer(
            f"🤖 Привет! Я отвечаю из локальной модели на домашнем ПК."
            f"{model_line}\n\n{HELP_TEXT}")

    async def cmd_help(self, message: Message) -> None:
        user_id = getattr(message.from_user, "id", None)
        if not self._check_access(user_id):
            await message.answer(REFUSAL_TEXT.format(user_id=user_id))
            return
        await message.answer(HELP_TEXT)

    async def cmd_clear(self, message: Message) -> None:
        user_id = getattr(message.from_user, "id", None)
        if not self._check_access(user_id):
            await message.answer(REFUSAL_TEXT.format(user_id=user_id))
            return
        removed = self.store.clear(user_id)
        await message.answer(f"🗑 История диалога очищена ({removed} сообщений).")

    async def cmd_model(self, message: Message) -> None:
        user_id = getattr(message.from_user, "id", None)
        if not self._check_access(user_id):
            await message.answer(REFUSAL_TEXT.format(user_id=user_id))
            return
        try:
            models = await self.client.list_models()
        except OllamaChatError as exc:
            await message.answer(f"⚠️ Не получил список моделей: {exc}")
            return
        if not models:
            await message.answer(
                "В Ollama нет установленных моделей — выполните на ПК: "
                "ollama pull qwen2.5:7b")
            return
        current = self.store.get_model(user_id) or self.turn.default_model
        buttons = [
            InlineKeyboardButton(
                text=("✅ " if name == current else "") + name,
                callback_data=f"model:{name}",
            )
            for name in models
        ]
        rows = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]
        await message.answer(
            "Выберите модель:", reply_markup=InlineKeyboardMarkup(
                inline_keyboard=rows))

    async def on_model_pick(self, callback: CallbackQuery) -> None:
        user_id = getattr(callback.from_user, "id", None)
        if not self._check_access(user_id):
            await callback.answer("⛔ Нет доступа", show_alert=True)
            return
        model = str(callback.data or "")[len("model:"):].strip()
        if not model:
            await callback.answer("Пустое имя модели")
            return
        self.store.set_model(user_id, model)
        await callback.answer(f"Модель: {model}")
        if callback.message is not None:
            await callback.message.edit_text(f"✅ Модель выбрана: {model}")

    # ------------------------------------------------------------------
    # Обычный ход чата
    # ------------------------------------------------------------------
    async def on_text(self, message: Message) -> None:
        user_id = getattr(message.from_user, "id", None)
        if not self._check_access(user_id):
            await message.answer(REFUSAL_TEXT.format(user_id=user_id))
            return
        text = str(message.text or "").strip()
        if not text:
            return
        chat_id = getattr(message.chat, "id", user_id)
        try:
            # Индикатор «печатает…» на всё время генерации
            async with ChatActionSender(bot=self.bot, chat_id=chat_id):
                parts = await self.turn.answer(user_id, text)
        except OllamaUnavailableError as exc:
            await message.answer(
                f"⚠️ Ollama недоступна: {exc}. Проверьте на ПК: "
                "ollama serve")
            return
        except OllamaChatError as exc:
            await message.answer(f"⚠️ Ошибка модели: {exc}")
            return
        except Exception:  # noqa: BLE001 — ход не должен ронять бота
            logger.exception("Ошибка хода чата (user_id=%s)", user_id)
            await message.answer("⚠️ Внутренняя ошибка — попробуйте ещё раз.")
            return
        for index, part in enumerate(parts):
            if index:
                await asyncio.sleep(SEND_PART_DELAY)
            await message.answer(part)

    # ------------------------------------------------------------------
    # Запуск
    # ------------------------------------------------------------------
    async def run(self) -> None:
        """Long polling до Ctrl+C; закрытие клиента Ollama — в finally."""
        if not self.token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN не задан")
        self.bot = Bot(token=self.token)
        dispatcher = Dispatcher()
        dispatcher.include_router(self.router)
        # Остатки webhook мешают getUpdates — сбрасываем
        await self.bot.delete_webhook(drop_pending_updates=True)
        logger.info("Бот запущен (long polling); разрешено пользователей: %s",
                    len(self.allowed))
        try:
            await dispatcher.start_polling(self.bot)
        finally:
            await self.client.aclose()


def _setup_logging() -> None:
    """Журнал в файл (UTF-8) + консоль; токены в лог не попадают."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def main() -> int:
    """Точка входа: настройки из .env → TelegramChatBot.run().

    ВАЖНО: .env читает только load_config() (как в GUI-приложении);
    get_config()/get_telegram_settings() переменные из файла НЕ
    подхватывают — без load_config() бот не увидит токен вообще.
    """
    _setup_logging()
    settings = get_telegram_settings(load_config())
    if not settings["token"]:
        print(
            "TELEGRAM_BOT_TOKEN не задан.\n"
            f"1) Впишите в .env ({ENV_PATH}):\n"
            "   TELEGRAM_BOT_TOKEN=123456:ABC-DEF...\n"
            "   TELEGRAM_ALLOWED_IDS=123456789\n"
            "   (строки должны быть раскомментированы, без пробела в начале)\n"
            "2) Токен — у @BotFather, свой числовой ID — у @userinfobot.\n"
            "Подробности: docs/telegram_bot.md")
        return 1
    if not settings["allowed_ids"]:
        logger.warning(
            "TELEGRAM_ALLOWED_IDS пуст — бот откажет всем пользователям.")
    app = TelegramChatBot(settings)
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        logger.info("Бот остановлен (Ctrl+C).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
