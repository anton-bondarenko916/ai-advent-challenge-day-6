"""Тесты Telegram-бота для локальной Ollama (ТЗ ai_1_31).

Проверяются офлайн (Telegram API и Ollama замоканы, БД временная):

* белый список: разбор CSV (пробелы/мусор), проверка доступа;
* split_message: короткий/граничный/длинный текст, границы слов,
  жёсткий разрез «слова» без разделителей, склейка восстанавливает;
* TelegramHistoryStore: append/load (хронология и лимит), clear,
  get/set_model, переживает переоткрытие, подрезка 200;
* OllamaChatClient (httpx.MockTransport): chat OK (URL, payload,
  stream=false), поле error, HTTP 5xx, пустой content, сеть →
  OllamaUnavailableError, list_models, base_url из окружения;
* ChatTurn: история уходит в модель с новым user-сообщением и
  дописывается с двух сторон; приоритет модели (выбор → дефолт →
  первая из /api/tags); ошибка — user-сообщение осталось,
  assistant-сообщения нет;
* хендлеры aiogram (если aiogram установлен): отказ постороннему,
  /clear, ход чата (typing + части), длинный ответ на части,
  /model с кнопками, выбор модели колбэком.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402

from src.config import get_telegram_settings, load_config  # noqa: E402
from src.tg_bot_logic import (  # noqa: E402
    ChatTurn,
    is_allowed,
    parse_allowed_ids,
    split_message,
)
from src.tg_ollama_client import (  # noqa: E402
    DEFAULT_OLLAMA_BASE,
    OllamaChatClient,
    OllamaChatError,
    OllamaUnavailableError,
)
from src.tg_storage import KEEP_PER_USER, TelegramHistoryStore  # noqa: E402

try:
    import telegram_bot as tb
    AIOGRAM_AVAILABLE = True
except ImportError:  # aiogram не установлен — хендлеры не тестируем
    AIOGRAM_AVAILABLE = False

USER = 42


class FakeClient:
    """OllamaChatClient-заглушка (запоминает вызовы chat)."""

    def __init__(self, reply: str = "готово", models=None,
                 error: Exception = None) -> None:
        self.reply = reply
        self.models = list(models or [])
        self.error = error
        self.chat_calls = []

    async def chat(self, model, messages):
        self.chat_calls.append((model, list(messages)))
        if self.error is not None:
            raise self.error
        return self.reply

    async def list_models(self):
        return list(self.models)

    async def aclose(self):
        return None


# ----------------------------------------------------------------------
# Белый список
# ----------------------------------------------------------------------
class AllowedIdsTest(unittest.TestCase):
    """parse_allowed_ids / is_allowed."""

    def test_parse_csv(self) -> None:
        self.assertEqual(parse_allowed_ids("1, 2 ,3"), frozenset({1, 2, 3}))

    def test_parse_garbage_skipped(self) -> None:
        self.assertEqual(parse_allowed_ids("1, x, ,2.5"), frozenset({1}))

    def test_parse_empty(self) -> None:
        self.assertEqual(parse_allowed_ids(""), frozenset())
        self.assertEqual(parse_allowed_ids(None), frozenset())

    def test_is_allowed(self) -> None:
        allowed = frozenset({42})
        self.assertTrue(is_allowed(42, allowed))
        self.assertFalse(is_allowed(43, allowed))
        self.assertFalse(is_allowed(None, allowed))


# ----------------------------------------------------------------------
# Разбивка длинных сообщений
# ----------------------------------------------------------------------
class SplitMessageTest(unittest.TestCase):

    def test_short_single_part(self) -> None:
        self.assertEqual(split_message("привет"), ["привет"])
        self.assertEqual(split_message(""), [""])

    def test_exact_limit_single_part(self) -> None:
        text = "а" * 4000
        self.assertEqual(split_message(text), [text])

    def test_long_splits_on_word_boundaries(self) -> None:
        words = " ".join(f"слово{i}" for i in range(1500))
        parts = split_message(words)
        self.assertGreater(len(parts), 1)
        for part in parts:
            self.assertLessEqual(len(part), 4000)
            self.assertTrue(part)
        self.assertEqual(" ".join(parts), words)

    def test_long_splits_on_newlines(self) -> None:
        text = "\n".join(f"строка {i} " + "x" * 50 for i in range(300))
        parts = split_message(text)
        for part in parts:
            self.assertLessEqual(len(part), 4000)

    def test_giant_word_hard_cut(self) -> None:
        text = "ж" * 9500
        parts = split_message(text)
        self.assertEqual("".join(parts), text)
        for part in parts[:-1]:
            self.assertEqual(len(part), 4000)

    def test_custom_limit(self) -> None:
        parts = split_message("ааа ббб ввв", limit=6)
        self.assertTrue(all(len(p) <= 6 for p in parts))


# ----------------------------------------------------------------------
# Хранилище истории
# ----------------------------------------------------------------------
class HistoryStoreTest(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "tg" / "bot.db"
        self.store = TelegramHistoryStore(self.path)

    def test_append_and_load_chronology(self) -> None:
        self.store.append(USER, "user", "первый")
        self.store.append(USER, "assistant", "ответ")
        self.store.append(USER, "user", "второй")
        self.assertEqual(self.store.load(USER, 10), [
            {"role": "user", "content": "первый"},
            {"role": "assistant", "content": "ответ"},
            {"role": "user", "content": "второй"},
        ])

    def test_load_limit_returns_tail(self) -> None:
        for i in range(5):
            self.store.append(USER, "user", f"msg{i}")
        loaded = self.store.load(USER, 2)
        self.assertEqual([m["content"] for m in loaded], ["msg3", "msg4"])

    def test_users_are_isolated(self) -> None:
        self.store.append(USER, "user", "мой")
        self.store.append(43, "user", "чужой")
        self.assertEqual(len(self.store.load(USER, 10)), 1)
        self.assertEqual(self.store.load(USER, 10)[0]["content"], "мой")

    def test_clear(self) -> None:
        self.store.append(USER, "user", "x")
        self.store.append(USER, "assistant", "y")
        self.assertEqual(self.store.clear(USER), 2)
        self.assertEqual(self.store.load(USER, 10), [])
        self.assertEqual(self.store.clear(USER), 0)

    def test_model_state_upsert(self) -> None:
        self.assertEqual(self.store.get_model(USER), "")
        self.assertTrue(self.store.set_model(USER, "qwen2.5:7b"))
        self.assertEqual(self.store.get_model(USER), "qwen2.5:7b")
        self.store.set_model(USER, "llama3.1:8b")
        self.assertEqual(self.store.get_model(USER), "llama3.1:8b")

    def test_survives_reopen(self) -> None:
        self.store.append(USER, "user", "переживи меня")
        self.store.set_model(USER, "m:1")
        reopened = TelegramHistoryStore(self.path)
        self.assertEqual(reopened.load(USER, 10)[0]["content"],
                         "переживи меня")
        self.assertEqual(reopened.get_model(USER), "m:1")

    def test_prune_keeps_last_200(self) -> None:
        for i in range(KEEP_PER_USER + 50):
            self.store.append(USER, "user", f"m{i}")
        loaded = self.store.load(USER, KEEP_PER_USER + 10)
        self.assertEqual(len(loaded), KEEP_PER_USER)
        self.assertEqual(loaded[0]["content"], "m50")
        self.assertEqual(loaded[-1]["content"], f"m{KEEP_PER_USER + 49}")


# ----------------------------------------------------------------------
# Клиент Ollama (httpx MockTransport)
# ----------------------------------------------------------------------
class OllamaClientTest(unittest.TestCase):

    def _client(self, handler) -> OllamaChatClient:
        return OllamaChatClient(
            transport=httpx.MockTransport(handler))

    def test_chat_ok(self) -> None:
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["payload"] = json.loads(request.content)
            return httpx.Response(200, json={
                "message": {"role": "assistant", "content": "привет!"}})

        client = self._client(handler)
        try:
            reply = asyncio.run(client.chat("qwen2.5:7b", [
                {"role": "user", "content": "тест"}]))
        finally:
            asyncio.run(client.aclose())
        self.assertEqual(reply, "привет!")
        self.assertIn("/api/chat", captured["url"])
        self.assertEqual(captured["payload"]["model"], "qwen2.5:7b")
        self.assertFalse(captured["payload"]["stream"])
        self.assertEqual(captured["payload"]["messages"],
                         [{"role": "user", "content": "тест"}])

    def test_chat_error_field(self) -> None:
        def handler(request):
            return httpx.Response(200, json={"error": "model not found"})

        client = self._client(handler)
        with self.assertRaises(OllamaChatError) as ctx:
            asyncio.run(client.chat("ghost", []))
        asyncio.run(client.aclose())
        self.assertIn("model not found", str(ctx.exception))

    def test_chat_http_500(self) -> None:
        def handler(request):
            return httpx.Response(500, text="boom")

        client = self._client(handler)
        with self.assertRaises(OllamaChatError):
            asyncio.run(client.chat("m", []))
        asyncio.run(client.aclose())

    def test_chat_empty_content(self) -> None:
        def handler(request):
            return httpx.Response(200, json={"message": {"content": ""}})

        client = self._client(handler)
        with self.assertRaises(OllamaChatError):
            asyncio.run(client.chat("m", []))
        asyncio.run(client.aclose())

    def test_unavailable_wraps_network_errors(self) -> None:
        def handler(request):
            raise httpx.ConnectError("connection refused")

        client = self._client(handler)
        with self.assertRaises(OllamaUnavailableError):
            asyncio.run(client.chat("m", []))
        asyncio.run(client.aclose())

    def test_list_models(self) -> None:
        def handler(request):
            self.assertIn("/api/tags", str(request.url))
            return httpx.Response(200, json={"models": [
                {"name": "qwen2.5:7b"}, {"name": ""}, {"name": "llama3.1:8b"}
            ]})

        client = self._client(handler)
        try:
            models = asyncio.run(client.list_models())
        finally:
            asyncio.run(client.aclose())
        self.assertEqual(models, ["qwen2.5:7b", "llama3.1:8b"])

    def test_base_url_from_env(self) -> None:
        with mock.patch.dict(os.environ,
                             {"OLLAMA_BASE_URL": "http://127.0.0.1:11500/"}):
            client = OllamaChatClient()
            self.assertEqual(client.base_url, "http://127.0.0.1:11500")
        client = OllamaChatClient()
        self.assertEqual(client.base_url, DEFAULT_OLLAMA_BASE)


# ----------------------------------------------------------------------
# Ход чата (ChatTurn)
# ----------------------------------------------------------------------
class ChatTurnTest(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = TelegramHistoryStore(Path(self._tmp.name) / "db.db")
        self.client = FakeClient(reply="короткий ответ",
                                 models=["заглушка:7b"])

    def test_answer_appends_both_sides(self) -> None:
        self.store.append(USER, "user", "старый вопрос")
        self.store.append(USER, "assistant", "старый ответ")
        turn = ChatTurn(self.client, self.store)
        parts = asyncio.run(turn.answer(USER, "новый вопрос"))
        self.assertEqual(parts, ["короткий ответ"])
        model, messages = self.client.chat_calls[0]
        self.assertEqual(messages, [
            {"role": "user", "content": "старый вопрос"},
            {"role": "assistant", "content": "старый ответ"},
            {"role": "user", "content": "новый вопрос"},
        ])
        self.assertEqual(
            [m["role"] for m in self.store.load(USER, 10)],
            ["user", "assistant", "user", "assistant"])

    def test_answer_long_reply_split(self) -> None:
        self.client.reply = " ".join(f"слово{i}" for i in range(1500))
        turn = ChatTurn(self.client, self.store)
        parts = asyncio.run(turn.answer(USER, "вопрос"))
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(p) <= 4000 for p in parts))

    def test_history_limit_sent_to_model(self) -> None:
        for i in range(10):
            self.store.append(USER, "user", f"m{i}")
        turn = ChatTurn(self.client, self.store, history_limit=4)
        asyncio.run(turn.answer(USER, "вопрос"))
        _, messages = self.client.chat_calls[0]
        # хвост истории (4) + новое сообщение
        self.assertEqual(len(messages), 5)
        self.assertEqual(messages[0]["content"], "m6")
        self.assertEqual(messages[-1]["content"], "вопрос")

    def test_error_keeps_user_message_without_assistant(self) -> None:
        self.client.error = OllamaChatError("модель упала")
        turn = ChatTurn(self.client, self.store)
        with self.assertRaises(OllamaChatError):
            asyncio.run(turn.answer(USER, "вопрос"))
        loaded = self.store.load(USER, 10)
        self.assertEqual([m["role"] for m in loaded], ["user"])
        self.assertEqual(loaded[0]["content"], "вопрос")

    def test_model_priority_picked_first(self) -> None:
        self.store.set_model(USER, "выбранная:8b")
        turn = ChatTurn(self.client, self.store, default_model="дефолт")
        asyncio.run(turn.answer(USER, "в"))
        self.assertEqual(self.client.chat_calls[0][0], "выбранная:8b")

    def test_model_priority_default_second(self) -> None:
        turn = ChatTurn(self.client, self.store, default_model="дефолт:7b")
        asyncio.run(turn.answer(USER, "в"))
        self.assertEqual(self.client.chat_calls[0][0], "дефолт:7b")

    def test_model_priority_first_installed_last(self) -> None:
        self.client.models = ["первая:7b", "вторая:8b"]
        turn = ChatTurn(self.client, self.store)
        asyncio.run(turn.answer(USER, "в"))
        self.assertEqual(self.client.chat_calls[0][0], "первая:7b")

    def test_no_models_raises(self) -> None:
        self.client.models = []
        turn = ChatTurn(self.client, self.store)
        with self.assertRaises(OllamaChatError):
            asyncio.run(turn.answer(USER, "в"))


# ----------------------------------------------------------------------
# Настройки (get_telegram_settings)
# ----------------------------------------------------------------------
class TelegramSettingsTest(unittest.TestCase):

    def test_from_env(self) -> None:
        with mock.patch.dict(os.environ, {
                "TELEGRAM_BOT_TOKEN": "tok",
                "TELEGRAM_ALLOWED_IDS": "1, 2",
                "TELEGRAM_MODEL": "qwen2.5:7b",
                "TELEGRAM_HISTORY_LIMIT": "7"}):
            settings = get_telegram_settings()
        self.assertEqual(settings["token"], "tok")
        self.assertEqual(settings["allowed_ids"], "1, 2")
        self.assertEqual(settings["model"], "qwen2.5:7b")
        self.assertEqual(settings["history_limit"], 7)
        self.assertTrue(Path(settings["db_path"]).is_absolute())

    def test_bad_limit_falls_back(self) -> None:
        settings = get_telegram_settings({"TELEGRAM_HISTORY_LIMIT": "abc"})
        self.assertEqual(settings["history_limit"], 20)

    def test_limit_clamped(self) -> None:
        self.assertEqual(get_telegram_settings(
            {"TELEGRAM_HISTORY_LIMIT": 1})["history_limit"], 2)
        self.assertEqual(get_telegram_settings(
            {"TELEGRAM_HISTORY_LIMIT": 1000})["history_limit"], 100)

    def test_env_file_reaches_settings_via_load_config(self) -> None:
        """Регрессия ai_1_31: .env читает ТОЛЬКО load_config().

        Раньше бот звал get_config() напрямую — переменные из .env не
        подхватывались и бот писал «TELEGRAM_BOT_TOKEN не задан», хотя
        файл был заполнен (GUI-приложение работало, т.к. его main.py
        вызывает load_config).
        """
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ):  # изоляция os.environ
            os.environ.pop("TELEGRAM_BOT_TOKEN", None)
            os.environ.pop("TELEGRAM_ALLOWED_IDS", None)
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "# комментарий\n"
                "TELEGRAM_BOT_TOKEN=test-token-123\n"
                "TELEGRAM_ALLOWED_IDS=42,43\n",
                encoding="utf-8")
            config = load_config(env_file=env_path)
            settings = get_telegram_settings(config)
        self.assertEqual(settings["token"], "test-token-123")
        self.assertIn("42", settings["allowed_ids"])

    def test_main_without_token_returns_1(self) -> None:
        """main() без токена печатает инструкцию и выходит с кодом 1."""
        import telegram_bot as tb

        with mock.patch.object(tb, "get_telegram_settings",
                               return_value={"token": "",
                                             "allowed_ids": "",
                                             "model": "",
                                             "history_limit": 20,
                                             "db_path": "x.db"}), \
                mock.patch("builtins.print") as printed:
            code = tb.main()
        self.assertEqual(code, 1)
        self.assertTrue(any("TELEGRAM_BOT_TOKEN" in str(call)
                            for call in printed.call_args_list))


# ----------------------------------------------------------------------
# Хендлеры aiogram (тонкий слой; только если aiogram установлен)
# ----------------------------------------------------------------------
def _message(user_id, text=""):
    return SimpleNamespace(
        from_user=SimpleNamespace(id=user_id),
        chat=SimpleNamespace(id=user_id),
        text=text,
        answer=mock.AsyncMock(),
    )


@unittest.skipUnless(AIOGRAM_AVAILABLE,
                     "aiogram не установлен — хендлеры недоступны")
class TelegramBotHandlersTest(unittest.TestCase):
    """Команды и ход чата (бот-мок, клиент-заглушка, временная БД)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = TelegramHistoryStore(Path(self._tmp.name) / "db.db")
        self.client = FakeClient(reply="готово", models=["m1:7b", "m2:8b"])
        self.settings = {
            "token": "test-token",
            "allowed_ids": f"{USER},43",
            "model": "",
            "history_limit": 20,
            "db_path": str(Path(self._tmp.name) / "db.db"),
        }
        self.bot = tb.TelegramChatBot(
            self.settings, client=self.client, store=self.store)
        self.bot.bot = mock.AsyncMock()

    def _msg(self, user_id=USER, text=""):
        return _message(user_id, text)

    def test_stranger_refused_without_echo(self) -> None:
        message = self._msg(user_id=999, text="дай мне всё")
        asyncio.run(self.bot.on_text(message))
        reply = message.answer.call_args.args[0]
        self.assertIn("Нет доступа", reply)
        self.assertIn("999", reply)
        # Эхо-утечка: текст постороннего НЕ повторяется
        self.assertNotIn("дай мне всё", reply)

    def test_stranger_logged_once(self) -> None:
        for _ in range(3):
            asyncio.run(self.bot.cmd_start(self._msg(user_id=999)))
        self.assertEqual(self.bot._refused_logged, {999})

    def test_start_greeting_mentions_model(self) -> None:
        self.store.set_model(USER, "m1:7b")
        message = self._msg()
        asyncio.run(self.bot.cmd_start(message))
        reply = message.answer.call_args.args[0]
        self.assertIn("Привет", reply)
        self.assertIn("m1:7b", reply)
        asyncio.run(self.bot.cmd_help(self._msg()))

    def test_clear_removes_history(self) -> None:
        self.store.append(USER, "user", "q")
        self.store.append(USER, "assistant", "a")
        message = self._msg()
        asyncio.run(self.bot.cmd_clear(message))
        self.assertIn("очищена (2", message.answer.call_args.args[0])
        self.assertEqual(self.store.load(USER, 10), [])

    def test_on_text_happy_path(self) -> None:
        message = self._msg(text="вопрос")
        asyncio.run(self.bot.on_text(message))
        message.answer.assert_awaited_once_with("готово")
        # Клиент получил историю с новым user-сообщением
        _, messages = self.client.chat_calls[0]
        self.assertEqual(messages[-1], {"role": "user",
                                        "content": "вопрос"})
        self.assertEqual([m["role"] for m in self.store.load(USER, 10)],
                         ["user", "assistant"])

    def test_on_text_long_answer_in_parts(self) -> None:
        self.client.reply = " ".join(f"слово{i}" for i in range(1500))
        message = self._msg(text="вопрос")
        with mock.patch.object(tb, "SEND_PART_DELAY", 0):
            asyncio.run(self.bot.on_text(message))
        self.assertGreater(message.answer.await_count, 1)
        for call in message.answer.await_args_list:
            self.assertLessEqual(len(call.args[0]), 4000)

    def test_on_text_ollama_unavailable(self) -> None:
        self.client.error = OllamaUnavailableError("сервер не отвечает")
        message = self._msg(text="вопрос")
        asyncio.run(self.bot.on_text(message))
        self.assertIn("недоступна", message.answer.call_args.args[0])

    def test_on_text_generic_error(self) -> None:
        self.client.error = RuntimeError("что-то совсем сломалось")
        message = self._msg(text="вопрос")
        asyncio.run(self.bot.on_text(message))
        self.assertIn("Внутренняя ошибка",
                      message.answer.call_args.args[0])

    def test_model_command_shows_buttons(self) -> None:
        message = self._msg()
        asyncio.run(self.bot.cmd_model(message))
        markup = message.answer.call_args.kwargs["reply_markup"]
        texts = [button.text for row in markup.inline_keyboard
                 for button in row]
        self.assertEqual(texts, ["m1:7b", "m2:8b"])

    def test_model_pick_saves_and_edits(self) -> None:
        callback = SimpleNamespace(
            from_user=SimpleNamespace(id=USER),
            data="model:m2:8b",
            answer=mock.AsyncMock(),
            message=SimpleNamespace(edit_text=mock.AsyncMock()),
        )
        asyncio.run(self.bot.on_model_pick(callback))
        self.assertEqual(self.store.get_model(USER), "m2:8b")
        callback.answer.assert_awaited_once()
        callback.message.edit_text.assert_awaited_once()
        self.assertIn("m2:8b",
                      callback.message.edit_text.call_args.args[0])
        # Выбор используется в следующем ходе
        asyncio.run(self.bot.on_text(self._msg(text="вопрос")))
        self.assertEqual(self.client.chat_calls[0][0], "m2:8b")

    def test_model_pick_refuses_stranger(self) -> None:
        callback = SimpleNamespace(
            from_user=SimpleNamespace(id=999),
            data="model:m1:7b",
            answer=mock.AsyncMock(),
            message=None,
        )
        asyncio.run(self.bot.on_model_pick(callback))
        self.assertEqual(self.store.get_model(999), "")
        callback.answer.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
