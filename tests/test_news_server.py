"""Тесты MCP-сервера «News» и agentic loop (search → summarize → save).

Покрывают сценарии приёмки ТЗ:

 1. Сервер запускается: ``python news_server.py`` → HTTP-порт отвечает;
 2. list_tools() возвращает search_news, summarize_text, save_to_file;
 3. search_news работает: фильтр по теме в RSS (локальная фикстура),
    недоступные ленты пропускаются;
 4. summarize_text работает: вызов DeepSeek API (мок) → Markdown-сводка;
 5. save_to_file работает: файл создан в output/ (временной каталог);
 6. Agentic Loop: LLM сама строит цепочку search_news → summarize_text →
    save_to_file, результат каждого шага возвращается в LLM, путь файла
    попадает в результат для кнопки «Открыть файл»;
 7. Кнопка «Открыть файл»: после сохранения в UI появляется кнопка,
    файл открывается (os.startfile замокан);
 8. Fallback summarize_text: при ошибке API возвращается сырой текст;
 9. Лимит итераций: зацикливание LLM обрывается лимитом раундов.

RSS для тестов — локальная фикстура tests/fixtures/rss_news.xml через
``file://`` URL (инструмент одинаково читает её и сетевые ленты).
Интеграционные тесты требуют SDK ``mcp``, иначе пропускаются.

Запуск из корня проекта:
    python -m unittest tests.test_news_server -v
"""

from __future__ import annotations

import asyncio
import json
import queue
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.config import DEFAULTS, get_news_settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RSS_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "rss_news.xml"
NEWS_SERVER = str(PROJECT_ROOT / "news_server.py")

TOOLS = ["search_news", "summarize_text", "save_to_file"]

try:  # pragma: no cover — зависит от окружения
    import tkinter as tk
except ImportError:  # pragma: no cover
    tk = None  # type: ignore

from tests import gui_helpers

try:  # pragma: no cover — зависит от окружения
    import mcp  # noqa: F401

    MCP_SDK = True
except ImportError:  # pragma: no cover
    MCP_SDK = False

if MCP_SDK:
    from src.reminder_client import ReminderMCPClient


def _free_port() -> int:
    """Свободный TCP-порт на 127.0.0.1 (для экземпляра сервера в тестах)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _unwrap(payload):
    """Снимает обёртку annotated structured content FastMCP."""
    if isinstance(payload, dict) and set(payload) == {"result"}:
        return payload["result"]
    return payload


# ---------------------------------------------------------------------------
# Юнит-тесты инструментов (без сети; RSS — локальная фикстура)
# ---------------------------------------------------------------------------


class NewsToolsUnitTest(unittest.TestCase):
    """Инструменты news_server: search/save напрямую, summarize с моком."""

    def setUp(self) -> None:
        import news_server

        self.ns = news_server
        tmp = Path(tempfile.mkdtemp(prefix="news_out_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(
            tmp, ignore_errors=True))
        self.output_dir = tmp
        patcher = mock.patch.object(self.ns, "OUTPUT_DIR", tmp)
        patcher.start()
        self.addCleanup(patcher.stop)
        # Ленты подменяем на локальную фикстуру (две «копии» — проверка
        # слияния результатов разных источников)
        patcher2 = mock.patch.object(self.ns, "RSS_FEEDS", {
            "test": RSS_FIXTURE.as_uri(),
        })
        patcher2.start()
        self.addCleanup(patcher2.stop)

    # -- ТЗ 3: поиск по теме --

    def test_search_news_filters_by_topic(self) -> None:
        """search_news фильтрует записи по теме (заголовок/описание)."""
        result = json.loads(self.ns.search_news("технологии"))
        self.assertNotIn("error", result)
        self.assertEqual(result["topic"], "технологии")
        self.assertEqual(result["count"], 2)
        titles = [n["title"] for n in result["news"]]
        self.assertTrue(all("технолог" in t.lower() for t in titles))
        self.assertEqual(result["news"][0]["source"], "test")
        self.assertTrue(result["news"][0]["link"])

    def test_search_news_no_matches(self) -> None:
        """Нет новостей по теме → {"error": ...} (ТЗ: пропускаем пустоту)."""
        result = json.loads(self.ns.search_news("кванторы"))
        self.assertIn("error", result)

    def test_search_news_skips_dead_feeds(self) -> None:
        """Недоступная лента пропускается, поиск идёт по остальным."""
        with mock.patch.object(self.ns, "RSS_FEEDS", {
            "dead": "file:///Z:/definitely/missing/feed.xml",
            "test": RSS_FIXTURE.as_uri(),
        }):
            result = json.loads(self.ns.search_news("технологии"))
        self.assertEqual(result["count"], 2)

    def test_search_news_limit(self) -> None:
        """limit ограничивает выдачу."""
        result = json.loads(self.ns.search_news("е", limit=1))
        self.assertEqual(result["count"], 1)

    # -- Поисковые ленты ({topic}-шаблон, например Google News) --

    def test_topic_feed_substitutes_and_encodes_topic(self) -> None:
        """{topic} подставляется в URL с URL-кодированием (кириллица)."""
        captured = {}

        def fake_fetch(url: str) -> bytes:
            captured["url"] = url
            return RSS_FIXTURE.read_bytes()

        with mock.patch.object(self.ns, "RSS_FEEDS", {
            "gq": "https://news.test/rss?q={topic}&hl=ru",
        }), mock.patch.object(self.ns, "_fetch_feed", side_effect=fake_fetch):
            result = json.loads(self.ns.search_news("технологии"))

        # Кириллица закодирована quote_plus («сырых» букв в URL нет)
        self.assertTrue(captured["url"].startswith(
            "https://news.test/rss?q=%D1%82"), captured["url"])
        # Фильтр не применялся: вернулись ВСЕ записи фикстуры (4), хотя
        # слово «технологи» есть только в двух
        self.assertEqual(result["count"], 4)

    def test_topic_feed_returns_results_without_word_match(self) -> None:
        """Для {topic}-ленты тема не обязана встречаться в тексте записи."""
        with mock.patch.object(self.ns, "RSS_FEEDS", {
            "gq": "https://news.test/rss?q={topic}",
        }), mock.patch.object(self.ns, "_fetch_feed",
                              return_value=RSS_FIXTURE.read_bytes()):
            result = json.loads(self.ns.search_news("квантовые сенсоры"))
        # Статическая лента дала бы 0, поисковая — все свои записи
        self.assertNotIn("error", result)
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["news"][0]["source"], "gq")

    def test_static_feed_still_filters_locally(self) -> None:
        """Обычная лента (без {topic}) фильтруется как раньше."""
        with mock.patch.object(self.ns, "RSS_FEEDS", {
            "test": RSS_FIXTURE.as_uri(),
        }):
            result = json.loads(self.ns.search_news("квантовые сенсоры"))
        self.assertIn("error", result)

    # -- ТЗ 4: суммаризация через DeepSeek API --

    def test_summarize_text_calls_api(self) -> None:
        """summarize_text вызывает DeepSeek API и возвращает сводку."""
        with mock.patch.object(
            self.ns, "_summarize_via_api",
            return_value="# Сводка\n\nТехнологии развиваются.",
        ) as api, mock.patch.dict(
            "os.environ", {"DEEPSEEK_API_KEY": "test-key"}
        ):
            result = self.ns.summarize_text("новости про технологии")
        self.assertEqual(result, "# Сводка\n\nТехнологии развиваются.")
        api.assert_called_once()
        args = api.call_args[0]
        self.assertEqual(args[0], "новости про технологии")
        self.assertEqual(args[1], "test-key")

    # -- ТЗ 8: fallback при ошибке API --

    def test_summarize_text_fallback_on_api_error(self) -> None:
        """Ошибка API → сообщение + сырой текст (продолжение цепочки)."""
        with mock.patch.object(
            self.ns, "_summarize_via_api",
            side_effect=RuntimeError("connection refused"),
        ), mock.patch.dict(
            "os.environ", {"DEEPSEEK_API_KEY": "test-key"}
        ):
            result = self.ns.summarize_text("сырые новости")
        self.assertIn("Не удалось сделать сводку", result)
        self.assertIn("сырые новости", result)

    def test_summarize_text_empty_content_falls_back(self) -> None:
        """Пустой content API (thinking-модель) — НЕ пустой результат.

        Раньше пустой content возвращался как "" — LLM-агент получал
        пустой результат и отвечал «Сводка недоступна». Теперь пустой
        ответ считается сбоем: fallback с сырыми данными.
        """
        with mock.patch.object(
            self.ns, "_summarize_via_api", return_value="   ",
        ), mock.patch.dict(
            "os.environ", {"DEEPSEEK_API_KEY": "test-key"}
        ):
            result = self.ns.summarize_text("сырые новости")
        self.assertIn("Не удалось сделать сводку", result)
        self.assertIn("API вернул пустой ответ", result)
        self.assertIn("сырые новости", result)

    def test_summarize_via_api_raises_on_empty_content(self) -> None:
        """_summarize_via_api: content=None от thinking-модели → RuntimeError."""
        message = SimpleNamespace(content=None, reasoning_content="мысли")
        choice = SimpleNamespace(message=message, finish_reason="stop")
        response = SimpleNamespace(choices=[choice])
        client = mock.Mock()
        client.chat.completions.create.return_value = response
        with mock.patch("openai.OpenAI", return_value=client):
            with self.assertRaises(RuntimeError) as ctx:
                self.ns._summarize_via_api("новости", "key", "model")
        self.assertIn("пустой ответ", str(ctx.exception))

    def test_summarize_text_retry_then_success(self) -> None:
        """Первая попытка пустая, вторая успешная — возвращается сводка."""
        with mock.patch.object(
            self.ns, "_summarize_via_api",
            side_effect=["", "# Сводка\n\nГотово."],
        ) as api, mock.patch.dict(
            "os.environ", {"DEEPSEEK_API_KEY": "test-key"}
        ):
            result = self.ns.summarize_text("сырые новости")
        self.assertEqual(result, "# Сводка\n\nГотово.")
        self.assertEqual(api.call_count, 2)

    def test_summarize_text_without_api_key(self) -> None:
        """Без ключа API — понятное сообщение и сырой текст."""
        with mock.patch.dict(
            "os.environ", {"DEEPSEEK_API_KEY": ""}, clear=False
        ):
            result = self.ns.summarize_text("сырые новости")
        self.assertIn("DEEPSEEK_API_KEY не задан", result)
        self.assertIn("сырые новости", result)

    # -- ТЗ 5: сохранение в output/ --

    def test_save_to_file_creates_markdown(self) -> None:
        """save_to_file пишет news_<штамп>.md и возвращает абсолютный путь."""
        path = self.ns.save_to_file("# Сводка\n\nТекст.")
        self.assertTrue(path.startswith("Ошибка") is False)
        file_path = Path(path)
        self.assertTrue(file_path.is_absolute())
        self.assertTrue(file_path.name.startswith("news_"))
        self.assertTrue(file_path.name.endswith(".md"))
        self.assertEqual(
            file_path.read_text(encoding="utf-8"), "# Сводка\n\nТекст."
        )

    def test_save_to_file_empty_content(self) -> None:
        """Пустой контент не сохраняется (ошибка, без файла)."""
        before = set(self.output_dir.glob("*.md"))
        result = self.ns.save_to_file("   ")
        self.assertTrue(result.startswith("Ошибка"))
        self.assertEqual(set(self.output_dir.glob("*.md")), before)


# ---------------------------------------------------------------------------
# HTTP-интеграция: сервер подпроцессом (ТЗ 1, 2, 3, 5, 8)
# ---------------------------------------------------------------------------


@unittest.skipUnless(MCP_SDK, "требуется SDK mcp")
class NewsServerHTTPTest(unittest.TestCase):
    """news_server на реальном HTTP: tools/list, инструменты, fallback."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp_dir = Path(tempfile.mkdtemp(prefix="news_http_"))
        cls.output_dir = cls.tmp_dir / "output"
        # Локальный HTTP-сервер с фикстурой: имитирует поисковую ленту
        # ({topic}-шаблон) без обращения к внешним сайтам
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        fixture_bytes = RSS_FIXTURE.read_bytes()

        class _RSSHandler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 — API http.server
                self.send_response(200)
                self.send_header("Content-Type", "application/rss+xml")
                self.end_headers()
                self.wfile.write(fixture_bytes)

            def log_message(self, *args):  # тишина в выводе тестов
                pass

        cls.rss_server = ThreadingHTTPServer(("127.0.0.1", 0), _RSSHandler)
        cls.rss_port = cls.rss_server.server_address[1]
        threading.Thread(
            target=cls.rss_server.serve_forever, daemon=True
        ).start()

        cls.port = _free_port()
        cls.url = f"http://127.0.0.1:{cls.port}/mcp"
        cls.log_file = cls.tmp_dir / "news_server.log"
        cls.process = subprocess.Popen(
            [
                sys.executable, "-u", NEWS_SERVER,
                "--port", str(cls.port),
                "--output-dir", str(cls.output_dir),
                # Поисковая лента ({topic}) на локальном HTTP-сервере +
                # статическая фикстура через file:// — всё офлайн
                "--feeds",
                "gq=http://127.0.0.1:%d/rss?q={topic};test=%s" % (
                    cls.rss_port, RSS_FIXTURE.as_uri(),
                ),
                # Пустой ключ — summarize_text уходит в fallback (ТЗ 8)
            ],
            stdout=open(cls.log_file, "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env={**__import__("os").environ, "DEEPSEEK_API_KEY": ""},
        )
        cls.client = ReminderMCPClient(cls.url)
        from src.reminder_launcher import HttpMCPServerLauncher

        deadline = time.monotonic() + 20.0
        launcher = HttpMCPServerLauncher(
            server_path=Path(NEWS_SERVER), port=cls.port,
        )
        while time.monotonic() < deadline:
            if launcher.is_server_running():
                return
            if cls.process.poll() is not None:
                raise RuntimeError(
                    f"Сервер упал при старте, код {cls.process.returncode}. "
                    f"Лог: {cls.log_file}"
                )
            time.sleep(0.3)
        raise RuntimeError("News-сервер не поднялся за 20 секунд")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.process.terminate()
        try:
            cls.process.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover
            cls.process.kill()
        cls.rss_server.shutdown()
        cls.rss_server.server_close()
        import shutil

        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def _call(self, name: str, arguments=None):
        return asyncio.run(self.client.call_tool(name, arguments or {}))

    # -- ТЗ 1: сервер запускается --

    def test_server_is_up(self) -> None:
        """News-сервер отвечает по HTTP на своём порту."""
        self.assertIsNone(self.process.poll())

    # -- ТЗ 2: три инструмента --

    def test_list_tools_returns_three(self) -> None:
        """list_tools возвращает search_news/summarize_text/save_to_file."""
        tools = asyncio.run(self.client.list_tools())
        self.assertEqual([t.name for t in tools], sorted(TOOLS))
        schemas = {t.name: t.input_schema for t in tools}
        descriptions = {t.name: t.description for t in tools}
        self.assertIn("topic", schemas["search_news"]["properties"])
        self.assertIn("text", schemas["summarize_text"]["properties"])
        self.assertIn("content", schemas["save_to_file"]["properties"])
        self.assertIn("RSS", descriptions["search_news"])

    # -- ТЗ 3: поиск по HTTP --

    def test_search_news_via_http(self) -> None:
        """search_news по HTTP: поисковая лента + статическая фикстура.

        Инструмент возвращает JSON-СТРОКУ — FastMCP оборачивает её в
        structuredContent ``{"result": "<json>"}``, поэтому после
        unwrap нужен второй json.loads. Поисковая лента gq отдаёт все
        4 записи (фильтр на её стороне), статическая test — 2 совпадения.
        """
        raw = _unwrap(json.loads(self._call(
            "search_news", {"topic": "технологии"},
        )))
        payload = json.loads(raw)
        self.assertNotIn("error", payload)
        self.assertEqual(payload["count"], 6)
        sources = {n["source"] for n in payload["news"]}
        self.assertEqual(sources, {"gq", "test"})

    def test_query_feed_skips_local_filter_via_http(self) -> None:
        """{topic}-лента отвечает записями без слова в теме (фильтр — там).

        Тема «квантовый планшет» в фикстуре не встречается: статическая
        лента не даст совпадений, а поисковая вернёт все свои записи —
        локальный фильтр к ней не применялся.
        """
        raw = _unwrap(json.loads(self._call(
            "search_news", {"topic": "квантовый планшет"},
        )))
        payload = json.loads(raw)
        self.assertNotIn("error", payload)
        self.assertEqual(payload["count"], 4)
        self.assertTrue(all(n["source"] == "gq" for n in payload["news"]))

    # -- ТЗ 8: fallback по HTTP --

    def test_summarize_fallback_via_http(self) -> None:
        """Без ключа API summarize_text возвращает сырой текст (fallback)."""
        result = str(self._call(
            "summarize_text", {"text": "тестовый текст новостей"},
        ))
        self.assertIn("DEEPSEEK_API_KEY не задан", result)
        self.assertIn("тестовый текст новостей", result)

    # -- ТЗ 5: сохранение по HTTP --

    def test_save_to_file_via_http(self) -> None:
        """save_to_file по HTTP создаёт файл в заданной папке output."""
        raw = _unwrap(json.loads(self._call(
            "save_to_file", {"content": "# Сводка\n\nПроверка."},
        )))
        result = str(raw)
        self.assertFalse(result.lower().startswith("ошибка"), result)
        file_path = Path(result)
        self.assertTrue(file_path.exists())
        self.assertIn(self.output_dir.name, str(file_path))
        self.assertEqual(
            file_path.read_text(encoding="utf-8"), "# Сводка\n\nПроверка."
        )


# ---------------------------------------------------------------------------
# Agentic loop: цепочка search → summarize → save (ТЗ 6, 9; офлайн)
# ---------------------------------------------------------------------------


class _FakeOpenAIClient:
    """Клиент OpenAI-формата: client.chat.completions.create → фейки."""

    def __init__(self, responses):
        self.completions = _FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)

    @property
    def calls(self):
        return self.completions.calls


class _FakeCompletions:
    """chat.completions.create — заглушка с очередью ответов."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:  # pragma: no cover — защита от зацикливания
            raise AssertionError("Неожиданный дополнительный вызов API")
        return self.responses.pop(0)


def _api_message(content=None, tool_calls=None, reasoning=None):
    return SimpleNamespace(
        content=content, tool_calls=tool_calls, reasoning_content=reasoning,
    )


def _api_response(message, finish_reason="stop", usage=None):
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage, model="test-model")


def _tool_call(call_id, name, arguments: dict):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


class _FakeNewsClient:
    """HTTP MCP-клиент news-сервера: журнал вызовов, готовые ответы."""

    def __init__(self):
        self.calls = []

    async def list_tools(self):
        from src.mcp_models import MCPTool

        return [
            MCPTool(name="search_news", description="Найти новости в RSS",
                    input_schema={"type": "object", "properties": {
                        "topic": {"type": "string"}}}),
            MCPTool(name="summarize_text", description="Сводка через DeepSeek",
                    input_schema={"type": "object", "properties": {
                        "text": {"type": "string"}}}),
            MCPTool(name="save_to_file", description="Сохранить в output/",
                    input_schema={"type": "object", "properties": {
                        "content": {"type": "string"}}}),
        ]

    async def call_tool(self, name, arguments=None):
        self.calls.append((name, dict(arguments or {})))
        if name == "search_news":
            return json.dumps({"topic": arguments["topic"], "count": 1,
                               "news": [{"title": "Технологии 5G",
                                         "link": "x", "summary": "s",
                                         "source": "test"}]}, ensure_ascii=False)
        if name == "summarize_text":
            return "# Сводка\n\nТехнологии развиваются."
        if name == "save_to_file":
            return json.dumps({"result": "C:\\tmp\\news_20260925_120000.md"})
        return "{}"


class _FakeReminderClient:
    """HTTP MCP-клиент напоминаний (проверка маршрутизации двух серверов)."""

    def __init__(self):
        self.calls = []

    async def list_tools(self):
        from src.mcp_models import MCPTool

        return [MCPTool(name="add_reminder", description="Создать напоминание",
                        input_schema={"type": "object", "properties": {
                            "text": {"type": "string"}}})]

    async def call_tool(self, name, arguments=None):
        self.calls.append((name, dict(arguments or {})))
        return json.dumps({"id": 1, "text": arguments.get("text", "")})


class AgenticLoopTest(unittest.TestCase):
    """LLM сама строит цепочку инструментов двух серверов (ТЗ 6)."""

    def _make_agent(self, fake_api, news_client, reminder_client=None):
        from src.agent import DeepSeekAgent
        from src.tokens import HeuristicCounter

        agent = DeepSeekAgent(
            api_key="test-key", mock_mode=False,
            token_counter=HeuristicCounter(),
            reminder_client=reminder_client,
            news_client=news_client,
        )
        agent._client = fake_api
        return agent

    def test_full_pipeline_chain(self) -> None:
        """«Прочитай новости и сохрани сводку» → 3 инструмента по цепочке."""
        saved_path = "C:\\tmp\\news_20260925_120000.md"
        api = _FakeOpenAIClient([
            # Шаг 1: LLM просит поиск
            _api_response(_api_message(
                content=None,
                tool_calls=[_tool_call("c1", "search_news",
                                       {"topic": "технологии"})],
            ), finish_reason="tool_calls"),
            # Шаг 2: получив новости, просит сводку
            _api_response(_api_message(
                content=None,
                tool_calls=[_tool_call("c2", "summarize_text",
                                       {"text": "новости про технологии"})],
            ), finish_reason="tool_calls"),
            # Шаг 3: со сводкой — сохранение в файл
            _api_response(_api_message(
                content=None,
                tool_calls=[_tool_call("c3", "save_to_file",
                                       {"content": "# Сводка"})],
            ), finish_reason="tool_calls"),
            # Финал: ответ пользователю
            _api_response(_api_message(
                content=f"Готово! Сохранил сводку в файл: {saved_path}",
            )),
        ])
        news = _FakeNewsClient()
        agent = self._make_agent(api, news)

        result = agent.send_message(
            "Прочитай свежие новости про технологии и сохрани сводку"
        )

        # Цепочка выполнена в правильном порядке
        self.assertEqual(
            [name for name, _ in news.calls],
            ["search_news", "summarize_text", "save_to_file"],
        )
        self.assertEqual(news.calls[0][1], {"topic": "технологии"})
        # Каждый шаг возвращался LLM (3 запроса с tool-результатами)
        self.assertEqual(len(api.calls), 4)
        # Путь файла попал в результат для кнопки UI
        self.assertEqual(result["file_path"], saved_path)
        self.assertEqual(result["tool_calls"][0]["name"], "search_news")
        self.assertEqual(result["tool_calls"][2]["name"], "save_to_file")
        self.assertEqual(
            result["text"], f"Готово! Сохранил сводку в файл: {saved_path}"
        )

    def test_tools_of_both_servers_in_request(self) -> None:
        """В запросе tools обоих серверов; вызовы маршрутизируются по имени."""
        api = _FakeOpenAIClient([
            _api_response(_api_message(
                content=None,
                tool_calls=[
                    _tool_call("c1", "add_reminder",
                               {"text": "перерыв", "interval_minutes": 30}),
                    _tool_call("c2", "search_news",
                               {"topic": "технологии"}),
                ],
            ), finish_reason="tool_calls"),
            _api_response(_api_message(content="Сделал оба дела.")),
        ])
        news = _FakeNewsClient()
        reminder = _FakeReminderClient()
        agent = self._make_agent(api, news, reminder_client=reminder)

        result = agent.send_message("напомни и найди новости")
        self.assertIsNone(result["error"])
        # Инструменты обоих серверов в одном запросе
        names = [t["function"]["name"]
                 for t in api.calls[0]["tools"]]
        self.assertIn("add_reminder", names)
        self.assertIn("search_news", names)
        self.assertEqual(len(names), 4)
        # Каждый вызов ушёл СВОЕМУ серверу
        self.assertEqual(reminder.calls, [("add_reminder", {
            "text": "перерыв", "interval_minutes": 30})])
        self.assertEqual(news.calls, [("search_news", {"topic": "технологии"})])
        self.assertIsNone(result.get("file_path"))

    def test_iteration_limit_stops_loop(self) -> None:
        """Зацикливание LLM обрывается лимитом раундов (ТЗ 9)."""
        looping = _api_response(_api_message(
            content=None,
            tool_calls=[_tool_call("c", "search_news", {"topic": "x"})],
        ), finish_reason="tool_calls")
        api = _FakeOpenAIClient([looping for _ in range(10)])
        agent = self._make_agent(api, _FakeNewsClient())
        agent._max_tool_rounds = 3

        result = agent.send_message("найди новости")
        self.assertIsNone(result["error"])
        self.assertLessEqual(len(api.calls), 4)

    def test_unknown_tool_returns_error_to_llm(self) -> None:
        """Вызов несуществующего инструмента не роняет ход (error → LLM)."""
        api = _FakeOpenAIClient([
            _api_response(_api_message(
                content=None,
                tool_calls=[_tool_call("c1", "no_such_tool", {})],
            ), finish_reason="tool_calls"),
            _api_response(_api_message(content="Обошёлся без инструмента.")),
        ])
        agent = self._make_agent(api, _FakeNewsClient())
        result = agent.send_message("что-нибудь")
        self.assertEqual(result["text"], "Обошёлся без инструмента.")
        tool_messages = [
            m for m in api.calls[1]["messages"]
            if isinstance(m, dict) and m.get("role") == "tool"
        ]
        self.assertIn("не найден", tool_messages[0]["content"])


# ---------------------------------------------------------------------------
# Конфигурация NEWS_*
# ---------------------------------------------------------------------------


class NewsSettingsTest(unittest.TestCase):
    """get_news_settings: дефолты, безопасность хоста, выключатель."""

    def test_defaults(self) -> None:
        settings = get_news_settings({})
        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["port"], 8001)
        self.assertEqual(settings["url"], "http://127.0.0.1:8001/mcp")
        self.assertEqual(settings["summary_model"], "")
        self.assertEqual(settings["output_dir"], "")

    def test_unsafe_host_rejected(self) -> None:
        """Внешний хост → 127.0.0.1 (хостить в интернете нельзя)."""
        self.assertEqual(
            get_news_settings({"NEWS_SERVER_HOST": "0.0.0.0"})["host"],
            "127.0.0.1",
        )

    def test_bad_port_falls_back(self) -> None:
        self.assertEqual(
            get_news_settings({"NEWS_SERVER_PORT": "abc"})["port"], 8001
        )

    def test_disabled_flag(self) -> None:
        self.assertFalse(
            get_news_settings({"NEWS_ENABLED": "false"})["enabled"]
        )


# ---------------------------------------------------------------------------
# UI: кнопка «Открыть файл» (ТЗ 7)
# ---------------------------------------------------------------------------


class NewsUITest(unittest.TestCase):
    """ChatApplication: блок сохранённого файла и кнопка открытия."""

    root = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = gui_helpers.get_root()
        if cls.root is None:
            raise unittest.SkipTest("графическая среда недоступна (tkinter/Tk)")

    def _make_app(self):
        from src.agent import DeepSeekAgent
        from src.tokens import HeuristicCounter
        from src.ui import ChatApplication

        gui_helpers.reset_children(self.root)
        agent = DeepSeekAgent(
            api_key="", mock_mode=True, token_counter=HeuristicCounter()
        )
        config = dict(DEFAULTS)
        config.update({
            "APP_STATE_PATH": str(Path(tempfile.mkdtemp()) / "state.db"),
        })
        app = ChatApplication(root=self.root, agent=agent, config=config)
        self.addCleanup(self.root.update)
        return app

    def _make_saved_file(self) -> Path:
        tmp = Path(tempfile.mkdtemp(prefix="news_ui_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(
            tmp, ignore_errors=True))
        file_path = tmp / "news_20260925_120000.md"
        file_path.write_text("# Сводка\n\nТест.", encoding="utf-8")
        return file_path

    def test_file_path_shows_button(self) -> None:
        """Результат с file_path → блок с путём и кнопка в чате."""
        app = self._make_app()
        file_path = self._make_saved_file()
        app._on_result({
            "text": "Готово! Сохранил сводку.",
            "model": "test", "tokens": None, "response_time": 0.1,
            "finish_reason": "stop", "error": None,
            "file_path": str(file_path),
        })
        chat_text = app.chat.get("1.0", "end-1c")
        self.assertIn("Файл сохранён", chat_text)
        self.assertIn(str(file_path), chat_text)
        self.assertIn(str(file_path), app._saved_files)

    def test_button_survives_chat_refresh(self) -> None:
        """Кнопка переживает перерисовку чата (источник — _saved_files)."""
        app = self._make_app()
        file_path = self._make_saved_file()
        app._on_result({
            "text": "Готово.", "model": "test", "tokens": None,
            "response_time": 0.1, "finish_reason": "stop", "error": None,
            "file_path": str(file_path),
        })
        app._refresh_chat()
        self.assertIn("Файл сохранён", app.chat.get("1.0", "end-1c"))

    def test_open_file_invokes_startfile(self) -> None:
        """_open_file вызывает os.startfile (Windows) для существующего файла."""
        app = self._make_app()
        file_path = self._make_saved_file()
        with mock.patch("os.startfile") as startfile:
            app._open_file(str(file_path))
        startfile.assert_called_once_with(str(file_path))

    def test_open_missing_file_shows_error(self) -> None:
        """Несуществующий файл — ошибка, startfile не вызывается."""
        app = self._make_app()
        with mock.patch("os.startfile") as startfile, \
                mock.patch("src.ui.messagebox.showerror") as showerror:
            app._open_file("Z:/нет/такого/файла.md")
        startfile.assert_not_called()
        showerror.assert_called_once()

    def test_saved_files_capped(self) -> None:
        """Хвост сохранённых файлов ограничен (чат не разбухает)."""
        app = self._make_app()
        for index in range(8):
            app._remember_saved_file(f"C:\\tmp\\news_{index}.md")
        self.assertEqual(len(app._saved_files), 5)
        self.assertEqual(app._saved_files[-1], "C:\\tmp\\news_7.md")

    def test_duplicate_path_not_added(self) -> None:
        """Повторное сохранение того же пути не дублирует блок."""
        app = self._make_app()
        self.assertTrue(app._remember_saved_file("C:\\tmp\\news.md"))
        self.assertFalse(app._remember_saved_file("C:\\tmp\\news.md"))
        self.assertEqual(len(app._saved_files), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
