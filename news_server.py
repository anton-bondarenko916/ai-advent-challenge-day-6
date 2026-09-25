"""MCP-сервер «News»: поиск новостей, сводка, сохранение в файл.

Транспорт: Streamable HTTP (stateful), только 127.0.0.1:8001 — сервер
НЕ хостится в интернете. Запускается приложением в отдельном окне
консоли (NewsServerLauncher, см. src/news_launcher.py) и живёт 24/7.

Инструменты (цепочку строит LLM через agentic loop агента):

* ``search_news``    — свежие новости по теме из RSS-лент (feedparser);
* ``summarize_text`` — краткая сводка текста через DeepSeek API
  (собственный вызов API внутри сервера, в историю чата не попадает;
  при ошибке — fallback: возврат сырого текста);
* ``save_to_file``   — сохранение Markdown в ``output/news_<штамп>.md``,
  возвращает абсолютный путь (UI показывает кнопку «Открыть файл»).

Конфигурация (env или CLI, см. main()): NEWS_RSS_FEEDS — список лент
``имя=url;имя=url`` (по умолчанию Google News с поиском по теме через
шаблон ``{topic}`` + статические cnews/3dnews/habr/kommersant/lenta),
NEWS_OUTPUT_DIR — папка результатов (по умолчанию output/ в корне
проекта), DEEPSEEK_API_KEY — ключ для summarize_text (читается из .env
проекта, если не задан в окружении).

Запуск вручную::

    python news_server.py                     # порт 8001, output/ проекта
    python news_server.py --port 8011         # свой порт (используют тесты)
    python news_server.py --feeds "test:file:///C:/tmp/feed.xml"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import feedparser
from mcp.server.fastmcp import FastMCP

# ─── Инициализация MCP-сервера ──────────────────────────────────────────────

mcp = FastMCP(
    "news-server",
    host=os.environ.get("NEWS_HOST", "127.0.0.1"),
    port=int(os.environ.get("NEWS_PORT", "8001")),
)

# Корень проекта (сервер лежит рядом с main.py)
PROJECT_ROOT = Path(__file__).resolve().parent

# RSS-ленты по умолчанию (ТЗ: lenta.ru + rssexport.rbc.ru; третья —
# запасные на случай, когда поисковая лента не отвечает — ошибки лент
# пропускаются, не роняя инструмент).
#
# Шаблон ``{topic}``: если URL содержит его, тема подставляется в запрос
# (URL-кодирование автоматическое) и поиск выполняется НА стороне
# провайдера — локальный текстовый фильтр для такой ленты не применяется
# (все её записи уже релевантны теме). Так подключается любой поисковый
# RSS (Google News и т.п.) одной строкой NEWS_RSS_FEEDS.
DEFAULT_RSS_FEEDS: Dict[str, str] = {
    "google": (
        "https://news.google.com/rss/search?q={topic}"
        "&hl=ru&gl=RU&ceid=RU:ru"
    ),
    "cnews": "https://www.cnews.ru/inc/rss/news.xml",
    "3dnews": "https://3dnews.ru/news/rss/",
    "habr": "https://habr.com/ru/rss/news/?fl=ru",
    "kommersant": "https://www.kommersant.ru/RSS/news.xml",
    "lenta": "https://lenta.ru/rss/news",
}


def _feeds_from_env() -> Dict[str, str]:
    """Разбирает NEWS_RSS_FEEDS вида ``имя=url;имя=url`` (или дефолт)."""
    raw = str(os.environ.get("NEWS_RSS_FEEDS") or "").strip()
    if not raw:
        return dict(DEFAULT_RSS_FEEDS)
    feeds: Dict[str, str] = {}
    for part in raw.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, _, url = part.partition("=")
        name, url = name.strip(), url.strip()
        if name and url:
            feeds[name] = url
    return feeds or dict(DEFAULT_RSS_FEEDS)


# Ленты читаются из окружения при импорте (переопределяются CLI/тестами)
RSS_FEEDS: Dict[str, str] = _feeds_from_env()

# Папка результатов (save_to_file); переопределяется NEWS_OUTPUT_DIR
OUTPUT_DIR = Path(
    os.environ.get("NEWS_OUTPUT_DIR") or (PROJECT_ROOT / "output")
)

# Модель сводки: по умолчанию — основная модель проекта
DEFAULT_SUMMARY_MODEL = "deepseek-v4-flash"

# Лимит символов описания новости в выдаче (экономия токенов LLM)
_SUMMARY_CHARS = 500

# Таймаут загрузки одной RSS-ленты, сек (недоступная лента пропускается)
_FEED_TIMEOUT = 10.0


def load_project_env() -> None:
    """Дочитывает .env проекта (для запуска сервера вне приложения).

    DEEPSEEK_API_KEY обычно попадает в окружение от родительского
    приложения (launcher наследует os.environ); здесь — запасной путь
    для ручного запуска ``python news_server.py``. Уже установленные
    переменные не перезаписываются.
    """
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:  # python-dotenv — зависимость проекта, но не критичная здесь
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=env_path, override=False)
    except ImportError:
        pass  # ключ можно задать и переменной окружения


# ─── Инструменты MCP ─────────────────────────────────────────────────────────


def _fetch_feed(url: str) -> bytes:
    """Скачивает RSS-ленту с таймаутом (file:// тоже поддержан).

    Raises:
        Exception: сеть недоступна, HTTP-ошибка — вызывающий код
            пропускает ленту и идёт к следующей.
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (news-mcp-server)"}
    )
    with urllib.request.urlopen(request, timeout=_FEED_TIMEOUT) as response:
        return response.read()


@mcp.tool()
def search_news(topic: str, limit: int = 10) -> str:
    """Найти свежие новости по теме в RSS-лентах.

    Поисковые ленты (URL с шаблоном ``{topic}``, например Google News)
    фильтруют тему НА своей стороне — их записи принимаются как есть.
    Обычные ленты фильтруются локально: вхождение темы в заголовок или
    описание без учёта регистра. Ленты опрашиваются по очереди;
    недоступная лента пропускается (не прерывает поиск).

    Args:
        topic: Тема для поиска (например, "технологии", "политика").
        limit: Максимальное количество новостей (по умолчанию 10).

    Returns:
        JSON-строка: {"topic", "count", "news": [{title, link, summary,
        source}]}. Если новостей по теме нет — {"error": "..."}.
    """
    topic_raw = str(topic or "").strip()
    topic_lower = topic_raw.lower()
    if not topic_lower:
        return json.dumps(
            {"error": "Укажите тему поиска (topic)"}, ensure_ascii=False
        )
    try:
        limit = max(1, min(int(limit), 50))
    except (TypeError, ValueError):
        limit = 10

    all_news: List[Dict[str, str]] = []
    errors: List[str] = []
    for source_name, feed_url in RSS_FEEDS.items():
        # Поисковая лента: подставляем тему в URL (кодирование —
        # quote_plus: кириллица и пробелы становятся %XX/+)
        is_query_feed = "{topic}" in feed_url
        url = (
            feed_url.replace("{topic}", urllib.parse.quote_plus(topic_raw))
            if is_query_feed else feed_url
        )
        try:
            feed = feedparser.parse(_fetch_feed(url))
        except Exception as exc:
            # Пропускаем недоступную ленту, идём к следующей (по ТЗ)
            errors.append(f"{source_name}: {exc}")
            continue
        for entry in feed.entries:
            title = str(getattr(entry, "title", "") or "")
            summary = str(getattr(entry, "summary", "") or "")
            # У поисковой ленты записи уже релевантны теме — локальный
            # буквальный фильтр не применяется
            if (not is_query_feed
                    and topic_lower not in title.lower()
                    and topic_lower not in summary.lower()):
                continue
            all_news.append({
                "title": title,
                "link": str(getattr(entry, "link", "") or ""),
                "summary": summary[:_SUMMARY_CHARS],
                "source": source_name,
            })
            if len(all_news) >= limit:
                break
        if len(all_news) >= limit:
            break

    if not all_news:
        message = f"Новости по теме '{topic}' не найдены"
        if errors:
            message += f" (ленты недоступны: {'; '.join(errors)})"
        return json.dumps({"error": message}, ensure_ascii=False)

    return json.dumps({
        "topic": topic,
        "count": len(all_news),
        "news": all_news,
    }, ensure_ascii=False, indent=2)


def _summarize_via_api(text: str, api_key: str, model: str) -> str:
    """Служебный вызов DeepSeek для сводки (изолирован для тестов).

    Это ОТДЕЛЬНЫЙ вызов API, не связанный с основным агентом: результат
    возвращается инструментом MCP и в историю чата не попадает.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompt = (
        "Сделай краткую сводку следующих новостей в формате Markdown.\n"
        "Для каждой новости укажи заголовок и 1-2 предложения сути.\n"
        "В конце добавь общий вывод (2-3 предложения).\n\n"
        f"Новости:\n{text}\n"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1500,
    )
    content = response.choices[0].message.content
    return str(content or "")


@mcp.tool()
def summarize_text(text: str) -> str:
    """Сделать краткую сводку текста через DeepSeek API.

    Args:
        text: Текст для суммаризации (обычно JSON со списком новостей
            из search_news).

    Returns:
        Краткая сводка в Markdown. При ошибке API — fallback: сообщение
        об ошибке и сырой текст (до 2000 символов), чтобы LLM могла
        продолжить цепочку без сводки.
    """
    body = str(text or "").strip()
    if not body:
        return "Ошибка: пустой текст для суммаризации"

    api_key = (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
    if not api_key:
        return (
            "Ошибка: DEEPSEEK_API_KEY не задан. "
            f"Сырые данные:\n{body[:2000]}"
        )
    model = (
        str(os.environ.get("NEWS_SUMMARY_MODEL") or "").strip()
        or DEFAULT_SUMMARY_MODEL
    )
    try:
        return _summarize_via_api(body, api_key, model)
    except Exception as exc:
        # Fallback по ТЗ: при ошибке API возвращаем сырой текст
        return (
            f"Не удалось сделать сводку: {exc}\n\n"
            f"Сырые данные:\n{body[:2000]}"
        )


@mcp.tool()
def save_to_file(content: str) -> str:
    """Сохранить контент в Markdown-файл в папке output/.

    Args:
        content: Текст для сохранения (Markdown).

    Returns:
        Абсолютный путь к сохранённому файлу (UI по нему показывает
        кнопку «Открыть файл»). При ошибке — строка «Ошибка ...».
    """
    body = str(content or "")
    if not body.strip():
        return "Ошибка сохранения: пустой контент"

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = OUTPUT_DIR / f"news_{timestamp}.md"
        filepath.write_text(body, encoding="utf-8")
        return str(filepath.absolute())
    except OSError as exc:
        return f"Ошибка сохранения: {exc}"


# ─── Точка входа ─────────────────────────────────────────────────────────────


def main() -> None:
    """Разбирает CLI, применяет переопределения и стартует HTTP-сервер."""
    global OUTPUT_DIR, RSS_FEEDS
    parser = argparse.ArgumentParser(
        description="MCP-сервер News (Streamable HTTP)"
    )
    parser.add_argument(
        "--host", default=os.environ.get("NEWS_HOST", "127.0.0.1"),
        help="адрес привязки (по умолчанию 127.0.0.1 — только локально)",
    )
    parser.add_argument(
        "--port", type=int,
        default=int(os.environ.get("NEWS_PORT", "8001")),
        help="порт HTTP-сервера (по умолчанию 8001)",
    )
    parser.add_argument(
        "--output-dir", default=os.environ.get("NEWS_OUTPUT_DIR"),
        help="папка результатов save_to_file (по умолчанию output/ проекта)",
    )
    parser.add_argument(
        "--feeds", default=os.environ.get("NEWS_RSS_FEEDS"),
        help="RSS-ленты в формате 'имя=url;имя=url' (file:// допустим)",
    )
    args = parser.parse_args()

    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    if args.feeds:
        os.environ["NEWS_RSS_FEEDS"] = args.feeds
        RSS_FEEDS = _feeds_from_env()

    # Синхронизируем настройки FastMCP с CLI
    mcp.settings.host = args.host
    mcp.settings.port = int(args.port)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    load_project_env()

    url = f"http://{args.host}:{args.port}/mcp"
    print(f"News MCP-сервер запущен на {url}")
    print(f"Папка для сохранения: {Path(OUTPUT_DIR).absolute()}")
    print(f"RSS-ленты: {', '.join(f'{n}={u}' for n, u in RSS_FEEDS.items())}")
    print("Окно можно закрыть вместе с завершением сервера (Ctrl+C).")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
