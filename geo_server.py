"""MCP-сервер «Geo»: определение города пользователя по IP.

Транспорт: Streamable HTTP (stateful), только 127.0.0.1:8002 — сервер
НЕ хостится в интернете. Запускается приложением в отдельном окне
консоли (GeoServerLauncher, см. src/geo_launcher.py) и живёт 24/7.

Инструмент первого звена цепочки «гео → погода» (её строит LLM через
agentic loop агента):

* ``get_user_city`` — город по IP-адресу через бесплатный ip-api.com
  (без API-ключа, лимит 45 запросов/мин). Возвращает название города
  (например, "Berlin") или "Unknown" при любой неудаче — LLM продолжит
  цепочку и объяснит пользователю, что город не определён.

Конфигурация (env или CLI, см. main()): GEO_HOST/GEO_PORT — адрес
привязки (по умолчанию 127.0.0.1:8002).

Запуск вручную::

    python geo_server.py                     # порт 8002
    python geo_server.py --port 8012         # свой порт (используют тесты)
"""

from __future__ import annotations

import argparse
import logging
import os

import httpx
from mcp.server.fastmcp import FastMCP

# ─── Инициализация MCP-сервера ──────────────────────────────────────────────

mcp = FastMCP(
    "geo-server",
    host=os.environ.get("GEO_HOST", "127.0.0.1"),
    port=int(os.environ.get("GEO_PORT", "8002")),
)

# Бесплатный сервис геолокации по IP: без ключа, status="success" при
# удаче; поле city — название города латиницей.
GEO_API_URL = "http://ip-api.com/json/"

# Таймаут запроса к geo-API, сек (сервер отвечает обычно быстрее 1 с)
_GEO_TIMEOUT = 5.0


def _fetch_geo_payload() -> dict:
    """Запрашивает данные геолокации по IP (изолировано для тестов).

    Raises:
        Exception: сеть недоступна, HTTP-ошибка, невалидный JSON —
            вызывающий код возвращает "Unknown", не роняя инструмент.
    """
    response = httpx.get(GEO_API_URL, timeout=_GEO_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, dict) else {}


# ─── Инструменты MCP ─────────────────────────────────────────────────────────


@mcp.tool()
def get_user_city() -> str:
    """Определить город пользователя по IP-адресу.

    Использует бесплатный ip-api.com (без API-ключа).

    Returns:
        Название города (например, "Berlin") или "Unknown", если город
        определить не удалось (нет сети, сервис недоступен, IP не
        определён).
    """
    try:
        data = _fetch_geo_payload()
    except Exception as exc:
        logging.getLogger("geo.tool").warning("Geo API недоступен: %s", exc)
        return "Unknown"
    if data.get("status") == "success":
        city = str(data.get("city") or "").strip()
        if city:
            return city
    return "Unknown"


# ─── Точка входа ─────────────────────────────────────────────────────────────


def main() -> None:
    """Разбирает CLI, применяет переопределения и стартует HTTP-сервер."""
    parser = argparse.ArgumentParser(
        description="MCP-сервер Geo (Streamable HTTP)"
    )
    parser.add_argument(
        "--host", default=os.environ.get("GEO_HOST", "127.0.0.1"),
        help="адрес привязки (по умолчанию 127.0.0.1 — только локально)",
    )
    parser.add_argument(
        "--port", type=int,
        default=int(os.environ.get("GEO_PORT", "8002")),
        help="порт HTTP-сервера (по умолчанию 8002)",
    )
    args = parser.parse_args()

    # Синхронизируем настройки FastMCP с CLI
    mcp.settings.host = args.host
    mcp.settings.port = int(args.port)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    url = f"http://{args.host}:{args.port}/mcp"
    print(f"Geo MCP-сервер запущен на {url}")
    print(f"Источник геолокации: {GEO_API_URL}")
    print("Окно можно закрыть вместе с завершением сервера (Ctrl+C).")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
