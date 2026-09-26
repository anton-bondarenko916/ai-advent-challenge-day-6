"""MCP-сервер «Weather»: текущая погода по названию города.

Транспорт: Streamable HTTP (stateful), только 127.0.0.1:8003 — сервер
НЕ хостится в интернете. Запускается приложением в отдельном окне
консоли (WeatherServerLauncher, см. src/weather_launcher.py) и живёт
24/7.

Инструмент второго звена цепочки «гео → погода» (её строит LLM через
agentic loop агента: get_user_city → get_weather):

* ``get_weather`` — погода по городу через бесплатный Open-Meteo
  (без API-ключа), два шага: геокодинг (geocoding-api.open-meteo.com)
  → прогноз (api.open-meteo.com). Возвращает JSON-строку с полями
  temperature, wind_speed, humidity либо {"error": "..."}.

Конфигурация (env или CLI, см. main()): WEATHER_HOST/WEATHER_PORT —
адрес привязки (по умолчанию 127.0.0.1:8003).

Запуск вручную::

    python weather_server.py                     # порт 8003
    python weather_server.py --port 8013         # свой порт (тесты)
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import httpx
from mcp.server.fastmcp import FastMCP

# ─── Инициализация MCP-сервера ──────────────────────────────────────────────

mcp = FastMCP(
    "weather-server",
    host=os.environ.get("WEATHER_HOST", "127.0.0.1"),
    port=int(os.environ.get("WEATHER_PORT", "8003")),
)

# Бесплатные сервисы Open-Meteo: геокодинг «название → координаты» и
# прогноз «координаты → погода». Без API-ключа.
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Таймауты запросов, сек
_GEOCODING_TIMEOUT = 10.0
_FORECAST_TIMEOUT = 10.0


def _geocode_city(city: str) -> dict:
    """Ищет координаты города (изолировано для тестов).

    Args:
        city: название города.

    Returns:
        Первая найденная запись геокодинга (latitude, longitude, name...).

    Raises:
        Exception: сеть недоступна, HTTP-ошибка, невалидный JSON.
        LookupError: город не найден (пустой список results).
    """
    params = {"name": city, "count": 1, "language": "en", "format": "json"}
    response = httpx.get(
        GEOCODING_URL, params=params, timeout=_GEOCODING_TIMEOUT
    )
    response.raise_for_status()
    data = response.json()
    results = data.get("results") if isinstance(data, dict) else None
    if not results:
        raise LookupError(f"город не найден: {city!r}")
    return results[0]


def _fetch_current_weather(latitude: float, longitude: float) -> dict:
    """Запрашивает текущую погоду по координатам (изолировано для тестов).

    Args:
        latitude: широта.
        longitude: долгота.

    Returns:
        Словарь ``current`` из ответа Open-Meteo (temperature_2m,
        wind_speed_10m, relative_humidity_2m).

    Raises:
        Exception: сеть недоступна, HTTP-ошибка, невалидный JSON.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,wind_speed_10m,relative_humidity_2m",
        "timezone": "auto",
    }
    response = httpx.get(
        FORECAST_URL, params=params, timeout=_FORECAST_TIMEOUT
    )
    response.raise_for_status()
    data = response.json()
    current = data.get("current") if isinstance(data, dict) else None
    return current if isinstance(current, dict) else {}


# ─── Инструменты MCP ─────────────────────────────────────────────────────────


def _normalize_city(raw: str) -> str:
    """Достаёт имя города из «сырого» ответа инструмента geo-сервера.

    LLM обычно передаёт чистое имя, но может переслать и ПОЛНЫЙ текст
    результата ``get_user_city``: клиент MCP показывает ответ FastMCP
    как pretty-JSON ``{"result": "Berlin"}`` (многострочный). Чтобы
    цепочка «гео → погода» не ломалась на таком переносе, JSON
    разворачивается (поля result/city/name), из многострочного текста
    берётся первый кандидат-строка.

    Args:
        raw: значение аргумента city из tool_calls.

    Returns:
        Имя города без обрамляющих кавычек/скобок; "" — ничего нет.
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if isinstance(data, dict):
            # Валидный JSON без узнаваемых полей — города в нём нет
            for key in ("result", "city", "name"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return ""
    if "\n" not in text:
        return text.strip().strip('"').strip()
    for line in text.splitlines():
        cleaned = line.strip().strip("{},\"").strip()
        if not cleaned or cleaned.startswith(("}", "{")):
            continue
        if ":" in cleaned:  # строка "result": "Berlin" → Berlin
            cleaned = cleaned.rsplit(":", 1)[-1]
        cleaned = cleaned.strip().strip('"').strip()
        if cleaned:
            return cleaned
    return ""


@mcp.tool()
def get_weather(city: str) -> str:
    """Получить текущую погоду в указанном городе.

    Использует бесплатный Open-Meteo API (без API-ключа): сначала
    геокодинг «название → координаты», затем запрос погоды.

    Args:
        city: название города (например, "Berlin" или результат
            инструмента get_user_city geo-сервера).

    Returns:
        JSON-строка с полями temperature (°C), wind_speed (км/ч),
        humidity (%). Если город не найден или сервис недоступен —
        {"error": "..."} (LLM объяснит проблему пользователю).
    """
    name = _normalize_city(city)
    if not name:
        return json.dumps(
            {"error": "Укажите город (city)"}, ensure_ascii=False
        )
    try:
        location = _geocode_city(name)
        current = _fetch_current_weather(
            location["latitude"], location["longitude"]
        )
    except LookupError:
        return json.dumps(
            {"error": f"Город '{name}' не найден"}, ensure_ascii=False
        )
    except Exception as exc:
        logging.getLogger("weather.tool").warning(
            "Open-Meteo недоступен: %s", exc
        )
        return json.dumps(
            {"error": f"Сервис погоды недоступен: {exc}"},
            ensure_ascii=False,
        )
    return json.dumps({
        "city": location.get("name", name),
        "temperature": current.get("temperature_2m"),
        "wind_speed": current.get("wind_speed_10m"),
        "humidity": current.get("relative_humidity_2m"),
    }, ensure_ascii=False)


# ─── Точка входа ─────────────────────────────────────────────────────────────


def main() -> None:
    """Разбирает CLI, применяет переопределения и стартует HTTP-сервер."""
    parser = argparse.ArgumentParser(
        description="MCP-сервер Weather (Streamable HTTP)"
    )
    parser.add_argument(
        "--host", default=os.environ.get("WEATHER_HOST", "127.0.0.1"),
        help="адрес привязки (по умолчанию 127.0.0.1 — только локально)",
    )
    parser.add_argument(
        "--port", type=int,
        default=int(os.environ.get("WEATHER_PORT", "8003")),
        help="порт HTTP-сервера (по умолчанию 8003)",
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
    print(f"Weather MCP-сервер запущен на {url}")
    print(f"Геокодинг: {GEOCODING_URL}")
    print(f"Прогноз: {FORECAST_URL}")
    print("Окно можно закрыть вместе с завершением сервера (Ctrl+C).")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
