"""MCP-сервер «CRM» (ТЗ ai_1_34): пользователи и тикеты поддержки.

Транспорт: Streamable HTTP (stateful), только 127.0.0.1:8004 — сервер
НЕ хостится в интернете. Запускается приложением в отдельном окне
консоли (CrmServerLauncher, см. src/crm_launcher.py) и живёт 24/7.

Данные — JSON-файл рядом со скриптом (crm_data.json): пользователи с
полями id/name/email/status/created_at и списком тикетов (id/subject/
status/created_at/priority). Файл читается на каждый вызов инструмента:
правки CRM подхватываются без перезапуска сервера.

Инструменты команды /support (SupportPipeline, app/support_pipeline.py):

* ``get_user_context(user_id)`` — агрегированный контекст: профиль +
  тикеты (основной инструмент пайплайна);
* ``get_user(user_id)`` / ``get_user_tickets(user_id)`` — по отдельности;
* ``list_users()`` — список пользователей для выпадающего списка в UI
  (user_id НЕ хардкодится — выбирается в чате).

Конфигурация (env или CLI, см. main()): CRM_HOST/CRM_PORT — адрес
привязки (по умолчанию 127.0.0.1:8004), CRM_DATA — путь к JSON.

Запуск вручную::

    python crm_server.py                     # порт 8004
    python crm_server.py --port 8014         # свой порт (используют тесты)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

from mcp.server.fastmcp import FastMCP

# ─── Инициализация MCP-сервера ──────────────────────────────────────────────

mcp = FastMCP(
    "crm-server",
    host=os.environ.get("CRM_HOST", "127.0.0.1"),
    port=int(os.environ.get("CRM_PORT", "8004")),
)

# JSON с пользователями и тикетами (по умолчанию — рядом со скриптом)
CRM_DATA_PATH = Path(os.environ.get(
    "CRM_DATA", str(Path(__file__).parent / "crm_data.json")))


def _load_crm_data() -> dict:
    """Загружает JSON с пользователями и тикетами.

    Отсутствующий или битый файл — пустой CRM (инструменты вернут
    «не найдено»), сервер не падает.
    """
    try:
        return json.loads(CRM_DATA_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"users": []}
    except (OSError, ValueError) as exc:
        logging.getLogger("crm.tool").warning(
            "CRM-данные не читаются (%s): %s", CRM_DATA_PATH, exc)
        return {"users": []}


def _find_user(data: dict, user_id: int) -> Dict | None:
    """Пользователь по ID (ID сравниваются как int: JSON может отдать str)."""
    try:
        wanted = int(user_id)
    except (TypeError, ValueError):
        return None
    for user in data.get("users", []):
        try:
            if int(user.get("id")) == wanted:
                return user
        except (TypeError, ValueError):
            continue
    return None


# ─── Инструменты MCP ─────────────────────────────────────────────────────────


@mcp.tool()
def list_users() -> List[Dict]:
    """Список пользователей CRM (для выпадающего списка в UI).

    Returns:
        Список словарей с полями: id, name, email, status
        (без тикетов — для выбора пользователя команды /support).
    """
    return [
        {
            "id": user.get("id"),
            "name": user.get("name", ""),
            "email": user.get("email", ""),
            "status": user.get("status", ""),
        }
        for user in _load_crm_data().get("users", [])
    ]


@mcp.tool()
def get_user(user_id: int) -> Dict:
    """Получить профиль пользователя по ID.

    Args:
        user_id: ID пользователя в CRM.

    Returns:
        Словарь с полями: id, name, email, status, created_at.
    """
    user = _find_user(_load_crm_data(), user_id)
    if user is None:
        return {"error": f"Пользователь {user_id} не найден"}
    return {
        "id": user.get("id"),
        "name": user.get("name", ""),
        "email": user.get("email", ""),
        "status": user.get("status", ""),
        "created_at": user.get("created_at", ""),
    }


@mcp.tool()
def get_user_tickets(user_id: int) -> List[Dict]:
    """Получить список тикетов пользователя.

    Args:
        user_id: ID пользователя в CRM.

    Returns:
        Список тикетов с полями: id, subject, status, created_at,
        priority. Неизвестный пользователь — пустой список.
    """
    user = _find_user(_load_crm_data(), user_id)
    if user is None:
        return []
    return list(user.get("tickets", []))


@mcp.tool()
def get_user_context(user_id: int) -> Dict:
    """Получить агрегированный контекст пользователя: профиль + тикеты.

    Args:
        user_id: ID пользователя в CRM.

    Returns:
        Словарь с полями: user (id/name/email/status), tickets.
    """
    user = _find_user(_load_crm_data(), user_id)
    if user is None:
        return {"error": f"Пользователь {user_id} не найден"}
    return {
        "user": {
            "id": user.get("id"),
            "name": user.get("name", ""),
            "email": user.get("email", ""),
            "status": user.get("status", ""),
        },
        "tickets": list(user.get("tickets", [])),
    }


# ─── Точка входа ─────────────────────────────────────────────────────────────


def main() -> None:
    """Разбирает CLI, применяет переопределения и стартует HTTP-сервер."""
    global CRM_DATA_PATH

    parser = argparse.ArgumentParser(
        description="MCP-сервер CRM (Streamable HTTP)"
    )
    parser.add_argument(
        "--host", default=os.environ.get("CRM_HOST", "127.0.0.1"),
        help="адрес привязки (по умолчанию 127.0.0.1 — только локально)",
    )
    parser.add_argument(
        "--port", type=int,
        default=int(os.environ.get("CRM_PORT", "8004")),
        help="порт HTTP-сервера (по умолчанию 8004)",
    )
    parser.add_argument(
        "--data", default=os.environ.get("CRM_DATA", ""),
        help="путь к crm_data.json (по умолчанию — рядом со скриптом)",
    )
    args = parser.parse_args()

    # Синхронизируем настройки FastMCP и путь данных с CLI
    mcp.settings.host = args.host
    mcp.settings.port = int(args.port)
    if args.data:
        CRM_DATA_PATH = Path(args.data)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    url = f"http://{args.host}:{args.port}/mcp"
    print(f"CRM MCP-сервер запущен на {url}")
    print(f"Данные: {CRM_DATA_PATH}")
    print("Окно можно закрыть вместе с завершением сервера (Ctrl+C).")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
