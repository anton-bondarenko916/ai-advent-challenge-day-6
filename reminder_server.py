"""MCP-сервер «Smart Reminder»: напоминания на HTTP-транспорте.

Транспорт: Streamable HTTP (stateful-сессии), только 127.0.0.1 — сервер
НЕ хостится в интернете. Приложение-клиент (src/reminder_client.py)
подключается по ``http://127.0.0.1:<порт>/mcp``, выполняет tools/list и
tools/call и держит постоянную сессию для приёма SSE-уведомлений.

Жизненный цикл: сервер живёт 24/7 в собственном окне консоли (его
запускает ReminderServerLauncher при старте приложения, см.
src/reminder_launcher.py). Повторный запуск приложения сервер не
перезапускает — порт уже занят, и лаунчер молча пропускает старт.

Инструменты (регистрируются декоратором ``@mcp.tool()``, типы параметров
превращаются в inputSchema, docstring — в описание для LLM):

* ``add_reminder``       — создать напоминание (текст, интервал, повтор);
* ``get_due_reminders``  — сработавшие (транзакционно сдвигает next_due_at);
* ``get_all_reminders``  — все активные;
* ``delete_reminder``    — удалить по ID;
* ``get_stats``          — агрегаты (total/active/triggered_today/avg_interval).

Планировщик: APScheduler (BackgroundScheduler) каждую минуту проверяет
due-напоминания и рассылает SSE-уведомления ``notifications/message``
всем зарегистрированным сессиям-подписчикам.

Как работает рассылка: MCP-сессия живёт внутри event loop HTTP-запроса,
а планировщик работает в своём потоке. Каждый вызов инструмента
регистрирует «свой» ServerSession + event loop в реестре
(:data:`_subscribers`); поток планировщика доставляет уведомления через
``asyncio.run_coroutine_threadsafe(session.send_log_message(...), loop)``.
Клиент, который только слушает (не вызывает инструменты), регистрирует
сессию служебным вызовом ``get_stats`` сразу после handshake.

Запуск вручную::

    python reminder_server.py                  # порт 8000, БД reminders.db
    python reminder_server.py --port 8010      # свой порт (используют тесты)
    python reminder_server.py --db path.db     # своя БД SQLite
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from mcp.server.fastmcp import FastMCP

# ─── Инициализация MCP-сервера ──────────────────────────────────────────────
# host/port подставляются из CLI-аргументов в main(); значения по умолчанию —
# из переменных окружения (REMINDER_HOST/REMINDER_PORT) либо 127.0.0.1:8000.
mcp = FastMCP(
    "smart-reminder",
    host=os.environ.get("REMINDER_HOST", "127.0.0.1"),
    port=int(os.environ.get("REMINDER_PORT", "8000")),
)

# Путь к БД: по умолчанию reminders.db рядом со скриптом (корень проекта);
# переопределяется переменной REMINDER_DB_PATH (тесты) или --db (CLI).
DB_PATH = Path(
    os.environ.get("REMINDER_DB_PATH")
    or (Path(__file__).parent / "reminders.db")
)

# ─── Реестр подписчиков SSE-уведомлений ─────────────────────────────────────
# {ServerSession: event loop, в котором выполняется запрос}. Наполняется
# при вызовах инструментов (см. _register_session), читается потоком
# APScheduler. Инструменты — async и выполняются прямо в event loop
# uvicorn, поэтому loop каждого запроса доступен через
# asyncio.get_running_loop() (у sync-инструментов его нет: anyio гоняет
# их в воркер-потоках). Мьютекс обязателен: рассылка идёт из чужого потока.
_subscribers: Dict[object, asyncio.AbstractEventLoop] = {}
_subscribers_lock = threading.Lock()


def _register_session() -> None:
    """Регистрирует текущую MCP-сессию как подписчика уведомлений.

    Вызывается изнутри инструментов (есть контекст запроса). Для клиента
    это точка подключения к SSE-рассылке: после первого вызова любого
    инструмента его сессия получает notifications/message.

    Ошибки не пробрасываются: регистрация не должна ломать сам вызов
    инструмента (например, вне контекста запроса — при прямом вызове
    функции планировщиком).
    """
    try:
        ctx = mcp.get_context()
        session = ctx.request_context.session
        loop = asyncio.get_running_loop()
        if session is None or loop is None:
            return
        with _subscribers_lock:
            _subscribers[session] = loop
    except Exception:
        # Вне запроса (планировщик вызывает impl-функции напрямую) контекста
        # и loop нет — это нормальный сценарий, не ошибка.
        pass


def broadcast(text: str) -> int:
    """Шлёт notifications/message всем подписанным сессиям (из любого потока).

    Args:
        text: текст уведомления (например, ``"REMINDER: выпить кофе"``).

    Returns:
        Количество сессий, которым доставка была запланирована успешно.
        Сессии с закрытым loop удаляются из реестра.
    """
    with _subscribers_lock:
        items = list(_subscribers.items())
    delivered = 0
    for session, loop in items:
        try:
            asyncio.run_coroutine_threadsafe(
                session.send_log_message(level="info", data=text), loop
            )
            delivered += 1
        except RuntimeError:
            # Loop закрыт — клиент отключился; вычищаем подписку.
            with _subscribers_lock:
                _subscribers.pop(session, None)
    return delivered


# ─── SQLite ──────────────────────────────────────────────────────────────────


def init_db() -> None:
    """Создаёт таблицы, если их нет (reminders + reminder_log)."""
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                interval_minutes INTEGER NOT NULL,
                repeat INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                next_due_at TIMESTAMP NOT NULL,
                last_triggered_at TIMESTAMP,
                trigger_count INTEGER DEFAULT 0,
                is_active INTEGER DEFAULT 1
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reminder_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reminder_id INTEGER NOT NULL,
                triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (reminder_id)
                    REFERENCES reminders(id) ON DELETE CASCADE
            )
        """)
        conn.commit()
    finally:
        conn.close()


def _connect() -> sqlite3.Connection:
    """Новое соединение SQLite (создаётся и закрывается в одном потоке).

    timeout=10: планировщик и инструменты FastMCP (event loop uvicorn)
    пишут в одну БД; короткие транзакции с ожиданием блокировки не
    создают конкуренции на практике.
    """
    return sqlite3.connect(DB_PATH, timeout=10.0)


# ─── Инструменты MCP ─────────────────────────────────────────────────────────


def _add_reminder_impl(text: str, interval_minutes: int,
                       repeat: bool = False) -> Dict:
    """Реализация add_reminder без контекста MCP (используется и тестами)."""
    if not str(text).strip():
        return {"error": "text не должен быть пустым"}
    try:
        interval = int(interval_minutes)
    except (TypeError, ValueError):
        return {"error": "interval_minutes должен быть целым числом"}
    if interval <= 0:
        return {"error": "interval_minutes должен быть больше 0"}

    next_due = datetime.now() + timedelta(minutes=interval)
    conn = _connect()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO reminders (text, interval_minutes, repeat, "
            "next_due_at) VALUES (?, ?, ?, ?)",
            (str(text), interval, int(bool(repeat)), next_due.isoformat()),
        )
        reminder_id = int(cursor.lastrowid)
        conn.commit()
    finally:
        conn.close()

    return {
        "id": reminder_id,
        "text": str(text),
        "interval_minutes": interval,
        "repeat": bool(repeat),
        "next_due_at": next_due.isoformat(),
    }


@mcp.tool()
async def add_reminder(text: str, interval_minutes: int, repeat: bool = False) -> Dict:
    """Создать напоминание.

    Args:
        text: Текст напоминания (например, "выпить кофе").
        interval_minutes: Через сколько минут напомнить (целое число > 0).
        repeat: Повторять ли напоминание (true = каждые interval_minutes).

    Returns:
        Словарь с полями: id, text, interval_minutes, repeat, next_due_at.
        При некорректных аргументах — {"error": "..."}.
    """
    _register_session()
    return _add_reminder_impl(text, interval_minutes, repeat)


def _get_due_reminders_impl() -> List[Dict]:
    """Реализация get_due_reminders (используется планировщиком и тестами).

    Транзакционно: выборка due-строк → запись в reminder_log → сдвиг
    next_due_at (repeat) либо деактивация (одноразовые).
    """
    now = datetime.now()
    conn = _connect()
    due: List[Dict] = []
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, text, interval_minutes, repeat FROM reminders "
            "WHERE is_active = 1 AND next_due_at <= ?",
            (now.isoformat(),),
        )
        rows = cursor.fetchall()

        for row in rows:
            reminder_id, text, interval_minutes, repeat = row
            due.append({
                "id": reminder_id,
                "text": text,
                "triggered_at": now.isoformat(),
            })

            cursor.execute(
                "INSERT INTO reminder_log (reminder_id, triggered_at) "
                "VALUES (?, ?)",
                (reminder_id, now.isoformat()),
            )
            if repeat:
                next_due = now + timedelta(minutes=int(interval_minutes))
                cursor.execute(
                    "UPDATE reminders SET next_due_at = ?, "
                    "last_triggered_at = ?, trigger_count = trigger_count + 1 "
                    "WHERE id = ?",
                    (next_due.isoformat(), now.isoformat(), reminder_id),
                )
            else:
                cursor.execute(
                    "UPDATE reminders SET is_active = 0, "
                    "last_triggered_at = ?, trigger_count = trigger_count + 1 "
                    "WHERE id = ?",
                    (now.isoformat(), reminder_id),
                )

        conn.commit()
    finally:
        conn.close()
    return due


@mcp.tool()
async def get_due_reminders() -> List[Dict]:
    """Получить напоминания, срок которых подошёл.

    Транзакционно обновляет next_due_at для повторяющихся и деактивирует
    одноразовые (повторный вызов не возвращает то же напоминание).

    Returns:
        Список словарей с полями: id, text, triggered_at.
    """
    _register_session()
    return _get_due_reminders_impl()


@mcp.tool()
async def get_all_reminders() -> List[Dict]:
    """Получить все активные напоминания.

    Returns:
        Список словарей с полями: id, text, interval_minutes, repeat,
        next_due_at, trigger_count. Отсортирован по времени срабатывания.
    """
    _register_session()
    conn = _connect()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, text, interval_minutes, repeat, next_due_at, "
            "trigger_count FROM reminders WHERE is_active = 1 "
            "ORDER BY next_due_at"
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    return [
        {
            "id": r[0],
            "text": r[1],
            "interval_minutes": r[2],
            "repeat": bool(r[3]),
            "next_due_at": r[4],
            "trigger_count": r[5],
        }
        for r in rows
    ]


def _delete_reminder_impl(reminder_id: int) -> Dict:
    """Реализация delete_reminder (используется и тестами)."""
    try:
        rid = int(reminder_id)
    except (TypeError, ValueError):
        return {"success": False, "error": "reminder_id должен быть числом"}

    conn = _connect()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM reminders WHERE id = ?", (rid,))
        deleted = cursor.rowcount > 0
        conn.commit()
    finally:
        conn.close()
    return {"success": deleted}


@mcp.tool()
async def delete_reminder(reminder_id: int) -> Dict:
    """Удалить напоминание по ID (в том числе неактивные записи).

    Args:
        reminder_id: ID напоминания (из add_reminder/get_all_reminders).

    Returns:
        {"success": true} — если запись удалена, иначе {"success": false}.
    """
    _register_session()
    return _delete_reminder_impl(reminder_id)


def _get_stats_impl() -> Dict:
    """Реализация get_stats (используется и слушателем уведомлений)."""
    conn = _connect()
    try:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM reminders")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM reminders WHERE is_active = 1")
        active = cursor.fetchone()[0]

        today = datetime.now().strftime("%Y-%m-%d")
        cursor.execute(
            "SELECT COUNT(*) FROM reminder_log WHERE DATE(triggered_at) = ?",
            (today,),
        )
        triggered_today = cursor.fetchone()[0]

        cursor.execute(
            "SELECT AVG(interval_minutes) FROM reminders WHERE is_active = 1"
        )
        avg_interval = cursor.fetchone()[0] or 0
    finally:
        conn.close()

    return {
        "total": total,
        "active": active,
        "triggered_today": triggered_today,
        "avg_interval": round(float(avg_interval), 1),
    }


@mcp.tool()
async def get_stats() -> Dict:
    """Получить агрегированную статистику по напоминаниям.

    Returns:
        Словарь с полями: total (всего записей), active (активных),
        triggered_today (срабатываний сегодня), avg_interval
        (средний интервал активных, минуты).
    """
    _register_session()
    return _get_stats_impl()


# ─── APScheduler (внутри сервера) ────────────────────────────────────────────

scheduler = BackgroundScheduler()

NOTIFICATION_PREFIX = "REMINDER:"


def check_due_reminders() -> None:
    """Вызывается каждую минуту: due-напоминания → SSE-уведомления.

    Срабатывания фиксируются транзакционно (_get_due_reminders_impl), а
    текст рассылается всем подписанным сессиям через их event loop.
    Ошибка доставки не должна ронять задачу планировщика.
    """
    try:
        due = _get_due_reminders_impl()
    except Exception as exc:
        logging.getLogger("reminder.scheduler").error(
            "Проверка напоминаний не удалась: %s", exc)
        return
    for reminder in due:
        text = f"{NOTIFICATION_PREFIX} {reminder['text']}"
        logging.getLogger("reminder.scheduler").info(text)
        broadcast(text)


# ─── Точка входа ─────────────────────────────────────────────────────────────


def main() -> None:
    """Разбирает CLI, инициализирует БД и стартует планировщик + сервер."""
    global DB_PATH
    parser = argparse.ArgumentParser(
        description="MCP-сервер Smart Reminder (Streamable HTTP)"
    )
    parser.add_argument(
        "--host", default=os.environ.get("REMINDER_HOST", "127.0.0.1"),
        help="адрес привязки (по умолчанию 127.0.0.1 — только локально)",
    )
    parser.add_argument(
        "--port", type=int,
        default=int(os.environ.get("REMINDER_PORT", "8000")),
        help="порт HTTP-сервера (по умолчанию 8000)",
    )
    parser.add_argument(
        "--db", default=os.environ.get("REMINDER_DB_PATH"),
        help="путь к SQLite-БД (по умолчанию reminders.db рядом со скриптом)",
    )
    parser.add_argument(
        "--check-seconds", type=float,
        default=float(os.environ.get("REMINDER_CHECK_SECONDS", "60")),
        help="интервал проверки due-напоминаний, сек (по умолчанию 60; "
             "меньшие значения — для тестов)",
    )
    args = parser.parse_args()

    if args.db:
        DB_PATH = Path(args.db)
    # Синхронизируем настройки FastMCP с CLI (порт может прийти аргументом)
    mcp.settings.host = args.host
    mcp.settings.port = int(args.port)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    init_db()
    scheduler.add_job(
        check_due_reminders,
        # По ТЗ проверка выполняется каждую минуту; --check-seconds
        # существует для быстрых интеграционных тестов.
        trigger=IntervalTrigger(seconds=max(1.0, float(args.check_seconds))),
        id="check_reminders",
        replace_existing=True,
    )
    scheduler.start()

    url = f"http://{args.host}:{args.port}/mcp"
    print(f"Smart Reminder MCP-сервер запущен на {url}")
    print(f"БД: {DB_PATH}")
    print("Окно можно закрыть вместе с завершением сервера (Ctrl+C).")
    try:
        # streamable-http, stateful: сессии живут между HTTP-запросами
        mcp.run(transport="streamable-http")
    finally:
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
