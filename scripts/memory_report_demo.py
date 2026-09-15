"""Отчёт «какие данные попали в каждый слой» (get_memory_report / preview_context).

Демонстрация для приёмки: заполняет все три слоя, печатает отчёт по слоям и
реальный список сообщений будущего запроса с пометкой слоя и токенов.
Файлы создаются в data/ и удаляются в конце.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import DeepSeekAgent  # noqa: E402
from src.config import DEFAULTS, get_memory_settings, get_strategy_settings  # noqa: E402
from src.tokens import HeuristicCounter  # noqa: E402


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    long_db = PROJECT_ROOT / "data" / "report_long_memory.db"
    chat_db = PROJECT_ROOT / "data" / "report_chat.db"
    for path in (long_db, chat_db):
        try:
            path.unlink()
        except OSError:
            pass

    settings = get_memory_settings(dict(DEFAULTS))
    settings["long_path"] = str(long_db)
    agent = DeepSeekAgent(
        api_key="k", model="deepseek-v4-flash", token_counter=HeuristicCounter(),
        strategy="sliding_window", strategy_settings=get_strategy_settings({}),
        memory_settings=settings,
    )
    # --- заполняем три слоя ---
    agent.remember_forever("profile", "имя", "Алекс")
    agent.remember_forever("profile", "любимый_цвет", "синий")
    agent.remember_forever("knowledge", "правило", "отвечать кратко")
    agent.set_task_field("goal", "написать отчёт")
    agent.set_task_field("status", "в работе")
    agent.set_task_field("plan", ["собрать данные", "написать текст"])
    agent.add_working_item("дедлайн", "20 мая", "note")
    agent.add_working_item("формат", "Markdown", "artifact")
    agent._append_message({"role": "user", "content":
                           "Меня зовут Алекс, любимый цвет — синий."})
    agent._append_message({"role": "assistant", "content": "Запомнил."})
    agent.pin_message(0)
    for i in range(8):
        agent._append_message({"role": "user", "content": f"вопрос {i}"})
        agent._append_message({"role": "assistant", "content": f"ответ {i}"})

    report = agent.get_memory_report()
    print("=== ОТЧЁТ ПО СЛОЯМ (get_memory_report) ===")
    print(f"включена: {report['enabled']}")
    short = report["short_term"]
    print(f"[short_term] сообщений: {short['messages']} · "
          f"закреплено: {short['pinned_count']} · источник: {short['source']}")
    for pin in short["items"]:
        print(f"    закреплено #{pin['index']} (заметка: {pin['note']})")
    working = report["working"]
    print(f"[working] элементов: {working['count']} · "
          f"≈{working['tokens']} ток. · источник: {working['source']}")
    print(f"    task.goal={working['task']['goal']!r} · "
          f"status={working['task']['status']!r} · plan={working['task']['plan']}")
    for item in working["items"]:
        print(f"    {item['kind']}: {item['key']} = {item['value']} "
              f"(source={item['source']}, updated={item['updated_at']})")
    long_term = report["long_term"]
    print(f"[long_term] записей: {long_term['count']} · "
          f"разделы: {long_term['sections']} · ≈{long_term['tokens']} ток. · "
          f"источник: {long_term['source']}")
    for item in long_term["items"]:
        print(f"    [{item['section']}] {item['key']} = {item['value']} "
              f"(source={item['source']}, created={item['created_at']})")
    print(f"pending (предложений): {len(report['pending'])} · "
          f"бюджет: {report['budget']}")

    print("\n=== ЧТО УХОДИТ В ЗАПРОС (preview_context) ===")
    for message in agent.preview_context():
        content = " ".join(str(message["content"]).split())
        if len(content) > 90:
            content = content[:87] + "..."
        print(f"[{message['layer']:<11}] {message['role']:<9} "
              f"≈{message['tokens']:>3} ток. | {content}")

    agent.close()
    for path in (long_db, chat_db):
        try:
            path.unlink()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
