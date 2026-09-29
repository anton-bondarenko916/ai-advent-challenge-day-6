"""Flet-приложение DeepSeek Chat: чат с RAG, база знаний, сравнение режимов.

Пакет построен поверх ядра ``src`` (DeepSeekAgent с профилями, памятью,
FSM задач, инвариантами и MCP-клиентами). Точка входа — ``app/main.py``:

    python app/main.py            # нативное окно (нужен DEEPSEEK_API_KEY)
    python app/main.py --mock     # mock-режим без обращения к API
    flet run app/main.py          # через flet CLI
"""

from __future__ import annotations

__all__ = ["agent", "main", "rag", "ui"]
