"""Временный репро: полное ChatApp, открытая вкладка «RAG-режимы».

Запуск: python scripts/rag_view_repro.py
Используется для визуальной отладки рендера вкладки (серая область).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import flet as ft

from app.main import ChatApp
from src.config import load_config


def start(page: ft.Page) -> None:
    config = load_config()
    config["MOCK_MODE"] = True
    from app.agent import RAGChatAgent
    agent = RAGChatAgent.with_config(config).bind_config(config)
    app = ChatApp(page, agent, config)
    # Сразу открыть вкладку «RAG-режимы» (индекс 3) для скриншота
    app.tabs.selected_index = 3
    app._on_tab_change()


if __name__ == "__main__":
    ft.run(start)
