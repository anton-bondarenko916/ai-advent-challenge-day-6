"""Вкладка «Чат»: история, ввод, переключатель RAG, панель источников.

Потокобезопасность: все обновления UI выполняются в async-обработчиках,
запущенных через page.run_task (event loop Flet); блокирующие вызовы
агента (SDK OpenAI, MCP) уходят в asyncio.to_thread внутри
RAGChatAgent.send_message. Фоновые потоки UI не трогают.

Методы *_async можно вызывать напрямую (тесты, asyncio.run) — работа с
page вынесена в schedule-обёртки (_on_send и т.п.).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

import flet as ft

from app.ui import components


class ChatView:
    """Чат с переключателем RAG и top_k.

    Attributes:
        agent: RAGChatAgent (ядро + RAG-пайплайн).
        app: владелец (page, notify); в тестах — заглушка.
    """

    def __init__(self, app: Any) -> None:
        self.app = app
        self.agent = app.agent
        self._running = False
        self._pending_bubble: Optional[ft.Container] = None

        self.messages_list = ft.ListView(
            expand=True,
            auto_scroll=True,
            spacing=8,
            padding=10,
        )
        self.input_field = ft.TextField(
            value="",
            multiline=True,
            min_lines=1,
            max_lines=4,
            shift_enter=True,
            hint_text="Сообщение (Enter — отправить, Shift+Enter — новая строка)",
            border_radius=8,
            expand=True,
            on_submit=self._on_send,
        )
        self.send_button = ft.IconButton(
            icon=ft.Icons.SEND,
            tooltip="Отправить",
            on_click=self._on_send,
        )
        self.rag_switch = ft.Switch(
            label="Использовать RAG",
            value=self.agent.rag_enabled,
            on_change=self._on_rag_toggle,
        )
        self.top_k_field = ft.TextField(
            value=str(self.agent.top_k),
            width=70,
            label="top_k",
            input_filter=ft.NumbersOnlyInputFilter(),
            on_blur=self._on_top_k_change,
            on_submit=self._on_top_k_change,
        )
        self.rag_status_text = ft.Text("", size=12, color=ft.Colors.BLUE_GREY_500)
        self.status_text = ft.Text(
            "Готов к работе", size=12, color=ft.Colors.BLUE_GREY_500, expand=True
        )
        self.controls_row = ft.Container(
            content=ft.Row([
                self.rag_switch,
                self.top_k_field,
                self.rag_status_text,
            ], spacing=12, wrap=True),
            padding=ft.Padding(left=10, top=0, right=10, bottom=0),
        )

    # ------------------------------------------------------------------
    # Компоновка
    # ------------------------------------------------------------------
    def build(self) -> ft.Column:
        """Собирает вкладку: чат, панель RAG, ввод, статус-строка."""
        return ft.Column([
            self.messages_list,
            self.controls_row,
            ft.Row([self.input_field, self.send_button], spacing=8),
            self.status_text,
        ], expand=True, spacing=6)

    def on_show(self) -> None:
        """Обновление статус-строки RAG при открытии вкладки."""
        self.refresh_rag_status()

    # ------------------------------------------------------------------
    # История сообщений
    # ------------------------------------------------------------------
    def restore_history(self, messages: List[Dict[str, Any]]) -> None:
        """Отображает сохранённую историю ядра (без панелей источников)."""
        for message in messages:
            role = str(message.get("role"))
            if role not in ("user", "assistant"):
                continue
            self.messages_list.controls.append(
                components.message_bubble(
                    str(message.get("content") or ""), role
                )
            )

    def add_user_message(self, text: str) -> None:
        """Добавляет пузырь пользователя и скроллит вниз."""
        self.messages_list.controls.append(
            components.message_bubble(text, "user")
        )

    def show_thinking(self) -> None:
        """Заглушка «Думаю…» на время запроса."""
        self._pending_bubble = components.message_bubble("Думаю…", "assistant")
        self.messages_list.controls.append(self._pending_bubble)

    def add_assistant_result(self, result: Dict[str, Any]) -> None:
        """Рисует ответ ассистента: текст + источники + инструменты + файл.

        Args:
            result: словарь от send_message (text, sources, tool_calls,
                file_path, error, rag_error, timestamp и т.д.).
        """
        bubble = self._replace_placeholder()
        content = bubble.content
        if result.get("error"):
            content.controls.append(components.markdown_text(
                f"⚠️ **Ошибка:** {result['error']}"))
            self._set_status(f"Ошибка: {result['error']}", error=True)
            return
        content.controls.append(components.markdown_text(
            str(result.get("text") or "(пустой ответ)")))
        rag_error = result.get("rag_error")
        if rag_error:
            content.controls.append(ft.Text(
                f"⚠️ RAG: {rag_error}", size=12,
                color=ft.Colors.ORANGE_700, selectable=True))
        sources_panel = components.sources_panel(result.get("sources") or [])
        if sources_panel is not None:
            content.controls.append(sources_panel)
        tools_panel = components.tools_panel(result.get("tool_calls") or [])
        if tools_panel is not None:
            content.controls.append(tools_panel)
        file_path = result.get("file_path")
        if file_path:
            content.controls.append(components.file_open_button(
                file_path, self._on_open_file))

        tokens = result.get("tokens") or {}
        response_time = result.get("response_time")
        rag_mark = " · RAG" if result.get("rag_used") else ""
        self._set_status(
            f"{result.get('model')}{rag_mark} · "
            f"{tokens.get('total_tokens', '?')} токенов · {response_time}с"
        )

    # ------------------------------------------------------------------
    # Обработчики (schedule → run_task; логика — в *_async)
    # ------------------------------------------------------------------
    def _on_send(self, event=None) -> None:
        self._schedule(self._send_async())

    def _on_rag_toggle(self, event=None) -> None:
        # Переключатель работает на лету: режим применяется со следующего хода
        self.agent.set_rag_enabled(bool(self.rag_switch.value))
        self.refresh_rag_status()

    def _on_top_k_change(self, event=None) -> None:
        try:
            self.agent.set_top_k(int(self.top_k_field.value or "3"))
        except ValueError:
            self.top_k_field.value = str(self.agent.top_k)
        self.refresh_rag_status()

    def _on_clear(self, event=None) -> None:
        self._schedule(self._clear_async())

    def _on_open_file(self, event=None) -> None:
        self._schedule(self._open_file_async(event))

    def _schedule(self, coroutine) -> None:
        """Запускает корутину на event loop Flet (или сразу в тестах)."""
        if self.app.page is not None:
            self.app.page.run_task(coroutine)
        else:
            # Офлайн-режим (тесты): корутину запускает вызывающий код
            coroutine.close()

    # ------------------------------------------------------------------
    # Async-логика
    # ------------------------------------------------------------------
    async def _send_async(self) -> None:
        """Полный цикл отправки: пузырь → запрос → ответ с панелями."""
        text = (self.input_field.value or "").strip()
        if not text or self._running:
            return
        self._running = True
        self.send_button.disabled = True
        self.input_field.value = ""
        self.add_user_message(text)
        self.show_thinking()
        self._set_status("Запрос к модели…")
        self.update()
        try:
            result = await self.agent.send_message(text)
        except Exception as exc:  # сеть/поток — показываем как ошибку хода
            result = {"error": str(exc)}
        finally:
            self.add_assistant_result(result)
            self._running = False
            self.send_button.disabled = False
            self.update()

    async def _clear_async(self) -> None:
        """Очищает историю (ядро + БД) и экран."""
        await asyncio.to_thread(self.agent.clear_history)
        self.messages_list.controls.clear()
        self._set_status("История очищена")
        self.update()

    async def _open_file_async(self, event) -> None:
        """Открывает сохранённый инструментом файл (os.startfile)."""
        control = getattr(event, "control", None)
        path = str(control.tooltip or "") if control else ""
        if path and os.path.exists(path):
            await asyncio.to_thread(os.startfile, path)  # noqa: S606
        else:
            self.app.notify(f"Файл не найден: {path}")

    # ------------------------------------------------------------------
    # Служебное
    # ------------------------------------------------------------------
    def refresh_rag_status(self) -> None:
        """Статус базы знаний: число чанков + top_k."""
        try:
            chunks = self.agent.count_chunks()
        except Exception:
            chunks = 0
        mode = "включён" if self.agent.rag_enabled else "выключен"
        self.rag_status_text.value = (
            f"RAG {mode} · база: {chunks} чанков · top_k: {self.agent.top_k}"
        )

    def _replace_placeholder(self) -> ft.Container:
        """Заменяет «Думаю…» на контейнер ответа (или создаёт новый)."""
        if self._pending_bubble is not None:
            bubble = self._pending_bubble
            self._pending_bubble = None
            bubble.content = ft.Column(spacing=2, tight=True)
            return bubble
        bubble = components.message_bubble("", "assistant")
        self.messages_list.controls.append(bubble)
        bubble.content = ft.Column(spacing=2, tight=True)
        return bubble

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status_text.value = text
        self.status_text.color = (
            ft.Colors.RED_600 if error else ft.Colors.BLUE_GREY_500
        )

    def update(self) -> None:
        """Перерисовка вкладки (только из event loop)."""
        if self.app.page is not None:
            self.app.page.update()
