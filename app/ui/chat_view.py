"""Вкладка «Чат»: история, ввод, RAG (всегда включён), память задачи.

Потокобезопасность: все обновления UI выполняются в async-обработчиках,
запущенных через page.run_task (event loop Flet); блокирующие вызовы
агента (SDK OpenAI, MCP, SQLite) уходят в asyncio.to_thread внутри
ChatHandler/RAGChatAgent. Фоновые потоки UI не трогают.

ТЗ ai_1_24: ход идёт через ChatHandler (память задачи + RAG-поиск +
гейт 0.6 + цитаты), под каждым ответом — свёрнутые источники с
цитатами, в шапке — кнопки «📋 Состояние диалога» (модалка с памятью
задачи) и «➕ Новый диалог» (новый session_id: сброс памяти и истории
с подтверждением). RAG в чате всегда включён (переключатель заблокирован).

Методы *_async можно вызывать напрямую (тесты, asyncio.run) — работа с
page вынесена в schedule-обёртки (_on_send и т.п.).
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any, Dict, List, Optional

import flet as ft

from app.chat_handler import STRATEGY_C_NOTE
from app.rag.pipeline import RAG_MODES, RAG_MODE_LABELS
from app.task_state import TaskState
from app.ui import components


def build_task_state_dialog(state: Dict[str, Any], on_reset,
                            on_close) -> ft.AlertDialog:
    """Модалка «Состояние диалога»: память задачи + действия (ТЗ ai_1_24).

    Args:
        state: словарь TaskState.to_dict().
        on_reset: обработчик кнопки «Сбросить состояние» (sync).
        on_close: обработчик кнопки «Закрыть» (sync).
    """
    return ft.AlertDialog(
        title=ft.Text("📋 Состояние диалога", weight=ft.FontWeight.BOLD),
        content=ft.Container(
            content=components.task_state_panel(state), width=460),
        actions=[
            ft.TextButton("Сбросить состояние", on_click=on_reset),
            ft.TextButton("Закрыть", on_click=on_close),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )


def build_new_dialog_dialog(on_cancel, on_confirm) -> ft.AlertDialog:
    """Модалка подтверждения «Новый диалог» (сброс без подтверждения
    запрещён ТЗ)."""
    return ft.AlertDialog(
        title=ft.Text("➕ Новый диалог", weight=ft.FontWeight.BOLD),
        content=ft.Container(
            content=ft.Text(
                "Начать новый диалог? Текущая история и память задачи "
                "будут сброшены.",
                size=14,
            ),
            width=380,
        ),
        actions=[
            ft.TextButton("Отмена", on_click=on_cancel),
            ft.FilledButton("Начать новый диалог", on_click=on_confirm),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )


class ChatView:
    """Чат: RAG всегда включён, источники с цитатами, память задачи.

    Attributes:
        agent: RAGChatAgent (ядро + RAG-пайплайн).
        app: владелец (page, notify, session_id, chat_handler);
            в тестах — заглушка (тогда ход идёт напрямую через agent).
        last_dialog: последняя модалка (для офлайн-тестов без page).
    """

    def __init__(self, app: Any) -> None:
        self.app = app
        self.agent = app.agent
        self._running = False
        self._pending_bubble: Optional[ft.Container] = None
        self.last_dialog: Optional[ft.AlertDialog] = None

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
            border=ft.OutlineInputBorder(border_radius=8),
            expand=True,
            on_submit=self._on_send,
        )
        self.send_button = ft.IconButton(
            icon=ft.Icons.SEND,
            tooltip="Отправить",
            on_click=self._on_send,
        )
        # ТЗ ai_1_24: RAG в чате всегда включён — переключатель заблокирован
        self.rag_switch = ft.Switch(
            label="Использовать RAG (всегда включён)",
            value=self.agent.rag_enabled,
            on_change=self._on_rag_toggle,
            disabled=True,
            tooltip="RAG в чате всегда включён: источники и цитаты "
                    "под каждым ответом",
        )
        self.state_button = ft.FilledTonalButton(
            "📋 Состояние диалога",
            on_click=self._on_dialog_state,
            tooltip="Память задачи: цель, уточнения, ограничения, термины",
        )
        self.new_dialog_button = ft.FilledTonalButton(
            "➕ Новый диалог",
            on_click=self._on_new_dialog,
            tooltip="Новый диалог: сброс памяти задачи и истории "
                    "(новый session_id)",
        )
        self.chat_actions_row = ft.Row(
            [self.state_button, self.new_dialog_button],
            spacing=8, wrap=True,
        )
        self.top_k_field = ft.TextField(
            value=str(self.agent.top_k),
            width=70,
            label="top_k",
            input_filter=ft.NumbersOnlyInputFilter(),
            on_blur=self._on_top_k_change,
            on_submit=self._on_top_k_change,
        )
        self.rag_mode_dropdown = ft.Dropdown(
            value=self.agent.rag_mode,
            width=190,
            label="RAG-режим",
            options=[
                ft.DropdownOption(key=mode, content=ft.Text(
                    RAG_MODE_LABELS[mode], size=12))
                for mode in RAG_MODES
            ],
            on_select=self._on_rag_mode_change,
            tooltip="Пайплайн поиска: rewrite / фильтр / реранкер",
        )
        self.threshold_field = ft.TextField(
            value=str(self.agent.similarity_threshold),
            width=110,
            label="порог",
            on_blur=self._on_threshold_change,
            on_submit=self._on_threshold_change,
        )
        self.rag_status_text = ft.Text("", size=12, color=ft.Colors.BLUE_GREY_500)
        self.status_text = ft.Text(
            "Готов к работе", size=12, color=ft.Colors.BLUE_GREY_500, expand=True
        )
        self.controls_row = ft.Container(
            content=ft.Row([
                self.rag_switch,
                self.rag_mode_dropdown,
                self.top_k_field,
                self.threshold_field,
                self.rag_status_text,
            ], spacing=12, wrap=True),
            padding=ft.Padding(left=10, top=0, right=10, bottom=0),
        )

    # ------------------------------------------------------------------
    # Компоновка
    # ------------------------------------------------------------------
    def build(self) -> ft.Column:
        """Собирает вкладку: чат, кнопки диалога, панель RAG, ввод, статус."""
        return ft.Column([
            self.messages_list,
            self.chat_actions_row,
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
                f"⚠️ **Ошибка:** {result['error']}",
                components.ERROR_TEXT_COLOR))
            self._set_status(f"Ошибка: {result['error']}", error=True)
            return
        content.controls.append(components.markdown_text(
            str(result.get("text") or "(пустой ответ)"),
            components.MESSAGE_TEXT_COLOR))
        rag_error = result.get("rag_error")
        if rag_error:
            content.controls.append(ft.Text(
                f"⚠️ RAG: {rag_error}", size=12,
                color=components.RAG_WARNING_COLOR, selectable=True))
        elif result.get("has_rag") is False:
            # Стратегия C (ТЗ ai_1_24): RAG не нашёл релевантных чанков —
            # ответ из общих знаний без источников
            content.controls.append(ft.Text(
                STRATEGY_C_NOTE, size=12,
                color=components.RAG_WARNING_COLOR, selectable=True))
        sources_panel = components.sources_panel(
            result.get("sources") or [],
            quotes=result.get("quotes") or None,
        )
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
        self._schedule(self._send_async)

    def _on_rag_toggle(self, event=None) -> None:
        # Переключатель работает на лету: режим применяется со следующего хода
        self.agent.set_rag_enabled(bool(self.rag_switch.value))
        self.refresh_rag_status()

    def _on_rag_mode_change(self, event=None) -> None:
        try:
            self.agent.set_rag_mode(str(self.rag_mode_dropdown.value or ""))
        except ValueError:
            self.rag_mode_dropdown.value = self.agent.rag_mode
        self.refresh_rag_status()

    def _on_threshold_change(self, event=None) -> None:
        try:
            value = float(
                str(self.threshold_field.value or "").replace(",", "."))
            self.agent.set_similarity_threshold(value)
        except ValueError:
            self.threshold_field.value = str(self.agent.similarity_threshold)
        self.refresh_rag_status()

    def _on_top_k_change(self, event=None) -> None:
        try:
            self.agent.set_top_k(int(self.top_k_field.value or "3"))
        except ValueError:
            self.top_k_field.value = str(self.agent.top_k)
        self.refresh_rag_status()

    def _on_clear(self, event=None) -> None:
        self._schedule(self._clear_async)

    def _on_open_file(self, event=None) -> None:
        self._schedule(self._open_file_async, event)

    # ------------------------------------------------------------------
    # Модалки: «Состояние диалога» и «Новый диалог» (ТЗ ai_1_24)
    # ------------------------------------------------------------------
    def _on_dialog_state(self, event=None) -> None:
        self._schedule(self._dialog_state_async)

    def _on_close_dialog(self, event=None) -> None:
        self._schedule(self._close_dialog_async)

    def _on_reset_task_state(self, event=None) -> None:
        self._schedule(self._reset_task_state_async)

    def _on_new_dialog(self, event=None) -> None:
        self._schedule(self._new_dialog_confirm_async)

    def _on_cancel_new_dialog(self, event=None) -> None:
        self._schedule(self._close_dialog_async)

    def _on_confirm_new_dialog(self, event=None) -> None:
        self._schedule(self._start_new_dialog_async)

    def _open_dialog(self, dialog: ft.AlertDialog) -> None:
        """Показ модалки; без page (офлайн-тесты) — в self.last_dialog."""
        if self.app.page is not None:
            self.app.page.show_dialog(dialog)
        else:
            self.last_dialog = dialog

    def _chat_handler(self):
        """ChatHandler приложения (None в legacy-заглушках тестов)."""
        return getattr(self.app, "chat_handler", None)

    def _session_id(self) -> str:
        return str(getattr(self.app, "session_id", "") or "")

    async def _dialog_state_async(self) -> None:
        """Модалка с текущей памятью задачи (цель/уточнения/термины)."""
        handler = self._chat_handler()
        if handler is not None:
            profile_id = (self.app.profile_id()
                          if hasattr(self.app, "profile_id") else 0)
            state = await asyncio.to_thread(
                handler.load_state, self._session_id(), profile_id)
        else:
            state = TaskState(session_id=self._session_id(), profile_id=0)
        self._open_dialog(build_task_state_dialog(
            state.to_dict(), self._on_reset_task_state,
            self._on_close_dialog))

    async def _close_dialog_async(self) -> None:
        """Закрывает верхнюю модалку (page.pop_dialog)."""
        if self.app.page is not None:
            self.app.page.pop_dialog()

    async def _reset_task_state_async(self) -> None:
        """«Сбросить состояние»: память задачи сессии удаляется из БД."""
        handler = self._chat_handler()
        if handler is not None and self._session_id():
            await asyncio.to_thread(
                handler.reset_session, self._session_id())
        await self._close_dialog_async()
        self._set_status("Память задачи сброшена")
        self.update()

    async def _new_dialog_confirm_async(self) -> None:
        """Подтверждение «Новый диалог» (сброс без подтверждения запрещён)."""
        self._open_dialog(build_new_dialog_dialog(
            self._on_cancel_new_dialog, self._on_confirm_new_dialog))

    async def _start_new_dialog_async(self) -> None:
        """Новый диалог: сброс памяти задачи + истории, новый session_id.

        Порядок из ТЗ: сброс памяти задачи старой сессии в SQLite →
        очистка истории ядра (и БД) → новый UUID → пустой чат.
        """
        old_session_id = self._session_id()
        handler = self._chat_handler()
        if handler is not None and old_session_id:
            await asyncio.to_thread(handler.reset_session, old_session_id)
        await asyncio.to_thread(self.agent.clear_history)
        self.app.session_id = str(uuid.uuid4())
        self.messages_list.controls.clear()
        await self._close_dialog_async()
        self._set_status(
            "Новый диалог: память задачи и история сброшены")
        self.update()

    def _schedule(self, handler, *args) -> None:
        """Запускает async-задачу на event loop Flet.

        Принимает КОРУТИННУЮ ФУНКЦИЮ и её аргументы: page.run_task сам
        создаёт корутину — передача готовой корутины даёт TypeError
        «handler must be a coroutine function» (и полноэкранный
        error-overlay Flet, блокирующий вкладки). Задача оборачивается
        в гард: исключение пишется в лог и показывается SnackBar'ом,
        а не роняет интерфейс.

        Без page (офлайн-тесты) ничего не делает: корутины запускает
        вызывающий код через asyncio.run.
        """
        if self.app.page is None:
            return
        self.app.page.run_task(
            components.guarded_task(self.app, handler), *args)

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
            handler = getattr(self.app, "chat_handler", None)
            if handler is not None:
                # ТЗ ai_1_24: ход через ChatHandler — память задачи,
                # RAG-гейт 0.6, цитаты, обновление состояния в SQLite
                profile_id = (self.app.profile_id()
                              if hasattr(self.app, "profile_id") else 0)
                result = await handler.handle_message(
                    str(getattr(self.app, "session_id", "") or ""),
                    profile_id, text)
            else:
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
        """Статус базы знаний: число чанков + режим + top_k + порог."""
        try:
            chunks = self.agent.count_chunks()
        except Exception:
            chunks = 0
        mode = "включён" if self.agent.rag_enabled else "выключен"
        self.rag_status_text.value = (
            f"RAG {mode} · база: {chunks} чанков · "
            f"{self.agent.rag_mode} · top_k: {self.agent.top_k}→"
            f"{self.agent.top_k_search} · порог: "
            f"{self.agent.similarity_threshold:.2f}"
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
