"""Вкладка «Чат»: история, ввод, RAG (всегда включён), память задачи.

Потокобезопасность: все обновления UI выполняются в async-обработчиках,
запущенных через page.run_task (event loop Flet); блокирующие вызовы
агента (SDK OpenAI, MCP, SQLite) уходят в asyncio.to_thread внутри
ChatHandler/RAGChatAgent. Фоновые потоки UI не трогают.

ТЗ ai_1_24: ход идёт через ChatHandler (память задачи + RAG-поиск +
гейт 0.6 + цитаты), под каждым ответом — свёрнутые источники с
цитатами, в шапке — кнопки «📋 Состояние диалога» (модалка с памятью
задачи) и «➕ Новый диалог» (новый session_id: сброс памяти и истории
с подтверждением).

ТЗ ai_1_25: в шапке — выпадающий список провайдера LLM (DeepSeek облако
↔ локальные модели Ollama). Переключение «на лету» через
core.switch_provider: перед ним лаунчер гарантирует, что Ollama запущен
и модель загружена (ollama pull при первом использовании); недоступный
провайдер даёт модалку ошибки и откат списка; выбор сохраняется в
app_state (SQLite) и восстанавливается при следующем запуске.

Команда /help: при вводе «/» в поле появляется подсказка автодополнения
(клик подставляет команду); сообщение уходит в ChatHandler, где
перехватывается HelpPipeline (документация проекта + git-контекст),
память задачи не затрагивается.

Команда /support (ТЗ ai_1_34): та же панель автодополнения предлагает
/support; рядом с провайдером — выпадающий список «Пользователь (для
/support)», заполняемый из CRM (list_users через crm_server.py) при
старте приложения. Выбранный user_id уходит в SupportPipeline через
app.support_user_id (НЕ хардкодится); недоступный CRM — список пуст,
ход /support отвечает только по FAQ (фолбэк).

RAG по умолчанию включён, но переключатель разблокирован: выключенное
положение отправляет вопрос ядру без поиска по базе знаний (память
задачи и инструменты MCP работают как раньше). Положение переключателя
сохраняется в app_state (ключ rag_enabled).

Методы *_async можно вызывать напрямую (тесты, asyncio.run) — работа с
page вынесена в schedule-обёртки (_on_send и т.п.).
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any, Dict, List, Optional

import flet as ft

from app.chat_handler import STRATEGY_C_NOTE
from app.rag.pipeline import RAG_MODES, RAG_MODE_LABELS
from app.task_state import TaskState
from app.ui import components
from src.llm_providers import (
    LLMProviderFactory,
    ProviderUnavailableError,
)
from src.ollama_launcher import OllamaLauncher
from src.utils import get_logger

logger = get_logger("deepseek-chat.chat-view")

# Ключ выбора провайдера в app_state (SQLite)
ACTIVE_PROVIDER_KEY = "active_provider"

# Ключ положения переключателя RAG в app_state (SQLite)
RAG_ENABLED_KEY = "rag_enabled"


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


def build_provider_error_dialog(message: str, on_close) -> ft.AlertDialog:
    """Модалка ошибки переключения провайдера (ТЗ ai_1_25)."""
    return ft.AlertDialog(
        title=ft.Text("❌ Провайдер недоступен", weight=ft.FontWeight.BOLD),
        content=ft.Container(content=ft.Text(message, size=14), width=420),
        actions=[ft.TextButton("Закрыть", on_click=on_close)],
        actions_alignment=ft.MainAxisAlignment.END,
    )


def build_provider_warning_dialog(message: str, on_close) -> ft.AlertDialog:
    """Модалка предупреждения (например, tool calling не поддерживается)."""
    return ft.AlertDialog(
        title=ft.Text("⚠️ Внимание", weight=ft.FontWeight.BOLD),
        content=ft.Container(content=ft.Text(message, size=14), width=420),
        actions=[ft.TextButton("Понятно", on_click=on_close)],
        actions_alignment=ft.MainAxisAlignment.END,
    )


def parse_crm_users(raw: Any) -> List[Dict[str, Any]]:
    """Список пользователей из ответа MCP (list_users CRM-сервера).

    FastMCP оборачивает список в ``{"result": [...]}`` — клиент
    возвращает его pretty-JSON'ом; принимается и сырой JSON-список.
    Элементы без id отбрасываются.
    """
    try:
        payload = json.loads(str(raw or ""))
    except (TypeError, ValueError):
        return []
    if isinstance(payload, dict) and set(payload) == {"result"}:
        payload = payload["result"]
    if not isinstance(payload, list):
        return []
    return [
        user for user in payload
        if isinstance(user, dict) and user.get("id") is not None
    ]


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
            hint_text="Сообщение (Enter — отправить, Shift+Enter — новая строка; /support — поддержка, /help — о проекте)",
            border=ft.OutlineInputBorder(border_radius=8),
            expand=True,
            on_change=self._on_input_change,
            on_submit=self._on_send,
        )
        # Автодополнение команд (ввод «/»): панель подсказок под полем
        self.suggestions = ft.Column(visible=False, spacing=0)
        self.send_button = ft.IconButton(
            icon=ft.Icons.SEND,
            tooltip="Отправить",
            on_click=self._on_send,
        )
        # RAG включён по умолчанию, переключатель разблокирован: выключенное
        # положение отправляет запрос без поиска по базе знаний
        self.rag_switch = ft.Switch(
            label="Использовать RAG",
            value=self._initial_rag_enabled(),
            on_change=self._on_rag_toggle,
            tooltip="Поиск по базе знаний + источники и цитаты под ответом. "
                    "Выключите, чтобы отправлять запросы без RAG",
        )
        # Положение из app_state применяется и к агенту (ход без RAG)
        self.agent.set_rag_enabled(bool(self.rag_switch.value))
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
        # ТЗ ai_1_25: переключатель провайдера LLM в шапке чата
        self.provider_dropdown = ft.Dropdown(
            width=250,
            label="Провайдер",
            options=[
                ft.DropdownOption(key=provider_id, content=ft.Text(
                    spec["name"], size=12))
                for provider_id, spec in LLMProviderFactory.PROVIDERS.items()
            ],
            value=self._current_provider_id(),
            on_select=self._on_provider_change,
            tooltip="DeepSeek — облако; Ollama — локальные модели "
                    "(нужен запущенный ollama serve)",
        )
        # ТЗ ai_1_34: пользователь CRM для /support (список заполняется
        # из CRM при старте приложения; user_id НЕ хардкодится)
        self.support_user_dropdown = ft.Dropdown(
            width=250,
            label="Пользователь (для /support)",
            options=[],
            on_select=self._on_support_user_change,
            tooltip="Контекст CRM (тикеты, статус) для команды /support",
        )
        self.chat_actions_row = ft.Row(
            [self.provider_dropdown, self.support_user_dropdown,
             self.state_button, self.new_dialog_button],
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
            self.suggestions,
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
        if result.get("is_support") and not result.get("crm_available"):
            # /support без CRM (ТЗ: фолбэк) — предупреждение под ответом
            crm_error = str(result.get("crm_error") or "CRM недоступен")
            content.controls.append(ft.Text(
                f"👤 {crm_error} — ответ дан только по FAQ поддержки.",
                size=12, color=components.RAG_WARNING_COLOR,
                selectable=True))
        if result.get("is_help") and result.get("git_context"):
            # /help: git-контекст хода (ветка или пометка о недоступности)
            content.controls.append(ft.Text(
                f"🌿 {result['git_context']}", size=12,
                color=components.RAG_WARNING_COLOR, selectable=True))
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

    # ------------------------------------------------------------------
    # Автодополнение команд при вводе «/» (ТЗ)
    # ------------------------------------------------------------------
    def _suggestion_controls(self) -> List[ft.Control]:
        """Подсказки команд (/help, /support — ТЗ ai_1_34)."""
        return [
            ft.Container(
                content=ft.Text(
                    "/support <вопрос> — поддержка пользователей "
                    "(контекст CRM + FAQ)",
                    size=13,
                ),
                on_click=lambda _: self._fill_suggestion("/support "),
                ink=True,
                border_radius=8,
                padding=ft.Padding(left=12, top=6, right=12, bottom=6),
            ),
            ft.Container(
                content=ft.Text(
                    "/help — вопросы о проекте (документация, архитектура, git)",
                    size=13,
                ),
                on_click=lambda _: self._fill_suggestion("/help "),
                ink=True,
                border_radius=8,
                padding=ft.Padding(left=12, top=6, right=12, bottom=6),
            ),
        ]

    def _on_input_change(self, event=None) -> None:
        """Ввод в поле: показывает подсказку, пока текст вида «/…»
        (моложе 10 символов — команда ещё набирается)."""
        text = str(self.input_field.value or "")
        if text.startswith("/") and len(text) < 10:
            self.suggestions.controls = self._suggestion_controls()
            self.suggestions.visible = True
        else:
            self.suggestions.visible = False
        self.update()

    def _fill_suggestion(self, value: str) -> None:
        """Клик по подсказке: подставляет команду и прячет панель."""
        self.input_field.value = value
        self.suggestions.visible = False
        if self.app.page is not None:
            try:
                self.input_field.focus()
            except Exception:  # noqa: BLE001 — фокус не критичен
                pass
        self.update()

    def _on_rag_toggle(self, event=None) -> None:
        # Переключатель работает на лету: режим применяется со следующего хода
        enabled = bool(self.rag_switch.value)
        self.agent.set_rag_enabled(enabled)
        self._save_rag_enabled(enabled)
        self._set_status(
            "RAG включён: поиск по базе знаний, источники и цитаты"
            if enabled
            else "RAG выключен: запросы уходят без поиска по базе знаний")
        self.refresh_rag_status()
        self.update()

    def _initial_rag_enabled(self) -> bool:
        """Начальное положение переключателя: выбор из app_state.

        Нет сохранённого значения (или хранилища) — состояние агента
        (RAG включён по умолчанию).
        """
        store = getattr(self.app, "app_state", None)
        saved = store.get(RAG_ENABLED_KEY) if store is not None else None
        if saved is None:
            return bool(self.agent.rag_enabled)
        return str(saved).strip() == "1"

    def _save_rag_enabled(self, enabled: bool) -> None:
        """Сохраняет положение переключателя в app_state (не фатально)."""
        store = getattr(self.app, "app_state", None)
        if store is not None:
            store.set(RAG_ENABLED_KEY, "1" if enabled else "0")

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

    def _on_provider_change(self, event=None) -> None:
        self._schedule(self._switch_provider_async,
                       str(self.provider_dropdown.value or ""))

    # ------------------------------------------------------------------
    # Пользователь CRM для /support (ТЗ ai_1_34)
    # ------------------------------------------------------------------
    def _on_support_user_change(self, event=None) -> None:
        """Выбор пользователя: user_id запоминается в приложении и
        используется следующим ходом /support."""
        try:
            user_id = int(self.support_user_dropdown.value or 0)
        except (TypeError, ValueError):
            user_id = 0
        self.app.support_user_id = user_id

    async def refresh_support_users_async(self) -> None:
        """Заполняет список пользователей из CRM (list_users, HTTP).

        Вызывается при старте приложения (page.run_task): ход не
        блокирует интерфейс. CRM недоступен — список остаётся пустым
        (ход /support возможен: фолбэк без контекста пользователя).
        """
        client = getattr(self.app, "support_client", None)
        if client is None:
            return
        try:
            raw = await client.call_tool("list_users", {})
            users = parse_crm_users(raw)
        except Exception as exc:  # noqa: BLE001 — деградация вместо падения
            logger.warning("Список пользователей CRM недоступен: %s", exc)
            users = []
        if users:
            self.support_user_dropdown.options = [
                ft.DropdownOption(
                    key=str(user.get("id")),
                    content=ft.Text(
                        f"{user.get('name', '')} ({user.get('id')})",
                        size=12),
                )
                for user in users
            ]
            if not self.selected_support_user_id():
                first = str(users[0].get("id"))
                self.support_user_dropdown.value = first
                self.app.support_user_id = int(first)
        else:
            self.support_user_dropdown.disabled = True
        self.update()

    def selected_support_user_id(self) -> int:
        """user_id выбранного пользователя (0 — не выбран)."""
        try:
            return int(getattr(self.app, "support_user_id", 0) or 0)
        except (TypeError, ValueError):
            return 0

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

    # ------------------------------------------------------------------
    # Переключение провайдера LLM (ТЗ ai_1_25)
    # ------------------------------------------------------------------
    def _core_agent(self):
        """Ядро DeepSeekAgent (в legacy-заглушках тестов — сам agent)."""
        return getattr(self.agent, "core", self.agent)

    def _current_provider_id(self) -> str:
        """Идентификатор активного провайдера ядра."""
        return str(getattr(self._core_agent(), "provider_id", "deepseek")
                   or "deepseek")

    def schedule_provider_switch(self, provider_id: str) -> None:
        """Переключение провайдера через event loop (восстановление выбора
        из app_state при старте приложения)."""
        self.provider_dropdown.value = provider_id
        self._schedule(self._switch_provider_async, provider_id)

    def _save_provider_choice(self, provider_id: str) -> None:
        """Сохраняет выбор в app_state (переживает перезапуск приложения).

        Хранилище «неубиваемое»: ошибка SQLite уже гасится внутри
        AppStateStore.set — выбор просто не сохранится.
        """
        store = getattr(self.app, "app_state", None)
        if store is not None:
            store.set(ACTIVE_PROVIDER_KEY, provider_id)

    def _prepare_and_switch(self, provider_id: str) -> Dict[str, Any]:
        """Подготовка локального провайдера + switch_provider ядра.

        Для ollama_*: лаунчер гарантирует запущенный сервер (автозапуск
        в новом окне, если не работает) и загруженную модель (ollama
        pull при первом использовании). Блокирующие вызовы — вызывающий
        поток (asyncio.to_thread), UI не блокируется.

        Raises:
            ProviderUnavailableError: сервер не поднялся / модель не
                загрузилась / провайдер недоступен.
            ValueError: неизвестное имя провайдера.
        """
        spec = LLMProviderFactory.spec(provider_id)
        core = self._core_agent()
        mock = bool(getattr(core, "_mock_mode", False))
        if spec["requires_ollama"] and not mock:
            launcher = OllamaLauncher()
            if not launcher.ensure_running(timeout=30.0):
                raise ProviderUnavailableError(
                    "Ollama не запущен и не смог запуститься "
                    "автоматически. Установите Ollama (ollama.com) и "
                    "проверьте, что команда «ollama serve» работает.")
            model_ok, model_message = launcher.ensure_model(spec["model"])
            if not model_ok:
                raise ProviderUnavailableError(model_message)
        return core.switch_provider(provider_id)

    async def _switch_provider_async(self, provider_id: str) -> None:
        """Полный цикл переключения: подготовка → ядро → app_state → UI.

        Недоступный провайдер не меняет состояние ядра: показывается
        модалка ошибки, выпадающий список откатывается к активному
        провайдеру.
        """
        provider_id = str(provider_id or "")
        if not provider_id:
            return
        # Выбранное значение выставляется и при прямом вызове
        # (восстановление из app_state), не только из on_select
        self.provider_dropdown.value = provider_id
        self._set_status("Переключение провайдера…")
        self.update()
        try:
            result = await asyncio.to_thread(
                self._prepare_and_switch, provider_id)
        except ProviderUnavailableError as exc:
            self._open_dialog(build_provider_error_dialog(
                str(exc), self._on_close_dialog))
            self.provider_dropdown.value = self._current_provider_id()
            self._set_status(f"Провайдер не переключён: {exc}", error=True)
            self.update()
            return
        except ValueError as exc:  # неизвестное имя провайдера
            self._open_dialog(build_provider_error_dialog(
                str(exc), self._on_close_dialog))
            self.provider_dropdown.value = self._current_provider_id()
            self._set_status(f"Провайдер не переключён: {exc}", error=True)
            self.update()
            return
        self._save_provider_choice(provider_id)
        if "warning" in result:
            self._open_dialog(build_provider_warning_dialog(
                str(result["warning"]), self._on_close_dialog))
        self._set_status(
            f"Провайдер: {result.get('name')} · модель {result.get('model')}")
        self.refresh_rag_status()
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
        self.suggestions.visible = False
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
        """Статус: активный провайдер + число чанков + режим + параметры."""
        core = self._core_agent()
        provider = getattr(core, "provider", None)
        if provider is not None:
            provider_label = provider.describe()
        else:
            provider_label = (
                f"{LLMProviderFactory.PROVIDERS['deepseek']['name']}"
                f" · {getattr(core, 'model', '')}"
            )
        try:
            chunks = self.agent.count_chunks()
        except Exception:
            chunks = 0
        mode = "включён" if self.agent.rag_enabled else "выключен"
        self.rag_status_text.value = (
            f"Провайдер: {provider_label} · RAG {mode} · база: {chunks} "
            f"чанков · {self.agent.rag_mode} · top_k: {self.agent.top_k}→"
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
