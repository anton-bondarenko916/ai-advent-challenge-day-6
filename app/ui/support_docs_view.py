"""Панель «Поддержка /support» (вкладка «Настройки»): путь + индексация.

Поле «Путь к папке поддержки» + кнопка «Проиндексировать FAQ»:
SupportRAG индексирует все *.md (рекурсивно) в отдельную коллекцию
Chroma ``support_faq`` (ТЗ ai_1_34: НЕ смешивать с другими
коллекциями). Выбранный путь сохраняется в app_state (ключ
support_docs_path, переживает перезапуск) и используется автоиндексацией
при старте приложения (когда коллекция пуста).

Структура — как у ProjectDocsPanel (ai_1_32): секция с рамкой,
статус-строка, офлайн-тесты зовут *_async напрямую (без page).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import flet as ft

from app.rag.embedder import EmbedderError
from app.rag.support_rag import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    SUPPORT_DOCS_PATH_KEY,
    SupportRAG,
    default_support_docs_path,
)

# Заголовок секции
PANEL_TITLE = "🛟 Поддержка (/support)"


class SupportDocsPanel:
    """Панель индексации FAQ поддержки (коллекция support_faq).

    Attributes:
        app: владелец (page, notify, app_state, support_rag);
            в офлайн-тестах — заглушка.
        support_rag: SupportRAG (создаётся лениво, если app его не даёт).
    """

    def __init__(self, app: Any) -> None:
        self.app = app
        agent = app.agent
        self.support_rag: SupportRAG = (
            getattr(app, "support_rag", None)
            or SupportRAG(agent.vector_store, agent.embedder)
        )
        self._indexing = False

        self.path_field = ft.TextField(
            label="Путь к папке поддержки (FAQ, *.md)",
            value=self._initial_path(),
            hint_text="По умолчанию support_docs/ в корне проекта",
            expand=True,
            on_blur=self._on_path_blur,
            on_submit=self._on_index,
        )
        self.pick_button = ft.IconButton(
            icon=ft.Icons.FOLDER_OPEN,
            tooltip="Выбрать папку поддержки",
            on_click=self._on_pick_folder,
        )
        self.index_button = ft.FilledTonalButton(
            "Проиндексировать FAQ поддержки",
            icon=ft.Icons.SYNC,
            on_click=self._on_index,
            tooltip="*.md рекурсивно → коллекция support_faq",
        )
        self.status_text = ft.Text(
            self._idle_status(), size=12,
            color=ft.Colors.BLUE_GREY_500, selectable=True, expand=True,
        )

    # ------------------------------------------------------------------
    # Компоновка
    # ------------------------------------------------------------------
    def build(self) -> ft.Container:
        """Секция с рамкой: путь, кнопки, статус."""
        return ft.Container(
            content=ft.Column([
                ft.Text(PANEL_TITLE, weight=ft.FontWeight.BOLD),
                ft.Row([self.path_field, self.pick_button], spacing=8),
                ft.Row([self.index_button], spacing=8),
                self.status_text,
            ], spacing=8),
            border=ft.Border.all(1, ft.Colors.BLUE_GREY_200),
            border_radius=10,
            padding=12,
        )

    def on_show(self) -> None:
        """Обновляет статус при открытии вкладки."""
        self.status_text.value = self._idle_status()
        self.status_text.color = ft.Colors.BLUE_GREY_500

    # ------------------------------------------------------------------
    # Обработчики
    # ------------------------------------------------------------------
    def _on_index(self, event=None) -> None:
        self._schedule(self._index_async)

    def _on_path_blur(self, event=None) -> None:
        """Путь запоминается сразу при уходе с поля (как в ai_1_28)."""
        self._save_path(self.path_field.value)

    def _on_pick_folder(self, event=None) -> None:
        self._schedule(self._pick_folder_async)

    def _schedule(self, handler, *args) -> None:
        """КОРУТИННАЯ ФУНКЦИЯ в page.run_task (готовая корутина даёт
        TypeError + error-overlay); без page (тесты) — no-op."""
        if self.app.page is None:
            return
        from app.ui.components import guarded_task
        self.app.page.run_task(guarded_task(self.app, handler), *args)

    # ------------------------------------------------------------------
    # Async-логика
    # ------------------------------------------------------------------
    async def _pick_folder_async(self) -> None:
        """Диалог выбора папки (FilePicker страницы)."""
        file_picker = getattr(self.app, "file_picker", None)
        if file_picker is None:
            self.app.notify("Диалог выбора папки недоступен")
            return
        path = await file_picker.get_directory_path(
            dialog_title="Папка поддержки (*.md)"
        )
        if path:
            self.path_field.value = str(path)
            self._save_path(path)
            self.update()

    async def _index_async(self) -> None:
        """Индексация FAQ поддержки (Ollama — в worker-потоке)."""
        if self._indexing:
            return
        raw_path = str(self.path_field.value or "").strip().strip('"')
        folder = Path(raw_path) if raw_path else None
        if folder is None or not folder.is_dir():
            self._set_status(f"Путь к папке поддержки не найден: {raw_path}",
                             error=True)
            self.update()
            return
        self._indexing = True
        self.index_button.disabled = True
        self._set_status("Индексация FAQ поддержки…")
        self.update()
        try:
            stats = await asyncio.to_thread(
                self.support_rag.index_support_docs, str(folder),
                DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP,
            )
        except EmbedderError as exc:
            self._set_status(f"Ошибка индексации: {exc}", error=True)
        except Exception as exc:  # noqa: BLE001 — ход UI не должен падать
            self._set_status(f"Ошибка индексации: {exc}", error=True)
        else:
            self._save_path(str(folder))
            summary = (f"Проиндексировано: {stats['files']} файлов, "
                       f"{stats['chunks']} чанков (коллекция "
                       f"{self.support_rag.collection_name}), "
                       f"{stats['duration']}с")
            if stats.get("errors"):
                summary += "\nПропущены: " + "; ".join(stats["errors"])
            self._set_status(summary, error=bool(stats.get("errors")))
        finally:
            self._indexing = False
            self.index_button.disabled = False
            self.update()

    # ------------------------------------------------------------------
    # Служебное
    # ------------------------------------------------------------------
    def _initial_path(self) -> str:
        """Путь из app_state → папка поддержки по умолчанию."""
        store = getattr(self.app, "app_state", None)
        saved = store.get(SUPPORT_DOCS_PATH_KEY) if store is not None else None
        if saved and Path(saved).is_dir():
            return saved
        return default_support_docs_path()

    def _save_path(self, path: Any) -> bool:
        """Сохраняет путь в app_state (не фатально, если хранилища нет)."""
        store = getattr(self.app, "app_state", None)
        if store is None:
            return False
        value = str(path or "").strip().strip('"')
        return bool(store.set(SUPPORT_DOCS_PATH_KEY, value))

    def _idle_status(self) -> str:
        try:
            chunks = self.support_rag.count()
        except Exception:  # noqa: BLE001 — статус не должен падать
            chunks = 0
        return (f"В коллекции {self.support_rag.collection_name}: "
                f"{chunks} чанков. /support отвечает по FAQ с контекстом "
                f"CRM (тикеты пользователя).")

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status_text.value = text
        self.status_text.color = (
            ft.Colors.RED_600 if error else ft.Colors.BLUE_GREY_500
        )

    def update(self) -> None:
        if self.app.page is not None:
            self.app.page.update()
