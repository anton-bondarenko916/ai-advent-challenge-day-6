"""Панель «Документация проекта» (вкладка «Настройки»): путь + индексация.

Поле «Путь к проекту» + кнопка «Проиндексировать документацию проекта»:
ProjectRAG индексирует README + docs/*.md + конфиги в отдельную
коллекцию Chroma ``project_docs`` (личная база знаний не затрагивается).
Выбранный путь сохраняется в app_state (ключ project_docs_path,
переживает перезапуск) и применяется к HelpPipeline команды /help.

Структура — как у OptimizationPanel/RemoteAccessPanel: секция с
рамкой, статус-строка, офлайн-тесты зовут *_async напрямую (без page).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import flet as ft

from app.rag.embedder import EmbedderError
from app.rag.project_rag import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    PROJECT_DOCS_PATH_KEY,
    ProjectRAG,
    default_project_path,
)

# Заголовок секции
PANEL_TITLE = "🗂 Документация проекта (/help)"


class ProjectDocsPanel:
    """Панель индексации документации проекта.

    Attributes:
        app: владелец (page, notify, app_state, project_rag,
            help_pipeline); в офлайн-тестах — заглушка.
        project_rag: ProjectRAG (создаётся лениво, если app его не даёт).
    """

    def __init__(self, app: Any) -> None:
        self.app = app
        agent = app.agent
        self.project_rag: ProjectRAG = (
            getattr(app, "project_rag", None)
            or ProjectRAG(agent.vector_store, agent.embedder)
        )
        self._indexing = False

        self.path_field = ft.TextField(
            label="Путь к проекту",
            value=self._initial_path(),
            hint_text="Путь к папке с README и docs/",
            expand=True,
            on_blur=self._on_path_blur,
            on_submit=self._on_index,
        )
        self.pick_button = ft.IconButton(
            icon=ft.Icons.FOLDER_OPEN,
            tooltip="Выбрать папку проекта",
            on_click=self._on_pick_folder,
        )
        self.index_button = ft.FilledTonalButton(
            "Проиндексировать документацию проекта",
            icon=ft.Icons.SYNC,
            on_click=self._on_index,
            tooltip="README + docs/*.md + конфиги → коллекция project_docs",
        )
        self.status_text = ft.Text(
            self._idle_status(), size=12,
            color=ft.Colors.BLUE_GREY_500, selectable=True, expand=True,
        )

    # ------------------------------------------------------------------
    # Компоновка
    # ------------------------------------------------------------------
    def build(self) -> ft.Container:
        """Секция с рамкой: путь, кнопки, статус (как в ai_1_28/ai_1_30)."""
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
            dialog_title="Папка проекта (README + docs/)"
        )
        if path:
            self.path_field.value = str(path)
            self._save_path(path)
            self.update()

    async def _index_async(self) -> None:
        """Индексация документации проекта (Ollama — в worker-потоке)."""
        if self._indexing:
            return
        raw_path = str(self.path_field.value or "").strip().strip('"')
        folder = Path(raw_path) if raw_path else None
        if folder is None or not folder.is_dir():
            self._set_status(f"Путь к проекту не найден: {raw_path}",
                             error=True)
            self.update()
            return
        self._indexing = True
        self.index_button.disabled = True
        self._set_status("Индексация документации проекта…")
        self.update()
        try:
            stats = await asyncio.to_thread(
                self.project_rag.index_project, str(folder),
                DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP,
            )
        except EmbedderError as exc:
            self._set_status(f"Ошибка индексации: {exc}", error=True)
        except Exception as exc:  # noqa: BLE001 — ход UI не должен падать
            self._set_status(f"Ошибка индексации: {exc}", error=True)
        else:
            # Путь валиден и проиндексирован — применяем к /help
            self._apply_project_path(str(folder))
            summary = (f"Проиндексировано: {stats['files']} файлов, "
                       f"{stats['chunks']} чанков "
                       f"(коллекция {self.project_rag.collection_name}), "
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
        """Путь из app_state → из HelpPipeline → корень приложения."""
        store = getattr(self.app, "app_state", None)
        saved = store.get(PROJECT_DOCS_PATH_KEY) if store is not None else None
        if saved and Path(saved).is_dir():
            return saved
        pipeline = getattr(self.app, "help_pipeline", None)
        if pipeline is not None:
            path = Path(str(getattr(pipeline, "project_path", "")))
            if path.is_dir():
                return str(path)
        return default_project_path()

    def _save_path(self, path: Any) -> bool:
        """Сохраняет путь в app_state (не фатально, если хранилища нет)."""
        store = getattr(self.app, "app_state", None)
        if store is None:
            return False
        value = str(path or "").strip().strip('"')
        return bool(store.set(PROJECT_DOCS_PATH_KEY, value))

    def _apply_project_path(self, path: str) -> None:
        """Проиндексированный путь становится путём git-контекста /help
        и сохраняется в app_state (переживает перезапуск)."""
        pipeline = getattr(self.app, "help_pipeline", None)
        if pipeline is not None:
            pipeline.project_path = path
        self._save_path(path)

    def _idle_status(self) -> str:
        try:
            chunks = self.project_rag.count()
        except Exception:  # noqa: BLE001 — статус не должен падать
            chunks = 0
        return (f"В коллекции {self.project_rag.collection_name}: "
                f"{chunks} чанков. /help отвечает по документации проекта "
                f"с git-контекстом.")

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status_text.value = text
        self.status_text.color = (
            ft.Colors.RED_600 if error else ft.Colors.BLUE_GREY_500
        )

    def update(self) -> None:
        if self.app.page is not None:
            self.app.page.update()
