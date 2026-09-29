"""Вкладка «База знаний»: индексация MD-файлов, настройки chunking.

Папка, chunk_size, overlap и стратегия (Fixed / Structure / Обе) —
только через UI, без хардкода (требование ТЗ).

Индексация идёт по файлам: каждый файл обрабатывается в
asyncio.to_thread, прогресс-бар обновляется в event loop между
await — фоновые потоки UI не трогают.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import flet as ft

from app.rag.chunkers import STRATEGY_FIXED, STRATEGY_STRUCTURE, validate_chunking
from app.rag.indexer import KnowledgeIndexer

# Значения Dropdown стратегий → последовательности стратегий chunking
STRATEGY_CHOICES: Dict[str, tuple] = {
    "fixed": (STRATEGY_FIXED,),
    "structure": (STRATEGY_STRUCTURE,),
    "both": (STRATEGY_FIXED, STRATEGY_STRUCTURE),
}
STRATEGY_LABELS = {
    "fixed": "Fixed (фиксированный размер)",
    "structure": "Structure (по заголовкам)",
    "both": "Обе стратегии",
}


def strategies_for_choice(choice: Optional[str]) -> tuple:
    """Маппинг выбора UI → стратегии ("Обе" → fixed+structure)."""
    return STRATEGY_CHOICES.get(str(choice or "fixed"), (STRATEGY_FIXED,))


class KnowledgeView:
    """Вкладка индексации документов.

    Attributes:
        agent: RAGChatAgent (используются его embedder/vector_store).
        file_picker: FilePicker страницы (может быть None в тестах).
    """

    def __init__(self, app: Any, default_folder: Optional[str] = None) -> None:
        self.app = app
        self.agent = app.agent
        self.indexer = KnowledgeIndexer(
            self.agent.embedder, self.agent.vector_store
        )
        self._indexing = False

        self.folder_field = ft.TextField(
            value=default_folder or "",
            label="Папка с MD-файлами",
            expand=True,
            on_submit=self._on_pick_folder,
        )
        self.pick_button = ft.IconButton(
            icon=ft.Icons.FOLDER_OPEN,
            tooltip="Выбрать папку",
            on_click=self._on_pick_folder,
        )
        self.chunk_size_field = ft.TextField(
            value="500", label="chunk_size", width=110, input_filter=ft.NumbersOnlyInputFilter(),
        )
        self.overlap_field = ft.TextField(
            value="50", label="overlap", width=110, input_filter=ft.NumbersOnlyInputFilter(),
        )
        self.strategy_dropdown = ft.Dropdown(
            label="Стратегия chunking",
            value="fixed",
            width=280,
            options=[
                ft.DropdownOption(key=key, content=ft.Text(STRATEGY_LABELS[key]))
                for key in ("fixed", "structure", "both")
            ],
        )
        self.index_button = ft.FilledButton(
            "Проиндексировать",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._on_index,
        )
        self.clear_button = ft.TextButton(
            "Очистить базу",
            icon=ft.Icons.DELETE_SWEEP,
            tooltip="Удалить все чанки коллекции",
            on_click=self._on_clear,
        )
        self.progress_bar = ft.ProgressBar(value=0, visible=False)
        self.status_text = ft.Text(
            self._idle_status(), size=12, color=ft.Colors.BLUE_GREY_500,
            selectable=True, expand=True,
        )

    # ------------------------------------------------------------------
    # Компоновка
    # ------------------------------------------------------------------
    def build(self) -> ft.Column:
        """Собирает вкладку: папка, настройки, кнопки, прогресс, статус."""
        return ft.Column([
            ft.Row([self.folder_field, self.pick_button], spacing=8),
            ft.Row([
                self.chunk_size_field,
                self.overlap_field,
                self.strategy_dropdown,
            ], spacing=12, wrap=True),
            ft.Row([self.index_button, self.clear_button], spacing=12),
            self.progress_bar,
            self.status_text,
        ], spacing=12)

    def on_show(self) -> None:
        """Обновляет статус при открытии вкладки."""
        self._set_status(self._idle_status())

    # ------------------------------------------------------------------
    # Обработчики
    # ------------------------------------------------------------------
    def _on_pick_folder(self, event=None) -> None:
        self._schedule(self._pick_folder_async())

    def _on_index(self, event=None) -> None:
        self._schedule(self._index_async())

    def _on_clear(self, event=None) -> None:
        self._schedule(self._clear_async())

    def _schedule(self, coroutine) -> None:
        if self.app.page is not None:
            self.app.page.run_task(coroutine)
        else:
            coroutine.close()

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
            dialog_title="Папка с MD-документами"
        )
        if path:
            self.folder_field.value = str(path)
            self._set_status(self._idle_status())
            self.update()

    async def _index_async(self) -> None:
        """Полная индексация: файлы → чанки → эмбеддинги → Chroma."""
        if self._indexing:
            return
        try:
            folder, strategies, chunk_size, overlap = self._read_settings()
        except ValueError as exc:
            self._set_status(f"Ошибка: {exc}", error=True)
            self.update()
            return
        files = self.indexer.collect_files(folder)
        if not files:
            self._set_status(
                f"В папке нет MD-файлов: {folder}", error=True)
            self.update()
            return
        self._indexing = True
        self.index_button.disabled = True
        self.progress_bar.visible = True
        self.progress_bar.value = 0
        try:
            total = len(files)
            for number, path in enumerate(files, start=1):
                # Блокирующая работа — в worker-поток, UI-луп свободен
                chunks = await asyncio.to_thread(
                    self.indexer.index_file, path, strategies,
                    chunk_size, overlap,
                )
                self.progress_bar.value = number / total
                self._set_status(
                    f"Файл {number}/{total}: {path.name} — чанков: {chunks}"
                )
                self.update()
            stats = await asyncio.to_thread(self.indexer.commit)
            summary = (
                f"Проиндексировано: {stats.files} файлов, {stats.chunks} чанков "
                f"({'+'.join(stats.strategies) or '—'}), "
                f"chunk_size={stats.chunk_size}, overlap={stats.overlap}, "
                f"{stats.duration}с"
            )
            if stats.errors:
                summary += "\nПропущены: " + "; ".join(stats.errors)
            self._set_status(summary, error=bool(stats.errors))
        except Exception as exc:  # Ollama недоступна и т.п.
            self._set_status(f"Ошибка индексации: {exc}", error=True)
        finally:
            self._indexing = False
            self.index_button.disabled = False
            self.progress_bar.visible = False
            self.update()
            chat_view = getattr(self.app, "chat_view", None)
            if chat_view is not None:
                chat_view.refresh_rag_status()

    async def _clear_async(self) -> None:
        """Очищает коллекцию Chroma."""
        await asyncio.to_thread(self.indexer.clear)
        self._set_status(self._idle_status())
        self.update()

    # ------------------------------------------------------------------
    # Служебное
    # ------------------------------------------------------------------
    def _read_settings(self):
        """Читает и валидирует настройки из полей UI.

        Returns:
            (папка Path, стратегии, chunk_size, overlap).

        Raises:
            ValueError: некорректные числа или путь.
        """
        folder = Path((self.folder_field.value or "").strip().strip('"'))
        if not folder or not folder.is_dir():
            raise ValueError(f"папка не найдена: {folder}")
        try:
            chunk_size = int(self.chunk_size_field.value)
            overlap = int(self.overlap_field.value)
        except (TypeError, ValueError):
            raise ValueError("chunk_size и overlap должны быть числами")
        validate_chunking(chunk_size, overlap)
        strategies = strategies_for_choice(self.strategy_dropdown.value)
        return folder, strategies, chunk_size, overlap

    def _idle_status(self) -> str:
        try:
            chunks = self.agent.count_chunks()
        except Exception:
            chunks = 0
        return f"В базе знаний: {chunks} чанков. Выберите папку с MD-файлами и нажмите «Проиндексировать»."

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status_text.value = text
        self.status_text.color = (
            ft.Colors.RED_600 if error else ft.Colors.BLUE_GREY_500
        )

    def update(self) -> None:
        if self.app.page is not None:
            self.app.page.update()
