"""Вкладка «Настройки»: профиль, температура, max_tokens, модель.

Профили и настройки ядра: ProfileRepository (SQLite) + update_config
DeepSeekAgent — та же функциональность, что и в Tkinter-версии.
Валидация — общая (src.utils): validate_temperature/max_tokens/model.
"""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional

import flet as ft

from app.ui import components
from app.ui.optimization_view import OptimizationPanel
from app.ui.project_docs_view import ProjectDocsPanel
from app.ui.remote_access_view import RemoteAccessPanel
from src.utils import (
    validate_max_tokens,
    validate_model,
    validate_temperature,
)

# Модели, доступные ядру (подписи из src.utils.validate_model)
MODEL_CHOICES = ("deepseek-v4-flash", "deepseek-v4-pro")


class SettingsView:
    """Настройки агента: профиль и параметры LLM."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.agent = app.agent

        self.profile_dropdown = ft.Dropdown(
            label="Профиль",
            width=280,
            options=self._profile_options(),
            value=self._active_profile_value(),
        )
        self.profile_button = ft.FilledTonalButton(
            "Переключить профиль", on_click=self._on_switch_profile)

        self.model_dropdown = ft.Dropdown(
            label="Модель",
            width=280,
            options=[
                ft.DropdownOption(key=model, content=ft.Text(model))
                for model in MODEL_CHOICES
            ],
            value=self.agent.model if self.agent.model in MODEL_CHOICES
            else MODEL_CHOICES[0],
        )
        self.temperature_field = ft.TextField(
            label="Температура (0.0–2.0)",
            value=str(self.agent.temperature),
            width=200,
        )
        self.max_tokens_field = ft.TextField(
            label="max_tokens",
            value=str(self.agent.max_tokens),
            width=160,
            input_filter=ft.NumbersOnlyInputFilter(),
        )
        self.save_button = ft.FilledButton(
            "Сохранить", icon=ft.Icons.SAVE, on_click=self._on_save)

        embedder = self.agent.embedder
        self.rag_info_text = ft.Text(
            f"RAG: Ollama {getattr(embedder, 'base_url', '—')}, "
            f"модель эмбеддингов «{getattr(embedder, 'model', '—')}» "
            f"(переопределяется переменными OLLAMA_BASE_URL / "
            f"OLLAMA_EMBED_MODEL)",
            size=12, color=ft.Colors.BLUE_GREY_500, selectable=True,
        )
        self.status_text = ft.Text("", size=12, expand=True)

        # Оптимизация локальной модели (ТЗ ai_1_28): секция с рамкой
        # под основными настройками; глобальные настройки в app_state
        self.optimization_panel = OptimizationPanel(app)
        # Удалённый доступ (ТЗ ai_1_30): URL прокси + API Key + DDNS
        self.remote_access_panel = RemoteAccessPanel(app)
        # Документация проекта (/help): путь + индексация project_docs
        self.project_docs_panel = ProjectDocsPanel(app)

    # ------------------------------------------------------------------
    # Компоновка
    # ------------------------------------------------------------------
    def build(self) -> ft.Column:
        """Собирает вкладку настроек."""
        return ft.Column([
            ft.Row([self.profile_dropdown, self.profile_button],
                   spacing=12, wrap=True),
            ft.Row([
                self.model_dropdown,
                self.temperature_field,
                self.max_tokens_field,
            ], spacing=12, wrap=True),
            ft.Row([self.save_button, self.status_text], spacing=12),
            self.rag_info_text,
            self.optimization_panel.build(),
            self.remote_access_panel.build(),
            self.project_docs_panel.build(),
        ], spacing=12, scroll=ft.ScrollMode.AUTO)

    def on_show(self) -> None:
        """Обновляет список профилей и моделей при открытии вкладки.

        Список моделей следует за активным провайдером (ТЗ ai_1_25):
        DeepSeek — облачные модели, Ollama — локальная модель провайдера.
        """
        self.profile_dropdown.options = self._profile_options()
        self.profile_dropdown.value = self._active_profile_value()
        choices = self._model_choices()
        self.model_dropdown.options = [
            ft.DropdownOption(key=model, content=ft.Text(model))
            for model in choices
        ]
        self.model_dropdown.value = (
            self._core_agent().model if self._core_agent().model in choices
            else choices[0]
        )
        self.optimization_panel.on_show()
        self.remote_access_panel.on_show()
        self.project_docs_panel.on_show()
        self.update()

    def _core_agent(self):
        """Ядро DeepSeekAgent (в legacy-заглушках тестов — сам agent)."""
        return getattr(self.agent, "core", self.agent)

    def _model_choices(self) -> tuple:
        """Модели активного провайдера (без провайдера — DeepSeek)."""
        provider = getattr(self._core_agent(), "provider", None)
        choices = getattr(provider, "model_choices", None)
        return tuple(choices()) if callable(choices) else MODEL_CHOICES

    # ------------------------------------------------------------------
    # Обработчики
    # ------------------------------------------------------------------
    def _on_switch_profile(self, event=None) -> None:
        self._schedule(self._switch_profile_async)

    def _on_save(self, event=None) -> None:
        self._schedule(self._save_async)

    def _schedule(self, handler, *args) -> None:
        """Запускает async-задачу: КОРУТИННУЮ ФУНКЦИЮ (не корутину!) в
        page.run_task — иначе TypeError и error-overlay Flet. Гард
        (components.guarded_task) пишет traceback в лог и показывает
        SnackBar вместо падения интерфейса. Без page (тесты) — no-op."""
        if self.app.page is None:
            return
        self.app.page.run_task(
            components.guarded_task(self.app, handler), *args)

    # ------------------------------------------------------------------
    # Async-логика
    # ------------------------------------------------------------------
    async def _switch_profile_async(self) -> None:
        """Переключает активный профиль (ядро перечитывает настройки)."""
        value = self.profile_dropdown.value
        if not value:
            self._set_status("Профиль не выбран", error=True)
            self.update()
            return
        info = await asyncio.to_thread(self.agent.set_profile, int(value))
        self._set_status(f"Активный профиль: {info.get('name', value)}")
        self.update()

    async def _save_async(self) -> None:
        """Валидирует и сохраняет модель/температуру/max_tokens."""
        model = self.model_dropdown.value
        try:
            validate_model(model)
            temperature = float(self.temperature_field.value or "0")
            validate_temperature(temperature)
            max_tokens = int(self.max_tokens_field.value or "0")
            validate_max_tokens(max_tokens)
        except (ValueError, TypeError) as exc:
            self._set_status(f"Ошибка: {exc}", error=True)
            self.update()
            return
        await asyncio.to_thread(
            self.agent.update_config,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._set_status(
            f"Сохранено: {model}, температура {temperature}, "
            f"max_tokens {max_tokens}")
        self.update()

    # ------------------------------------------------------------------
    # Служебное
    # ------------------------------------------------------------------
    def _profile_options(self) -> List[ft.DropdownOption]:
        repository = self.agent.profile_repository
        if repository is None:
            return []
        options = []
        try:
            profiles = repository.get_all()
        except Exception:
            return []
        for profile in profiles:
            options.append(ft.DropdownOption(
                key=str(profile.id),
                content=ft.Text(f"{profile.name} (#{profile.id})", size=13),
            ))
        return options

    def _active_profile_value(self) -> Optional[str]:
        repository = self.agent.profile_repository
        if repository is None:
            return None
        try:
            profile = repository.get_active()
        except Exception:
            return None
        return str(profile.id) if profile else None

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status_text.value = text
        self.status_text.color = (
            ft.Colors.RED_600 if error else ft.Colors.BLUE_GREY_500
        )

    def update(self) -> None:
        if self.app.page is not None:
            self.app.page.update()
