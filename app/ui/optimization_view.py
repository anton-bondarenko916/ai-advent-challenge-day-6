"""Панель «Оптимизация локальной модели (Ollama)» (ТЗ ai_1_28).

Вкладывается во вкладку «Настройки». Логика:

* переключатель «Включить оптимизацию»: снят — все поля заблокированы,
  запросы уходят с дефолтами Ollama; включён — параметры применяются
  к запросам активного Ollama-провайдера (температура, max_tokens →
  num_predict, num_ctx, top_p, repeat_penalty);
* системный промпт добавляется СВЕРХУ промпта агента (не заменяет);
* квант: список тегов из реестра ollama-models (при недоступности —
  fallback-список), отметка локально скачанных и кнопка загрузки;
* сохранение глобальное — таблица app_state (ключ ollama_optimization),
  переживает перезапуск; применение — set_ollama_optimization ядра.

Офлайн-режим (page=None в тестах): изменения применяются к ядру и
хранилищу сразу, обновления UI и фоновые задачи не выполняются.

Валидация полей: БЕЗ input_filter — NumbersOnlyInputFilter в Flet это
регэксп «только цифры» (точка/запятая блокируются, отклонённые правки
откатываются форматтером → ломается backspace). Вместо него своя
проверка формата на каждый ввод (свойство TextField.error — в Flet 1.0
нет error_text! — красная рамка и текст под полем) и перед
применением на blur; запятая считается равнозначной точке (русская
раскладка). Разрешено: «0.7», «0,7», «0.» (промежуточное при вводе),
пустое поле (при применении подставится минимум диапазона).
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional

import flet as ft

from app.ui import components
from src.llm_optimization import OptimizationSettings, OllamaTagManager

# Формат числа: цифры с необязательным хвостом-дробью (хвост может быть
# пустым — «0.» валидно на середине ввода); запятая равнозначна точке
FLOAT_FORMAT = re.compile(r"^\d+([.,]\d*)?$")
INT_FORMAT = re.compile(r"^\d+$")


class OptimizationPanel:
    """Панель оптимизации локальной модели (секция вкладки «Настройки»)."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.agent = app.agent
        self.settings: Dict[str, Any] = self._load_settings()
        # Метаданные тегов последнего обновления списка (tag →
        # {"quant": str, "local": bool}) и локальность выбранного тега;
        # ВНЕ settings: sanitize хранит только профиль инференса
        self._tag_meta: Dict[str, Dict[str, Any]] = {}
        self._selected_local: Optional[bool] = None

        self.enabled_switch = ft.Switch(
            label="Включить оптимизацию",
            value=bool(self.settings.get("enabled")),
            on_change=self._on_toggle,
        )

        # (поле, формат): float — точка/запятая разрешены, int — цифры.
        # Без input_filter: он блокировал точку и ломал стирание символов
        self._numeric_fields: List[Any] = []

        def _numeric(label: str, key: str, kind: str,
                     width: int = 160) -> ft.TextField:
            field = ft.TextField(
                label=label,
                value=str(self.settings.get(key, "")),
                width=width,
                on_change=self._on_field_change,
                on_blur=self._on_field_blur,
                disabled=not self.settings.get("enabled"),
            )
            self._numeric_fields.append((field, kind))
            return field

        self.temperature_field = _numeric(
            "Temperature (0.0–2.0)", "temperature", "float")
        self.max_tokens_field = _numeric(
            "Max tokens (num_predict)", "max_tokens", "int")
        self.num_ctx_field = _numeric(
            "Context window (num_ctx)", "num_ctx", "int")
        self.top_p_field = _numeric("Top-P (0.0–1.0)", "top_p", "float")
        self.repeat_penalty_field = _numeric(
            "Repeat penalty (0.0–2.0)", "repeat_penalty", "float")

        self.system_prompt_field = ft.TextField(
            label="Системный промпт (добавляется сверху к промпту агента)",
            multiline=True,
            min_lines=3,
            max_lines=6,
            value=str(self.settings.get("system_prompt") or ""),
            on_blur=self._on_field_blur,
            disabled=not self.settings.get("enabled"),
        )

        self.quant_dropdown = ft.Dropdown(
            label="Квант (тег модели из реестра)",
            width=340,
            options=[],
            value=None,
            on_select=self._on_quant_change,
            disabled=not self.settings.get("enabled"),
        )
        self.refresh_tags_button = ft.FilledTonalButton(
            "🔄 Обновить список",
            on_click=self._on_refresh_tags,
            disabled=not self.settings.get("enabled"),
        )
        self.quant_status_text = ft.Text("", size=12, expand=True)
        self.download_button = ft.FilledTonalButton(
            "⬇️ Скачать модель",
            on_click=self._on_download,
            visible=False,
        )
        self.reset_button = ft.OutlinedButton(
            "Сбросить к значениям по умолчанию",
            on_click=self._on_reset,
        )
        self.status_text = ft.Text("", size=12, expand=True)

        self._sync_quant_controls()

    # ------------------------------------------------------------------
    # Компоновка
    # ------------------------------------------------------------------
    def build(self) -> ft.Container:
        """Собирает панель (секция с рамкой во вкладке «Настройки»)."""
        return ft.Container(
            content=ft.Column([
                ft.Text("Оптимизация локальной модели (Ollama)",
                        size=15, weight=ft.FontWeight.BOLD),
                self.enabled_switch,
                ft.Row([
                    self.temperature_field,
                    self.max_tokens_field,
                    self.num_ctx_field,
                ], spacing=12, wrap=True),
                ft.Row([
                    self.top_p_field,
                    self.repeat_penalty_field,
                ], spacing=12, wrap=True),
                self.system_prompt_field,
                ft.Row([
                    self.quant_dropdown,
                    self.refresh_tags_button,
                    self.download_button,
                ], spacing=12, wrap=True),
                self.quant_status_text,
                ft.Row([self.reset_button, self.status_text], spacing=12),
            ], spacing=10),
            border=ft.Border.all(1, ft.Colors.BLUE_GREY_200),
            border_radius=8,
            padding=14,
        )

    def on_show(self) -> None:
        """Перечитывает настройки при открытии вкладки (ядро — источник)."""
        self.settings = self._load_settings()
        self._selected_local = None
        self._sync_fields_from_settings()
        self.update()

    # ------------------------------------------------------------------
    # Загрузка/применение настроек
    # ------------------------------------------------------------------
    def _core_agent(self):
        """Ядро DeepSeekAgent (в legacy-заглушках тестов — сам agent)."""
        return getattr(self.agent, "core", self.agent)

    def _store(self):
        """Хранилище app_state (None — сохранение отключено)."""
        return getattr(self.app, "app_state", None)

    def _load_settings(self) -> Dict[str, Any]:
        return OptimizationSettings.load(self._store())

    def _collect_settings(self) -> Optional[Dict[str, Any]]:
        """Собирает настройки из полей UI.

        Returns:
            Словарь настроек или None при некорректном числе (в status
            выводится ошибка, предыдущие значения не затираются).
        """
        try:
            return {
                "enabled": bool(self.enabled_switch.value),
                "temperature": self._parse_number(
                    self.temperature_field.value, float),
                "max_tokens": self._parse_number(
                    self.max_tokens_field.value, int),
                "num_ctx": self._parse_number(
                    self.num_ctx_field.value, int),
                "top_p": self._parse_number(self.top_p_field.value, float),
                "repeat_penalty": self._parse_number(
                    self.repeat_penalty_field.value, float),
                "system_prompt": str(self.system_prompt_field.value or ""),
                "selected_quant": str(self.settings.get("selected_quant")
                                      or ""),
                "selected_tag": str(self.settings.get("selected_tag") or ""),
            }
        except (TypeError, ValueError):
            self._set_status(
                "Ошибка: параметры должны быть числами", error=True)
            return None

    @staticmethod
    def _parse_number(text: Any, cast: type) -> Any:
        """Число из текста поля; запятая равнозначна точке.

        Пустое поле → 0 (при применении sanitize подставит минимум
        диапазона), «0.» → 0.0 (валидное промежуточное состояние ввода).
        """
        clean = str(text or "").strip().replace(",", ".")
        return cast(clean) if clean else cast(0)

    # ------------------------------------------------------------------
    # Валидация формата полей (без input_filter: тот блокировал точку
    # и откатом правок ломал стирание символов)
    # ------------------------------------------------------------------
    @staticmethod
    def _format_error(text: Any, kind: str) -> str:
        """Текст ошибки формата («» — формат корректен)."""
        value = str(text or "").strip()
        if not value:
            return ""  # пустое поле допустимо: применится минимум диапазона
        pattern = FLOAT_FORMAT if kind == "float" else INT_FORMAT
        if pattern.fullmatch(value):
            return ""
        if kind == "float":
            return "Только число: цифры и одна точка (или запятая)"
        return "Только целое число (без точки)"

    def _kind_of(self, field: Any) -> str:
        """Формат поля (float/int) из реестра числовых полей."""
        for known_field, kind in self._numeric_fields:
            if known_field is field:
                return kind
        return "float"

    def _validate_field_formats(self, only: Any = None) -> bool:
        """Проверяет формат числовых полей, выставляет TextField.error
        (в Flet 1.0 свойства error_text нет — только error).

        Args:
            only: проверить одно поле (живая валидация при вводе);
                None — все (перед применением и после сброса).

        Returns:
            True — все проверенные поля в валидном формате.
        """
        valid = True
        for field, kind in self._numeric_fields:
            if only is not None and field is not only:
                continue
            error = self._format_error(field.value, kind)
            field.error = error or None
            if error:
                valid = False
        return valid

    def _on_field_change(self, event=None) -> None:
        """Живая валидация при вводе: только подсветка (TextField.error),
        без записи настроек — применение происходит на blur."""
        field = getattr(event, "control", None) if event is not None else None
        if field is None:
            # Прямой вызов (тесты, сброс): проверить все поля
            self._validate_field_formats()
            self.update()
            return
        self._validate_field_formats(only=field)
        page = getattr(self.app, "page", None)
        if page is not None:
            try:
                field.update()
            except Exception:  # noqa: BLE001 — подсветка не критична
                pass

    def _on_field_blur(self, event=None) -> None:
        """Потеря фокуса: проверить ВСЕ числовые поля и применить.

        При ошибке формата настройки не применяются — красная рамка и
        статус подсказывают, какое поле исправить.
        """
        if not self._validate_field_formats():
            self._set_status(
                "Ошибка: исправьте выделенные поля — настройки не применены",
                error=True)
            self.update()
            return
        settings = self._collect_settings()
        if settings is not None:
            self._apply(settings)

    def _apply(self, settings: Dict[str, Any]) -> None:
        """Санитизирует, сохраняет в app_state и применяет к ядру."""
        self.settings = OptimizationSettings.sanitize(settings)
        OptimizationSettings.save(self._store(), self.settings)
        try:
            self._core_agent().set_ollama_optimization(self.settings)
        except Exception as exc:  # noqa: BLE001 — legacy-заглушки тестов
            self._set_status(f"Не применено: {exc}", error=True)
            self.update()
            return
        self._set_status(
            "Оптимизация применена" if self.settings.get("enabled")
            else "Оптимизация выключена — дефолты Ollama")
        self.update()

    def _sync_fields_from_settings(self) -> None:
        """Выставляет значения полей из self.settings (сброс/загрузка)."""
        self.enabled_switch.value = bool(self.settings.get("enabled"))
        self.temperature_field.value = str(
            self.settings.get("temperature", ""))
        self.max_tokens_field.value = str(
            self.settings.get("max_tokens", ""))
        self.num_ctx_field.value = str(self.settings.get("num_ctx", ""))
        self.top_p_field.value = str(self.settings.get("top_p", ""))
        self.repeat_penalty_field.value = str(
            self.settings.get("repeat_penalty", ""))
        self.system_prompt_field.value = str(
            self.settings.get("system_prompt") or "")
        self._update_fields_state()
        # Ошибки формата сбрасываются: значения из настроек валидны
        self._validate_field_formats()
        self._sync_quant_controls()

    def _update_fields_state(self) -> None:
        """Снятый переключатель блокирует все поля (дефолты Ollama)."""
        disabled = not bool(self.settings.get("enabled"))
        for field in (self.temperature_field, self.max_tokens_field,
                      self.num_ctx_field, self.top_p_field,
                      self.repeat_penalty_field, self.system_prompt_field,
                      self.quant_dropdown, self.refresh_tags_button):
            field.disabled = disabled

    # ------------------------------------------------------------------
    # Обработчики
    # ------------------------------------------------------------------
    def _on_toggle(self, event=None) -> None:
        # Подсветить кривые форматы и при переключателе: пользователь мог
        # испортить поле и переключить свитч, не выйдя из поля (blur не был)
        self._validate_field_formats()
        settings = self._collect_settings() or dict(self.settings)
        settings["enabled"] = bool(
            getattr(event.control, "value", None)
            if event is not None else self.enabled_switch.value)
        self._apply(settings)
        self._update_fields_state()
        self.update()

    def _on_reset(self, event=None) -> None:
        self.settings = OptimizationSettings.reset()
        self._sync_fields_from_settings()
        self._apply(dict(self.settings))

    def _on_quant_change(self, event=None) -> None:
        """Выбор кванта: запоминаем тег, показываем статус локальности."""
        value = getattr(event.control, "value", None) if event is not None \
            else self.quant_dropdown.value
        if not value:
            return
        meta = self._tag_meta.get(str(value)) or {}
        settings = dict(self.settings)
        settings["selected_tag"] = str(value)
        settings["selected_quant"] = str(meta.get("quant") or "")
        self._selected_local = bool(meta.get("local"))
        self._apply(settings)
        self._sync_quant_controls()

    def _on_refresh_tags(self, event=None) -> None:
        self._schedule(self._refresh_tags_async)

    def _on_download(self, event=None) -> None:
        if not self.settings.get("selected_tag"):
            self._set_status("Сначала выберите квант (тег) из списка",
                             error=True)
            self.update()
            return
        self._schedule(self._download_async)

    # ------------------------------------------------------------------
    # Кванты: список тегов, локальность, загрузка
    # ------------------------------------------------------------------
    def _ollama_model_name(self) -> str:
        """Модель для списка тегов: активная локальная или из фабрики."""
        provider = getattr(self._core_agent(), "provider", None)
        model = getattr(provider, "model", "") if getattr(
            provider, "local", False) else ""
        if model:
            return str(model)
        from src.llm_providers import LLMProviderFactory
        return str(LLMProviderFactory.PROVIDERS["ollama_qwen"]["model"])

    def _sync_quant_controls(self) -> None:
        """Статус кванта и видимость кнопки загрузки (по данным панели)."""
        tag = str(self.settings.get("selected_tag") or "")
        if not tag:
            self.quant_status_text.value = (
                "Список квантов не загружен — нажмите «Обновить список».")
            self.download_button.visible = False
            return
        is_local = bool(self._selected_local)
        self.download_button.visible = not is_local
        quant = str(self.settings.get("selected_quant") or "")
        label = f"{quant} — {tag}" if quant else tag
        self.quant_status_text.value = (
            f"✅ Скачан: {label}" if is_local else f"⬇️ Не скачан: {label}")

    async def _refresh_tags_async(self) -> None:
        """Список тегов реестра + отметки локальности (в потоке)."""
        model_name = self._ollama_model_name()
        self._set_status(f"Загрузка списка квантов для {model_name}…")
        self.update()
        try:
            tags = await asyncio.to_thread(
                OllamaTagManager.get_available_tags, model_name)
        except Exception as exc:  # noqa: BLE001 — гард уже страхует
            tags = []
            self._set_status(f"Ошибка получения тегов: {exc}", error=True)
            self.update()
            return
        if not tags:
            self._set_status(
                "Теги не найдены (реестр недоступен, модель неизвестна)",
                error=True)
            self.update()
            return
        selected_tag = str(self.settings.get("selected_tag") or "")
        options: List[ft.DropdownOption] = []
        self._tag_meta = {
            str(item.get("tag") or ""): {
                "quant": str(item.get("quant") or ""),
                "local": bool(item.get("local")),
            }
            for item in tags if item.get("tag")
        }
        for item in tags:
            tag = str(item.get("tag") or "")
            quant = str(item.get("quant") or "")
            label = f"{quant} — {tag}" if quant else tag
            if item.get("local"):
                label = f"{label} · ✅ скачан"
            options.append(ft.DropdownOption(
                key=tag, content=ft.Text(label, size=13)))
        self.quant_dropdown.options = options
        if selected_tag and selected_tag not in self._tag_meta:
            # сохранённый тег пропал из реестра — сбрасываем выбор
            selected_tag = ""
        self.quant_dropdown.value = selected_tag or None
        self._selected_local = bool(
            self._tag_meta.get(selected_tag, {}).get("local"))
        self._sync_quant_controls()
        found = sum(1 for item in tags if item.get("local"))
        self._set_status(
            f"Теги {model_name}: {len(tags)} "
            f"(локально скачано: {found})")
        self.update()

    async def _download_async(self) -> None:
        """Загрузка выбранного тега (ollama pull, в потоке)."""
        tag = str(self.settings.get("selected_tag") or "")
        self._set_status(f"Загрузка модели {tag}… (может занять время)")
        self.update()
        try:
            ok, message = await asyncio.to_thread(
                OllamaTagManager.pull_model, tag)
        except Exception as exc:  # noqa: BLE001 — гард уже страхует
            ok, message = False, str(exc)
        self._selected_local = bool(ok)
        self._sync_quant_controls()
        self._set_status(
            ("✅ " if ok else "⚠️ ") + str(message), error=not ok)
        self.update()

    # ------------------------------------------------------------------
    # Служебное
    # ------------------------------------------------------------------
    def _set_status(self, text: str, error: bool = False) -> None:
        self.status_text.value = text
        self.status_text.color = (
            ft.Colors.RED_600 if error else ft.Colors.BLUE_GREY_500
        )

    def _schedule(self, handler, *args) -> None:
        """Запускает async-задачу: КОРУТИННУЮ ФУНКЦИЮ в page.run_task
        (готовая корутина даёт TypeError и error-overlay Flet). Без
        page (офлайн-тесты) — no-op: вызывающий код запускает сам."""
        page = getattr(self.app, "page", None)
        if page is None:
            return
        page.run_task(components.guarded_task(self.app, handler), *args)

    def update(self) -> None:
        page = getattr(self.app, "page", None)
        if page is not None:
            page.update()
