"""Общие компоненты Flet-интерфейса: сообщения чата, панели источников.

Чистая сборка контролов без обращения к page — вкладки тестируются
офлайн (контролы Flet 1.0 — обычные dataclass-объекты).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import flet as ft

# Варианты оценки на вкладке «Сравнение» (порядок = порядок в сводке)
ASSESSMENT_OPTIONS = ("Хорошо", "Частично", "Плохо")

PREVIEW_TOOL_RESULT = 220


def truncate(text: Optional[str], limit: int = PREVIEW_TOOL_RESULT) -> str:
    """Обрезает длинный текст для панели инструментов."""
    value = str(text or "")
    if len(value) > limit:
        return value[: limit - 1] + "…"
    return value


def markdown_text(value: str) -> ft.Markdown:
    """Selectable Markdown (копирование из чата) с переносом строк."""
    return ft.Markdown(
        value or "",
        selectable=True,
        extension_set=ft.MarkdownExtensionSet.GITHUB_FLAVORED,
        soft_line_break=True,
    )


def message_bubble(text: str, role: str, timestamp: str = "") -> ft.Container:
    """Пузырь сообщения чата (пользователь — справа, ассистент — слева).

    Args:
        text: текст сообщения (Markdown).
        role: "user" | "assistant".
        timestamp: время ответа (опционально).

    Returns:
        Готовый Container для ListView.
    """
    is_user = role == "user"
    header_role = "Вы" if is_user else "Ассистент"
    header = ft.Text(
        f"{header_role}{' · ' + timestamp if timestamp else ''}",
        size=11,
        weight=ft.FontWeight.BOLD,
        color=ft.Colors.BLUE_GREY_400,
    )
    body = ft.Column([header, markdown_text(text)], spacing=2, tight=True)
    return ft.Container(
        content=body,
        bgcolor=ft.Colors.BLUE_50 if is_user else ft.Colors.GREY_100,
        border_radius=10,
        padding=10,
        margin=ft.Margin(left=40 if is_user else 0,
                         top=0, right=0 if is_user else 40, bottom=0),
    )


def source_line(source: Dict[str, Any]) -> ft.Text:
    """Строка источника: 📄 installation.md → ## Windows (structure, d=0.12)."""
    path = source.get("source") or "?"
    section = source.get("section") or "—"
    strategy = source.get("strategy") or "fixed"
    distance = source.get("distance")
    suffix = f" · {strategy}" + (f" · d={distance}" if distance is not None else "")
    return ft.Text(
        f"📄 {path} → {section}{suffix}",
        size=12,
        color=ft.Colors.BLUE_GREY_700,
        selectable=True,
    )


def source_preview(source: Dict[str, Any]) -> ft.Text:
    """Фрагмент текста чанка в кавычках."""
    preview = source.get("preview") or ""
    return ft.Text(
        f"   «{preview}»",
        size=11,
        italic=True,
        color=ft.Colors.GREY_700,
        selectable=True,
    )


def sources_panel(sources: List[Dict[str, Any]]) -> Optional[ft.ExpansionTile]:
    """Сворачиваемая панель «Источники» под ответом LLM.

    Args:
        sources: список метаданных чанков (Retriever.format_sources).

    Returns:
        ExpansionTile или None, если источников нет.
    """
    if not sources:
        return None
    rows: List[ft.Control] = []
    for source in sources:
        rows.append(source_line(source))
        rows.append(source_preview(source))
    return ft.ExpansionTile(
        title=ft.Text(f"📚 Источники ({len(sources)})", size=13,
                      weight=ft.FontWeight.BOLD),
        controls=rows,
        expanded=False,
        tile_padding=6,
    )


def tools_panel(tool_calls: List[Dict[str, Any]]) -> Optional[ft.ExpansionTile]:
    """Панель выполненных MCP-инструментов под ответом (напоминания,
    новости, гео/погода — функциональность agentic loop из src)."""
    if not tool_calls:
        return None
    rows: List[ft.Control] = []
    for call in tool_calls:
        name = call.get("name") or "?"
        arguments = call.get("arguments") or {}
        result = truncate(str(call.get("result")))
        rows.append(ft.Text(
            f"🛠 {name} {arguments}\n   → {result}",
            size=12,
            selectable=True,
            color=ft.Colors.TEAL_800,
        ))
    return ft.ExpansionTile(
        title=ft.Text(f"🛠 Инструменты ({len(tool_calls)})", size=13,
                      weight=ft.FontWeight.BOLD),
        controls=rows,
        expanded=False,
        tile_padding=6,
    )


def file_open_button(file_path: str, on_open) -> ft.FilledTonalButton:
    """Кнопка «📂 Открыть файл» (save_to_file news-сервера)."""
    return ft.FilledTonalButton(
        f"📂 Открыть файл: {file_path}",
        icon=ft.Icons.FOLDER_OPEN,
        on_click=on_open,
        tooltip=file_path,
    )


def assessment_dropdown(value: str = "", on_select=None) -> ft.Dropdown:
    """Dropdown оценки (Хорошо/Частично/Плохо) для вкладки «Сравнение»."""
    return ft.Dropdown(
        value=value or None,
        options=[
            ft.DropdownOption(key=option, content=ft.Text(option, size=12))
            for option in ASSESSMENT_OPTIONS
        ],
        width=140,
        text_size=12,
        on_select=on_select,
        hint_text="не оценено",
    )


def aggregate_assessments(
    values: List[Optional[str]],
) -> Dict[str, Optional[float]]:
    """Агрегирует оценки в проценты (статистика вкладки «Сравнение»).

    Args:
        values: оценки ("Хорошо"/"Частично"/"Плохо") или None.

    Returns:
        {"Хорошо": 70.0, "Частично": 20.0, "Плохо": 10.0, "Оценено": 10}
        — проценты считаются от числа выставленных оценок; при нуле
        оценок проценты None.
    """
    total = sum(1 for value in values if value in ASSESSMENT_OPTIONS)
    result: Dict[str, Optional[float]] = {
        option: None for option in ASSESSMENT_OPTIONS
    }
    result["Оценено"] = total
    if not total:
        return result
    for option in ASSESSMENT_OPTIONS:
        count = sum(1 for value in values if value == option)
        result[option] = round(100.0 * count / total, 1)
    return result


def format_summary(label: str, stats: Dict[str, Optional[float]]) -> str:
    """Строка сводки по ТЗ: «Без RAG: 70% хорошо, 20% частично, 10% плохо»."""
    if not stats.get("Оценено"):
        return f"{label}: нет оценок"
    parts = []
    for option in ASSESSMENT_OPTIONS:
        percent = stats.get(option)
        parts.append(f"{percent:.0f}% {option.lower()}")
    return f"{label}: " + ", ".join(parts)
