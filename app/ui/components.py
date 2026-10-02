"""Общие компоненты Flet-интерфейса: сообщения чата, панели источников,
гард фоновых задач.

Чистая сборка контролов без обращения к page — вкладки тестируются
офлайн (контролы Flet 1.0 — обычные dataclass-объекты).
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional

import flet as ft

from src.utils import get_logger

logger = get_logger("deepseek-chat.flet.ui")

# Варианты оценки на вкладке «Сравнение» (порядок = порядок в сводке)
ASSESSMENT_OPTIONS = ("Хорошо", "Частично", "Плохо")

PREVIEW_TOOL_RESULT = 220


def guarded_task(app: Any, handler) -> Any:
    """Обёртка async-задачи вкладки: исключение не «роняет» интерфейс.

    Без обёртки неперехваченное исключение в async-обработчике Flet
    показывает полноэкранный error-overlay — вернуться на другие
    вкладки нельзя. Гард пишет полный traceback в лог (терминал) и
    показывает пользователю SnackBar с текстом ошибки.

    Args:
        app: владелец вкладок (используется app.notify).
        handler: корутинная функция задачи.

    Returns:
        Корутинная функция-обёртку (для page.run_task).
    """

    async def wrapper(*args) -> None:
        try:
            await handler(*args)
        except Exception as exc:  # noqa: BLE001 — гард по определению широкий
            logger.error("Ошибка в %s:\n%s",
                         getattr(handler, "__qualname__", handler),
                         traceback.format_exc())
            try:
                app.notify(f"⚠️ Ошибка: {exc} (подробности — в логе)")
            except Exception:
                logger.error("Не удалось показать SnackBar об ошибке")

    return wrapper


def truncate(text: Optional[str], limit: int = PREVIEW_TOOL_RESULT) -> str:
    """Обрезает длинный текст для панели инструментов."""
    value = str(text or "")
    if len(value) > limit:
        return value[: limit - 1] + "…"
    return value


# Дизайн чата: тёмные пузыри с явным светлым текстом — контраст не
# зависит от светлой/тёмной темы окна (по умолчанию Flet берёт тему
# системы, и «светлый текст на светлом пузыре» делал сообщения
# нечитаемыми).
USER_BUBBLE_COLOR = ft.Colors.BLUE_800
ASSISTANT_BUBBLE_COLOR = ft.Colors.BLUE_GREY_800
USER_HEADER_COLOR = ft.Colors.BLUE_200
ASSISTANT_HEADER_COLOR = ft.Colors.BLUE_GREY_200
MESSAGE_TEXT_COLOR = ft.Colors.GREY_50
ERROR_TEXT_COLOR = ft.Colors.RED_200
RAG_WARNING_COLOR = ft.Colors.ORANGE_300


def _message_style_sheet(color: str) -> ft.MarkdownStyleSheet:
    """Явные цвета текста Markdown (абзацы, заголовки, списки, код)."""
    style = ft.TextStyle(color=color)
    return ft.MarkdownStyleSheet(
        p_text_style=style,
        h1_text_style=style,
        h2_text_style=style,
        h3_text_style=style,
        h4_text_style=style,
        h5_text_style=style,
        h6_text_style=style,
        em_text_style=style,
        strong_text_style=style,
        del_text_style=style,
        list_bullet_text_style=style,
        blockquote_text_style=style,
        code_text_style=style,
    )


def markdown_text(value: str, color: Optional[str] = None) -> ft.Markdown:
    """Selectable Markdown (копирование из чата) с переносом строк.

    Args:
        value: текст сообщения (Markdown).
        color: явный цвет текста — задавайте внутри цветных контейнеров
            (пузыри чата), где цвет темы может слиться с фоном; None —
            цвет темы (для текста на поверхности страницы/карточек).
    """
    return ft.Markdown(
        value or "",
        selectable=True,
        extension_set=ft.MarkdownExtensionSet.GITHUB_FLAVORED,
        soft_line_break=True,
        md_style_sheet=_message_style_sheet(color) if color else None,
    )


def message_bubble(text: str, role: str, timestamp: str = "") -> ft.Container:
    """Пузырь сообщения чата (пользователь — справа, ассистент — слева).

    Тёмный контрастный фон + светлый текст (константы выше) — читаемо
    и в светлой, и в тёмной теме окна.

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
        color=USER_HEADER_COLOR if is_user else ASSISTANT_HEADER_COLOR,
    )
    body = ft.Column(
        [header, markdown_text(text, MESSAGE_TEXT_COLOR)],
        spacing=2, tight=True,
    )
    return ft.Container(
        content=body,
        bgcolor=USER_BUBBLE_COLOR if is_user else ASSISTANT_BUBBLE_COLOR,
        border_radius=10,
        padding=10,
        margin=ft.Margin(left=40 if is_user else 0,
                         top=0, right=0 if is_user else 40, bottom=0),
    )


def source_line(source: Dict[str, Any]) -> ft.Text:
    """Строка источника (формат ТЗ ai_1_24):
    📄 installation.md → ## Windows (chunk_id: 12 · structure · d=0.12).

    Светлый цвет — панель лежит на тёмном пузыре сообщения."""
    path = source.get("source") or "?"
    section = source.get("section") or "—"
    chunk_id = source.get("chunk_id")
    suffix = f"chunk_id: {chunk_id}" if chunk_id is not None else ""
    strategy = source.get("strategy") or "fixed"
    distance = source.get("distance")
    if strategy:
        suffix += (", " if suffix else "") + str(strategy)
    if distance is not None:
        suffix += f" · d={distance}"
    suffix = f" ({suffix})" if suffix else ""
    return ft.Text(
        f"📄 {path} → {section}{suffix}",
        size=12,
        color=ft.Colors.BLUE_200,
        selectable=True,
    )


def source_preview(source: Dict[str, Any]) -> ft.Text:
    """Фрагмент текста чанка в кавычках (светлый — на тёмном пузыре)."""
    preview = source.get("preview") or ""
    return ft.Text(
        f"   «{preview}»",
        size=11,
        italic=True,
        color=ft.Colors.GREY_300,
        selectable=True,
    )


def source_quote(quote: Dict[str, Any]) -> ft.Text:
    """Цитата LLM из чанка: «…» + пометка ⚠️ для недословной (ТЗ ai_1_23:
    цитаты не выбрасываются — неверная помечается valid: False)."""
    mark = "" if quote.get("valid", True) else " ⚠️ не дословная"
    return ft.Text(
        f"   «{quote.get('text') or ''}»{mark}",
        size=11,
        italic=True,
        color=ft.Colors.BLUE_100 if quote.get("valid", True)
        else ft.Colors.ORANGE_200,
        selectable=True,
    )


def sources_panel(sources: List[Dict[str, Any]],
                  quotes: Optional[List[Dict[str, Any]]] = None
                  ) -> Optional[ft.ExpansionTile]:
    """Сворачиваемая панель «Источники» под ответом LLM (свёрнута).

    С цитатами (ТЗ ai_1_24): под каждым источником показываются
    дословные цитаты LLM из этого чанка (invalid — с пометкой ⚠️);
    без цитат — прежний фрагмент превью чанка.

    Args:
        sources: список метаданных чанков (Retriever.format_sources).
        quotes: валидированные цитаты LLM (validate_quotes) или None.

    Returns:
        ExpansionTile или None, если источников нет.
    """
    if not sources:
        return None
    quotes_by_chunk: Dict[Any, List[Dict[str, Any]]] = {}
    for quote in quotes or []:
        quotes_by_chunk.setdefault(quote.get("chunk_id"), []).append(quote)
    rows: List[ft.Control] = []
    for source in sources:
        rows.append(source_line(source))
        chunk_quotes = quotes_by_chunk.get(source.get("chunk_id"))
        if chunk_quotes:
            rows.extend(source_quote(quote) for quote in chunk_quotes)
        else:
            rows.append(source_preview(source))
    return ft.ExpansionTile(
        title=ft.Text(f"📚 Источники ({len(sources)})", size=13,
                      weight=ft.FontWeight.BOLD,
                      color=ft.Colors.BLUE_200),
        controls=rows,
        expanded=False,
        tile_padding=6,
        icon_color=ft.Colors.BLUE_200,
        collapsed_icon_color=ft.Colors.BLUE_200,
        text_color=ft.Colors.BLUE_100,
        collapsed_text_color=ft.Colors.BLUE_200,
    )


def tools_panel(tool_calls: List[Dict[str, Any]]) -> Optional[ft.ExpansionTile]:
    """Панель выполненных MCP-инструментов под ответом (напоминания,
    новости, гео/погода — функциональность agentic loop из src).
    Светлые цвета — на тёмном пузыре сообщения."""
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
            color=ft.Colors.TEAL_200,
        ))
    return ft.ExpansionTile(
        title=ft.Text(f"🛠 Инструменты ({len(tool_calls)})", size=13,
                      weight=ft.FontWeight.BOLD,
                      color=ft.Colors.TEAL_200),
        controls=rows,
        expanded=False,
        tile_padding=6,
        icon_color=ft.Colors.TEAL_200,
        collapsed_icon_color=ft.Colors.TEAL_200,
        text_color=ft.Colors.TEAL_100,
        collapsed_text_color=ft.Colors.TEAL_200,
    )


def file_open_button(file_path: str, on_open) -> ft.FilledTonalButton:
    """Кнопка «📂 Открыть файл» (save_to_file news-сервера)."""
    return ft.FilledTonalButton(
        f"📂 Открыть файл: {file_path}",
        icon=ft.Icons.FOLDER_OPEN,
        on_click=on_open,
        tooltip=file_path,
    )


def task_state_panel(state: Dict[str, Any]) -> ft.Column:
    """Панель памяти задачи для модалки «Состояние диалога» (ТЗ ai_1_24:
    🎯 цель, 📝 уточнения, ⚠️ ограничения, 📚 термины).

    Args:
        state: словарь TaskState.to_dict() (goal/clarifications/
            constraints/terms).

    Returns:
        Готовая Column (без обращения к page — офлайн-тесты).
    """
    rows: List[ft.Control] = []
    goal = str(state.get("goal") or "").strip()
    if goal:
        rows.append(ft.Text(f"🎯 Цель: {goal}", size=14, selectable=True))
    for title, key, icon in (
        ("📝 Уточнения:", "clarifications", "  • "),
        ("⚠️ Ограничения:", "constraints", "  • "),
    ):
        items = [str(item) for item in state.get(key) or [] if str(item).strip()]
        if items:
            rows.append(ft.Text(title, size=13, weight=ft.FontWeight.BOLD))
            rows.extend(ft.Text(f"{icon}{item}", size=13, selectable=True)
                        for item in items)
    terms = state.get("terms") or {}
    if terms:
        rows.append(ft.Text("📚 Термины:", size=13,
                            weight=ft.FontWeight.BOLD))
        rows.extend(
            ft.Text(f"  • {term}: {definition}", size=13, selectable=True)
            for term, definition in terms.items())
    if not rows:
        rows.append(ft.Text("Состояние диалога пусто.", size=13,
                            color=ft.Colors.BLUE_GREY_400))
    return ft.Column(rows, spacing=4, tight=True, scroll=ft.ScrollMode.AUTO)


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
