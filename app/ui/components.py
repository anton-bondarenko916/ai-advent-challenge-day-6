"""Общие компоненты Flet-интерфейса: сообщения чата, панели источников,
гард фоновых задач.

Чистая сборка контролов без обращения к page — вкладки тестируются
офлайн (контролы Flet 1.0 — обычные dataclass-объекты).
"""

from __future__ import annotations

import json
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


# Иконки инструментов лога шагов (ТЗ ai_1_35, Модуль 3): fs-сервер +
# узнаваемые инструменты других серверов; неизвестный — 🔧
TOOL_ICONS: Dict[str, str] = {
    "search_within_files": "🔍",
    "search_files": "🔍",
    "read_file": "📄",
    "read_multiple_files": "📚",
    "write_file": "✍️",
    "list_directory": "📁",
    "tree": "🌳",
    "add_reminder": "⏰",
    "get_user_city": "📍",
    "get_weather": "⛅",
    "search_news": "📰",
    "get_user_context": "👤",
    "list_users": "👥",
}


def tool_icon(name: str) -> str:
    """Иконка инструмента для лога шагов (неизвестный — 🔧)."""
    return TOOL_ICONS.get(str(name or ""), "🔧")


def tools_panel(tool_calls: List[Dict[str, Any]]) -> Optional[ft.ExpansionTile]:
    """Панель выполненных MCP-инструментов под ответом — лог шагов
    agentic loop (напоминания, новости, гео/погода, файловая система).
    Каждый шаг: номер, иконка, имя, аргументы, обрезанный результат.
    Скролл с ограничением высоты: длинный лог не растягивает чат
    (ТЗ ai_1_35). Светлые цвета — на тёмном пузыре сообщения."""
    if not tool_calls:
        return None
    rows: List[ft.Control] = []
    for step, call in enumerate(tool_calls, 1):
        name = call.get("name") or "?"
        arguments = call.get("arguments") or {}
        result = truncate(str(call.get("result")))
        rows.append(ft.Text(
            f"Шаг {step}: {tool_icon(name)} {name} {arguments}\n"
            f"   → {result}",
            size=12,
            selectable=True,
            color=ft.Colors.TEAL_200,
        ))
    # Высота панели растёт с числом шагов, но ограничена: дальше — скролл
    panel_height = min(340.0, 40.0 + 64.0 * len(rows))
    return ft.ExpansionTile(
        title=ft.Text(f"🛠 Инструменты ({len(tool_calls)})", size=13,
                      weight=ft.FontWeight.BOLD,
                      color=ft.Colors.TEAL_200),
        controls=[ft.Column(
            rows,
            scroll=ft.ScrollMode.AUTO,
            height=panel_height,
            spacing=6,
        )],
        expanded=False,
        tile_padding=6,
        icon_color=ft.Colors.TEAL_200,
        collapsed_icon_color=ft.Colors.TEAL_200,
        text_color=ft.Colors.TEAL_100,
        collapsed_text_color=ft.Colors.TEAL_200,
    )


# ----------------------------------------------------------------------
# Сводка файловых операций в самом сообщении (ТЗ ai_1_35)
# ----------------------------------------------------------------------
FS_SUMMARY_MAX_HITS = 8    # записей search_within_files в сообщении
FS_SUMMARY_MAX_FILES = 10  # путей search_files в сообщении
FS_SUMMARY_MAX_BLOCKS = 3  # поисковых блоков в сообщении (модель может
                           # искать несколько раз — поток блоков топит ответ)
FS_SUMMARY_SNIPPET = 90    # символов текста найденной строки

# Цвета сводки (светлые — на тёмном пузыре сообщения)
FS_SEARCH_COLOR = ft.Colors.TEAL_200
FS_CREATED_COLOR = ft.Colors.GREEN_300


def parse_tool_result_json(text: Any) -> Any:
    """Значение из JSON-результата MCP-инструмента.

    FastMCP оборачивает значение в ``{"result": …}``; принимается и
    «сырой» JSON (тесты вызывают инструменты напрямую). Не-JSON — None.
    """
    try:
        payload = json.loads(str(text or ""))
    except (TypeError, ValueError):
        return None
    if isinstance(payload, dict) and set(payload) == {"result"}:
        payload = payload["result"]
    return payload


def _is_error_payload(payload: Any) -> bool:
    """True, если результат инструмента — ошибка ({"error": …})."""
    if isinstance(payload, dict):
        return "error" in payload
    if isinstance(payload, list) and payload:
        first = payload[0]
        return isinstance(first, dict) and "error" in first
    if isinstance(payload, str):
        return payload.lstrip().startswith('{"error"')
    return False


def fs_summary_controls(tool_calls: List[Dict[str, Any]]) -> List[ft.Control]:
    """Сводка файловых операций в ТЕЛЕ сообщения (шаги — в панели ниже).

    Поиск (search_within_files/search_files): что искали и что нашлось —
    файлы, строки, фрагменты. Создание (write_file): путь сохранённого
    файла. Ошибки инструментов в сводку не попадают — они видны в логе
    шагов. Не-fs инструменты игнорируются. Поисковых блоков — не больше
    FS_SUMMARY_MAX_BLOCKS (модель может искать несколько раз подряд,
    поток блоков топит ответ).
    """
    controls: List[ft.Control] = []
    search_blocks = 0
    search_calls = sum(
        1 for call in (tool_calls or [])
        if str(call.get("name") or "") in ("search_within_files",
                                           "search_files"))
    for call in tool_calls or []:
        name = str(call.get("name") or "")
        if name in ("search_within_files", "search_files"):
            if search_blocks >= FS_SUMMARY_MAX_BLOCKS:
                continue
            block = (_search_within_summary(call)
                     if name == "search_within_files"
                     else _search_files_summary(call))
            if block:
                search_blocks += 1
                if (search_blocks == FS_SUMMARY_MAX_BLOCKS
                        and search_calls > FS_SUMMARY_MAX_BLOCKS):
                    block.append(_summary_text(
                        f"… ещё {search_calls - FS_SUMMARY_MAX_BLOCKS} "
                        f"поисков — полные результаты в шагах ниже",
                        FS_SEARCH_COLOR))
            controls.extend(block)
        elif name == "write_file":
            controls.extend(_write_file_summary(call))
    return controls


def _summary_text(text: str, color: Any) -> ft.Text:
    """Абзац сводки (выделаемый, размер как у заметок под ответом)."""
    return ft.Text(text, size=12, selectable=True, color=color)


def _plural(count: int, one: str, few: str, many: str) -> str:
    """Русская форма существительного по числу (1 вхождение, 2 входжения,
    5 вхождений)."""
    n = abs(int(count)) % 100
    if 11 <= n <= 19:
        return many
    digit = n % 10
    if digit == 1:
        return one
    if 2 <= digit <= 4:
        return few
    return many


def _search_within_summary(call: Dict[str, Any]) -> List[ft.Control]:
    """«🔍 Поиск «X»: N вхождений в M файлах» + список находок."""
    hits = parse_tool_result_json(call.get("result"))
    if not isinstance(hits, list) or not hits or _is_error_payload(hits):
        return []
    entries = [hit for hit in hits if isinstance(hit, dict) and hit.get("file")]
    if not entries:
        return []

    arguments = call.get("arguments") or {}
    needle = str(arguments.get("substring") or "").strip()
    directory = str(arguments.get("directory") or "").strip()
    where = f" (в {directory})" if directory not in ("", ".") else ""
    files = {str(hit.get("file")) for hit in entries}
    header = (f"🔍 Поиск «{needle}»{where}: найдено {len(entries)} "
              f"{_plural(len(entries), 'вхождение', 'вхождения', 'вхождений')} "
              f"в {len(files)} "
              f"{_plural(len(files), 'файле', 'файлах', 'файлах')}")
    lines = [header]
    for hit in entries[:FS_SUMMARY_MAX_HITS]:
        snippet = str(hit.get("content") or "")[:FS_SUMMARY_SNIPPET]
        lines.append(f"   • {hit.get('file')}:{hit.get('line')} — {snippet}")
    hidden = len(entries) - FS_SUMMARY_MAX_HITS
    if hidden > 0:
        lines.append(f"   … ещё {hidden} "
                     f"{_plural(hidden, 'вхождение', 'вхождения', 'вхождений')} "
                     f"(полный список — в шагах)")
    return [_summary_text("\n".join(lines), FS_SEARCH_COLOR)]


def _search_files_summary(call: Dict[str, Any]) -> List[ft.Control]:
    """«🔍 Файлы по шаблону «X»: N» + список путей."""
    paths = parse_tool_result_json(call.get("result"))
    if not isinstance(paths, list) or not paths or _is_error_payload(paths):
        return []
    entries = [str(path) for path in paths if str(path)]
    if not entries:
        return []

    pattern = str((call.get("arguments") or {}).get("pattern") or "*")
    header = (f"🔍 Файлы по шаблону «{pattern}»: "
              f"найдено {len(entries)}")
    lines = [header]
    lines.extend(f"   • {path}" for path in entries[:FS_SUMMARY_MAX_FILES])
    hidden = len(entries) - FS_SUMMARY_MAX_FILES
    if hidden > 0:
        lines.append(f"   … ещё {hidden} "
                     f"{_plural(hidden, 'файл', 'файла', 'файлов')} "
                     f"(полный список — в шагах)")
    return [_summary_text("\n".join(lines), FS_SEARCH_COLOR)]


def _write_file_summary(call: Dict[str, Any]) -> List[ft.Control]:
    """«✍️ Создан файл: <полный путь>» (путь — куда реально сохранился)."""
    message = parse_tool_result_json(call.get("result"))
    if not isinstance(message, str):
        return []
    marker = "Файл создан:"
    if not message.startswith(marker):
        return []  # ошибка записи — только в логе шагов
    path = message[len(marker):].strip().strip('"')
    if not path:
        return []
    return [_summary_text(f"✍️ Создан файл: {path}", FS_CREATED_COLOR)]


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
