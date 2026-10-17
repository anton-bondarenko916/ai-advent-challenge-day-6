"""PDF-конспект темы через WeasyPrint (ТЗ ai_1_36, Модуль 5).

Конспект собирается как HTML+CSS (заголовок, описание, теория и
задание каждой подтемы, фидбек AI по выполненным) и рендерится
локально WeasyPrint'ом — без внешних сервисов.

WeasyPrint — ОПЦИОНАЛЬНАЯ зависимость: на Windows ему нужен GTK
runtime с Pango ≥ 1.44 (актуальный: «GTK3-Runtime Win64» —
github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer;
пакет winget GTK.GTK3 больше не существует, а GtkD.GtkPlusRuntime
несёт Pango 1.43 — слишком старый). Модуль импортируется и без
него; при попытке рендера без установленного пакета/библиотек
поднимается :class:`PDFError` с инструкцией по установке.

Каталоги стандартных установок GTK (:func:`ensure_gtk_on_path`)
добавляются в PATH ТЕКУЩЕГО процесса перед импортом: установленный
рантайм подхватывается без перезапуска приложения (PATH процесса
иначе застывает на момент старта — прецедент: error 0x7e сразу
после установки рантайма). Тесты подставляют свой renderer.
"""

from __future__ import annotations

import html as html_module
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.utils import get_logger

logger = get_logger("deepseek-chat.learning.pdf")

WEASYPRINT_HINT = (
    "WeasyPrint не смог загрузить GTK (нужен Pango ≥ 1.44). Установите "
    "GTK3-Runtime Win64: github.com/tschoonj/GTK-for-Windows-Runtime-"
    "Environment-Installer/releases (файл gtk3-runtime-*-ts-win64.exe), "
    "затем перезапустите приложение. Если рантайм установлен в "
    "нестандартный каталог — укажите его в переменной окружения "
    "GTK_RUNTIME_BIN.")

PANGO_TOO_OLD_HINT = (
    "GTK-рантайм найден, но Pango в нём старше 1.44 — WeasyPrint с ним "
    "не работает. Установите актуальный GTK3-Runtime Win64 "
    "(github.com/tschoonj/GTK-for-Windows-Runtime-Environment-"
    "Installer) и перезапустите приложение.")

# DLL, по наличию которой распознаётся каталог GTK-рантайма
GTK_MARKER_DLL = "libgobject-2.0-0.dll"

PDF_CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 2cm; color: #222; }
h1 { color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 6px; }
h2 { color: #333; margin-top: 1.2cm; }
h3 { color: #444; }
.meta { color: #666; font-size: 10pt; }
.theory { background: #f5f5f5; padding: 12px; border-radius: 6px;
          margin: 8px 0; }
.task { background: #fff3e0; padding: 12px; border-left: 4px solid #ff9800;
        margin: 8px 0; }
.rubric { background: #e8f0fe; padding: 12px; border-left: 4px solid #1a73e8;
          margin: 8px 0; }
.feedback { background: #e6f4ea; padding: 12px; border-left: 4px solid
            #34a853; margin: 8px 0; font-size: 10pt; }
code { background: #eee; padding: 2px 5px; font-family: Consolas, monospace; }
pre { background: #f0f0f0; padding: 8px; white-space: pre-wrap;
      font-family: Consolas, monospace; }
"""

Renderer = Callable[[str, Path], None]


def _escape_markdown(text: str) -> str:
    """Markdown-текст (теория/задание) → безопасный HTML-фрагмент.

    Экранирование HTML обязательно (текст приходит от LLM), переносы
    строк и ```-блоки кода сохраняются простыми преобразованиями —
    для конспекта этого достаточно.
    """
    value = str(text or "")
    parts: List[str] = []
    # ```-блоки — как <pre>, остальное — абзацы с <br>
    chunks = re.split(r"```[a-zA-Z0-9_-]*\n?", value)
    for index, chunk in enumerate(chunks):
        escaped = html_module.escape(chunk)
        parts.append(
            f"<pre>{escaped}</pre>" if index % 2 else escaped.replace(
                "\n", "<br>"))
    return "".join(parts)


def render_topic_html(topic: Dict[str, Any],
                      subtopics: List[Dict[str, Any]],
                      user_name: str = "") -> str:
    """HTML конспекта: тема + теория/задание/рубрика каждой подтемы +
    фидбек AI по завершённым (если в подтеме есть ai_feedback)."""
    title = html_module.escape(str(topic.get("title") or "Тема курса"))
    description = _escape_markdown(topic.get("description"))
    generated = str(topic.get("generated_at") or "").split(".")[0]
    meta_parts = ["Уровень курса: Android-разработка"]
    if user_name:
        meta_parts.append(f"Ученик: {html_module.escape(user_name)}")
    if generated:
        meta_parts.append(f"Сгенерировано: {html_module.escape(generated)}")

    sections: List[str] = []
    for index, sub in enumerate(subtopics, 1):
        sub_title = html_module.escape(str(sub.get("title") or ""))
        completed = " ✅" if sub.get("is_completed") else ""
        block = [f"<h2>{index}. {sub_title}{completed}</h2>"]
        block.append(f'<div class="theory">{_escape_markdown(sub.get("theory"))}</div>')
        block.append(
            '<div class="task"><b>Задание:</b> '
            f'{_escape_markdown(sub.get("task"))}</div>')
        block.append(
            '<div class="rubric"><b>Критерии оценки:</b> '
            f'{_escape_markdown(sub.get("rubric"))}</div>')
        feedback = str(sub.get("ai_feedback") or "").strip()
        if feedback:
            score = sub.get("score")
            header = f"Фидбек AI{f' (оценка {score}/5)' if score else ''}:"
            block.append(
                f'<div class="feedback"><b>{html_module.escape(header)}</b> '
                f'{_escape_markdown(feedback)}</div>')
        sections.append("\n".join(block))

    return f"""<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><style>{PDF_CSS}</style></head>
<body>
<h1>{title}</h1>
<p class="meta">{" · ".join(meta_parts)}</p>
<p>{description}</p>
{chr(10).join(sections)}
</body>
</html>"""


def _gtk_candidate_dirs() -> List[Path]:
    """Стандартные каталоги установки GTK-рантаймов на Windows.

    Порядок важен: первым — tschoonj «GTK3-Runtime Win64» (Pango 1.50),
    последним — «Gtk-Runtime» от winget-пакета GtkD.GtkPlusRuntime
    (Pango 1.43: импортируется, но WeasyPrint'у не подходит).
    GTK_RUNTIME_BIN (если задан) — всегда первым.
    """
    candidates: List[Path] = []
    override = os.environ.get("GTK_RUNTIME_BIN", "").strip()
    if override:
        candidates.append(Path(override))
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    program_files_x86 = Path(
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    candidates.extend([
        program_files / "GTK3-Runtime Win64" / "bin",
        *(  # пользовательская установка без прав администратора
            [Path(local_appdata) / "Programs" / "GTK3-Runtime Win64" / "bin"]
            if local_appdata else []),
        program_files_x86 / "GTK3-Runtime Win64" / "bin",
        program_files / "Gtk-Runtime" / "bin",
    ])
    return candidates


def ensure_gtk_on_path(candidates: Optional[List[Path]] = None
                       ) -> Optional[Path]:
    """Добавляет каталог GTK-DLL в PATH ТЕКУЩЕГО процесса.

    WeasyPrint грузит ``libgobject-2.0-0`` и Pango через ctypes из PATH
    процесса, который застывает на момент старта приложения: рантайм,
    установленный «вживую», не виден до перезапуска (error 0x7e).
    Функция находит установку по стандартным каталогам
    (:func:`_gtk_candidate_dirs`) и чинит PATH на месте; повторные
    вызовы ничего не портят (без дублей в PATH).

    Args:
        candidates: явный список каталогов (тесты); None — стандартные.

    Returns:
        Найденный каталог рантайма (даже если он уже был в PATH) или
        None, если ни один кандидат не содержит GTK-DLL.
    """
    if sys.platform != "win32":
        return None
    for candidate in (candidates if candidates is not None
                      else _gtk_candidate_dirs()):
        if not (candidate / GTK_MARKER_DLL).is_file():
            continue
        current = os.environ.get("PATH", "")
        entries = [entry.casefold() for entry in current.split(os.pathsep)
                   if entry.strip()]
        if str(candidate).casefold() not in entries:
            os.environ["PATH"] = str(candidate) + os.pathsep + current
        try:
            os.add_dll_directory(str(candidate))
        except (OSError, AttributeError):  # не Windows/нет прав — не критично
            pass
        return candidate
    return None


def weasyprint_renderer(html: str, target: Path,
                        gtk_candidates: Optional[List[Path]] = None) -> None:
    """Renderer по умолчанию: WeasyPrint (ленивый импорт — модуль
    работает и без установленного пакета)."""
    ensure_gtk_on_path(gtk_candidates)
    try:
        from weasyprint import HTML
    except Exception as exc:  # noqa: BLE001 — ImportError и DLL-ошибки GTK
        logger.warning("WeasyPrint недоступен: %s", exc)
        raise PDFError(WEASYPRINT_HINT) from exc
    try:
        HTML(string=html).write_pdf(str(target))
    except PDFError:
        raise
    except Exception as exc:  # noqa: BLE001 — ошибки рендера/записи
        str_exc = str(exc)
        if "not found in library" in str_exc or "error 0x7f" in str_exc:
            # Рантайм найден, но Pango старше 1.44 (например,
            # GtkD.GtkPlusRuntime из winget)
            logger.warning("Pango в GTK-рантайме слишком старый: %s", exc)
            raise PDFError(PANGO_TOO_OLD_HINT) from exc
        raise PDFError(f"Не удалось создать PDF: {exc}") from exc


class PDFError(RuntimeError):
    """Ошибка генерации PDF (WeasyPrint недоступен / запись не удалась)."""


class PDFService:
    """PDF-конспекты тем (output/pdf по умолчанию).

    Attributes:
        output_dir: папка сохранения конспектов (создаётся лениво).
        renderer: html, путь → файл; по умолчанию WeasyPrint, тесты
            подставляют фейк.
    """

    def __init__(self, output_dir: Optional[str | Path] = None,
                 renderer: Optional[Renderer] = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else (
            Path(__file__).resolve().parents[2] / "output" / "pdf")
        self.renderer = renderer or weasyprint_renderer

    def generate_summary(self, topic: Dict[str, Any],
                         subtopics: List[Dict[str, Any]],
                         user_name: str = "") -> str:
        """Создаёт PDF-конспект темы; возвращает абсолютный путь файла.

        Файл: ``summary_<topic_id>_<YYYYMMDD>.pdf`` (формат ТЗ),
        каталог создаётся автоматически (parent.mkdir).
        """
        if not subtopics:
            raise PDFError("Тема ещё не сгенерирована — конспекта нет")
        html = render_topic_html(topic, subtopics, user_name=user_name)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d")
        target = self.output_dir / f"summary_{topic.get('id')}_{stamp}.pdf"
        try:
            self.renderer(html, target)
        except PDFError:
            raise
        except Exception as exc:  # noqa: BLE001 — ошибки рендера/записи
            raise PDFError(f"Не удалось создать PDF: {exc}") from exc
        logger.info("PDF-конспект создан: %s", target)
        return str(target.absolute())
