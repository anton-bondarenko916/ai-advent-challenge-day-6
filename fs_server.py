"""MCP-сервер «Filesystem» (ТЗ ai_1_35): файловые операции проекта.

Транспорт: Streamable HTTP (stateful), только 127.0.0.1:8005 — сервер
НЕ хостится в интернете. Запускается приложением в отдельном окне
консоли (FsServerLauncher, см. src/fs_launcher.py) и живёт 24/7.

Инструменты agentic loop агента (LLM сама решает, что вызвать):

* ``read_file`` / ``read_multiple_files`` — чтение одного/нескольких
  файлов (большой файл обрезается по символам — защита контекста LLM);
* ``write_file`` — создание/перезапись файла (parent.mkdir — новые
  каталоги создаются автоматически, ТЗ);
* ``search_files`` — glob-поиск имён файлов;
* ``search_within_files`` — поиск подстроки по СОДЕРЖИМОМУ (сценарий
  «найти все места использования компонента»: файл/строка/текст);
* ``list_directory`` / ``tree`` — навигация (список / дерево JSON).

Безопасность: все пути проверяются ``_validate_path`` — доступ ТОЛЬКО
внутри разрешённого корня (ALLOWED_ROOT: по умолчанию сам проект,
переопределяется FS_ROOT/--root). Обход через «..», абсолютные пути
наружу и «соседние» каталоги с похожим префиксом блокируются.
Служебные каталоги (venv/.git/__pycache__/node_modules) при обходе
пропускаются, бинарные/большие файлы при поиске — тоже.

Конфигурация (env или CLI, см. main()): FS_HOST/FS_PORT — адрес
привязки (по умолчанию 127.0.0.1:8005), FS_ROOT — разрешённый корень.

Запуск вручную::

    python fs_server.py                      # корень = папка скрипта
    python fs_server.py --root C:/projects/demo --port 8015
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

from mcp.server.fastmcp import FastMCP

# ─── Инициализация MCP-сервера ──────────────────────────────────────────────

mcp = FastMCP(
    "fs-server",
    host=os.environ.get("FS_HOST", "127.0.0.1"),
    port=int(os.environ.get("FS_PORT", "8005")),
)

# Разрешённая директория (по умолчанию — сам проект, где лежит скрипт)
ALLOWED_ROOT = Path(os.environ.get(
    "FS_ROOT", str(Path(__file__).parent))).resolve()

# Служебные каталоги, которые НЕ обходятся при поиске/дереве
EXCLUDED_PARTS = ("venv", ".venv", "__pycache__", ".git", "node_modules",
                  ".idea", ".vscode", ".inspectra")

# Расширения текстовых файлов для search_within_files (код + документация)
TEXT_SUFFIXES = (
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg",
    ".ini", ".html", ".css", ".js", ".ts", ".sh", ".bat",
)

# Лимиты защиты контекста LLM и производительности
MAX_READ_CHARS = 24_000        # обрезка read_file/read_multiple_files:
                               # большие файлы целиком раздувают запрос
                               # (прецедент: 5 read_file → 300K+ символов
                               # → ответы модели обрезались в пустоту)
MAX_SEARCH_FILE_BYTES = 1_000_000   # файлы больше — не сканируются
MAX_SEARCH_RESULTS = 500            # жёсткий предел search_files
DEFAULT_SEARCH_RESULTS = 30         # дефолт search_within_files: результат
                                    # уходит в контекст LLM — сотни вхождений
                                    # раздувают запрос (прецедент: запрос
                                    # 325K символов → пустые ответы)
MAX_TREE_DEPTH = 8                  # предел глубины tree
MAX_BATCH_FILES = 20                # файлов за один read_multiple_files
SEARCH_SNIPPET_CHARS = 120          # обрезка строки-находки


def _validate_path(path: str) -> Path:
    """Проверяет, что путь внутри разрешённой директории (ТЗ).

    Относительные пути отсчитываются от ALLOWED_ROOT. Сравнение через
    ``relative_to`` (а не startswith): «C:/proj-сосед» не должен пройти
    проверку префиксом, «..» и абсолютные пути наружу блокируются.

    Raises:
        ValueError: путь вне ALLOWED_ROOT.
    """
    raw = str(path or "").strip() or "."
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = ALLOWED_ROOT / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(ALLOWED_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Доступ запрещён: {path} (вне разрешённой директории "
            f"{ALLOWED_ROOT})") from exc
    return resolved


def _is_excluded(path: Path) -> bool:
    """True, если путь проходит через служебный каталог (venv/.git/…)."""
    return any(part in EXCLUDED_PARTS for part in path.parts)


def _read_capped(path: Path) -> str:
    """Читает текст файла, обрезая длинные файлы (защита контекста LLM)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > MAX_READ_CHARS:
        text = (text[:MAX_READ_CHARS]
                + f"\n\n…(файл обрезан: показано {MAX_READ_CHARS} "
                  f"из {len(text)} символов)")
    return text


# ─── Инструменты MCP ─────────────────────────────────────────────────────────


@mcp.tool()
def read_file(path: str) -> str:
    """Прочитать содержимое файла проекта.

    Args:
        path: путь файла (относительный — от корня проекта, или
            абсолютный внутри разрешённой директории).

    Returns:
        Содержимое файла (длинные файлы обрезаются) либо JSON с error.
    """
    try:
        return _read_capped(_validate_path(path))
    except Exception as exc:  # noqa: BLE001 — ошибка уходит LLM текстом
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@mcp.tool()
def read_multiple_files(paths: List[str]) -> Dict[str, str]:
    """Прочитать несколько файлов за одну операцию.

    Args:
        paths: список путей файлов (до 20 за вызов).

    Returns:
        Словарь «путь → содержимое (или {"error": …})».
    """
    if len(paths) > MAX_BATCH_FILES:
        return {"error": f"Слишком много файлов за вызов "
                         f"({len(paths)} > {MAX_BATCH_FILES})"}
    result: Dict[str, str] = {}
    for path in paths:
        try:
            result[str(path)] = _read_capped(_validate_path(path))
        except Exception as exc:  # noqa: BLE001 — ошибка по каждому файлу
            result[str(path)] = json.dumps({"error": str(exc)},
                                           ensure_ascii=False)
    return result


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Создать или перезаписать файл (каталоги создаются автоматически).

    Args:
        path: путь файла внутри разрешённой директории.
        content: полное содержимое файла.

    Returns:
        Подтверждение с путём либо JSON с error.
    """
    try:
        target = _validate_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")
        return f"Файл создан: {target}"
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@mcp.tool()
def search_files(pattern: str, directory: str = ".") -> List[str]:
    """Найти файлы по glob-паттерну (например ``*.md`` или ``test_*.py``).

    Args:
        pattern: glob-паттерн имени файла.
        directory: каталог поиска (по умолчанию — корень проекта).

    Returns:
        Список относительных путей (служебные каталоги пропускаются)
        либо JSON с error.
    """
    try:
        root = _validate_path(directory)
        results: List[str] = []
        for found in root.rglob(str(pattern or "*")):
            if not found.is_file() or _is_excluded(found):
                continue
            results.append(str(found.relative_to(root)))
            if len(results) >= MAX_SEARCH_RESULTS:
                break
        return results
    except Exception as exc:  # noqa: BLE001
        return [json.dumps({"error": str(exc)}, ensure_ascii=False)]


@mcp.tool()
def search_within_files(substring: str, directory: str = ".",
                        max_results: int = 30) -> List[Dict]:
    """Найти все вхождения подстроки в текстовых файлах проекта.

    Сценарий «найти все места использования компонента/API»: сканируются
    файлы кода и документации (см. TEXT_SUFFIXES), служебные каталоги
    (venv, .git, __pycache__, …) пропускаются.

    Args:
        substring: искомая подстрока (регистр учитывается).
        directory: каталог поиска (по умолчанию — корень проекта).
        max_results: предел записей (по умолчанию 30 — результат уходит
            в контекст LLM: сотни вхождений раздувают запрос и обрезают
            ответы модели).

    Returns:
        Список ``[{"file", "line", "content"}]`` либо ``[{"error": …}]``.
    """
    try:
        root = _validate_path(directory)
    except Exception as exc:  # noqa: BLE001
        return [{"error": str(exc)}]
    try:
        limit = min(max(int(max_results), 1), MAX_SEARCH_RESULTS)
    except (TypeError, ValueError):
        limit = DEFAULT_SEARCH_RESULTS
    needle = str(substring or "")
    if not needle.strip():
        return [{"error": "Пустая подстрока для поиска"}]

    results: List[Dict] = []
    for file_path in root.rglob("*"):
        if (not file_path.is_file()
                or file_path.suffix.lower() not in TEXT_SUFFIXES
                or _is_excluded(file_path)):
            continue
        try:
            if file_path.stat().st_size > MAX_SEARCH_FILE_BYTES:
                continue
            lines = file_path.read_text(
                encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line_no, line in enumerate(lines, 1):
            if needle in line:
                results.append({
                    "file": str(file_path.relative_to(root)),
                    "line": line_no,
                    "content": line.strip()[:SEARCH_SNIPPET_CHARS],
                })
                if len(results) >= limit:
                    return results
    return results


@mcp.tool()
def list_directory(path: str = ".") -> List[str]:
    """Список файлов и папок в директории (каталоги — с суффиксом «/»).

    Args:
        path: каталог (по умолчанию — корень проекта).

    Returns:
        Список имён либо JSON с error.
    """
    try:
        root = _validate_path(path)
        return sorted(
            entry.name + ("/" if entry.is_dir() else "")
            for entry in root.iterdir()
        )
    except Exception as exc:  # noqa: BLE001
        return [json.dumps({"error": str(exc)}, ensure_ascii=False)]


@mcp.tool()
def tree(path: str = ".", depth: int = 3) -> str:
    """Дерево каталогов проекта (JSON-строка: имя → поддерево/«file»).

    Args:
        path: корень дерева (по умолчанию — корень проекта).
        depth: глубина (по умолчанию 3, максимум 8).

    Returns:
        JSON-строка дерева либо JSON с error.
    """
    try:
        root = _validate_path(path)
        try:
            level = min(max(int(depth), 0), MAX_TREE_DEPTH)
        except (TypeError, ValueError):
            level = 3

        def _walk(current: Path, current_depth: int) -> Dict:
            result: Dict = {}
            try:
                entries = sorted(current.iterdir(),
                                 key=lambda item: item.name.lower())
            except OSError:
                return {}
            for item in entries:
                if _is_excluded(item):
                    continue
                if item.is_dir():
                    result[item.name] = (
                        _walk(item, current_depth + 1)
                        if current_depth < level else "dir")
                else:
                    result[item.name] = "file"
            return result

        return json.dumps(_walk(root, 0), ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


# ─── Точка входа ─────────────────────────────────────────────────────────────


def main() -> None:
    """Разбирает CLI, применяет переопределения и стартует HTTP-сервер."""
    global ALLOWED_ROOT

    parser = argparse.ArgumentParser(
        description="MCP-сервер Filesystem (Streamable HTTP)"
    )
    parser.add_argument(
        "--host", default=os.environ.get("FS_HOST", "127.0.0.1"),
        help="адрес привязки (по умолчанию 127.0.0.1 — только локально)",
    )
    parser.add_argument(
        "--port", type=int,
        default=int(os.environ.get("FS_PORT", "8005")),
        help="порт HTTP-сервера (по умолчанию 8005)",
    )
    parser.add_argument(
        "--root", default=os.environ.get("FS_ROOT", ""),
        help="разрешённая директория (по умолчанию — папка скрипта)",
    )
    args = parser.parse_args()

    # Синхронизируем настройки FastMCP и корень доступа с CLI
    mcp.settings.host = args.host
    mcp.settings.port = int(args.port)
    if args.root:
        ALLOWED_ROOT = Path(args.root).resolve()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print(f"FS MCP-сервер запущен на "
          f"http://{args.host}:{args.port}/mcp")
    print(f"Разрешённая директория: {ALLOWED_ROOT}")
    print("Окно можно закрыть вместе с завершением сервера (Ctrl+C).")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
