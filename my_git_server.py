"""Собственный MCP-сервер для работы с локальным Git (FastMCP, stdio).

Запускается приложением как подпроцесс: MCP-клиент (src/mcp_client.py)
поднимает ``python -u my_git_server.py``, обмен идёт через stdin/stdout
(JSON-RPC, транспорт stdio — никакой сети и хостинга).

Инструменты (регистрируются декоратором ``@mcp.tool()``, типы параметров
автоматически превращаются в inputSchema, docstring — в описание для LLM):

* ``get_git_log``          — последние N коммитов (hash, author, date, message);
* ``get_git_status``       — статус рабочей директории (modified/staged/untracked);
* ``get_git_branches``     — список локальных веток;
* ``get_current_branch``   — имя текущей ветки (git-контекст команды /help).

Запуск вручную (для проверки): ``python -u my_git_server.py`` — сервер
читает JSON-RPC из stdin до EOF. Флаг ``-u`` обязателен: без него stdout
подпроцесса буферизуется и клиент не видит ответы.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List

from mcp.server.fastmcp import FastMCP

# Создаём сервер: имя видно клиенту в информации о сервере.
mcp = FastMCP("git-server")

# Границы разумного значения limit для get_git_log: 0 и отрицательные —
# некорректный аргумент ``git log -N``, тысячи коммитов — не сценарий UI.
MIN_LIMIT = 1
MAX_LIMIT = 500


def _run_git(repo_path: str, *args: str) -> subprocess.CompletedProcess:
    """Выполняет ``git -C <repo> <args>`` и возвращает результат.

    ВАЖНО: ``stdin=subprocess.DEVNULL`` обязателен. Дочерний процесс не
    должен наследовать stdin самого сервера (JSON-RPC-канал с клиентом):
    на Windows + anyio stdio такой spawn зависает навсегда — ответ
    tools/call никогда не уходит, и клиент упирается в таймаут.

    Остальное: ``-c core.quotepath=false`` — неэкранированные пути в
    статусе (кириллица в именах файлов остаётся читаемой);
    ``errors="replace"`` защищает разбор текста от «битых» байт.
    """
    return subprocess.run(
        ["git", "-C", repo_path, "-c", "core.quotepath=false", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
    )


@mcp.tool()
def get_git_log(repo_path: str, limit: int = 10) -> List[Dict[str, str]]:
    """Возвращает последние N коммитов в репозитории.

    Args:
        repo_path: Абсолютный путь к git-репозиторию.
        limit: Максимальное количество коммитов, от 1 до 500 (по умолчанию 10).

    Returns:
        Список словарей с полями: hash (короткий), author, date, message.
        При ошибке — список из одного словаря {"error": "..."}.
    """
    if not Path(repo_path).exists():
        return [{"error": f"Путь не существует: {repo_path}"}]

    limit = max(MIN_LIMIT, min(int(limit), MAX_LIMIT))

    result = _run_git(
        repo_path, "log", f"-{limit}",
        "--pretty=format:%H|%an|%ad|%s", "--date=short",
    )

    if result.returncode != 0:
        # Репозиторий без единого коммита (unborn HEAD) — не ошибка вызова.
        if "does not have any commits" in (result.stderr or ""):
            return []
        return [{"error": f"Ошибка git: {(result.stderr or '').strip()}"}]

    commits: List[Dict[str, str]] = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        # maxsplit=3: символ '|' в сообщении коммита не ломает разбор.
        parts = line.split("|", 3)
        if len(parts) == 4:
            commits.append({
                "hash": parts[0][:8],      # короткий hash
                "author": parts[1],
                "date": parts[2],
                "message": parts[3],
            })
    return commits


@mcp.tool()
def get_git_status(repo_path: str) -> Dict[str, List[str]]:
    """Возвращает статус рабочей директории Git.

    Args:
        repo_path: Абсолютный путь к git-репозиторию.

    Returns:
        Словарь с полями: modified (изменено, не индексировано),
        staged (индексировано), untracked (не отслеживается).
        При ошибке — {"error": ["..."]}.
    """
    if not Path(repo_path).exists():
        return {"error": [f"Путь не существует: {repo_path}"]}

    result = _run_git(repo_path, "status", "--porcelain")

    if result.returncode != 0:
        return {"error": [(result.stderr or "").strip()]}

    modified: List[str] = []
    staged: List[str] = []
    untracked: List[str] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        status, path = line[:2], line[3:]
        # Переименование: porcelain пишет "R  old -> new" — берём новый путь.
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if status == "??":
            untracked.append(path)
        elif status[0] != " ":
            staged.append(path)
        else:
            modified.append(path)

    return {
        "modified": modified,
        "staged": staged,
        "untracked": untracked,
    }


@mcp.tool()
def get_git_branches(repo_path: str) -> List[str]:
    """Возвращает список локальных веток в репозитории.

    Args:
        repo_path: Абсолютный путь к git-репозиторию.

    Returns:
        Список имён веток. При ошибке — список из одного элемента
        с текстом ошибки.
    """
    if not Path(repo_path).exists():
        return [f"Путь не существует: {repo_path}"]

    result = _run_git(repo_path, "branch", "--format=%(refname:short)")

    if result.returncode != 0:
        return [(result.stderr or "").strip()]

    return [b.strip() for b in result.stdout.splitlines() if b.strip()]


@mcp.tool()
def get_current_branch(repo_path: str) -> str:
    """Возвращает имя текущей ветки Git-репозитория.

    Args:
        repo_path: Абсолютный путь к git-репозиторию.

    Returns:
        Имя текущей ветки (например, "main"). В detached HEAD — строка
        "detached HEAD (<короткий хеш>)". При ошибке — текст ошибки
        строкой (вызов tools/call при этом не помечается isError:
        пайплайн /help сам решает, как деградировать без ветки).
    """
    if not Path(repo_path).exists():
        return f"Путь не существует: {repo_path}"

    # branch --show-current: имя из HEAD-symref; работает и в репозитории
    # без коммитов (unborn HEAD), в отличие от rev-parse HEAD.
    result = _run_git(repo_path, "branch", "--show-current")

    if result.returncode != 0:
        return f"Ошибка git: {(result.stderr or '').strip()}"

    branch = result.stdout.strip()
    if branch:
        return branch

    # Пустое имя — detached HEAD: берём короткий хеш коммита.
    head = _run_git(repo_path, "rev-parse", "--short", "HEAD")
    if head.returncode == 0 and head.stdout.strip():
        return f"detached HEAD ({head.stdout.strip()})"
    return "(репозиторий без коммитов)"


if __name__ == "__main__":
    # Запуск с stdio-транспортом (по умолчанию): сервер работает,
    # пока открыт stdin со стороны клиента-родителя.
    mcp.run()
