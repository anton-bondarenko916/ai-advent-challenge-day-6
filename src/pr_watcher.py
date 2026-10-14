"""Watcher новых PR: polling GitHub API каждые N минут (ТЗ ai_1_33).

Цикл: GET /repos/{owner}/{repo}/pulls?state=open&sort=updated →
фильтр «(pr_number, head_sha) ещё не просмотрен» → callback(PR) →
пометка просмотренным в SQLite (reviewed_prs, UNIQUE(pr_number,
head_sha)). Новый коммит в PR = новый head_sha = повторное ревью.

Решение по ТЗ-критерию «ошибка — watcher продолжает»: ошибка ОДНОГО
callback не прерывает обработку остальных PR цикла и сам цикл polling;
PR помечается просмотренным ПОСЛЕ попытки ревью даже при ошибке — иначе
сломавшийся PR ретраился бы каждые 5 минут вечно (повторное ревью всё
равно придёт с новым head_sha — новым коммитом).

Ловушки проекта соблюдены: короткоживущие соединения sqlite3 с
обязательным close() (Windows держит файл, пока соединение открыто),
таблица создаётся при каждом подключении, ошибки БД деградируют
(лог + False), а не роняют watcher. httpx-transport подставляется в
конструкторе — тесты идут офлайн через MockTransport.
"""

from __future__ import annotations

import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import httpx

from .utils import get_logger

logger = get_logger("deepseek-chat.pr-watcher")

GITHUB_API = "https://api.github.com"
API_VERSION = "2022-11-28"

# "owner/repo": без пробелов и без второго слэша
_REPO_RE = re.compile(r"^[^/\s]+/[^/\s]+$")

# Полные ссылки, из которых достаём owner/repo: HTTPS-URL (с www и
# .git на конце) и SSH-форма git@github.com:owner/repo.git
_REPO_URL_RES = (
    re.compile(
        r"^(?:https?://)?(?:www\.)?github\.com/([^/\s]+)/([^/\s]+?)"
        r"(?:\.git)?/?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^git@github\.com:([^/\s]+)/([^/\s]+?)(?:\.git)?$",
        re.IGNORECASE,
    ),
)


def normalize_repo(raw: str) -> str:
    """Приводит 'https://github.com/owner/repo(.git)' и SSH-форму к 'owner/repo'.

    Значения уже в формате 'owner/repo' (и вообще всё остальное)
    возвращаются как есть — валидацию делает вызывающий.
    """
    value = str(raw or "").strip().strip('"').strip("'")
    for pattern in _REPO_URL_RES:
        match = pattern.match(value)
        if match:
            return f"{match.group(1)}/{match.group(2)}"
    return value

DEFAULT_INTERVAL_MINUTES = 5
DEFAULT_PER_PAGE = 50


class PRWatcher:
    """Polling GitHub API: новые открытые PR → callback.

    Attributes:
        token: персональный токен GitHub (права repo — нужны для
            комментариев к PR через Inspectra).
        repo: репозиторий в формате "owner/repo".
        interval: период polling в секундах.
        db_path: путь к SQLite с таблицей reviewed_prs.
    """

    def __init__(
        self,
        token: str,
        repo: str,
        interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
        db_path: Any = "data/reviewed_prs.db",
        per_page: int = DEFAULT_PER_PAGE,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        token = str(token or "").strip()
        if not token:
            raise ValueError(
                "GITHUB_TOKEN не задан — watcher без токена не запускается "
                "(токен нужен с правами repo: и чтение PR, и комментарии)"
            )
        repo = normalize_repo(repo)
        if not _REPO_RE.match(repo):
            raise ValueError(
                f"GITHUB_REPO={repo!r} — ожидается 'owner/repo' "
                "(например, 'octocat/hello-world') или ссылка на репозиторий "
                "(https://github.com/owner/repo)"
            )
        if int(interval_minutes) < 1:
            raise ValueError(
                f"interval_minutes={interval_minutes} — минимум 1 минута"
            )
        self.token = token
        self.repo = repo
        self.interval = int(interval_minutes) * 60
        self.per_page = max(1, int(per_page))
        self.db_path = Path(db_path)
        if str(self.db_path.parent):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._transport = transport

    # ------------------------------------------------------------------
    # Учёт просмотренных PR (SQLite)
    # ------------------------------------------------------------------
    def is_reviewed(self, pr_number: int, head_sha: str) -> bool:
        """Просмотрен ли уже этот коммит этого PR (ошибка БД → True:
        повторное ревью лучше лишней попытки)."""
        try:
            with self._db() as conn:
                row = conn.execute(
                    "SELECT 1 FROM reviewed_prs"
                    " WHERE pr_number = ? AND head_sha = ?",
                    (int(pr_number), str(head_sha)),
                ).fetchone()
        except sqlite3.Error as exc:
            logger.warning("reviewed_prs: is_reviewed не удался: %s", exc)
            return True
        return row is not None

    def mark_reviewed(self, pr_number: int, head_sha: str) -> bool:
        """Помечает коммит PR просмотренным (INSERT OR IGNORE)."""
        try:
            with self._db() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO reviewed_prs"
                    " (pr_number, head_sha) VALUES (?, ?)",
                    (int(pr_number), str(head_sha)),
                )
            return True
        except sqlite3.Error as exc:
            logger.warning("reviewed_prs: mark_reviewed не удался: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------
    def poll(self) -> List[Dict[str, Any]]:
        """Возвращает список непросмотренных открытых PR.

        Returns:
            [{"number", "head_sha", "title", "branch", "url", "draft"}].

        Raises:
            httpx.HTTPError: сеть недоступна или GitHub вернул ошибку
                (обрабатывает цикл run — polling не умирает).
        """
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
        }
        url = (
            f"{GITHUB_API}/repos/{self.repo}/pulls"
            f"?state=open&sort=updated&direction=desc&per_page={self.per_page}"
        )
        with self._client() as client:
            response = client.get(url, headers=headers, timeout=15.0)
        response.raise_for_status()

        new_prs: List[Dict[str, Any]] = []
        for item in response.json():
            pr = self._parse_pr(item)
            if pr is None:
                continue
            if not self.is_reviewed(pr["number"], pr["head_sha"]):
                new_prs.append(pr)
        return new_prs

    def process_new(self, callback: Callable[[Dict[str, Any]], Any]) -> int:
        """Один цикл: poll → callback для каждого нового PR → пометка.

        Ошибка callback НЕ прерывает остальные PR цикла и не пробрасывается
        (критерий приёмки «watcher продолжает»); PR помечается просмотренным
        после попытки ревью независимо от результата — иначе сломавшийся PR
        ретраился бы бесконечно (см. докстринг модуля).

        Returns:
            Число найденных новых PR.

        Raises:
            httpx.HTTPError: ошибка poll (сетевые ошибки отдаёт наверх —
                их ловит run, чтобы ретраить на следующем цикле).
        """
        new_prs = self.poll()
        for pr in new_prs:
            logger.info(
                "Найден новый PR #%s: %s", pr["number"], pr["title"])
            try:
                callback(pr)
            except Exception:  # noqa: BLE001 — цикл не должен умирать
                logger.exception(
                    "Ошибка обработки PR #%s", pr["number"])
            self.mark_reviewed(pr["number"], pr["head_sha"])
        return len(new_prs)

    def run(self, callback: Callable[[Dict[str, Any]], Any]) -> None:
        """Запускает polling loop до Ctrl+C (KeyboardInterrupt пробрасывается).

        Args:
            callback: обработчик нового PR (запуск Inspectra).
        """
        logger.info(
            "Watcher запущен. Polling каждые %s мин. Репозиторий: %s",
            self.interval // 60, self.repo)
        while True:
            try:
                self.process_new(callback)
            except Exception:  # noqa: BLE001 — ошибка цикла → ждём и заново
                logger.exception("Watcher error")
            time.sleep(self.interval)

    # ------------------------------------------------------------------
    # Внутреннее
    # ------------------------------------------------------------------
    def _client(self) -> httpx.Client:
        """httpx.Client с подстановкой транспорта (тесты — MockTransport)."""
        return httpx.Client(timeout=15.0, transport=self._transport)

    @staticmethod
    def _parse_pr(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Извлекает поля PR; битая запись (нет number/head.sha) — None."""
        try:
            head = item.get("head") or {}
            head_sha = str(head.get("sha") or "").strip()
            number = int(item["number"])
            if not head_sha:
                raise KeyError("head.sha")
        except (KeyError, TypeError, ValueError):
            logger.warning("Пропускаю битую запись PR: %s", str(item)[:200])
            return None
        return {
            "number": number,
            "head_sha": head_sha,
            "title": str(item.get("title") or ""),
            "branch": str((head.get("ref") or "")),
            "url": str(item.get("html_url") or ""),
            "draft": bool(item.get("draft", False)),
        }

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        """Короткоживущее соединение с гарантией close().

        ВАЖНО (ловушка Windows): ``with conn:`` у sqlite3 — это
        ТРАНЗАКЦИЯ, соединение при выходе не закрывается, и файл
        остаётся залочен. Поэтому try/finally + явный close.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS reviewed_prs ("
                    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    " pr_number INTEGER NOT NULL,"
                    " head_sha TEXT NOT NULL,"
                    " reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
                    " UNIQUE(pr_number, head_sha))"
                )
                yield conn
        finally:
            conn.close()
