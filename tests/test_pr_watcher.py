"""Тесты watcher'а PR (src/pr_watcher.py) и его настроек — офлайн.

GitHub API подменяется httpx.MockTransport (запросы запоминаются),
SQLite — во временной папке. Проверяются: валидация (токен, формат
"owner/repo", интервал), poll (новые PR, фильтр просмотренных, битые
записи, HTTP/сетевые ошибки), process_new (callback по каждому PR,
ошибка callback не останавливает остальные и не пробрасывается,
пометка просмотренным), настройки get_pr_watcher_settings (дефолты,
окружение, некорректные значения) и регрессия «.env виден только через
load_config()» + main() без токена завершается с кодом 1.

Запуск из корня проекта:
    python -m unittest tests.test_pr_watcher -v
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from src.config import (
    PROJECT_ROOT,
    get_pr_watcher_settings,
    load_config,
)
from src.pr_watcher import PRWatcher, normalize_repo


def pr_item(number: int, sha: str, title: str = "Fix bug",
            **extra) -> dict:
    """Запись PR в формате GitHub API."""
    item = {
        "number": number,
        "title": title,
        "head": {"sha": sha, "ref": f"feature-{number}"},
        "html_url": f"https://github.com/owner/repo/pull/{number}",
        "draft": False,
    }
    item.update(extra)
    return item


def github_transport(captured: list, items: list = None,
                     status: int = 200) -> httpx.MockTransport:
    """MockTransport: запоминает запросы, отдаёт список PR."""

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if status != 200:
            return httpx.Response(status, json={"message": "boom"},
                                  request=request)
        return httpx.Response(200, json=items or [], request=request)

    return httpx.MockTransport(handler)


def failing_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    return httpx.MockTransport(handler)


def make_watcher(db_path: Path, transport: httpx.MockTransport,
                 interval: int = 5) -> PRWatcher:
    return PRWatcher(
        token="test-token", repo="owner/repo",
        interval_minutes=interval, db_path=db_path,
        transport=transport,
    )


class PRWatcherValidationTest(unittest.TestCase):
    """Валидация конструктора (ТЗ: без токена watcher не запускается)."""

    def test_empty_token_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "GITHUB_TOKEN"):
            PRWatcher(token="", repo="owner/repo", db_path="x.db")

    def test_bad_repo_format_raises(self) -> None:
        for repo in ("justname", "a/b/c", "owner/ with space", ""):
            with self.assertRaisesRegex(ValueError, "owner/repo"):
                PRWatcher(token="tok", repo=repo, db_path="x.db")

    def test_zero_interval_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "интервал|минут"):
            PRWatcher(token="tok", repo="a/b", interval_minutes=0,
                      db_path="x.db")

    def test_repo_accepts_full_urls(self) -> None:
        """Полная ссылка (как в git remote) приводится к owner/repo."""
        cases = {
            "https://github.com/octocat/hello-world": "octocat/hello-world",
            "http://github.com/octocat/hello-world": "octocat/hello-world",
            "github.com/octocat/hello-world": "octocat/hello-world",
            "https://www.github.com/octocat/hello-world.git":
                "octocat/hello-world",
            "https://github.com/octocat/hello-world/": "octocat/hello-world",
            "git@github.com:octocat/hello-world.git": "octocat/hello-world",
            '"https://github.com/octocat/hello-world"':
                "octocat/hello-world",  # с кавычками из .env
            "octocat/hello-world": "octocat/hello-world",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_repo(raw), expected)
                watcher = PRWatcher(token="tok", repo=raw, db_path="x.db")
                self.assertEqual(watcher.repo, expected)


class PRWatcherPollTest(unittest.TestCase):
    """poll/process_new на MockTransport, SQLite во временной папке."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.db = self.tmp / "reviewed_prs.db"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_poll_returns_new_prs_and_sends_headers(self) -> None:
        captured: list = []
        items = [pr_item(1, "sha1", "Add feature"),
                 pr_item(2, "sha2", "Fix bug")]
        watcher = make_watcher(self.db, github_transport(captured, items))
        new_prs = watcher.poll()

        self.assertEqual(len(new_prs), 2)
        self.assertEqual(new_prs[0]["number"], 1)
        self.assertEqual(new_prs[0]["head_sha"], "sha1")
        self.assertEqual(new_prs[0]["title"], "Add feature")
        self.assertEqual(new_prs[0]["branch"], "feature-1")

        request = captured[0]
        self.assertIn("repos/owner/repo/pulls", str(request.url))
        self.assertEqual(request.url.params["state"], "open")
        self.assertEqual(request.url.params["sort"], "updated")
        self.assertEqual(request.headers["Authorization"],
                         "Bearer test-token")
        self.assertIn("vnd.github+json", request.headers["Accept"])

    def test_poll_filters_reviewed(self) -> None:
        captured: list = []
        items = [pr_item(1, "sha1"), pr_item(2, "sha2")]
        watcher = make_watcher(self.db, github_transport(captured, items))
        watcher.mark_reviewed(1, "sha1")
        new_prs = watcher.poll()
        self.assertEqual([pr["number"] for pr in new_prs], [2])

    def test_same_sha_not_repolled_new_sha_is(self) -> None:
        """Повторное ревью: тот же (pr, sha) — нет, новый коммит — да."""
        captured: list = []
        items = [pr_item(7, "sha-old")]
        watcher = make_watcher(self.db, github_transport(captured, items))
        watcher.mark_reviewed(7, "sha-old")
        self.assertEqual(watcher.poll(), [])
        items[:] = [pr_item(7, "sha-new")]  # новый коммит в PR
        self.assertEqual([pr["head_sha"] for pr in watcher.poll()],
                         ["sha-new"])

    def test_poll_http_error_raises(self) -> None:
        captured: list = []
        watcher = make_watcher(self.db, github_transport(captured,
                                                         status=401))
        with self.assertRaises(httpx.HTTPStatusError):
            watcher.poll()

    def test_poll_network_error_raises(self) -> None:
        watcher = make_watcher(self.db, failing_transport())
        with self.assertRaises(httpx.ConnectError):
            watcher.poll()

    def test_poll_skips_malformed_entries(self) -> None:
        captured: list = []
        items = [
            {"title": "нет number и head"},
            {"number": 3, "head": {}},           # нет sha
            {"number": "abc", "head": {"sha": "s"}},  # битый number
            pr_item(4, "sha4"),
        ]
        watcher = make_watcher(self.db, github_transport(captured, items))
        new_prs = watcher.poll()
        self.assertEqual([pr["number"] for pr in new_prs], [4])

    # ------------------------------------------------------------------
    # process_new: callback + пометка просмотренным
    # ------------------------------------------------------------------
    def test_process_new_calls_callback_and_marks(self) -> None:
        captured: list = []
        items = [pr_item(1, "sha1"), pr_item(2, "sha2")]
        watcher = make_watcher(self.db, github_transport(captured, items))
        seen: list = []

        processed = watcher.process_new(seen.append)

        self.assertEqual(processed, 2)
        self.assertEqual([pr["number"] for pr in seen], [1, 2])
        self.assertTrue(watcher.is_reviewed(1, "sha1"))
        self.assertTrue(watcher.is_reviewed(2, "sha2"))
        # Второй цикл: всё просмотрено, callback не зовётся
        seen.clear()
        self.assertEqual(watcher.process_new(seen.append), 0)
        self.assertEqual(seen, [])

    def test_callback_error_logged_and_rest_processed(self) -> None:
        """Ошибка callback не роняет process_new и не блокирует остальные
        PR (критерий приёмки: watcher продолжает)."""
        captured: list = []
        items = [pr_item(1, "sha1"), pr_item(2, "sha2")]
        watcher = make_watcher(self.db, github_transport(captured, items))
        seen: list = []

        def failing_on_first(pr: dict) -> None:
            seen.append(pr["number"])
            if pr["number"] == 1:
                raise RuntimeError("Inspectra упал")

        processed = watcher.process_new(failing_on_first)

        self.assertEqual(processed, 2)
        self.assertEqual(seen, [1, 2])  # второй PR обработан
        # PR помечен просмотренным даже при ошибке: иначе ретраился бы
        # каждые 5 минут вечно (новый head_sha → новое ревью)
        self.assertTrue(watcher.is_reviewed(1, "sha1"))
        self.assertTrue(watcher.is_reviewed(2, "sha2"))

    def test_db_error_degrades_reviewed_true(self) -> None:
        """Ошибка БД в is_reviewed → True (лучше пропустить, чем
        ретраить сломанный PR бесконечно)."""
        captured: list = []
        watcher = make_watcher(self.db, github_transport(captured))
        # Каталог вместо файла БД → sqlite3 error на каждое обращение
        bad = self.tmp / "dir_as_db"
        bad.mkdir()
        watcher.db_path = bad
        self.assertTrue(watcher.is_reviewed(1, "sha"))
        self.assertFalse(watcher.mark_reviewed(1, "sha"))


class PRWatcherSettingsTest(unittest.TestCase):
    """get_pr_watcher_settings: дефолты, окружение, некорректные значения."""

    def test_defaults(self) -> None:
        with mock.patch.dict(os.environ):  # изоляция os.environ
            for key in ("GITHUB_TOKEN", "GITHUB_REPO",
                        "PR_WATCHER_PROJECT_PATH", "PR_WATCHER_INTERVAL",
                        "PR_WATCHER_DB_PATH"):
                os.environ.pop(key, None)
            settings = get_pr_watcher_settings({})
        self.assertEqual(settings["token"], "")
        self.assertEqual(settings["repo"], "")
        self.assertEqual(settings["project_path"], str(PROJECT_ROOT))
        self.assertEqual(settings["interval_minutes"], 5)
        self.assertEqual(settings["model"], "qwen2.5-coder-14b-32k")
        self.assertEqual(settings["provider"], "ollama")
        self.assertTrue(Path(settings["db_path"]).is_absolute())
        self.assertIn("data", Path(settings["db_path"]).parts)
        self.assertEqual(settings["timeout"], 600.0)
        self.assertEqual(settings["top_k"], 5)

    def test_env_override(self) -> None:
        env = {
            "GITHUB_TOKEN": "ghp_test",
            "GITHUB_REPO": "me/myrepo",
            "PR_WATCHER_INTERVAL": "10",
            "PR_WATCHER_MODEL": "custom-model",
            "PR_WATCHER_TOP_K": "3",
        }
        with mock.patch.dict(os.environ, env):
            settings = get_pr_watcher_settings()
        self.assertEqual(settings["token"], "ghp_test")
        self.assertEqual(settings["repo"], "me/myrepo")
        self.assertEqual(settings["interval_minutes"], 10)
        self.assertEqual(settings["model"], "custom-model")
        self.assertEqual(settings["top_k"], 3)

    def test_repo_url_normalized_in_settings(self) -> None:
        """Регрессия: полный URL в GITHUB_REPO должен превращаться в
        'owner/repo' УЖЕ в настройках — иначе inspectra получает
        GITHUB_REPOSITORY=URL и падает с «PR not found» (404)."""
        with mock.patch.dict(os.environ):
            settings = get_pr_watcher_settings({
                "GITHUB_REPO":
                    "https://github.com/octocat/hello-world",
            })
        self.assertEqual(settings["repo"], "octocat/hello-world")

    def test_invalid_values_fall_back_to_defaults(self) -> None:
        with mock.patch.dict(os.environ):
            settings = get_pr_watcher_settings({
                "PR_WATCHER_INTERVAL": "abc",
                "PR_WATCHER_TIMEOUT": "0",     # ниже минимума 60
                "PR_WATCHER_TOP_K": "999",     # выше максимума 20
            })
        self.assertEqual(settings["interval_minutes"], 5)
        self.assertEqual(settings["timeout"], 600.0)
        self.assertEqual(settings["top_k"], 5)

    def test_relative_project_path_resolved_from_root(self) -> None:
        settings = get_pr_watcher_settings({
            "PR_WATCHER_PROJECT_PATH": "sub/dir",
        })
        self.assertEqual(settings["project_path"],
                         str(PROJECT_ROOT / "sub" / "dir"))

    def test_env_file_reaches_settings_via_load_config(self) -> None:
        """Регрессия (по образцу ai_1_31): .env читает ТОЛЬКО load_config()."""
        tmp = Path(tempfile.mkdtemp())
        try:
            env_path = tmp / ".env"
            env_path.write_text(
                "GITHUB_TOKEN=ghp_from_env_file\n"
                "GITHUB_REPO=env/repo\n", encoding="utf-8")
            with mock.patch.dict(os.environ):  # изоляция os.environ
                for key in ("GITHUB_TOKEN", "GITHUB_REPO"):
                    os.environ.pop(key, None)
                config = load_config(env_file=env_path)
                settings = get_pr_watcher_settings(config)
            self.assertEqual(settings["token"], "ghp_from_env_file")
            self.assertEqual(settings["repo"], "env/repo")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_main_without_token_exits_1(self) -> None:
        """main() без токена печатает подсказку и возвращает 1 (без сети)."""
        import run_pr_watcher

        with mock.patch.dict(os.environ,
                             {"GITHUB_TOKEN": "", "GITHUB_REPO": ""}):
            code = run_pr_watcher.main()
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
