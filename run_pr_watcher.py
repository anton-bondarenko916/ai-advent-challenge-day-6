"""AI-ревью PR: watcher GitHub + RAG по коду + Inspectra — точка входа.

Схема: polling GitHub API каждые N минут (только ИСХОДЯЩИЕ соединения)
→ новый открытый PR (по паре pr_number + head_sha) → RAG по коду
проекта (коллекция project_code в Chroma) → промпт «баги и edge cases»
в .inspectra.yml → ``inspectra review --post-comment --pr-summary``
локальной моделью qwen2.5-coder-14b-32k → комментарий в PR.

Запуск: ``python run_pr_watcher.py`` (токен и репозиторий — в .env,
см. docs/pr_review_watcher.md). Перед первым запуском создать модель:

    ollama pull qwen2.5-coder:14b
    ollama create qwen2.5-coder-14b-32k -f Modelfile_qwen_coder_32k

Точка входа тонкая: вся логика в src/pr_watcher.py,
src/inspectra_runner.py и app/rag/code_rag.py — и тестируется офлайн.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from app.rag.code_rag import CodeRAG
from app.rag.embedder import EmbedderError, OllamaEmbedder
from app.rag.vector_store import VectorStore
from src.config import ENV_PATH, get_pr_watcher_settings, load_config
from src.inspectra_runner import InspectraRunner
from src.pr_watcher import PRWatcher

logger = logging.getLogger("deepseek-chat.pr-watcher-main")

LOG_PATH = ENV_PATH.parent / "run_pr_watcher.log"


def _setup_logging() -> None:
    """Журнал в файл (UTF-8) + консоль; токен в лог не попадает."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def main() -> int:
    """Точка входа: настройки из .env → индексация кода → polling loop.

    ВАЖНО: .env читает только load_config() (как в GUI-приложении и
    telegram_bot.py); get_config()/get_pr_watcher_settings() переменные
    из файла НЕ подхватывают — без load_config() watcher не увидит
    токен вообще.
    """
    _setup_logging()
    settings = get_pr_watcher_settings(load_config())

    if not settings["token"]:
        print(
            "GITHUB_TOKEN не задан.\n"
            f"1) Впишите в .env ({ENV_PATH}):\n"
            "   GITHUB_TOKEN=ghp_...\n"
            "   GITHUB_REPO=owner/repo\n"
            "   (строки должны быть раскомментированы, без пробела в начале)\n"
            "2) Токен — GitHub → Settings → Developer settings → "
            "Personal access tokens (права repo).\n"
            "Подробности: docs/pr_review_watcher.md")
        return 1
    if not settings["repo"]:
        print(
            "GITHUB_REPO не задан.\n"
            f"Впишите в .env ({ENV_PATH}): GITHUB_REPO=owner/repo")
        return 1

    # 1. Индексация кода проекта в коллекцию project_code
    embedder = OllamaEmbedder()
    store = VectorStore()
    code_rag = CodeRAG(vector_store=store, embedder=embedder)
    try:
        stats = code_rag.index_code(settings["project_path"])
    except EmbedderError as exc:
        logger.error(
            "Индексация кода не удалась: %s. Проверьте 'ollama serve' "
            "и модель эмбеддингов nomic-embed-text.", exc)
        return 1
    logger.info(
        "Индексация кода: %s файлов, %s чанков (ошибок: %s)",
        stats["files"], stats["chunks"], len(stats["errors"]))

    # 2. Обёртка Inspectra и watcher
    runner = InspectraRunner(
        project_path=settings["project_path"],
        code_rag=code_rag,
        model=settings["model"],
        provider=settings["provider"],
        timeout=settings["timeout"],
        top_k=settings["top_k"],
        repo=settings["repo"],
    )
    watcher = PRWatcher(
        token=settings["token"],
        repo=settings["repo"],
        interval_minutes=settings["interval_minutes"],
        db_path=settings["db_path"],
    )

    def on_new_pr(pr: Dict[str, Any]) -> None:
        logger.info("Запуск ревью PR #%s (%s)...", pr["number"], pr["title"])
        result = runner.review_pr(pr["number"], pr["title"])
        if result["success"]:
            logger.info("PR #%s отревьюен успешно.", pr["number"])
        else:
            logger.error(
                "Ошибка ревью PR #%s: %s",
                pr["number"], (result["stderr"] or "")[-500:])

    # 3. Цикл polling (до Ctrl+C)
    try:
        watcher.run(on_new_pr)
    except KeyboardInterrupt:
        logger.info("Watcher остановлен (Ctrl+C).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
