"""Обёртка над Inspectra CLI: AI-ревью PR с RAG-контекстом из кода.

Схема ревью одного PR:

1. RAG по коду (CodeRAG, коллекция project_code): по заголовку PR
   ищутся релевантные чанки *.py — они уходят в промпт как контекст;
2. во временный ``.inspectra.yml`` проекта пишется промпт (фокус —
   баги и edge cases: логические ошибки, необработанные ветки,
   ошибки в условиях и циклах);
3. запускается ``inspectra review --pr N --provider ollama --model
   qwen2.5-coder-14b-32k --post-comment --pr-summary --config ...``
   (обязателен ``--post-comment``: по умолчанию его НЕТ);
4. временный конфиг удаляется. Если у проекта уже был свой
   ``.inspectra.yml`` — он восстанавливается байт в байт.

Inspectra используется ТОЛЬКО как CLI (не библиотека, по ТЗ). Любая
ошибка запуска (CLI не найден, таймаут, ненулевой код возврата)
превращается в {"success": False, ...} — watcher продолжает работу.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from app.rag.code_rag import NO_CONTEXT_MESSAGE

from .utils import get_logger

logger = get_logger("deepseek-chat.inspectra")

DEFAULT_MODEL = "qwen2.5-coder-14b-32k"
DEFAULT_PROVIDER = "ollama"
DEFAULT_TIMEOUT = 600.0      # сек на один запуск Inspectra
DEFAULT_TOP_K = 5

CONFIG_NAME = ".inspectra.yml"
CONTEXT_CHUNK_LIMIT = 500    # символов чанка в контексте промпта

# Промпт ревью: фокус ТОЛЬКО баги и edge cases (ТЗ), severity +
# рекомендация для каждой находки
REVIEW_PROMPT = """Ты — опытный code reviewer. Проанализируй diff на баги и edge cases.

[КОНТЕКСТ ИЗ КОДА ПРОЕКТА]
{rag_context}

[ИНСТРУКЦИИ]
Найди:
- Потенциальные баги (логические ошибки, неверные условия)
- Edge cases, которые не обработаны
- Логические ошибки в условиях и циклах

Для каждой находки укажи severity (critical/high/medium/low) и рекомендацию.

Не пиши про стиль кода, форматирование и именование — только баги и edge cases.
"""


class InspectraRunner:
    """Запуск ``inspectra review`` с RAG-контекстом из кода проекта.

    Args:
        project_path: корень ПРОЕКТА-ревью (cwd запуска CLI; там же
            создаётся и удаляется временный .inspectra.yml).
        code_rag: CodeRAG с проиндексированной коллекцией project_code.
        model: модель Ollama для ревью (Modelfile с num_ctx 32768).
        provider: провайдер Inspectra (локальная Ollama).
        timeout: таймаут одного запуска CLI, сек.
        top_k: сколько RAG-чанков уходит в промпт.
        binary: имя/путь исполняемого файла Inspectra.
        repo: "owner/repo" для режима --pr: Inspectra читает репозиторий
            из переменной окружения GITHUB_REPOSITORY (конвенция GitHub
            Actions), а не из нашего GITHUB_REPO — подставляем сами.
    """

    def __init__(
        self,
        project_path: Union[str, Path],
        code_rag: Any,
        model: str = DEFAULT_MODEL,
        provider: str = DEFAULT_PROVIDER,
        timeout: float = DEFAULT_TIMEOUT,
        top_k: int = DEFAULT_TOP_K,
        binary: str = "inspectra",
        repo: str = "",
    ) -> None:
        self.project_path = Path(project_path)
        self.code_rag = code_rag
        self.model = str(model)
        self.provider = str(provider)
        self.timeout = float(timeout)
        self.top_k = max(1, int(top_k))
        self.binary = str(binary)
        self.repo = str(repo or "").strip()

    # ------------------------------------------------------------------
    # RAG-контекст
    # ------------------------------------------------------------------
    def build_rag_context(
        self,
        pr_title: str,
        pr_number: Optional[int] = None,
    ) -> str:
        """Собирает RAG-контекст промпта по заголовку PR.

        Ошибка RAG (Ollama недоступна, пустая коллекция) НЕ срывает
        ревью: возвращается заглушка — Inspectra отревьюит diff без
        контекста проекта.

        Args:
            pr_title: заголовок PR (по нему ищутся чанки).
            pr_number: номер PR (только для строки в промпте).

        Returns:
            Текст блока [КОНТЕКСТ ИЗ КОДА ПРОЕКТА] (не пустой).
        """
        try:
            results: List[Dict[str, Any]] = self.code_rag.search(
                pr_title, top_k=self.top_k)
        except Exception as exc:  # noqa: BLE001 — ревью важнее RAG
            logger.warning("RAG-поиск не удался, контекст пуст: %s", exc)
            results = []
        if not results:
            return NO_CONTEXT_MESSAGE
        header = "RAG-контекст из кода проекта:"
        if pr_number is not None:
            header = f"RAG-контекст из кода проекта (к PR #{pr_number}):"
        parts = [header]
        for result in results:
            metadata = result.get("metadata") or {}
            path = str(metadata.get("path")
                       or metadata.get("source") or "unknown.py")
            text = str(result.get("text") or "")[:CONTEXT_CHUNK_LIMIT]
            parts.append(f"\n--- {path} ---\n{text}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Конфиг и запуск
    # ------------------------------------------------------------------
    def write_inspectra_config(
        self,
        rag_context: str,
        pr_number: Optional[int] = None,
    ) -> Path:
        """Пишет временный .inspectra.yml с промптом ревью.

        Args:
            rag_context: текст из build_rag_context.
            pr_number: номер PR (для заголовка контекста).

        Returns:
            Путь к записанному конфигу.
        """
        prompt = REVIEW_PROMPT.format(
            rag_context=rag_context,
        ) + (f"\n[ПУЛЛ-РЕКВЕСТ]\n#{pr_number}\n" if pr_number is not None
             else "")
        config_path = self.project_path / CONFIG_NAME
        config_path.parent.mkdir(parents=True, exist_ok=True)
        # allow_unicode=False — файл в чистом ASCII (\uXXXX-экранирование):
        # inspectra открывает конфиг БЕЗ encoding= (config/loader.py:35),
        # на Windows это кодовая страница локали, и UTF-8-кириллица
        # роняла его с "charmap codec can't decode". ASCII читается любой
        # локалью; safe_load восстанавливает исходный текст один в один.
        config_path.write_text(
            yaml.dump({"prompt": prompt}, allow_unicode=False),
            encoding="utf-8",
        )
        return config_path

    def review_pr(self, pr_number: int, pr_title: str) -> Dict[str, Any]:
        """Полный цикл ревью PR; ошибки НЕ пробрасываются.

        Args:
            pr_number: номер PR.
            pr_title: заголовок PR (запрос RAG-поиска).

        Returns:
            {"success", "stdout", "stderr", "returncode"}: success —
            код возврата 0; у ошибок запуска (нет CLI, таймаут)
            returncode -1 и объяснение в stderr.
        """
        rag_context = self.build_rag_context(pr_title, pr_number)
        # Пользовательский конфиг бэкапится ДО записи временного,
        # иначе затёрли бы его безвозвратно
        original = self._backup_config()
        config_path = self.write_inspectra_config(rag_context, pr_number)

        cmd = [
            self.binary, "review",
            "--pr", str(int(pr_number)),
            "--provider", self.provider,
            "--model", self.model,
            "--post-comment",     # по умолчанию НЕТ — обязателен (ТЗ)
            "--pr-summary",
            "--config", str(config_path),
        ]
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_path),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                env=self._subprocess_env(),
            )
        except FileNotFoundError:
            return self._failure(
                f"Исполняемый файл {self.binary!r} не найден — установите "
                "Inspectra и проверьте, что он в PATH")
        except subprocess.TimeoutExpired:
            return self._failure(
                f"Inspectra не уложился в таймаут {self.timeout:.0f} с "
                "(PR слишком большой или модель не отвечает)")
        except OSError as exc:
            return self._failure(f"Запуск {self.binary!r} не удался: {exc}")
        finally:
            if original is None:
                self._cleanup_config(config_path)
            else:
                # Был свой .inspectra.yml — возвращаем байт в байт
                self._restore_config(original)

        success = result.returncode == 0
        if success:
            logger.info("PR #%s отревьюен (Inspectra, %s)",
                        pr_number, self.model)
        else:
            logger.warning(
                "Inspectra вернул код %s для PR #%s: %s",
                result.returncode, pr_number,
                str(result.stderr or "")[-500:])
        return {
            "success": success,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
            "returncode": result.returncode,
        }

    # ------------------------------------------------------------------
    # Внутреннее
    # ------------------------------------------------------------------
    def _subprocess_env(self) -> Dict[str, str]:
        """Окружение для CLI: os.environ + GITHUB_REPOSITORY + UTF-8.

        Inspectra (как и GitHub Actions) читает имя репозитория из
        GITHUB_REPOSITORY; наш ключ называется GITHUB_REPO, поэтому
        подставляем дубликат — токен GITHUB_TOKEN наследуется из .env
        как есть. PYTHONUTF8=1 включает UTF-8-режим Python'а inspectra
        (PEP 540): без него он читает конфиг и git-diff в кодировке
        локали Windows (cp1251) и падает на кириллице.
        """
        env = dict(os.environ)
        if self.repo:
            env["GITHUB_REPOSITORY"] = self.repo
        env["PYTHONUTF8"] = "1"
        return env

    @staticmethod
    def _cleanup_config(config_path: Path) -> None:
        """Удаляет временный конфиг (с коротким ретраем).

        После убийства процесса по таймауту Windows может ещё держать
        файл открытым несколько мгновений — первый unlink падает с
        PermissionError (найдено живым запуском: .inspectra.yml
        оставался в корне репозитория после таймаут-прогона).
        """
        for attempt in range(3):
            try:
                config_path.unlink(missing_ok=True)
                return
            except OSError as exc:
                if attempt == 2:
                    logger.warning(
                        "Не удалил временный конфиг %s: %s "
                        "(перезапишется следующим ревью)", config_path, exc)
                else:
                    time.sleep(0.5)

    @staticmethod
    def _failure(message: str) -> Dict[str, Any]:
        """Стандартный ответ об ошибке запуска (лог + success=False)."""
        logger.warning("Ревью не состоялось: %s", message)
        return {
            "success": False,
            "stdout": "",
            "stderr": message,
            "returncode": -1,
        }

    def _backup_config(self) -> Optional[bytes]:
        """Содержимое пользовательского .inspectra.yml (если был)."""
        config_path = self.project_path / CONFIG_NAME
        if not config_path.is_file():
            return None
        try:
            return config_path.read_bytes()
        except OSError as exc:
            logger.warning("Не прочитал существующий %s: %s",
                           config_path, exc)
            return None

    def _restore_config(self, original: Optional[bytes]) -> None:
        """Восстанавливает пользовательский конфиг поверх временного."""
        if original is None:
            return
        config_path = self.project_path / CONFIG_NAME
        try:
            config_path.write_bytes(original)
        except OSError as exc:
            logger.error("Не восстановил пользовательский %s: %s",
                         config_path, exc)
