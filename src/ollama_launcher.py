"""Автозапуск локального сервера Ollama и загрузка моделей (ТЗ ai_1_25).

Повторяет идею :class:`src.reminder_launcher.HttpMCPServerLauncher`:

* :meth:`OllamaLauncher.is_running` — TCP-проверка порта 11434: если
  Ollama уже работает, процесс НЕ перезапускается (повторный запуск
  приложения не дублирует сервер);
* :meth:`OllamaLauncher.launch` — старт ``ollama serve`` в НОВОМ окне
  консоли (``CREATE_NEW_CONSOLE``): логи сервера видны пользователю,
  процесс переживает закрытие приложения;
* :meth:`OllamaLauncher.ensure_running` — проверка → запуск → ожидание
  выхода на порт (используется при переключении провайдера);
* :meth:`OllamaLauncher.ensure_model` — проверка модели по /api/tags и
  ``ollama pull`` при первом использовании (загрузка занимает время и
  блокирует только вызвавший поток — UI вызывает её из asyncio.to_thread).

Если утилита ``ollama`` не найдена в PATH, возвращается понятное
сообщение об ошибке вместо падения — переключение провайдера просто
не выполняется.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from typing import Optional, Tuple

import httpx

from .utils import get_logger

logger = get_logger("deepseek-chat.ollama-launcher")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434
MODEL_PULL_TIMEOUT = 1800.0  # 30 минут на скачивание модели


class OllamaLauncher:
    """Управляет сервером Ollama и загрузкой моделей (без остановки).

    Attributes:
        host: адрес сервера (только 127.0.0.1).
        port: порт сервера (11434).
        api_base: база нативного API (http://127.0.0.1:11434).
        label: имя для сообщений пользователю.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        api_base: Optional[str] = None,
        label: str = "Ollama",
    ) -> None:
        """Настраивает лаунчер.

        Args:
            host: адрес сервера.
            port: порт сервера.
            api_base: база API; None — OLLAMA_BASE_URL или host:port.
            label: имя для сообщений.
        """
        self.host = str(host or DEFAULT_HOST)
        self.port = int(port or DEFAULT_PORT)
        self.api_base = (api_base or os.getenv("OLLAMA_BASE_URL")
                         or f"http://{self.host}:{self.port}").rstrip("/")
        self.label = str(label or "Ollama")

    @property
    def server_url(self) -> str:
        """База сервера (для сообщений)."""
        return f"http://{self.host}:{self.port}"

    def ollama_exe(self) -> Optional[str]:
        """Путь к утилите ``ollama`` (None — не найдена в PATH)."""
        return shutil.which("ollama")

    def is_running(self) -> bool:
        """True, если порт Ollama отвечает (TCP-соединение устанавливается)."""
        try:
            with socket.create_connection((self.host, self.port), timeout=0.5):
                return True
        except OSError:
            return False

    # Совместимость с именованием MCP-лаунчеров
    is_server_running = is_running

    def launch(self) -> Tuple[bool, str]:
        """Запускает ``ollama serve`` в новом окне, если ещё не работает.

        Returns:
            (started, message): started=True — процесс запущен;
            started=False — уже работает или запустить не удалось
            (текст ошибки — приложение не должно падать).
        """
        if self.is_running():
            return False, (f"{self.label} уже запущен на {self.server_url} "
                           f"— не перезапускаем.")
        ollama_exe = self.ollama_exe()
        if not ollama_exe:
            return False, (
                f"{self.label} не найден в PATH. Установите Ollama "
                f"(https://ollama.com) и проверьте, что команда "
                f"«ollama» доступна в терминале.")
        command = [ollama_exe, "serve"]
        try:
            if sys.platform == "win32":
                subprocess.Popen(  # noqa: S603 — фиксированный список аргументов
                    command,
                    creationflags=subprocess.CREATE_NEW_CONSOLE,  # type: ignore[attr-defined]
                    stdin=subprocess.DEVNULL,
                )
            else:
                subprocess.Popen(  # noqa: S603
                    command,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
        except OSError as exc:
            return False, f"Не удалось запустить {self.label}: {exc}"
        return True, f"{self.label} запускается в новом окне ({self.server_url})."

    def ensure_running(self, timeout: float = 30.0,
                       poll: float = 0.5) -> bool:
        """Гарантирует, что сервер работает: проверяет, запускает, ждёт порт.

        Args:
            timeout: сколько секунд ждать выхода сервера на порт.
            poll: интервал опроса порта.

        Returns:
            True — сервер отвечает (работал или поднят), False — не удалось.
        """
        if self.is_running():
            return True
        if self.ollama_exe() is None:
            logger.warning(
                "%s не найден в PATH — автозапуск невозможен.", self.label)
            return False
        _started, message = self.launch()
        logger.info("%s", message)
        deadline = time.monotonic() + max(0.0, float(timeout))
        while time.monotonic() < deadline:
            if self.is_running():
                return True
            time.sleep(poll)
        return False

    def has_model(self, model: str) -> bool:
        """Загружена ли модель (по /api/tags; сервер должен работать)."""
        base = str(model or "").split(":", 1)[0].strip().lower()
        try:
            response = httpx.get(f"{self.api_base}/api/tags", timeout=5.0)
        except Exception:
            return False
        if response.status_code != 200:
            return False
        try:
            models = response.json().get("models") or []
        except ValueError:
            return False
        return any(str(item.get("name", "")).lower().startswith(base)
                   for item in models)

    def ensure_model(self, model: str,
                     timeout: float = MODEL_PULL_TIMEOUT) -> Tuple[bool, str]:
        """Проверяет модель и загружает её через ``ollama pull``.

        Args:
            model: имя модели (например, ``qwen2.5:7b``).
            timeout: лимит времени на скачивание (pull большой модели
                занимает десятки минут — блокируется только вызвавший
                поток, UI живёт).

        Returns:
            (ok, message): ok=True — модель есть (была или скачана).
        """
        if self.has_model(model):
            return True, f"Модель {model} уже загружена."
        ollama_exe = self.ollama_exe()
        if not ollama_exe:
            return False, (
                f"{self.label} не найден в PATH — не удалось загрузить "
                f"модель {model}.")
        logger.info("Модель %s не найдена — загружаю (ollama pull)…", model)
        try:
            subprocess.run(  # noqa: S603 — фиксированный список аргументов
                [ollama_exe, "pull", model],
                check=True,
                timeout=float(timeout),
            )
        except subprocess.TimeoutExpired:
            return False, (f"Загрузка модели {model} превысила лимит "
                           f"{int(timeout)} с.")
        except (OSError, subprocess.CalledProcessError) as exc:
            return False, f"Не удалось загрузить модель {model}: {exc}"
        return True, f"Модель {model} загружена."

    def stop(self) -> None:
        """Ничего не делает: сервер должен жить после закрытия приложения."""
        return None
