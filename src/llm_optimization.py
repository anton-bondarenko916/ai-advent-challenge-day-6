"""Оптимизация локальной модели Ollama: настройки и теги квантов (ТЗ ai_1_28).

Две сущности:

* :class:`OptimizationSettings` — параметрический профиль инференса
  (temperature, num_predict, num_ctx, top_p, repeat_penalty) плюс
  опциональный системный промпт и выбранный квант. Хранится глобально
  в таблице ``app_state`` (ключ ``ollama_optimization``, JSON) и
  переживает перезапуск приложения. Хранилище передаётся клиентом
  (AppStateStore): модуль знает только протокол ``get``/``set``.
* :class:`OllamaTagManager` — список доступных тегов (квантов) модели
  из реестра ollama-models (пакет ``ollama-models``, клиент
  ``OllamaModelsClient.get_model``) с отметкой локально скачанных
  (по /api/tags работающего Ollama). Реестр внешний и может быть
  недоступен (сеть, пакет не установлен) — тогда используется
  захардкоженный список известных тегов (fallback по ТЗ).

Всё «неубиваемое»: ошибки сети/реестра/JSON превращаются в значения
по умолчанию или пустые списки, никогда — в исключение наверх.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .utils import get_logger

logger = get_logger("deepseek-chat.llm-optimization")

# Ключ в таблице app_state (глобальные настройки, вне профилей)
OPTIMIZATION_KEY = "ollama_optimization"

DEFAULT_OLLAMA_API_BASE = "http://localhost:11434"
OLLAMA_TAGS_TIMEOUT = 3.0


class OptimizationSettings:
    """Настройки оптимизации локальной модели (профиль инференса).

    Значения по умолчанию соответствуют поведению Ollama без
    оптимизации с рекомендациями ТЗ: num_predict 2048 (аналог
    max_tokens), num_ctx 8192 (для RAG-чатов, где контекст длинный).

    Attributes:
        enabled: применяются ли параметры к запросам (False — запрос
            уходит как есть, с дефолтами Ollama).
        temperature: температура (0.0–2.0).
        max_tokens: лимит токенов ответа (в Ollama — num_predict).
        num_ctx: размер контекстного окна (KV-кэш растёт линейно —
            большие значения могут не влезть в память).
        top_p: nucleus sampling (0.0–1.0).
        repeat_penalty: штраф за повторы (0.0–2.0).
        system_prompt: кастомный системный промпт; добавляется СВЕРХУ
            к промпту агента (не заменяет).
        selected_quant: выбранный квант (подпись из списка тегов).
        selected_tag: полный тег модели реестра (например,
            ``qwen2.5:7b-instruct-q4_K_M``).
    """

    DEFAULTS: Dict[str, Any] = {
        "enabled": False,
        "temperature": 0.7,
        "max_tokens": 2048,
        "num_ctx": 8192,
        "top_p": 0.95,
        "repeat_penalty": 1.1,
        "system_prompt": "",
        "selected_quant": "q4_K_M",
        "selected_tag": "",
    }

    # Допустимые диапазоны (обрезание, а не ошибка: кривое значение из
    # UI не должно ломать сохранение — приводим к ближайшему валидному)
    LIMITS: Dict[str, Tuple[float, float]] = {
        "temperature": (0.0, 2.0),
        "max_tokens": (1, 32768),
        "num_ctx": (512, 262144),
        "top_p": (0.0, 1.0),
        "repeat_penalty": (0.0, 2.0),
    }

    @staticmethod
    def sanitize(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Приводит настройки к валидному виду (типы и диапазоны).

        Неизвестные ключи отбрасываются, кривые значения заменяются
        дефолтами, числовые обрезаются до допустимого диапазона.

        Args:
            raw: сырые настройки (например, из JSON app_state или UI).

        Returns:
            Чистый словарь настроек (копия, без ссылок на raw).
        """
        settings = OptimizationSettings.DEFAULTS.copy()
        if not isinstance(raw, dict):
            return settings
        settings["enabled"] = bool(raw.get("enabled", False))
        for key, (low, high) in OptimizationSettings.LIMITS.items():
            try:
                value = float(raw.get(key, settings[key]))
            except (TypeError, ValueError):
                continue  # кривое значение — остаётся дефолт
            if value != value:  # NaN
                continue
            settings[key] = type(settings[key])(min(max(value, low), high))
        prompt = raw.get("system_prompt")
        if isinstance(prompt, str):
            settings["system_prompt"] = prompt.strip()
        for key in ("selected_quant", "selected_tag"):
            value = raw.get(key)
            if isinstance(value, str):
                settings[key] = value.strip()
        return settings

    @staticmethod
    def load(store: Any) -> Dict[str, Any]:
        """Загружает настройки из app_state (или дефолт).

        Args:
            store: хранилище с протоколом AppStateStore (``get``/``set``);
                None — сразу дефолт.

        Returns:
            Словарь настроек (санитизированный).
        """
        if store is None:
            return OptimizationSettings.DEFAULTS.copy()
        try:
            raw = store.get(OPTIMIZATION_KEY)
        except Exception as exc:  # noqa: BLE001 — хранилище не критично
            logger.warning("Оптимизация: чтение не удалось: %s", exc)
            return OptimizationSettings.DEFAULTS.copy()
        if not raw:
            return OptimizationSettings.DEFAULTS.copy()
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            logger.warning(
                "Оптимизация: кривой JSON в %s — откат к дефолтам",
                OPTIMIZATION_KEY)
            return OptimizationSettings.DEFAULTS.copy()
        return OptimizationSettings.sanitize(data)

    @staticmethod
    def save(store: Any, settings: Dict[str, Any]) -> bool:
        """Сохраняет настройки в app_state (upsert, JSON).

        Args:
            store: хранилище (``get``/``set``); None — ничего не делать.
            settings: настройки (санитизируются перед записью).

        Returns:
            True — сохранено; False — хранилище недоступно (не фатально).
        """
        if store is None:
            return False
        clean = OptimizationSettings.sanitize(settings)
        try:
            return bool(store.set(
                OPTIMIZATION_KEY,
                json.dumps(clean, ensure_ascii=False),
            ))
        except Exception as exc:  # noqa: BLE001 — хранилище не критично
            logger.warning("Оптимизация: запись не удалась: %s", exc)
            return False

    @staticmethod
    def reset() -> Dict[str, Any]:
        """Дефолтные настройки (новый словарь)."""
        return OptimizationSettings.DEFAULTS.copy()


class OllamaTagManager:
    """Список тегов (квантов) модели: реестр ollama-models + fallback.

    Реестр — внешний сервис (пакет ``ollama-models``, хост
    ollama.devcomfort.me). Любая ошибка (пакета нет, сеть недоступна,
    4xx/5xx) переводит список в fallback — захардкоженные теги моделей,
    известных приложению (реестр LLMProviderFactory: qwen2.5, llama3.1).
    """

    # Fallback по ТЗ: известные теги для моделей из фабрики провайдеров
    FALLBACK_TAGS: Dict[str, List[str]] = {
        "qwen2.5": [
            "qwen2.5:7b",
            "qwen2.5:7b-instruct-q4_K_M",
            "qwen2.5:7b-instruct-q5_K_M",
            "qwen2.5:7b-instruct-q8_0",
        ],
        "llama3.1": [
            "llama3.1:8b",
            "llama3.1:8b-instruct-q4_K_M",
            "llama3.1:8b-instruct-q5_K_M",
            "llama3.1:8b-instruct-q8_0",
        ],
    }

    @staticmethod
    def get_available_tags(model_name: str,
                           api_base: Optional[str] = None) -> List[Dict[str, Any]]:
        """Доступные теги модели с отметкой локально скачанных.

        Args:
            model_name: имя модели (``qwen2.5:7b``) — базовое имя идёт
                в реестр, размер (``7b``) фильтрует нерелевантные теги
                (например, 0.5b/14b у того же qwen2.5).
            api_base: база нативного API Ollama (для /api/tags);
                None — OLLAMA_BASE_URL или localhost:11434.

        Returns:
            Список ``{"tag": str, "quant": str, "local": bool}``,
            отсортированный как в реестре. Пустой список — модель
            неизвестна ни реестру, ни fallback'у.
        """
        base_name = str(model_name or "").split(":", 1)[0].strip().lower()
        if not base_name:
            return []
        size = OllamaTagManager._size_token(model_name)
        local_names = OllamaTagManager.get_local_tags(api_base)
        tags = OllamaTagManager._fetch_registry_tags(base_name)
        if tags is None:  # реестр недоступен — fallback по ТЗ
            tags = list(OllamaTagManager.FALLBACK_TAGS.get(base_name, []))
        if size:
            relevant = [t for t in tags
                        if size in str(t).lower()
                        or str(t).lower().endswith("latest")]
            if relevant:  # пустой фильтр — оставляем весь список
                tags = relevant
        result: List[Dict[str, Any]] = []
        for tag in tags:
            full_tag = str(tag)
            if ":" not in full_tag:
                full_tag = f"{base_name}:{full_tag}"
            result.append({
                "tag": full_tag,
                "quant": OllamaTagManager._extract_quant(full_tag),
                "local": OllamaTagManager._is_local(full_tag, local_names),
            })
        return result

    @staticmethod
    def get_local_tags(api_base: Optional[str] = None) -> List[str]:
        """Локально скачанные теги (по /api/tags; ошибка — пусто)."""
        base = (str(api_base).rstrip("/") if api_base
                else os.getenv("OLLAMA_BASE_URL",
                               DEFAULT_OLLAMA_API_BASE).rstrip("/"))
        try:
            response = httpx.get(f"{base}/api/tags",
                                 timeout=OLLAMA_TAGS_TIMEOUT)
        except Exception:  # noqa: BLE001 — сервер не запущен: это норма
            return []
        if response.status_code != 200:
            return []
        try:
            models = response.json().get("models") or []
        except ValueError:
            return []
        return [str(item.get("name") or "") for item in models
                if item.get("name")]

    @staticmethod
    def pull_model(tag: str) -> Tuple[bool, str]:
        """Скачивает модель с указанным тегом (``ollama pull``).

        Делегирует :meth:`src.ollama_launcher.OllamaLauncher.ensure_model`
        (та же утилита, что и переключение провайдера: если модель уже
        есть — мгновенный пропуск; загрузка блокирует только вызывающий
        поток, UI зовёт из asyncio.to_thread).

        Returns:
            (успех, сообщение для пользователя).
        """
        from .ollama_launcher import OllamaLauncher  # лениво: без циклов

        clean_tag = str(tag or "").strip()
        if not clean_tag:
            return False, "Тег модели не выбран."
        try:
            return OllamaLauncher().ensure_model(clean_tag)
        except Exception as exc:  # noqa: BLE001 — загрузка не критична
            logger.warning("Ошибка загрузки модели %s: %s", clean_tag, exc)
            return False, f"Ошибка загрузки модели {clean_tag}: {exc}"

    # ------------------------------------------------------------------
    # Внутреннее
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_registry_tags(base_name: str) -> Optional[List[str]]:
        """Теги из реестра ollama-models (None — реестр недоступен).

        Пакет ``ollama-models`` опциональный: ImportError — обычная
        ситуация (fallback), не ошибка конфигурации.
        """
        try:
            from ollama_models import OllamaModelsClient
        except ImportError:
            logger.info(
                "Пакет ollama-models не установлен — список квантов "
                "собирается из fallback-списка.")
            return None
        try:
            model = OllamaModelsClient().get_model(base_name)
        except Exception as exc:  # noqa: BLE001 — сеть/4xx/5xx
            logger.warning(
                "Реестр ollama-models недоступен (%s) — fallback.", exc)
            return None
        tags = [str(tag) for tag in (getattr(model, "tags", None) or [])]
        default_tag = getattr(model, "default_tag", None)
        if default_tag and str(default_tag) not in tags:
            tags.insert(0, str(default_tag))
        return tags

    @staticmethod
    def _size_token(model_name: str) -> str:
        """Токен размера из имени модели (``qwen2.5:7b`` → ``7b``)."""
        match = re.search(r"(\d+(?:\.\d+)?b)", str(model_name or "").lower())
        return match.group(1) if match else ""

    @staticmethod
    def _extract_quant(tag: str) -> str:
        """Квант из тега (``7b-instruct-q4_K_M`` → ``Q4_K_M``).

        Покрывает форматы q8_0 / q4_K_M / q6_K / q5_K_S; квант не
        найден — пустая строка (UI покажет тег целиком).
        """
        match = re.search(r"q\d+(?:_\d+)?(?:_[kK](?:_[sSmMlL])?)?",
                          str(tag or "").lower())
        return match.group(0).upper() if match else ""

    @staticmethod
    def _is_local(full_tag: str, local_names: List[str]) -> bool:
        """Скачан ли тег (``:latest`` в локальном списке опускается)."""
        def normalize(name: str) -> str:
            return str(name or "").lower().removesuffix(":latest")

        target = normalize(full_tag)
        return any(normalize(name) == target for name in local_names)
