"""Тесты оптимизации локальной модели Ollama (ТЗ ai_1_28).

Проверяются офлайн (без сети, реестра и запущенного Ollama):

* OptimizationSettings: дефолты, санитизация (типы, диапазоны, кривой
  JSON), сохранение/загрузка в app_state (SQLite), сброс;
* OllamaProvider: при выключенной оптимизации запрос не меняется, при
  включенной — extra_body={"options": {...}} (num_predict вместо
  max_tokens!) и системный промпт СВЕРХУ (копия — история не мутируется);
* ядро: set_ollama_optimization (горячее применение, перенос при
  switch_provider), применение только в основном ходе (_call_api),
  DeepSeek-путь не затрагивается;
* OllamaTagManager: теги из реестра ollama-models (клиент замокан),
  отметки локальности по /api/tags, кванты из тегов, fallback при
  недоступном реестре и без пакета, pull_model;
* UI-панель: блокировка полей выключенным переключателем, применение и
  сохранение, сброс, выбор кванта, кнопка загрузки, ошибка кривого числа.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import DeepSeekAgent  # noqa: E402
from src.app_state import AppStateStore  # noqa: E402
from src.llm_optimization import (  # noqa: E402
    OPTIMIZATION_KEY,
    OptimizationSettings,
    OllamaTagManager,
)
from src.llm_providers import (  # noqa: E402
    DeepSeekProvider,
    LLMProviderFactory,
    OllamaProvider,
)

from tests.test_app_agent import make_agent  # noqa: E402
from tests.test_flet_views import make_app  # noqa: E402

ENABLED = {
    "enabled": True,
    "temperature": 0.1,
    "max_tokens": 512,
    "num_ctx": 4096,
    "top_p": 0.8,
    "repeat_penalty": 1.3,
    "system_prompt": "Отвечай кратко.",
}


# ----------------------------------------------------------------------
# OptimizationSettings: дефолты, санитизация, app_state
# ----------------------------------------------------------------------
class OptimizationSettingsTest(unittest.TestCase):
    """Санитизация и хранение настроек в app_state."""

    def test_defaults(self) -> None:
        settings = OptimizationSettings.reset()
        self.assertFalse(settings["enabled"])
        self.assertEqual(settings["temperature"], 0.7)
        self.assertEqual(settings["max_tokens"], 2048)
        self.assertEqual(settings["num_ctx"], 8192)
        self.assertEqual(settings["top_p"], 0.95)
        self.assertEqual(settings["repeat_penalty"], 1.1)
        self.assertEqual(settings["system_prompt"], "")
        self.assertEqual(settings["selected_quant"], "q4_K_M")

    def test_sanitize_keeps_known_keys_only(self) -> None:
        settings = OptimizationSettings.sanitize({
            "enabled": True,
            "temperature": 0.2,
            "unknown_key": "мусор",
        })
        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["temperature"], 0.2)
        self.assertNotIn("unknown_key", settings)
        # Остальные параметры — дефолты
        self.assertEqual(settings["max_tokens"],
                         OptimizationSettings.DEFAULTS["max_tokens"])

    def test_sanitize_clamps_ranges_and_repairs_types(self) -> None:
        settings = OptimizationSettings.sanitize({
            "temperature": 5.0,        # > 2.0
            "max_tokens": "512",       # строка с числом
            "num_ctx": 100,            # < 512
            "top_p": "кривое",         # не число → дефолт
            "repeat_penalty": -1,      # < 0
        })
        self.assertEqual(settings["temperature"], 2.0)
        self.assertEqual(settings["max_tokens"], 512)
        self.assertEqual(settings["num_ctx"], 512)
        self.assertEqual(settings["top_p"], 0.95)
        self.assertEqual(settings["repeat_penalty"], 0.0)

    def test_sanitize_non_dict_returns_defaults(self) -> None:
        for raw in (None, "строка", 42, ["список"]):
            self.assertEqual(
                OptimizationSettings.sanitize(raw),
                OptimizationSettings.DEFAULTS.copy())

    def test_save_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AppStateStore(Path(tmp) / "app_state.db")
            settings = dict(ENABLED, system_prompt="Тест промпт")
            self.assertTrue(OptimizationSettings.save(store, settings))
            loaded = OptimizationSettings.load(store)
            self.assertTrue(loaded["enabled"])
            self.assertEqual(loaded["temperature"], 0.1)
            self.assertEqual(loaded["system_prompt"], "Тест промпт")
            # Ключ в app_state — ровно тот, что указан в ТЗ
            import json
            raw = json.loads(store.get(OPTIMIZATION_KEY))
            self.assertEqual(raw["selected_quant"], "q4_K_M")

    def test_load_missing_and_corrupted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AppStateStore(Path(tmp) / "app_state.db")
            # Ключа нет → дефолт
            self.assertEqual(OptimizationSettings.load(store),
                             OptimizationSettings.DEFAULTS.copy())
            # Кривой JSON → дефолт (не исключение)
            store.set(OPTIMIZATION_KEY, "{не-json")
            self.assertEqual(OptimizationSettings.load(store),
                             OptimizationSettings.DEFAULTS.copy())

    def test_load_without_store_and_broken_store(self) -> None:
        self.assertEqual(OptimizationSettings.load(None),
                         OptimizationSettings.DEFAULTS.copy())

        class _Broken:
            def get(self, key):
                raise RuntimeError("db locked")

        self.assertEqual(OptimizationSettings.load(_Broken()),
                         OptimizationSettings.DEFAULTS.copy())

    def test_save_without_store_is_noop(self) -> None:
        self.assertFalse(OptimizationSettings.save(None, dict(ENABLED)))


# ----------------------------------------------------------------------
# OllamaProvider: применение оптимизации к запросу
# ----------------------------------------------------------------------
class OllamaProviderOptimizationTest(unittest.TestCase):
    """apply_optimization: выключено/включено, промпт, копии сообщений."""

    def test_disabled_request_untouched(self) -> None:
        provider = OllamaProvider(model="qwen2.5:7b")
        self.assertFalse(provider.enabled)
        request = {"model": "qwen2.5:7b",
                   "messages": [{"role": "user", "content": "тест"}]}
        # Возвращается тот же объект — модификаций нет вовсе
        self.assertIs(provider.apply_optimization(request), request)

    def test_enabled_injects_ollama_options(self) -> None:
        provider = OllamaProvider(model="qwen2.5:7b",
                                  optimization=dict(ENABLED))
        self.assertTrue(provider.enabled)
        patched = provider.apply_optimization({
            "model": "qwen2.5:7b",
            "messages": [{"role": "user", "content": "тест"}],
            "temperature": 0.7,
            "max_tokens": 1000,
        })
        options = patched["extra_body"]["options"]
        # temperature / top_p / repeat_penalty — как в настройках
        self.assertEqual(options["temperature"], 0.1)
        self.assertEqual(options["top_p"], 0.8)
        self.assertEqual(options["repeat_penalty"], 1.3)
        # max_tokens транслируется в num_predict (аналог в Ollama)
        self.assertEqual(options["num_predict"], 512)
        self.assertNotIn("max_tokens", options)
        self.assertEqual(options["num_ctx"], 4096)

    def test_enabled_prepends_system_prompt_on_top(self) -> None:
        provider = OllamaProvider(model="qwen2.5:7b",
                                  optimization=dict(ENABLED))
        base_prompt = "Ты полезный ассистент."
        patched = provider.apply_optimization({
            "model": "qwen2.5:7b",
            "messages": [
                {"role": "system", "content": base_prompt},
                {"role": "user", "content": "привет"},
            ],
        })
        system = patched["messages"][0]
        self.assertEqual(system["role"], "system")
        self.assertTrue(system["content"].startswith("Отвечай кратко."))
        self.assertIn(base_prompt, system["content"])
        # Ровно один system-блок, промпт ДОБАВЛЕН СВЕРХУ, не заменён
        self.assertEqual(
            patched["messages"][0]["content"].count("system"), 0)
        self.assertEqual(len(patched["messages"]), 2)

    def test_enabled_inserts_system_when_missing(self) -> None:
        provider = OllamaProvider(model="qwen2.5:7b",
                                  optimization=dict(ENABLED))
        patched = provider.apply_optimization({
            "model": "qwen2.5:7b",
            "messages": [{"role": "user", "content": "привет"}],
        })
        self.assertEqual(patched["messages"][0],
                         {"role": "system", "content": "Отвечай кратко."})
        self.assertEqual(patched["messages"][1]["role"], "user")

    def test_source_messages_not_mutated(self) -> None:
        """История ядра не мутируется: патчится копия сообщений."""
        provider = OllamaProvider(model="qwen2.5:7b",
                                  optimization=dict(ENABLED))
        system = {"role": "system", "content": "база"}
        user = {"role": "user", "content": "вопрос"}
        messages = [system, user]
        provider.apply_optimization({"messages": messages})
        self.assertEqual(system["content"], "база")
        self.assertEqual(user["content"], "вопрос")
        self.assertEqual(len(messages), 2)

    def test_empty_system_prompt_disables_prepend(self) -> None:
        settings = dict(ENABLED, system_prompt="   ")
        provider = OllamaProvider(model="qwen2.5:7b",
                                  optimization=settings)
        patched = provider.apply_optimization({
            "messages": [{"role": "system", "content": "база"}],
        })
        self.assertEqual(patched["messages"][0]["content"], "база")

    def test_extra_body_merges_with_existing(self) -> None:
        provider = OllamaProvider(model="qwen2.5:7b",
                                  optimization=dict(ENABLED))
        patched = provider.apply_optimization({
            "messages": [],
            "extra_body": {"custom": 1},
        })
        self.assertEqual(patched["extra_body"]["custom"], 1)
        self.assertIn("options", patched["extra_body"])

    def test_set_optimization_hot_toggle(self) -> None:
        provider = OllamaProvider(model="qwen2.5:7b")
        request = {"messages": [{"role": "user", "content": "т"}]}
        self.assertIs(provider.apply_optimization(request), request)
        provider.set_optimization(dict(ENABLED))
        self.assertTrue(provider.enabled)
        patched = provider.apply_optimization(request)
        self.assertIn("options", patched["extra_body"])
        provider.set_optimization({"enabled": False, "temperature": 0.5})
        self.assertFalse(provider.enabled)
        self.assertIs(provider.apply_optimization(request), request)

    def test_chat_completion_applies_optimization(self) -> None:
        captured = {}

        class _Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                message = SimpleNamespace(content="ответ", tool_calls=None)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=message)], usage=None)

        provider = OllamaProvider(model="qwen2.5:7b",
                                  optimization=dict(ENABLED))
        provider.client = SimpleNamespace(
            chat=SimpleNamespace(completions=_Completions()))
        result = provider.chat_completion(
            [{"role": "system", "content": "база"},
             {"role": "user", "content": "тест"}])
        self.assertEqual(result["content"], "ответ")
        self.assertEqual(captured["extra_body"]["options"]["num_predict"],
                         512)
        self.assertTrue(
            captured["messages"][0]["content"].startswith("Отвечай кратко."))


# ----------------------------------------------------------------------
# Фабрика и ядро: перенос настроек в провайдеров
# ----------------------------------------------------------------------
class FactoryAndCoreOptimizationTest(unittest.TestCase):
    """Фабрика получает optimization; ядро хранит и переносит настройки."""

    def test_factory_passes_optimization_to_ollama(self) -> None:
        provider = LLMProviderFactory.create(
            "ollama_qwen", optimization=dict(ENABLED))
        self.assertIsInstance(provider, OllamaProvider)
        self.assertTrue(provider.enabled)
        self.assertEqual(provider.optimization["temperature"], 0.1)
        # DeepSeek-провайдер создаётся без ошибок (настройка игнорируется)
        deepseek = LLMProviderFactory.create(
            "deepseek", api_key="k", optimization=dict(ENABLED))
        self.assertIsInstance(deepseek, DeepSeekProvider)

    def test_factory_without_optimization_keeps_defaults(self) -> None:
        provider = LLMProviderFactory.create("ollama_qwen")
        self.assertFalse(provider.enabled)
        self.assertEqual(provider.optimization, {})

    def test_set_ollama_optimization_sanitizes_and_stores(self) -> None:
        core = DeepSeekAgent(api_key="test", mock_mode=True, storage=None)
        clean = core.set_ollama_optimization(
            dict(ENABLED, temperature=99.0))
        self.assertEqual(clean["temperature"], 2.0)
        self.assertEqual(core.ollama_optimization["temperature"], 2.0)
        # Свойство возвращает копию
        core.ollama_optimization["enabled"] = "мусор"
        self.assertTrue(core.ollama_optimization["enabled"])

    def test_set_applies_to_active_ollama_provider(self) -> None:
        provider = OllamaProvider(model="qwen2.5:7b")
        core = DeepSeekAgent(api_key="test", mock_mode=True, storage=None,
                             provider=provider)
        core.set_ollama_optimization(dict(ENABLED))
        self.assertTrue(provider.enabled)
        self.assertEqual(provider.optimization["num_ctx"], 4096)
        # Выключение — тоже горячее
        core.set_ollama_optimization({"enabled": False})
        self.assertFalse(provider.enabled)

    def test_switch_provider_carries_optimization(self) -> None:
        core = DeepSeekAgent(api_key="test", mock_mode=True, storage=None)
        core.set_ollama_optimization(dict(ENABLED))
        core.switch_provider("ollama_qwen")
        provider = core.provider
        self.assertIsInstance(provider, OllamaProvider)
        self.assertTrue(provider.enabled)
        self.assertEqual(provider.optimization["repeat_penalty"], 1.3)
        self.assertTrue(provider.optimization["system_prompt"] != "")


# ----------------------------------------------------------------------
# Ядро: оптимизация в основном ходе диалога
# ----------------------------------------------------------------------
class _FakeResponseFactory:
    """Ответ OpenAI SDK-заглушка (content без tool_calls)."""

    @staticmethod
    def make(content: str = "локальный ответ"):
        message = SimpleNamespace(content=content, tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)], usage=None)


class CallApiOptimizationTest(unittest.TestCase):
    """_call_api: параметры уходят в API только для Ollama-провайдера."""

    def _core_with_client(self, provider) -> tuple:
        core = DeepSeekAgent(api_key="test", mock_mode=False, storage=None,
                             provider=provider)
        captured = {}

        def _create(**kwargs):
            captured.update(kwargs)
            return _FakeResponseFactory.make()

        core._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=_create)))
        return core, captured

    def test_ollama_request_contains_options_and_prompt(self) -> None:
        provider = OllamaProvider(model="qwen2.5:7b",
                                  optimization=dict(ENABLED))
        core, captured = self._core_with_client(provider)
        core.send_message("привет")
        options = captured["extra_body"]["options"]
        self.assertEqual(options["num_predict"], 512)
        self.assertEqual(options["num_ctx"], 4096)
        # Кастомный промпт СВЕРХУ системного промпта агента
        system = captured["messages"][0]["content"]
        self.assertTrue(system.startswith("Отвечай кратко."))
        self.assertIn("Ты полезный AI-ассистент", system)
        # История ядра без «Отвечай кратко» — патчилась копия
        history_system = core._message_history[0]["content"]
        self.assertNotIn("Отвечай кратко", history_system)

    def test_disabled_ollama_request_has_no_extra_body(self) -> None:
        provider = OllamaProvider(model="qwen2.5:7b",
                                  optimization={"enabled": False})
        core, captured = self._core_with_client(provider)
        core.send_message("привет")
        self.assertNotIn("extra_body", captured)

    def test_deepseek_request_untouched(self) -> None:
        core, captured = self._core_with_client(None)
        core.send_message("привет")
        self.assertNotIn("extra_body", captured)
        system = captured["messages"][0]["content"]
        self.assertNotIn("Отвечай кратко", system)

    def test_service_calls_not_optimized(self) -> None:
        """Служебный вызов (_call_service_api) без options и промпта."""
        provider = OllamaProvider(model="qwen2.5:7b",
                                  optimization=dict(ENABLED))
        core, captured = self._core_with_client(provider)
        core._call_service_api({
            "model": core.model,
            "messages": [{"role": "system", "content": "служба"},
                         {"role": "user", "content": "данные"}],
            "max_tokens": 100,
        })
        self.assertNotIn("extra_body", captured)
        self.assertEqual(captured["messages"][0]["content"], "служба")


# ----------------------------------------------------------------------
# OllamaTagManager: реестр, локальность, fallback
# ----------------------------------------------------------------------
class _FakeModelTags:
    """ModelTags-заглушка реестра ollama-models."""

    def __init__(self, tags, default_tag=None) -> None:
        self.tags = list(tags)
        self.default_tag = default_tag


class ExtractQuantTest(unittest.TestCase):
    """Извлечение кванта и размера из тегов."""

    def test_extract_quant(self) -> None:
        extract = OllamaTagManager._extract_quant
        self.assertEqual(extract("qwen2.5:7b-instruct-q4_K_M"), "Q4_K_M")
        self.assertEqual(extract("qwen2.5:7b-instruct-q8_0"), "Q8_0")
        self.assertEqual(extract("llama3.1:8b-instruct-q6_k"), "Q6_K")
        self.assertEqual(extract("model:7b-q5_K_S"), "Q5_K_S")
        self.assertEqual(extract("qwen2.5:7b"), "")

    def test_size_token(self) -> None:
        size = OllamaTagManager._size_token
        self.assertEqual(size("qwen2.5:7b"), "7b")
        self.assertEqual(size("llama3.1:8b"), "8b")
        self.assertEqual(size("qwen2.5:0.5b"), "0.5b")
        self.assertEqual(size("qwen2.5"), "")


class OllamaTagManagerTest(unittest.TestCase):
    """Список тегов: реестр (замокан), локальность, fallback."""

    def test_registry_tags_filtered_by_size_and_marked_local(self) -> None:
        registry = [
            "qwen2.5:0.5b",
            "qwen2.5:7b",
            "qwen2.5:7b-instruct-q4_K_M",
            "qwen2.5:7b-instruct-q8_0",
            "qwen2.5:14b",
        ]
        with mock.patch.object(
                OllamaTagManager, "_fetch_registry_tags",
                return_value=list(registry)), \
                mock.patch.object(
                    OllamaTagManager, "get_local_tags",
                    return_value=["qwen2.5:7b",
                                  "qwen2.5:7b-instruct-q8_0"]):
            tags = OllamaTagManager.get_available_tags("qwen2.5:7b")
        by_tag = {item["tag"]: item for item in tags}
        # Нерелевантные размеры отфильтрованы
        self.assertNotIn("qwen2.5:0.5b", by_tag)
        self.assertNotIn("qwen2.5:14b", by_tag)
        self.assertTrue(by_tag["qwen2.5:7b"]["local"])
        self.assertEqual(by_tag["qwen2.5:7b-instruct-q4_K_M"]["quant"],
                         "Q4_K_M")
        self.assertFalse(by_tag["qwen2.5:7b-instruct-q4_K_M"]["local"])
        self.assertTrue(by_tag["qwen2.5:7b-instruct-q8_0"]["local"])

    def test_bare_registry_tag_gets_base_name_prefix(self) -> None:
        with mock.patch.object(OllamaTagManager, "_fetch_registry_tags",
                               return_value=["latest", "7b"]), \
                mock.patch.object(OllamaTagManager, "get_local_tags",
                                  return_value=[]):
            tags = OllamaTagManager.get_available_tags("qwen2.5:7b")
        self.assertEqual([item["tag"] for item in tags],
                         ["qwen2.5:latest", "qwen2.5:7b"])

    def test_fallback_when_registry_unavailable(self) -> None:
        with mock.patch.object(OllamaTagManager, "_fetch_registry_tags",
                               return_value=None), \
                mock.patch.object(OllamaTagManager, "get_local_tags",
                                  return_value=["qwen2.5:7b"]):
            tags = OllamaTagManager.get_available_tags("qwen2.5:7b")
        quants = [item["quant"] for item in tags]
        self.assertIn("Q4_K_M", quants)
        self.assertIn("Q8_0", quants)
        # Локальность отмечается и в fallback-списке
        by_tag = {item["tag"]: item for item in tags}
        self.assertTrue(by_tag["qwen2.5:7b"]["local"])

    def test_fallback_llama3_1(self) -> None:
        with mock.patch.object(OllamaTagManager, "_fetch_registry_tags",
                               return_value=None), \
                mock.patch.object(OllamaTagManager, "get_local_tags",
                                  return_value=[]):
            tags = OllamaTagManager.get_available_tags("llama3.1:8b")
        self.assertTrue(tags)
        self.assertTrue(all(item["tag"].startswith("llama3.1")
                            for item in tags))

    def test_unknown_model_without_registry_returns_empty(self) -> None:
        with mock.patch.object(OllamaTagManager, "_fetch_registry_tags",
                               return_value=None), \
                mock.patch.object(OllamaTagManager, "get_local_tags",
                                  return_value=[]):
            self.assertEqual(
                OllamaTagManager.get_available_tags("mistral:7b"), [])

    def test_empty_model_name_returns_empty(self) -> None:
        self.assertEqual(OllamaTagManager.get_available_tags(""), [])
        self.assertEqual(OllamaTagManager.get_available_tags(None), [])

    def test_fetch_registry_tags_without_package(self) -> None:
        """Пакет ollama-models не установлен → fallback (не падение)."""
        with mock.patch.dict(sys.modules, {"ollama_models": None}):
            self.assertIsNone(
                OllamaTagManager._fetch_registry_tags("qwen2.5"))

    def test_fetch_registry_tags_on_http_error(self) -> None:
        from ollama_models import OllamaModelsClient

        with mock.patch.object(OllamaModelsClient, "get_model",
                               side_effect=OSError("403 Forbidden")):
            self.assertIsNone(
                OllamaTagManager._fetch_registry_tags("qwen2.5"))

    def test_fetch_registry_tags_returns_tags_and_default(self) -> None:
        from ollama_models import OllamaModelsClient

        fake = _FakeModelTags(
            ["qwen2.5:7b", "qwen2.5:7b-instruct-q4_K_M"],
            default_tag="qwen2.5:7b")
        with mock.patch.object(OllamaModelsClient, "get_model",
                               return_value=fake):
            tags = OllamaTagManager._fetch_registry_tags("qwen2.5")
        self.assertEqual(tags,
                         ["qwen2.5:7b", "qwen2.5:7b-instruct-q4_K_M"])

    def test_get_local_tags_reads_api(self) -> None:
        import src.llm_optimization as module

        response = SimpleNamespace(status_code=200, json=lambda: {
            "models": [{"name": "qwen2.5:7b"},
                       {"name": "nomic-embed-text"}]})
        with mock.patch.object(module.httpx, "get",
                               return_value=response) as mocked:
            names = OllamaTagManager.get_local_tags()
        self.assertEqual(names, ["qwen2.5:7b", "nomic-embed-text"])
        self.assertIn("/api/tags", mocked.call_args.args[0])

    def test_get_local_tags_on_error_returns_empty(self) -> None:
        import src.llm_optimization as module

        with mock.patch.object(module.httpx, "get",
                               side_effect=OSError("refused")):
            self.assertEqual(OllamaTagManager.get_local_tags(), [])
        response = SimpleNamespace(status_code=500, json=lambda: {})
        with mock.patch.object(module.httpx, "get",
                               return_value=response):
            self.assertEqual(OllamaTagManager.get_local_tags(), [])

    def test_pull_model_delegates_to_launcher(self) -> None:
        from src.ollama_launcher import OllamaLauncher

        with mock.patch.object(OllamaLauncher, "ensure_model",
                               return_value=(True, "модель загружена")) as m:
            ok, message = OllamaTagManager.pull_model("qwen2.5:7b")
        self.assertTrue(ok)
        self.assertEqual(message, "модель загружена")
        m.assert_called_once_with("qwen2.5:7b")

    def test_pull_model_empty_tag(self) -> None:
        ok, message = OllamaTagManager.pull_model("  ")
        self.assertFalse(ok)
        self.assertIn("не выбран", message)


# ----------------------------------------------------------------------
# UI-панель оптимизации (офлайн, page=None)
# ----------------------------------------------------------------------
class _StubAppState:
    """AppStateStore-заглушка (словарь в памяти)."""

    def __init__(self) -> None:
        self.saved: dict = {}

    def get(self, key: str):
        return self.saved.get(key)

    def set(self, key: str, value: str) -> bool:
        self.saved[key] = value
        return True


class OptimizationPanelTest(unittest.TestCase):
    """Панель «Оптимизация локальной модели»: поля, сохранение, сброс."""

    def setUp(self) -> None:
        from app.ui.optimization_view import OptimizationPanel

        self.store = _StubAppState()
        self.agent = make_agent()
        self.app = make_app(self.agent)
        self.app.app_state = self.store
        self.panel = OptimizationPanel(self.app)
        self.panel.build()

    def _enable(self) -> None:
        self.panel.enabled_switch.value = True
        self.panel._on_toggle()

    def test_fields_disabled_by_default(self) -> None:
        self.assertFalse(self.panel.enabled_switch.value)
        for field in (self.panel.temperature_field,
                      self.panel.max_tokens_field,
                      self.panel.num_ctx_field,
                      self.panel.top_p_field,
                      self.panel.repeat_penalty_field,
                      self.panel.system_prompt_field,
                      self.panel.quant_dropdown):
            self.assertTrue(field.disabled)

    def test_toggle_enables_fields_and_applies_to_core(self) -> None:
        self._enable()
        self.assertTrue(self.panel.settings["enabled"])
        self.assertFalse(self.panel.temperature_field.disabled)
        # Применено к ядру и сохранено в app_state
        self.assertTrue(
            self.agent.core.ollama_optimization["enabled"])
        import json
        raw = json.loads(self.store.saved[OPTIMIZATION_KEY])
        self.assertTrue(raw["enabled"])

    def test_toggle_off_blocks_fields(self) -> None:
        self._enable()
        self.panel.enabled_switch.value = False
        self.panel._on_toggle()
        self.assertTrue(self.panel.temperature_field.disabled)
        self.assertFalse(
            self.agent.core.ollama_optimization["enabled"])

    def test_field_changes_saved_and_applied(self) -> None:
        self._enable()
        self.panel.temperature_field.value = "0.15"
        self.panel.max_tokens_field.value = "777"
        self.panel.num_ctx_field.value = "4096"
        self.panel._on_field_blur()
        core_settings = self.agent.core.ollama_optimization
        self.assertEqual(core_settings["temperature"], 0.15)
        self.assertEqual(core_settings["max_tokens"], 777)
        self.assertEqual(core_settings["num_ctx"], 4096)
        # Провайдер Ollama (после переключения) получает значения,
        # max_tokens транслируется в num_predict при патче запроса
        self.agent.core.switch_provider("ollama_qwen")
        provider = self.agent.core.provider
        self.assertEqual(provider.optimization["max_tokens"], 777)
        patched = provider.apply_optimization({"messages": []})
        self.assertEqual(patched["extra_body"]["options"]["num_predict"],
                         777)

    def test_numeric_fields_have_no_input_filter(self) -> None:
        """Регрессия ai_1_28-fix: NumbersOnlyInputFilter блокировал
        точку и откатом отклонённых правок ломал backspace."""
        for field, _kind in self.panel._numeric_fields:
            self.assertIsNone(getattr(field, "input_filter", None))

    def test_decimal_point_and_comma_accepted(self) -> None:
        """Точка и запятая — равнозначные разделители дробной части."""
        self._enable()
        self.panel.temperature_field.value = "0.35"
        self.panel.top_p_field.value = "0,9"
        self.panel._on_field_blur()
        core_settings = self.agent.core.ollama_optimization
        self.assertEqual(core_settings["temperature"], 0.35)
        self.assertEqual(core_settings["top_p"], 0.9)
        # Поле с точкой не помечено ошибкой
        self.assertIsNone(self.panel.temperature_field.error)
        self.assertIsNone(self.panel.top_p_field.error)

    def test_trailing_point_is_valid_while_typing(self) -> None:
        """«0.» — валидное промежуточное состояние (не красная рамка)."""
        self._enable()
        self.panel.temperature_field.value = "0."
        self.panel._on_field_change()
        self.assertIsNone(self.panel.temperature_field.error)

    def test_live_validation_marks_invalid_and_clears_on_fix(self) -> None:
        self._enable()
        self.panel.temperature_field.value = "ноль-семь"
        self.panel._on_field_change()
        self.assertIn("Только число",
                      str(self.panel.temperature_field.error))
        # Исправили на валидное — ошибка ушла
        self.panel.temperature_field.value = "0.7"
        self.panel._on_field_change()
        self.assertIsNone(self.panel.temperature_field.error)

    def test_int_field_rejects_decimal_format(self) -> None:
        """Целые поля не принимают дробный формат; blur не применяет."""
        self._enable()
        good_max_tokens = self.agent.core.ollama_optimization["max_tokens"]
        self.panel.num_ctx_field.value = "8192.5"
        self.panel._on_field_change()
        self.assertIn("Только целое число",
                      str(self.panel.num_ctx_field.error))
        self.panel._on_field_blur()
        self.assertIn("не применены", self.panel.status_text.value)
        self.assertEqual(
            self.agent.core.ollama_optimization["max_tokens"],
            good_max_tokens)
        # Исправили — применилось, ошибка ушла
        self.panel.num_ctx_field.value = "4096"
        self.panel._on_field_blur()
        self.assertEqual(
            self.agent.core.ollama_optimization["num_ctx"], 4096)
        self.assertIsNone(self.panel.num_ctx_field.error)

    def test_empty_field_is_allowed(self) -> None:
        """Пустое поле не помечается ошибкой; применится минимум диапазона."""
        self._enable()
        self.panel.temperature_field.value = ""
        self.panel._on_field_change()
        self.assertIsNone(self.panel.temperature_field.error)
        self.panel._on_field_blur()
        self.assertEqual(
            self.agent.core.ollama_optimization["temperature"], 0.0)

    def test_invalid_number_keeps_previous_settings(self) -> None:
        self._enable()
        good = self.agent.core.ollama_optimization["temperature"]
        self.panel.temperature_field.value = "абракадабра"
        self.panel._on_field_blur()
        self.assertIn("Ошибка", self.panel.status_text.value)
        self.assertIn("Только число",
                      str(self.panel.temperature_field.error))
        self.assertEqual(
            self.agent.core.ollama_optimization["temperature"], good)

    def test_system_prompt_saved(self) -> None:
        self._enable()
        self.panel.system_prompt_field.value = "Пиши по-русски."
        self.panel._on_field_blur()
        self.assertEqual(
            self.agent.core.ollama_optimization["system_prompt"],
            "Пиши по-русски.")

    def test_reset_restores_defaults(self) -> None:
        self._enable()
        self.panel.temperature_field.value = "1.7"
        self.panel._on_field_blur()
        self.panel._on_reset()
        settings = self.agent.core.ollama_optimization
        self.assertEqual(settings, OptimizationSettings.DEFAULTS.copy())
        self.assertEqual(self.panel.temperature_field.value, "0.7")
        self.assertTrue(self.panel.temperature_field.disabled)
        import json
        raw = json.loads(self.store.saved[OPTIMIZATION_KEY])
        self.assertFalse(raw["enabled"])

    def test_settings_survive_panel_restart(self) -> None:
        """Сохранённые настройки восстанавливаются при перезапуске."""
        self._enable()
        self.panel.temperature_field.value = "0.2"
        self.panel._on_field_blur()
        from app.ui.optimization_view import OptimizationPanel

        reloaded = OptimizationPanel(self.app)
        self.assertTrue(reloaded.enabled_switch.value)
        self.assertEqual(reloaded.settings["temperature"], 0.2)
        self.assertFalse(reloaded.temperature_field.disabled)

    def test_quant_selection_saves_tag_and_status(self) -> None:
        self._enable()
        tags = [
            {"tag": "qwen2.5:7b", "quant": "", "local": True},
            {"tag": "qwen2.5:7b-instruct-q8_0", "quant": "Q8_0",
             "local": False},
        ]
        with mock.patch.object(OllamaTagManager, "get_available_tags",
                               return_value=tags):
            asyncio.run(self.panel._refresh_tags_async())
        keys = [option.key for option in self.panel.quant_dropdown.options]
        self.assertEqual(keys, ["qwen2.5:7b",
                                "qwen2.5:7b-instruct-q8_0"])
        # Выбор НЕ скачанного кванта: статус и кнопка загрузки
        self.panel.quant_dropdown.value = "qwen2.5:7b-instruct-q8_0"
        self.panel._on_quant_change()
        settings = self.agent.core.ollama_optimization
        self.assertEqual(settings["selected_tag"],
                         "qwen2.5:7b-instruct-q8_0")
        self.assertEqual(settings["selected_quant"], "Q8_0")
        self.assertIn("Не скачан", self.panel.quant_status_text.value)
        self.assertTrue(self.panel.download_button.visible)
        # Выбор локального тега: кнопка загрузки скрыта
        self.panel.quant_dropdown.value = "qwen2.5:7b"
        self.panel._on_quant_change()
        self.assertIn("Скачан", self.panel.quant_status_text.value)
        self.assertFalse(self.panel.download_button.visible)

    def test_refresh_tags_empty_shows_error(self) -> None:
        self._enable()
        with mock.patch.object(OllamaTagManager, "get_available_tags",
                               return_value=[]):
            asyncio.run(self.panel._refresh_tags_async())
        self.assertIn("Теги не найдены", self.panel.status_text.value)

    def test_download_pulls_selected_tag(self) -> None:
        self._enable()
        self.panel.settings["selected_tag"] = "qwen2.5:7b-instruct-q8_0"
        with mock.patch.object(OllamaTagManager, "pull_model",
                               return_value=(True, "модель загружена")) as m:
            asyncio.run(self.panel._download_async())
        m.assert_called_once_with("qwen2.5:7b-instruct-q8_0")
        self.assertIn("модель загружена", self.panel.status_text.value)
        self.assertTrue(self.panel._selected_local)
        self.assertFalse(self.panel.download_button.visible)

    def test_download_without_tag_shows_hint(self) -> None:
        self._enable()
        self.panel.settings["selected_tag"] = ""
        with mock.patch.object(OllamaTagManager, "pull_model") as m:
            self.panel._on_download()
        m.assert_not_called()
        self.assertIn("выберите", self.panel.status_text.value.lower())

    def test_tag_meta_not_leaked_into_saved_settings(self) -> None:
        """Служебные отметки (_tag_meta) не попадают в хранилище."""
        self._enable()
        self.panel.temperature_field.value = "0.3"
        self.panel._on_field_blur()
        import json
        raw = json.loads(self.store.saved[OPTIMIZATION_KEY])
        self.assertNotIn("_selected_local", raw)


class SettingsViewIntegrationTest(unittest.TestCase):
    """Панель встроена во вкладку «Настройки»."""

    def setUp(self) -> None:
        from app.ui.settings_view import SettingsView

        self.store = _StubAppState()
        self.agent = make_agent()
        self.app = make_app(self.agent)
        self.app.app_state = self.store
        self.view = SettingsView(self.app)

    def test_panel_present_in_build(self) -> None:
        column = self.view.build()
        texts = [str(control) for control in column.controls]
        self.assertTrue(any("Container" in text for text in texts))

    def test_on_show_restores_settings(self) -> None:
        store_values = dict(OptimizationSettings.DEFAULTS)
        store_values["enabled"] = True
        import json
        self.store.saved[OPTIMIZATION_KEY] = json.dumps(store_values)
        self.view.optimization_panel.enabled_switch.value = False
        self.view.optimization_panel.temperature_field.disabled = True
        self.view.on_show()
        self.assertTrue(self.view.optimization_panel.enabled_switch.value)
        # Оптимизация включена — поля активны
        self.assertFalse(
            self.view.optimization_panel.temperature_field.disabled)


if __name__ == "__main__":
    unittest.main()
