"""Тесты переключения LLM-провайдеров DeepSeek ↔ Ollama (ТЗ ai_1_25).

Проверяются офлайн (без сети и запущенного Ollama): фабрика провайдеров,
доступность и capability Ollama (httpx замокан), chat_completion c
очисткой reasoning_effort/thinking, switch_provider ядра (успех, ошибка
доступности с сохранением состояния, отключение MCP без tool calling),
guard профиля/настроек от подмены модели чужого провайдера, лаунчер
Ollama (launch/ensure_model), сохранение выбора в app_state и UI
(выпадающий список, модалки, откат выбора).

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import llm_providers  # noqa: E402
from src.agent import DeepSeekAgent  # noqa: E402
from src.llm_providers import (  # noqa: E402
    DEEPSEEK_MODELS,
    DeepSeekProvider,
    LLMProviderFactory,
    OllamaProvider,
    ProviderUnavailableError,
)
from src.ollama_launcher import OllamaLauncher  # noqa: E402
from src.utils import AgentConfigError  # noqa: E402

from tests.test_app_agent import make_agent  # noqa: E402
from tests.test_flet_views import make_app  # noqa: E402


class _FakeHTTPResponse:
    """Ответ httpx-заглушка (status_code + json())."""

    def __init__(self, status_code: int = 200, payload: dict = None) -> None:
        self.status_code = int(status_code)
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


# ----------------------------------------------------------------------
# Фабрика провайдеров
# ----------------------------------------------------------------------
class FactoryTest(unittest.TestCase):
    """LLMProviderFactory: реестр, создание, доступность."""

    def test_registry_specs(self) -> None:
        self.assertEqual(
            LLMProviderFactory.PROVIDERS["deepseek"]["model"],
            "deepseek-v4-flash")
        self.assertFalse(
            LLMProviderFactory.PROVIDERS["deepseek"]["requires_ollama"])
        self.assertEqual(
            LLMProviderFactory.PROVIDERS["ollama_qwen"]["model"],
            "qwen2.5:7b")
        self.assertEqual(
            LLMProviderFactory.PROVIDERS["ollama_llama"]["model"],
            "llama3.1:8b")
        self.assertTrue(
            LLMProviderFactory.PROVIDERS["ollama_qwen"]["requires_ollama"])

    def test_create_deepseek(self) -> None:
        provider = LLMProviderFactory.create("deepseek", api_key="k")
        self.assertIsInstance(provider, DeepSeekProvider)
        self.assertEqual(provider.id, "deepseek")
        self.assertEqual(provider.model, "deepseek-v4-flash")
        self.assertTrue(provider.is_available())
        self.assertTrue(provider.supports_tool_calling())
        self.assertTrue(provider.owns_model("deepseek-v4-pro"))
        self.assertFalse(provider.owns_model("qwen2.5:7b"))
        self.assertEqual(provider.model_choices(), DEEPSEEK_MODELS)

    def test_create_deepseek_key_from_env_and_explicit(self) -> None:
        with mock.patch.dict(os.environ,
                             {"DEEPSEEK_API_KEY": "env-key"}):
            provider = LLMProviderFactory.create("deepseek")
            self.assertEqual(provider.client.api_key, "env-key")
        provider = LLMProviderFactory.create("deepseek", api_key="explicit")
        self.assertEqual(provider.client.api_key, "explicit")

    def test_create_ollama(self) -> None:
        for provider_name, model in (("ollama_qwen", "qwen2.5:7b"),
                                     ("ollama_llama", "llama3.1:8b")):
            provider = LLMProviderFactory.create(provider_name)
            self.assertIsInstance(provider, OllamaProvider)
            self.assertEqual(provider.id, provider_name)
            self.assertEqual(provider.model, model)
            # OpenAI-совместимый эндпоинт и игнорируемый ключ SDK
            self.assertTrue(provider.base_url.endswith("/v1"))
            self.assertEqual(provider.client.api_key, "ollama")
            self.assertEqual(provider.model_choices(), (model,))

    def test_create_unknown_raises(self) -> None:
        with self.assertRaises(ValueError):
            LLMProviderFactory.create("gpt5")
        with self.assertRaises(ValueError):
            LLMProviderFactory.spec("")

    def test_deepseek_provider_without_key_falls_back(self) -> None:
        """Без ключа SDK не падает: заглушка «missing-api-key»."""
        with mock.patch.dict(os.environ,
                             {"DEEPSEEK_API_KEY": ""}, clear=False):
            provider = DeepSeekProvider()
        self.assertEqual(provider.client.api_key, "missing-api-key")

    def test_ollama_base_url_from_env(self) -> None:
        with mock.patch.dict(os.environ,
                             {"OLLAMA_BASE_URL": "http://127.0.0.1:11500"}):
            provider = LLMProviderFactory.create("ollama_qwen")
            self.assertEqual(provider.base_url,
                             "http://127.0.0.1:11500/v1")
            self.assertEqual(provider.api_base,
                             "http://127.0.0.1:11500")

    def test_get_available_providers(self) -> None:
        with mock.patch.object(OllamaProvider, "is_available",
                               return_value=False):
            items = LLMProviderFactory.get_available_providers()
        by_id = {item["id"]: item for item in items}
        self.assertTrue(by_id["deepseek"]["available"])
        self.assertFalse(by_id["ollama_qwen"]["available"])
        self.assertFalse(by_id["ollama_llama"]["available"])


# ----------------------------------------------------------------------
# OllamaProvider: доступность и tool calling (httpx замокан)
# ----------------------------------------------------------------------
class OllamaProviderTest(unittest.TestCase):
    """is_available / supports_tool_calling / chat_completion."""

    def setUp(self) -> None:
        self.provider = OllamaProvider(model="qwen2.5:7b")

    def test_is_available_true_when_model_listed(self) -> None:
        response = _FakeHTTPResponse(200, {"models": [
            {"name": "qwen2.5:7b"}, {"name": "nomic-embed-text"}]})
        with mock.patch.object(llm_providers.httpx, "get",
                               return_value=response) as mocked:
            self.assertTrue(self.provider.is_available())
        self.assertIn("/api/tags", mocked.call_args.args[0])

    def test_is_available_false_cases(self) -> None:
        empty = _FakeHTTPResponse(200, {"models": []})
        with mock.patch.object(llm_providers.httpx, "get",
                               return_value=empty):
            self.assertFalse(self.provider.is_available())
        not_found = _FakeHTTPResponse(404, {})
        with mock.patch.object(llm_providers.httpx, "get",
                               return_value=not_found):
            self.assertFalse(self.provider.is_available())
        with mock.patch.object(llm_providers.httpx, "get",
                               side_effect=OSError("connection refused")):
            self.assertFalse(self.provider.is_available())

    def test_supports_tool_calling_capability(self) -> None:
        with_tools = _FakeHTTPResponse(
            200, {"capabilities": ["completion", "tools"]})
        with mock.patch.object(llm_providers.httpx, "post",
                               return_value=with_tools) as mocked:
            self.assertTrue(self.provider.supports_tool_calling())
        self.assertIn("/api/show", mocked.call_args.args[0])
        self.assertEqual(mocked.call_args.kwargs["json"],
                         {"model": "qwen2.5:7b"})

    def test_supports_tool_calling_without_capability(self) -> None:
        no_tools = _FakeHTTPResponse(200, {"capabilities": ["completion"]})
        with mock.patch.object(llm_providers.httpx, "post",
                               return_value=no_tools):
            self.assertFalse(self.provider.supports_tool_calling())
        with mock.patch.object(llm_providers.httpx, "post",
                               return_value=_FakeHTTPResponse(500, {})):
            self.assertFalse(self.provider.supports_tool_calling())
        with mock.patch.object(llm_providers.httpx, "post",
                               side_effect=OSError("down")):
            self.assertFalse(self.provider.supports_tool_calling())

    def test_chat_completion_strips_cloud_only_kwargs(self) -> None:
        captured = {}

        class _Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                message = SimpleNamespace(content="привет", tool_calls=None)
                choice = SimpleNamespace(message=message)
                return SimpleNamespace(choices=[choice], usage=None)

        self.provider.client = SimpleNamespace(
            chat=SimpleNamespace(completions=_Completions()))
        result = self.provider.chat_completion(
            [{"role": "user", "content": "тест"}],
            temperature=0.3,
            reasoning_effort="high",
            thinking={"type": "enabled"},
        )
        self.assertEqual(result["content"], "привет")
        self.assertIsNone(result["tool_calls"])
        self.assertNotIn("reasoning_effort", captured)
        self.assertNotIn("thinking", captured)
        self.assertEqual(captured["model"], "qwen2.5:7b")
        self.assertEqual(captured["temperature"], 0.3)

    def test_owns_model_by_base_name(self) -> None:
        self.assertTrue(self.provider.owns_model("qwen2.5:7b"))
        self.assertTrue(self.provider.owns_model("Qwen2.5:14b"))
        self.assertFalse(self.provider.owns_model("llama3.1:8b"))


# ----------------------------------------------------------------------
# switch_provider ядра (mock-режим: без сети)
# ----------------------------------------------------------------------
class SwitchProviderMockTest(unittest.TestCase):
    """switch_provider в mock-режиме: переключение без обращений к сети."""

    def setUp(self) -> None:
        self.core = DeepSeekAgent(api_key="test", model="deepseek-v4-flash",
                                  mock_mode=True, storage=None)

    def test_default_provider_is_deepseek(self) -> None:
        self.assertIsNone(self.core.provider)
        self.assertEqual(self.core.provider_id, "deepseek")

    def test_switch_to_ollama_and_back(self) -> None:
        result = self.core.switch_provider("ollama_qwen")
        # Локальная модель: MCP отключается политикой (не capability)
        self.assertIn("warning", result)
        self.assertIn("только к ней", result["warning"])
        self.assertEqual(self.core.provider_id, "ollama_qwen")
        self.assertEqual(self.core.model, "qwen2.5:7b")
        self.assertEqual(self.core.provider.base_url,
                         "http://localhost:11434/v1")
        self.assertFalse(self.core._provider_tools_enabled)
        # Обратно на DeepSeek: инструменты включаются снова
        result = self.core.switch_provider("deepseek", api_key="k")
        self.assertTrue(result["success"])
        self.assertEqual(self.core.provider_id, "deepseek")
        self.assertEqual(self.core.model, "deepseek-v4-flash")
        self.assertTrue(self.core._provider_tools_enabled)

    def test_switch_preserves_history(self) -> None:
        self.core._append_message(
            {"role": "user", "content": "запомни меня"})
        before = len(self.core._message_history)
        self.core.switch_provider("ollama_qwen")
        self.assertEqual(len(self.core._message_history), before)
        self.assertEqual(self.core._message_history[-1]["content"],
                         "запомни меня")

    def test_switch_unknown_provider(self) -> None:
        with self.assertRaises(ValueError):
            self.core.switch_provider("gpt5")

    def test_switch_back_uses_agent_api_key(self) -> None:
        """UI вызывает switch_provider без api_key — ключ берётся у агента."""
        self.core.switch_provider("ollama_qwen")
        result = self.core.switch_provider("deepseek")
        self.assertTrue(result["success"])
        self.assertEqual(self.core.provider.client.api_key, "test")

    def test_mock_switch_to_local_disables_tools(self) -> None:
        """Политика «локальная модель — без MCP» работает и в mock-режиме
        (local-флаг статичен, HTTP не нужен)."""
        result = self.core.switch_provider("ollama_qwen")
        self.assertIn("warning", result)
        self.assertFalse(self.core._provider_tools_enabled)
        result = self.core.switch_provider("deepseek")
        self.assertNotIn("warning", result)
        self.assertTrue(self.core._provider_tools_enabled)


class SwitchProviderAvailabilityTest(unittest.TestCase):
    """Доступность и capability в боёвом режиме (httpx замокан)."""

    def setUp(self) -> None:
        self.core = DeepSeekAgent(api_key="test", model="deepseek-v4-flash",
                                  mock_mode=False, storage=None)

    def _patch(self, get=None, post=None):
        patchers = []
        if get is not None:
            patchers.append(mock.patch.object(llm_providers.httpx, "get",
                                              get))
        if post is not None:
            patchers.append(mock.patch.object(llm_providers.httpx, "post",
                                              post))
        for patcher in patchers:
            patcher.start()
        return patchers

    def tearDown(self) -> None:
        # mock.patch.object(started=True) снимается глобально
        mock.patch.stopall()

    def test_unavailable_provider_raises_and_keeps_state(self) -> None:
        self._patch(get=mock.Mock(side_effect=OSError("refused")))
        with self.assertRaises(ProviderUnavailableError):
            self.core.switch_provider("ollama_qwen")
        # Состояние не изменилось
        self.assertIsNone(self.core.provider)
        self.assertEqual(self.core.provider_id, "deepseek")
        self.assertEqual(self.core.model, "deepseek-v4-flash")

    def test_switch_to_local_never_enables_mcp(self) -> None:
        """Даже при capability tools локальный провайдер работает без MCP."""
        post = mock.Mock(return_value=_FakeHTTPResponse(
            200, {"capabilities": ["tools"]}))
        self._patch(
            get=mock.Mock(return_value=_FakeHTTPResponse(
                200, {"models": [{"name": "qwen2.5:7b"}]})),
            post=post,
        )
        result = self.core.switch_provider("ollama_qwen")
        self.assertIn("warning", result)
        self.assertEqual(self.core.provider_id, "ollama_qwen")
        self.assertEqual(self.core.model, "qwen2.5:7b")
        self.assertFalse(self.core._provider_tools_enabled)
        # Для политики capability /api/show не запрашивается
        post.assert_not_called()

    def test_switch_without_tool_calling_disables_mcp(self) -> None:
        stub_client = SimpleNamespace(
            list_tools=mock.AsyncMock(return_value=[
                SimpleNamespace(name="add_reminder", description="d",
                                input_schema={"type": "object",
                                              "properties": {}}),
            ]))
        core = DeepSeekAgent(api_key="test", mock_mode=False, storage=None,
                             reminder_client=stub_client)
        self._patch(
            get=mock.Mock(return_value=_FakeHTTPResponse(
                200, {"models": [{"name": "qwen2.5:7b"}]})),
            post=mock.Mock(return_value=_FakeHTTPResponse(
                200, {"capabilities": ["completion"]})),
        )
        result = core.switch_provider("ollama_qwen")
        self.assertIn("warning", result)
        self.assertIn("Локальная модель", result["warning"])
        self.assertFalse(core._provider_tools_enabled)
        # Без tool calling tools не попадают в запрос
        self.assertEqual(core._get_tools_for_llm(), [])
        # Обратно на DeepSeek: инструменты включаются снова
        result = core.switch_provider("deepseek", api_key="k")
        self.assertNotIn("warning", result)
        self.assertTrue(core._provider_tools_enabled)
        tools = core._get_tools_for_llm()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["function"]["name"], "add_reminder")


# ----------------------------------------------------------------------
# Guard профиля и настроек: модель чужого провайдера не применяется
# ----------------------------------------------------------------------
class ProviderModelGuardTest(unittest.TestCase):
    """Профиль/update_config не подменяют модель активного провайдера."""

    def setUp(self) -> None:
        self.core = DeepSeekAgent(api_key="test", mock_mode=True,
                                  storage=None,
                                  provider=OllamaProvider(
                                      model="qwen2.5:7b"))
        self.assertEqual(self.core.model, "qwen2.5:7b")

    def test_profile_model_of_other_provider_ignored(self) -> None:
        self.core._profile_settings = SimpleNamespace(
            model="deepseek-v4-flash", temperature=0.1, max_tokens=123)
        self.core._apply_profile_settings()
        self.assertEqual(self.core.model, "qwen2.5:7b")
        self.assertEqual(self.core.temperature, 0.1)
        self.assertEqual(self.core.max_tokens, 123)

    def test_profile_model_of_same_provider_applied(self) -> None:
        self.core._profile_settings = SimpleNamespace(
            model="qwen2.5:7b", temperature=0.9, max_tokens=50)
        self.core._apply_profile_settings()
        self.assertEqual(self.core.model, "qwen2.5:7b")
        self.assertEqual(self.core.temperature, 0.9)

    def test_update_config_rejects_foreign_model(self) -> None:
        with self.assertRaises(AgentConfigError):
            self.core.update_config(model="deepseek-v4-flash")
        self.assertEqual(self.core.model, "qwen2.5:7b")
        # Остальные параметры обновляются без проблем
        self.core.update_config(temperature=0.2)
        self.assertEqual(self.core.temperature, 0.2)

    def test_deepseek_agent_without_provider_keeps_old_behavior(self) -> None:
        core = DeepSeekAgent(api_key="test", mock_mode=True, storage=None)
        core.update_config(model="deepseek-v4-pro")
        self.assertEqual(core.model, "deepseek-v4-pro")


# ----------------------------------------------------------------------
# OllamaLauncher
# ----------------------------------------------------------------------
class _FreePort:
    """Контекст: свободный TCP-порт на 127.0.0.1."""

    def __enter__(self) -> int:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    def __exit__(self, *args) -> None:
        return None


class OllamaLauncherTest(unittest.TestCase):
    """launch/ensure_running/ensure_model (процессы замоканы)."""

    def setUp(self) -> None:
        with _FreePort() as port:
            self.launcher = OllamaLauncher(port=port)

    def test_is_running_false_on_closed_port(self) -> None:
        self.assertFalse(self.launcher.is_running())

    def test_ensure_running_true_when_already_running(self) -> None:
        with mock.patch.object(OllamaLauncher, "is_running",
                               return_value=True):
            self.assertTrue(self.launcher.ensure_running(timeout=0.1))

    def test_ensure_running_false_without_ollama_in_path(self) -> None:
        with mock.patch.object(OllamaLauncher, "is_running",
                               return_value=False), \
                mock.patch.object(OllamaLauncher, "ollama_exe",
                                  return_value=None):
            self.assertFalse(self.launcher.ensure_running(timeout=0.1))

    def test_launch_spawns_serve_in_new_console(self) -> None:
        popen = mock.Mock()
        with mock.patch.object(OllamaLauncher, "is_running",
                               return_value=False), \
                mock.patch.object(OllamaLauncher, "ollama_exe",
                                  return_value="C:/fake/ollama.exe"), \
                mock.patch.object(llm_providers, "httpx"), \
                mock.patch("src.ollama_launcher.subprocess.Popen", popen):
            started, message = OllamaLauncher().launch()
        self.assertTrue(started)
        command = popen.call_args.args[0]
        self.assertEqual(command, ["C:/fake/ollama.exe", "serve"])

    def test_launch_skips_when_already_running(self) -> None:
        with mock.patch.object(OllamaLauncher, "is_running",
                               return_value=True):
            started, message = self.launcher.launch()
        self.assertFalse(started)
        self.assertIn("не перезапускаем", message)

    def test_launch_without_ollama_reports_path(self) -> None:
        with mock.patch.object(OllamaLauncher, "is_running",
                               return_value=False), \
                mock.patch.object(OllamaLauncher, "ollama_exe",
                                  return_value=None):
            started, message = self.launcher.launch()
        self.assertFalse(started)
        self.assertIn("PATH", message)

    def test_ensure_model_already_loaded(self) -> None:
        response = _FakeHTTPResponse(200, {"models":
                                           [{"name": "qwen2.5:7b"}]})
        run = mock.Mock()
        with mock.patch("src.ollama_launcher.httpx.get",
                        return_value=response), \
                mock.patch("src.ollama_launcher.subprocess.run", run):
            ok, message = self.launcher.ensure_model("qwen2.5:7b")
        self.assertTrue(ok)
        self.assertIn("уже загружена", message)
        run.assert_not_called()

    def test_ensure_model_pulls_when_missing(self) -> None:
        response = _FakeHTTPResponse(200, {"models": []})
        run = mock.Mock()
        with mock.patch("src.ollama_launcher.httpx.get",
                        return_value=response), \
                mock.patch("src.ollama_launcher.subprocess.run", run), \
                mock.patch.object(OllamaLauncher, "ollama_exe",
                                  return_value="C:/fake/ollama.exe"):
            ok, message = self.launcher.ensure_model("qwen2.5:7b")
        self.assertTrue(ok)
        self.assertIn("загружена", message)
        command = run.call_args.args[0]
        self.assertEqual(command,
                         ["C:/fake/ollama.exe", "pull", "qwen2.5:7b"])
        self.assertTrue(run.call_args.kwargs.get("check"))

    def test_ensure_model_pull_failure(self) -> None:
        response = _FakeHTTPResponse(200, {"models": []})
        run = mock.Mock(
            side_effect=subprocess.CalledProcessError(1, "pull"))
        with mock.patch("src.ollama_launcher.httpx.get",
                        return_value=response), \
                mock.patch("src.ollama_launcher.subprocess.run", run), \
                mock.patch.object(OllamaLauncher, "ollama_exe",
                                  return_value="C:/fake/ollama.exe"):
            ok, message = self.launcher.ensure_model("llama3.1:8b")
        self.assertFalse(ok)
        self.assertIn("llama3.1:8b", message)

    def test_has_model_false_when_server_down(self) -> None:
        with mock.patch("src.ollama_launcher.httpx.get",
                        side_effect=OSError("down")):
            self.assertFalse(self.launcher.has_model("qwen2.5:7b"))


# ----------------------------------------------------------------------
# app_state: выбор провайдера переживает перезапуск
# ----------------------------------------------------------------------
class ProviderPersistenceTest(unittest.TestCase):
    """AppStateStore: ключ active_provider."""

    def test_provider_roundtrip(self) -> None:
        from src.app_state import AppStateStore

        with tempfile.TemporaryDirectory() as tmp:
            store = AppStateStore(Path(tmp) / "app_state.db")
            self.assertIsNone(store.get("active_provider"))
            store.set("active_provider", "ollama_qwen")
            self.assertEqual(store.get("active_provider"), "ollama_qwen")
            store.set("active_provider", "deepseek")
            self.assertEqual(store.get("active_provider"), "deepseek")


# ----------------------------------------------------------------------
# UI вкладки «Чат»: выпадающий список и обработка переключения
# ----------------------------------------------------------------------
class _StubAppState:
    """AppStateStore-заглушка (запоминает set)."""

    def __init__(self) -> None:
        self.saved: dict = {}

    def get(self, key: str):
        return self.saved.get(key)

    def set(self, key: str, value: str) -> bool:
        self.saved[key] = value
        return True


class ChatViewProviderTest(unittest.TestCase):
    """Переключение провайдера из UI (офлайн, page=None)."""

    def setUp(self) -> None:
        self.app_state = _StubAppState()
        self.agent = make_agent()
        self.app = make_app(self.agent)
        self.app.app_state = self.app_state
        self.app.session_id = "session-1"
        from app.ui.chat_view import ChatView

        self.view = ChatView(self.app)

    def test_dropdown_options_and_default(self) -> None:
        keys = [option.key
                for option in self.view.provider_dropdown.options]
        self.assertEqual(keys, ["deepseek", "ollama_qwen", "ollama_llama"])
        self.assertEqual(self.view.provider_dropdown.value, "deepseek")

    def test_status_bar_shows_provider(self) -> None:
        self.view.refresh_rag_status()
        self.assertIn("DeepSeek", self.view.rag_status_text.value)
        self.assertIn("deepseek-v4-flash",
                      self.view.rag_status_text.value)

    def test_switch_provider_from_ui(self) -> None:
        asyncio.run(self.view._switch_provider_async("ollama_qwen"))
        core = self.agent.core
        self.assertEqual(core.provider_id, "ollama_qwen")
        self.assertEqual(core.model, "qwen2.5:7b")
        self.assertEqual(self.view.provider_dropdown.value, "ollama_qwen")
        self.assertEqual(self.app_state.saved.get("active_provider"),
                         "ollama_qwen")
        self.assertIn("Ollama", self.view.status_text.value)
        self.assertIn("Ollama", self.view.rag_status_text.value)
        # Локальная модель → предупреждение «MCP отключены» (модалка)
        self.assertIsNotNone(self.view.last_dialog)
        self.assertIn("Внимание", str(self.view.last_dialog.title.value))
        self.assertFalse(core._provider_tools_enabled)

    def test_unavailable_provider_shows_error_and_reverts(self) -> None:
        def _raise(provider_id: str):
            raise ProviderUnavailableError("Ollama не запущен.")

        self.view._prepare_and_switch = _raise
        asyncio.run(self.view._switch_provider_async("ollama_llama"))
        # Модалка ошибки, выбор в списке и ядре не изменился
        self.assertIsNotNone(self.view.last_dialog)
        self.assertIn("недоступен", str(self.view.last_dialog.title.value))
        self.assertEqual(self.view.provider_dropdown.value, "deepseek")
        self.assertEqual(self.agent.core.provider_id, "deepseek")
        self.assertNotIn("active_provider", self.app_state.saved)

    def test_tool_calling_warning_dialog(self) -> None:
        def _warning(provider_id: str):
            return {"warning": "Модель не поддерживает tool calling. "
                               "MCP-инструменты отключены.",
                    "provider": provider_id,
                    "name": "Ollama — Qwen 2.5 7B",
                    "model": "qwen2.5:7b"}

        self.view._prepare_and_switch = _warning
        asyncio.run(self.view._switch_provider_async("ollama_qwen"))
        self.assertIsNotNone(self.view.last_dialog)
        self.assertIn("Внимание", str(self.view.last_dialog.title.value))
        # Выбор при предупреждении сохраняется (переключение состоялось)
        self.assertEqual(self.app_state.saved.get("active_provider"),
                         "ollama_qwen")

    def test_schedule_provider_switch_sets_dropdown(self) -> None:
        page = SimpleNamespace(run_task=mock.Mock(), update=mock.Mock(),
                               show_dialog=mock.Mock())
        self.app.page = page
        self.view.schedule_provider_switch("ollama_qwen")
        self.assertEqual(self.view.provider_dropdown.value, "ollama_qwen")
        page.run_task.assert_called_once()
        # run_task получает КОРУТИННУЮ ФУНКЦИЮ и аргумент
        handler, argument = page.run_task.call_args.args
        self.assertTrue(asyncio.iscoroutinefunction(handler))
        self.assertEqual(argument, "ollama_qwen")


if __name__ == "__main__":
    unittest.main()
