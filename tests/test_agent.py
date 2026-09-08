"""Тесты агента DeepSeekAgent без обращения к реальному API.

Запуск из корня проекта:
    python -m unittest discover -s tests -v

Mock-режим агента имитирует ответ, поэтому сеть не требуется.
"""

from __future__ import annotations

import unittest

from src.agent import DeepSeekAgent
from src.utils import AgentConfigError, humanize_api_error


class DeepSeekAgentMockTest(unittest.TestCase):
    """Проверка поведения агента в mock-режиме."""

    def setUp(self) -> None:
        self.agent = DeepSeekAgent(
            api_key="",
            model="deepseek-v4-flash",
            system_prompt="Тестовая система.",
            temperature=0.5,
            max_tokens=500,
            mock_mode=True,
        )

    def test_initial_history_has_system_prompt(self) -> None:
        history = self.agent.history
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["role"], "system")
        self.assertEqual(history[0]["content"], "Тестовая система.")

    def test_send_message_success_shape(self) -> None:
        result = self.agent.send_message("Привет!")
        self.assertIsNone(result["error"])
        self.assertTrue(result["text"])
        self.assertEqual(result["model"], "deepseek-v4-flash")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertIsInstance(result["response_time"], float)
        self.assertIsNotNone(result["tokens"])

    def test_history_preserved_between_calls(self) -> None:
        self.agent.send_message("Первый вопрос")
        self.agent.send_message("Второй вопрос")
        roles = [m["role"] for m in self.agent.history]
        # system + user + assistant + user + assistant
        self.assertEqual(
            roles, ["system", "user", "assistant", "user", "assistant"]
        )

    def test_clear_history_keeps_only_system(self) -> None:
        self.agent.send_message("Вопрос")
        self.agent.clear_history()
        history = self.agent.history
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["role"], "system")

    def test_empty_input_returns_error(self) -> None:
        before = len(self.agent.history)
        result = self.agent.send_message("   ")
        self.assertIsNotNone(result["error"])
        self.assertIsNone(result["text"])
        self.assertEqual(len(self.agent.history), before)  # ничего не добавлено

    def test_update_config_temperature(self) -> None:
        self.agent.update_config(temperature=1.2)
        self.assertEqual(self.agent.temperature, 1.2)

    def test_update_config_rejects_invalid_temperature(self) -> None:
        with self.assertRaises(AgentConfigError):
            self.agent.update_config(temperature=5.0)

    def test_update_config_rejects_unknown_model(self) -> None:
        with self.assertRaises(AgentConfigError):
            self.agent.update_config(model="gpt-999")

    def test_update_config_unknown_key(self) -> None:
        with self.assertRaises(AgentConfigError):
            self.agent.update_config(unknown_param=1)

    def test_change_system_prompt_resets_history(self) -> None:
        self.agent.send_message("Вопрос")
        self.agent.update_config(system_prompt="Новый промпт.")
        history = self.agent.history
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["content"], "Новый промпт.")

    def test_history_limited(self) -> None:
        self.agent.update_config(max_history_messages=4)
        for i in range(10):
            self.agent.send_message(f"Вопрос {i}")
        # system + не более 4 сообщений
        self.assertLessEqual(len(self.agent.history), 5)
        self.assertEqual(self.agent.history[0]["role"], "system")


class ValidationTest(unittest.TestCase):
    """Проверка валидации параметров конструктора."""

    def test_invalid_model_raises(self) -> None:
        with self.assertRaises(AgentConfigError):
            DeepSeekAgent(api_key="k", model="bad-model", mock_mode=True)

    def test_invalid_temperature_raises(self) -> None:
        with self.assertRaises(AgentConfigError):
            DeepSeekAgent(api_key="k", temperature=99, mock_mode=True)

    def test_invalid_max_tokens_raises(self) -> None:
        with self.assertRaises(AgentConfigError):
            DeepSeekAgent(api_key="k", max_tokens=0, mock_mode=True)

    def test_default_parameters(self) -> None:
        agent = DeepSeekAgent(api_key="k", mock_mode=True)
        self.assertEqual(agent.model, "deepseek-v4-flash")
        self.assertEqual(agent.temperature, 0.7)
        self.assertEqual(agent.max_tokens, 1000)


class HumanizeErrorTest(unittest.TestCase):
    """Проверка перевода ошибок API в понятный текст."""

    def test_auth_error_does_not_leak_key(self) -> None:
        text = humanize_api_error(
            Exception("Incorrect API key provided: sk-1234567890. 401")
        )
        self.assertIn("аутентификации", text)
        self.assertNotIn("sk-1234567890", text)  # ключ не раскрывается

    def test_network_error(self) -> None:
        text = humanize_api_error(Exception("Connection error: timed out"))
        self.assertIn("Сетевая ошибка", text)


if __name__ == "__main__":
    unittest.main()
