"""Проверка конфигурации и вспомогательных функций.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from src.config import get_config, load_config, validate_api_key
from src.utils import (
    AgentConfigError,
    format_tokens,
    format_time,
    validate_max_tokens,
    validate_model,
    validate_system_prompt,
    validate_temperature,
)


class ConfigTest(unittest.TestCase):
    """Тесты модуля config."""

    def test_defaults_without_env(self) -> None:
        # Убираем влияние пользовательского окружения
        for key in (
            "DEEPSEEK_API_KEY", "DEFAULT_MODEL", "DEFAULT_TEMPERATURE",
            "DEFAULT_MAX_TOKENS", "UI_TITLE", "UI_WIDTH", "UI_HEIGHT",
            "MOCK_MODE", "MAX_HISTORY_MESSAGES",
        ):
            os.environ.pop(key, None)
        config = get_config()
        self.assertEqual(config["DEFAULT_MODEL"], "deepseek-v4-flash")
        self.assertEqual(config["DEFAULT_TEMPERATURE"], 0.7)
        self.assertEqual(config["DEFAULT_MAX_TOKENS"], 1000)
        self.assertEqual(config["UI_TITLE"], "DeepSeek Chat")
        self.assertFalse(config["MOCK_MODE"])

    def test_validate_api_key_raises_when_empty(self) -> None:
        with self.assertRaises(ValueError):
            validate_api_key({"DEEPSEEK_API_KEY": "", "MOCK_MODE": False})

    def test_validate_api_key_allowed_in_mock(self) -> None:
        validate_api_key({"DEEPSEEK_API_KEY": "", "MOCK_MODE": True})

    def test_load_config_from_file(self) -> None:
        # Фикстура лежит в tests/fixtures/test.env — тест не создаёт файлы
        # на лету (в песочнице может быть запрещена запись/удаление).
        fixture = Path(__file__).resolve().parent / "fixtures" / "test.env"
        self.assertTrue(fixture.exists(), "fixture test.env не найден")

        # Очищаем ключи, которые могут остаться от предыдущих запусков
        for key in ("DEEPSEEK_API_KEY", "DEFAULT_TEMPERATURE",
                    "MOCK_MODE", "UI_WIDTH"):
            os.environ.pop(key, None)
        try:
            config = load_config(env_file=fixture)
        finally:
            for key in ("DEEPSEEK_API_KEY", "DEFAULT_TEMPERATURE",
                        "MOCK_MODE", "UI_WIDTH"):
                os.environ.pop(key, None)

        self.assertEqual(config["DEEPSEEK_API_KEY"], "sk-test123")
        self.assertEqual(config["DEFAULT_TEMPERATURE"], 1.3)
        self.assertTrue(config["MOCK_MODE"])
        self.assertEqual(config["UI_WIDTH"], 800)


class UtilsValidationTest(unittest.TestCase):
    """Тесты валидаторов utils."""

    def test_validate_model(self) -> None:
        validate_model("deepseek-v4-flash")
        validate_model("deepseek-v4-pro")
        with self.assertRaises(AgentConfigError):
            validate_model("unknown-model")

    def test_validate_temperature_bounds(self) -> None:
        self.assertEqual(validate_temperature(0.0), 0.0)
        self.assertEqual(validate_temperature(1.24), 1.2)
        with self.assertRaises(AgentConfigError):
            validate_temperature(2.5)

    def test_validate_max_tokens(self) -> None:
        self.assertEqual(validate_max_tokens(100), 100)
        with self.assertRaises(AgentConfigError):
            validate_max_tokens(0)
        with self.assertRaises(AgentConfigError):
            validate_max_tokens("abc")

    def test_validate_system_prompt(self) -> None:
        self.assertEqual(validate_system_prompt("  Привет  "), "Привет")
        with self.assertRaises(AgentConfigError):
            validate_system_prompt("   ")

    def test_format_time(self) -> None:
        value = format_time()
        self.assertRegex(value, r"^\d{2}:\d{2}:\d{2}$")

    def test_format_tokens(self) -> None:
        self.assertEqual(format_tokens(None), "—")
        self.assertEqual(
            format_tokens({"prompt_tokens": 10, "completion_tokens": 5,
                           "total_tokens": 15}),
            "in: 10 · out: 5 · total: 15",
        )


if __name__ == "__main__":
    unittest.main()
