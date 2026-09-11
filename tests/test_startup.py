"""Тесты стартовой конфигурации приложения (main.create_agent).

Именно этот путь ломался при запуске `python main.py` (неимпортированный
модуль в config.py), поэтому здесь проверяется вся сборка зависимостей
без запуска GUI и без сети.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import main as app_main
from src.config import get_config, get_summary_settings
from src.utils import AgentConfigError

_TESTS_TMP = Path(__file__).resolve().parent / "tmp_startup"


def _config(**overrides) -> dict:
    """Конфигурация для тестов: без хранилища и с фиктивным ключом."""
    config = get_config()
    config.update({
        "DEEPSEEK_API_KEY": "test-key",
        "MOCK_MODE": False,
        "STORAGE_ENABLED": False,
        "DEFAULT_MODEL": "deepseek-v4-flash",
    })
    config.update(overrides)
    return config


class CreateAgentTest(unittest.TestCase):
    """Сборка агента из конфигурации (как в main.py)."""

    def tearDown(self) -> None:
        shutil.rmtree(_TESTS_TMP, ignore_errors=True)

    def test_create_agent_with_defaults(self) -> None:
        """Стартовая сборка проходит и переносит настройки в агента."""
        agent = app_main.create_agent(_config())
        try:
            self.assertEqual(agent.model, "deepseek-v4-flash")
            self.assertFalse(agent.is_mock)
            # Настройки сжатия контекста применены из конфигурации
            settings = agent.summary_settings
            self.assertTrue(settings["enabled"])
            self.assertEqual(settings["keep_last"], 10)
            self.assertEqual(settings["every_n"], 10)
        finally:
            agent.close()

    def test_create_agent_mock_mode(self) -> None:
        agent = app_main.create_agent(_config(MOCK_MODE=True))
        try:
            self.assertTrue(agent.is_mock)
        finally:
            agent.close()

    def test_create_agent_applies_custom_summary_settings(self) -> None:
        agent = app_main.create_agent(_config(
            SUMMARY_KEEP_LAST_N=4, SUMMARY_EVERY_N=3, SUMMARY_MAX_TOKENS=120,
            SUMMARY_ENABLED=False,
        ))
        try:
            settings = agent.summary_settings
            self.assertFalse(settings["enabled"])
            self.assertEqual(settings["keep_last"], 4)
            self.assertEqual(settings["every_n"], 3)
            self.assertEqual(settings["max_tokens"], 120)
        finally:
            agent.close()

    def test_create_agent_with_storage(self) -> None:
        """Хранилище подключается и указывает на файл из конфигурации."""
        _TESTS_TMP.mkdir(parents=True, exist_ok=True)
        db_path = _TESTS_TMP / f"start_{uuid.uuid4().hex}.db"
        agent = app_main.create_agent(_config(
            STORAGE_ENABLED=True, STORAGE_PATH=str(db_path),
        ))
        try:
            # Сохранение создаёт файл БД (ленивое создание при первой записи)
            agent._persist()
            self.assertTrue(db_path.exists())
        finally:
            agent.close()

    def test_invalid_model_raises_config_error(self) -> None:
        with self.assertRaises(AgentConfigError):
            app_main.create_agent(_config(DEFAULT_MODEL="no-such-model"))

    def test_summary_settings_validation_clamps(self) -> None:
        """Некорректные значения заменяются значениями по умолчанию."""
        settings = get_summary_settings({
            "SUMMARY_ENABLED": True,
            "SUMMARY_KEEP_LAST_N": "abc",
            "SUMMARY_EVERY_N": 9999,
            "SUMMARY_MAX_TOKENS": 10,
            "SUMMARY_TEMPERATURE": 5.0,
        })
        self.assertEqual(settings["keep_last"], 10)
        self.assertEqual(settings["every_n"], 10)
        self.assertEqual(settings["max_tokens"], 400)
        self.assertEqual(settings["temperature"], 0.2)


if __name__ == "__main__":
    unittest.main()
