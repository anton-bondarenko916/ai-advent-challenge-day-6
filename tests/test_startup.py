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
from src.config import get_config, get_strategy_settings
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
            # Стратегия управления контекстом применена из конфигурации
            strategy = agent.get_strategy()
            self.assertEqual(strategy["name"], "sliding_window")
            self.assertEqual(strategy["window"], 20)
            self.assertIn("branching", strategy["available"])
            # Лимиты фактов из конфигурации
            settings = agent.strategy_settings
            self.assertTrue(settings["facts_enabled"])
            self.assertEqual(settings["facts_max_items"], 50)
            self.assertEqual(settings["facts_max_tokens"], 300)
            self.assertEqual(settings["branch_max"], 20)
        finally:
            agent.close()

    def test_create_agent_mock_mode(self) -> None:
        agent = app_main.create_agent(_config(MOCK_MODE=True))
        try:
            self.assertTrue(agent.is_mock)
        finally:
            agent.close()

    def test_create_agent_applies_custom_strategy_settings(self) -> None:
        agent = app_main.create_agent(_config(
            STRATEGY="sticky_facts", WINDOW_KEEP_LAST_N=4,
            FACTS_ENABLED=False, FACTS_MAX_ITEMS=7, FACTS_MAX_TOKENS=120,
            BRANCH_MAX=3,
        ))
        try:
            strategy = agent.get_strategy()
            self.assertEqual(strategy["name"], "sticky_facts")
            self.assertEqual(strategy["window"], 4)
            settings = agent.strategy_settings
            self.assertFalse(settings["facts_enabled"])
            self.assertEqual(settings["facts_max_items"], 7)
            self.assertEqual(settings["facts_max_tokens"], 120)
            self.assertEqual(settings["branch_max"], 3)
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

    def test_invalid_strategy_falls_back_to_default(self) -> None:
        """Неизвестная стратегия в конфиге не роняет запуск."""
        settings = get_strategy_settings({"STRATEGY": "нет-такой"})
        self.assertEqual(settings["name"], "sliding_window")

    def test_strategy_settings_validation_clamps(self) -> None:
        """Некорректные значения заменяются значениями по умолчанию."""
        settings = get_strategy_settings({
            "STRATEGY": "sticky_facts",
            "WINDOW_KEEP_LAST_N": "abc",
            "FACTS_MAX_TOKENS": 1,
            "FACTS_MAX_ITEMS": 99999,
            "FACTS_TEMPERATURE": 5.0,
            "BRANCH_MAX": 0,
        })
        self.assertEqual(settings["name"], "sticky_facts")
        self.assertEqual(settings["window"], 20)
        self.assertEqual(settings["facts_max_tokens"], 300)
        self.assertEqual(settings["facts_max_items"], 50)
        self.assertEqual(settings["facts_temperature"], 0.0)
        self.assertEqual(settings["branch_max"], 20)

    def test_branching_strategy_keeps_default_sub_strategy(self) -> None:
        settings = get_strategy_settings({"STRATEGY": "branching"})
        self.assertEqual(settings["name"], "branching")
        self.assertEqual(settings["branch_strategy"], "sliding_window")


if __name__ == "__main__":
    unittest.main()
