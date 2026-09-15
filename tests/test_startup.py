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
from src.config import (
    PROJECT_ROOT,
    get_config,
    get_memory_settings,
    get_strategy_settings,
)
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


class MemorySettingsTest(unittest.TestCase):
    """Модель памяти: настройки передаются агенту и валидируются."""

    def tearDown(self) -> None:
        shutil.rmtree(_TESTS_TMP, ignore_errors=True)

    def test_create_agent_passes_memory_settings(self) -> None:
        """create_agent() получает memory_settings из конфигурации."""
        agent = app_main.create_agent(_config())
        try:
            settings = agent.get_memory_settings()
            self.assertTrue(settings["enabled"])
            self.assertTrue(settings["working_enabled"])
            self.assertTrue(settings["long_enabled"])
            self.assertEqual(settings["long_path"], "data/long_memory.db")
            self.assertEqual(settings["pin_max"], 5)
            self.assertEqual(settings["max_tokens"], 600)
            self.assertTrue(agent.get_memory_report()["enabled"])
        finally:
            agent.close()

    def test_long_path_resolved_relative_to_project_root(self) -> None:
        """Путь БД долговременной памяти считается от PROJECT_ROOT."""
        agent = app_main.create_agent(
            _config(MEMORY_LONG_PATH="data/test_long_startup.db")
        )
        try:
            self.assertEqual(
                agent._long_memory.db_path,
                PROJECT_ROOT / "data" / "test_long_startup.db",
            )
        finally:
            agent.close()

    def test_memory_enabled_false_disables_layers(self) -> None:
        """MEMORY_ENABLED=false отключает модель памяти целиком."""
        agent = app_main.create_agent(_config(MEMORY_ENABLED=False))
        try:
            self.assertFalse(agent.get_memory_settings()["enabled"])
            self.assertEqual(agent._memory_blocks(), [])
            report = agent.get_memory_report()
            self.assertFalse(report["enabled"])
        finally:
            agent.close()

    def test_memory_disabled_by_default_without_settings(self) -> None:
        """Агент без memory_settings ведёт себя как раньше (память выключена)."""
        from src.agent import DeepSeekAgent
        agent = DeepSeekAgent(api_key="k", model="deepseek-v4-flash")
        try:
            self.assertEqual(agent._memory_blocks(), [])
        finally:
            agent.close()

    def test_memory_settings_validation_clamps(self) -> None:
        """Некорректные значения заменяются дефолтом, бюджеты согласованы."""
        settings = get_memory_settings({
            "MEMORY_PIN_MAX": 999,
            "MEMORY_MAX_TOKENS": 50,
            "MEMORY_MIN_CONFIDENCE": 5.0,
            "MEMORY_LONG_MAX_TOKENS": 100000,
            "MEMORY_WORKING_MAX_TOKENS": 100000,
        })
        self.assertEqual(settings["pin_max"], 5)
        self.assertEqual(settings["max_tokens"], 600)
        self.assertEqual(settings["min_confidence"], 0.6)
        self.assertLessEqual(
            settings["long_max_tokens"] + settings["working_max_tokens"],
            settings["max_tokens"],
        )

    def test_empty_long_path_disables_long_layer(self) -> None:
        """Явная пустая строка MEMORY_LONG_PATH выключает долговременный слой."""
        settings = get_memory_settings({"MEMORY_LONG_PATH": ""})
        self.assertEqual(settings["long_path"], "")
        self.assertFalse(settings["long_enabled"])

    def test_mock_mode_does_not_create_long_db(self) -> None:
        """В mock-режиме долговременная БД не создаётся."""
        _TESTS_TMP.mkdir(parents=True, exist_ok=True)
        long_db = _TESTS_TMP / f"long_{uuid.uuid4().hex}.db"
        agent = app_main.create_agent(_config(
            MOCK_MODE=True, MEMORY_LONG_PATH=str(long_db),
        ))
        try:
            agent.remember_forever("profile", "имя", "Алекс")
            agent.send_message("привет")
            self.assertFalse(long_db.exists())
        finally:
            agent.close()
        self.assertFalse(long_db.exists())


if __name__ == "__main__":
    unittest.main()
