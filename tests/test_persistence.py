"""Тесты сохранения/восстановления контекста (persistence) без сети.

Проверяют:
- агент сохраняет диалог и параметры в SQLite после каждого обмена;
- «перезапуск» (новый экземпляр агента на тот же файл БД) восстанавливает
  историю и параметры, переопределяя значения из конструктора;
- clear_history() и смена системного промпта очищают данные и на диске;
- mock-режим полностью изолирован от хранилища (не читает и не пишет);
- при ошибке API (откате сообщения) файл БД не создаётся.

Реальные HTTP-вызовы исключены: метод _call_api подменяется через
unittest.mock (mock-режим агента для этого не годится — он изолирован
от хранилища).

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path
from unittest import mock

from src.agent import DeepSeekAgent
from src.storage import ContextStore
from src.utils import ApiError

# Временная папка тестов ВНУТРИ репозитория: в песочнице системный TEMP
# может быть недоступен для записи/удаления.
_TESTS_TMP = Path(__file__).resolve().parent / "tmp_persistence"

_PAYLOAD = {
    "content": "Тестовый ответ",
    "model": "deepseek-v4-flash",
    "tokens": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    "finish_reason": "stop",
}


def _make_db() -> Path:
    """Возвращает уникальный путь к файлу БД внутри tests/tmp_persistence."""
    return _TESTS_TMP / f"test_{uuid.uuid4().hex}.db"


class PersistenceBase(unittest.TestCase):
    """Общая обвязка: уникальная БД на тест + очистка после теста."""

    def setUp(self) -> None:
        self.db_path = _make_db()

    def tearDown(self) -> None:
        # Лучший-effort: в песочнице удаление каталогов может быть
        # запрещено — ошибки игнорируем, каталог в .gitignore.
        shutil.rmtree(_TESTS_TMP, ignore_errors=True)

    def make_store(self) -> ContextStore:
        return ContextStore(db_path=self.db_path, enabled=True)

    @staticmethod
    def _payload(content: str) -> dict:
        return {**_PAYLOAD, "content": content}


class ContextRoundTripTest(PersistenceBase):
    """Сохранить → «перезапустить» → контекст восстановлен."""

    def test_restore_history_and_params(self) -> None:
        # --- Первый «запуск»: агент с хранилищем ---
        store1 = self.make_store()
        agent1 = DeepSeekAgent(
            api_key="k1",
            model="deepseek-v4-flash",
            system_prompt="Ты тестовая система.",
            temperature=0.3,
            max_tokens=777,
            storage=store1,
        )
        payloads = [
            self._payload("Ответ 1"),
            self._payload("Ответ 2"),
        ]
        with mock.patch.object(agent1, "_call_api", side_effect=payloads):
            r1 = agent1.send_message("Вопрос 1")
            r2 = agent1.send_message("Вопрос 2")
        self.assertIsNone(r1["error"])
        self.assertIsNone(r2["error"])
        # Файл БД создан после первого успешного обмена
        self.assertTrue(self.db_path.exists())

        expected_history = [
            {"role": "system", "content": "Ты тестовая система."},
            {"role": "user", "content": "Вопрос 1"},
            {"role": "assistant", "content": "Ответ 1"},
            {"role": "user", "content": "Вопрос 2"},
            {"role": "assistant", "content": "Ответ 2"},
        ]
        self.assertEqual(agent1.history, expected_history)
        agent1.close()

        # --- «Перезапуск»: новый агент на тот же файл БД ---
        # Намеренно другие значения в конструкторе — восстановленные
        # параметры должны их переопределить.
        store2 = self.make_store()
        agent2 = DeepSeekAgent(
            api_key="k2",
            model="deepseek-v4-pro",
            system_prompt="Другая система.",
            temperature=1.9,
            max_tokens=3000,
            storage=store2,
        )
        self.assertEqual(agent2.model, "deepseek-v4-flash")
        self.assertEqual(agent2.system_prompt, "Ты тестовая система.")
        self.assertEqual(agent2.temperature, 0.3)
        self.assertEqual(agent2.max_tokens, 777)
        self.assertEqual(agent2.history, expected_history)
        agent2.close()

    def test_restore_after_close_flush_without_exchange(self) -> None:
        """Параметры, изменённые без отправки сообщений, тоже сохраняются."""
        store1 = self.make_store()
        agent1 = DeepSeekAgent(
            api_key="k1", model="deepseek-v4-pro",
            temperature=1.5, max_tokens=2000, storage=store1,
        )
        agent1.update_config(model="deepseek-v4-flash", temperature=0.9)
        agent1.close()  # flush без единого обмена

        store2 = self.make_store()
        agent2 = DeepSeekAgent(
            api_key="k2", model="deepseek-v4-pro",
            temperature=1.5, max_tokens=2000, storage=store2,
        )
        self.assertEqual(agent2.model, "deepseek-v4-flash")
        self.assertEqual(agent2.temperature, 0.9)
        self.assertEqual(agent2.history, [
            {"role": "system", "content": agent2.system_prompt},
        ])
        agent2.close()

    def test_broken_db_does_not_crash_agent(self) -> None:
        """Битая БД не роняет запуск: агент стартует с состоянием по умолчанию."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.write_text("это не sqlite", encoding="utf-8")
        store = self.make_store()
        agent = DeepSeekAgent(
            api_key="k", model="deepseek-v4-flash", storage=store,
        )
        self.assertEqual(agent.model, "deepseek-v4-flash")
        self.assertEqual(len(agent.history), 1)
        agent.close()


class ClearSemanticsTest(PersistenceBase):
    """Очистка истории и смена промпта очищают данные и на диске."""

    def _seed_dialog(self) -> DeepSeekAgent:
        store = self.make_store()
        agent = DeepSeekAgent(
            api_key="k", model="deepseek-v4-flash",
            system_prompt="СП до очистки.", storage=store,
        )
        with mock.patch.object(agent, "_call_api", return_value=dict(_PAYLOAD)):
            agent.send_message("Вопрос")
        return agent

    def test_clear_history_clears_disk(self) -> None:
        agent1 = self._seed_dialog()
        agent1.clear_history()
        agent1.close()

        store2 = self.make_store()
        agent2 = DeepSeekAgent(
            api_key="k2", storage=store2,
            # Значение из «.env» отличается от сохранённого — восстановленные
            # параметры должны переопределить его.
            system_prompt="Промпт-заглушка из .env.",
        )
        # История пуста (только system), параметры (промпт) сохранены
        self.assertEqual(len(agent2.history), 1)
        self.assertEqual(agent2.history[0]["content"], "СП до очистки.")
        agent2.close()

    def test_system_prompt_change_clears_disk(self) -> None:
        agent1 = self._seed_dialog()
        agent1.update_config(system_prompt="СП новый.")
        agent1.close()

        store2 = self.make_store()
        agent2 = DeepSeekAgent(
            api_key="k2", storage=store2,
            system_prompt="Другой промпт для переопределения.",
        )
        self.assertEqual(len(agent2.history), 1)
        self.assertEqual(agent2.system_prompt, "СП новый.")
        self.assertEqual(agent2.history[0]["content"], "СП новый.")
        agent2.close()


class ErrorAndMockIsolationTest(PersistenceBase):
    """Ошибки API и mock-режим не пишут/не читают хранилище."""

    def test_api_error_does_not_create_db(self) -> None:
        store = self.make_store()
        agent = DeepSeekAgent(api_key="k", storage=store)
        with mock.patch.object(
            agent, "_call_api", side_effect=ApiError("Сетевая ошибка: таймаут.")
        ):
            result = agent.send_message("Вопрос")
        self.assertIsNotNone(result["error"])
        self.assertFalse(self.db_path.exists())
        agent.close()
        # close() без диалога и файла не создаёт БД «просто так»
        self.assertFalse(self.db_path.exists())

    def test_disabled_store_does_not_write(self) -> None:
        store = ContextStore(db_path=self.db_path, enabled=False)
        agent = DeepSeekAgent(api_key="k", storage=store)
        with mock.patch.object(agent, "_call_api", return_value=dict(_PAYLOAD)):
            result = agent.send_message("Вопрос")
        self.assertIsNone(result["error"])
        self.assertFalse(self.db_path.exists())
        agent.close()

    def test_mock_mode_never_creates_db(self) -> None:
        store = ContextStore(db_path=self.db_path, enabled=True)
        agent = DeepSeekAgent(api_key="", mock_mode=True, storage=store)
        result = agent.send_message("Привет из mock")
        self.assertIsNone(result["error"])
        self.assertFalse(self.db_path.exists())
        agent.close()
        self.assertFalse(self.db_path.exists())

    def test_mock_mode_does_not_load_existing_context(self) -> None:
        # 1) Реальный агент сохраняет диалог на диск
        real_store = self.make_store()
        real = DeepSeekAgent(api_key="k", storage=real_store)
        with mock.patch.object(real, "_call_api", return_value=dict(_PAYLOAD)):
            real.send_message("Секретный вопрос")
        real.close()

        # 2) Mock-агент на том же файле не должен загружать этот диалог
        mock_store = self.make_store()
        mock_agent = DeepSeekAgent(
            api_key="", mock_mode=True, storage=mock_store
        )
        self.assertEqual(len(mock_agent.history), 1)  # только system
        mock_agent.send_message("Ещё вопрос")
        mock_agent.close()

        # 3) Данные реального диалога на диске не тронуты
        check_store = self.make_store()
        state = check_store.load_state()
        self.assertIsNotNone(state)
        roles = [m["role"] for m in state["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant"])
        check_store.close()


if __name__ == "__main__":
    unittest.main()
