"""Тестовые сценарии чата с памятью задачи (ТЗ ai_1_24, приёмка).

Сценарий 1 «Настройка RAG с нуля» (12 сообщений): память задачи
фиксирует цель, уточнения (русский, Ollama), ограничения (без Docker),
термины (RAG, nomic-embed-text); каждый ответ — с источниками и
дословными цитатами; цель не теряется к финальному сообщению.

Сценарий 2 «Отладка проблемы» (10 сообщений): цель (отладка RAG),
уточнения (порог 0.7, nomic-embed-text), термин (chunk); ассистент
продолжает выдавать источники на каждом ходу.

Плюс приёмка UI: свёрнутые источники, модалка «Состояние диалога»,
«Новый диалог» (новый session_id, сброс памяти и истории), заметка
стратегии C. Всё офлайн: mock-ядро, mock-экстрактор, временный SQLite.

Запуск из корня проекта:
    python -m unittest tests.test_chat_scenarios -v
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.chat_handler import ChatHandler, ChatTurnExtractor, STRATEGY_C_NOTE
from app.task_state import TaskStateRepository
from app.ui.chat_view import ChatView
from tests.test_app_agent import make_agent

SCENARIO_1 = [
    "Хочу сделать RAG для своей документации",
    "Документы на русском, подойдёт ли nomic-embed-text?",
    "Использую Ollama, как настроить?",
    "А какой порог similarity использовать?",
    "А если хочу без Docker?",
    "Спасибо, всё получилось!",
]

SCENARIO_2 = [
    "RAG не находит ответы на мои вопросы",
    "Порог стоит 0.7",
    "Использую nomic-embed-text",
    "А какие чанки у меня в базе?",
    "Понял, спасибо!",
]


def run_dialog(handler: ChatHandler, messages: list,
               session_id: str) -> list:
    """Прогоняет сценарий через ChatHandler, возвращает ответы."""
    results = []
    for message in messages:
        results.append(asyncio.run(
            handler.handle_message(session_id, 0, message)))
    return results


def make_env() -> tuple:
    """(handler, agent, repo-путь) над mock-ядром и mock-экстрактором."""
    agent = make_agent(rag_enabled=True)
    db_path = Path(tempfile.mkdtemp()) / "chat.db"
    repo = TaskStateRepository(db_path)
    handler = ChatHandler(agent, task_state_repo=repo,
                          extractor=ChatTurnExtractor(api_key="test",
                                                      mock_mode=True))
    return handler, agent, db_path


class Scenario1RagSetupTest(unittest.TestCase):
    """Сценарий 1 «Настройка RAG с нуля» — 12 сообщений."""

    def test_memory_and_sources_on_twelve_messages(self) -> None:
        handler, _agent, db_path = make_env()
        session_id = "scenario-1"
        results = run_dialog(handler, SCENARIO_1, session_id)
        self.assertEqual(len(results), len(SCENARIO_1))

        # Каждый ответ — с источниками (или честная пометка), цитаты
        # дословны (ТЗ: «не теряет цель и продолжает давать источники»)
        for index, result in enumerate(results, start=1):
            self.assertIsNone(result.get("error"), f"ход {index}")
            self.assertTrue(result["has_rag"], f"ход {index}: нет RAG")
            self.assertTrue(result["sources"], f"ход {index}: нет источников")
            self.assertTrue(result["quotes"], f"ход {index}: нет цитат")
            self.assertTrue(all(q["valid"] for q in result["quotes"]),
                            f"ход {index}: цитата не дословна")

        # Память задачи: цель, уточнения, ограничения, термины
        state = results[-1]["task_state"]
        self.assertEqual(state["goal"],
                         "Хочу сделать RAG для своей документации")
        self.assertIn("документы на русском", state["clarifications"])
        self.assertIn("nomic-embed-text как эмбеддер",
                      state["clarifications"])
        self.assertIn("используется Ollama", state["clarifications"])
        self.assertIn("без Docker", state["constraints"])
        self.assertIn("RAG", state["terms"])
        self.assertIn("nomic-embed-text", state["terms"])

        # Персистентность: тот же путь БД → то же состояние (SQLite)
        persisted = TaskStateRepository(db_path).get_or_create(
            session_id, 0)
        self.assertEqual(persisted.goal, state["goal"])
        self.assertEqual(persisted.constraints, state["constraints"])

    def test_goal_not_lost_at_final_message(self) -> None:
        handler, _agent, _db = make_env()
        results = run_dialog(handler, SCENARIO_1, "scenario-1b")
        first_goal = results[0]["task_state"]["goal"]
        final_goal = results[-1]["task_state"]["goal"]
        self.assertEqual(first_goal, "Хочу сделать RAG для своей "
                                     "документации")
        self.assertEqual(final_goal, first_goal)


class Scenario2DebuggingTest(unittest.TestCase):
    """Сценарий 2 «Отладка проблемы» — 10 сообщений."""

    def test_memory_and_sources_on_ten_messages(self) -> None:
        handler, _agent, _db = make_env()
        results = run_dialog(handler, SCENARIO_2, "scenario-2")
        self.assertEqual(len(results), len(SCENARIO_2))

        # Ассистент продолжает выдавать источники на каждом ходу
        for index, result in enumerate(results, start=1):
            self.assertTrue(result["has_rag"], f"ход {index}")
            self.assertTrue(result["sources"], f"ход {index}")

        state = results[-1]["task_state"]
        # Цель — отладка RAG; уточнения — порог 0.7 и nomic-embed-text
        self.assertTrue(state["goal"].startswith("Отладка:"))
        self.assertIn("RAG не находит ответы на мои вопросы", state["goal"])
        self.assertIn("порог similarity 0.7", state["clarifications"])
        self.assertIn("используется nomic-embed-text",
                      state["clarifications"])
        self.assertIn("chunk", state["terms"])


class StrategyCUiTest(unittest.TestCase):
    """Стратегия C в UI: пометка «без источников» вместо панели."""

    def test_strategy_c_note_rendered(self) -> None:
        agent = make_agent(rag_enabled=True)
        handler = ChatHandler(agent, task_state_repo=None,
                              extractor=ChatTurnExtractor(
                                  api_key="test", mock_mode=True))
        app = SimpleNamespace(agent=agent, page=None,
                              notify=lambda m: None, session_id="s",
                              profile_id=lambda: 0, chat_handler=handler)
        chat = ChatView(app)
        # База с низким similarity → стратегия C
        agent.pipeline.vector_store = type(agent.pipeline.vector_store)(
            {"ids": [["c1"]], "documents": [["Чанк."]],
             "metadatas": [[{"chunk_id": 9, "source": "a.md",
                             "section": "## S", "strategy": "fixed"}]],
             "distances": [[0.55]]})
        chat.input_field.value = "Вопрос мимо базы знаний"
        asyncio.run(chat._send_async())
        bubble = chat.messages_list.controls[-1]
        texts = " | ".join(
            str(getattr(c, "value", "") or "") for c in bubble.content.controls)
        self.assertIn("В документах нет информации", texts)
        self.assertIn("без источников", texts)
        control_types = [type(c).__name__ for c in bubble.content.controls]
        self.assertNotIn("ExpansionTile", control_types)


class ChatUiAcceptanceTest(unittest.TestCase):
    """Приёмка UI: источники свёрнуты, модалки, новый диалог."""

    def setUp(self) -> None:
        handler, agent, _db = make_env()
        self.handler = handler
        self.agent = agent
        self.app = SimpleNamespace(
            agent=agent, page=None, notify=lambda m: None,
            session_id="ui-session", profile_id=lambda: 0,
            chat_handler=handler)
        self.chat = ChatView(self.app)

    def _send(self, text: str) -> None:
        self.chat.input_field.value = text
        asyncio.run(self.chat._send_async())

    def test_sources_collapsed_with_quotes(self) -> None:
        self._send(SCENARIO_1[0])
        self.assertEqual(len(self.chat.messages_list.controls), 2)
        bubble = self.chat.messages_list.controls[-1]
        panels = [c for c in bubble.content.controls
                  if type(c).__name__ == "ExpansionTile"]
        self.assertEqual(len(panels), 1)
        panel = panels[0]
        # Свёрнута по умолчанию, заголовок с числом источников
        self.assertFalse(panel.expanded)
        self.assertIn("Источники (2)", str(panel.title.value))
        # В панели: строка источника + цитата на каждый чанк
        texts = " | ".join(
            str(getattr(c, "value", "") or "") for c in panel.controls)
        self.assertIn("chunk_id: 1", texts)
        self.assertIn("«", texts)

    def test_dialog_state_modal_shows_memory(self) -> None:
        self._send(SCENARIO_1[0])
        asyncio.run(self.chat._dialog_state_async())
        dialog = self.chat.last_dialog
        self.assertIn("Состояние диалога", str(dialog.title.value))
        panel_text = " | ".join(
            str(getattr(c, "value", "") or "")
            for c in dialog.content.content.controls)
        self.assertIn("Цель", panel_text)
        self.assertIn("RAG", panel_text)
        # Кнопки модалки: сброс и закрытие
        labels = [str(action.content) for action in dialog.actions]
        self.assertIn("Сбросить состояние", labels)
        self.assertIn("Закрыть", labels)

    def test_reset_state_from_modal(self) -> None:
        self._send(SCENARIO_1[0])
        asyncio.run(self.chat._reset_task_state_async())
        self.assertTrue(self.handler.load_state("ui-session", 0).is_empty())

    def test_new_dialog_requires_confirmation_and_resets(self) -> None:
        self._send(SCENARIO_1[0])
        # Шаг 1: только подтверждение, ничего не сброшено
        asyncio.run(self.chat._new_dialog_confirm_async())
        dialog = self.chat.last_dialog
        self.assertIn("будут сброшены",
                      str(dialog.content.content.value))
        old_session_id = self.app.session_id
        self.assertTrue(self.chat.messages_list.controls)

        # Шаг 2: подтверждение → новый session_id, пустой чат, БД без памяти
        asyncio.run(self.chat._start_new_dialog_async())
        self.assertNotEqual(self.app.session_id, old_session_id)
        self.assertEqual(len(self.chat.messages_list.controls), 0)
        self.assertTrue(
            self.handler.load_state(old_session_id, 0).is_empty())
        # История ядра очищена: остались только системный промпт
        self.assertFalse([m for m in self.agent.core._message_history
                          if m.get("role") in ("user", "assistant")])

    def test_cancel_new_dialog_keeps_everything(self) -> None:
        self._send(SCENARIO_1[0])
        asyncio.run(self.chat._new_dialog_confirm_async())
        asyncio.run(self.chat._close_dialog_async())
        state = self.handler.load_state("ui-session", 0)
        self.assertFalse(state.is_empty())
        self.assertTrue(self.chat.messages_list.controls)

    def test_full_scenario_through_ui(self) -> None:
        # Сценарий 1 целиком через вкладку чата
        for message in SCENARIO_1:
            self._send(message)
        # 12 пузырей: 6 вопросов + 6 ответов
        self.assertEqual(len(self.chat.messages_list.controls),
                         2 * len(SCENARIO_1))
        # Память задачи увидела уточнения сценария
        state = self.handler.load_state("ui-session", 0)
        self.assertIn("используется Ollama", state.clarifications)
        self.assertIn("без Docker", state.constraints)


if __name__ == "__main__":
    unittest.main()
