"""Тесты управления контекстом: сжатие истории в конспект (summary).

Проверяют офлайн (без сети):
- условие срабатывания и границы блока сжатия;
- сборку запроса: после сжатия уходят system-промпт + system-конспект +
  последние N сообщений, а покрытые сообщения в запрос не попадают;
- инкрементальное обновление конспекта (старый конспект + новый блок);
- отдельный учёт служебных токенов (не смешивается со статистикой диалога);
- сохранение и восстановление конспекта при «перезапуске»;
- отсутствие потерь при ошибке сжатия и повторную попытку;
- очистку конспекта (вручную, вместе с историей, при смене промпта);
- поведение при выключенном сжатии и в mock-режиме;
- обратную совместимость со старой БД.

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
from src.summarizer import (
    SUMMARY_MARKER,
    ContextSummarizer,
    should_summarize,
)
from src.tokens import HeuristicCounter
from src.utils import ApiError
from tests import gui_helpers

_TESTS_TMP = Path(__file__).resolve().parent / "tmp_summary"

# Быстрые настройки для тестов: сжимаем каждые 2 хода, держим 2 сообщения
_FAST_SETTINGS = {
    "enabled": True,
    "keep_last": 2,
    "every_n": 4,
    "max_tokens": 200,
    "temperature": 0.2,
    "model": "",
    "prompt": "",
}


def _make_db() -> Path:
    return _TESTS_TMP / f"summary_{uuid.uuid4().hex}.db"


def _chat_payload(content: str = "Ответ", prompt: int = 10,
                  completion: int = 5) -> dict:
    """Ответ обычного запроса диалога (как _handle_response)."""
    return {
        "content": content,
        "model": "deepseek-v4-flash",
        "tokens": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "estimated": False,
            "cache_hit_tokens": None,
            "cache_miss_tokens": None,
        },
        "finish_reason": "stop",
    }


def _summary_payload(text: str = "Факты: пользователя зовут Алекс.",
                     prompt: int = 50, completion: int = 20) -> dict:
    """Ответ служебного запроса сжатия."""
    return {
        "content": text,
        "model": "deepseek-v4-flash",
        "tokens": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "estimated": False,
            "cache_hit_tokens": None,
            "cache_miss_tokens": None,
        },
        "finish_reason": "stop",
    }


class SummarizerUnitTest(unittest.TestCase):
    """Юнит-тесты суммаризатора (без агента)."""

    def test_should_summarize_thresholds(self) -> None:
        # keep_last=2, every_n=4
        self.assertFalse(should_summarize(5, 0, 2, 4))   # лишних 3
        self.assertFalse(should_summarize(6, 0, 2, 4) is False)  # ровно 4 → True
        self.assertTrue(should_summarize(6, 0, 2, 4))
        self.assertFalse(should_summarize(8, 4, 2, 4))   # уже покрыто 4
        self.assertTrue(should_summarize(10, 4, 2, 4))   # накопилось 4 новых
        self.assertFalse(should_summarize(0, 0, 10, 10))

    def test_chunk_bounds(self) -> None:
        # диалог 12 сообщений, держим 2 → сжимаем [0, 10)
        self.assertEqual(ContextSummarizer.chunk_bounds(12, 0, 2), (0, 10))
        # уже покрыто 4 → сжимаем [4, 10)
        self.assertEqual(ContextSummarizer.chunk_bounds(12, 4, 2), (4, 10))
        # нечего сжимать
        self.assertEqual(ContextSummarizer.chunk_bounds(2, 0, 2), (0, 0))
        # covered больше границы — не уходим за неё
        self.assertEqual(ContextSummarizer.chunk_bounds(12, 10, 2), (10, 10))

    def test_build_request_contains_marker_and_previous_summary(self) -> None:
        summarizer = ContextSummarizer()
        request = summarizer.build_request(
            "Старый конспект.",
            [{"role": "user", "content": "вопрос"},
             {"role": "assistant", "content": "ответ"}],
        )
        joined = "\n".join(m["content"] for m in request["messages"])
        self.assertIn(SUMMARY_MARKER, joined)
        self.assertIn("Старый конспект.", joined)
        self.assertIn("Пользователь: вопрос", joined)
        self.assertIn("Ассистент: ответ", joined)

    def test_truncate_respects_limit(self) -> None:
        counter = HeuristicCounter()
        long_text = " ".join(f"факт{i}" for i in range(500))
        truncated = ContextSummarizer.truncate(long_text, 30, counter)
        self.assertLessEqual(counter.count_text(truncated), 30)
        self.assertTrue(truncated.endswith("…"))
        short_text = "короткий конспект"
        self.assertEqual(
            ContextSummarizer.truncate(short_text, 100, counter), short_text
        )


class AgentSummarizationTest(unittest.TestCase):
    """Сжатие контекста внутри агента."""

    def setUp(self) -> None:
        self.db_path = _make_db()
        self.service_requests: list = []
        self.service_texts: list = []
        self._turn_number = 0

    def tearDown(self) -> None:
        shutil.rmtree(_TESTS_TMP, ignore_errors=True)

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _make_agent(self, settings=None, storage=None,
                    counter=None) -> DeepSeekAgent:
        return DeepSeekAgent(
            api_key="k",
            model="deepseek-v4-flash",
            system_prompt="Системный промпт.",
            max_history_messages=50,
            storage=storage,
            token_counter=counter or HeuristicCounter(),
            prices={"default": {"input": 1.0, "output": 1.0}, "models": {}},
            summary_settings=dict(settings if settings is not None
                                  else _FAST_SETTINGS),
        )

    def _patch(self, agent: DeepSeekAgent, reply: str = "Ответ",
               summary_text: str = "Факты: пользователя зовут Алекс.",
               service_error: bool = False):
        """Подменяет обычный и служебный вызовы API (сеть не используется)."""
        answers = {"n": 0}

        def fake_chat():
            answers["n"] += 1
            return _chat_payload(f"Ответ {answers['n']}")

        def fake_service(request):
            self.service_requests.append(request)
            self.service_texts.append(summary_text)
            if service_error:
                raise ApiError("Сетевая ошибка: служебный запрос не прошёл.")
            return _summary_payload(summary_text)

        agent._call_api = fake_chat
        agent._call_service_api = fake_service
        return answers

    def _run_turns(self, agent: DeepSeekAgent, count: int) -> list:
        """Отправляет ``count`` сообщений с непрерывной нумерацией ходов."""
        results = []
        for _ in range(count):
            self._turn_number += 1
            results.append(agent.send_message(f"Вопрос {self._turn_number}"))
        return results

    # ------------------------------------------------------------------
    # Сборка запроса и триггер
    # ------------------------------------------------------------------

    def test_no_summary_before_threshold(self) -> None:
        agent = self._make_agent()
        self._patch(agent)
        self._run_turns(agent, 2)  # диалог 4 сообщения, лишних 2 < 4
        self.assertEqual(len(self.service_requests), 0)
        self.assertFalse(agent.get_summary()["active"])
        agent.close()

    def test_summary_triggered_and_request_shrinks(self) -> None:
        agent = self._make_agent()
        self._patch(agent, summary_text="Факты: Алекс, любимый цвет — синий.")
        sent_messages: list = []

        # Перехватываем, что реально уходит в API на обычном запросе
        def fake_chat_capture():
            sent_messages.append(agent._build_request()["messages"])
            return _chat_payload("Ответ")

        agent._call_api = fake_chat_capture
        self._run_turns(agent, 3)   # диалог 6 → при 4-м ходе сработает сжатие
        self.assertEqual(len(self.service_requests), 0)
        self._run_turns(agent, 1)   # 4-й ход: сжатие + запрос
        self.assertEqual(len(self.service_requests), 1)

        summary = agent.get_summary()
        self.assertTrue(summary["active"])
        self.assertEqual(summary["covered"], 4)  # сжаты первые 2 обмена

        # Проверяем содержимое последнего запроса к API
        messages = sent_messages[-1]
        contents = [m["content"] for m in messages]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Системный промпт.", contents[0])
        self.assertEqual(messages[1]["role"], "system")
        self.assertIn("Краткое содержание предыдущего диалога", contents[1])
        self.assertIn("Алекс", contents[1])
        # Покрытые сообщения дословно не отправляются
        self.assertNotIn("Вопрос 1", "\n".join(contents))
        self.assertNotIn("Ответ 1", "\n".join(contents))
        # Последние keep_last сообщений — «как есть»
        self.assertIn("Вопрос 3", "\n".join(contents))
        agent.close()

    def test_incremental_summary_uses_previous(self) -> None:
        agent = self._make_agent()
        self._patch(agent, summary_text="Факты: шаг 1.")
        self._run_turns(agent, 4)   # первое сжатие
        first_count = len(self.service_requests)
        self.assertGreaterEqual(first_count, 1)
        # Первый служебный запрос идёт без предыдущего конспекта
        first_joined = "\n".join(
            m["content"] for m in self.service_requests[0]["messages"]
        )
        self.assertNotIn("Предыдущий конспект:", first_joined)

        # Меняем ответ сжатия и накапливаем ещё сообщения до новых триггеров
        self._patch(agent, summary_text="Факты: шаг 1 и шаг 2.")
        self._run_turns(agent, 4)
        self.assertGreater(len(self.service_requests), first_count)

        # В последнем служебном запросе есть предыдущий конспект (результат
        # предыдущего сжатия) и новый фрагмент диалога
        last_request = self.service_requests[-1]
        previous_text = self.service_texts[-2]  # что вернуло сжатие до него
        joined = "\n".join(m["content"] for m in last_request["messages"])
        self.assertIn("Предыдущий конспект:", joined)
        self.assertIn(previous_text, joined)
        self.assertIn("Новый фрагмент диалога:", joined)
        self.assertIn(SUMMARY_MARKER, joined)
        # Конспект действительно накапливается: покрыто больше, чем за 1 раз
        self.assertGreater(agent.get_summary()["covered"], 4)
        agent.close()

    def test_service_tokens_counted_separately(self) -> None:
        agent = self._make_agent()
        captures: list = []

        def fake_chat():
            return _chat_payload("Ответ", prompt=10, completion=5)

        def fake_service(request):
            captures.append(request)
            return _summary_payload(prompt=100, completion=40)

        agent._call_api = fake_chat
        agent._call_service_api = fake_service
        results = self._run_turns(agent, 4)

        stats = agent.get_token_stats()
        self.assertEqual(stats["session"]["summary_requests"], 1)
        self.assertEqual(stats["session"]["summary_prompt_tokens"], 100)
        self.assertEqual(stats["session"]["summary_completion_tokens"], 40)
        # Стоимость сжатия: 100/1M*1 + 40/1M*1 = 0.00014
        self.assertAlmostEqual(stats["session"]["summary_cost_usd"], 0.00014,
                               places=9)
        # Статистика диалога содержит только 4 обычных обмена
        self.assertEqual(stats["session"]["requests"], 4)
        self.assertEqual(stats["session"]["total_tokens"], 4 * 15)
        # Флаг «сжатие было в этом ходе» выставлен только на 4-м ходе
        self.assertFalse(results[2]["tokens"]["summarized_this_turn"])
        self.assertTrue(results[3]["tokens"]["summarized_this_turn"])
        self.assertEqual(results[3]["tokens"]["summary_covered"], 4)
        agent.close()

    def test_summary_error_keeps_history_and_retries(self) -> None:
        agent = self._make_agent()
        self._patch(agent, service_error=True)
        self._run_turns(agent, 4)   # попытка сжатия падает

        self.assertEqual(len(self.service_requests), 1)
        summary = agent.get_summary()
        self.assertFalse(summary["active"])
        self.assertEqual(summary["covered"], 0)
        self.assertIsNotNone(summary["error"])
        # История не потеряна
        self.assertEqual(len(agent.history), 1 + 8)

        # Следующий ход повторяет попытку (уже успешную)
        self._patch(agent, summary_text="Факты: восстановлено.")
        self._run_turns(agent, 1)
        self.assertEqual(len(self.service_requests), 2)
        self.assertTrue(agent.get_summary()["active"])
        agent.close()

    def test_disabled_by_default_on_plain_agent(self) -> None:
        """Без явных настроек сжатие выключено (обратная совместимость)."""
        agent = DeepSeekAgent(
            api_key="k", model="deepseek-v4-flash",
            token_counter=HeuristicCounter(),
            max_history_messages=4,
        )
        agent._call_api = lambda: _chat_payload("Ответ")
        service_calls = []
        agent._call_service_api = lambda request: (
            service_calls.append(request) or _summary_payload()
        )
        self._run_turns(agent, 6)
        self.assertEqual(service_calls, [])
        self.assertFalse(agent.get_summary()["enabled"])
        # Старое поведение сохранилось: история урезается по лимиту
        self.assertLessEqual(len(agent.history), 5)
        agent.close()

    def test_mock_mode_has_no_summary(self) -> None:
        agent = DeepSeekAgent(
            api_key="", mock_mode=True,
            token_counter=HeuristicCounter(),
            summary_settings=dict(_FAST_SETTINGS),
        )
        self._run_turns(agent, 6)
        self.assertFalse(agent.get_summary()["active"])
        self.assertEqual(agent.get_token_stats()["session"]["summary_requests"], 0)
        agent.close()

    # ------------------------------------------------------------------
    # Управление конспектом
    # ------------------------------------------------------------------

    def test_clear_summary_and_clear_history(self) -> None:
        agent = self._make_agent()
        self._patch(agent)
        self._run_turns(agent, 4)
        self.assertTrue(agent.get_summary()["active"])

        agent.clear_summary()
        self.assertFalse(agent.get_summary()["active"])
        self.assertEqual(agent.get_summary()["covered"], 0)
        # История диалога не тронута
        self.assertEqual(len(agent.history), 1 + 8)

        # Сжатие снова сработает и создаст конспект заново
        self._run_turns(agent, 4)
        self.assertTrue(agent.get_summary()["active"])

        agent.clear_history()
        self.assertFalse(agent.get_summary()["active"])
        self.assertEqual(len(agent.history), 1)
        agent.close()

    def test_system_prompt_change_clears_summary(self) -> None:
        agent = self._make_agent()
        self._patch(agent)
        self._run_turns(agent, 4)
        self.assertTrue(agent.get_summary()["active"])
        agent.update_config(system_prompt="Новый системный промпт.")
        self.assertFalse(agent.get_summary()["active"])
        self.assertEqual(len(agent.history), 1)
        agent.close()

    def test_summarize_now_force(self) -> None:
        agent = self._make_agent()
        self._patch(agent)
        self._run_turns(agent, 2)          # ещё до порога
        self.assertFalse(agent.get_summary()["active"])
        result = agent.summarize_now(force=True)
        self.assertTrue(result["updated"])
        self.assertEqual(result["covered"], 2)  # всё, кроме последних 2
        self.assertTrue(agent.get_summary()["active"])
        agent.close()

    def test_summarize_now_when_disabled(self) -> None:
        settings = dict(_FAST_SETTINGS, enabled=False)
        agent = self._make_agent(settings=settings)
        result = agent.summarize_now(force=True)
        self.assertFalse(result["updated"])
        self.assertIn("отключено", result["error"])
        agent.close()


class SummaryPersistenceTest(unittest.TestCase):
    """Конспект сохраняется в БД и восстанавливается при перезапуске."""

    def setUp(self) -> None:
        self.db_path = _make_db()
        self.service_requests: list = []
        self.service_texts: list = []
        self._turn_number = 0

    def tearDown(self) -> None:
        shutil.rmtree(_TESTS_TMP, ignore_errors=True)

    def _agent(self) -> DeepSeekAgent:
        agent = DeepSeekAgent(
            api_key="k", model="deepseek-v4-flash",
            system_prompt="Системный промпт.",
            storage=ContextStore(db_path=self.db_path, enabled=True),
            token_counter=HeuristicCounter(),
            summary_settings=dict(_FAST_SETTINGS),
        )

        def fake_chat():
            return _chat_payload("Ответ")

        def fake_service(request):
            self.service_requests.append(request)
            return _summary_payload("Факты: Алекс, любимый цвет — синий.")

        agent._call_api = fake_chat
        agent._call_service_api = fake_service
        return agent

    def test_summary_restored_without_resummarize(self) -> None:
        agent1 = self._agent()
        for index in range(4):
            agent1.send_message(f"Вопрос {index + 1}")
        summary_before = agent1.get_summary()
        self.assertTrue(summary_before["active"])
        self.assertEqual(summary_before["covered"], 4)
        agent1.close()
        service_calls_first_run = len(self.service_requests)

        # «Перезапуск»: новый агент на том же файле БД
        agent2 = self._agent()
        summary_after = agent2.get_summary()
        self.assertTrue(summary_after["active"])
        self.assertEqual(summary_after["covered"], 4)
        self.assertEqual(summary_after["text"], summary_before["text"])
        # На старте новые служебные запросы не выполнялись
        self.assertEqual(len(self.service_requests), service_calls_first_run)
        # Контекст второго агента содержит конспект
        contents = "\n".join(
            m["content"] for m in agent2._build_request()["messages"]
        )
        self.assertIn("Краткое содержание предыдущего диалога", contents)
        agent2.close()

    def test_cleared_summary_not_restored(self) -> None:
        agent1 = self._agent()
        for index in range(4):
            agent1.send_message(f"Вопрос {index + 1}")
        agent1.clear_summary()
        agent1.close()

        agent2 = self._agent()
        self.assertFalse(agent2.get_summary()["active"])
        agent2.close()

    def test_old_db_without_summary_key(self) -> None:
        store = ContextStore(db_path=self.db_path, enabled=True)
        store.save_state(  # summary не передан — как в старой версии
            messages=[
                {"role": "system", "content": "Системный промпт."},
                {"role": "user", "content": "вопрос"},
                {"role": "assistant", "content": "ответ"},
            ],
            params={"model": "deepseek-v4-flash",
                    "system_prompt": "Системный промпт.",
                    "temperature": 0.7, "max_tokens": 100},
        )
        store.close()

        agent = self._agent()
        self.assertEqual(len(agent.history), 3)
        self.assertFalse(agent.get_summary()["active"])
        agent.close()

    def test_inconsistent_summary_ignored(self) -> None:
        """Конспект, противоречащий истории (covered больше), не загружается."""
        store = ContextStore(db_path=self.db_path, enabled=True)
        store.save_state(
            messages=[
                {"role": "system", "content": "Системный промпт."},
                {"role": "user", "content": "вопрос"},
            ],
            params={"model": "deepseek-v4-flash",
                    "system_prompt": "Системный промпт."},
            summary={"text": "битый конспект", "covered": 99},
        )
        store.close()

        agent = self._agent()
        self.assertFalse(agent.get_summary()["active"])
        agent.close()


class ContextStatsTest(unittest.TestCase):
    """Статистика контекста и конспекта для UI."""

    def setUp(self) -> None:
        self.db_path = _make_db()

    def tearDown(self) -> None:
        shutil.rmtree(_TESTS_TMP, ignore_errors=True)

    def test_context_stats_expose_summary(self) -> None:
        agent = DeepSeekAgent(
            api_key="k", model="deepseek-v4-flash",
            system_prompt="Системный промпт.",
            token_counter=HeuristicCounter(),
            summary_settings=dict(_FAST_SETTINGS),
        )
        agent._call_api = lambda: _chat_payload("Ответ")
        agent._call_service_api = lambda request: _summary_payload("Факты: Алекс.")
        for index in range(4):
            agent.send_message(f"Вопрос {index + 1}")

        stats = agent.get_token_stats()
        context = stats["context"]
        self.assertTrue(context["summary_active"])
        self.assertEqual(context["summary_covered"], 4)
        self.assertGreater(context["summary_tokens"], 0)
        self.assertGreater(context["messages_raw"], context["messages_sent"])
        self.assertTrue(stats["summary"]["active"])
        self.assertIsNotNone(stats["summary"]["updated_at"])
        agent.close()


class ContextDialogTest(unittest.TestCase):
    """Окно «Контекст»: просмотр, очистка и пустое состояние.

    Требует графической среды; без дисплея тесты пропускаются.
    """

    root = None
    _ChatApplication = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = gui_helpers.get_root()
        if cls.root is None:
            raise unittest.SkipTest("графическая среда недоступна (tkinter/Tk)")
        from src.ui import ChatApplication

        cls._ChatApplication = ChatApplication

    @classmethod
    def tearDownClass(cls) -> None:
        # Общий root живёт до конца процесса (см. tests/gui_helpers.py)
        pass

    def setUp(self) -> None:
        previous = getattr(self, "app", None)
        if previous is not None:
            previous._cancel_autosize()
        gui_helpers.reset_children(self.root)
        self.agent = DeepSeekAgent(
            api_key="k", model="deepseek-v4-flash",
            system_prompt="Системный промпт.",
            token_counter=HeuristicCounter(),
            summary_settings=dict(_FAST_SETTINGS),
        )
        self.agent._call_api = lambda: _chat_payload("Ответ")
        self.agent._call_service_api = lambda request: _summary_payload(
            "Факты: пользователя зовут Алекс, любимый цвет — синий."
        )
        self.app = self._ChatApplication(root=self.root, agent=self.agent)
        self.root.update()

    def tearDown(self) -> None:
        self.app._close_context_dialog()
        self.agent.close()

    def test_dialog_shows_summary_and_clears(self) -> None:
        for index in range(4):
            self.agent.send_message(f"Вопрос {index + 1}")
        self.app._open_context_dialog()
        self.root.update()

        text = self.app.context_text.get("1.0", "end-1c")
        self.assertIn("Алекс", text)
        info = self.app.context_info_label.cget("text")
        self.assertIn("Покрыто сообщений: 4", info)
        self.assertIn("включено", info)
        self.assertEqual(
            str(self.app.context_clear_btn.cget("state")), "normal"
        )

        # «Очистить» (подтверждение диалога подменяем)
        with mock.patch("tkinter.messagebox.askyesno", return_value=True):
            self.app._on_context_clear()
        self.root.update()
        self.assertFalse(self.agent.get_summary()["active"])
        self.assertIn("Конспекта пока нет", self.app.context_text.get("1.0", "end-1c"))
        self.assertEqual(
            str(self.app.context_clear_btn.cget("state")), "disabled"
        )
        self.assertIn("очищен", self.app.status_label.cget("text"))

    def test_dialog_empty_state(self) -> None:
        self.app._open_context_dialog()
        self.root.update()
        self.assertIn(
            "Конспекта пока нет", self.app.context_text.get("1.0", "end-1c")
        )
        self.assertIn("Покрыто сообщений: 0",
                      self.app.context_info_label.cget("text"))

    def test_dialog_reuse_single_window(self) -> None:
        self.app._open_context_dialog()
        first = self.app._context_window
        self.app._open_context_dialog()   # повторный вызов — то же окно
        self.assertIs(self.app._context_window, first)
        self.app._close_context_dialog()
        self.assertIsNone(self.app._context_window)


if __name__ == "__main__":
    unittest.main()
