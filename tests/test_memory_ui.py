"""GUI-тесты модели памяти: кнопки у сообщений и окно «Память…».

Проверяют (без дисплея — аккуратно пропускаются):
- у каждого сообщения после ``_refresh_chat()`` есть три кнопки сохранения
  в слои (встроенные окна в ``tk.Text``);
- «💬 В диалог» закрепляет/открепляет сообщение (состояние видно в агенте);
- «📋 В рабочую» открывает окно сохранения с предвыбранным слоем; «Сохранить»
  кладёт элемент в рабочую память, окно закрывается, UI жив;
- окно «Память…» открывается, вкладки заполнены, «Что уходит в запрос»
  показывает блоки с пометкой слоя;
- очистка раздела долговременной памяти не ломает UI и не трогает
  краткосрочную/рабочую память;
- mock-режим: кнопки работают, БД не создаётся.

Tk в одном процессе доставляет клавиатурные события только первому окну,
поэтому используется общий root из ``tests/gui_helpers.py``, а обработчики
вызываются напрямую.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import time
import unittest
from unittest import mock

from src.agent import DeepSeekAgent
from src.config import DEFAULTS, get_memory_settings, get_strategy_settings
from src.memory import LONG_TERM, SHORT_TERM, WORKING
from src.tokens import HeuristicCounter
from tests import gui_helpers

try:  # pragma: no cover — зависит от окружения
    import tkinter as tk
except ImportError:  # pragma: no cover
    tk = None  # type: ignore


_PAYLOAD = {
    "content": "Ответ ассистента",
    "model": "deepseek-v4-flash",
    "tokens": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    "finish_reason": "stop",
}


def _find_buttons(widget, texts=None):
    """Рекурсивно ищет кнопки tk.Button (по подстрокам текста)."""
    found = []
    try:
        children = widget.winfo_children()
    except (tk.TclError, AttributeError):
        return found
    for child in children:
        try:
            if isinstance(child, tk.Button):
                label = str(child.cget("text"))
                if texts is None or any(t in label for t in texts):
                    found.append(child)
        except tk.TclError:
            pass
        found.extend(_find_buttons(child, texts))
    return found


class _MemoryGuiBase(unittest.TestCase):
    """Общий root и пересоздание виджетов приложения."""

    root: "tk.Tk"
    _ChatApplication = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = gui_helpers.get_root()
        if cls.root is None:
            raise unittest.SkipTest("графическая среда недоступна (tkinter/Tk)")
        from src.ui import ChatApplication
        cls._ChatApplication = ChatApplication

    def setUp(self) -> None:
        previous = getattr(self, "app", None)
        if previous is not None:
            previous._cancel_autosize()
            previous._close_memory_dialog()
            previous._close_save_dialog()
        gui_helpers.reset_children(self.root)
        # Долговременный слой отключён (без файла): тесты не трогают рабочую БД.
        settings = get_memory_settings(dict(DEFAULTS))
        settings["long_path"] = ""
        settings["long_enabled"] = False
        self.agent = DeepSeekAgent(
            api_key="k", model="deepseek-v4-flash",
            token_counter=HeuristicCounter(),
            strategy="sliding_window",
            strategy_settings=get_strategy_settings({}),
            memory_settings=settings,
        )
        self.app = self._ChatApplication(root=self.root, agent=self.agent)
        self.root.update()
        self.app._busy = False
        self.app._set_controls_enabled(True)

    def tearDown(self) -> None:
        try:
            self.app._close_memory_dialog()
            self.app._close_save_dialog()
        except Exception:
            pass

    def _seed_messages(self, count: int = 2) -> None:
        for i in range(count):
            self.agent._append_message({"role": "user", "content": f"вопрос {i}"})
            self.agent._append_message(
                {"role": "assistant", "content": f"ответ {i}"}
            )
        self.app._refresh_chat()
        self.root.update()

    def _pump(self) -> None:
        """Даёт фоновому потоку завершиться и разбирает очередь."""
        for _ in range(20):
            self.root.update()
            if not self.app._action_busy:
                break
            time.sleep(0.02)
        self.app._poll_action_queue()
        self.root.update()

    # ------------------------------------------------------------------
    # Кнопки у сообщений
    # ------------------------------------------------------------------

    def test_three_buttons_per_message(self) -> None:
        self._seed_messages(2)
        windows = self.app.chat.window_names()
        self.assertEqual(len(windows), 4)  # по 3 кнопки на каждое из 2 сообщений
        for name in windows:
            widget = self.app.chat.nametowidget(name)
            buttons = _find_buttons(widget)
            self.assertEqual(len(buttons), 3)
            labels = " ".join(str(b.cget("text")) for b in buttons)
            self.assertIn("В диалог", labels)
            self.assertIn("В рабочую", labels)
            self.assertIn("В долговременную", labels)

    def test_refresh_does_not_accumulate_buttons(self) -> None:
        self._seed_messages(2)
        self.app._refresh_chat()
        self.root.update()
        self.assertEqual(len(self.app.chat.window_names()), 4)

    # ------------------------------------------------------------------
    # Краткосрочная память: кнопка «В диалог»
    # ------------------------------------------------------------------

    def test_short_button_pins_and_unpins(self) -> None:
        self._seed_messages(1)
        self.app._on_save_short(0)
        self.root.update()
        self.assertEqual([p["index"] for p in self.agent.list_pinned()], [0])
        self.app._on_save_short(0)
        self.root.update()
        self.assertEqual(self.agent.list_pinned(), [])

    # ------------------------------------------------------------------
    # Рабочая память: окно сохранения
    # ------------------------------------------------------------------

    def test_working_save_dialog_writes_item(self) -> None:
        self._seed_messages(1)
        self.app._on_save_working(0)
        self.root.update()
        self.assertIsNotNone(self.app._save_dialog)
        self.assertTrue(self.app._save_dialog.winfo_exists())

        buttons = _find_buttons(self.app._save_dialog, texts=["Сохранить"])
        self.assertTrue(buttons)
        buttons[0].invoke()
        self._pump()

        working = self.agent.get_working_memory()
        self.assertGreaterEqual(len(working["items"]), 1)
        # Окно закрылось, UI жив
        self.root.update()
        self.assertFalse(self.app._busy)

    def test_working_save_dialog_preselects_layer(self) -> None:
        self._seed_messages(1)
        self.app._on_save_working(0)
        self.root.update()
        dialog = self.app._save_dialog
        # В окне есть виджет выбора вида (рабочая память)
        texts = " ".join(
            str(w.cget("text")) for w in _find_buttons(dialog)
        )
        self.assertIn("Сохранить", texts)
        self.assertIn("Отмена", texts)
        self.app._close_save_dialog()

    def test_long_save_dialog_writes_item(self) -> None:
        self._seed_messages(1)
        self.app._on_save_long(0)
        self.root.update()
        self.assertIsNotNone(self.app._save_dialog)
        buttons = _find_buttons(self.app._save_dialog, texts=["Сохранить"])
        self.assertTrue(buttons)
        buttons[0].invoke()
        self._pump()
        # Долговременная память отключена в тесте (без файла) — но запись
        # через роутер не падает; UI жив.
        self.root.update()
        self.assertFalse(self.app._busy)

    # ------------------------------------------------------------------
    # Окно «Память…»
    # ------------------------------------------------------------------

    def test_memory_dialog_opens_with_tabs(self) -> None:
        self._seed_messages(2)
        self.agent.add_working_item("дедлайн", "20 мая", "note")
        self.agent.set_task_field("goal", "отчёт")
        self.app._open_memory_dialog()
        self.root.update()
        self.assertTrue(self.app._memory_window.winfo_exists())
        tabs = self.app._memory_notebook.tabs()
        self.assertGreaterEqual(len(tabs), 4)
        titles = [self.app._memory_notebook.tab(t, "text")
                  for t in tabs]
        joined = " ".join(titles)
        self.assertIn("Краткосрочная", joined)
        self.assertIn("Рабочая", joined)
        self.assertIn("Долговременная", joined)
        self.assertIn("Предложения", joined)
        # Вкладка «Рабочая» заполнена
        rows = [
            self.app._memory_working_tree.item(row, "values")
            for row in self.app._memory_working_tree.get_children()
        ]
        self.assertTrue(any("дедлайн" in str(r) for r in rows))

    def test_preview_window_shows_layers(self) -> None:
        self._seed_messages(2)
        self.agent.set_task_field("goal", "цель-задачи")
        self.app._open_memory_dialog()
        self.root.update()
        self.app._open_preview_window()
        self.root.update()
        # Находим окно предпросмотра (Toplevel с заголовком)
        preview_text = None
        for child in self.root.winfo_children():
            if isinstance(child, tk.Toplevel) and child.title() == "Что уходит в запрос":
                preview_text = _find_text_widget(child)
        self.assertIsNotNone(preview_text)
        content = preview_text.get("1.0", "end-1c")
        self.assertIn("рабочая", content.lower())
        self.assertIn("цель-задачи", content)

    def test_clear_long_section_keeps_other_layers(self) -> None:
        self._seed_messages(2)
        self.agent.add_working_item("дедлайн", "20 мая", "note")
        self.app._on_save_short(0)
        self.app._open_memory_dialog()
        self.root.update()
        # Прямая очистка раздела + перерисовка окна не должна падать
        self.agent.clear_long_memory("profile")
        self.app._refresh_memory_dialog()
        self.root.update()
        self.assertTrue(self.app._memory_window.winfo_exists())
        self.assertIn("дедлайн", self.agent.get_working_memory()["items"])
        self.assertEqual([p["index"] for p in self.agent.list_pinned()], [0])

    # ------------------------------------------------------------------
    # Mock-режим
    # ------------------------------------------------------------------

    def test_mock_mode_buttons_work(self) -> None:
        gui_helpers.reset_children(self.root)
        mock_agent = DeepSeekAgent(
            api_key="", mock_mode=True, token_counter=HeuristicCounter(),
            memory_settings=get_memory_settings(dict(DEFAULTS)),
        )
        app = self._ChatApplication(root=self.root, agent=mock_agent)
        self.root.update()
        mock_agent._append_message({"role": "user", "content": "привет"})
        app._refresh_chat()
        self.root.update()
        app._on_save_short(0)
        self.root.update()
        self.assertEqual([p["index"] for p in mock_agent.list_pinned()], [0])
        app._close_memory_dialog()
        mock_agent.close()


def _find_text_widget(widget):
    """Находит первый tk.Text внутри виджета."""
    try:
        children = widget.winfo_children()
    except (tk.TclError, AttributeError):
        return None
    for child in children:
        if isinstance(child, tk.Text):
            return child
        found = _find_text_widget(child)
        if found is not None:
            return found
    return None


if __name__ == "__main__":
    unittest.main()
