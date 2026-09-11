"""Регрессионные тесты поля ввода GUI (вставка и «повторный ввод»).

Проверяют, что:
- подсказка («Введите сообщение...») — отдельная метка и НЕ попадает
  в содержимое поля, поэтому после первой отправки можно сразу писать снова;
- вставка работает в ЛЮБОЙ раскладке клавиатуры: при русской раскладке Tk
  присылает keysym ``Cyrillic_em`` (иногда ``"??"``) вместо ``v``, и привязки
  вида ``<Control-v>`` не срабатывают — обработчик определяет клавишу по
  физическому коду (``keycode``);
- текст любого размера принимается без обрезки;
- «Отправить» видит вставленный/введённый текст.

Тесты требуют графической среды (Tk). Если дисплей недоступен, тесты
аккуратно пропускаются.

Замечание по реализации тестов: Tk в одном процессе доставляет сгенерированные
клавиатурные события только первому окну, поэтому окно создаётся один раз на
класс (``setUpClass``), а виджеты приложения пересоздаются перед каждым тестом.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from unittest import mock

from src.agent import DeepSeekAgent
from src.tokens import HeuristicCounter
from src.ui import PLACEHOLDER
from tests import gui_helpers

try:  # pragma: no cover — зависит от окружения
    import tkinter as tk
except ImportError:  # pragma: no cover
    tk = None  # type: ignore


class _FakeKeyEvent:
    """Минимальная замена события Tk для проверки обработчика Ctrl+клавиша."""

    def __init__(self, keysym: str, keycode: int | None = None) -> None:
        self.keysym = keysym
        self.keycode = keycode


class InputFieldRegressionTest(unittest.TestCase):
    """Поведение поля ввода: подсказка, вставка, повторный ввод."""

    root: "tk.Tk"
    _ChatApplication = None

    # ------------------------------------------------------------------
    # Общее окно на весь процесс (иначе Tk не доставляет клавиатурные события)
    # ------------------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = gui_helpers.get_root()
        if cls.root is None:
            raise unittest.SkipTest("графическая среда недоступна (tkinter/Tk)")

        from src.ui import ChatApplication

        cls._ChatApplication = ChatApplication

    @classmethod
    def tearDownClass(cls) -> None:
        # Общий root живёт до конца процесса (см. tests/gui_helpers.py):
        # при уничтожении окна Tk перестаёт доставлять клавиатурные события.
        pass

    def setUp(self) -> None:
        # Пересоздаём виджеты приложения на общем окне
        previous = getattr(self, "app", None)
        if previous is not None:
            previous._cancel_autosize()
        gui_helpers.reset_children(self.root)
        self.agent = DeepSeekAgent(
            api_key="", mock_mode=True, token_counter=HeuristicCounter()
        )
        self.app = self._ChatApplication(root=self.root, agent=self.agent)
        self.root.update()
        self.app._busy = False
        self.app._set_controls_enabled(True)
        self.app._clear_input()
        self.root.update()

    def _set_clipboard(self, text: str) -> None:
        """Кладёт текст в буфер обмена и даёт Tk стать его владельцем."""
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()  # без этого clipboard_get() может вернуть пусто

    # ------------------------------------------------------------------
    # Подсказка не является текстом поля
    # ------------------------------------------------------------------

    def test_placeholder_is_not_widget_text(self) -> None:
        """Поле пустое, а подсказка — отдельная метка."""
        self.assertEqual(self.app._input_text(), "")
        self.assertEqual(self.app.placeholder_label.cget("text"), PLACEHOLDER)
        self.assertEqual(self.app.placeholder_label.winfo_manager(), "place")

    def test_placeholder_hidden_when_text_typed(self) -> None:
        self.app.input.insert("1.0", "привет")
        self.app._on_input_modified()
        self.assertTrue(self.app._input_text().startswith("привет"))
        self.assertEqual(self.app.placeholder_label.winfo_manager(), "")

    # ------------------------------------------------------------------
    # Баг №2: повторный ввод после отправки
    # ------------------------------------------------------------------

    def test_second_input_after_send_is_visible(self) -> None:
        """После отправки следующий текст должен читаться как сообщение."""
        self.app.input.insert("1.0", "первый запрос")
        self.app._on_input_modified()
        self.assertEqual(self.app._get_input_text(), "первый запрос")

        self.app._on_send_clicked()
        self.root.update()
        self.app._poll_queue()
        self.root.update()

        self.assertEqual(self.app._input_text(), "")
        self.assertEqual(self.app.placeholder_label.winfo_manager(), "place")

        # Второй ввод — БЕЗ клика по полю (фокус уже в нём): это и был баг
        self.app.input.insert("1.0", "второй запрос")
        self.app._on_input_modified()
        self.assertEqual(self.app._get_input_text(), "второй запрос")
        self.assertNotIn(PLACEHOLDER, self.app._input_text())

    # ------------------------------------------------------------------
    # Баг №1: вставка из буфера обмена (в т.ч. в русской раскладке)
    # ------------------------------------------------------------------

    def test_paste_via_event(self) -> None:
        """<<Paste>> вставляет содержимое буфера ровно один раз."""
        self._set_clipboard("вставленный текст")
        self.app.input.focus_set()
        self.app.input.event_generate("<<Paste>>", when="now")
        self.root.update()
        self.assertEqual(self.app._get_input_text(), "вставленный текст")

    def test_ctrl_v_binding_is_layout_independent(self) -> None:
        """Ctrl+V обрабатывается общим обработчиком, а не привязкой <Control-v>."""
        self.assertTrue(self.app.input.bind("<Control-KeyPress>"))
        # Зависимых от латинской раскладки привязок быть не должно
        self.assertEqual(self.app.input.bind("<Control-v>"), "")
        self.assertEqual(self.app.input.bind("<Control-a>"), "")

    def test_ctrl_v_with_cyrillic_keysym(self) -> None:
        """Ctrl+V с кириллическим keysym вставляет текст."""
        self._set_clipboard("вставка по-русски")
        result = self.app._on_input_ctrl_key(_FakeKeyEvent("Cyrillic_em", 86))
        self.assertEqual(result, "break")
        self.assertEqual(self.app._get_input_text(), "вставка по-русски")

    def test_ctrl_v_with_unmapped_keysym_uses_keycode(self) -> None:
        """Если keysym не распознан ("??"), клавиша определяется по keycode."""
        self._set_clipboard("по физическому коду")
        self.app._on_input_ctrl_key(_FakeKeyEvent("??", 86))
        self.assertEqual(self.app._get_input_text(), "по физическому коду")

    @unittest.skipUnless(sys.platform == "win32",
                         "физические коды клавиш проверяются в Windows")
    def test_ctrl_v_real_keypress_in_any_layout(self) -> None:
        """Реальное нажатие клавиши V с Ctrl вставляет текст ровно один раз."""
        self._set_clipboard("реальное нажатие")
        self.app.input.focus_set()
        self.app.input.event_generate("<Control-KeyPress>", when="now",
                                      keycode=86)
        self.root.update()
        self.assertEqual(self.app._get_input_text(), "реальное нажатие")

    def test_ctrl_a_with_cyrillic_keysym(self) -> None:
        """Ctrl+A выделяет всё и при кириллическом keysym."""
        self.app._insert_into_input("текст для выделения")
        result = self.app._on_input_ctrl_key(_FakeKeyEvent("Cyrillic_ef", 65))
        self.assertEqual(result, "break")
        self.assertTrue(self.app.input.tag_ranges("sel"))

    def test_ctrl_z_undo_with_cyrillic_keysym(self) -> None:
        """Ctrl+Z отменяет ввод и при кириллическом keysym."""
        self.app._insert_into_input("отменяемый текст")
        self.app._on_input_ctrl_key(_FakeKeyEvent("Cyrillic_ya", 90))
        self.assertEqual(self.app._get_input_text(), "")

    def test_paste_with_empty_clipboard_reports_status(self) -> None:
        """Пустой буфер обмена: понятное сообщение, без исключений."""
        self.root.clipboard_clear()
        self.root.update()
        self.app._on_paste()
        self.assertIn("буфере обмена", self.app.status_label.cget("text"))
        self.assertEqual(self.app._get_input_text(), "")

    def test_paste_large_text_is_not_truncated(self) -> None:
        """Текст большого размера принимается полностью."""
        big = ("строка большого текста " * 5000).strip()  # ~115k символов
        self.app._insert_into_input(big)
        self.root.update()
        self.assertEqual(len(self.app._get_input_text()), len(big))
        self.app._autosize_input()
        self.assertLessEqual(int(self.app.input.cget("height")), 10)

    def test_paste_replaces_selection(self) -> None:
        self.app._insert_into_input("старый текст")
        self.app.input.tag_add("sel", "1.0", "end-1c")
        self.app._insert_into_input("новый")
        self.assertEqual(self.app._get_input_text(), "новый")

    def test_paste_is_ignored_while_busy(self) -> None:
        self._set_clipboard("нельзя")
        self.app._busy = True
        self.app._on_paste()
        self.assertEqual(self.app._get_input_text(), "")
        self.app._busy = False

    # ------------------------------------------------------------------
    # Отправка и очистка
    # ------------------------------------------------------------------

    def test_send_sees_pasted_text(self) -> None:
        """Кнопка «Отправить» использует вставленный текст, а не пустую строку."""
        self.app._insert_into_input("проверка отправки")
        with mock.patch.object(
            self.app.agent, "send_message", return_value={
                "text": "ответ", "model": "deepseek-v4-flash", "tokens": None,
                "response_time": 0.0, "finish_reason": "stop", "error": None,
            }
        ) as mocked:
            self.app._on_send_clicked()
            self.root.update()
            self.app._poll_queue()
            self.root.update()
        mocked.assert_called_once_with("проверка отправки")

    def test_paste_then_send_through_real_handler(self) -> None:
        """Ctrl+V в русской раскладке → текст попадает в агент."""
        self._set_clipboard("вставлено и отправлено")
        sent = []
        with mock.patch.object(
            self.app.agent, "send_message",
            side_effect=lambda text: (
                sent.append(text) or {
                    "text": "ok", "model": "deepseek-v4-flash", "tokens": None,
                    "response_time": 0.0, "finish_reason": "stop", "error": None,
                }
            ),
        ):
            self.app._on_input_ctrl_key(_FakeKeyEvent("Cyrillic_em", 86))
            self.app._on_send_clicked()
            self.root.update()
            self.app._poll_queue()
            self.root.update()
        self.assertEqual(sent, ["вставлено и отправлено"])

    def test_clear_input_restores_placeholder(self) -> None:
        self.app._insert_into_input("что-то")
        self.app._clear_input()
        self.assertEqual(self.app._input_text(), "")
        self.assertEqual(self.app.placeholder_label.winfo_manager(), "place")
        self.app._insert_into_input("снова текст")
        self.assertEqual(self.app._get_input_text(), "снова текст")

    def test_empty_input_not_sent(self) -> None:
        with mock.patch.object(self.app.agent, "send_message") as mocked:
            self.app._on_send_clicked()
            self.root.update()
        mocked.assert_not_called()
        self.assertIn("Введите сообщение", self.app.status_label.cget("text"))

    def test_select_all_input(self) -> None:
        self.app._insert_into_input("выделить меня")
        self.app._select_all_input()
        self.assertTrue(self.app.input.tag_ranges("sel"))


if __name__ == "__main__":
    unittest.main()
