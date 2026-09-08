"""Графический интерфейс приложения DeepSeek Chat (Tkinter).

Класс ChatApplication строит главное окно, организует панели
(история/чат, параметры, ввод, статус-бар), обрабатывает действия
пользователя и запускает запросы к агенту в фоновом потоке, чтобы
интерфейс не «зависал».

Обмен с фоновым потоком выполняется через queue.Queue: рабочий поток
кладёт результат, главный поток опрашивает очередь через root.after —
это потокобезопасный паттерн для Tkinter.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Dict, Optional

import tkinter as tk
from tkinter import messagebox, ttk

from . import utils
from .agent import DeepSeekAgent
from .config import get_config

# ---------------------------------------------------------------------------
# Константы оформления
# ---------------------------------------------------------------------------

COLORS = {
    "bg": "#F5F5F5",            # общий фон окна
    "user_bg": "#DCF8C6",       # фон сообщений пользователя
    "assistant_bg": "#FFFFFF",  # фон сообщений ассистента
    "assistant_border": "#E0E0E0",
    "params_bg": "#E8E8E8",     # панель параметров
    "button": "#007AFF",        # кнопки (iOS-стиль)
    "button_active": "#0066D6",
    "button_disabled": "#9DB8D8",
    "status_bg": "#333333",     # статус-бар
    "status_fg": "#FFFFFF",
    "input_bg": "#FFFFFF",
    "input_border": "#CCCCCC",
    "text": "#1A1A1A",
    "muted": "#888888",
}

FONTS = {
    "default": ("Segoe UI", 10),
    "header": ("Segoe UI", 10, "bold"),
    "small": ("Segoe UI", 8),
    "status": ("Segoe UI", 9),
}

PLACEHOLDER = "Введите сообщение..."

TEMP_MIN, TEMP_MAX = 0.0, 2.0
TOKENS_MIN, TOKENS_MAX = 50, 4000


class ChatApplication:
    """Главное окно и логика взаимодействия пользователя с агентом.

    Attributes:
        root: корневое окно Tk.
        agent: экземпляр DeepSeekAgent, которым управляет интерфейс.
        config: словарь конфигурации (из config.get_config).
    """

    def __init__(self, root: tk.Tk, agent: DeepSeekAgent,
                 config: Optional[Dict[str, Any]] = None) -> None:
        """Создаёт элементы интерфейса.

        Args:
            root: корневое окно Tk.
            agent: готовый экземпляр DeepSeekAgent.
            config: конфигурация; если None — читается автоматически.
        """
        self.root = root
        self.agent = agent
        self.config = config if config is not None else get_config()

        # Потокобезопасная очередь результатов запросов
        self._result_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._busy = False          # идёт ли запрос к API
        self._placeholder_active = True  # показан ли плейсхолдер во вводе
        self._last_retry_text: Optional[str] = None  # текст для «Повторить»

        self._configure_window()
        self._build_widgets()
        self._bind_events()
        self._apply_config_to_controls()
        self._refresh_chat()
        self._set_status(
            "Готов" + (" · mock-режим (без реального API)" if agent.is_mock else "")
        )

    # ------------------------------------------------------------------
    # Построение интерфейса
    # ------------------------------------------------------------------

    def _configure_window(self) -> None:
        """Настраивает главное окно: заголовок, размеры, минимум."""
        title = str(self.config.get("UI_TITLE", "DeepSeek Chat"))
        width = int(self.config.get("UI_WIDTH", 900))
        height = int(self.config.get("UI_HEIGHT", 700))

        self.root.title(title)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(640, 480)
        self.root.configure(bg=COLORS["bg"])
        # Опционально: self.root.iconbitmap("icon.ico")

    def _build_widgets(self) -> None:
        """Создаёт все виджеты и раскладывает их по панелям."""
        root = self.root

        # ---- 1. Верхняя панель (40 px): заголовок + кнопка очистки ----
        self.top_bar = tk.Frame(root, bg=COLORS["bg"], height=40)
        self.top_bar.pack(fill="x")
        self.top_bar.pack_propagate(False)

        tk.Label(
            self.top_bar, text="История чата", bg=COLORS["bg"],
            font=FONTS["header"], fg=COLORS["text"],
        ).pack(side="left", padx=12)

        self.clear_top_btn = self._make_button(
            self.top_bar, "Очистить историю", self._on_clear_history
        )
        self.clear_top_btn.pack(side="right", padx=10, pady=6)

        # ---- 2. Область чата (растягивается) ----
        chat_wrap = tk.Frame(root, bg=COLORS["bg"])
        chat_wrap.pack(fill="both", expand=True, padx=8, pady=(4, 4))

        self.chat = tk.Text(
            chat_wrap,
            wrap="word",
            state="disabled",
            bg=COLORS["bg"],
            fg=COLORS["text"],
            font=FONTS["default"],
            relief="flat",
            padx=8,
            pady=6,
            cursor="arrow",
            highlightthickness=1,
            highlightbackground=COLORS["input_border"],
            undo=False,
        )
        scrollbar = ttk.Scrollbar(
            chat_wrap, orient="vertical", command=self.chat.yview
        )
        self.chat.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.chat.pack(side="left", fill="both", expand=True)

        self._configure_text_tags()

        # ---- 3. Панель параметров (110 px) ----
        self.params_bar = tk.Frame(
            root, bg=COLORS["params_bg"], height=110
        )
        self.params_bar.pack(fill="x")
        self.params_bar.pack_propagate(False)

        self._build_params_widgets()

        # ---- 4. Панель ввода (нижняя) ----
        self.input_bar = tk.Frame(root, bg=COLORS["bg"])
        self.input_bar.pack(fill="x", padx=8, pady=(0, 6))

        self.input = tk.Text(
            self.input_bar,
            height=3,
            wrap="word",
            bg=COLORS["input_bg"],
            fg=COLORS["text"],
            font=FONTS["default"],
            relief="solid",
            bd=1,
            highlightthickness=1,
            highlightbackground=COLORS["input_border"],
            insertbackground=COLORS["text"],
        )
        self.input.pack(side="left", fill="x", expand=True)

        send_btn_wrap = tk.Frame(self.input_bar, bg=COLORS["bg"])
        send_btn_wrap.pack(side="right", padx=(8, 0), fill="y")

        self.send_btn = self._make_button(
            send_btn_wrap, "Отправить", self._on_send_clicked
        )
        self.send_btn.pack(fill="x", pady=(0, 6))
        self.send_btn.config(width=12)

        self.clear_input_btn = self._make_button(
            send_btn_wrap, "Очистить", self._on_clear_history, small=True
        )
        self.clear_input_btn.pack(fill="x")
        self.clear_input_btn.config(width=12)

        self._show_placeholder()

        # ---- 5. Статус-бар (30 px) ----
        self.status_bar = tk.Frame(root, bg=COLORS["status_bg"], height=30)
        self.status_bar.pack(fill="x")
        self.status_bar.pack_propagate(False)

        self.status_label = tk.Label(
            self.status_bar, text="", bg=COLORS["status_bg"],
            fg=COLORS["status_fg"], font=FONTS["status"], anchor="w",
        )
        self.status_label.pack(side="left", padx=10)

        self.info_label = tk.Label(
            self.status_bar, text="", bg=COLORS["status_bg"],
            fg=COLORS["status_fg"], font=FONTS["status"], anchor="e",
        )
        self.info_label.pack(side="right", padx=10)

    def _build_params_widgets(self) -> None:
        """Строит содержимое панели параметров (2 ряда)."""
        bar = self.params_bar

        # --- Ряд 1: модель / температура / max tokens ---
        row1 = tk.Frame(bar, bg=COLORS["params_bg"])
        row1.pack(fill="x", padx=10, pady=(8, 2))

        tk.Label(
            row1, text="Модель:", bg=COLORS["params_bg"], font=FONTS["default"]
        ).pack(side="left")

        self.model_var = tk.StringVar(value=self.agent.model)
        self.model_combo = ttk.Combobox(
            row1,
            textvariable=self.model_var,
            values=list(utils.AVAILABLE_MODELS),
            state="readonly",
            width=20,
        )
        self.model_combo.pack(side="left", padx=(4, 16))

        tk.Label(
            row1, text="Температура:", bg=COLORS["params_bg"],
            font=FONTS["default"],
        ).pack(side="left")

        self.temp_var = tk.DoubleVar(value=self.agent.temperature)
        self.temp_scale = tk.Scale(
            row1,
            from_=TEMP_MIN,
            to=TEMP_MAX,
            resolution=0.1,
            orient="horizontal",
            variable=self.temp_var,
            length=140,
            showvalue=False,
            bg=COLORS["params_bg"],
            fg=COLORS["text"],
            highlightthickness=0,
            troughcolor="#D0D0D0",
            command=self._on_temperature_change,
        )
        self.temp_scale.pack(side="left")

        self.temp_value_label = tk.Label(
            row1, text="", width=4, bg=COLORS["params_bg"],
            font=FONTS["default"], fg=COLORS["text"],
        )
        self.temp_value_label.pack(side="left", padx=(2, 16))

        tk.Label(
            row1, text="Max tokens:", bg=COLORS["params_bg"],
            font=FONTS["default"],
        ).pack(side="left")

        self.tokens_var = tk.StringVar(value=str(self.agent.max_tokens))
        self.tokens_spin = tk.Spinbox(
            row1,
            from_=TOKENS_MIN,
            to=TOKENS_MAX,
            increment=50,
            textvariable=self.tokens_var,
            width=7,
            font=FONTS["default"],
            command=self._on_max_tokens_change,
        )
        self.tokens_spin.pack(side="left", padx=(4, 0))

        # --- Ряд 2: системный промпт + кнопка «Применить» ---
        row2 = tk.Frame(bar, bg=COLORS["params_bg"])
        row2.pack(fill="x", padx=10, pady=(2, 8))

        tk.Label(
            row2, text="Системный промпт:", bg=COLORS["params_bg"],
            font=FONTS["default"],
        ).pack(side="left")

        self.prompt_entry = tk.Entry(
            row2,
            bg=COLORS["input_bg"],
            font=FONTS["default"],
            relief="solid",
            bd=1,
        )
        self.prompt_entry.pack(side="left", fill="x", expand=True, padx=(4, 8))

        self.apply_prompt_btn = self._make_button(
            row2, "Применить", self._on_apply_prompt
        )
        self.apply_prompt_btn.pack(side="left")

    # ------------------------------------------------------------------
    # Вспомогательные конструкторы
    # ------------------------------------------------------------------

    def _make_button(self, parent, text: str, command,
                     small: bool = False) -> tk.Button:
        """Создаёт кнопку в фирменном стиле (#007AFF, белый текст)."""
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=COLORS["button"],
            fg="#FFFFFF",
            activebackground=COLORS["button_active"],
            activeforeground="#FFFFFF",
            disabledforeground="#FFFFFF",
            relief="flat",
            bd=0,
            padx=10,
            pady=2 if not small else 1,
            font=FONTS["small"] if small else FONTS["default"],
            cursor="hand2",
        )

    def _configure_text_tags(self) -> None:
        """Настраивает теги форматирования сообщений в области чата."""
        chat = self.chat

        # Сообщения пользователя: светло-зелёный фон, выравнивание вправо
        chat.tag_configure(
            "user", background=COLORS["user_bg"],
            foreground=COLORS["text"], justify="right", lmargin2=20,
        )
        chat.tag_configure(
            "user_hdr", font=FONTS["header"], foreground="#3A6B2E",
        )

        # Сообщения ассистента: белый фон, выравнивание влево
        chat.tag_configure(
            "assistant", background=COLORS["assistant_bg"],
            foreground=COLORS["text"], justify="left",
        )
        chat.tag_configure(
            "assistant_hdr", font=FONTS["header"], foreground="#555555",
        )

    def _bind_events(self) -> None:
        """Привязывает события клавиатуры и фокуса."""
        # Ctrl+Enter — отправка сообщения
        self.input.bind("<Control-Return>", self._on_send_clicked)
        # Плейсхолдер
        self.input.bind("<FocusIn>", self._on_input_focus_in)
        self.input.bind("<FocusOut>", self._on_input_focus_out)
        # Применение max_tokens по Enter / уходу с поля
        self.tokens_spin.bind("<Return>", self._on_max_tokens_change)
        self.tokens_spin.bind("<FocusOut>", self._on_max_tokens_change)
        # Применение системного промпта по Enter в поле промпта
        self.prompt_entry.bind("<Return>", self._on_apply_prompt)
        # Модель
        self.model_combo.bind(
            "<<ComboboxSelected>>", self._on_model_change
        )
        # Закрытие окна
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Инициализация значений из конфигурации агента
    # ------------------------------------------------------------------

    def _apply_config_to_controls(self) -> None:
        """Синхронизирует виджеты с текущей конфигурацией агента."""
        self.model_var.set(self.agent.model)
        if self.agent.model not in self.model_combo["values"]:
            self.model_combo["values"] = (self.agent.model,) + tuple(
                utils.AVAILABLE_MODELS
            )

        self.temp_var.set(self.agent.temperature)
        self.temp_value_label.config(text=f"{self.agent.temperature:.1f}")

        self.tokens_var.set(str(self.agent.max_tokens))
        self.prompt_entry.delete(0, "end")
        self.prompt_entry.insert(0, self.agent.system_prompt)

    # ------------------------------------------------------------------
    # Плейсхолдер поля ввода
    # ------------------------------------------------------------------

    def _show_placeholder(self) -> None:
        """Вставляет серый текст-подсказку, если поле пустое."""
        if not self.input.get("1.0", "end-1c"):
            self.input.configure(fg=COLORS["muted"])
            self.input.insert("1.0", PLACEHOLDER)
            self._placeholder_active = True

    def _on_input_focus_in(self, _event=None) -> None:
        """Удаляет плейсхолдер при получении фокуса."""
        if self._placeholder_active:
            self.input.delete("1.0", "end")
            self.input.configure(fg=COLORS["text"])
            self._placeholder_active = False

    def _on_input_focus_out(self, _event=None) -> None:
        """Возвращает плейсхолдер, если поле осталось пустым."""
        if not self._busy and not self.input.get("1.0", "end-1c"):
            self._show_placeholder()

    def _get_input_text(self) -> str:
        """Возвращает введённый текст (пустая строка, если это плейсхолдер)."""
        if self._placeholder_active:
            return ""
        return self.input.get("1.0", "end-1c").strip()

    def _clear_input(self) -> None:
        """Очищает поле ввода и восстанавливает плейсхолдер."""
        self.input.delete("1.0", "end")
        self._placeholder_active = False
        self._show_placeholder()

    # ------------------------------------------------------------------
    # Отображение чата
    # ------------------------------------------------------------------

    def _refresh_chat(self) -> None:
        """Перерисовывает область чата из истории агента (источник истины).

        Системный промпт в чате не показывается.
        """
        self.chat.configure(state="normal")
        try:
            self.chat.delete("1.0", "end")
            for message in self.agent.history:
                role = message.get("role")
                if role in ("user", "assistant"):
                    self._insert_message(role, message.get("content", ""))
        finally:
            self.chat.configure(state="disabled")
        self._scroll_to_end()

    def _insert_message(self, role: str, content: str) -> None:
        """Вставляет одно сообщение (пузырь) в область чата.

        Args:
            role: "user" или "assistant".
            content: текст сообщения.
        """
        base_tag = "user" if role == "user" else "assistant"
        name = "Вы" if role == "user" else "Ассистент"
        timestamp = utils.format_time()

        header = f"[{timestamp}] {name}:"
        paragraphs = content.split("\n") or [""]

        self.chat.insert("end", "\n")  # отступ между сообщениями

        # Заголовок: время и роль
        self.chat.insert("end", header + "\n", (base_tag, f"{base_tag}_hdr"))
        # Тело сообщения
        for paragraph in paragraphs:
            self.chat.insert("end", paragraph + "\n", (base_tag,))

    def _scroll_to_end(self) -> None:
        """Прокручивает область чата вниз."""
        self.chat.see("end")
        self.root.update_idletasks()

    # ------------------------------------------------------------------
    # Статус-бар
    # ------------------------------------------------------------------

    def _set_status(self, text: str) -> None:
        """Показывает текст статуса слева в статус-баре."""
        self.status_label.config(text=text)

    def _set_info(self, text: str) -> None:
        """Показывает служебную информацию справа в статус-баре."""
        self.info_label.config(text=text)

    def _set_controls_enabled(self, enabled: bool) -> None:
        """Блокирует/разблокирует органы управления на время запроса.

        Для ttk.Combobox используется собственный метод state(), так как
        его обычное состояние — "readonly", а не "normal".
        """
        # tk-виджеты
        for widget in (
            self.send_btn,
            self.clear_top_btn,
            self.clear_input_btn,
            self.apply_prompt_btn,
            self.temp_scale,
            self.tokens_spin,
            self.prompt_entry,
            self.input,
        ):
            widget.configure(state="normal" if enabled else "disabled")

        # ttk.Combobox
        if enabled:
            self.model_combo.configure(state="readonly")
        else:
            self.model_combo.configure(state="disabled")

    # ------------------------------------------------------------------
    # Обработчики параметров
    # ------------------------------------------------------------------

    def _on_model_change(self, _event=None) -> None:
        """Применяет выбранную модель к агенту «на лету»."""
        model = self.model_var.get()
        try:
            self.agent.update_config(model=model)
            self._set_status(f"Модель изменена: {model}")
        except utils.AgentConfigError as exc:
            self._set_status(f"Ошибка: {exc}")

    def _on_temperature_change(self, raw_value: str) -> None:
        """Применяет новую температуру (вызывается ползунком)."""
        try:
            temperature = round(float(raw_value), 1)
        except (TypeError, ValueError):
            return
        self.temp_value_label.config(text=f"{temperature:.1f}")
        try:
            self.agent.update_config(temperature=temperature)
        except utils.AgentConfigError as exc:
            self._set_status(f"Ошибка: {exc}")

    def _on_max_tokens_change(self, _event=None) -> None:
        """Читает поле max tokens, валидирует и применяет значение."""
        raw = self.tokens_var.get().strip()
        try:
            value = int(raw)
            clamped = max(TOKENS_MIN, min(TOKENS_MAX, value))
            if clamped != value:
                self.tokens_var.set(str(clamped))
                value = clamped
            self.agent.update_config(max_tokens=value)
        except (ValueError, utils.AgentConfigError) as exc:
            self.tokens_var.set(str(self.agent.max_tokens))
            self._set_status(f"Ошибка: неверное значение max tokens — {exc}")

    def _on_apply_prompt(self, _event=None) -> None:
        """Применяет новый системный промпт (с предупреждением и сбросом)."""
        if self._busy:
            return
        new_prompt = self.prompt_entry.get().strip()
        if not new_prompt:
            self._set_status("Системный промпт не может быть пустым.")
            return
        if new_prompt == self.agent.system_prompt:
            self._set_status("Системный промпт не изменился.")
            return

        confirmed = messagebox.askyesno(
            "Смена системного промпта",
            "Изменение системного промпта очистит историю диалога.\n\n"
            "Продолжить?",
            parent=self.root,
        )
        if not confirmed:
            return

        try:
            self.agent.update_config(system_prompt=new_prompt)
            self._refresh_chat()
            self._set_status("Системный промпт применён, история очищена.")
        except utils.AgentConfigError as exc:
            self._set_status(f"Ошибка: {exc}")

    # ------------------------------------------------------------------
    # Обработчики отправки / очистки
    # ------------------------------------------------------------------

    def _on_clear_history(self) -> None:
        """Очищает историю диалога (с подтверждением)."""
        if self._busy:
            return
        if not self.agent.history[1:]:  # кроме системного промпта пусто
            self._set_status("История уже пуста.")
            return
        confirmed = messagebox.askyesno(
            "Очистить историю",
            "Удалить всю историю диалога?",
            parent=self.root,
        )
        if not confirmed:
            return
        self.agent.clear_history()
        self._refresh_chat()
        self._set_status("История очищена.")
        self._set_info("")

    def _on_send_clicked(self, _event=None) -> None:
        """Читает поле ввода и запускает отправку."""
        if self._busy:
            return
        text = self._get_input_text()
        if not text:
            self._set_status("Введите сообщение перед отправкой.")
            self.root.bell()
            self.input.focus_set()
            return
        self._clear_input()
        self._send_text(text)

    def _send_text(self, text: str) -> None:
        """Отправляет текст агенту в фоновом потоке.

        Сообщение пользователя немедленно показывается в чате, интерфейс
        блокируется, а результат обрабатывается по завершении потока.
        """
        if self._busy:
            return

        self._busy = True
        self._last_retry_text = text
        self._set_controls_enabled(False)
        self._set_status("Отправка...")
        self._set_info("")

        # Показываем сообщение пользователя сразу (для обратной связи)
        self.chat.configure(state="normal")
        try:
            self._insert_message("user", text)
        finally:
            self.chat.configure(state="disabled")
        self._scroll_to_end()

        worker = threading.Thread(
            target=self._worker, args=(text,), daemon=True, name="deepseek-request"
        )
        worker.start()

        # Запускаем опрос очереди результатов
        self.root.after(100, self._poll_queue)

    def _worker(self, text: str) -> None:
        """Рабочий поток: вызывает агента и кладёт результат в очередь.

        Args:
            text: текст сообщения пользователя.
        """
        try:
            result = self.agent.send_message(text)
        except Exception as exc:  # агент не должен бросать, но страхуемся
            result = {
                "text": None,
                "model": self.agent.model,
                "tokens": None,
                "response_time": 0.0,
                "finish_reason": None,
                "error": f"Непредвиденная ошибка: {exc}",
            }
        self._result_queue.put(result)

    def _poll_queue(self) -> None:
        """Опрашивает очередь результатов (вызывается через root.after)."""
        try:
            while True:
                result = self._result_queue.get_nowait()
                self._on_result(result)
        except queue.Empty:
            pass
        finally:
            if self._busy:
                self.root.after(100, self._poll_queue)

    def _on_result(self, result: Dict[str, Any]) -> None:
        """Обрабатывает результат запроса (вызывается в главном потоке)."""
        self._busy = False
        self._set_controls_enabled(True)

        # Перерисовываем чат из истории агента — единственный источник правды
        self._refresh_chat()

        if result.get("error"):
            self._handle_error(result)
            return

        text = result.get("text") or ""
        model = result.get("model") or self.agent.model
        tokens = result.get("tokens")
        response_time = result.get("response_time")
        finish_reason = result.get("finish_reason")

        token_str = utils.format_tokens(tokens)
        time_str = f"{response_time:.2f} с" if response_time is not None else "—"
        reason_str = finish_reason or "—"

        self._set_status("Готов.")
        self._set_info(f"{model} · токены {token_str} · {time_str} · {reason_str}")

    def _handle_error(self, result: Dict[str, Any]) -> None:
        """Показывает ошибку: статус-бар + диалог с кнопкой «Повторить».

        Args:
            result: словарь-результат с непустым полем error.
        """
        error_text = result.get("error") or "Неизвестная ошибка."
        short = error_text if len(error_text) <= 160 else error_text[:157] + "..."
        self._set_status(f"Ошибка: {short}")
        self._set_info("")

        retry_text = self._last_retry_text
        retry = messagebox.askretrycancel(
            "Ошибка запроса",
            f"Не удалось получить ответ от API:\n\n{error_text}\n\n"
            "Нажмите «Повторить», чтобы отправить сообщение снова.",
            parent=self.root,
        )
        if retry:
            if retry_text:
                self._send_text(retry_text)
        else:
            self.input.focus_set()

    def _on_close(self) -> None:
        """Корректно завершает приложение при закрытии окна."""
        self.root.destroy()
