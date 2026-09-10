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

# Поле ввода: минимальная/максимальная высота в строках и порог, после
# которого авто-подгонка высоты отключается (чтобы не тормозить на
# очень больших вставках).
INPUT_MIN_LINES = 3
INPUT_MAX_LINES = 10
_AUTOSIZE_MAX_CHARS = 20_000

# Максимум символов сообщения, отображаемых в области чата (в поле ввода
# ограничений нет: там текст любого размера). Длинные сообщения рисуются
# частично, чтобы перерисовка чата не подвисала.
_CHAT_DISPLAY_MAX_CHARS = 20_000

# Виртуальные коды клавиш (Windows VK) для Ctrl-сочетаний.
# ВАЖНО: при русской (или любой нелатинской) раскладке Tk присылает в
# event.keysym значение вида "Cyrillic_em" (а иногда и вовсе "??"), поэтому
# стандартные привязки <Control-v> НЕ срабатывают и вставка «молчит».
# Определяем клавишу по физическому коду, а keysym используем как запасной путь.
VK_SELECT_ALL = 65   # A
VK_COPY = 67         # C
VK_CUT = 88          # X
VK_REDO = 89         # Y
VK_PASTE = 86        # V
VK_UNDO = 90         # Z
VK_RETURN = 13       # Enter

# Запасные keysym-имена (латиница + кириллическая раскладка ЙЦУКЕН)
_PASTE_KEYSYMS = {"v", "cyrillic_em"}
_SELECT_ALL_KEYSYMS = {"a", "cyrillic_ef"}
_COPY_KEYSYMS = {"c", "cyrillic_es"}
_CUT_KEYSYMS = {"x", "cyrillic_che"}
_UNDO_KEYSYMS = {"z", "cyrillic_ya"}
_REDO_KEYSYMS = {"y", "cyrillic_en"}
_RETURN_KEYSYMS = {"return", "kp_enter"}


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
        self._last_retry_text: Optional[str] = None  # текст для «Повторить»
        self._autosize_job: Optional[str] = None     # отложенная подгонка высоты

        self._configure_window()
        self._build_widgets()
        self._bind_events()
        # ВАЖНО: агент уже загрузил сохранённый контекст в своём __init__,
        # поэтому синхронизация виджетов ниже отражает восстановленные
        # параметры (модель, температура, max tokens, системный промпт).
        self._apply_config_to_controls()
        self._refresh_chat()
        self._set_initial_status()
        # Показываем статистику токенов (в т.ч. восстановленную сессию)
        self._update_token_stats()

    def _set_initial_status(self) -> None:
        """Показывает стартовый статус в статус-баре.

        Если агент восстановил сохранённый контекст (в истории больше
        одного сообщения), сообщаем об этом пользователю.
        """
        history_len = len(self.agent.history) - 1  # минус системный промпт
        if self.agent.is_mock:
            self._set_status("Готов · mock-режим (без реального API)")
        elif history_len > 0:
            self._set_status(
                f"Контекст восстановлен: {history_len} сообщений · "
                f"модель {self.agent.model}"
            )
        else:
            self._set_status("Готов")

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

        # ---- 3. Панель параметров (150 px: модель/темп/токены, промпт, статистика) ----
        self.params_bar = tk.Frame(
            root, bg=COLORS["params_bg"], height=150
        )
        self.params_bar.pack(fill="x")
        self.params_bar.pack_propagate(False)

        self._build_params_widgets()

        # ---- 4. Панель ввода (нижняя) ----
        self.input_bar = tk.Frame(root, bg=COLORS["bg"])
        self.input_bar.pack(fill="x", padx=8, pady=(0, 6))

        # Поле ввода со скроллбаром: принимает текст любого размера
        self.input_wrap = tk.Frame(self.input_bar, bg=COLORS["bg"])
        self.input_wrap.pack(side="left", fill="both", expand=True)

        self.input = tk.Text(
            self.input_wrap,
            height=INPUT_MIN_LINES,
            wrap="word",
            bg=COLORS["input_bg"],
            fg=COLORS["text"],
            font=FONTS["default"],
            relief="solid",
            bd=1,
            highlightthickness=1,
            highlightbackground=COLORS["input_border"],
            insertbackground=COLORS["text"],
            undo=True,
            maxundo=-1,
            autoseparators=True,
        )
        input_scroll = ttk.Scrollbar(
            self.input_wrap, orient="vertical", command=self.input.yview
        )
        self.input.configure(yscrollcommand=input_scroll.set)
        input_scroll.pack(side="right", fill="y")
        self.input.pack(side="left", fill="both", expand=True)

        # Подсказка — отдельная метка ПОВЕРХ поля, а не текст внутри него:
        # так ввод и вставка никогда не смешиваются с плейсхолдером.
        self.placeholder_label = tk.Label(
            self.input_wrap,
            text=PLACEHOLDER,
            bg=COLORS["input_bg"],
            fg=COLORS["muted"],
            font=FONTS["default"],
            cursor="xterm",
        )
        self.placeholder_label.bind(
            "<Button-1>", lambda _event: self.input.focus_set()
        )

        # Контекстное меню поля ввода (в т.ч. «Вставить» правой кнопкой)
        self.input_menu = tk.Menu(self.root, tearoff=0)
        self.input_menu.add_command(
            label="Вырезать", command=lambda: self.input.event_generate("<<Cut>>")
        )
        self.input_menu.add_command(
            label="Копировать",
            command=lambda: self.input.event_generate("<<Copy>>"),
        )
        self.input_menu.add_command(label="Вставить", command=self._on_paste)
        self.input_menu.add_separator()
        self.input_menu.add_command(
            label="Выделить всё", command=self._select_all_input
        )

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

        self._refresh_input_placeholder()

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

        # --- Ряд 3: статистика токенов (запрос / ответ / контекст / сессия) ---
        row3 = tk.Frame(bar, bg=COLORS["params_bg"])
        row3.pack(fill="x", padx=10, pady=(0, 8))

        tk.Label(
            row3, text="Токены:", bg=COLORS["params_bg"],
            font=FONTS["header"], fg=COLORS["text"],
        ).pack(side="left", padx=(0, 8))

        self.token_request_label = tk.Label(
            row3, text="Запрос: —", bg=COLORS["params_bg"],
            font=FONTS["default"], fg=COLORS["text"],
        )
        self.token_request_label.pack(side="left", padx=(0, 12))

        self.token_response_label = tk.Label(
            row3, text="Ответ: —", bg=COLORS["params_bg"],
            font=FONTS["default"], fg=COLORS["text"],
        )
        self.token_response_label.pack(side="left", padx=(0, 12))

        self.token_context_label = tk.Label(
            row3, text="Контекст: —", bg=COLORS["params_bg"],
            font=FONTS["default"], fg=COLORS["text"],
        )
        self.token_context_label.pack(side="left", padx=(0, 12))

        self.token_session_label = tk.Label(
            row3, text="Сессия: —", bg=COLORS["params_bg"],
            font=FONTS["default"], fg=COLORS["text"],
        )
        self.token_session_label.pack(side="left", padx=(0, 12))

        self.token_counter_label = tk.Label(
            row3, text="", bg=COLORS["params_bg"],
            font=FONTS["small"], fg=COLORS["muted"],
        )
        self.token_counter_label.pack(side="right")

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
        """Привязывает события клавиатуры, вставки и фокуса."""
        # Ctrl+Enter — отправка сообщения
        self.input.bind("<Control-Return>", self._on_send_clicked)
        # Любое изменение текста (набор, вставка, вырезание, undo) —
        # обновляем подсказку и высоту поля
        self.input.bind("<<Modified>>", self._on_input_modified)
        # Ctrl+<клавиша> — раскладко-независимая обработка (вставка, копирование,
        # выделение, отмена). При русской раскладке Tk НЕ вызывает <Control-v>,
        # поэтому используется общий обработчик с проверкой физического кода.
        self.input.bind("<Control-KeyPress>", self._on_input_ctrl_key)
        # Виртуальное событие вставки + классические Shift+Insert / Ctrl+Insert
        self.input.bind("<<Paste>>", self._on_paste)
        self.input.bind("<Control-Insert>", self._on_paste)
        self.input.bind("<Shift-Insert>", self._on_paste)
        # Правая кнопка мыши — контекстное меню
        self.input.bind("<Button-3>", self._show_input_menu)
        # Подсказка показывается/скрывается по фокусу
        self.input.bind("<FocusIn>", lambda _e: self._refresh_input_placeholder())
        self.input.bind("<FocusOut>", lambda _e: self._refresh_input_placeholder())
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
    # Поле ввода: подсказка, вставка, авторазмер
    # ------------------------------------------------------------------

    def _widgets_alive(self) -> bool:
        """True, если виджеты приложения ещё существуют.

        Защищает отложенные колбэки (авто-высота и т.п.) от обращения к уже
        уничтоженным виджетам — иначе Tk печатает TclError в фоне.
        """
        try:
            return bool(self.input.winfo_exists())
        except (tk.TclError, AttributeError):
            return False

    def _input_text(self) -> str:
        """Возвращает сырой текст поля ввода (без обрезки пробелов)."""
        return self.input.get("1.0", "end-1c")

    def _refresh_input_placeholder(self) -> None:
        """Показывает подсказку, только если поле пусто и не заблокировано.

        Подсказка — отдельная метка поверх поля, поэтому ввод и вставка
        никогда не смешиваются с ней (исправляет «залипание» плейсхолдера).
        """
        if not self._widgets_alive():
            return
        if self._busy or self._input_text().strip():
            self.placeholder_label.place_forget()
        else:
            self.placeholder_label.place(x=10, y=6)
            self.placeholder_label.lift()

    def _cancel_autosize(self) -> None:
        """Отменяет отложенную подгонку высоты поля ввода."""
        if self._autosize_job is None:
            return
        try:
            self.root.after_cancel(self._autosize_job)
        except tk.TclError:
            pass
        self._autosize_job = None

    def _on_input_modified(self, _event=None) -> None:
        """Реакция на любое изменение текста (в т.ч. вставку из буфера)."""
        if not self._widgets_alive():
            return
        # Сбрасываем флаг, иначе Tk не пришлёт <<Modified>> в следующий раз
        self.input.edit_modified(False)
        self._refresh_input_placeholder()
        # Высоту пересчитываем с небольшой задержкой (крупный текст/вставка)
        self._cancel_autosize()
        self._autosize_job = self.root.after(60, self._autosize_input)

    def _set_input_height(self, lines: int) -> None:
        """Устанавливает высоту поля ввода в строках (с ограничениями)."""
        lines = max(INPUT_MIN_LINES, min(INPUT_MAX_LINES, int(lines)))
        try:
            if int(self.input.cget("height")) != lines:
                self.input.configure(height=lines)
        except (tk.TclError, ValueError):
            pass

    def _autosize_input(self) -> None:
        """Подгоняет высоту поля под содержимое (3–10 строк).

        Для очень больших текстов подсчёт строк не выполняется — высота
        сразу выставляется максимальной, чтобы UI не подвисал.
        """
        self._autosize_job = None
        if not self._widgets_alive():
            return
        text = self._input_text()
        if len(text) > _AUTOSIZE_MAX_CHARS:
            self._set_input_height(INPUT_MAX_LINES)
            return
        try:
            self.input.update_idletasks()
            counted = self.input.count("1.0", "end-1c", "displaylines")
            lines = int(counted[0]) if counted else 1
        except (tk.TclError, TypeError, IndexError, ValueError):
            lines = len(text.split("\n")) or 1
        self._set_input_height(lines + 1)

    def _insert_into_input(self, text: str) -> None:
        """Вставляет текст в позицию курсора, заменяя выделение.

        Поддерживает текст произвольного размера (вставка не ограничена).
        Для очень больших вставок прокрутка к курсору не выполняется:
        ``see()`` заставляет Tk пересчитать разметку всего текста и надолго
        подвешивает интерфейс.
        """
        if not text:
            return
        try:
            if self.input.tag_ranges("sel"):
                self.input.delete("sel.first", "sel.last")
            self.input.insert("insert", text)
            if len(text) <= _AUTOSIZE_MAX_CHARS:
                self.input.see("insert")
            else:
                self.input.yview_moveto(0.0)
        except tk.TclError:
            return
        self._on_input_modified()

    def _clipboard_text(self) -> Optional[str]:
        """Читает текст из буфера обмена, пробуя несколько форматов.

        Returns:
            Текст из буфера или None, если текста там нет/он недоступен.
        """
        for kind in (None, "UTF8_STRING", "STRING"):
            try:
                data = (self.root.clipboard_get(type=kind) if kind
                        else self.root.clipboard_get())
            except tk.TclError:
                continue
            if isinstance(data, str) and data:
                return data
        return None

    def _on_input_ctrl_key(self, event):
        """Обрабатывает Ctrl+<клавиша> независимо от раскладки клавиатуры.

        Ключевая причина бага со вставкой: при русской раскладке физическая
        клавиша V приходит с ``event.keysym == "Cyrillic_em"`` (в некоторых
        сборках Tk — вовсе ``"??"``), поэтому привязка ``<Control-v>`` не
        срабатывает. Здесь клавиша определяется по ``event.keycode``
        (физический код клавиши), а keysym используется как запасной вариант.

        Args:
            event: событие нажатия клавиши с зажатым Ctrl.

        Returns:
            "break", если событие обработано, иначе None.
        """
        key = (event.keysym or "").lower()
        code = getattr(event, "keycode", None)

        if code == VK_PASTE or key in _PASTE_KEYSYMS:
            return self._on_paste()
        if code == VK_SELECT_ALL or key in _SELECT_ALL_KEYSYMS:
            return self._select_all_input()
        if code == VK_COPY or key in _COPY_KEYSYMS:
            self.input.event_generate("<<Copy>>")
            return "break"
        if code == VK_CUT or key in _CUT_KEYSYMS:
            self.input.event_generate("<<Cut>>")
            return "break"
        if code == VK_UNDO or key in _UNDO_KEYSYMS:
            return self._undo_input()
        if code == VK_REDO or key in _REDO_KEYSYMS:
            return self._redo_input()
        if code == VK_RETURN or key in _RETURN_KEYSYMS:
            # Ctrl+Enter (на случай, если сработала общая привязка)
            return self._on_send_clicked()
        return None

    def _undo_input(self):
        """Отменяет последнее изменение текста в поле ввода."""
        try:
            self.input.edit_undo()
        except tk.TclError:
            return "break"  # стек отмены пуст
        self._on_input_modified()
        return "break"

    def _redo_input(self):
        """Повторяет отменённое изменение текста в поле ввода."""
        try:
            self.input.edit_redo()
        except tk.TclError:
            return "break"  # нечего повторять
        self._on_input_modified()
        return "break"

    def _on_paste(self, _event=None):
        """Вставляет текст из буфера обмена.

        Вызывается из Ctrl+V (через _on_input_ctrl_key), Shift+Insert,
        виртуального события ``<<Paste>>`` и пункта меню «Вставить».
        Всегда возвращает "break", чтобы стандартная привязка Tk не вставила
        текст второй раз.

        Returns:
            "break".
        """
        if self._busy:  # во время запроса поле заблокировано
            self._set_status("Дождитесь ответа — поле ввода временно заблокировано.")
            return "break"
        text = self._clipboard_text()
        if not text:
            self._set_status("В буфере обмена нет текста для вставки.")
            return "break"
        self._insert_into_input(text)
        return "break"

    def _select_all_input(self, _event=None):
        """Выделяет весь текст в поле ввода (Ctrl+A в любой раскладке)."""
        try:
            self.input.tag_add("sel", "1.0", "end-1c")
            self.input.mark_set("insert", "1.0")
        except tk.TclError:
            pass
        return "break"

    def _show_input_menu(self, event) -> None:
        """Показывает контекстное меню поля ввода по правой кнопке мыши."""
        self.input.focus_set()
        try:
            self.input_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.input_menu.grab_release()

    def _get_input_text(self) -> str:
        """Возвращает введённый текст (подсказка — отдельная метка, не текст)."""
        return self._input_text().strip()

    def _clear_input(self) -> None:
        """Очищает поле ввода и возвращает подсказку."""
        self.input.delete("1.0", "end")
        self.input.edit_modified(False)
        self._autosize_input()
        self._refresh_input_placeholder()

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
        # Очень длинные сообщения рисуем частично — иначе перерисовка
        # чата заметно подтормаживает (в поле ввода ограничений нет).
        truncated = len(content) > _CHAT_DISPLAY_MAX_CHARS
        display_text = content[:_CHAT_DISPLAY_MAX_CHARS] if truncated else content
        paragraphs = display_text.split("\n") or [""]

        self.chat.insert("end", "\n")  # отступ между сообщениями

        # Заголовок: время и роль
        self.chat.insert("end", header + "\n", (base_tag, f"{base_tag}_hdr"))
        # Тело сообщения
        for paragraph in paragraphs:
            self.chat.insert("end", paragraph + "\n", (base_tag,))
        if truncated:
            self.chat.insert(
                "end",
                f"… сообщение показано частично "
                f"({len(content)} символов, лимит отображения "
                f"{_CHAT_DISPLAY_MAX_CHARS})\n",
                (base_tag,),
            )

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

    # ------------------------------------------------------------------
    # Статистика токенов
    # ------------------------------------------------------------------

    def _update_token_stats(self) -> None:
        """Обновляет блок «Токены» на панели параметров.

        Точные значения API показываются без пометки, оценочные — с «≈»,
        неизвестные — «—». Ошибки агента/счётчика не должны ломать UI.
        """
        try:
            stats = self.agent.get_token_stats()
        except Exception:
            return

        def _fmt(value, estimated: bool) -> str:
            if value is None:
                return "—"
            return f"≈{value}" if estimated else str(value)

        request = stats.get("request") or {}
        response = stats.get("response") or {}
        context = stats.get("context") or {}
        session = stats.get("session") or {}

        self.token_request_label.config(
            text="Запрос: " + _fmt(request.get("tokens"),
                                   bool(request.get("estimated")))
        )
        self.token_response_label.config(
            text="Ответ: " + _fmt(response.get("tokens"),
                                  bool(response.get("estimated")))
        )
        # Контекст всегда оценивается локально
        self.token_context_label.config(
            text="Контекст: " + _fmt(context.get("tokens"), True)
        )

        total = int(session.get("total_with_estimates", 0) or 0)
        estimated_any = int(session.get("estimated_total_tokens", 0) or 0) > 0
        if not session.get("prices_set"):
            cost_text = "—"
        else:
            cost = float(session.get("cost_usd") or 0.0) + float(
                session.get("estimated_cost_usd") or 0.0
            )
            cost_text = f"{'≈' if estimated_any else ''}${cost:.6f}"
        self.token_session_label.config(
            text=f"Сессия: {'≈' if estimated_any else ''}{total} токенов · {cost_text}"
        )
        self.token_counter_label.config(
            text=f"счётчик: {stats.get('counter', '')}"
        )

    def _token_context_text(self) -> str:
        """Возвращает строку с оценкой контекста для статус-бара."""
        try:
            stats = self.agent.get_token_stats()
        except Exception:
            return "ctx: —"
        context = stats.get("context") or {}
        tokens = context.get("tokens")
        if tokens is None:
            return "ctx: —"
        return f"ctx: ≈{tokens}"

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

        # Подсказка не должна висеть поверх заблокированного поля
        self._refresh_input_placeholder()

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
            self._update_token_stats()
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
        self._update_token_stats()
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
        # Обновляем статистику токенов (контекст мог измениться/откатиться)
        self._update_token_stats()

        if result.get("error"):
            self._handle_error(result)
            return

        text = result.get("text") or ""
        model = result.get("model") or self.agent.model
        tokens = result.get("tokens")
        response_time = result.get("response_time")
        finish_reason = result.get("finish_reason")

        token_str = utils.format_tokens(tokens)
        if isinstance(tokens, dict) and tokens.get("estimated"):
            token_str = f"≈{token_str}"
        time_str = f"{response_time:.2f} с" if response_time is not None else "—"
        reason_str = finish_reason or "—"
        context_str = self._token_context_text()

        self._set_status("Готов.")
        self._set_info(
            f"{model} · {token_str} · {context_str} · {time_str} · {reason_str}"
        )

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
        """Корректно завершает приложение при закрытии окна.

        Сначала — финальный flush контекста агентом (сохранение истории
        и параметров в хранилище), затем закрытие окна. Метод агента
        close() идемпотентен, повторный вызов из main.py безопасен.
        """
        self._cancel_autosize()
        self.agent.close()
        self.root.destroy()
