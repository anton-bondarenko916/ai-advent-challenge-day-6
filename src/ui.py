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

import json
import queue
import threading
from typing import Any, Dict, List, Optional

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from . import utils
from .agent import DeepSeekAgent
from .config import get_config
from .invariant_models import (
    SCOPE_GLOBAL,
    SCOPE_PROFILE,
    Invariant,
    InvariantCategory,
)
from .models import ProfileAgentSettings, UserProfile
from .task_models import (
    AWAITING_STAGES,
    MODE_TITLES,
    STAGE_TITLES,
    WORKING_STAGES,
    Task,
    TaskMode,
    TaskStage,
)

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

PLACEHOLDER = "Введите сообщение... (/задача — поставить задачу)"

# Команды чата, создающие задачу: остальной текст сообщения — описание.
# Задачи создаётся в ручном режиме (пользователь подтверждает план и
# итог); режим можно сменить на лету в панели задачи.
TASK_COMMANDS = ("/задача", "/task")

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
        # Окно «Контекст» и его фоновый воркер обновления фактов
        self._context_window: Optional[tk.Toplevel] = None
        self._context_busy = False
        self._context_pending_mode: Optional[str] = None
        self._summary_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()

        # Окно «Ветки» и общий механизм фоновых операций (ветки/факты)
        self._branches_window: Optional[tk.Toplevel] = None
        self._branches_busy = False
        self._action_queue: "queue.Queue[Any]" = queue.Queue()
        self._action_busy = False

        # Окно «Память…» и модальное окно сохранения в память
        self._memory_window: Optional[tk.Toplevel] = None
        self._save_dialog: Optional[tk.Toplevel] = None

        # Инварианты: модалка «Показать инварианты» в панели задачи,
        # форма инварианта, форма категории и данные ожидающего сохранения
        # (пока идёт LLM-проверка конфликтов).
        self._invariants_window: Optional[tk.Toplevel] = None
        self._invariant_dialog: Optional[tk.Toplevel] = None
        self._category_dialog: Optional[tk.Toplevel] = None
        self._pending_invariant_save: Optional[Dict[str, Any]] = None
        self._invariant_conflict_dialog: Optional[tk.Toplevel] = None

        # Подсистема задач (FSM): очередь результатов и флаг занятости.
        # Тяжёлые LLM-этапы задачи выполняются в фоновом потоке, UI
        # опрашивает очередь через root.after (как и обычные запросы).
        self._task_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._task_busy = False
        self._task_steps_visible = False   # раскрыт ли список шагов
        self._task_status_text = ""        # хвост статус-бара про задачу
        self._status_base_text = ""        # основной текст статус-бара

        # Индекс сообщения диалога (без system) под правой кнопкой мыши:
        # используется кнопкой «Создать от выбранного сообщения».
        self._chat_selected_index: Optional[int] = None
        self._chat_message_count = 0

        self._configure_window()
        self._build_widgets()
        self._bind_events()
        # ВАЖНО: агент уже загрузил сохранённый контекст в своём __init__,
        # поэтому синхронизация виджетов ниже отражает восстановленные
        # параметры (модель, температура, max tokens, системный промпт).
        self._apply_config_to_controls()
        self._refresh_chat()
        self._set_initial_status()
        # Персонализация: заполняем переключатель профиля и статус-бар
        self._update_profile_status()
        # Инварианты: счётчики в статус-баре и вкладка (активный профиль)
        self._update_invariant_status()
        self._refresh_invariants_tab()
        # Показываем статистику токенов (в т.ч. восстановленную сессию)
        self._update_token_stats()
        self._set_info(self._token_context_text())
        # Подсистема задач: восстанавливаем активную задачу профиля
        # (без авто-запуска; панель появится, если задача есть)
        self._restore_task_on_startup()

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

        # Подсистема задач: очистка всей истории задач (с подтверждением).
        # Отдельной кнопки создания задачи нет — задача создаётся
        # сообщением «/задача <описание>» в чате (см. _send_text).
        self.clear_tasks_btn = self._make_button(
            self.top_bar, "Очистить задачи", self._on_clear_tasks
        )
        self.clear_tasks_btn.pack(side="right", pady=6)

        # ---- 2. Вкладки: Чат / Профиль / Настройки агента ----
        # Вкладка «Чат» содержит область чата; «Профиль» — управление
        # профилями пользователя; «Настройки агента» — параметры, привязанные
        # к активному профилю. Панель ввода и статус-бар остаются общими.
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=(4, 0))

        chat_tab = tk.Frame(self.notebook, bg=COLORS["bg"])
        self.notebook.add(chat_tab, text="Чат")

        chat_wrap = tk.Frame(chat_tab, bg=COLORS["bg"])
        chat_wrap.pack(fill="both", expand=True, pady=(4, 4))

        # Панель задачи (FSM) над областью чата: этап, шаг, кнопки
        # управления. Показывается только при активной задаче.
        self._build_task_panel(chat_wrap)

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

        # Контекстное меню области чата: создание ветки и сохранение в память
        self.chat_menu = tk.Menu(self.root, tearoff=0)
        self.chat_menu.add_command(
            label="Создать ветку отсюда…", command=self._on_create_branch_from_chat
        )
        self.chat_menu.add_command(
            label="Сохранить в краткосрочную (закрепить/открепить)",
            command=self._on_menu_save_short,
        )
        self.chat_menu.add_command(
            label="Сохранить в рабочую память…", command=self._on_menu_save_working
        )
        self.chat_menu.add_command(
            label="Сохранить в долговременную память…",
            command=self._on_menu_save_long,
        )
        self.chat_menu.add_command(
            label="Показать, что сохранено об этом сообщении",
            command=self._on_menu_show_saved,
        )

        # ---- 2б. Вкладки «Профиль», «Инварианты», «Настройки агента» ----
        self._build_profile_tab()
        self._build_invariants_tab()
        self._build_agent_settings_tab()

        # ---- 3. Панель параметров (186 px: модель/темп/токены, промпт, стратегия, статистика) ----
        self.params_bar = tk.Frame(
            root, bg=COLORS["params_bg"], height=186
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

        # Персонализация: быстрый переключатель активного профиля
        tk.Label(
            self.status_bar, text="Профиль:", bg=COLORS["status_bg"],
            fg=COLORS["status_fg"], font=FONTS["status"],
        ).pack(side="left")
        self.profile_switch_var = tk.StringVar()
        self.profile_switch = ttk.Combobox(
            self.status_bar,
            textvariable=self.profile_switch_var,
            state="readonly",
            width=16,
        )
        self.profile_switch.pack(side="left", padx=(4, 10), pady=4)
        self.profile_switch.bind(
            "<<ComboboxSelected>>", self._on_status_profile_switch
        )

        self.info_label = tk.Label(
            self.status_bar, text="", bg=COLORS["status_bg"],
            fg=COLORS["status_fg"], font=FONTS["status"], anchor="e",
        )
        self.info_label.pack(side="right", padx=10)

        # Персонализация: активный профиль · модель · temperature
        self.profile_info_label = tk.Label(
            self.status_bar, text="", bg=COLORS["status_bg"],
            fg=COLORS["status_fg"], font=FONTS["status"], anchor="e",
        )
        self.profile_info_label.pack(side="right", padx=10)

        # Инварианты: счётчик активных правил (global + активный профиль)
        self.invariant_info_label = tk.Label(
            self.status_bar, text="", bg=COLORS["status_bg"],
            fg=COLORS["status_fg"], font=FONTS["status"], anchor="e",
        )
        self.invariant_info_label.pack(side="right", padx=10)

    # ------------------------------------------------------------------
    # Панель задачи (FSM): этап, шаги, кнопки управления
    # ------------------------------------------------------------------

    def _build_task_panel(self, parent: tk.Frame) -> None:
        """Создаёт панель задачи над областью чата (изначально скрыта).

        Состав: название задачи, переключатель режима (auto/manual),
        этап, ожидаемое действие, прогресс-бар по шагам и кнопки
        Пауза / Продолжить / Стоп / Принять / Отклонить / Удалить,
        состояние которых отражает текущую стадию FSM.
        """
        self.task_panel = tk.Frame(
            parent, bg=COLORS["params_bg"], bd=1,
            highlightthickness=1, highlightbackground=COLORS["assistant_border"],
        )
        # Панель НЕ пакуется здесь: показ/скрытие — в _refresh_task_panel.

        # --- Ряд 1: название задачи + режим + раскрытие списка шагов ---
        row1 = tk.Frame(self.task_panel, bg=COLORS["params_bg"])
        row1.pack(fill="x", padx=10, pady=(6, 2))

        self.task_title_label = tk.Label(
            row1, text="", bg=COLORS["params_bg"],
            font=FONTS["header"], fg=COLORS["text"], anchor="w",
        )
        self.task_title_label.pack(side="left")

        tk.Label(
            row1, text="Режим:", bg=COLORS["params_bg"], font=FONTS["small"],
            fg=COLORS["muted"],
        ).pack(side="left", padx=(14, 2))
        self.task_mode_var = tk.StringVar(value=MODE_TITLES[TaskMode.MANUAL])
        self.task_mode_combo = ttk.Combobox(
            row1, textvariable=self.task_mode_var, state="readonly",
            width=9, values=(MODE_TITLES[TaskMode.AUTO],
                             MODE_TITLES[TaskMode.MANUAL]),
        )
        self.task_mode_combo.pack(side="left")
        self.task_mode_combo.bind(
            "<<ComboboxSelected>>", self._on_task_mode_change
        )

        self.task_steps_toggle = self._make_button(
            row1, "Шаги ▸", self._on_toggle_task_steps, small=True
        )
        self.task_steps_toggle.pack(side="right")

        # --- Ряд 2: этап + ожидаемое действие ---
        row2 = tk.Frame(self.task_panel, bg=COLORS["params_bg"])
        row2.pack(fill="x", padx=10, pady=2)
        self.task_stage_label = tk.Label(
            row2, text="", bg=COLORS["params_bg"],
            font=FONTS["small"], fg=COLORS["text"], anchor="w",
        )
        self.task_stage_label.pack(side="left")
        self.task_expected_label = tk.Label(
            row2, text="", bg=COLORS["params_bg"],
            font=FONTS["small"], fg=COLORS["muted"], anchor="w",
        )
        self.task_expected_label.pack(side="left", padx=(16, 0))

        # --- Ряд 3: прогресс-бар по шагам ---
        row3 = tk.Frame(self.task_panel, bg=COLORS["params_bg"])
        row3.pack(fill="x", padx=10, pady=(2, 4))
        self.task_progress = ttk.Progressbar(row3, maximum=1, value=0)
        self.task_progress.pack(side="left", fill="x", expand=True)
        self.task_step_label = tk.Label(
            row3, text="", bg=COLORS["params_bg"], font=FONTS["small"],
            fg=COLORS["text"], anchor="e",
        )
        self.task_step_label.pack(side="left", padx=(10, 0))

        # --- Ряд 4: кнопки управления ---
        row4 = tk.Frame(self.task_panel, bg=COLORS["params_bg"])
        row4.pack(fill="x", padx=10, pady=(0, 6))
        self.task_pause_btn = self._make_button(
            row4, "⏸ Пауза", self._on_task_pause, small=True
        )
        self.task_resume_btn = self._make_button(
            row4, "▶ Продолжить", self._on_task_resume, small=True
        )
        self.task_stop_btn = self._make_button(
            row4, "⏹ Стоп", self._on_task_stop, small=True
        )
        self.task_approve_btn = self._make_button(
            row4, "✓ Принять", self._on_task_approve, small=True
        )
        self.task_reject_btn = self._make_button(
            row4, "✗ Отклонить", self._on_task_reject, small=True
        )
        # Инварианты: модальное окно с активными правилами (global +
        # профиль активной задачи). Доступна всегда при открытой панели.
        self.task_invariants_btn = self._make_button(
            row4, "📋 Инварианты", self._on_show_invariants, small=True
        )
        self.task_delete_btn = self._make_button(
            row4, "✕ Удалить", self._on_task_delete, small=True
        )
        for button in (self.task_pause_btn, self.task_resume_btn,
                       self.task_stop_btn, self.task_approve_btn,
                       self.task_reject_btn, self.task_invariants_btn,
                       self.task_delete_btn):
            button.pack(side="left", padx=(0, 6), pady=2)

        # --- Список шагов (раскрывается кнопкой «Шаги») ---
        self.task_steps_text = tk.Text(
            self.task_panel, height=6, wrap="word", state="disabled",
            bg=COLORS["assistant_bg"], fg=COLORS["text"], font=FONTS["small"],
            relief="flat", padx=8, pady=4,
            highlightthickness=1,
            highlightbackground=COLORS["assistant_border"],
        )
        # Не пакуется: показывается в _on_toggle_task_steps.

    def _on_toggle_task_steps(self) -> None:
        """Раскрывает/скрывает список шагов плана с их статусами."""
        self._task_steps_visible = not self._task_steps_visible
        task = self.agent.active_task
        if self._task_steps_visible and task is not None:
            self.task_steps_text.pack(
                fill="x", padx=10, pady=(0, 6),
                before=self.task_panel.winfo_children()[0],
            )
            self._refresh_task_steps_text(task)
            self.task_steps_toggle.config(text="Шаги ▾")
        else:
            self.task_steps_text.pack_forget()
            self.task_steps_toggle.config(text="Шаги ▸")

    def _refresh_task_steps_text(self, task: Task) -> None:
        """Перерисовывает список шагов (текущий подсвечен, статусы ✓/⏳/✗)."""
        marks = {"done": "✓", "in_progress": "⏳", "rejected": "✗",
                 "failed": "✗", "pending": "·"}
        lines = []
        for step in task.plan:
            mark = marks.get(step.status, "·")
            current = " ←" if step.step_index == task.current_step_index \
                and task.stage in (TaskStage.EXECUTION,
                                   TaskStage.AWAITING_STEP_APPROVAL) else ""
            lines.append(f"{mark} {step.step_index + 1}. {step.description}"
                         f"{current}")
        self.task_steps_text.configure(state="normal")
        self.task_steps_text.delete("1.0", "end")
        self.task_steps_text.insert("1.0", "\n".join(lines) or "Плана ещё нет.")
        self.task_steps_text.configure(state="disabled")

    def _refresh_task_panel(self, task: Optional[Task]) -> None:
        """Перерисовывает панель задачи и статус-бар под состояние FSM.

        Кнопки активны/неактивны строго по матрице переходов:
        - Пауза — в рабочих стадиях (planning/execution/validation);
        - Продолжить — на паузе (и для задачи, прерванной перезапуском);
        - Принять/Отклонить — в стадиях ожидания подтверждения;
        - Стоп — всегда, кроме done/cancelled.
        """
        if not self._widgets_alive():
            return
        if task is None:
            self.task_panel.pack_forget()
            self.task_steps_text.pack_forget()
            self._task_steps_visible = False
            self._set_task_status_text("")
            return

        self.task_title_label.config(
            text=f"ЗАДАЧА: {task.title or task.description}"
        )
        self.task_mode_var.set(MODE_TITLES.get(task.mode, task.mode.value))
        stage_title = STAGE_TITLES.get(task.stage, task.stage.value)
        self.task_stage_label.config(text=f"Этап: {stage_title}")
        self.task_expected_label.config(
            text=f"Ожидается: {task.expected_action or '—'}"
        )

        total = len(task.plan)
        done = task.done_steps_count()
        self.task_progress.configure(
            maximum=max(1, total), value=done,
        )
        if total and task.stage in (TaskStage.EXECUTION,
                                    TaskStage.AWAITING_STEP_APPROVAL,
                                    TaskStage.AWAITING_FINAL_APPROVAL):
            step = task.current_step()
            step_text = f"Шаг {task.current_step_index + 1} из {total}"
            if step is not None:
                step_text += f": {step.description[:60]}"
        elif total:
            step_text = f"Шаги: {done} из {total} выполнено"
        else:
            step_text = ""
        self.task_step_label.config(text=step_text)

        stage = task.stage
        interrupted = "прервана" in (task.expected_action or "")
        self.task_pause_btn.config(
            state="normal" if stage in WORKING_STAGES else "disabled")
        self.task_resume_btn.config(
            state="normal" if (stage == TaskStage.PAUSED
                               or (stage in WORKING_STAGES and interrupted))
            else "disabled")
        self.task_stop_btn.config(
            state="normal" if stage not in (TaskStage.DONE,
                                            TaskStage.CANCELLED)
            else "disabled")
        awaiting = stage in AWAITING_STAGES
        self.task_approve_btn.config(
            state="normal" if awaiting else "disabled")
        self.task_reject_btn.config(
            state="normal" if awaiting else "disabled")
        # Завершённую задачу можно убрать с панели и из БД;
        # активную — только после «Стоп»
        self.task_delete_btn.config(
            state="normal" if task.is_terminal() else "disabled")
        self.task_mode_combo.config(state="readonly")

        # Панель должна идти до области чата (chat — первый ребёнок wrap)
        if not self.task_panel.winfo_ismapped():
            try:
                self.task_panel.pack(
                    fill="x", before=self.chat, pady=(0, 4))
            except tk.TclError:
                self.task_panel.pack(fill="x", pady=(0, 4))
        if self._task_steps_visible:
            self._refresh_task_steps_text(task)

        # Статус-бар: «Задача: execution (2/5)»
        status = f"Задача: {stage.value}"
        if stage in (TaskStage.EXECUTION, TaskStage.AWAITING_STEP_APPROVAL) \
                and total:
            status += f" ({task.current_step_index + 1}/{total})"
        elif stage in (TaskStage.DONE, TaskStage.CANCELLED,
                       TaskStage.FAILED):
            status += " (новая — «/задача …» в чате, убрать — «✕ Удалить»)"
        self._set_task_status_text(status)

    def _set_task_status_text(self, text: str) -> None:
        """Обновляет задачный хвост статус-бара и перерисовывает бар."""
        self._task_status_text = text
        combined = self._status_base_text
        if text:
            combined = f"{combined} · {text}" if combined else text
        if self._widgets_alive():
            self.status_label.config(text=combined)

    def _restore_task_on_startup(self) -> None:
        """Восстанавливает активную задачу после перезапуска приложения.

        Задача не запускается автоматически: если она была прервана в
        рабочей стадии, показываем «Задача прервана. Продолжить?» и ждём
        явного решения пользователя (кнопка «Продолжить»).
        """
        try:
            result = self.agent.restore_task()
        except Exception as exc:  # задачи не должны ломать запуск
            utils.get_logger("deepseek-chat.ui").warning(
                "Не удалось восстановить задачу: %s", exc)
            return
        task = result.get("task")
        if task is None:
            return
        self._refresh_task_panel(task)
        if result.get("interrupted"):
            self._set_status("Задача прервана. Нажмите «Продолжить», "
                             "чтобы возобновить работу.")

    # ---- фоновое выполнение задачных операций ------------------------

    def _run_task_action(self, fn) -> None:
        """Запускает задачную операцию агента в фоновом потоке.

        Пока операция идёт, кнопки панели блокируются (повторные клики
        не должны ломать FSM).
        """
        if self._task_busy:
            return
        self._task_busy = True
        for button in (self.task_pause_btn, self.task_resume_btn,
                       self.task_stop_btn, self.task_approve_btn,
                       self.task_reject_btn, self.task_delete_btn,
                       self.task_mode_combo):
            try:
                button.config(state="disabled")
            except tk.TclError:
                pass
        worker = threading.Thread(
            target=self._task_worker, args=(fn,), daemon=True,
            name="task-fsm",
        )
        worker.start()
        self.root.after(100, self._poll_task_queue)

    def _task_worker(self, fn) -> None:
        """Рабочий поток задачной операции: результат → очередь."""
        try:
            result = fn()
        except Exception as exc:  # агент не должен бросать, но страхуемся
            result = {"task": None, "error": f"Непредвиденная ошибка: {exc}"}
        self._task_queue.put(result)

    def _poll_task_queue(self) -> None:
        """Опрашивает очередь результатов задач (вызывается через after)."""
        try:
            while True:
                result = self._task_queue.get_nowait()
                self._on_task_result(result)
        except queue.Empty:
            pass
        finally:
            if self._task_busy:
                self.root.after(100, self._poll_task_queue)

    def _on_task_result(self, result: Dict[str, Any]) -> None:
        """Обрабатывает результат задачной операции (в главном потоке)."""
        self._task_busy = False
        task = result.get("task")
        self._refresh_task_panel(task)

        error = result.get("error")
        if error:
            short = error if len(error) <= 120 else error[:117] + "..."
            self._set_status(f"Задача: {short}")
            self.root.bell()
            return

        # Очистка: удаление одной задачи или всех задач сразу
        if result.get("deleted"):
            self._set_task_status_text("")
            self._set_status("Задача удалена.")
            return
        if result.get("cleared") is not None:
            self._set_task_status_text("")
            self._set_status(f"Удалено задач: {result['cleared']}.")
            return

        if task is None:
            return
        stage_title = STAGE_TITLES.get(task.stage, task.stage.value)
        self._set_status(f"Задача: {stage_title}.")

        # Завершённая задача дописывает итог в чат как сообщение
        # ассистента (задача начиналась сообщением пользователя)
        if task.stage == TaskStage.DONE and task.result:
            self.agent.note_task_message(
                f"Задача «{task.title or task.description}» завершена.\n\n"
                f"{task.result}",
                role="assistant",
            )
            self._refresh_chat()

        # Авто-продолжение внутри этапа: шаги и проверка цепочкой идут
        # сами в обоих режимах (пользователь подтверждает только переходы
        # между этапами). Один этап за раз — пауза и закрытие окна
        # остаются безопасными между вызовами.
        if result.get("auto"):
            self.root.after(
                120, lambda: self._run_task_action(self.agent.continue_task)
            )

    # ---- обработчики кнопок панели задачи ----------------------------

    def _on_task_pause(self) -> None:
        """Ставит задачу на паузу (текущая стадия сохраняется в БД)."""
        self._run_task_action(self.agent.pause_task)

    def _on_task_resume(self) -> None:
        """Продолжает задачу: после паузы или прерывания перезапуском."""
        task = self.agent.active_task
        if task is not None and task.stage == TaskStage.PAUSED:
            self._run_task_action(self.agent.resume_task)
        else:
            self._run_task_action(self.agent.continue_task)

    def _on_task_stop(self) -> None:
        """Останавливает задачу (с подтверждением)."""
        if not messagebox.askyesno(
                "Остановка задачи",
                "Отменить задачу? Её прогресс останется в журнале, "
                "но выполнение прекратится.",
                parent=self.root):
            return
        self._run_task_action(self.agent.cancel_task)

    def _on_task_approve(self) -> None:
        """«Принять»: подтверждает переход этапа (план или итог).

        Шаги внутри выполнения идут автоматически — пользователь
        подтверждает только переходы между этапами.
        """
        task = self.agent.active_task
        if task is None:
            return
        if task.stage == TaskStage.AWAITING_PLAN_APPROVAL:
            self._run_task_action(self.agent.approve_plan)
        elif task.stage == TaskStage.AWAITING_FINAL_APPROVAL:
            self._run_task_action(self.agent.approve_final)

    def _on_task_reject(self) -> None:
        """«Отклонить»: запрос причины и возврат этапа на переделку."""
        task = self.agent.active_task
        if task is None:
            return
        reason = simpledialog.askstring(
            "Причина отклонения",
            "Укажите, что именно не так (необязательно):",
            parent=self.root,
        )
        if task.stage == TaskStage.AWAITING_PLAN_APPROVAL:
            self._run_task_action(
                lambda: self.agent.reject_plan(reason or ""))
        elif task.stage == TaskStage.AWAITING_FINAL_APPROVAL:
            self._run_task_action(
                lambda: self.agent.reject_final(reason or ""))

    def _on_task_mode_change(self, _event=None) -> None:
        """Смена режима (auto/manual) на лету, прямо из панели задачи."""
        mode = (TaskMode.AUTO
                if self.task_mode_var.get() == MODE_TITLES[TaskMode.AUTO]
                else TaskMode.MANUAL)
        task = self.agent.active_task
        if task is None or task.mode == mode:
            return
        self._run_task_action(lambda: self.agent.set_task_mode(mode))

    def _on_task_delete(self) -> None:
        """«✕ Удалить»: убирает завершённую задачу с панели и из БД.

        Активную задачу сначала нужно остановить («Стоп») — кнопка
        удаления активна только в конечных стадиях (done/failed/cancelled).
        """
        task = self.agent.active_task
        if task is None:
            return
        if not messagebox.askyesno(
                "Удаление задачи",
                f"Удалить задачу «{task.title or task.description}»?\n\n"
                "Шаги и журнал событий будут удалены вместе с ней.",
                parent=self.root):
            return
        self._run_task_action(self.agent.delete_task)

    def _on_clear_tasks(self) -> None:
        """«Очистить задачи»: удаляет ВСЕ задачи после подтверждения.

        Каскадно удаляются шаги и журналы всех задач, включая активную.
        """
        repo = self.agent.task_repository
        count = repo.count_tasks() if repo is not None else 0
        if count <= 0:
            self._set_status("Задач нет — очищать нечего.")
            return
        message = (f"Удалить все задачи ({count})?\n\n"
                   "Шаги и журналы событий будут удалены безвозвратно.")
        if self.agent.active_task is not None:
            message += ("\n\nВнимание: активная задача будет удалена "
                        "вместе с прогрессом.")
        if not messagebox.askyesno("Очистка задач", message,
                                   parent=self.root):
            return
        self._run_task_action(self.agent.clear_tasks)

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

        # Кнопка окна управления контекстом (стратегия, ветка, блок фактов)
        self.context_btn = self._make_button(
            row2, "Контекст…", self._open_context_dialog, small=True
        )
        self.context_btn.pack(side="left", padx=(8, 0))

        # Кнопка окна веток диалога
        self.branches_btn = self._make_button(
            row2, "Ветки…", self._open_branches_dialog, small=True
        )
        self.branches_btn.pack(side="left", padx=(8, 0))

        # Кнопка окна «Память…» (три слоя + pending + предпросмотр запроса)
        self.memory_btn = self._make_button(
            row2, "Память…", self._open_memory_dialog, small=True
        )
        self.memory_btn.pack(side="left", padx=(8, 0))

        # --- Ряд 2б: стратегия управления контекстом ---
        row_strategy = tk.Frame(bar, bg=COLORS["params_bg"])
        row_strategy.pack(fill="x", padx=10, pady=(0, 2))

        tk.Label(
            row_strategy, text="Стратегия:", bg=COLORS["params_bg"],
            font=FONTS["default"],
        ).pack(side="left")

        self.strategy_var = tk.StringVar(value="")
        self.strategy_combo = ttk.Combobox(
            row_strategy,
            textvariable=self.strategy_var,
            values=self._available_strategies(),
            state="readonly",
            width=18,
        )
        self.strategy_combo.pack(side="left", padx=(4, 12))

        self.strategy_info_label = tk.Label(
            row_strategy, text="", bg=COLORS["params_bg"],
            font=FONTS["small"], fg=COLORS["muted"], anchor="w",
        )
        self.strategy_info_label.pack(side="left")

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

        # Факты/окно — отдельная метка справа, рядом со счётчиком
        self.token_summary_label = tk.Label(
            row3, text="Факты: —", bg=COLORS["params_bg"],
            font=FONTS["small"], fg=COLORS["muted"],
        )
        self.token_summary_label.pack(side="right", padx=(0, 12))

        # Память: крат./раб./долг. и бюджет
        self.token_memory_label = tk.Label(
            row3, text="Память: —", bg=COLORS["params_bg"],
            font=FONTS["small"], fg=COLORS["muted"],
        )
        self.token_memory_label.pack(side="right", padx=(0, 12))

        self.token_counter_label = tk.Label(
            row3, text="", bg=COLORS["params_bg"],
            font=FONTS["small"], fg=COLORS["muted"],
        )
        self.token_counter_label.pack(side="right")

    # ------------------------------------------------------------------
    # Персонализация: вкладка «Профиль»
    # ------------------------------------------------------------------

    def _profile_repo(self):
        """Репозиторий профилей агента (None — персонализация выключена)."""
        return getattr(self.agent, "profile_repository", None)

    def _build_profile_tab(self) -> None:
        """Строит вкладку «Профиль»: список профилей и форма редактирования.

        Сверху — выпадающий список профилей и кнопки управления; ниже —
        форма: имя, отображаемое имя, стиль, формат, язык, ограничения
        (многострочно, по одному в строке) и контекст (многострочно,
        строки вида ``ключ: значение``).
        """
        tab = tk.Frame(self.notebook, bg=COLORS["bg"])
        self.notebook.add(tab, text="Профиль")

        # --- Верхняя строка: выбор профиля и кнопки управления ---
        top = tk.Frame(tab, bg=COLORS["bg"])
        top.pack(fill="x", padx=10, pady=(10, 4))

        tk.Label(
            top, text="Профиль:", bg=COLORS["bg"], font=FONTS["default"]
        ).pack(side="left")

        self.profile_var = tk.StringVar()
        self.profile_combo = ttk.Combobox(
            top,
            textvariable=self.profile_var,
            state="readonly",
            width=24,
        )
        self.profile_combo.pack(side="left", padx=(4, 12))
        self.profile_combo.bind(
            "<<ComboboxSelected>>", self._on_profile_form_selected
        )

        self.profile_new_btn = self._make_button(
            top, "Новый", self._on_profile_new, small=True
        )
        self.profile_new_btn.pack(side="left", padx=(0, 6))

        self.profile_save_btn = self._make_button(
            top, "Сохранить", self._on_profile_save, small=True
        )
        self.profile_save_btn.pack(side="left", padx=(0, 6))

        self.profile_delete_btn = self._make_button(
            top, "Удалить", self._on_profile_delete, small=True
        )
        self.profile_delete_btn.pack(side="left", padx=(0, 6))

        self.profile_activate_btn = self._make_button(
            top, "Сделать активным", self._on_profile_activate, small=True
        )
        self.profile_activate_btn.pack(side="left")

        # --- Форма редактирования профиля ---
        form = tk.LabelFrame(
            tab, text="Редактирование профиля", bg=COLORS["bg"],
            font=FONTS["default"], fg=COLORS["text"],
        )
        form.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        def _entry_row(label: str) -> tk.Entry:
            """Строка формы: подпись + однострочное поле."""
            row = tk.Frame(form, bg=COLORS["bg"])
            row.pack(fill="x", padx=8, pady=(6, 0))
            tk.Label(
                row, text=label, bg=COLORS["bg"], font=FONTS["default"],
                width=18, anchor="w",
            ).pack(side="left")
            entry = tk.Entry(
                row, bg=COLORS["input_bg"], font=FONTS["default"],
                relief="solid", bd=1,
            )
            entry.pack(side="left", fill="x", expand=True)
            return entry

        self.profile_name_entry = _entry_row("Имя профиля:")
        self.profile_display_entry = _entry_row("Отображаемое имя:")
        self.profile_style_entry = _entry_row("Стиль:")
        self.profile_format_entry = _entry_row("Формат:")
        self.profile_language_entry = _entry_row("Язык:")

        def _text_row(label: str, height: int) -> tk.Text:
            """Строка формы: подпись + многострочное поле."""
            row = tk.Frame(form, bg=COLORS["bg"])
            row.pack(fill="both", expand=True, padx=8, pady=(6, 0))
            tk.Label(
                row, text=label, bg=COLORS["bg"], font=FONTS["default"],
                width=18, anchor="n",
            ).pack(side="left")
            text = tk.Text(
                row, height=height, wrap="word", bg=COLORS["input_bg"],
                fg=COLORS["text"], font=FONTS["default"], relief="solid",
                bd=1, undo=True,
            )
            text.pack(side="left", fill="both", expand=True)
            return text

        self.profile_constraints_text = _text_row(
            "Ограничения\n(по одному\nв строке):", 4
        )
        self.profile_context_text = _text_row(
            "Контекст\n(ключ: значение):", 4
        )

        self._profile_form_id: Optional[int] = None
        self._refresh_profile_tab()

    # ------------------------------------------------------------------
    # Персонализация: вкладка «Настройки агента»
    # ------------------------------------------------------------------

    def _build_agent_settings_tab(self) -> None:
        """Строит вкладку «Настройки агента», привязанные к активному профилю.

        Модель, temperature, max_tokens и базовый системный промпт
        редактируются здесь и сохраняются в ``profile_agent_settings``.
        """
        tab = tk.Frame(self.notebook, bg=COLORS["bg"])
        self.notebook.add(tab, text="Настройки агента")

        header = tk.Label(
            tab,
            text="Настройки привязаны к активному профилю",
            bg=COLORS["bg"], font=FONTS["header"], fg=COLORS["muted"],
        )
        header.pack(anchor="w", padx=10, pady=(10, 2))

        # --- Модель ---
        model_row = tk.Frame(tab, bg=COLORS["bg"])
        model_row.pack(fill="x", padx=10, pady=(6, 0))
        tk.Label(
            model_row, text="Модель:", bg=COLORS["bg"], font=FONTS["default"]
        ).pack(side="left")
        self.profile_model_var = tk.StringVar()
        self.profile_model_combo = ttk.Combobox(
            model_row,
            textvariable=self.profile_model_var,
            values=list(utils.AVAILABLE_MODELS),
            state="readonly",
            width=24,
        )
        self.profile_model_combo.pack(side="left", padx=(8, 0))

        # --- Температура ---
        temp_row = tk.Frame(tab, bg=COLORS["bg"])
        temp_row.pack(fill="x", padx=10, pady=(8, 0))
        tk.Label(
            temp_row, text="Temperature:", bg=COLORS["bg"],
            font=FONTS["default"],
        ).pack(side="left")
        self.profile_temp_var = tk.DoubleVar(value=0.7)
        self.profile_temp_scale = tk.Scale(
            temp_row,
            from_=TEMP_MIN,
            to=TEMP_MAX,
            resolution=0.1,
            orient="horizontal",
            variable=self.profile_temp_var,
            length=220,
            showvalue=False,
            bg=COLORS["bg"],
            fg=COLORS["text"],
            highlightthickness=0,
            troughcolor="#D0D0D0",
            command=lambda _v: self._update_profile_temp_label(),
        )
        self.profile_temp_scale.pack(side="left", padx=(8, 4))
        self.profile_temp_label = tk.Label(
            temp_row, text="0.7", width=4, bg=COLORS["bg"],
            font=FONTS["default"], fg=COLORS["text"],
        )
        self.profile_temp_label.pack(side="left")

        # --- Max tokens ---
        tokens_row = tk.Frame(tab, bg=COLORS["bg"])
        tokens_row.pack(fill="x", padx=10, pady=(8, 0))
        tk.Label(
            tokens_row, text="Max tokens:", bg=COLORS["bg"],
            font=FONTS["default"],
        ).pack(side="left")
        self.profile_tokens_var = tk.StringVar(value="1000")
        self.profile_tokens_spin = tk.Spinbox(
            tokens_row,
            from_=TOKENS_MIN,
            to=TOKENS_MAX,
            increment=50,
            textvariable=self.profile_tokens_var,
            width=7,
            font=FONTS["default"],
        )
        self.profile_tokens_spin.pack(side="left", padx=(8, 0))

        # --- Базовый системный промпт ---
        prompt_row = tk.Frame(tab, bg=COLORS["bg"])
        prompt_row.pack(fill="both", expand=True, padx=10, pady=(8, 0))
        tk.Label(
            prompt_row, text="Base system prompt:", bg=COLORS["bg"],
            font=FONTS["default"], anchor="n",
        ).pack(side="left", anchor="n")
        self.profile_prompt_text = tk.Text(
            prompt_row, height=8, wrap="word", bg=COLORS["input_bg"],
            fg=COLORS["text"], font=FONTS["default"], relief="solid",
            bd=1, undo=True,
        )
        self.profile_prompt_text.pack(
            side="left", fill="both", expand=True, padx=(8, 0)
        )

        # --- Сохранение ---
        save_row = tk.Frame(tab, bg=COLORS["bg"])
        save_row.pack(fill="x", padx=10, pady=(8, 10))
        self.profile_settings_save_btn = self._make_button(
            save_row, "Сохранить настройки", self._on_profile_settings_save
        )
        self.profile_settings_save_btn.pack(side="left")

        self._refresh_agent_settings_tab()

    # ------------------------------------------------------------------
    # Персонализация: работа с профилями (форма, переключение, статус)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_constraints_text(text: str) -> List[str]:
        """Каждая непустая строка Text-поля формы — отдельное ограничение."""
        return [line.strip() for line in (text or "").splitlines()
                if line.strip()]

    @staticmethod
    def _parse_context_text(text: str) -> Dict[str, Any]:
        """Каждая строка формы вида ``ключ: значение`` — пара контекста.

        Значение пытается приводиться к JSON-типу (числа, списки,
        true/false); не удалось — остаётся строкой. Строки без ``:``
        пропускаются (это ограничения, а не контекст).
        """
        context: Dict[str, Any] = {}
        for line in (text or "").splitlines():
            line = line.strip()
            if not line:
                continue
            key, sep, value = line.partition(":")
            if not sep or not key.strip():
                continue
            try:
                context[key.strip()] = json.loads(value.strip())
            except ValueError:
                context[key.strip()] = value.strip()
        return context

    def _refresh_profile_tab(self, select_id: Optional[int] = None) -> None:
        """Перечитывает профили из БД и заполняет список и форму.

        Args:
            select_id: профиль, который нужно выбрать (после сохранения);
                None — оставить текущий выбор или взять активный профиль.
        """
        repo = self._profile_repo()
        profiles: List[UserProfile] = repo.get_all() if repo is not None else []
        self._profiles_by_name = {p.name: p for p in profiles}
        names = [p.name for p in profiles]
        try:
            self.profile_combo.configure(values=names)
        except tk.TclError:
            return

        if repo is None:
            for widget in (self.profile_new_btn, self.profile_save_btn,
                           self.profile_delete_btn, self.profile_activate_btn):
                widget.configure(state="disabled")
            self.profile_combo.configure(state="disabled")

        chosen: Optional[UserProfile] = None
        wanted = select_id if select_id is not None else self._profile_form_id
        if wanted is not None:
            chosen = next((p for p in profiles if p.id == wanted), None)
        if chosen is None:
            chosen = self.agent.profile or (profiles[0] if profiles else None)
        if chosen is not None:
            self.profile_var.set(chosen.name)
            self._load_profile_into_form(chosen)
        else:
            self.profile_var.set("")
            self._clear_profile_form()

    def _load_profile_into_form(self, profile: UserProfile) -> None:
        """Заполняет форму редактирования данными профиля."""
        self._profile_form_id = profile.id

        def _set_entry(entry: tk.Entry, value: str) -> None:
            entry.delete(0, "end")
            entry.insert(0, value)

        _set_entry(self.profile_name_entry, profile.name)
        _set_entry(self.profile_display_entry, profile.display_name)
        _set_entry(self.profile_style_entry, profile.style)
        _set_entry(self.profile_format_entry, profile.format)
        _set_entry(self.profile_language_entry, profile.language or "русский")

        self.profile_constraints_text.delete("1.0", "end")
        if profile.constraints:
            self.profile_constraints_text.insert(
                "1.0", "\n".join(profile.constraints)
            )
        self.profile_context_text.delete("1.0", "end")
        if profile.context:
            lines = []
            for key, value in profile.context.items():
                if isinstance(value, (list, tuple, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                lines.append(f"{key}: {value}")
            self.profile_context_text.insert("1.0", "\n".join(lines))

    def _clear_profile_form(self) -> None:
        """Очищает форму под создание нового профиля."""
        self._profile_form_id = None
        for entry in (self.profile_name_entry, self.profile_display_entry,
                      self.profile_style_entry, self.profile_format_entry):
            entry.delete(0, "end")
        self.profile_language_entry.delete(0, "end")
        self.profile_language_entry.insert(0, "русский")
        self.profile_constraints_text.delete("1.0", "end")
        self.profile_context_text.delete("1.0", "end")

    def _profile_from_form(self) -> UserProfile:
        """Собирает UserProfile из полей формы (id не проставляется)."""
        return UserProfile(
            name=self.profile_name_entry.get().strip(),
            display_name=self.profile_display_entry.get().strip(),
            style=self.profile_style_entry.get().strip(),
            format=self.profile_format_entry.get().strip(),
            language=self.profile_language_entry.get().strip() or "русский",
            constraints=self._parse_constraints_text(
                self.profile_constraints_text.get("1.0", "end-1c")
            ),
            context=self._parse_context_text(
                self.profile_context_text.get("1.0", "end-1c")
            ),
        )

    def _on_profile_form_selected(self, _event=None) -> None:
        """Выбор профиля в списке — загрузка его в форму редактирования."""
        profile = self._profiles_by_name.get(self.profile_var.get())
        if profile is not None:
            self._load_profile_into_form(profile)

    def _on_profile_new(self, _event=None) -> None:
        """Кнопка «Новый»: очистка формы под создание профиля."""
        self._clear_profile_form()
        self._set_status(
            "Новый профиль: заполните форму и нажмите «Сохранить»."
        )

    def _on_profile_save(self, _event=None) -> None:
        """Кнопка «Сохранить»: создание или обновление профиля из формы.

        После сохранения активного профиля агент перечитывает его и
        перестраивает системный промпт (без очистки истории).
        """
        if self._busy:
            return
        repo = self._profile_repo()
        if repo is None:
            self._set_status("Профили недоступны (нет репозитория).")
            return
        profile = self._profile_from_form()
        if not profile.name:
            self._set_status("Имя профиля не может быть пустым.")
            return
        try:
            if self._profile_form_id is None:
                saved = repo.create(profile)
            else:
                profile.id = self._profile_form_id
                saved = repo.update(profile)
        except ValueError as exc:
            self._set_status(f"Ошибка: {exc}")
            return
        self._refresh_profile_tab(select_id=saved.id)
        # Редактировали активный профиль — перестраиваем промпт в агенте
        active = self.agent.profile
        if active is not None and active.id == saved.id:
            result = self.agent.refresh_profile()
            if result.get("error"):
                self._set_status(f"Ошибка: {result['error']}")
                return
            self._apply_config_to_controls()
        self._update_profile_status()
        self._set_status(f'Профиль "{saved.name}" сохранён.')

    def _on_profile_delete(self, _event=None) -> None:
        """Кнопка «Удалить»: подтверждение и каскадное удаление профиля.

        История чата и настройки профиля удаляются вместе с ним; если
        удалён активный профиль — агент переключается на оставшийся.
        """
        if self._busy:
            return
        repo = self._profile_repo()
        profile_id = self._profile_form_id
        name = self.profile_name_entry.get().strip()
        if repo is None or profile_id is None:
            self._set_status("Сначала выберите профиль в списке.")
            return
        confirmed = messagebox.askyesno(
            "Удалить профиль",
            f'Удалить профиль "{name}"?\n\n'
            "Будут также удалены его настройки и вся история чата "
            "этого профиля.",
            parent=self.root,
        )
        if not confirmed:
            return
        # Если удаляем активный профиль — сначала переключаем агента на
        # другой (иначе агент останется ссылаться на несуществующий id).
        if self.agent.profile is not None and self.agent.profile.id == profile_id:
            other = next(
                (p for p in repo.get_all() if p.id != profile_id), None
            )
            if other is not None:
                self.agent.set_profile(other.id)
        if not repo.delete(profile_id):
            self._set_status(
                "Ошибка: профиль не удалён (последний профиль удалить нельзя)."
            )
            return
        self._refresh_profile_tab()
        self._refresh_agent_settings_tab()
        self._apply_profile_change_ui()
        self._set_status(f'Профиль "{name}" удалён вместе с его историей.')

    def _on_profile_activate(self, _event=None) -> None:
        """Кнопка «Сделать активным»: переключение агента на профиль из формы."""
        if self._busy:
            return
        repo = self._profile_repo()
        profile_id = self._profile_form_id
        if repo is None or profile_id is None:
            self._set_status("Сначала выберите или сохраните профиль.")
            return
        result = self.agent.set_profile(profile_id)
        if result.get("error"):
            self._set_status(f"Ошибка: {result['error']}")
            return
        self._apply_profile_change_ui()
        self._set_status(
            f'Активный профиль: "{result["profile"].name}".'
        )

    def _on_status_profile_switch(self, _event=None) -> None:
        """Переключение активного профиля из статус-бара.

        История чата перезагружается под выбранный профиль (у каждого
        профиля своя), настройки и системный промпт применяются сразу.
        """
        if self._busy:
            self._update_profile_status()
            self._set_status(
                "Дождитесь ответа — переключение профиля временно заблокировано."
            )
            return
        repo = self._profile_repo()
        profile = self._profiles_by_name.get(self.profile_switch_var.get())
        if repo is None or profile is None:
            return
        result = self.agent.set_profile(profile.id)
        if result.get("error"):
            self._set_status(f"Ошибка: {result['error']}")
            self._update_profile_status()
            return
        self._apply_profile_change_ui()
        self._refresh_profile_tab()
        self._refresh_agent_settings_tab()
        self._set_status(f'Профиль переключён: "{profile.name}".')

    def _apply_profile_change_ui(self) -> None:
        """Синхронизирует весь UI после переключения/удаления профиля."""
        self._refresh_chat()
        self._apply_config_to_controls()
        self._update_token_stats()
        self._update_strategy_controls()
        self._refresh_branches_dialog()
        self._refresh_context_dialog()
        self._refresh_memory_dialog()
        self._update_profile_status()
        # Инварианты: счётчики и вкладка следуют за активным профилем
        self._update_invariant_status()
        self._refresh_invariants_tab()

    def _update_profile_status(self) -> None:
        """Показывает в статус-баре активный профиль, модель и temperature."""
        try:
            repo = self._profile_repo()
            names = [p.name for p in repo.get_all()] if repo is not None else []
            self.profile_switch.configure(values=names)
        except (tk.TclError, AttributeError):
            return
        profile = self.agent.profile
        if profile is not None and profile.name in names:
            if self.profile_switch_var.get() != profile.name:
                self.profile_switch_var.set(profile.name)
            self.profile_info_label.config(
                text=(
                    f'Профиль "{profile.name}" · {self.agent.model} · '
                    f"temperature={self.agent.temperature:.1f}"
                )
            )
        else:
            self.profile_switch_var.set("")
            self.profile_info_label.config(text="")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Инварианты: вкладка, формы, модалка в панели задачи
    # ------------------------------------------------------------------

    def _invariant_repo(self):
        """Репозиторий инвариантов агента (None — подсистема выключена)."""
        return getattr(self.agent, "invariant_repository", None)

    def _build_invariants_tab(self) -> None:
        """Строит вкладку «Инварианты»: две панели — категории/инварианты.

        Слева — список категорий с приоритетами и кнопками управления;
        справа — инварианты выбранной категории, действующие для выбранного
        профиля + глобальные (маркеры ``[G]``/``[P]``), с кнопками
        создания/редактирования/удаления.
        """
        tab = tk.Frame(self.notebook, bg=COLORS["bg"])
        self.notebook.add(tab, text="Инварианты")
        self.invariants_tab = tab

        # --- Верхняя строка: целевой профиль + новая категория ---
        top = tk.Frame(tab, bg=COLORS["bg"])
        top.pack(fill="x", padx=10, pady=(8, 4))

        tk.Label(
            top, text="Профиль:", bg=COLORS["bg"], font=FONTS["small"],
            fg=COLORS["muted"],
        ).pack(side="left")
        self.invariants_profile_var = tk.StringVar()
        self.invariants_profile_combo = ttk.Combobox(
            top, textvariable=self.invariants_profile_var,
            state="readonly", width=18,
        )
        self.invariants_profile_combo.pack(side="left", padx=(4, 10))
        self.invariants_profile_combo.bind(
            "<<ComboboxSelected>>", self._on_invariants_profile_change
        )
        self.category_new_btn = self._make_button(
            top, "[+] Новая категория", self._on_new_category, small=True
        )
        self.category_new_btn.pack(side="right")

        # --- Основная область: две панели ---
        main = tk.Frame(tab, bg=COLORS["bg"])
        main.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Левая панель: категории
        left = tk.Frame(main, bg=COLORS["bg"])
        left.pack(side="left", fill="y", padx=(0, 8))
        tk.Label(
            left, text="КАТЕГОРИИ", bg=COLORS["bg"],
            font=FONTS["small"], fg=COLORS["muted"], anchor="w",
        ).pack(fill="x")
        list_wrap = tk.Frame(left, bg=COLORS["bg"])
        list_wrap.pack(fill="y", expand=True)
        self.invariants_category_list = tk.Listbox(
            list_wrap, width=26, exportselection=False,
            bg=COLORS["input_bg"], fg=COLORS["text"], font=FONTS["small"],
            relief="solid", bd=1,
            highlightthickness=1,
            highlightbackground=COLORS["input_border"],
            activestyle="dotbox",
        )
        category_scroll = ttk.Scrollbar(
            list_wrap, orient="vertical",
            command=self.invariants_category_list.yview,
        )
        self.invariants_category_list.configure(
            yscrollcommand=category_scroll.set
        )
        category_scroll.pack(side="right", fill="y")
        self.invariants_category_list.pack(
            side="left", fill="y", expand=True
        )
        self.invariants_category_list.bind(
            "<<ListboxSelect>>", self._on_invariants_category_select
        )
        category_btns = tk.Frame(left, bg=COLORS["bg"])
        category_btns.pack(fill="x", pady=(6, 0))
        self.category_edit_btn = self._make_button(
            category_btns, "Редактировать", self._on_edit_category, small=True
        )
        self.category_edit_btn.pack(side="left", padx=(0, 6))
        self.category_delete_btn = self._make_button(
            category_btns, "Удалить", self._on_delete_category, small=True
        )
        self.category_delete_btn.pack(side="left")

        # Правая панель: инварианты выбранной категории
        right = tk.Frame(main, bg=COLORS["bg"])
        right.pack(side="left", fill="both", expand=True)
        self.invariants_header_label = tk.Label(
            right, text="", bg=COLORS["bg"], font=FONTS["small"],
            fg=COLORS["muted"], anchor="w",
        )
        self.invariants_header_label.pack(fill="x")
        inv_wrap = tk.Frame(right, bg=COLORS["bg"])
        inv_wrap.pack(fill="both", expand=True)
        self.invariants_list = tk.Listbox(
            inv_wrap, exportselection=False,
            bg=COLORS["input_bg"], fg=COLORS["text"], font=FONTS["small"],
            relief="solid", bd=1,
            highlightthickness=1,
            highlightbackground=COLORS["input_border"],
            activestyle="dotbox",
        )
        invariant_scroll = ttk.Scrollbar(
            inv_wrap, orient="vertical", command=self.invariants_list.yview
        )
        self.invariants_list.configure(yscrollcommand=invariant_scroll.set)
        invariant_scroll.pack(side="right", fill="y")
        self.invariants_list.pack(side="left", fill="both", expand=True)
        invariant_btns = tk.Frame(right, bg=COLORS["bg"])
        invariant_btns.pack(fill="x", pady=(6, 0))
        self.invariant_new_btn = self._make_button(
            invariant_btns, "[+] Новый инвариант", self._on_new_invariant,
            small=True,
        )
        self.invariant_new_btn.pack(side="left", padx=(0, 6))
        self.invariant_edit_btn = self._make_button(
            invariant_btns, "Редактировать", self._on_edit_invariant,
            small=True,
        )
        self.invariant_edit_btn.pack(side="left", padx=(0, 6))
        self.invariant_delete_btn = self._make_button(
            invariant_btns, "Удалить", self._on_delete_invariant, small=True
        )
        self.invariant_delete_btn.pack(side="left")

        # Состояние выбора вкладки (категория/инварианты перечитываются)
        self._invariant_categories: List[InvariantCategory] = []
        self._invariants_profiles: Dict[str, UserProfile] = {}
        self._invariants_selected_category_id: Optional[int] = None
        self._invariant_rows: List[Invariant] = []
        self._refresh_invariants_tab()

    def _refresh_invariants_tab(self) -> None:
        """Перечитывает категории/инварианты из БД и перерисовывает вкладку."""
        repo = self._invariant_repo()
        profile_repo = self._profile_repo()
        if repo is None or not repo.enabled:
            self.invariants_category_list.delete(0, "end")
            self.invariants_list.delete(0, "end")
            self.invariants_header_label.config(
                text="Инварианты недоступны (нет хранилища)."
            )
            return

        # Переключатель целевого профиля: имена профилей из репозитория
        profiles = (profile_repo.get_all()
                    if profile_repo is not None else [])
        self._invariants_profiles = {
            profile.name: profile for profile in profiles
        }
        names = list(self._invariants_profiles)
        current = self.invariants_profile_var.get()
        if current not in names:
            active = self.agent.profile
            current = (active.name if active is not None
                       and active.name in names else "")
        self.invariants_profile_combo.configure(values=names)
        self.invariants_profile_var.set(current)

        # Категории по приоритету; выделение сохраняется между обновлениями
        self._invariant_categories = repo.get_all_categories()
        category_list = self.invariants_category_list
        category_list.delete(0, "end")
        selected_id = self._invariants_selected_category_id
        restore_index = 0 if selected_id is None else None
        for index, category in enumerate(self._invariant_categories):
            category_list.insert(
                "end", f"{category.display_name} ({category.priority})"
            )
            if selected_id is not None and category.id == selected_id:
                restore_index = index
        if restore_index is not None and self._invariant_categories:
            category_list.selection_set(restore_index)
        elif not self._invariant_categories:
            self._invariants_selected_category_id = None
        self._refresh_invariants_list()

    def _selected_invariant_category(self) -> Optional[InvariantCategory]:
        """Категория, выделенная в списке (None — выбор не определён)."""
        selection = self.invariants_category_list.curselection()
        if not selection:
            return None
        index = int(selection[0])
        if 0 <= index < len(self._invariant_categories):
            category = self._invariant_categories[index]
            self._invariants_selected_category_id = category.id
            return category
        return None

    def _invariants_target_profile_id(self) -> Optional[int]:
        """Профиль, для которого показываются инварианты (None — без профилей)."""
        profile = self._invariants_profiles.get(
            self.invariants_profile_var.get()
        )
        return profile.id if profile is not None else None

    def _refresh_invariants_list(self) -> None:
        """Перерисовывает список инвариантов выбранной категории."""
        repo = self._invariant_repo()
        if repo is None or not repo.enabled:
            return
        category = self._selected_invariant_category()
        profile_id = self._invariants_target_profile_id()
        profile_name = self.invariants_profile_var.get() or "—"
        if category is not None:
            header = (f"ИНВАРИАНТЫ ({profile_name} + Global) · "
                      f"{category.display_name}")
            invariants = repo.get_invariants_by_category(
                profile_id, category.id
            )
        else:
            header = f"ИНВАРИАНТЫ ({profile_name} + Global)"
            invariants = repo.get_invariants_for_profile(profile_id)
        self.invariants_header_label.config(text=header)

        invariants_list = self.invariants_list
        invariants_list.delete(0, "end")
        self._invariant_rows = list(invariants)
        for invariant in self._invariant_rows:
            marker = "[G]" if invariant.scope == SCOPE_GLOBAL else "[P]"
            details = invariant.description.strip() or invariant.rule.strip()
            if len(details) > 60:
                details = details[:57] + "…"
            invariants_list.insert(
                "end", f"{marker} {invariant.title} — {details}"
            )

    def _on_invariants_category_select(self, _event=None) -> None:
        """Выбор категории в левой панели — обновить список инвариантов."""
        self._refresh_invariants_list()

    def _on_invariants_profile_change(self, _event=None) -> None:
        """Смена целевого профиля — обновить список инвариантов."""
        self._refresh_invariants_list()

    def _selected_invariant(self) -> Optional[Invariant]:
        """Инвариант, выделенный в правой панели (None — ничего не выбрано)."""
        selection = self.invariants_list.curselection()
        if not selection:
            return None
        index = int(selection[0])
        if 0 <= index < len(self._invariant_rows):
            return self._invariant_rows[index]
        return None

    # ------------------------------------------------------------------
    # Инварианты: форма инварианта + LLM-проверка конфликтов
    # ------------------------------------------------------------------

    def _on_new_invariant(self) -> None:
        """«[+] Новый инвариант»: открывает пустую форму."""
        self._open_invariant_dialog(None)

    def _on_edit_invariant(self) -> None:
        """«Редактировать»: форма с данными выделенного инварианта."""
        invariant = self._selected_invariant()
        if invariant is None:
            self._set_status("Выберите инвариант в списке.")
            return
        self._open_invariant_dialog(invariant)

    def _on_delete_invariant(self) -> None:
        """«Удалить»: подтверждение и удаление выделенного инварианта."""
        invariant = self._selected_invariant()
        if invariant is None:
            self._set_status("Выберите инвариант в списке.")
            return
        if not messagebox.askyesno(
            "Удаление инварианта",
            f"Удалить инвариант «{invariant.title}»?",
            parent=self.root,
        ):
            return
        repo = self._invariant_repo()
        if repo is None or not repo.delete_invariant(invariant.id or 0):
            self._set_status("Не удалось удалить инвариант.")
            return
        self._refresh_invariants_tab()
        self._update_invariant_status()
        self._set_status(f"Инвариант удалён: {invariant.title}.")

    def _open_invariant_dialog(self, invariant: Optional[Invariant]) -> None:
        """Открывает модальную форму инварианта (создание/редактирование).

        Поля: категория (Combobox), название (Entry), описание (Text),
        правило для LLM (Text), область действия Global/Profile (Radio),
        профиль (Combobox, активен только для scope='profile').
        При сохранении запускается LLM-проверка конфликтов (не блокирует:
        при конфликте показывается предупреждение с выбором).
        """
        repo = self._invariant_repo()
        if repo is None:
            self._set_status("Инварианты недоступны (нет хранилища).")
            return
        if self._invariant_dialog is not None:
            self._invariant_dialog.lift()
            return

        dialog = tk.Toplevel(self.root)
        self._invariant_dialog = dialog
        dialog.title(
            "Редактирование инварианта" if invariant is not None
            else "Новый инвариант"
        )
        dialog.transient(self.root)
        dialog.configure(bg=COLORS["bg"])
        dialog.resizable(False, False)

        categories = self._invariant_categories or repo.get_all_categories()
        if not categories:
            self._close_invariant_dialog()
            self._set_status("Сначала создайте категорию.")
            return

        row = tk.Frame(dialog, bg=COLORS["bg"])
        row.pack(fill="x", padx=12, pady=(10, 2))
        tk.Label(
            row, text="Категория:", bg=COLORS["bg"], font=FONTS["small"],
            fg=COLORS["muted"],
        ).pack(side="left")
        category_var = tk.StringVar()
        category_combo = ttk.Combobox(
            row, textvariable=category_var, state="readonly", width=28,
            values=[category.display_name for category in categories],
        )
        category_combo.pack(side="left", padx=(4, 0))

        tk.Label(
            dialog, text="Название:", bg=COLORS["bg"], font=FONTS["small"],
            fg=COLORS["muted"],
        ).pack(fill="x", padx=12, pady=(8, 0))
        title_entry = tk.Entry(
            dialog, bg=COLORS["input_bg"], fg=COLORS["text"],
            font=FONTS["default"], relief="solid", bd=1,
            highlightthickness=1,
            highlightbackground=COLORS["input_border"],
        )
        title_entry.pack(fill="x", padx=12)

        tk.Label(
            dialog, text="Описание:", bg=COLORS["bg"], font=FONTS["small"],
            fg=COLORS["muted"],
        ).pack(fill="x", padx=12, pady=(8, 0))
        description_text = tk.Text(
            dialog, height=3, wrap="word", bg=COLORS["input_bg"],
            fg=COLORS["text"], font=FONTS["default"], relief="solid", bd=1,
            highlightthickness=1,
            highlightbackground=COLORS["input_border"],
        )
        description_text.pack(fill="x", padx=12)

        tk.Label(
            dialog, text="Правило для LLM:", bg=COLORS["bg"],
            font=FONTS["small"], fg=COLORS["muted"],
        ).pack(fill="x", padx=12, pady=(8, 0))
        rule_text = tk.Text(
            dialog, height=4, wrap="word", bg=COLORS["input_bg"],
            fg=COLORS["text"], font=FONTS["default"], relief="solid", bd=1,
            highlightthickness=1,
            highlightbackground=COLORS["input_border"],
        )
        rule_text.pack(fill="x", padx=12)

        scope_row = tk.Frame(dialog, bg=COLORS["bg"])
        scope_row.pack(fill="x", padx=12, pady=(10, 0))
        scope_var = tk.StringVar(value=SCOPE_PROFILE)
        tk.Radiobutton(
            scope_row, text="Глобальный (для всех профилей)",
            variable=scope_var, value=SCOPE_GLOBAL, bg=COLORS["bg"],
            font=FONTS["small"], command=lambda: _toggle_scope(),
        ).pack(side="left")
        tk.Radiobutton(
            scope_row, text="Профильный",
            variable=scope_var, value=SCOPE_PROFILE, bg=COLORS["bg"],
            font=FONTS["small"], command=lambda: _toggle_scope(),
        ).pack(side="left", padx=(10, 0))

        profile_row = tk.Frame(dialog, bg=COLORS["bg"])
        profile_row.pack(fill="x", padx=12, pady=(6, 0))
        tk.Label(
            profile_row, text="Профиль:", bg=COLORS["bg"],
            font=FONTS["small"], fg=COLORS["muted"],
        ).pack(side="left")
        profile_var = tk.StringVar()
        profile_combo = ttk.Combobox(
            profile_row, textvariable=profile_var, state="readonly",
            width=22, values=list(self._invariants_profiles),
        )
        profile_combo.pack(side="left", padx=(4, 0))
        if self.invariants_profile_var.get() in self._invariants_profiles:
            profile_var.set(self.invariants_profile_var.get())

        def _toggle_scope() -> None:
            """Профильный Combobox активен только для scope='profile'."""
            if scope_var.get() == SCOPE_PROFILE:
                profile_combo.configure(state="readonly")
            else:
                profile_combo.configure(state="disabled")

        buttons = tk.Frame(dialog, bg=COLORS["bg"])
        buttons.pack(fill="x", padx=12, pady=(12, 10))
        save_btn = self._make_button(buttons, "Сохранить",
                                     self._on_invariant_dialog_save)
        save_btn.pack(side="right")
        cancel_btn = self._make_button(buttons, "Отмена",
                                       self._close_invariant_dialog)
        cancel_btn.pack(side="right", padx=(0, 8))

        dialog.invariant_id = invariant.id if invariant is not None else None
        dialog.invariant_category_combo = category_combo  # type: ignore
        dialog.invariant_category_var = category_var  # type: ignore
        dialog.invariant_title_entry = title_entry  # type: ignore
        dialog.invariant_description_text = description_text  # type: ignore
        dialog.invariant_rule_text = rule_text  # type: ignore
        dialog.invariant_scope_var = scope_var  # type: ignore
        dialog.invariant_profile_var = profile_var  # type: ignore
        dialog.invariant_save_btn = save_btn  # type: ignore
        dialog.invariant_categories = categories  # type: ignore

        if invariant is not None:
            for index, category in enumerate(categories):
                if category.id == invariant.category_id:
                    category_var.set(category.display_name)
                    break
            title_entry.insert(0, invariant.title)
            description_text.insert("1.0", invariant.description)
            rule_text.insert("1.0", invariant.rule)
            scope_var.set(invariant.scope)
            if invariant.profile_id is not None:
                for name, profile in self._invariants_profiles.items():
                    if profile.id == invariant.profile_id:
                        profile_var.set(name)
                        break
        elif categories:
            category_var.set(categories[0].display_name)
        _toggle_scope()

        dialog.protocol("WM_DELETE_WINDOW", self._close_invariant_dialog)
        dialog.grab_set()
        title_entry.focus_set()

    def _close_invariant_dialog(self) -> None:
        """Закрывает форму инварианта и сбрасывает ожидающее сохранение."""
        self._pending_invariant_save = None
        dialog = self._invariant_dialog
        self._invariant_dialog = None
        if dialog is not None:
            try:
                dialog.grab_release()
                dialog.destroy()
            except tk.TclError:
                pass

    def _invariant_from_dialog(self) -> Optional[Invariant]:
        """Собирает Invariant из формы (None — ошибка в статус-баре)."""
        dialog = self._invariant_dialog
        if dialog is None:
            return None
        categories: List[InvariantCategory] = dialog.invariant_categories
        category_name = dialog.invariant_category_var.get()
        category = next(
            (item for item in categories
             if item.display_name == category_name),
            None,
        )
        title = dialog.invariant_title_entry.get().strip()
        description = dialog.invariant_description_text.get(
            "1.0", "end"
        ).strip()
        rule = dialog.invariant_rule_text.get("1.0", "end").strip()
        scope = dialog.invariant_scope_var.get()
        if category is None:
            self._set_status("Выберите категорию инварианта.")
            return None
        if not title:
            self._set_status("Укажите название инварианта.")
            return None
        if not rule:
            self._set_status("Укажите правило для LLM.")
            return None
        profile_id: Optional[int] = None
        if scope == SCOPE_PROFILE:
            profile = self._invariants_profiles.get(
                dialog.invariant_profile_var.get()
            )
            if profile is None:
                self._set_status("Выберите профиль для профильного инварианта.")
                return None
            profile_id = profile.id
        return Invariant(
            id=dialog.invariant_id,
            category_id=int(category.id or 0),
            title=title,
            description=description,
            rule=rule,
            scope=scope,
            profile_id=profile_id,
        )

    def _on_invariant_dialog_save(self) -> None:
        """«Сохранить»: собирает данные и запускает LLM-проверку конфликтов.

        Проверка идёт в фоновом потоке (_run_action): при найденных
        конфликтах показывается предупреждение с выбором, иначе инвариант
        сохраняется сразу. Проверка НЕ блокирует сохранение.
        """
        dialog = self._invariant_dialog
        repo = self._invariant_repo()
        if dialog is None or repo is None:
            return
        invariant = self._invariant_from_dialog()
        if invariant is None:
            return
        dialog.invariant_save_btn.configure(state="disabled")

        target_profile_id = (invariant.profile_id
                             if invariant.scope == SCOPE_PROFILE else None)
        existing = [
            item for item in repo.get_invariants_for_profile(target_profile_id)
            if item.id != invariant.id
        ]
        categories = repo.get_all_categories()
        self._pending_invariant_save = {
            "invariant": invariant,
            "existing": existing,
            "categories": categories,
        }
        self._set_status("Проверка конфликтов инварианта…")
        self._run_action(
            lambda: self.agent.check_invariant_conflicts(
                invariant, existing, categories
            ),
            self._on_invariant_conflict_check_done,
        )

    def _on_invariant_conflict_check_done(self, conflicts: Any) -> None:
        """Результат LLM-проверки конфликтов (вызывается в главном потоке).

        Ошибка фоновой операции (``{"error": ...}`` от _run_action) не
        блокирует сохранение: LLM-проверка — только предупреждение.
        """
        if self._invariant_dialog is None:
            return  # форма закрыта, пока шла проверка
        if isinstance(conflicts, dict) and conflicts.get("error"):
            self._set_status(
                "Проверка конфликтов не удалась — сохраняю без проверки."
            )
            self._finish_invariant_save()
            return
        if conflicts:
            self._show_invariant_conflict_dialog(list(conflicts))
        else:
            self._finish_invariant_save()

    def _show_invariant_conflict_dialog(self, conflicts: list) -> None:
        """Предупреждение о конфликте: «[Всё равно сохранить] [Отменить]»."""
        dialog = self._invariant_dialog
        if dialog is None:
            return
        lines = ["⚠️ Обнаружен конфликт:"]
        categories = {
            int(category.id): category.display_name
            for category in self._invariant_categories
        }
        repo = self._invariant_repo()
        for conflict in conflicts[:5]:
            existing = None
            if repo is not None and conflict.invariant_id is not None:
                existing = repo.get_invariant_by_id(conflict.invariant_id)
            if existing is not None:
                category_name = categories.get(
                    int(existing.category_id), ""
                )
                lines.append(
                    f"   С инвариантом «{existing.title}» ({category_name})"
                )
            else:
                lines.append(f"   С инвариантом id={conflict.invariant_id}")
            if conflict.reason:
                lines.append(f"   Причина: {conflict.reason}")
        lines.append("")
        lines.append("Сохранить всё равно? (правила не блокируют сохранение)")

        window = tk.Toplevel(self.root)
        self._invariant_conflict_dialog = window
        window.title("Конфликт инвариантов")
        window.transient(dialog)
        window.configure(bg=COLORS["bg"])
        window.resizable(False, False)
        text = tk.Label(
            window, text="\n".join(lines), bg=COLORS["bg"],
            fg=COLORS["text"], font=FONTS["small"], justify="left",
            anchor="w",
        )
        text.pack(fill="both", expand=True, padx=14, pady=(12, 4))
        buttons = tk.Frame(window, bg=COLORS["bg"])
        buttons.pack(fill="x", padx=14, pady=(4, 12))
        self._make_button(
            buttons, "Всё равно сохранить",
            self._on_invariant_conflict_save_anyway,
        ).pack(side="right")
        self._make_button(
            buttons, "Отменить", self._on_invariant_conflict_cancel
        ).pack(side="right", padx=(0, 8))
        window.protocol(
            "WM_DELETE_WINDOW", self._on_invariant_conflict_cancel
        )
        window.grab_set()

    def _on_invariant_conflict_save_anyway(self) -> None:
        """«Всё равно сохранить»: конфликт не блокирует сохранение."""
        self._close_invariant_conflict_dialog()
        self._finish_invariant_save()

    def _on_invariant_conflict_cancel(self) -> None:
        """«Отменить»: конфликт найден, сохранение отложено (форма открыта)."""
        self._close_invariant_conflict_dialog()
        self._pending_invariant_save = None
        dialog = self._invariant_dialog
        if dialog is not None:
            dialog.invariant_save_btn.configure(state="normal")
        self._set_status("Сохранение инварианта отменено (конфликт).")

    def _close_invariant_conflict_dialog(self) -> None:
        """Закрывает окно предупреждения о конфликте."""
        window = self._invariant_conflict_dialog
        self._invariant_conflict_dialog = None
        if window is not None:
            try:
                window.grab_release()
                window.destroy()
            except tk.TclError:
                pass

    def _finish_invariant_save(self) -> None:
        """Записывает инвариант в БД и обновляет UI (главный поток)."""
        data = self._pending_invariant_save
        self._pending_invariant_save = None
        repo = self._invariant_repo()
        if data is None or repo is None:
            self._close_invariant_dialog()
            return
        invariant: Invariant = data["invariant"]
        try:
            if invariant.id:
                repo.update_invariant(invariant)
                message = f"Инвариант обновлён: {invariant.title}."
            else:
                repo.create_invariant(invariant)
                message = f"Инвариант сохранён: {invariant.title}."
        except ValueError as exc:
            self._set_status(f"Ошибка: {exc}")
            self._close_invariant_dialog()
            return
        self._close_invariant_dialog()
        self._refresh_invariants_tab()
        self._update_invariant_status()
        self._set_status(message)

    # ------------------------------------------------------------------
    # Инварианты: форма категории
    # ------------------------------------------------------------------

    def _on_new_category(self) -> None:
        """«[+] Новая категория»: пустая форма категории."""
        self._open_category_dialog(None)

    def _on_edit_category(self) -> None:
        """«Редактировать» (категория): форма с данными выделенной."""
        category = self._selected_invariant_category()
        if category is None:
            self._set_status("Выберите категорию в списке.")
            return
        self._open_category_dialog(category)

    def _on_delete_category(self) -> None:
        """«Удалить» (категория): системные категории удалить нельзя."""
        category = self._selected_invariant_category()
        if category is None:
            self._set_status("Выберите категорию в списке.")
            return
        if category.is_system:
            self._set_status(
                f"Категорию «{category.display_name}» удалить нельзя "
                "(системная)."
            )
            return
        if not messagebox.askyesno(
            "Удаление категории",
            f"Удалить категорию «{category.display_name}» "
            "вместе с её инвариантами?",
            parent=self.root,
        ):
            return
        repo = self._invariant_repo()
        if repo is None or not repo.delete_category(category.id or 0):
            self._set_status("Не удалось удалить категорию.")
            return
        self._invariants_selected_category_id = None
        self._refresh_invariants_tab()
        self._update_invariant_status()
        self._set_status(f"Категория удалена: {category.display_name}.")

    def _open_category_dialog(self, category: Optional[InvariantCategory]) -> None:
        """Открывает модальную форму категории (name/display_name/desc/priority)."""
        if self._category_dialog is not None:
            self._category_dialog.lift()
            return
        dialog = tk.Toplevel(self.root)
        self._category_dialog = dialog
        dialog.title(
            "Редактирование категории" if category is not None
            else "Новая категория"
        )
        dialog.transient(self.root)
        dialog.configure(bg=COLORS["bg"])
        dialog.resizable(False, False)

        tk.Label(
            dialog, text="Имя (латиница, уникальное):", bg=COLORS["bg"],
            font=FONTS["small"], fg=COLORS["muted"],
        ).pack(fill="x", padx=12, pady=(10, 0))
        name_entry = tk.Entry(
            dialog, bg=COLORS["input_bg"], fg=COLORS["text"],
            font=FONTS["default"], relief="solid", bd=1,
            highlightthickness=1,
            highlightbackground=COLORS["input_border"],
        )
        name_entry.pack(fill="x", padx=12)

        tk.Label(
            dialog, text="Отображаемое имя:", bg=COLORS["bg"],
            font=FONTS["small"], fg=COLORS["muted"],
        ).pack(fill="x", padx=12, pady=(8, 0))
        display_entry = tk.Entry(
            dialog, bg=COLORS["input_bg"], fg=COLORS["text"],
            font=FONTS["default"], relief="solid", bd=1,
            highlightthickness=1,
            highlightbackground=COLORS["input_border"],
        )
        display_entry.pack(fill="x", padx=12)

        tk.Label(
            dialog, text="Описание:", bg=COLORS["bg"], font=FONTS["small"],
            fg=COLORS["muted"],
        ).pack(fill="x", padx=12, pady=(8, 0))
        description_entry = tk.Entry(
            dialog, bg=COLORS["input_bg"], fg=COLORS["text"],
            font=FONTS["default"], relief="solid", bd=1,
            highlightthickness=1,
            highlightbackground=COLORS["input_border"],
        )
        description_entry.pack(fill="x", padx=12)

        priority_row = tk.Frame(dialog, bg=COLORS["bg"])
        priority_row.pack(fill="x", padx=12, pady=(8, 0))
        tk.Label(
            priority_row, text="Приоритет (меньше = важнее):",
            bg=COLORS["bg"], font=FONTS["small"], fg=COLORS["muted"],
        ).pack(side="left")
        priority_var = tk.StringVar(value="100")
        priority_spin = ttk.Spinbox(
            priority_row, from_=1, to=1000, width=6,
            textvariable=priority_var,
        )
        priority_spin.pack(side="left", padx=(8, 0))

        buttons = tk.Frame(dialog, bg=COLORS["bg"])
        buttons.pack(fill="x", padx=12, pady=(12, 10))
        self._make_button(
            buttons, "Сохранить", self._on_category_dialog_save
        ).pack(side="right")
        self._make_button(
            buttons, "Отмена", self._close_category_dialog
        ).pack(side="right", padx=(0, 8))

        dialog.category_name_entry = name_entry  # type: ignore
        dialog.category_display_entry = display_entry  # type: ignore
        dialog.category_description_entry = description_entry  # type: ignore
        dialog.category_priority_var = priority_var  # type: ignore
        dialog.category_id = category.id if category is not None else None

        if category is not None:
            name_entry.insert(0, category.name)
            display_entry.insert(0, category.display_name)
            description_entry.insert(0, category.description)
            priority_var.set(str(category.priority))

        dialog.protocol("WM_DELETE_WINDOW", self._close_category_dialog)
        dialog.grab_set()
        name_entry.focus_set()

    def _close_category_dialog(self) -> None:
        """Закрывает форму категории."""
        dialog = self._category_dialog
        self._category_dialog = None
        if dialog is not None:
            try:
                dialog.grab_release()
                dialog.destroy()
            except tk.TclError:
                pass

    def _on_category_dialog_save(self) -> None:
        """«Сохранить» (категория): валидация и запись в БД."""
        dialog = self._category_dialog
        repo = self._invariant_repo()
        if dialog is None or repo is None:
            return
        name = dialog.category_name_entry.get().strip()
        display_name = dialog.category_display_entry.get().strip()
        description = dialog.category_description_entry.get().strip()
        try:
            priority = int(dialog.category_priority_var.get())
        except (TypeError, ValueError):
            priority = 100
        if not name:
            self._set_status("Укажите имя категории (латиница).")
            return
        if not display_name:
            self._set_status("Укажите отображаемое имя категории.")
            return
        category = InvariantCategory(
            id=dialog.category_id,
            name=name,
            display_name=display_name,
            description=description,
            priority=priority,
            is_system=False,
        )
        try:
            if category.id:
                repo.update_category(category)
                message = f"Категория обновлена: {category.display_name}."
            else:
                repo.create_category(category)
                message = f"Категория создана: {category.display_name}."
        except ValueError as exc:
            self._set_status(f"Ошибка: {exc}")
            return
        self._close_category_dialog()
        self._refresh_invariants_tab()
        self._update_invariant_status()
        self._set_status(message)

    # ------------------------------------------------------------------
    # Инварианты: модалка «Показать инварианты» в панели задачи
    # ------------------------------------------------------------------

    def _on_show_invariants(self) -> None:
        """«📋 Инварианты»: модальное окно с активными правилами.

        Показываются инварианты активного профиля + глобальные,
        сгруппированные по категориям (маркеры ``[G]``/``[P]``).
        «Перейти к редактированию» открывает вкладку «Инварианты».
        """
        if self._invariants_window is not None:
            self._invariants_window.lift()
            return
        repo = self._invariant_repo()
        if repo is None or not repo.enabled:
            self._set_status("Инварианты недоступны (нет хранилища).")
            return

        profile = self.agent.profile
        profile_id = profile.id if profile is not None else None
        profile_name = profile.name if profile is not None else "—"
        invariants = repo.get_invariants_for_profile(profile_id)
        categories = repo.get_all_categories()

        window = tk.Toplevel(self.root)
        self._invariants_window = window
        window.title("Активные инварианты")
        window.transient(self.root)
        window.configure(bg=COLORS["bg"])
        header = tk.Label(
            window,
            text=f"Активные инварианты ({profile_name} + Global)",
            bg=COLORS["bg"], font=FONTS["header"], fg=COLORS["text"],
            anchor="w",
        )
        header.pack(fill="x", padx=14, pady=(10, 4))

        body = tk.Text(
            window, width=62, height=16, wrap="word", state="disabled",
            bg=COLORS["assistant_bg"], fg=COLORS["text"],
            font=FONTS["small"], relief="flat", padx=10, pady=8,
            highlightthickness=1,
            highlightbackground=COLORS["assistant_border"],
        )
        body.pack(fill="both", expand=True, padx=14)

        # Группировка по категориям в порядке приоритета (важные выше)
        by_id = {int(c.id): c for c in categories if c.id is not None}
        grouped: Dict[int, List[Invariant]] = {}
        for invariant in invariants:
            grouped.setdefault(int(invariant.category_id), []).append(
                invariant
            )
        ordered = sorted(
            grouped,
            key=lambda cid: (by_id[cid].priority if cid in by_id else 10_000,
                             cid),
        )
        lines: List[str] = []
        for category_id in ordered:
            category = by_id.get(category_id)
            display = (category.display_name if category is not None
                       else "Без категории")
            lines.append(display)
            for invariant in grouped[category_id]:
                marker = "[G]" if invariant.scope == SCOPE_GLOBAL else "[P]"
                lines.append(f"  {marker} {invariant.title}")
            lines.append("")
        body.configure(state="normal")
        body.delete("1.0", "end")
        body.insert(
            "1.0", "\n".join(lines).strip()
            or "Инвариантов нет — добавьте их на вкладке «Инварианты»."
        )
        body.configure(state="disabled")

        buttons = tk.Frame(window, bg=COLORS["bg"])
        buttons.pack(fill="x", padx=14, pady=(4, 12))
        self._make_button(
            buttons, "Перейти к редактированию",
            self._on_invariants_goto_edit,
        ).pack(side="right")
        self._make_button(
            buttons, "Закрыть", self._close_invariants_window
        ).pack(side="right", padx=(0, 8))
        window.protocol("WM_DELETE_WINDOW", self._close_invariants_window)
        window.grab_set()

    def _on_invariants_goto_edit(self) -> None:
        """«Перейти к редактированию»: закрыть модалку, открыть вкладку."""
        self._close_invariants_window()
        try:
            self.notebook.select(self.invariants_tab)
        except tk.TclError:
            pass

    def _close_invariants_window(self) -> None:
        """Закрывает модальное окно активных инвариантов."""
        window = self._invariants_window
        self._invariants_window = None
        if window is not None:
            try:
                window.grab_release()
                window.destroy()
            except tk.TclError:
                pass

    def _update_invariant_status(self) -> None:
        """Показывает в статус-баре: Инвариантов: N (глобальных: X, профильных: Y)."""
        repo = self._invariant_repo()
        if repo is None or not repo.enabled:
            self.invariant_info_label.config(text="")
            return
        profile = self.agent.profile
        profile_id = profile.id if profile is not None else None
        try:
            counts = repo.count_invariants(profile_id)
        except Exception:
            self.invariant_info_label.config(text="")
            return
        self.invariant_info_label.config(text=(
            f"Инвариантов: {counts['total']} "
            f"(глобальных: {counts['global']}, "
            f"профильных: {counts['profile']})"
        ))

    # Персонализация: вкладка «Настройки агента» (обработчики)
    # ------------------------------------------------------------------

    def _refresh_agent_settings_tab(self) -> None:
        """Загружает настройки АКТИВНОГО профиля в виджеты вкладки."""
        settings = self.agent.profile_settings
        if settings is None:
            for widget in (self.profile_model_combo, self.profile_temp_scale,
                           self.profile_tokens_spin, self.profile_prompt_text,
                           self.profile_settings_save_btn):
                try:
                    widget.configure(state="disabled")
                except tk.TclError:
                    pass
            return
        self.profile_model_var.set(settings.model)
        self.profile_temp_var.set(float(settings.temperature))
        self._update_profile_temp_label()
        self.profile_tokens_var.set(str(settings.max_tokens))
        self.profile_prompt_text.delete("1.0", "end")
        self.profile_prompt_text.insert("1.0", settings.base_system_prompt or "")

    def _update_profile_temp_label(self) -> None:
        """Обновляет подпись значения temperature на вкладке настроек."""
        try:
            value = round(float(self.profile_temp_var.get()), 1)
        except (tk.TclError, TypeError, ValueError):
            return
        self.profile_temp_label.config(text=f"{value:.1f}")

    def _on_profile_settings_save(self, _event=None) -> None:
        """Сохраняет настройки активного профиля и применяет их к агенту.

        Новый temperature/max_tokens/модель действуют со следующего
        запроса без перезапуска приложения.
        """
        if self._busy:
            return
        repo = self._profile_repo()
        profile = self.agent.profile
        if repo is None or profile is None:
            self._set_status("Профили недоступны (нет репозитория).")
            return
        try:
            tokens = int(self.profile_tokens_var.get().strip())
            tokens = max(TOKENS_MIN, min(TOKENS_MAX, tokens))
            temperature = round(float(self.profile_temp_var.get()), 1)
        except (TypeError, ValueError):
            self._set_status("Ошибка: неверное значение max tokens.")
            return
        settings = ProfileAgentSettings(
            profile_id=profile.id,
            model=self.profile_model_var.get(),
            temperature=temperature,
            max_tokens=tokens,
            base_system_prompt=self.profile_prompt_text.get(
                "1.0", "end-1c"
            ).strip(),
        )
        if not repo.update_settings(settings):
            self._set_status("Ошибка: не удалось сохранить настройки профиля.")
            return
        result = self.agent.refresh_profile()
        if result.get("error"):
            self._set_status(f"Ошибка: {result['error']}")
            return
        self._apply_config_to_controls()
        self._update_profile_status()
        self._set_status(
            f'Настройки профиля "{profile.name}" сохранены: '
            f"{settings.model}, temperature={settings.temperature:.1f}, "
            f"max_tokens={settings.max_tokens}."
        )

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
        # Стратегия управления контекстом
        self.strategy_combo.bind(
            "<<ComboboxSelected>>", self._on_strategy_change
        )
        # Правая кнопка мыши в области чата — «Создать ветку отсюда…»
        self.chat.bind("<Button-3>", self._show_chat_menu)
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

        Системный промпт в чате не показывается. Каждое сообщение получает
        тег ``msg_<индекс>``, где индекс считается без учёта system
        (первый показанный пузырь — ``msg_0``): по этому тегу правая кнопка
        мыши определяет, от какого сообщения создавать ветку.
        """
        self.chat.configure(state="normal")
        dialog_index = 0
        try:
            self.chat.delete("1.0", "end")
            for message in self.agent.history:
                role = message.get("role")
                if role in ("user", "assistant"):
                    self._insert_message(
                        role, message.get("content", ""), index=dialog_index
                    )
                    dialog_index += 1
        finally:
            self.chat.configure(state="disabled")
        self._chat_message_count = dialog_index
        if (self._chat_selected_index is not None
                and self._chat_selected_index >= dialog_index):
            self._chat_selected_index = None
        self._scroll_to_end()

    def _insert_message(self, role: str, content: str,
                        index: Optional[int] = None) -> None:
        """Вставляет одно сообщение (пузырь) в область чата.

        Args:
            role: "user" или "assistant".
            content: текст сообщения.
            index: индекс сообщения в диалоге без учёта system; если задан,
                к сообщению добавляется тег ``msg_<index>`` (для веток) и
                строка кнопок памяти. Если None — индекс берётся из
                ``self._chat_message_count`` (сообщение, вставляемое до
                перерисовки чата).
        """
        base_tag = "user" if role == "user" else "assistant"
        name = "Вы" if role == "user" else "Ассистент"
        timestamp = utils.format_time()
        effective_index = index if index is not None else self._chat_message_count
        message_tag = f"msg_{effective_index}"

        def tags(*extra: str) -> tuple:
            """Дополняет набор тегов тегом сообщения (если он есть)."""
            return extra + (message_tag,)

        header = f"[{timestamp}] {name}:"
        # Очень длинные сообщения рисуем частично — иначе перерисовка
        # чата заметно подтормаживает (в поле ввода ограничений нет).
        truncated = len(content) > _CHAT_DISPLAY_MAX_CHARS
        display_text = content[:_CHAT_DISPLAY_MAX_CHARS] if truncated else content
        paragraphs = display_text.split("\n") or [""]

        self.chat.insert("end", "\n", tags())  # отступ между сообщениями

        # Заголовок: время и роль
        self.chat.insert("end", header + "\n", tags(base_tag, f"{base_tag}_hdr"))
        # Тело сообщения
        for paragraph in paragraphs:
            self.chat.insert("end", paragraph + "\n", tags(base_tag))
        if truncated:
            self.chat.insert(
                "end",
                f"… сообщение показано частично "
                f"({len(content)} символов, лимит отображения "
                f"{_CHAT_DISPLAY_MAX_CHARS})\n",
                tags(base_tag),
            )

        # Строка кнопок памяти (не храним ссылки: чат перерисовывается целиком)
        self._insert_memory_buttons(effective_index, role)

    def _message_saved_state(self, index: int) -> Dict[str, bool]:
        """Состояние «сохранено в слой» для сообщения (источник истины — агент).

        Returns:
            ``{"short": pinned, "working": bool, "long": bool}``.
        """
        state = {"short": False, "working": False, "long": False}
        try:
            pinned = {int(p["index"]) for p in self.agent.list_pinned()}
            state["short"] = int(index) in pinned
        except Exception:
            pass
        try:
            working = self.agent.get_working_memory()
            for item in (working.get("items") or {}).values():
                if item.get("source_message") == int(index):
                    state["working"] = True
                    break
        except Exception:
            pass
        try:
            for item in self.agent.get_long_memory():
                if item.get("source_message") == int(index):
                    state["long"] = True
                    break
        except Exception:
            pass
        return state

    def _insert_memory_buttons(self, index: int, role: str) -> None:
        """Вставляет строку кнопок сохранения в память под сообщением.

        Кнопки — встроенные окна (``window_create``). Вставка выполняется при
        ``state="normal"`` (чат переключается вызывающим кодом), поэтому
        клики по кнопкам работают и при заблокированном тексте.
        """
        state = self._message_saved_state(index)
        row = tk.Frame(self.chat, bg=COLORS["bg"])

        short_label = "📌 Открепить" if state["short"] else "💬 В диалог"
        btn_short = tk.Button(
            row, text=short_label, font=FONTS["small"], relief="flat",
            takefocus=0, cursor="hand2", padx=4, pady=0,
            bg=COLORS["bg"], fg=COLORS["button"], activebackground="#E8E8E8",
            command=lambda i=index: self._on_save_short(i),
        )
        btn_short.pack(side="left", padx=(0, 4))

        working_label = "✓ В рабочую" if state["working"] else "📋 В рабочую"
        btn_working = tk.Button(
            row, text=working_label, font=FONTS["small"], relief="flat",
            takefocus=0, cursor="hand2", padx=4, pady=0,
            bg=COLORS["bg"], fg=COLORS["button"], activebackground="#E8E8E8",
            command=lambda i=index: self._on_save_working(i),
        )
        btn_working.pack(side="left", padx=(0, 4))

        long_label = "✓ В долг." if state["long"] else "🗄 В долговременную"
        btn_long = tk.Button(
            row, text=long_label, font=FONTS["small"], relief="flat",
            takefocus=0, cursor="hand2", padx=4, pady=0,
            bg=COLORS["bg"], fg=COLORS["button"], activebackground="#E8E8E8",
            command=lambda i=index: self._on_save_long(i),
        )
        btn_long.pack(side="left")

        self.chat.window_create("end", window=row)
        self.chat.insert("end", "\n", ("msg_buttons",))

    def _dialog_message(self, index: int) -> Optional[Dict[str, Any]]:
        """Сообщение диалога по индексу (без system)."""
        history = self.agent.history[1:]
        if 0 <= int(index) < len(history):
            message = history[int(index)]
            return {"role": message.get("role"), "content": message.get("content", "")}
        return None

    def _on_save_short(self, index: int) -> None:
        """Кнопка «💬 В диалог»: закрепить/открепить (краткосрочная память)."""
        try:
            pinned = {int(p["index"]) for p in self.agent.list_pinned()}
        except Exception:
            pinned = set()
        if int(index) in pinned:
            result = self.agent.unpin_message(int(index))
            status = "Откреплено." if result.get("ok") else result.get("error")
        else:
            result = self.agent.pin_message(int(index))
            status = "Закреплено (краткосрочная память)." if result.get("ok") \
                else result.get("error")
        self._refresh_chat()
        self._update_token_stats()
        self._refresh_memory_dialog()
        self._set_status(status)

    def _on_save_working(self, index: int) -> None:
        """Кнопка «📋 В рабочую»: окно сохранения в рабочую память."""
        self._open_save_dialog(index, "working")

    def _on_save_long(self, index: int) -> None:
        """Кнопка «🗄 В долговременную»: окно сохранения в долговременную."""
        self._open_save_dialog(index, "long_term")

    def _scroll_to_end(self) -> None:
        """Прокручивает область чата вниз."""
        self.chat.see("end")
        self.root.update_idletasks()

    # ------------------------------------------------------------------
    # Статус-бар
    # ------------------------------------------------------------------

    def _set_status(self, text: str) -> None:
        """Показывает текст статуса слева в статус-баре.

        К основному тексту добавляется краткая информация о задаче
        (подсистема FSM): «… · Задача: execution (2/5)».
        """
        self._status_base_text = text
        combined = text
        if self._task_status_text:
            combined = (f"{text} · {self._task_status_text}"
                        if text else self._task_status_text)
        self.status_label.config(text=combined)

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

        # Стратегия с фактами: «Факты: N · ≈X токенов», скользящее окно:
        # «Окно: N сообщений». Ключи stats["facts"] / stats["strategy"].
        strategy = stats.get("strategy") or {}
        facts = stats.get("facts") or {}
        if facts.get("active"):
            facts_count = int(facts.get("count") or 0)
            facts_tokens = int(facts.get("tokens") or 0)
            self.token_summary_label.config(
                text=f"Факты: {facts_count} · ≈{facts_tokens} токенов"
            )
        else:
            window = strategy.get("window")
            if window is None:
                self.token_summary_label.config(text="Окно: —")
            else:
                self.token_summary_label.config(
                    text=f"Окно: {int(window)} сообщений"
                )

        # Метка стратегии и выпадающий список всегда синхронны с агентом
        self._update_strategy_controls()

        # Память: крат./раб./долг. и бюджет
        memory = stats.get("memory") or {}
        try:
            if memory.get("enabled"):
                short = memory.get("short_term") or {}
                working = memory.get("working") or {}
                long_term = memory.get("long_term") or {}
                self.token_memory_label.config(
                    text=(
                        f"Память: крат. {int(short.get('pinned') or 0)} · "
                        f"раб. {int(working.get('items') or 0)} · "
                        f"долг. {int(long_term.get('items') or 0)} "
                        f"(≈{int(memory.get('total_tokens') or 0)} ток., "
                        f"бюджет ≈{int(memory.get('budget') or 0)})"
                    )
                )
            else:
                self.token_memory_label.config(text="Память: выкл.")
        except tk.TclError:
            pass

    def _token_context_text(self) -> str:
        """Строка о контексте для статус-бара.

        Формат: ``Ветка: main · Стратегия: sliding_window ·
        ctx: ≈120 (8 сообщ. отправлено)`` (плюс память при включении).
        """
        try:
            stats = self.agent.get_token_stats()
        except Exception:
            return "ctx: —"
        context = stats.get("context") or {}
        branch = stats.get("branch") or {}
        strategy = stats.get("strategy") or {}
        memory = stats.get("memory") or {}
        tokens = context.get("tokens")
        sent = int(context.get("messages_sent") or 0)
        if tokens is None:
            ctx = "ctx: —"
        else:
            ctx = f"ctx: ≈{tokens} ({sent} сообщ. отправлено)"
        branch_name = branch.get("name") or "—"
        strategy_name = strategy.get("name") or "—"
        parts = [f"Ветка: {branch_name}", f"Стратегия: {strategy_name}"]
        if memory.get("enabled"):
            short = memory.get("short_term") or {}
            working = memory.get("working") or {}
            long_term = memory.get("long_term") or {}
            parts.append(
                f"Память: {int(short.get('pinned') or 0)}/"
                f"{int(working.get('items') or 0)}/"
                f"{int(long_term.get('items') or 0)}"
            )
        parts.append(ctx)
        return " · ".join(parts)

    # ------------------------------------------------------------------
    # Стратегии управления контекстом
    # ------------------------------------------------------------------

    def _available_strategies(self) -> list:
        """Список стратегий агента (с безопасным запасным значением)."""
        try:
            names = list(self.agent.available_strategies())
        except Exception:
            names = []
        return names or ["sliding_window", "sticky_facts", "branching"]

    def _facts_limit(self) -> int:
        """Максимум фактов из настроек стратегии (для подписи)."""
        try:
            settings = self.agent.strategy_settings
            return int(settings.get("facts_max_items", 50) or 50)
        except Exception:
            return 50

    def _strategy_info_text(self, info: Dict[str, Any]) -> str:
        """Подпись текущего окна/лимитов рядом с выпадающим списком."""
        name = info.get("name")
        window = info.get("window")
        facts_count = int(info.get("facts_count") or 0)
        if name == "branching":
            window_text = "окно: —" if window is None else f"окно: {window}"
            return f"Ветки · {window_text}"
        if name == "sticky_facts":
            window_text = "Окно: —" if window is None else f"Окно: {window}"
            return f"{window_text} · фактов: {facts_count}/{self._facts_limit()}"
        if name == "sliding_window":
            if window is None:
                return "Окно: —"
            return f"Окно: {window} сообщ."
        return ""

    def _update_strategy_controls(self) -> None:
        """Синхронизирует список стратегий и подпись с состоянием агента."""
        try:
            info = self.agent.get_strategy()
        except Exception:
            return
        # Legacy-режим (агент без стратегии) показываем как скользящее окно:
        # пустой выпадающий список выглядел бы как сломанный UI.
        name = info.get("name") or (self._available_strategies() or [""])[0]
        display_info = dict(info)
        display_info["name"] = name
        try:
            self.strategy_combo.configure(values=self._available_strategies())
            if name and self.strategy_var.get() != name:
                self.strategy_var.set(name)
            self.strategy_info_label.config(
                text=self._strategy_info_text(display_info)
            )
        except tk.TclError:
            return

    def _on_strategy_change(self, _event=None) -> None:
        """Применяет выбранную стратегию управления контекстом сразу."""
        try:
            name = (self.strategy_var.get() or "").strip()
        except tk.TclError:
            return
        if not name:
            return
        if self._busy:
            self._set_status("Дождитесь ответа на текущий запрос.")
            self._update_strategy_controls()
            return
        try:
            info = self.agent.set_strategy(name)
        except utils.AgentConfigError as exc:
            self._set_status(f"Ошибка: {exc}")
            self._update_strategy_controls()
            return
        except Exception as exc:  # страховка: агент не должен бросать
            self._set_status(f"Ошибка: {exc}")
            self._update_strategy_controls()
            return

        self._update_strategy_controls()
        self._update_token_stats()
        self._refresh_context_dialog()
        self._refresh_branches_dialog()
        self._set_status(f"Стратегия: {info.get('name') or name}")
        self._set_info(self._token_context_text())

    # ------------------------------------------------------------------
    # Окно «Контекст»: просмотр, обновление и очистка конспекта
    # ------------------------------------------------------------------

    def _open_context_dialog(self) -> None:
        """Открывает окно «Контекст».

        Окно работает в двух режимах:
        - legacy-суммаризация (стратегия не задана): текст конспекта;
        - стратегии контекста: стратегия/окно/ветка и блок фактов.
        Режим определяется в :meth:`_context_mode`.
        """
        window = self._context_window
        if window is not None and window.winfo_exists():
            self._refresh_context_dialog()
            window.lift()
            window.focus_set()
            return

        window = tk.Toplevel(self.root)
        self._context_window = window
        window.title("Контекст и стратегия")
        window.geometry("720x540")
        window.minsize(520, 380)
        window.configure(bg=COLORS["bg"])
        window.transient(self.root)
        window.protocol("WM_DELETE_WINDOW", self._close_context_dialog)

        self.context_title_label = tk.Label(
            window, text="Стратегия управления контекстом",
            bg=COLORS["bg"], font=FONTS["header"], fg=COLORS["text"],
        )
        self.context_title_label.pack(anchor="w", padx=10, pady=(10, 4))

        self.context_info_label = tk.Label(
            window, text="", bg=COLORS["bg"], font=FONTS["small"],
            fg=COLORS["muted"], justify="left", anchor="w",
        )
        self.context_info_label.pack(fill="x", padx=10, pady=(0, 6))

        # Блок фактов (виден только в режиме стратегии с фактами)
        self._context_facts_heading = tk.Label(
            window, text="Факты диалога", bg=COLORS["bg"],
            font=FONTS["header"], fg=COLORS["text"],
        )
        self._context_facts_heading.pack(anchor="w", padx=10)

        self._context_tree_wrap = tk.Frame(window, bg=COLORS["bg"])
        self._context_tree_wrap.pack(fill="both", expand=True, padx=10,
                                     pady=(2, 0))

        self.context_facts_tree = ttk.Treeview(
            self._context_tree_wrap,
            columns=("category", "key", "value"),
            show="headings",
            height=8,
            selectmode="browse",
        )
        for column, title, width in (
            ("category", "Категория", 130),
            ("key", "Ключ", 150),
            ("value", "Значение", 380),
        ):
            self.context_facts_tree.heading(column, text=title)
            self.context_facts_tree.column(column, width=width, anchor="w")
        tree_scroll = ttk.Scrollbar(
            self._context_tree_wrap, orient="vertical",
            command=self.context_facts_tree.yview,
        )
        self.context_facts_tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.pack(side="right", fill="y")
        self.context_facts_tree.pack(side="left", fill="both", expand=True)

        self._context_text_heading = tk.Label(
            window, text="Факты в том виде, в каком они уходят в запрос",
            bg=COLORS["bg"], font=FONTS["small"], fg=COLORS["muted"],
        )
        self._context_text_heading.pack(anchor="w", padx=10, pady=(8, 2))

        self._context_text_wrap = tk.Frame(window, bg=COLORS["bg"])
        self._context_text_wrap.pack(fill="both", expand=True, padx=10)

        self.context_text = tk.Text(
            self._context_text_wrap,
            wrap="word",
            height=6,
            bg=COLORS["input_bg"],
            fg=COLORS["text"],
            font=FONTS["default"],
            relief="solid",
            bd=1,
            highlightthickness=1,
            highlightbackground=COLORS["input_border"],
            cursor="arrow",
        )
        scrollbar = ttk.Scrollbar(
            self._context_text_wrap, orient="vertical",
            command=self.context_text.yview,
        )
        self.context_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.context_text.pack(side="left", fill="both", expand=True)

        buttons = tk.Frame(window, bg=COLORS["bg"])
        buttons.pack(fill="x", padx=10, pady=(8, 10))

        self.context_refresh_btn = self._make_button(
            buttons, "Обновить факты", self._on_context_refresh
        )
        self.context_refresh_btn.pack(side="left")

        self.context_clear_btn = self._make_button(
            buttons, "Очистить факты", self._on_context_clear
        )
        self.context_clear_btn.pack(side="left", padx=(8, 0))

        self._make_button(buttons, "Закрыть", self._close_context_dialog).pack(
            side="right"
        )

        self._refresh_context_dialog()

    def _context_mode(self) -> str:
        """Режим окна «Контекст»: ``legacy``, ``facts`` или ``strategy``."""
        try:
            strategy = self.agent.get_strategy()
        except Exception:
            return "legacy"
        if not strategy.get("name"):
            # Стратегия не задана — работает прежняя суммаризация конспекта
            return "legacy"
        return "facts" if strategy.get("facts") else "strategy"

    def _pack_context_facts(self, show: bool) -> None:
        """Показывает/скрывает блок фактов, сохраняя порядок виджетов."""
        for widget in (self._context_facts_heading, self._context_tree_wrap):
            try:
                widget.pack_forget()
            except tk.TclError:
                pass
        if not show:
            return
        try:
            self._context_facts_heading.pack(
                anchor="w", padx=10, before=self._context_text_heading
            )
            self._context_tree_wrap.pack(
                fill="both", expand=True, padx=10, pady=(2, 0),
                before=self._context_text_heading,
            )
        except tk.TclError:
            pass

    def _refresh_context_dialog(self) -> None:
        """Перерисовывает окно «Контекст» по текущему состоянию агента.

        В legacy-режиме (стратегия не задана) окно ведёт себя как раньше:
        показывает текст конспекта. В режиме стратегий — стратегию, размер
        окна, активную ветку и блок фактов (для стратегии без фактов —
        понятную надпись).
        """
        window = self._context_window
        if window is None or not window.winfo_exists():
            return

        mode = self._context_mode()
        if mode == "legacy":
            self._refresh_context_summary()
        else:
            self._refresh_context_facts(mode)

    def _refresh_context_summary(self) -> None:
        """Legacy-режим: текст конспекта и состояние сжатия."""
        try:
            summary = self.agent.get_summary()
        except Exception as exc:
            summary = {"active": False, "text": "", "enabled": False,
                       "error": str(exc)}

        text = str(summary.get("text") or "")
        self._pack_context_facts(False)
        try:
            self.context_title_label.config(
                text="Конспект предыдущей части диалога"
            )
            self._context_text_heading.config(text="Текст конспекта")
            self.context_text.configure(height=14)
        except tk.TclError:
            pass

        self.context_text.configure(state="normal")
        self.context_text.delete("1.0", "end")
        if text:
            self.context_text.insert("1.0", text)
        else:
            self.context_text.insert(
                "1.0",
                "Конспекта пока нет.\n\n"
                "Конспект создаётся автоматически, когда история выходит за "
                "окно последних сообщений (см. SUMMARY_KEEP_LAST_N / "
                "SUMMARY_EVERY_N в .env). Кнопка «Обновить» сожмёт историю "
                "принудительно.",
            )
        self.context_text.configure(state="disabled")

        info = (
            f"Покрыто сообщений: {summary.get('covered', 0)} · "
            f"токены конспекта: ≈{summary.get('tokens', 0)} · "
            f"обновлён: {summary.get('updated_at') or '—'} · "
            f"модель: {summary.get('model') or '—'} · "
            f"сжатие: {'включено' if summary.get('enabled') else 'выключено'}"
        )
        if summary.get("error"):
            info += f"\nОшибка сжатия: {summary['error']}"
        self.context_info_label.config(text=info)

        try:
            self.context_refresh_btn.config(
                text="Обновить",
                state="disabled" if self._context_busy else "normal",
            )
            self.context_clear_btn.config(
                text="Очистить",
                state="normal" if (text and summary.get("enabled"))
                else "disabled",
            )
        except tk.TclError:
            pass

    def _refresh_context_facts(self, mode: str) -> None:
        """Режим стратегий: стратегия, окно, ветка и блок фактов."""
        try:
            strategy = self.agent.get_strategy()
        except Exception as exc:
            strategy = {"error": str(exc)}
        try:
            facts = self.agent.get_facts()
        except Exception as exc:
            facts = {"items": [], "text": "", "count": 0, "tokens": 0,
                     "active": False, "enabled": False, "error": str(exc),
                     "updated_at": None}

        facts_active = mode == "facts"
        count = int(facts.get("count") or 0)
        tokens = int(facts.get("tokens") or 0)
        branch = strategy.get("branch") or {}
        name = strategy.get("name") or "—"
        sub_strategy = strategy.get("sub_strategy")
        window_size = strategy.get("window")

        lines = []
        if sub_strategy and sub_strategy != name:
            lines.append(f"Стратегия: {name} · подстратегия ветки: {sub_strategy}")
        else:
            lines.append(f"Стратегия: {name}")
        if window_size is not None:
            lines.append(f"Окно последних сообщений: {window_size}")
        lines.append(
            f"Ветка: {branch.get('name') or '—'} (id={branch.get('id', '—')}) · "
            f"сообщений: {branch.get('messages', 0)} · "
            f"веток: {branch.get('branches_count', 0)}"
        )
        if facts_active:
            lines.append(
                f"Фактов: {count} · ≈{tokens} токенов · "
                f"обновлено: {facts.get('updated_at') or '—'} · "
                f"{'включены' if facts.get('enabled') else 'выключены'}"
            )
        else:
            lines.append("Факты не используются в этой стратегии.")
        error = strategy.get("error") or facts.get("error")
        if error:
            lines.append(f"Ошибка: {error}")
        self.context_info_label.config(text="\n".join(lines))

        self._pack_context_facts(facts_active)
        try:
            self.context_title_label.config(
                text="Стратегия управления контекстом"
            )
        except tk.TclError:
            pass

        # Таблица фактов: категория / ключ / значение
        try:
            self.context_facts_tree.delete(
                *self.context_facts_tree.get_children()
            )
            for item in facts.get("items") or []:
                self.context_facts_tree.insert(
                    "", "end",
                    values=(item.get("category", ""), item.get("key", ""),
                            item.get("value", "")),
                )
        except tk.TclError:
            pass

        # Текстовый блок фактов (как они попадают в запрос)
        text = str(facts.get("text") or "").strip()
        try:
            self._context_text_heading.config(
                text="Факты в том виде, в каком они уходят в запрос"
            )
            self.context_text.configure(height=6)
        except tk.TclError:
            pass
        self.context_text.configure(state="normal")
        self.context_text.delete("1.0", "end")
        if not facts_active:
            self.context_text.insert(
                "1.0",
                "Факты не используются в этой стратегии.\n\n"
                "Блок фактов доступен для стратегий «sticky_facts» и "
                "«branching» (с подстратегией sticky_facts). Переключите "
                "стратегию в выпадающем списке на панели параметров.",
            )
        elif text:
            self.context_text.insert("1.0", text)
        else:
            self.context_text.insert(
                "1.0",
                "Фактов пока нет.\n\n"
                "Нажмите «Обновить факты», чтобы извлечь их из текущего "
                "диалога.",
            )
        self.context_text.configure(state="disabled")

        # Кнопки: обновлять/очищать имеет смысл только при активных фактах
        try:
            self.context_refresh_btn.config(
                text="Обновить факты",
                state="normal" if (facts_active and not self._context_busy)
                else "disabled",
            )
            self.context_clear_btn.config(
                text="Очистить факты",
                state="normal" if (facts_active and count > 0
                                   and not self._context_busy)
                else "disabled",
            )
        except tk.TclError:
            pass

    def _on_context_refresh(self) -> None:
        """Обновляет контекст в фоновом потоке (UI не «зависает»).

        В legacy-режиме принудительно сжимает историю в конспект
        (``summarize_now``), в режиме стратегий — обновляет факты
        (``update_facts_now``).
        """
        if self._context_busy:
            return
        if self._busy:
            self._set_status("Дождитесь ответа на текущий запрос.")
            return

        mode = self._context_mode()
        self._context_pending_mode = mode
        self._context_busy = True
        try:
            self.context_refresh_btn.configure(state="disabled")
        except tk.TclError:
            pass
        self._set_status(
            "Сжимаю историю…" if mode == "legacy" else "Обновляю факты…"
        )
        self._refresh_context_dialog()

        def worker() -> None:
            try:
                if mode == "legacy":
                    result = self.agent.summarize_now(force=True)
                else:
                    result = self.agent.update_facts_now(force=True)
            except Exception as exc:  # агент не должен бросать, но страхуемся
                result = {"updated": False, "count": 0,
                          "error": f"Непредвиденная ошибка: {exc}"}
            self._summary_queue.put(result)

        threading.Thread(
            target=worker, daemon=True, name="context-refresh"
        ).start()
        self.root.after(100, self._poll_summary_queue)

    def _poll_summary_queue(self) -> None:
        """Опрашивает очередь результата обновления контекста (root.after)."""
        try:
            while True:
                result = self._summary_queue.get_nowait()
                self._on_summary_result(result)
        except queue.Empty:
            pass
        finally:
            if self._context_busy:
                self.root.after(100, self._poll_summary_queue)

    def _on_summary_result(self, result: Dict[str, Any]) -> None:
        """Обрабатывает результат обновления контекста в главном потоке."""
        mode = getattr(self, "_context_pending_mode", "facts")
        self._context_busy = False
        self._context_pending_mode = None
        self._update_token_stats()
        self._refresh_context_dialog()
        self._refresh_branches_dialog()
        error = result.get("error")
        if mode == "legacy":
            if error:
                self._set_status(f"Сжатие не удалось: {error}")
            elif result.get("updated"):
                self._set_status(
                    f"Конспект обновлён: покрыто "
                    f"{result.get('covered', 0)} сообщений."
                )
            else:
                self._set_status("Сжимать нечего: история ещё короткая.")
            return
        if error:
            self._set_status(f"Факты: {error}")
        elif result.get("updated"):
            self._set_status(
                f"Факты обновлены: {int(result.get('count') or 0)}."
            )
        else:
            self._set_status("Факты не изменились.")

    def _on_context_clear(self) -> None:
        """Очищает конспект (legacy) или блок фактов (стратегии)."""
        parent = self._context_window or self.root
        legacy = self._context_mode() == "legacy"
        if legacy:
            title = "Очистить конспект"
            question = (
                "Удалить конспект диалога?\n\n"
                "Сама история сообщений сохранится."
            )
        else:
            title = "Очистить факты"
            question = (
                "Удалить все факты диалога?\n\n"
                "Сама история сообщений сохранится."
            )
        if not messagebox.askyesno(title, question, parent=parent):
            return
        if legacy:
            self.agent.clear_summary()
        else:
            self.agent.clear_facts()
        self._refresh_context_dialog()
        self._update_token_stats()
        self._refresh_branches_dialog()
        self._set_status("Конспект очищен." if legacy else "Факты очищены.")

    def _close_context_dialog(self) -> None:
        """Закрывает окно «Контекст»."""
        window = self._context_window
        self._context_window = None
        if window is not None:
            try:
                window.destroy()
            except tk.TclError:
                pass

    # ------------------------------------------------------------------
    # Ветки: контекстное меню чата и создание ветки от сообщения
    # ------------------------------------------------------------------

    def _message_index_at(self, index: str) -> Optional[int]:
        """Индекс сообщения диалога по позиции в области чата.

        Args:
            index: индекс Tk (например, "5.3"), полученный через
                ``self.chat.index("@x,y")``.

        Returns:
            Индекс сообщения без учёта system или None, если под позицией
            сообщения нет.
        """
        if not index:
            return None
        try:
            names = self.chat.tag_names(index)
        except tk.TclError:
            return None
        for tag in names:
            if isinstance(tag, str) and tag.startswith("msg_"):
                try:
                    return int(tag[4:])
                except ValueError:
                    continue
        return None

    def _selected_chat_index(self) -> Optional[int]:
        """Индекс выбранного в чате сообщения (None — сообщение не выбрано)."""
        index = self._chat_selected_index
        if index is None:
            return None
        if index < 0 or index >= self._chat_message_count:
            return None
        return index

    def _show_chat_menu(self, event) -> None:
        """Показывает контекстное меню чата по правой кнопке мыши."""
        try:
            index = self.chat.index("@%d,%d" % (event.x, event.y))
        except tk.TclError:
            index = None
        message_index = self._message_index_at(index)
        self._chat_selected_index = message_index
        try:
            enabled = "normal" if message_index is not None else "disabled"
            for position in (0, 1, 2, 3, 4):
                self.chat_menu.entryconfigure(position, state=enabled)
        except tk.TclError:
            pass
        try:
            self.chat_menu.tk_popup(event.x_root, event.y_root)
        finally:
            try:
                self.chat_menu.grab_release()
            except tk.TclError:
                pass

    def _on_create_branch_from_chat(self) -> None:
        """Создаёт ветку от сообщения, выбранного правой кнопкой в чате."""
        index = self._selected_chat_index()
        if index is None:
            self._set_status(
                "Ветку можно создать только от сообщения: щёлкните по нему "
                "правой кнопкой мыши."
            )
            return
        self._create_branch(from_index=index)

    def _on_menu_save_short(self) -> None:
        """Меню: закрепить/открепить выбранное сообщение."""
        index = self._selected_chat_index()
        if index is None:
            self._set_status("Выберите сообщение правой кнопкой мыши.")
            return
        self._on_save_short(index)

    def _on_menu_save_working(self) -> None:
        """Меню: сохранить выбранное сообщение в рабочую память."""
        index = self._selected_chat_index()
        if index is None:
            self._set_status("Выберите сообщение правой кнопкой мыши.")
            return
        self._on_save_working(index)

    def _on_menu_save_long(self) -> None:
        """Меню: сохранить выбранное сообщение в долговременную память."""
        index = self._selected_chat_index()
        if index is None:
            self._set_status("Выберите сообщение правой кнопкой мыши.")
            return
        self._on_save_long(index)

    def _on_menu_show_saved(self) -> None:
        """Меню: показать, что сохранено об этом сообщении."""
        index = self._selected_chat_index()
        if index is None:
            self._set_status("Выберите сообщение правой кнопкой мыши.")
            return
        state = self._message_saved_state(index)
        lines = [f"Сообщение №{index + 1}:"]
        lines.append(
            "Краткосрочная: " + ("закреплено" if state["short"] else "нет")
        )
        lines.append(
            "Рабочая: " + ("сохранено" if state["working"] else "нет")
        )
        lines.append(
            "Долговременная: " + ("сохранено" if state["long"] else "нет")
        )
        messagebox.showinfo("Сохранено об этом сообщении", "\n".join(lines),
                            parent=self.root)

    # ------------------------------------------------------------------
    # Окно «Ветки» диалога
    # ------------------------------------------------------------------

    def _open_branches_dialog(self) -> None:
        """Открывает окно со списком веток диалога."""
        window = self._branches_window
        if window is not None and window.winfo_exists():
            self._refresh_branches_dialog()
            window.lift()
            window.focus_set()
            return

        window = tk.Toplevel(self.root)
        self._branches_window = window
        window.title("Ветки диалога")
        window.geometry("780x430")
        window.minsize(560, 320)
        window.configure(bg=COLORS["bg"])
        window.transient(self.root)
        window.protocol("WM_DELETE_WINDOW", self._close_branches_dialog)

        tk.Label(
            window, text="Ветки диалога", bg=COLORS["bg"],
            font=FONTS["header"], fg=COLORS["text"],
        ).pack(anchor="w", padx=10, pady=(10, 4))

        self.branches_info_label = tk.Label(
            window, text="", bg=COLORS["bg"], font=FONTS["small"],
            fg=COLORS["muted"], justify="left", anchor="w",
        )
        self.branches_info_label.pack(fill="x", padx=10, pady=(0, 6))

        tree_wrap = tk.Frame(window, bg=COLORS["bg"])
        tree_wrap.pack(fill="both", expand=True, padx=10)

        self.branches_tree = ttk.Treeview(
            tree_wrap,
            columns=("name", "strategy", "messages", "facts",
                     "created", "parent"),
            show="headings",
            selectmode="browse",
        )
        for column, title, width, anchor in (
            ("name", "Имя", 190, "w"),
            ("strategy", "Стратегия", 120, "w"),
            ("messages", "Сообщений", 90, "center"),
            ("facts", "Фактов", 70, "center"),
            ("created", "Создана", 150, "w"),
            ("parent", "Родитель", 120, "w"),
        ):
            self.branches_tree.heading(column, text=title)
            self.branches_tree.column(column, width=width, anchor=anchor)
        # Активная ветка выделяется фоном
        self.branches_tree.tag_configure("active", background=COLORS["user_bg"])
        tree_scroll = ttk.Scrollbar(
            tree_wrap, orient="vertical", command=self.branches_tree.yview
        )
        self.branches_tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.pack(side="right", fill="y")
        self.branches_tree.pack(side="left", fill="both", expand=True)
        self.branches_tree.bind(
            "<Double-1>", lambda _event: self._on_branches_switch()
        )
        self.branches_tree.bind(
            "<<TreeviewSelect>>",
            lambda _event: self._update_branches_buttons_state(),
        )

        buttons_top = tk.Frame(window, bg=COLORS["bg"])
        buttons_top.pack(fill="x", padx=10, pady=(8, 4))

        self.branches_create_btn = self._make_button(
            buttons_top, "Создать от конца", self._on_branches_checkpoint,
            small=True,
        )
        self.branches_create_btn.pack(side="left")

        self.branches_create_msg_btn = self._make_button(
            buttons_top, "Создать от выбранного сообщения",
            self._on_branches_create_from_message, small=True,
        )
        self.branches_create_msg_btn.pack(side="left", padx=(8, 0))

        buttons_bottom = tk.Frame(window, bg=COLORS["bg"])
        buttons_bottom.pack(fill="x", padx=10, pady=(0, 10))

        self.branches_switch_btn = self._make_button(
            buttons_bottom, "Переключиться", self._on_branches_switch, small=True
        )
        self.branches_switch_btn.pack(side="left")

        self.branches_rename_btn = self._make_button(
            buttons_bottom, "Переименовать", self._on_branches_rename, small=True
        )
        self.branches_rename_btn.pack(side="left", padx=(8, 0))

        self.branches_delete_btn = self._make_button(
            buttons_bottom, "Удалить", self._on_branches_delete, small=True
        )
        self.branches_delete_btn.pack(side="left", padx=(8, 0))

        self._make_button(
            buttons_bottom, "Закрыть", self._close_branches_dialog, small=True
        ).pack(side="right")

        self._refresh_branches_dialog()

    def _selected_branch_id(self) -> Optional[int]:
        """Идентификатор выбранной в таблице ветки (None — нет выбора)."""
        window = self._branches_window
        if window is None:
            return None
        try:
            selection = self.branches_tree.selection()
        except (tk.TclError, AttributeError):
            return None
        if not selection:
            return None
        try:
            return int(selection[0])
        except (TypeError, ValueError):
            return None

    def _branch_name(self, branch_id: int) -> str:
        """Имя ветки по её идентификатору (для диалогов подтверждения)."""
        try:
            for branch in self.agent.list_branches():
                if int(branch.get("id") or 0) == int(branch_id):
                    return str(branch.get("name") or f"ветка-{branch_id}")
        except Exception:
            pass
        return f"ветка-{branch_id}"

    def _refresh_branches_dialog(self) -> None:
        """Перерисовывает таблицу веток и состояние кнопок."""
        window = self._branches_window
        if window is None or not window.winfo_exists():
            return
        try:
            branches = list(self.agent.list_branches())
        except Exception:
            branches = []

        names = {}
        for branch in branches:
            try:
                names[int(branch.get("id") or 0)] = str(
                    branch.get("name") or ""
                )
            except (TypeError, ValueError):
                continue

        selected = self._selected_branch_id()
        tree = self.branches_tree
        try:
            tree.delete(*tree.get_children())
        except tk.TclError:
            return

        active_id: Optional[int] = None
        for branch in branches:
            try:
                branch_id = int(branch.get("id") or 0)
            except (TypeError, ValueError):
                continue
            is_active = bool(branch.get("active"))
            if is_active:
                active_id = branch_id
            parent_id = branch.get("parent_id")
            parent_name = "—"
            if parent_id is not None:
                try:
                    parent_name = names.get(int(parent_id)) or f"id={parent_id}"
                except (TypeError, ValueError):
                    parent_name = str(parent_id)
            try:
                tree.insert(
                    "", "end", iid=str(branch_id),
                    values=(
                        branch.get("name") or f"ветка-{branch_id}",
                        branch.get("strategy") or "—",
                        int(branch.get("messages") or 0),
                        int(branch.get("facts_count") or 0),
                        str(branch.get("created_at") or "—"),
                        parent_name,
                    ),
                    tags=("active",) if is_active else (),
                )
            except tk.TclError:
                continue

        target = selected if selected is not None else active_id
        if target is not None and tree.exists(str(target)):
            try:
                tree.selection_set(str(target))
                tree.focus(str(target))
                tree.see(str(target))
            except tk.TclError:
                pass

        active = next(
            (b for b in branches if b.get("active")), {}
        ) or {}
        try:
            self.branches_info_label.config(
                text=(
                    f"Активная ветка: {active.get('name') or '—'} "
                    f"(id={active.get('id', '—')}) · всего веток: "
                    f"{len(branches)}"
                )
            )
        except tk.TclError:
            pass
        self._update_branches_buttons_state()

    def _update_branches_buttons_state(self) -> None:
        """Включает/выключает кнопки окна веток."""
        window = self._branches_window
        if window is None or not window.winfo_exists():
            return
        busy = self._branches_busy
        has_selection = self._selected_branch_id() is not None
        states = {
            "branches_create_btn": not busy,
            "branches_create_msg_btn": not busy,
            "branches_switch_btn": has_selection and not busy,
            "branches_rename_btn": has_selection and not busy,
            "branches_delete_btn": has_selection and not busy,
        }
        for attribute, enabled in states.items():
            widget = getattr(self, attribute, None)
            if widget is None:
                continue
            try:
                widget.configure(state="normal" if enabled else "disabled")
            except tk.TclError:
                continue

    def _on_branches_checkpoint(self) -> None:
        """Создаёт checkpoint — ветку от конца текущего диалога."""
        self._create_branch(from_index=None)

    def _on_branches_create_from_message(self) -> None:
        """Создаёт ветку от выбранного в чате сообщения.

        Если сообщение не выбрано, поведение совпадает с checkpoint.
        """
        index = self._selected_chat_index()
        if index is None:
            self._set_status(
                "Сообщение не выбрано — создаю ветку от конца диалога."
            )
            self._create_branch(from_index=None)
            return
        self._create_branch(from_index=index)

    def _create_branch(self, from_index: Optional[int] = None) -> None:
        """Спрашивает имя и создаёт ветку в фоновом потоке."""
        if self._busy or self._branches_busy or self._action_busy:
            self._set_status("Дождитесь завершения текущей операции.")
            return
        parent = self._branches_window or self.root
        if from_index is None:
            title = "Новая ветка"
            prompt = "Имя новой ветки (пусто — автоматически):"
        else:
            title = "Ветка от сообщения"
            prompt = (
                f"Имя ветки от сообщения №{from_index + 1} "
                "(пусто — автоматически):"
            )
        name = simpledialog.askstring(title, prompt, parent=parent)
        if name is None:  # отмена
            return
        name = name.strip() or None

        if from_index is None:
            def action() -> Dict[str, Any]:
                return self.agent.checkpoint(name=name)
        else:
            def action() -> Dict[str, Any]:
                return self.agent.create_branch(from_index=from_index, name=name)

        self._branch_action(action)

    def _on_branches_switch(self) -> None:
        """Переключается на выбранную ветку."""
        branch_id = self._selected_branch_id()
        if branch_id is None:
            self._set_status("Выберите ветку в списке.")
            return
        self._branch_action(lambda: self.agent.switch_branch(branch_id))

    def _on_branches_rename(self) -> None:
        """Переименовывает выбранную ветку."""
        branch_id = self._selected_branch_id()
        if branch_id is None:
            self._set_status("Выберите ветку в списке.")
            return
        parent = self._branches_window or self.root
        name = simpledialog.askstring(
            "Переименовать ветку", "Новое имя ветки:",
            initialvalue=self._branch_name(branch_id), parent=parent,
        )
        if name is None:
            return
        name = name.strip()
        if not name:
            self._set_status("Имя ветки не может быть пустым.")
            return
        self._branch_action(lambda: self.agent.rename_branch(branch_id, name))

    def _on_branches_delete(self) -> None:
        """Удаляет выбранную ветку (с подтверждением)."""
        branch_id = self._selected_branch_id()
        if branch_id is None:
            self._set_status("Выберите ветку в списке.")
            return
        parent = self._branches_window or self.root
        name = self._branch_name(branch_id)
        if not messagebox.askyesno(
            "Удалить ветку",
            f"Удалить ветку «{name}»?\n\nДочерние ветки останутся, но "
            "потеряют родителя.",
            parent=parent,
        ):
            return
        self._branch_action(lambda: self.agent.delete_branch(branch_id))

    def _branch_action(self, action) -> None:
        """Выполняет операцию с ветками в фоновом потоке."""
        if self._branches_busy or self._action_busy:
            self._set_status("Дождитесь завершения операции с ветками.")
            return
        self._branches_busy = True
        self._update_branches_buttons_state()
        self._set_status("Операция с ветками…")
        self._run_action(action, self._on_branch_result)

    def _on_branch_result(self, result: Dict[str, Any]) -> None:
        """Обрабатывает результат операции с ветками в главном потоке."""
        self._branches_busy = False
        self._refresh_branches_dialog()
        self._refresh_chat()
        self._update_token_stats()
        self._refresh_context_dialog()
        self._update_branches_buttons_state()

        error = result.get("error") if isinstance(result, dict) else None
        if error:
            self._set_status(f"Ветки: {error}")
            messagebox.showerror(
                "Ветки", str(error), parent=self._branches_window or self.root
            )
            return
        data = result if isinstance(result, dict) else {}
        deleted = data.get("deleted") or []
        branch = data.get("branch") or {}
        if deleted:
            self._set_status(f"Удалено веток: {len(deleted)}.")
        elif isinstance(branch, dict) and branch.get("name"):
            self._set_status(f"Активная ветка: {branch.get('name')}.")
        else:
            self._set_status("Готово.")
        self._set_info(self._token_context_text())

    def _close_branches_dialog(self) -> None:
        """Закрывает окно «Ветки»."""
        window = self._branches_window
        self._branches_window = None
        if window is not None:
            try:
                window.destroy()
            except tk.TclError:
                pass

    # ------------------------------------------------------------------
    # Окно «Сохранить в память» (рабочая / долговременная)
    # ------------------------------------------------------------------

    def _open_save_dialog(self, index: int, layer: str) -> None:
        """Открывает модальное окно сохранения сообщения в выбранный слой."""
        message = self._dialog_message(index)
        if message is None:
            self._set_status("Сообщение не найдено.")
            return
        existing = self._save_dialog
        if existing is not None:
            try:
                existing.destroy()
            except tk.TclError:
                pass

        is_working = layer == "working"
        dialog = tk.Toplevel(self.root)
        self._save_dialog = dialog
        dialog.title("В рабочую память" if is_working
                     else "В долговременную память")
        dialog.geometry("560x430")
        dialog.minsize(460, 320)
        dialog.configure(bg=COLORS["bg"])
        dialog.transient(self.root)
        dialog.protocol("WM_DELETE_WINDOW", self._close_save_dialog)

        role_name = "Вы" if message.get("role") == "user" else "Ассистент"
        quote = str(message.get("content") or "")[:200]

        # Цитата-источник (read-only)
        tk.Label(dialog, text=f"Сообщение №{index + 1} · {role_name}",
                 bg=COLORS["bg"], font=FONTS["header"], fg=COLORS["text"],
                 ).pack(anchor="w", padx=10, pady=(10, 2))
        quote_text = tk.Text(dialog, wrap="word", height=4, bg=COLORS["input_bg"],
                             fg=COLORS["text"], font=FONTS["small"],
                             relief="solid", bd=1, state="disabled")
        quote_text.pack(fill="x", padx=10)
        quote_text.configure(state="normal")
        quote_text.insert("1.0", quote)
        quote_text.configure(state="disabled")

        # Слой предвыбран и не может быть не выбран
        tk.Label(dialog, text="Слой: " + ("Рабочая (задача)" if is_working
                                           else "Долговременная (профиль, "
                                                "решения, знания)"),
                 bg=COLORS["bg"], font=FONTS["small"], fg=COLORS["muted"],
                 ).pack(anchor="w", padx=10, pady=(6, 2))

        # Поле значения (редактируемое)
        tk.Label(dialog, text="Значение (можно сократить/уточнить):",
                 bg=COLORS["bg"], font=FONTS["default"], fg=COLORS["text"],
                 ).pack(anchor="w", padx=10)
        value_text = tk.Text(dialog, wrap="word", height=5, bg=COLORS["input_bg"],
                             fg=COLORS["text"], font=FONTS["default"],
                             relief="solid", bd=1,
                             highlightthickness=1,
                             highlightbackground=COLORS["input_border"])
        value_text.pack(fill="x", padx=10, pady=(2, 6))
        value_text.insert("1.0", str(message.get("content") or ""))

        # Поля ключа/вида/раздела
        fields = tk.Frame(dialog, bg=COLORS["bg"])
        fields.pack(fill="x", padx=10)
        tk.Label(fields, text="Ключ:", bg=COLORS["bg"], font=FONTS["default"]
                 ).pack(side="left")
        key_var = tk.StringVar()
        key_entry = tk.Entry(fields, textvariable=key_var, font=FONTS["default"],
                             relief="solid", bd=1)
        key_entry.pack(side="left", fill="x", expand=True, padx=(4, 10))

        kind_var = tk.StringVar(value="note")
        section_var = tk.StringVar(value="profile")
        if is_working:
            tk.Label(fields, text="Вид:", bg=COLORS["bg"],
                     font=FONTS["default"]).pack(side="left")
            ttk.Combobox(fields, textvariable=kind_var,
                         values=("decision", "artifact", "note"), state="readonly",
                         width=12).pack(side="left", padx=(4, 0))
        else:
            tk.Label(fields, text="Раздел:", bg=COLORS["bg"],
                     font=FONTS["default"]).pack(side="left")
            ttk.Combobox(fields, textvariable=section_var,
                         values=("profile", "decisions", "knowledge"),
                         state="readonly", width=12).pack(side="left", padx=(4, 0))

        buttons = tk.Frame(dialog, bg=COLORS["bg"])
        buttons.pack(fill="x", padx=10, pady=(10, 10))

        def _confirm() -> None:
            value = value_text.get("1.0", "end-1c").strip()
            if not value:
                self._set_status("Значение не может быть пустым.")
                return
            # Значения Tk-переменных читаем в главном потоке: обращение к
            # Tk из фонового потока небезопасно.
            key = key_var.get().strip() or None
            kind = kind_var.get()
            section = section_var.get()
            if is_working:
                self._run_action(
                    lambda: self.agent.save_to_memory(
                        "working", value, key=key, kind=kind,
                        source="manual", message_index=index,
                    ),
                    self._on_save_result,
                )
            else:
                self._run_action(
                    lambda: self.agent.save_to_memory(
                        "long_term", value, key=key, section=section,
                        source="manual", message_index=index,
                    ),
                    self._on_save_result,
                )
            self._close_save_dialog()

        self._make_button(buttons, "Сохранить", _confirm).pack(side="left")
        self._make_button(buttons, "Отмена",
                          self._close_save_dialog).pack(side="left", padx=(8, 0))

        # Предзаполнение локальными правилами (без сети, не блокирует UI)
        self._prefill_save_dialog(message.get("content") or "", key_var,
                                  kind_var, section_var, is_working)

    def _prefill_save_dialog(self, text: str, key_var, kind_var, section_var,
                             is_working: bool) -> None:
        """Заполняет ключ/вид/раздел по локальным правилам классификации."""
        try:
            from .memory import MemoryClassifier
            local = MemoryClassifier.classify_local(text)
        except Exception:
            return
        if is_working:
            candidates = local.get("working") or []
            if candidates:
                first = candidates[0]
                if first.get("key") and not key_var.get():
                    key_var.set(first["key"])
                if first.get("kind"):
                    kind_var.set(first["kind"])
        else:
            candidates = local.get("long_term") or []
            if candidates:
                first = candidates[0]
                if first.get("key") and not key_var.get():
                    key_var.set(first["key"])
                if first.get("section"):
                    section_var.set(first["section"])

    def _close_save_dialog(self) -> None:
        """Закрывает окно сохранения в память."""
        dialog = self._save_dialog
        self._save_dialog = None
        if dialog is not None:
            try:
                dialog.destroy()
            except tk.TclError:
                pass

    def _on_save_result(self, result: Dict[str, Any]) -> None:
        """Обрабатывает результат сохранения в память."""
        self._refresh_chat()
        self._update_token_stats()
        self._refresh_memory_dialog()
        if result.get("ok"):
            layer = result.get("layer")
            title = {"working": "рабочая память",
                     "long_term": "долговременная память"}.get(layer, layer)
            self._set_status(
                f"Сохранено: {title} · {result.get('key')} = {result.get('value')}"
            )
        else:
            self._set_status(f"Ошибка: {result.get('error')}")

    # ------------------------------------------------------------------
    # Окно «Память…» (три вкладки + pending + предпросмотр запроса)
    # ------------------------------------------------------------------

    def _open_memory_dialog(self) -> None:
        """Открывает окно «Память…» с тремя вкладками и предпросмотром."""
        window = self._memory_window
        if window is not None and window.winfo_exists():
            self._refresh_memory_dialog()
            window.lift()
            window.focus_set()
            return

        window = tk.Toplevel(self.root)
        self._memory_window = window
        window.title("Память")
        window.geometry("820x560")
        window.minsize(640, 420)
        window.configure(bg=COLORS["bg"])
        window.transient(self.root)
        window.protocol("WM_DELETE_WINDOW", self._close_memory_dialog)

        notebook = ttk.Notebook(window)
        notebook.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        self._memory_notebook = notebook

        # --- Вкладка 1: краткосрочная ---
        short_tab = tk.Frame(notebook, bg=COLORS["bg"])
        notebook.add(short_tab, text="Краткосрочная (диалог)")
        self._short_header = tk.Label(short_tab, text="", bg=COLORS["bg"],
                                      font=FONTS["small"], fg=COLORS["muted"],
                                      anchor="w")
        self._short_header.pack(fill="x", padx=8, pady=(6, 2))
        self._build_tree(short_tab, "short", ("index", "role", "preview", "pinned"))
        short_btns = tk.Frame(short_tab, bg=COLORS["bg"])
        short_btns.pack(fill="x", padx=8, pady=(4, 8))
        self._make_button(short_btns, "Закрепить/Открепить",
                          self._on_short_toggle, small=True).pack(side="left")
        self._make_button(short_btns, "Открепить все",
                          self._on_short_unpin_all, small=True).pack(
            side="left", padx=(8, 0))

        # --- Вкладка 2: рабочая ---
        working_tab = tk.Frame(notebook, bg=COLORS["bg"])
        notebook.add(working_tab, text="Рабочая (задача)")
        self._working_fields = {}
        for label, key, height in (
            ("Цель", "goal", 1), ("Статус", "status", 1),
            ("План (по строке)", "plan", 3),
            ("Ограничения (по строке)", "constraints", 3),
            ("Открытые вопросы (по строке)", "open_questions", 3),
        ):
            row = tk.Frame(working_tab, bg=COLORS["bg"])
            row.pack(fill="x", padx=8, pady=(2, 0))
            tk.Label(row, text=label + ":", bg=COLORS["bg"],
                     font=FONTS["small"], fg=COLORS["text"]).pack(anchor="w")
            widget = tk.Text(row, wrap="word", height=height,
                             bg=COLORS["input_bg"], fg=COLORS["text"],
                             font=FONTS["default"], relief="solid", bd=1,
                             highlightthickness=1,
                             highlightbackground=COLORS["input_border"])
            widget.pack(fill="x", pady=(2, 0))
            self._working_fields[key] = widget
        self._build_tree(working_tab, "working",
                         ("key", "value", "kind", "source", "updated"))
        working_btns = tk.Frame(working_tab, bg=COLORS["bg"])
        working_btns.pack(fill="x", padx=8, pady=(4, 8))
        self._make_button(working_btns, "Сохранить поля",
                          self._on_working_save_fields, small=True).pack(side="left")
        self._make_button(working_btns, "Добавить элемент",
                          self._on_working_add_item, small=True).pack(
            side="left", padx=(8, 0))
        self._make_button(working_btns, "Удалить элемент",
                          self._on_working_delete_item, small=True).pack(
            side="left", padx=(8, 0))
        self._make_button(working_btns, "Новая задача",
                          self._on_working_new_task, small=True).pack(
            side="left", padx=(8, 0))
        self._make_button(working_btns, "Очистить рабочую",
                          self._on_working_clear, small=True).pack(
            side="left", padx=(8, 0))

        # --- Вкладка 3: долговременная ---
        long_tab = tk.Frame(notebook, bg=COLORS["bg"])
        notebook.add(long_tab, text="Долговременная")
        search_row = tk.Frame(long_tab, bg=COLORS["bg"])
        search_row.pack(fill="x", padx=8, pady=(6, 2))
        tk.Label(search_row, text="Поиск:", bg=COLORS["bg"],
                 font=FONTS["small"]).pack(side="left")
        self._long_search = tk.Entry(search_row, font=FONTS["default"],
                                     relief="solid", bd=1)
        self._long_search.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self._long_search.bind("<KeyRelease>", lambda _e: self._refresh_long_tab())
        self._build_tree(long_tab, "long",
                         ("section", "key", "value", "source", "created", "updated"))
        long_btns = tk.Frame(long_tab, bg=COLORS["bg"])
        long_btns.pack(fill="x", padx=8, pady=(4, 8))
        self._make_button(long_btns, "Добавить", self._on_long_add,
                          small=True).pack(side="left")
        self._make_button(long_btns, "Забыть", self._on_long_forget,
                          small=True).pack(side="left", padx=(8, 0))
        self._make_button(long_btns, "Очистить раздел", self._on_long_clear_section,
                          small=True).pack(side="left", padx=(8, 0))
        self._make_button(long_btns, "Очистить долговременную",
                          self._on_long_clear_all, small=True).pack(
            side="left", padx=(8, 0))
        self._make_button(long_btns, "Экспорт…", self._on_long_export,
                          small=True).pack(side="left", padx=(8, 0))
        self._make_button(long_btns, "Импорт…", self._on_long_import,
                          small=True).pack(side="left", padx=(8, 0))

        # --- Вкладка 4: предложения ---
        pending_tab = tk.Frame(notebook, bg=COLORS["bg"])
        notebook.add(pending_tab, text="Предложения (pending)")
        self._build_tree(pending_tab, "pending",
                         ("id", "layer", "section", "key", "value", "confidence"))
        pending_btns = tk.Frame(pending_tab, bg=COLORS["bg"])
        pending_btns.pack(fill="x", padx=8, pady=(4, 8))
        self._make_button(pending_btns, "Принять в …", self._on_pending_accept,
                          small=True).pack(side="left")
        self._make_button(pending_btns, "Отклонить", self._on_pending_reject,
                          small=True).pack(side="left", padx=(8, 0))
        self._make_button(pending_btns, "Отклонить все",
                          self._on_pending_reject_all, small=True).pack(
            side="left", padx=(8, 0))

        # --- Низ окна: сводка + «Что уходит в запрос» ---
        bottom = tk.Frame(window, bg=COLORS["bg"])
        bottom.pack(fill="x", padx=8, pady=(4, 8))
        self._memory_summary_label = tk.Label(
            bottom, text="", bg=COLORS["bg"], font=FONTS["small"],
            fg=COLORS["muted"], anchor="w")
        self._memory_summary_label.pack(side="left", fill="x", expand=True)
        self._make_button(bottom, "Что уходит в запрос",
                          self._open_preview_window, small=True).pack(side="right")

        self._refresh_memory_dialog()

    def _build_tree(self, parent, name: str, columns) -> None:
        """Создаёт Treeview с горизонтальной прокруткой и кнопками."""
        wrap = tk.Frame(parent, bg=COLORS["bg"])
        wrap.pack(fill="both", expand=True, padx=8)
        tree = ttk.Treeview(wrap, columns=columns, show="headings",
                            selectmode="browse", height=8)
        for column in columns:
            tree.heading(column, text=column)
            tree.column(column, width=120, anchor="w")
        tree.pack(side="left", fill="both", expand=True)
        yscroll = ttk.Scrollbar(wrap, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=yscroll.set)
        yscroll.pack(side="right", fill="y")
        setattr(self, f"_memory_{name}_tree", tree)

    def _memory_tree(self, name: str):
        """Возвращает Treeview вкладки (None, если окна нет)."""
        window = self._memory_window
        if window is None or not window.winfo_exists():
            return None
        return getattr(self, f"_memory_{name}_tree", None)

    def _selected_memory_item(self, name: str) -> Optional[tuple]:
        """iid выбранной строки таблицы (None, если нет)."""
        tree = self._memory_tree(name)
        if tree is None:
            return None
        try:
            selection = tree.selection()
        except (tk.TclError, AttributeError):
            return None
        if not selection:
            return None
        try:
            values = tree.item(selection[0], "values")
            return tuple(values)
        except (tk.TclError, AttributeError):
            return None

    def _refresh_memory_dialog(self) -> None:
        """Перерисовывает окно «Память…» по состоянию агента."""
        window = self._memory_window
        if window is None or not window.winfo_exists():
            return
        self._refresh_short_tab()
        self._refresh_working_tab()
        self._refresh_long_tab()
        self._refresh_pending_tab()
        self._refresh_memory_summary()

    def _refresh_short_tab(self) -> None:
        tree = self._memory_tree("short")
        if tree is None:
            return
        try:
            tree.delete(*tree.get_children())
            pinned = {int(p["index"]) for p in self.agent.list_pinned()}
            history = self.agent.history[1:]
            for i, message in enumerate(history):
                role = "Вы" if message.get("role") == "user" else "Ассистент"
                preview = " ".join(str(message.get("content") or "").split())[:60]
                tree.insert("", "end", values=(
                    str(i + 1), role, preview,
                    "✓" if i in pinned else "",
                ))
            self._short_header.config(
                text=f"{len(history)} сообщений · закреплено {len(pinned)}"
            )
        except tk.TclError:
            pass

    def _refresh_working_tab(self) -> None:
        tree = self._memory_tree("working")
        if tree is None:
            return
        try:
            working = self.agent.get_working_memory()
            task = working.get("task") or {}
            items = working.get("items") or {}
            for key, widget in self._working_fields.items():
                widget.configure(state="normal")
                widget.delete("1.0", "end")
                if key in ("plan", "constraints", "open_questions"):
                    widget.insert("1.0", "\n".join(task.get(key, [])))
                else:
                    widget.insert("1.0", task.get(key, ""))
            tree.delete(*tree.get_children())
            for key, item in items.items():
                tree.insert("", "end", values=(
                    key, item.get("value", ""), item.get("kind", "note"),
                    item.get("source", "manual"),
                    (item.get("updated_at") or "")[:16],
                ))
        except tk.TclError:
            pass

    def _refresh_long_tab(self) -> None:
        tree = self._memory_tree("long")
        if tree is None:
            return
        try:
            query = ""
            try:
                query = self._long_search.get().strip()
            except (tk.TclError, AttributeError):
                pass
            items = self.agent.get_long_memory()
            if query:
                needle = query.lower()
                items = [i for i in items
                         if needle in str(i.get("key", "")).lower()
                         or needle in str(i.get("value", "")).lower()]
            tree.delete(*tree.get_children())
            for item in items:
                tree.insert("", "end", values=(
                    item.get("section", ""), item.get("key", ""),
                    item.get("value", ""), item.get("source", "manual"),
                    (item.get("created_at") or "")[:16],
                    (item.get("updated_at") or "")[:16],
                ))
        except tk.TclError:
            pass

    def _refresh_pending_tab(self) -> None:
        tree = self._memory_tree("pending")
        if tree is None:
            return
        try:
            tree.delete(*tree.get_children())
            for item in self.agent.get_pending_memory():
                tree.insert("", "end", values=(
                    item.get("id"), item.get("layer"), item.get("section") or "",
                    item.get("key") or "", item.get("value", ""),
                    item.get("confidence"),
                ))
        except tk.TclError:
            pass

    def _refresh_memory_summary(self) -> None:
        try:
            report = self.agent.get_memory_report()
        except Exception:
            return
        short = report.get("short_term") or {}
        working = report.get("working") or {}
        long_term = report.get("long_term") or {}
        budget = report.get("budget") or {}
        text = (
            f"Краткосрочная: {int(short.get('pinned_count') or 0)}/≈0 ток. · "
            f"Рабочая: {int(working.get('count') or 0)}/"
            f"≈{int(working.get('tokens') or 0)} · "
            f"Долговременная: {int(long_term.get('count') or 0)}/"
            f"≈{int(long_term.get('tokens') or 0)} · "
            f"бюджет ≈{int(budget.get('max_tokens') or 0)} ток."
        )
        try:
            self._memory_summary_label.config(text=text)
        except (tk.TclError, AttributeError):
            pass

    def _close_memory_dialog(self) -> None:
        """Закрывает окно «Память…»."""
        window = self._memory_window
        self._memory_window = None
        if window is not None:
            try:
                window.destroy()
            except tk.TclError:
                pass

    def _open_preview_window(self) -> None:
        """Окно «Что уходит в запрос» — построчный предпросмотр контекста."""
        try:
            preview = self.agent.preview_context()
        except Exception as exc:
            messagebox.showerror("Предпросмотр", str(exc), parent=self._memory_window)
            return
        window = tk.Toplevel(self.root)
        window.title("Что уходит в запрос")
        window.geometry("720x520")
        window.configure(bg=COLORS["bg"])
        window.transient(self._memory_window or self.root)
        text = tk.Text(window, wrap="word", bg=COLORS["input_bg"],
                       fg=COLORS["text"], font=FONTS["default"],
                       relief="flat", padx=8, pady=6, cursor="arrow")
        text.pack(fill="both", expand=True, padx=8, pady=8)
        layer_titles = {"system": "system", "long_term": "долговременная",
                        "working": "рабочая", "pinned": "закреплённые",
                        "short_term": "краткосрочная"}
        for item in preview:
            layer = layer_titles.get(item.get("layer"), item.get("layer"))
            text.insert("end", f"[{layer} · {item.get('tokens', 0)} ток.]\n")
            text.insert("end", f"{item.get('role')}: {item.get('content')}\n\n")
        text.configure(state="disabled")

    # --- Обработчики вкладок окна «Память…» ---

    def _on_short_toggle(self) -> None:
        selected = self._selected_memory_item("short")
        if selected is None:
            self._set_status("Выберите сообщение в таблице.")
            return
        index = int(selected[0]) - 1  # в таблице нумерация с 1
        self._on_save_short(index)

    def _on_short_unpin_all(self) -> None:
        for pin in list(self.agent.list_pinned()):
            self.agent.unpin_message(int(pin["index"]))
        self._refresh_chat()
        self._update_token_stats()
        self._refresh_memory_dialog()
        self._set_status("Все закрепления сняты.")

    def _on_working_save_fields(self) -> None:
        fields = {}
        for key, widget in self._working_fields.items():
            fields[key] = widget.get("1.0", "end-1c")
        self._run_action(
            lambda: self._save_working_fields(fields), self._on_memory_action_done
        )

    def _save_working_fields(self, fields: Dict[str, str]) -> Dict[str, Any]:
        for key, value in fields.items():
            if key in ("plan", "constraints", "open_questions"):
                self.agent.set_task_field(key, value.split("\n"))
            else:
                self.agent.set_task_field(key, value)
        return {"ok": True, "error": None}

    def _on_working_add_item(self) -> None:
        parent = self._memory_window or self.root
        key = simpledialog.askstring("Элемент", "Ключ:", parent=parent)
        if key is None:
            return
        value = simpledialog.askstring("Элемент", "Значение:", parent=parent)
        if value is None:
            return
        kind = simpledialog.askstring(
            "Элемент", "Вид (decision/artifact/note):",
            initialvalue="note", parent=parent,
        ) or "note"
        self._run_action(
            lambda: self.agent.add_working_item(key, value, kind),
            self._on_memory_action_done,
        )

    def _on_working_delete_item(self) -> None:
        selected = self._selected_memory_item("working")
        if selected is None:
            self._set_status("Выберите элемент в таблице.")
            return
        key = selected[0]
        self._run_action(
            lambda: self.agent.forget_working_item(key),
            self._on_memory_action_done,
        )

    def _on_working_new_task(self) -> None:
        parent = self._memory_window or self.root
        if not messagebox.askyesno(
            "Новая задача",
            "Начать новую задачу?\n\nРабочая память будет очищена. "
            "Перенести решения (kind=decision) в долговременную память?",
            parent=parent,
        ):
            # Отмена: просто очищаем без переноса
            self._run_action(
                lambda: self.agent.clear_working_memory(),
                self._on_memory_action_done,
            )
            return
        working = self.agent.get_working_memory()
        decisions = [
            key for key, item in (working.get("items") or {}).items()
            if item.get("kind") == "decision"
        ]
        self._run_action(
            lambda: self.agent.start_new_task(carry_to_long=decisions),
            self._on_memory_action_done,
        )

    def _on_working_clear(self) -> None:
        parent = self._memory_window or self.root
        if not messagebox.askyesno("Очистить рабочую память",
                                   "Очистить рабочую память задачи?",
                                   parent=parent):
            return
        self._run_action(lambda: self.agent.clear_working_memory(),
                         self._on_memory_action_done)

    def _on_long_add(self) -> None:
        parent = self._memory_window or self.root
        section = simpledialog.askstring(
            "Добавить", "Раздел (profile/decisions/knowledge):",
            initialvalue="profile", parent=parent,
        )
        if section is None:
            return
        key = simpledialog.askstring("Добавить", "Ключ:", parent=parent)
        if key is None:
            return
        value = simpledialog.askstring("Добавить", "Значение:", parent=parent)
        if value is None:
            return
        self._run_action(
            lambda: self.agent.remember_forever(section, key, value),
            self._on_memory_action_done,
        )

    def _on_long_forget(self) -> None:
        selected = self._selected_memory_item("long")
        if selected is None:
            self._set_status("Выберите запись в таблице.")
            return
        section, key = selected[0], selected[1]
        self._run_action(
            lambda: self.agent.forget_forever(section, key),
            self._on_memory_action_done,
        )

    def _on_long_clear_section(self) -> None:
        selected = self._selected_memory_item("long")
        if selected is None:
            self._set_status("Выберите запись в таблице (для определения раздела).")
            return
        section = selected[0]
        parent = self._memory_window or self.root
        if not messagebox.askyesno(
            "Очистить раздел",
            f"Очистить раздел «{section}» долговременной памяти?\n\n"
            "Краткосрочная и рабочая память не пострадают.",
            parent=parent,
        ):
            return
        self._run_action(
            lambda: self.agent.clear_long_memory(section),
            self._on_memory_action_done,
        )

    def _on_long_clear_all(self) -> None:
        parent = self._memory_window or self.root
        if not messagebox.askyesno(
            "Очистить долговременную память",
            "Удалить ВСЮ долговременную память?\n\nЭто действие необратимо.",
            parent=parent,
        ):
            return
        self._run_action(
            lambda: self.agent.clear_long_memory(None),
            self._on_memory_action_done,
        )

    def _on_long_export(self) -> None:
        parent = self._memory_window or self.root
        path = simpledialog.askstring(
            "Экспорт", "Путь к файлу Markdown:", parent=parent,
        )
        if path is None:
            return
        self._run_action(
            lambda: self.agent.export_long_memory(path),
            self._on_memory_action_done,
        )

    def _on_long_import(self) -> None:
        parent = self._memory_window or self.root
        path = simpledialog.askstring(
            "Импорт", "Путь к файлу Markdown:", parent=parent,
        )
        if path is None:
            return
        self._run_action(
            lambda: self.agent.import_long_memory(path),
            self._on_memory_action_done,
        )

    def _on_pending_accept(self) -> None:
        selected = self._selected_memory_item("pending")
        if selected is None:
            self._set_status("Выберите предложение в таблице.")
            return
        pending_id = int(selected[0])
        self._run_action(
            lambda: self.agent.accept_pending(pending_id),
            self._on_memory_action_done,
        )

    def _on_pending_reject(self) -> None:
        selected = self._selected_memory_item("pending")
        if selected is None:
            self._set_status("Выберите предложение в таблице.")
            return
        pending_id = int(selected[0])
        self._run_action(
            lambda: self.agent.reject_pending(pending_id),
            self._on_memory_action_done,
        )

    def _on_pending_reject_all(self) -> None:
        self._run_action(lambda: self.agent.clear_pending_memory(),
                         self._on_memory_action_done)

    def _on_memory_action_done(self, result: Dict[str, Any]) -> None:
        """Обрабатывает результат фоновой операции окна «Память…»."""
        self._refresh_chat()
        self._update_token_stats()
        self._refresh_memory_dialog()
        error = result.get("error") if isinstance(result, dict) else None
        if error:
            self._set_status(f"Ошибка: {error}")
        else:
            self._set_status("Готово.")

    # ------------------------------------------------------------------
    # Фоновые операции UI (ветки и прочие вызовы API по кнопкам)
    # ------------------------------------------------------------------

    def _run_action(self, action, on_done) -> None:
        """Выполняет ``action`` в фоновом потоке и отдаёт результат в UI.

        Результат передаётся в главный поток через очередь и ``root.after``,
        чтобы Tkinter не вызывался из рабочего потока.
        """
        if self._action_busy:
            self._set_status("Дождитесь завершения операции.")
            return
        self._action_busy = True

        def worker() -> None:
            try:
                result = action()
            except Exception as exc:  # агент не должен бросать, но страхуемся
                result = {"error": f"Непредвиденная ошибка: {exc}"}
            self._action_queue.put((result, on_done))

        threading.Thread(target=worker, daemon=True, name="ui-action").start()
        self.root.after(80, self._poll_action_queue)

    def _poll_action_queue(self) -> None:
        """Опрашивает очередь фоновых операций (вызывается через root.after)."""
        try:
            while True:
                result, on_done = self._action_queue.get_nowait()
                self._action_busy = False
                try:
                    on_done(result)
                except Exception:
                    pass
        except queue.Empty:
            pass
        finally:
            if self._action_busy:
                self.root.after(80, self._poll_action_queue)

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
            self.context_btn,
            self.branches_btn,
            self.memory_btn,
            self.temp_scale,
            self.tokens_spin,
            self.prompt_entry,
            self.input,
            # Персонализация: вкладка «Профиль»
            self.profile_new_btn,
            self.profile_save_btn,
            self.profile_delete_btn,
            self.profile_activate_btn,
            self.profile_name_entry,
            self.profile_display_entry,
            self.profile_style_entry,
            self.profile_format_entry,
            self.profile_language_entry,
            self.profile_constraints_text,
            self.profile_context_text,
            # Персонализация: вкладка «Настройки агента»
            self.profile_temp_scale,
            self.profile_tokens_spin,
            self.profile_prompt_text,
            self.profile_settings_save_btn,
            # Инварианты: вкладка «Инварианты»
            self.invariant_new_btn,
            self.invariant_edit_btn,
            self.invariant_delete_btn,
            self.category_new_btn,
            self.category_edit_btn,
            self.category_delete_btn,
            self.invariants_category_list,
            self.invariants_list,
        ):
            widget.configure(state="normal" if enabled else "disabled")

        # ttk.Combobox: обычное состояние — "readonly", а не "normal"
        for combo in (self.model_combo, self.strategy_combo,
                      self.profile_combo, self.profile_model_combo,
                      self.profile_switch, self.invariants_profile_combo):
            combo.configure(state="readonly" if enabled else "disabled")

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
            self._update_strategy_controls()
            self._refresh_branches_dialog()
            self._refresh_context_dialog()
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
            "Удалить всю историю диалога?\n\n"
            "Будут очищены сообщения активной ветки, её закрепления и рабочая "
            "память. Долговременная память (профиль, решения, знания) "
            "сохранится.",
            parent=self.root,
        )
        if not confirmed:
            return
        self.agent.clear_history()
        self._refresh_chat()
        self._update_token_stats()
        self._update_strategy_controls()
        self._refresh_branches_dialog()
        self._refresh_context_dialog()
        self._refresh_memory_dialog()
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

    @staticmethod
    def _parse_task_command(text: str) -> Optional[str]:
        """Разбирает команду создания задачи.

        Задача задаётся сообщением в чате: ``/задача <описание>`` или
        ``/task <описание>`` (регистр не важен).

        Returns:
            Описание задачи (пустая строка — команда без описания) или
            None, если это обычное сообщение, а не команда.
        """
        stripped = text.strip()
        lowered = stripped.lower()
        for command in TASK_COMMANDS:
            if lowered == command:
                return ""
            if lowered.startswith(command + " "):
                return stripped[len(command):].strip()
        return None

    def _start_task_from_chat(self, message: str,
                              description: str) -> None:
        """Создаёт задачу из сообщения чата (вместо отдельной кнопки).

        Сообщение показывается в чате как обычно и остаётся в истории
        диалога; задача создаётся в ручном режиме — пользователь
        управляет переходами между этапами (подтверждает план и итог).
        """
        if not description:
            self._set_status(
                "Укажите описание после команды: /задача что нужно сделать.")
            self.root.bell()
            return
        if self._task_busy:
            self._set_status(
                "Дождитесь завершения текущей операции с задачей.")
            self.root.bell()
            return

        # Показываем сообщение пользователя сразу (это обычное сообщение
        # чата, просто оно ещё и создаёт задачу)
        self.chat.configure(state="normal")
        try:
            self._insert_message("user", message)
        finally:
            self.chat.configure(state="disabled")
        self._scroll_to_end()
        self._set_status("Создание задачи...")

        def action():
            # Сообщение остаётся в истории диалога (без вызова LLM)
            self.agent.note_task_message(message)
            return self.agent.start_task(description, TaskMode.MANUAL)

        self._run_task_action(action)

    def _send_text(self, text: str) -> None:
        """Отправляет текст агенту в фоновом потоке.

        Сообщение пользователя немедленно показывается в чате, интерфейс
        блокируется, а результат обрабатывается по завершении потока.
        Сообщение «/задача …» вместо обычного запроса создаёт задачу.
        """
        if self._busy:
            return

        description = self._parse_task_command(text)
        if description is not None:
            self._start_task_from_chat(text.strip(), description)
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
        self._close_context_dialog()
        self._close_branches_dialog()
        self._close_memory_dialog()
        self._close_save_dialog()
        self.agent.close()
        self.root.destroy()
