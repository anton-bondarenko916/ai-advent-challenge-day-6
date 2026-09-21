"""Панель инструментов MCP рядом с областью чата (Tkinter).

MCPPanel — сворачиваемый Frame, который встраивается в правую часть
вкладки «Чат»: индикатор состояния, кнопка «Обновить», скроллируемый
список карточек инструментов. Клик по карточке открывает модальное окно
с описанием и inputSchema (JSON) инструмента.

На данном этапе панель ТОЛЬКО получает и показывает список. Вызов
инструментов — следующий этап подсистемы.

Потокобезопасность: запрос списка выполняет MCPWorker в отдельном
потоке; его callback'и кладут результат в queue.Queue, а главный поток
опрашивает очередь через ``root.after`` — тот же паттерн, что у задач
(FSM) в ui.py, поэтому UI никогда не блокируется и не обновляется из
чужого потока.

Состояние сворачивания сохраняется в SQLite (AppStateStore, таблица
app_state) и восстанавливается при следующем запуске.
"""

from __future__ import annotations

import queue
from typing import Any, Callable, Dict, List, Optional

import tkinter as tk
from tkinter import messagebox, ttk

from .app_state import AppStateStore
from .mcp_models import MCPTool
from .mcp_worker import MCPWorker

# Состояния индикатора панели: цвет точки ● и подпись.
STATUS_IDLE = "idle"            # серый:  «Не подключён» (старт)
STATUS_LOADING = "loading"      # жёлтый: «Загрузка...»
STATUS_CONNECTED = "connected"  # зелёный: «Подключён | N инструментов»
STATUS_ERROR = "error"          # красный: «Ошибка: …»

_STATUS_COLORS = {
    STATUS_IDLE: "#9E9E9E",
    STATUS_LOADING: "#F9A825",
    STATUS_CONNECTED: "#2E7D32",
    STATUS_ERROR: "#C62828",
}

_WIDTH = 262          # ширина панели (px)
_CARD_LIMIT = 80      # лимит символов краткого описания в карточке

# Подсказка для случая, когда Node.js/npx не установлен (см. MCPClient).
NODEJS_HINT_MARK = "https://nodejs.org"

COLLAPSED_KEY = "mcp_panel_collapsed"


class MCPPanel(tk.Frame):
    """Панель «MCP»: статус, список инструментов, кнопка обновления.

    Attributes:
        settings: настройки из config.get_mcp_settings (enabled,
            server_command, timeout, log_path).
        app_state: хранилище состояния UI; None — сворачивание не
            сохраняется.
        tools: последний полученный список инструментов.
        collapsed: свёрнута ли панель (остаётся только заголовок).
    """

    def __init__(
        self,
        master: tk.Misc,
        settings: Dict[str, Any],
        app_state: Optional[AppStateStore] = None,
        worker_factory: Optional[Callable[[Any, Any], MCPWorker]] = None,
    ) -> None:
        """Строит панель.

        Args:
            master: родительский виджет (обёртка вкладки «Чат»).
            settings: словарь настроек MCP (get_mcp_settings).
            app_state: хранилище состояния UI (сворачивание панели).
            worker_factory: фабрика воркера ``f(on_success, on_error)``;
                None — стандартный MCPWorker. Для тестов подаётся фейк.
        """
        super().__init__(master, bg="#ECECEC", width=_WIDTH)
        self.settings = settings
        self.app_state = app_state
        self.tools: List[MCPTool] = []
        self.collapsed = False
        self._busy = False
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._poll_job: Optional[str] = None
        self._status_kind = STATUS_IDLE
        self._status_text = "Не подключён"

        if worker_factory is not None:
            self._make_worker = worker_factory
        else:
            self._make_worker = self._default_worker_factory

        self._build_widgets()

    # ------------------------------------------------------------------
    # Построение виджетов
    # ------------------------------------------------------------------

    def _build_widgets(self) -> None:
        """Создаёт заголовок (всегда виден) и тело (скрывается).

        Геометрия: pack_propagate НЕ отключаем — при сворачивании панель
        должна сама сжиматься до высоты заголовка.
        """
        # ---- Заголовок: точка-индикатор, «MCP», кнопка сворачивания ----
        self.header = tk.Frame(self, bg="#ECECEC")
        self.header.pack(side="top", fill="x", padx=8, pady=(6, 2))

        self.status_dot = tk.Label(
            self.header, text="●", bg="#ECECEC",
            fg=_STATUS_COLORS[STATUS_IDLE], font=FONTS_DOT,
        )
        self.status_dot.pack(side="left")

        tk.Label(
            self.header, text=" MCP", bg="#ECECEC", fg="#1A1A1A",
            font=FONTS_HEADER,
        ).pack(side="left")

        self.toggle_btn = tk.Button(
            self.header, text="◀", command=self._on_toggle_collapsed,
            relief="flat", bg="#ECECEC", fg="#666666",
            font=FONTS_SMALL, bd=0, cursor="hand2",
            activebackground="#E0E0E0",
        )
        self.toggle_btn.pack(side="right")

        # ---- Тело: статус, кнопка обновления, список инструментов ----
        self.body = tk.Frame(self, bg="#ECECEC")
        self.body.pack(side="top", fill="both", expand=True, padx=8)

        self.status_label = tk.Label(
            self.body, text="", bg="#ECECEC", fg="#1A1A1A",
            font=FONTS_SMALL, anchor="w", justify="left",
            wraplength=_WIDTH - 24,
        )
        self.status_label.pack(side="top", fill="x", pady=(0, 4))
        self._render_status()

        self.refresh_btn = tk.Button(
            self.body, text="🔄 Обновить", command=self.refresh,
            bg="#007AFF", fg="#FFFFFF", activebackground="#0066D6",
            activeforeground="#FFFFFF", relief="flat", cursor="hand2",
            font=FONTS_SMALL,
        )
        self.refresh_btn.pack(side="top", fill="x", pady=(0, 6))

        # Скроллируемая область карточек инструментов
        list_wrap = tk.Frame(self.body, bg="#ECECEC")
        list_wrap.pack(side="top", fill="both", expand=True)

        self.canvas = tk.Canvas(
            list_wrap, bg="#ECECEC", highlightthickness=0, bd=0,
        )
        scrollbar = ttk.Scrollbar(
            list_wrap, orient="vertical", command=self.canvas.yview
        )
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.tools_frame = tk.Frame(self.canvas, bg="#ECECEC")
        self._canvas_window = self.canvas.create_window(
            (0, 0), window=self.tools_frame, anchor="nw"
        )
        self.tools_frame.bind(
            "<Configure>", self._on_tools_frame_configure
        )
        self.canvas.bind(
            "<Configure>", self._on_canvas_configure
        )
        # Колесо мыши: прокрутка списка (Windows)
        self.canvas.bind_all("<MouseWheel>", self._on_mouse_wheel)

        self._show_placeholder(
            "Список ещё не запрашивался.\nНажмите «Обновить»."
        )

    # ------------------------------------------------------------------
    # Статус
    # ------------------------------------------------------------------

    def get_status_text(self) -> str:
        """Текст статуса без цветовой точки (для тестов и статус-бара)."""
        return self._status_text

    def get_status_kind(self) -> str:
        """Текущее состояние: idle | loading | connected | error."""
        return self._status_kind

    def _set_status(self, kind: str, text: str) -> None:
        """Обновляет индикатор и текст статуса (только главный поток)."""
        self._status_kind = kind
        self._status_text = text
        self._render_status()

    def _render_status(self) -> None:
        """Рисует текущий статус на виджетах."""
        try:
            self.status_dot.config(fg=_STATUS_COLORS[self._status_kind])
            self.status_label.config(text=self._status_text)
        except tk.TclError:
            pass  # панель уже уничтожена (закрытие приложения)

    # ------------------------------------------------------------------
    # Обновление списка (запрос в отдельном потоке)
    # ------------------------------------------------------------------

    def _default_worker_factory(self, on_success, on_error) -> MCPWorker:
        """Стандартная фабрика: MCPWorker с настройками из config."""
        return MCPWorker(
            on_success=on_success,
            on_error=on_error,
            server_command=self.settings.get("server_command"),
            timeout=self.settings.get("timeout"),
            log_path=str(self.settings.get("log_path") or ""),
        )

    def refresh(self) -> None:
        """Запускает запрос tools/list в фоновом потоке.

        Пока идёт запрос: статус «Загрузка...», кнопка заблокирована.
        Повторные клики игнорируются (никаких параллельных сессий).
        """
        if self._busy:
            return
        self._busy = True
        self.refresh_btn.config(state="disabled")
        self._clear_tools()
        self._set_status(STATUS_LOADING, "Загрузка...")

        worker = self._make_worker(
            on_success=lambda tools: self._queue.put({"tools": tools}),
            on_error=lambda message: self._queue.put({"error": message}),
        )
        worker.start()
        self._poll_job = self.after(100, self._poll)

    def _poll(self) -> None:
        """Опрашивает очередь результатов воркера (главный поток)."""
        self._poll_job = None
        try:
            while True:
                result = self._queue.get_nowait()
                if "tools" in result:
                    self._on_tools_loaded(result["tools"])
                elif "error" in result:
                    self._on_error(result["error"])
        except queue.Empty:
            pass
        finally:
            if self._busy:
                self._poll_job = self.after(100, self._poll)

    def _on_tools_loaded(self, tools: List[MCPTool]) -> None:
        """Успешный ответ: перерисовать список (без дублей)."""
        self._busy = False
        self.tools = list(tools)
        self.refresh_btn.config(state="normal")
        count = len(self.tools)
        self._set_status(
            STATUS_CONNECTED,
            f"Подключён | {count} {_plural_tools(count)}"
        )
        self._clear_tools()
        if not self.tools:
            # Пустой список — НЕ ошибка.
            self._show_placeholder(
                "Сервер ответил, но инструментов не вернул."
            )
            return
        for tool in self.tools:
            self._add_tool_card(tool)

    def _on_error(self, message: str) -> None:
        """Ошибка запроса: красный статус + messagebox с текстом."""
        self._busy = False
        self.tools = []
        self.refresh_btn.config(state="normal")
        short = message if len(message) <= 120 else message[:117] + "..."
        self._set_status(STATUS_ERROR, f"Ошибка: {short}")
        self._show_placeholder("Нет данных: последний запрос завершился ошибкой.")

        dialog_text = message
        if NODEJS_HINT_MARK in message:
            dialog_text += (
                "\n\nNode.js не установлен или не виден в PATH.\n"
                f"Скачайте установщик: {NODEJS_HINT_MARK}"
            )
        try:
            messagebox.showerror("Ошибка MCP", dialog_text, parent=self)
        except tk.TclError:
            pass  # окно приложения закрывается

    # ------------------------------------------------------------------
    # Карточки инструментов
    # ------------------------------------------------------------------

    def _clear_tools(self) -> None:
        """Удаляет все карточки и плейсхолдеры из списка."""
        for child in self.tools_frame.winfo_children():
            try:
                child.destroy()
            except tk.TclError:
                pass

    def _show_placeholder(self, text: str) -> None:
        """Серая подпись вместо списка (старт / пусто / ошибка)."""
        label = tk.Label(
            self.tools_frame, text=text, bg="#ECECEC", fg="#888888",
            font=FONTS_SMALL, justify="left", wraplength=_WIDTH - 24,
            anchor="w",
        )
        label.pack(side="top", fill="x", pady=(8, 0))

    def _add_tool_card(self, tool: MCPTool) -> None:
        """Добавляет карточку инструмента (клик — модалка с inputSchema)."""
        card = tk.Frame(
            self.tools_frame, bg="#FFFFFF", bd=1, relief="solid",
            cursor="hand2",
        )
        card.pack(side="top", fill="x", padx=2, pady=3)

        name_label = tk.Label(
            card, text=f"▸ {tool.name}", bg="#FFFFFF", fg="#1A1A1A",
            font=FONTS_CARD_NAME, anchor="w", justify="left",
            wraplength=_WIDTH - 40,
        )
        name_label.pack(side="top", fill="x", padx=6, pady=(4, 0))

        desc_text = tool.short_description(limit=_CARD_LIMIT)
        if desc_text:
            desc_label = tk.Label(
                card, text=desc_text, bg="#FFFFFF", fg="#888888",
                font=FONTS_SMALL, anchor="w", justify="left",
                wraplength=_WIDTH - 40,
            )
            desc_label.pack(side="top", fill="x", padx=6, pady=(0, 4))

        # Клик по карточке и по её дочерним меткам — открываем схему
        for widget in (card, name_label, desc_label if desc_text else None):
            if widget is None:
                continue
            widget.bind(
                "<Button-1>",
                lambda _event, t=tool: self.open_schema_dialog(t),
            )

    # ------------------------------------------------------------------
    # Модальное окно inputSchema
    # ------------------------------------------------------------------

    def open_schema_dialog(self, tool: MCPTool) -> None:
        """Модальное окно: имя, описание и JSON inputSchema инструмента.

        Вызывается только из главного потока (обработчик клика).
        """
        dialog = tk.Toplevel(self)
        dialog.title(f"Инструмент: {tool.name}")
        dialog.transient(self.winfo_toplevel())
        dialog.resizable(True, True)

        tk.Label(
            dialog, text=f"Инструмент: {tool.name}", anchor="w",
            justify="left", font=FONTS_HEADER,
        ).pack(side="top", fill="x", padx=10, pady=(10, 2))

        if tool.description:
            tk.Label(
                dialog, text="Описание:", anchor="w", font=FONTS_SMALL,
            ).pack(side="top", fill="x", padx=10)
            tk.Label(
                dialog, text=tool.description, anchor="nw", justify="left",
                wraplength=420, font=FONTS_SMALL, fg="#444444",
            ).pack(side="top", fill="x", padx=10, pady=(0, 6))

        tk.Label(
            dialog, text="Параметры (inputSchema):", anchor="w",
            font=FONTS_SMALL,
        ).pack(side="top", fill="x", padx=10)

        schema_text = tk.Text(
            dialog, height=14, width=56, wrap="none", relief="solid", bd=1,
            font=FONTS_MONO, bg="#FFFFFF", fg="#1A1A1A",
        )
        schema_scroll = ttk.Scrollbar(
            dialog, orient="vertical", command=schema_text.yview
        )
        schema_text.configure(yscrollcommand=schema_scroll.set)
        schema_scroll.pack(side="right", fill="y")
        schema_text.pack(side="top", fill="both", expand=True, padx=10)
        schema_text.insert("1.0", tool.schema_json())
        schema_text.config(state="disabled")  # только чтение

        buttons = tk.Frame(dialog)
        buttons.pack(side="top", fill="x", padx=10, pady=10)

        def _copy_json() -> None:
            dialog.clipboard_clear()
            dialog.clipboard_append(tool.schema_json())

        tk.Button(
            buttons, text="Копировать JSON", command=_copy_json,
            bg="#007AFF", fg="#FFFFFF", activebackground="#0066D6",
            activeforeground="#FFFFFF", relief="flat", cursor="hand2",
        ).pack(side="left")

        tk.Button(
            buttons, text="Закрыть", command=dialog.destroy,
            relief="flat", cursor="hand2",
        ).pack(side="right")

        dialog.geometry("+160+120")
        try:
            dialog.grab_set()
        except tk.TclError:
            pass  # окно ещё не отображено (тесты без update())
        return dialog

    # ------------------------------------------------------------------
    # Сворачивание панели
    # ------------------------------------------------------------------

    def _on_toggle_collapsed(self) -> None:
        """Кнопка ◀/▶: скрыть/показать тело панели."""
        self.set_collapsed(not self.collapsed)

    def set_collapsed(self, collapsed: bool) -> None:
        """Меняет состояние панели и сохраняет его в app_state.

        Args:
            collapsed: True — свернуть (останется заголовок),
                False — развернуть.
        """
        self.collapsed = bool(collapsed)
        self._apply_collapsed_layout()
        if self.app_state is not None:
            self.app_state.set_bool(COLLAPSED_KEY, self.collapsed)

    def apply_collapsed_state(self) -> None:
        """Применяет сохранённое состояние сворачивания после упаковки.

        Вызывается из ui.py СРАЗУ после ``panel.pack(...)``: геометрия
        сворачивания зависит от текущих pack-опций самой панели.
        """
        if self.app_state is not None:
            self.collapsed = self.app_state.get_bool(COLLAPSED_KEY, False)
        self._apply_collapsed_layout()

    def _apply_collapsed_layout(self) -> None:
        """Приводит упаковку панели к текущему состоянию collapsed.

        Свёрнутая панель: тело скрыто, размер — только заголовок
        (высота И ширина, иначе за заголовком останется пустая колонка).
        Развёрнутая: тело на месте, панель снова тянется по высоте чата.
        """
        if self.collapsed:
            self.body.pack_forget()
            self.toggle_btn.config(text="▶")
            self.pack_propagate(False)
            try:
                self.configure(
                    width=max(self.header.winfo_reqwidth(), 80),
                    height=max(self.header.winfo_reqheight(), 24) + 10,
                )
            except tk.TclError:
                return  # панель уже уничтожена
        else:
            self.toggle_btn.config(text="◀")
            self.pack_propagate(True)
            self.body.pack(side="top", fill="both", expand=True, padx=8)
            try:
                self.configure(width=_WIDTH)
            except tk.TclError:
                return

        # Переупаковка самой панели: свёрнутой нужен fill="none" + anchor
        # сверху, развёрнутой — fill="y". Повторный pack() ставит виджет
        # В КОНЕЦ очереди упаковки (панель уехала бы левее скроллбара
        # чата), поэтому позицию восстанавливаем опцией before=<следующий>.
        try:
            pack_info = self.pack_info()
        except tk.TclError:
            return  # панель ещё не упакована — переупаковка не нужна
        if self.collapsed:
            pack_info.update({"fill": "none", "anchor": "n"})
        else:
            pack_info.update({"fill": "y", "anchor": "n"})
        try:
            siblings = self.master.pack_slaves()
            index = siblings.index(self)
            if index + 1 < len(siblings):
                pack_info["before"] = siblings[index + 1]
        except (tk.TclError, ValueError):
            pass  # нет соседей — порядок не важен
        self.pack(**pack_info)

    # ------------------------------------------------------------------
    # Скроллинг списка
    # ------------------------------------------------------------------

    def _on_tools_frame_configure(self, _event=None) -> None:
        """Содержимое списка изменилось — пересчитать scrollregion."""
        self.canvas.configure(
            scrollregion=self.canvas.bbox("all")
        )

    def _on_canvas_configure(self, event) -> None:
        """Карточки растягиваются на ширину канвы."""
        self.canvas.itemconfigure(self._canvas_window, width=event.width)

    def _on_mouse_wheel(self, event) -> None:
        """Прокрутка колесом мыши, только когда курсор над панелью."""
        try:
            under = self.winfo_containing(event.x_root, event.y_root)
            if under is None:
                return
            path = str(under)
            if path == str(self) or path.startswith(str(self) + "."):
                self.canvas.yview_scroll(-1 * (event.delta // 120), "units")
        except (tk.TclError, AttributeError):
            pass

    def destroy(self) -> None:
        """Уничтожение панели: снять привязку колеса и отложенные задачи."""
        if self._poll_job is not None:
            try:
                self.after_cancel(self._poll_job)
            except tk.TclError:
                pass
            self._poll_job = None
        try:
            self.unbind_all("<MouseWheel>")
        except tk.TclError:
            pass
        super().destroy()


# Шрифты панели (отдельный словарь, чтобы не трогать FONTS в ui.py).
FONTS_DOT = ("Segoe UI", 11)
FONTS_HEADER = ("Segoe UI", 10, "bold")
FONTS_SMALL = ("Segoe UI", 8)
FONTS_CARD_NAME = ("Segoe UI", 9, "bold")
FONTS_MONO = ("Consolas", 9)


def _plural_tools(count: int) -> str:
    """Русская форма слова «инструмент» для числа ``count``.

    1 инструмент / 2 инструмента / 5 инструментов / 11 инструментов.
    """
    if count % 10 == 1 and count % 100 != 11:
        return "инструмент"
    if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        return "инструмента"
    return "инструментов"
