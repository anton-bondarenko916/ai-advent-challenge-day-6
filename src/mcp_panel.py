"""Панель инструментов MCP рядом с областью чата (Tkinter).

MCPPanel — сворачиваемый Frame, который встраивается в правую часть
вкладки «Чат»: индикатор состояния, кнопка «Обновить», скроллируемый
список карточек инструментов, сгруппированных по серверам (заголовок
«● имя сервера» + карточки его инструментов). Источники: stdio-сервер
(команда MCP_SERVER_COMMAND, обычно Git) и HTTP-серверы (напоминания,
news, geo, weather — http_sources). Карточка имеет кнопки:

* «Вызвать» — открывает диалог вызова (open_call_dialog): поля
  параметров строятся по inputSchema инструмента, результат выводится
  в то же окно после выполнения в фоновом потоке; вызов уходит на
  сервер, которому принадлежит инструмент (HTTP — через
  MCPRemoteCallWorker, stdio — через MCPToolCallWorker);
* «Параметры» — модальное окно с описанием и inputSchema (JSON);
  туда же ведёт клик по карточке.

Сервер, не ответивший на tools/list, не скрывает остальные: его группа
помечается красным с текстом ошибки (как в agentic loop — недоступный
сервер не ломает диалог).

Потокобезопасность: и запрос списка, и вызов инструмента выполняют
воркеры в отдельных daemon-потоках (MultiServerListWorker /
MCPToolCallWorker / MCPRemoteCallWorker); их callback'и кладут
результат в queue.Queue, а главный поток опрашивает очередь через
``root.after`` — тот же паттерн, что у задач (FSM) в ui.py, поэтому UI
никогда не блокируется и не обновляется из чужого потока.

Состояние сворачивания сохраняется в SQLite (AppStateStore, таблица
app_state) и восстанавливается при следующем запуске.
"""

from __future__ import annotations

import queue
from typing import Any, Callable, Dict, List, Optional

import tkinter as tk
from tkinter import messagebox, ttk

from .app_state import AppStateStore
from .config import PROJECT_ROOT
from .mcp_models import MCPTool, ServerTools
from .mcp_worker import (
    MCPRemoteCallWorker,
    MCPToolCallWorker,
    MultiServerListWorker,
)

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
        http_sources: HTTP-серверы для панели — список
            ``{"name", "url"}`` (напоминания, news, geo, weather).
        app_state: хранилище состояния UI; None — сворачивание не
            сохраняется.
        tools: последний полученный список инструментов (плоский список
            всех серверов).
        tool_sources: маршрут вызовов «имя инструмента → url сервера»
            ("" — stdio-сервер панели).
        collapsed: свёрнута ли панель (остаётся только заголовок).
    """

    def __init__(
        self,
        master: tk.Misc,
        settings: Dict[str, Any],
        app_state: Optional[AppStateStore] = None,
        worker_factory: Optional[Callable[[Any, Any], MultiServerListWorker]] = None,
        call_worker_factory: Optional[Callable[..., Any]] = None,
        http_sources: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Строит панель.

        Args:
            master: родительский виджет (обёртка вкладки «Чат»).
            settings: словарь настроек MCP (get_mcp_settings).
            app_state: хранилище состояния UI (сворачивание панели).
            worker_factory: фабрика воркера списка ``f(on_success, on_error)``;
                None — стандартный MultiServerListWorker (stdio + http_sources).
                Для тестов подаётся фейк.
            call_worker_factory: фабрика воркера вызова
                ``f(tool_name, arguments, on_success, on_error)``
                (для инструментов HTTP-серверов — с kwarg ``source_url``);
                None — стандартные MCPToolCallWorker/MCPRemoteCallWorker.
            http_sources: HTTP-серверы — список ``{"name", "url"}``.
        """
        super().__init__(master, bg="#ECECEC", width=_WIDTH)
        self.settings = settings
        self.http_sources: List[Dict[str, Any]] = list(http_sources or [])
        self.app_state = app_state
        self.tools: List[MCPTool] = []
        self.tool_sources: Dict[str, str] = {}
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

        if call_worker_factory is not None:
            self._make_call_worker = call_worker_factory
        else:
            self._make_call_worker = self._default_call_worker_factory

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

    def _default_worker_factory(self, on_success, on_error) -> MultiServerListWorker:
        """Стандартная фабрика: опрос stdio-сервера и всех HTTP-серверов."""
        return MultiServerListWorker(
            on_success=on_success,
            on_error=on_error,
            server_command=self.settings.get("server_command"),
            timeout=self.settings.get("timeout"),
            log_path=str(self.settings.get("log_path") or ""),
            http_sources=self.http_sources,
        )

    def _default_call_worker_factory(
        self, tool_name, arguments, on_success, on_error, source_url: str = "",
    ):
        """Фабрика вызова: HTTP-сервер (source_url) или stdio панели."""
        if source_url:
            return MCPRemoteCallWorker(
                tool_name=tool_name,
                arguments=arguments,
                on_success=on_success,
                on_error=on_error,
                server_url=source_url,
            )
        return MCPToolCallWorker(
            tool_name=tool_name,
            arguments=arguments,
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

    def _on_tools_loaded(self, result: Any) -> None:
        """Успешный ответ: перерисовать список по группам серверов.

        Принимает оба формата воркера: список групп ServerTools
        (MultiServerListWorker) или плоский список MCPTool — ответы
        одиночного сервера (старые воркеры и тесты) рисуются как одна
        группа без заголовка.
        """
        self._busy = False
        self.refresh_btn.config(state="normal")
        groups = self._normalize_groups(result)
        self.tools = [tool for group in groups for tool in group.tools]
        self.tool_sources = {
            tool.name: group.url for group in groups for tool in group.tools
        }
        total = len(self.tools)
        failed = [group for group in groups if group.error]
        if total:
            status = f"Подключён | {total} {_plural_tools(total)}"
            if failed:
                status += " · нет связи: " + ", ".join(
                    group.name or "сервер" for group in failed
                )
            self._set_status(STATUS_CONNECTED, status)
        elif failed:
            short = failed[0].error
            if len(short) > 120:
                short = short[:117] + "..."
            self._set_status(STATUS_ERROR, f"Ошибка: {short}")
        else:
            self._set_status(
                STATUS_CONNECTED,
                "Подключён | 0 " + _plural_tools(0),
            )
        self._clear_tools()
        if not groups:
            self._show_placeholder(
                "Серверы не настроены.\nПроверьте MCP_ENABLED и enabled у "
                "HTTP-серверов."
            )
            return
        if not self.tools and not failed:
            # Пустой список — НЕ ошибка.
            self._show_placeholder(
                "Серверы ответили, но инструментов не вернули."
            )
            return
        for group in groups:
            self._render_group(group)

    @staticmethod
    def _normalize_groups(result: Any) -> List[ServerTools]:
        """Приводит ответ воркера к списку групп ServerTools.

        Args:
            result: список ServerTools (многосерверный воркер) либо
                список MCPTool (одиночный сервер — обратная совместимость).

        Returns:
            Список групп; плоский список инструментов становится одной
            безымянной группой (без заголовка в панели).
        """
        if isinstance(result, ServerTools):
            return [result]
        if result and all(isinstance(item, MCPTool) for item in result):
            return [ServerTools(name="", tools=list(result))]
        return [item for item in result if isinstance(item, ServerTools)]

    def _render_group(self, group: ServerTools) -> None:
        """Рисует одну группу: заголовок сервера, ошибка, карточки.

        Args:
            group: группа ServerTools (имя, инструменты, ошибка, url).
        """
        if group.name:
            color = "#C62828" if group.error else "#2E7D32"
            header = tk.Label(
                self.tools_frame,
                text=f"● {group.name}",
                bg="#ECECEC", fg=color,
                font=FONTS_SERVER_NAME, anchor="w",
            )
            header.pack(side="top", fill="x", padx=2, pady=(6, 0))
        if group.error and not group.tools:
            error_label = tk.Label(
                self.tools_frame,
                text=_shorten(group.error, _CARD_LIMIT),
                bg="#ECECEC", fg="#C62828",
                font=FONTS_SMALL, anchor="w", justify="left",
                wraplength=_WIDTH - 24,
            )
            error_label.pack(side="top", fill="x", padx=6)
        for tool in group.tools:
            self._add_tool_card(tool, source_url=group.url)

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

    def _add_tool_card(self, tool: MCPTool, source_url: str = "") -> None:
        """Добавляет карточку инструмента с кнопками «Вызвать»/«Параметры».

        Клик по карточке (или меткам) открывает модалку с inputSchema —
        как и кнопка «Параметры»; кнопка «Вызвать» открывает диалог
        вызова инструмента (на сервере ``source_url``, "" — stdio).

        Args:
            tool: инструмент для карточки.
            source_url: MCP-эндпоинт сервера инструмента; "" — stdio.
        """
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

        # Кнопки карточки. Собственные виджеты Button: клик по ним НЕ
        # попадает в binding карточки (parent не входит в bindtags
        # кнопки), поэтому «Вызвать» не открывает заодно модалку схемы.
        buttons = tk.Frame(card, bg="#FFFFFF")
        buttons.pack(side="top", fill="x", padx=6, pady=(0, 6))

        tk.Button(
            buttons, text="▶ Вызвать",
            command=lambda t=tool, u=source_url: self.open_call_dialog(t, u),
            bg="#007AFF", fg="#FFFFFF", activebackground="#0066D6",
            activeforeground="#FFFFFF", relief="flat", cursor="hand2",
            font=FONTS_SMALL,
        ).pack(side="left")

        tk.Button(
            buttons, text="Параметры",
            command=lambda t=tool: self.open_schema_dialog(t),
            bg="#FFFFFF", fg="#444444", activebackground="#E8E8E8",
            relief="flat", cursor="hand2", font=FONTS_SMALL,
        ).pack(side="left", padx=(6, 0))

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
    # Диалог вызова инструмента
    # ------------------------------------------------------------------

    def open_call_dialog(self, tool: MCPTool, source_url: str = "") -> "_ToolCallDialog":
        """Открывает диалог вызова инструмента (кнопка «Вызвать»).

        Форма параметров строится по inputSchema; вызов выполняется в
        отдельном потоке, результат появляется в том же окне.

        Args:
            tool: инструмент для вызова.
            source_url: MCP-эндпоинт сервера инструмента; "" — stdio.

        Returns:
            Созданное окно (_ToolCallDialog) — для тестов и вызывателя.
        """
        return _ToolCallDialog(self, tool, source_url)

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


class _ToolCallDialog(tk.Toplevel):
    """Модальное окно вызова инструмента MCP (кнопка «Вызвать»).

    Поля формы строятся по inputSchema инструмента (properties,
    required, default): string — произвольный текст, integer/number —
    с проверкой числа, boolean — true/false. Пустые необязательные
    поля на сервер НЕ передаются (там применится значение по умолчанию).

    Вызов выполняет воркер панели (по умолчанию MCPToolCallWorker) в
    отдельном daemon-потоке; результат доставляется через собственную
    queue.Queue и опрашивается ``after``-циклом окна — UI не блокируется.
    Окно можно закрыть во время выполнения: after-задача отменяется,
    callback воркера уйдёт в очередь, которую больше никто не читает
    (это безопасно, поток daemon).
    """

    def __init__(self, panel: MCPPanel, tool: MCPTool,
                 source_url: str = "") -> None:
        """Строит окно вызова.

        Args:
            panel: панель MCP (источник настроек и фабрики воркера).
            tool: вызываемый инструмент.
            source_url: MCP-эндпоинт сервера инструмента; "" — stdio
                сервер панели.
        """
        super().__init__(panel)
        self.panel = panel
        self.tool = tool
        self.source_url = str(source_url or "")
        self._busy = False
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._poll_job: Optional[str] = None
        self._fields: List[Dict[str, Any]] = []

        self.title(f"Вызов: {tool.name}")
        self.transient(panel.winfo_toplevel())
        self.resizable(True, True)
        self.geometry("+150+110")
        self._build_widgets()
        try:
            self.grab_set()
        except tk.TclError:
            pass  # окно ещё не отображено (тесты без update())

    # ------------------------------------------------------------------
    # Построение виджетов
    # ------------------------------------------------------------------

    def _build_widgets(self) -> None:
        """Заголовок, форма параметров, статус, кнопки, поле результата."""
        tk.Label(
            self, text=f"Инструмент: {self.tool.name}", anchor="w",
            justify="left", font=FONTS_HEADER,
        ).pack(side="top", fill="x", padx=10, pady=(10, 2))

        fields_frame = tk.Frame(self)
        fields_frame.pack(side="top", fill="x", padx=10)

        schema = self.tool.input_schema or {}
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])

        if not properties:
            tk.Label(
                fields_frame, text="Инструмент без параметров.",
                fg="#888888", font=FONTS_SMALL, anchor="w",
            ).pack(side="top", fill="x", pady=2)

        for name, spec in properties.items():
            spec = spec if isinstance(spec, dict) else {}
            ptype = str(spec.get("type") or "string")
            is_required = name in required
            row = tk.Frame(fields_frame)
            row.pack(side="top", fill="x", pady=2)
            tk.Label(
                row, text=self._field_label(name, ptype, is_required),
                font=FONTS_SMALL, anchor="w", justify="left", width=30,
            ).pack(side="left")
            var = tk.StringVar(value=self._prefill(name, ptype, spec))
            entry = tk.Entry(
                row, textvariable=var, font=FONTS_MONO, width=34,
            )
            entry.pack(side="left", fill="x", expand=True)
            self._fields.append({
                "name": name,
                "type": ptype,
                "required": is_required,
                "var": var,
                "entry": entry,
            })

        self.status_label = tk.Label(
            self, text="Заполните параметры и нажмите «Выполнить».",
            fg="#666666", font=FONTS_SMALL, anchor="w", justify="left",
            wraplength=440,
        )
        self.status_label.pack(side="top", fill="x", padx=10, pady=(6, 2))

        buttons = tk.Frame(self)
        buttons.pack(side="top", fill="x", padx=10, pady=(2, 6))

        self.run_btn = tk.Button(
            buttons, text="Выполнить", command=self._execute,
            bg="#007AFF", fg="#FFFFFF", activebackground="#0066D6",
            activeforeground="#FFFFFF", relief="flat", cursor="hand2",
            font=FONTS_SMALL,
        )
        self.run_btn.pack(side="left")

        self.copy_btn = tk.Button(
            buttons, text="Копировать результат", command=self._copy_result,
            state="disabled", relief="flat", cursor="hand2",
            font=FONTS_SMALL,
        )
        self.copy_btn.pack(side="left", padx=(6, 0))

        tk.Button(
            buttons, text="Закрыть", command=self.destroy,
            relief="flat", cursor="hand2", font=FONTS_SMALL,
        ).pack(side="right")

        tk.Label(
            self, text="Результат:", font=FONTS_SMALL, anchor="w",
        ).pack(side="top", fill="x", padx=10)

        result_wrap = tk.Frame(self)
        result_wrap.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 10))

        self.result_text = tk.Text(
            result_wrap, height=14, width=56, wrap="none", relief="solid",
            bd=1, font=FONTS_MONO, bg="#FFFFFF", state="disabled",
        )
        result_scroll = ttk.Scrollbar(
            result_wrap, orient="vertical", command=self.result_text.yview
        )
        self.result_text.configure(yscrollcommand=result_scroll.set)
        result_scroll.pack(side="right", fill="y")
        self.result_text.pack(side="left", fill="both", expand=True)

    @staticmethod
    def _field_label(name: str, ptype: str, is_required: bool) -> str:
        """Подпись поля: имя, тип и звезда у обязательных."""
        return f"{name}{' *' if is_required else ''} ({ptype})"

    def _prefill(self, name: str, ptype: str, spec: Dict[str, Any]) -> str:
        """Начальное значение поля.

        ``repo_path`` подставляется из настроек (MCP_GIT_REPO_PATH,
        по умолчанию — корень проекта), остальные — из ``default``
        схемы, иначе пустая строка.
        """
        if name == "repo_path" and ptype == "string":
            return str(self.panel.settings.get("git_repo_path") or PROJECT_ROOT)
        default = spec.get("default")
        return "" if default is None else str(default)

    # ------------------------------------------------------------------
    # Сбор аргументов и запуск
    # ------------------------------------------------------------------

    def _collect_arguments(self) -> Optional[Dict[str, Any]]:
        """Собирает и валидирует аргументы из формы.

        Returns:
            Словарь аргументов либо None (ошибка валидации — предупреждение
            уже показано).
        """
        args: Dict[str, Any] = {}
        for field in self._fields:
            raw = field["var"].get().strip()
            name, ptype = field["name"], field["type"]
            if field["required"] and not raw:
                self._warn(f"Заполните обязательное поле: {name}")
                field["entry"].focus_set()
                return None
            if not raw:
                continue  # необязательное пустое поле — не передаём
            try:
                args[name] = _parse_argument(raw, ptype)
            except ValueError as exc:
                self._warn(f"Некорректное значение поля {name}: {exc}")
                field["entry"].focus_set()
                return None
        return args

    def _execute(self) -> None:
        """«Выполнить»: валидация → воркер в отдельном потоке → опрос."""
        if self._busy:
            return
        arguments = self._collect_arguments()
        if arguments is None:
            return
        self._busy = True
        self.run_btn.config(state="disabled")
        self._set_status("Выполняется...", "#F9A825")
        if self.source_url:
            # Инструмент HTTP-сервера (geo/weather/напоминания/news):
            # вызов уходит на его сервер
            worker = self.panel._make_call_worker(
                self.tool.name,
                arguments,
                on_success=lambda text: self._queue.put({"ok": text}),
                on_error=lambda message: self._queue.put({"error": message}),
                source_url=self.source_url,
            )
        else:
            worker = self.panel._make_call_worker(
                self.tool.name,
                arguments,
                on_success=lambda text: self._queue.put({"ok": text}),
                on_error=lambda message: self._queue.put({"error": message}),
            )
        worker.start()
        self._poll_job = self.after(100, self._poll)

    def _poll(self) -> None:
        """Опрашивает очередь результата воркера (главный поток)."""
        self._poll_job = None
        try:
            while True:
                item = self._queue.get_nowait()
                self._busy = False
                try:
                    self.run_btn.config(state="normal")
                except tk.TclError:
                    pass
                if "ok" in item:
                    text = str(item["ok"])
                    self._show_result(text)
                    self._set_status(
                        f"Готово ({len(text)} символов).", "#2E7D32"
                    )
                elif "error" in item:
                    self._set_status(f"Ошибка: {item['error']}", "#C62828")
                    try:
                        messagebox.showerror(
                            "Ошибка вызова", item["error"], parent=self
                        )
                    except tk.TclError:
                        pass  # окно приложения закрывается
        except queue.Empty:
            pass
        finally:
            if self._busy:
                self._poll_job = self.after(100, self._poll)

    # ------------------------------------------------------------------
    # Отображение результата и служебные
    # ------------------------------------------------------------------

    def _show_result(self, text: str) -> None:
        """Кладёт текст в поле результата и включает кнопку копирования."""
        try:
            self.result_text.config(state="normal")
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", text)
            self.result_text.config(state="disabled")
            self.copy_btn.config(state="normal")
        except tk.TclError:
            pass  # окно уже закрыто

    def _copy_result(self) -> None:
        """Копирует текст результата в буфер обмена."""
        try:
            text = self.result_text.get("1.0", "end-1c")
            self.clipboard_clear()
            self.clipboard_append(text)
        except tk.TclError:
            pass

    def _set_status(self, text: str, color: str) -> None:
        """Обновляет строку статуса (цвет: жёлтый — идёт, зелёный — ок)."""
        try:
            self.status_label.config(text=text, fg=color)
        except tk.TclError:
            pass

    def _warn(self, message: str) -> None:
        """Предупреждение валидации формы (не прерывает приложение)."""
        try:
            messagebox.showwarning(
                "Вызов инструмента", message, parent=self
            )
        except tk.TclError:
            pass

    def get_result_text(self) -> str:
        """Текст поля результата (для тестов)."""
        return self.result_text.get("1.0", "end-1c")

    def destroy(self) -> None:
        """Уничтожение окна: снять отложенный опрос очереди."""
        if self._poll_job is not None:
            try:
                self.after_cancel(self._poll_job)
            except tk.TclError:
                pass
            self._poll_job = None
        super().destroy()


def _parse_argument(raw: str, ptype: str) -> Any:
    """Преобразует строку поля к типу из inputSchema.

    Args:
        raw: непустое значение поля (без краевых пробелов).
        ptype: значение ``type`` из схемы (string/integer/number/boolean);
            неизвестный тип трактуется как строка.

    Returns:
        Значение соответствующего типа.

    Raises:
        ValueError: значение не приводится к типу (текст ошибки —
            для показа пользователю).
    """
    kind = ptype.lower()
    if kind in ("integer", "int"):
        return int(raw)
    if kind == "number":
        return float(raw)
    if kind in ("boolean", "bool"):
        low = raw.lower()
        if low in ("1", "true", "yes", "да"):
            return True
        if low in ("0", "false", "no", "нет"):
            return False
        raise ValueError(f"ожидалось true/false, получено: {raw!r}")
    return raw


def _shorten(text: str, limit: int) -> str:
    """Обрезает текст до ``limit`` символов с многоточием (для группы)."""
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


# Шрифты панели (отдельный словарь, чтобы не трогать FONTS в ui.py).
FONTS_DOT = ("Segoe UI", 11)
FONTS_HEADER = ("Segoe UI", 10, "bold")
FONTS_SERVER_NAME = ("Segoe UI", 9, "bold")
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
