"""Вкладка «🎓 Обучение» (ТЗ ai_1_36): мок-вход, дашборд, план, подтема.

Экраны (перерисовка одного Column — без вложенной навигации Flet):

* вход по логину/паролю (3 мок-пользователя, SHA-256 в БД);
* дашборд: приветствие, уровень, streak, сводка прогресса и 5 тем
  курса (контент темы генерируется ЛЕНИВО при первом открытии);
* экран темы: описание, подтемы со статусами, PDF-конспект;
* экран подтемы: теория, задание, проверка через fs MCP (путь к папке
  проекта + фидбек AI с оценкой), ручная отметка «выполнено».

Структура — как у остальных вкладок: без обращения к page в
конструкторе (офлайн-тесты с заглушкой app), фоновые задачи — только
корутинные ФУНКЦИИ через page.run_task (гард components.guarded_task),
ошибки не «роняют» интерфейс.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import flet as ft

from app.ui import components

# Ключ app_state: последний путь к проекту пользователя (переживает
# перезапуск — проверять задание удобно в ту же папку)
LEARNING_PROJECT_PATH_KEY = "learning_project_path"

# Значки статусов тем/подтем (единственный источник — для UI и тестов)
STATUS_ICONS = {
    "not_generated": "🆕",
    "not_started": "▶",
    "in_progress": "◐",
    "completed": "✅",
}
STATUS_LABELS = {
    "not_generated": "не сгенерирована (откройте — создам план)",
    "not_started": "не начата",
    "in_progress": "в процессе",
    "completed": "завершена",
}

DEMO_USERS_HINT = ("Демо-пользователи: ivan / ivan123, maria / maria123, "
                   "alex / alex123")


def topic_badge(topic: Dict[str, Any]) -> str:
    """Строка статуса темы для карточки дашборда."""
    status = str(topic.get("display_status") or "not_generated")
    total = int(topic.get("subtopics_total") or 0)
    completed = int(topic.get("subtopics_completed") or 0)
    progress = f" ({completed}/{total})" if total else ""
    return (f"{STATUS_ICONS.get(status, '•')} "
            f"{STATUS_LABELS.get(status, status)}{progress}")


class LearningView:
    """Вкладка платформы обучения.

    Attributes:
        app: владелец (page, notify, learning_* сервисы, file_picker,
            app_state); в офлайн-тестах — заглушка.
        user: сессия — словарь пользователя после входа (user_id).
    """

    def __init__(self, app: Any) -> None:
        self.app = app
        self.user: Optional[Dict[str, Any]] = None
        self.current_topic_id: Optional[int] = None
        self.current_subtopic_id: Optional[int] = None

        self.username_field = ft.TextField(
            label="Логин", autofocus=True, on_submit=self._on_login)
        self.password_field = ft.TextField(
            label="Пароль", password=True, can_reveal_password=True,
            on_submit=self._on_login)
        self.login_error = ft.Text("", color=ft.Colors.RED_600, size=13)
        self.login_button = ft.FilledButton(
            "Войти", icon=ft.Icons.LOGIN, on_click=self._on_login)

        self.status_text = ft.Text("", size=13, selectable=True,
                                   expand=True)

        self.screen = ft.Column(
            [], expand=True, scroll=ft.ScrollMode.AUTO, spacing=10)

    # ------------------------------------------------------------------
    # Компоновка и навигация
    # ------------------------------------------------------------------
    def build(self) -> ft.Container:
        """Корень вкладки (перерисовывается целиком при навигации)."""
        self._show_login()
        return ft.Container(content=self.screen, expand=True,
                            padding=ft.Padding(0, 8, 0, 0))

    def on_show(self) -> None:
        """Возврат на вкладку: гость видит вход, пользователь — дашборд."""
        if self.user is not None and self.app.page is not None:
            self._schedule(self._dashboard_async)

    def _render(self, controls: List[ft.Control]) -> None:
        self.screen.controls = list(controls)
        self.update()

    def update(self) -> None:
        if self.app.page is not None:
            self.app.page.update()

    def _schedule(self, handler, *args) -> None:
        """КОРУТИННАЯ ФУНКЦИЯ в page.run_task (см. project_docs_view);
        без page (офлайн-тесты) — no-op."""
        if self.app.page is None:
            return
        self.app.page.run_task(
            components.guarded_task(self.app, handler), *args)

    def _repo(self) -> Any:
        return getattr(self.app, "learning_repo", None)

    def _notify(self, message: str) -> None:
        try:
            self.app.notify(message)
        except Exception:  # noqa: BLE001 — уведомление не критично
            pass

    # ------------------------------------------------------------------
    # Экран 1: вход
    # ------------------------------------------------------------------
    def _show_login(self, error: str = "") -> None:
        self.login_error.value = error
        self._render([
            ft.Container(
                ft.Column([
                    ft.Text("🎓 Android Learning Assistant",
                            size=28, weight=ft.FontWeight.BOLD),
                    ft.Text("Платформа обучения Android-разработке",
                            size=14, color=ft.Colors.BLUE_GREY_500),
                    self.username_field,
                    self.password_field,
                    self.login_error,
                    self.login_button,
                    ft.Text(DEMO_USERS_HINT, size=12,
                            color=ft.Colors.BLUE_GREY_400),
                ], spacing=10, tight=True),
                width=420, padding=20,
                border=ft.Border.all(1, ft.Colors.BLUE_GREY_200),
                border_radius=12,
            ),
        ])

    def _on_login(self, event=None) -> None:
        self._schedule(self._login_async)

    async def _login_async(self) -> None:
        """Проверка логина/пароля (SHA-256) и переход на дашборд."""
        repo = self._repo()
        if repo is None:
            self._show_login("Модуль обучения недоступен (нет БД)")
            return
        username = str(self.username_field.value or "").strip()
        user = repo.authenticate(username,
                                 str(self.password_field.value or ""))
        if user is None:
            self._show_login("Неверный логин или пароль")
            return
        self.user = user
        self._notify(f"Добро пожаловать, {user.get('display_name')}!")
        await self._dashboard_async()

    def _logout(self, event=None) -> None:
        self.user = None
        self.password_field.value = ""
        self._show_login()

    # ------------------------------------------------------------------
    # Экран 2: дашборд (прогресс, streak, 5 тем)
    # ------------------------------------------------------------------
    async def _dashboard_async(self) -> None:
        plan = getattr(self.app, "learning_plan", None)
        repo = self._repo()
        if plan is None or repo is None or self.user is None:
            return
        topics = plan.ensure_topics(self.user["id"])
        summary = repo.progress_summary(self.user["id"])
        streak = repo.streak_days(self.user["id"])
        self._show_dashboard(topics, summary, streak)

    def _show_dashboard(self, topics: List[Dict[str, Any]],
                        summary: Dict[str, int],
                        streak: int) -> None:
        user = self.user or {}
        header = ft.Row([
            ft.Column([
                ft.Text(
                    f"Привет, {user.get('display_name') or 'ученик'}! "
                    f"(уровень: {user.get('level') or 'beginner'})",
                    size=20, weight=ft.FontWeight.BOLD),
                ft.Text(
                    f"🔥 Streak: {streak} дн. · Прогресс: "
                    f"{summary.get('completed', 0)} из "
                    f"{summary.get('total', 0)} подтем завершено",
                    size=13, color=ft.Colors.BLUE_GREY_600),
            ], spacing=2, expand=True),
            ft.Row([
                ft.IconButton(ft.Icons.REFRESH, tooltip="Обновить",
                              on_click=self._on_refresh),
                ft.FilledTonalButton("Выйти", icon=ft.Icons.LOGOUT,
                                     on_click=self._logout),
            ], spacing=4, tight=True),
        ], spacing=8)

        cards: List[ft.Control] = []
        for topic in topics:
            cards.append(ft.Container(
                content=ft.Row([
                    ft.FilledButton(
                        f"{topic['order_index'] + 1}. {topic['title']}",
                        on_click=self._make_topic_opener(topic["id"]),
                        expand=True),
                    ft.Text(topic_badge(topic), size=12,
                            color=ft.Colors.BLUE_GREY_600, no_wrap=True),
                ], spacing=8),
                padding=6,
                border=ft.Border.all(1, ft.Colors.BLUE_GREY_100),
                border_radius=8,
            ))
        cards.append(self.status_text)
        self._render([header, ft.Divider()] + cards)

    def _make_topic_opener(self, topic_id: int):
        """Замыкание-обработчик открытия темы (нужен id в цикле)."""
        def handler(event=None) -> None:
            self._schedule(self._open_topic_async, topic_id)
        return handler

    def _on_refresh(self, event=None) -> None:
        self._schedule(self._dashboard_async)

    # ------------------------------------------------------------------
    # Экран 3: тема (ленивая генерация, подтемы, PDF)
    # ------------------------------------------------------------------
    async def _open_topic_async(self, topic_id: int) -> None:
        plan = getattr(self.app, "learning_plan", None)
        if plan is None or self.user is None:
            return
        self._set_busy("Готовлю тему (при первом открытии — генерация "
                       "плана через LLM)…")
        self.update()
        try:
            topic = await plan.get_or_generate_topic(self.user["id"],
                                                     topic_id)
        except Exception as exc:  # noqa: BLE001 — гард покажет SnackBar,
            # но интерфейс надо вернуть на дашборд
            self.status_text.value = ""
            await self._dashboard_async()
            self._notify(f"⚠️ Не удалось открыть тему: {exc}")
            return
        self.current_topic_id = topic_id
        self._show_topic(topic)
        if topic.get("reminder_status"):
            self._notify(f"⏰ {topic['reminder_status']}")

    def _show_topic(self, topic: Dict[str, Any]) -> None:
        rows: List[ft.Control] = [
            ft.Row([
                ft.TextButton("← К списку тем", on_click=self._on_refresh),
                ft.Container(expand=True),
                ft.FilledButton(
                    "📥 PDF-конспект темы",
                    icon=ft.Icons.PICTURE_AS_PDF,
                    on_click=self._on_pdf,
                    tooltip="HTML+CSS → PDF (WeasyPrint, локально)"),
            ], spacing=8),
            ft.Text(topic["title"], size=20, weight=ft.FontWeight.BOLD),
            components.markdown_text(
                str(topic.get("description") or "").strip()
                or "(описание отсутствует)"),
            ft.Divider(),
        ]
        for sub in topic.get("subtopics") or []:
            mark = "✅" if sub.get("is_completed") else "⬜"
            rows.append(ft.Container(
                content=ft.Row([
                    ft.FilledTonalButton(
                        f"{mark} {sub['title']}",
                        on_click=self._make_subtopic_opener(sub["id"]),
                        expand=True),
                ], spacing=8),
                padding=ft.Padding(4, 2, 4, 2),
            ))
        rows.append(self.status_text)
        self._render(rows)

    def _make_subtopic_opener(self, subtopic_id: int):
        def handler(event=None) -> None:
            self._schedule(self._open_subtopic_async, subtopic_id)
        return handler

    async def _open_subtopic_async(self, subtopic_id: int) -> None:
        repo = self._repo()
        if repo is None or self.user is None:
            return
        subtopic = repo.get_subtopic_for_user(self.user["id"], subtopic_id)
        if subtopic is None:
            self._notify("⚠️ Подтема не найдена")
            return
        self.current_subtopic_id = subtopic_id
        self._show_subtopic(subtopic)

    # ------------------------------------------------------------------
    # Экран 4: подтема (теория, задание, проверка, отметка)
    # ------------------------------------------------------------------
    def _project_path_field(self, on_pick) -> ft.Row:
        """Поле «путь к проекту» + кнопка выбора папки (FilePicker)."""
        field = ft.TextField(
            label="Путь к папке с проектом",
            value=self._saved_project_path(),
            hint_text="Например: C:/projects/MyFirstApp",
            expand=True, on_submit=self._on_review)
        self._path_field = field
        return ft.Row([
            field,
            ft.IconButton(ft.Icons.FOLDER_OPEN,
                          tooltip="Выбрать папку проекта",
                          on_click=on_pick),
        ], spacing=4)

    def _saved_project_path(self) -> str:
        store = getattr(self.app, "app_state", None)
        saved = store.get(LEARNING_PROJECT_PATH_KEY) \
            if store is not None else None
        return str(saved or "")

    def _show_subtopic(self, subtopic: Dict[str, Any]) -> None:
        score = subtopic.get("score")
        completed = bool(subtopic.get("is_completed"))
        feedback = str(subtopic.get("ai_feedback") or "").strip()

        rows: List[ft.Control] = [
            ft.TextButton("← К теме", on_click=self._on_back_to_topic),
            ft.Text(subtopic["title"], size=20, weight=ft.FontWeight.BOLD),
            ft.Text(f"Тема: {subtopic.get('topic_title') or ''}",
                    size=13, color=ft.Colors.BLUE_GREY_500),
            ft.Text("📖 Теория", size=15, weight=ft.FontWeight.BOLD),
            components.markdown_text(
                str(subtopic.get("theory") or "(теория отсутствует)")),
            # Фон контейнеров фиксированно светлый — текст внутри явно
            # тёмный (в тёмной теме цвет темы даёт «светлое на светлом»,
            # урок контраста ai_1_21)
            ft.Container(
                content=ft.Column([
                    ft.Text("📋 Задание", size=15,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.ORANGE_900),
                    components.markdown_text(
                        str(subtopic.get("task") or ""),
                        ft.Colors.GREY_900),
                    ft.Text(
                        "🎯 Критерии оценки: "
                        + str(subtopic.get("rubric") or ""),
                        size=12, color=ft.Colors.BLUE_GREY_700,
                        selectable=True),
                ], spacing=6, tight=True),
                bgcolor=ft.Colors.ORANGE_50,
                padding=12, border_radius=8,
            ),
            ft.Divider(),
            ft.Text("🔎 Проверка задания", size=15,
                    weight=ft.FontWeight.BOLD),
            ft.Text(
                "AI прочитает файлы проекта через fs-сервер (MCP) и "
                "оценит выполнение по критериям. Укажите папку "
                "Android-проекта с выполненным заданием.",
                size=12, color=ft.Colors.BLUE_GREY_600),
        ]

        self.review_button = ft.FilledButton(
            "Проверить задание", icon=ft.Icons.FACT_CHECK,
            on_click=self._on_review)
        rows.append(self._project_path_field(self._on_pick_folder))
        rows.append(self.review_button)
        rows.append(self.status_text)

        if feedback:
            rows.extend([
                ft.Container(
                    content=ft.Column([
                        ft.Text(
                            f"🤖 Фидбек AI — оценка: "
                            f"{score if score is not None else '?'}/5"
                            + (" · подтема завершена ✅" if completed
                               else ""),
                            size=14, weight=ft.FontWeight.BOLD,
                            color=ft.Colors.GREEN_900),
                        components.markdown_text(
                            feedback, ft.Colors.GREY_900),
                    ], spacing=6, tight=True),
                    bgcolor=ft.Colors.GREEN_50,
                    padding=12, border_radius=8,
                ),
            ])
        toggle_label = ("↩️ Снять отметку «выполнено»" if completed
                        else "✅ Отметить выполненной")
        rows.append(ft.FilledTonalButton(
            toggle_label, on_click=self._on_toggle_completed))
        self._render(rows)

    def _on_back_to_topic(self, event=None) -> None:
        if self.current_topic_id is not None:
            self._schedule(self._open_topic_async, self.current_topic_id)

    # ------------------------------------------------------------------
    # Обработчики проверки / отметки / PDF
    # ------------------------------------------------------------------
    def _on_review(self, event=None) -> None:
        self._schedule(self._review_async)

    async def _review_async(self) -> None:
        review = getattr(self.app, "learning_review", None)
        if review is None or self.user is None:
            return
        path_field = getattr(self, "_path_field", None)
        path = str(path_field.value or "").strip().strip('"') \
            if path_field is not None else ""
        if not path:
            self._set_status("Укажите путь к папке проекта", error=True)
            self.update()
            return
        self._set_busy("Проверяю задание (fs MCP + LLM, это занимает до "
                       "минуты)…")
        self.review_button.disabled = True
        self.update()
        try:
            result = await review.review_task(
                self.user["id"], self.current_subtopic_id, path)
        except Exception as exc:  # noqa: BLE001 — показываем в статусе
            self._set_status(f"Ошибка проверки: {exc}", error=True)
            self.review_button.disabled = False
            self.update()
            return
        self._save_project_path(path)
        repo = self._repo()
        subtopic = repo.get_subtopic_for_user(
            self.user["id"], self.current_subtopic_id)
        self._set_status("")
        self.review_button.disabled = False
        if subtopic is not None:
            self._show_subtopic(subtopic)
        verdict = "подтема завершена ✅" if result.get("completed") \
            else "продолжайте работу"
        self._notify(f"Оценка: {result.get('score')}/5 — {verdict}")

    def _on_toggle_completed(self, event=None) -> None:
        self._schedule(self._toggle_completed_async)

    async def _toggle_completed_async(self) -> None:
        repo = self._repo()
        if repo is None or self.user is None:
            return
        subtopic = repo.get_subtopic_for_user(
            self.user["id"], self.current_subtopic_id)
        repo.mark_completed(self.user["id"], self.current_subtopic_id,
                            not bool(subtopic and subtopic["is_completed"]))
        await self._open_topic_async(self.current_topic_id)

    def _on_pdf(self, event=None) -> None:
        self._schedule(self._pdf_async)

    async def _pdf_async(self) -> None:
        pdf_service = getattr(self.app, "learning_pdf", None)
        repo = self._repo()
        if pdf_service is None or repo is None or self.user is None:
            return
        self._set_busy("Собираю PDF-конспект (WeasyPrint)…")
        self.update()
        try:
            topic = await asyncio.to_thread(
                repo.get_topic, self.user["id"], self.current_topic_id)
            path = await asyncio.to_thread(
                pdf_service.generate_summary, topic,
                topic.get("subtopics") or [],
                self.user.get("display_name") or "")
        except Exception as exc:  # noqa: BLE001 — статус вместо падения
            self._set_status(f"Ошибка PDF: {exc}", error=True)
            self.update()
            return
        self._set_status(f"PDF-конспект: {path}")
        self._notify(f"📄 PDF создан: {path}")

    async def _pick_folder_async(self) -> None:
        """Диалог выбора папки проекта (FilePicker страницы)."""
        file_picker = getattr(self.app, "file_picker", None)
        if file_picker is None:
            self._notify("Диалог выбора папки недоступен")
            return
        path = await file_picker.get_directory_path(
            dialog_title="Папка Android-проекта (выполненное задание)")
        if path and getattr(self, "_path_field", None) is not None:
            self._path_field.value = str(path)
            self._save_project_path(str(path))
            self.update()

    def _on_pick_folder(self, event=None) -> None:
        self._schedule(self._pick_folder_async)

    def _save_project_path(self, path: str) -> None:
        store = getattr(self.app, "app_state", None)
        if store is not None:
            try:
                store.set(LEARNING_PROJECT_PATH_KEY, str(path))
            except Exception:  # noqa: BLE001 — не критично
                pass

    # ------------------------------------------------------------------
    # Служебное
    # ------------------------------------------------------------------
    def _set_busy(self, message: str) -> None:
        self.status_text.value = message
        self.status_text.color = ft.Colors.BLUE_GREY_500

    def _set_status(self, message: str, error: bool = False) -> None:
        self.status_text.value = message
        self.status_text.color = (ft.Colors.RED_600 if error
                                  else ft.Colors.BLUE_GREY_500)
