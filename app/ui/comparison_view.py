"""Вкладка «Сравнение»: 10 контрольных вопросов × 2 режима + статистика.

Чистая логика (загрузка вопросов, агрегация оценок, сохранение
результатов) вынесена в функции модуля — покрывается офлайн-тестами
без запуска Flet.

Прогон: для каждого вопроса два изолированных запроса на «чистом»
агенте (RAGChatAgent.answer_once, без хранилища) — история чата не
засоряется. Оценки выставляются пользователем (Dropdown) и сохраняются
в app/data/comparison_results.json.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import flet as ft

from app.ui import components

DEFAULT_QUESTIONS_PATH = Path(__file__).resolve().parents[1] / "data" / "test_questions.json"
DEFAULT_RESULTS_PATH = Path(__file__).resolve().parents[1] / "data" / "comparison_results.json"

QUESTION_FIELDS = ("id", "question", "expected", "sources")


def load_test_questions(path: Path = DEFAULT_QUESTIONS_PATH) -> List[Dict[str, Any]]:
    """Читает и валидирует контрольные вопросы.

    Args:
        path: путь к test_questions.json.

    Returns:
        Список словарей {id, question, expected, sources}.

    Raises:
        ValueError: файл не список, пуст, поля отсутствуют, id дублируются.
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Не читается {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Некорректный JSON в {path}: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: ожидался непустой список вопросов")
    seen_ids = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Вопрос #{index}: не объект")
        for field in QUESTION_FIELDS:
            if field not in item:
                raise ValueError(f"Вопрос #{index}: нет поля {field!r}")
        if item["id"] in seen_ids:
            raise ValueError(f"Дублирующийся id: {item['id']}")
        seen_ids.add(item["id"])
        if not str(item["question"]).strip():
            raise ValueError(f"Вопрос #{index}: пустой вопрос")
        if not isinstance(item["sources"], list) or not item["sources"]:
            raise ValueError(f"Вопрос #{index}: sources должен быть непустым списком")
    return raw


def normalize_assessment(value: Any) -> Optional[str]:
    """Приводит значение Dropdown к оценке (или None)."""
    if value in components.ASSESSMENT_OPTIONS:
        return str(value)
    return None


def save_results(
    results: Dict[int, Dict[str, Any]],
    path: Path = DEFAULT_RESULTS_PATH,
) -> None:
    """Сохраняет ответы и оценки прогона в JSON (переживает перезапуск)."""
    payload = {
        str(question_id): data for question_id, data in results.items()
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_results(path: Path = DEFAULT_RESULTS_PATH) -> Dict[int, Dict[str, Any]]:
    """Читает сохранённый прогон (отсутствует файл → пустой словарь)."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        int(question_id): data
        for question_id, data in raw.items()
        if isinstance(data, dict)
    }


class ComparisonView:
    """Таблица вопросов, прогон двух режимов, оценки, сводка."""

    def __init__(
        self,
        app: Any,
        questions_path: Path = DEFAULT_QUESTIONS_PATH,
        results_path: Path = DEFAULT_RESULTS_PATH,
    ) -> None:
        self.app = app
        self.agent = app.agent
        self.results_path = results_path
        self.questions = load_test_questions(questions_path)
        self._running = False

        saved = load_results(results_path)
        self.rows: List[Dict[str, Any]] = []
        questions_column = ft.Column(spacing=10, scroll=ft.ScrollMode.AUTO, expand=True)
        for question in self.questions:
            row = self._build_row(question, saved.get(int(question["id"]), {}))
            self.rows.append(row)
            questions_column.controls.append(row["card"])

        self.run_button = ft.FilledButton(
            f"Прогнать все ({len(self.questions)})",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._on_run,
        )
        self.progress_bar = ft.ProgressBar(value=0, visible=False)
        self.progress_text = ft.Text("", size=12)
        # Сводка: тёмная панель с явным светлым текстом — контраст не
        # зависит от темы окна (голубой BLUE_GREY_50 на тёмной теме
        # делал «Без RAG: нет оценок» нечитаемым)
        self.summary_text_color = ft.Colors.GREY_50
        self.summary_bgcolor = ft.Colors.BLUE_GREY_900
        self.summary_no_rag = ft.Text("Без RAG: нет оценок", size=13,
                                      weight=ft.FontWeight.BOLD,
                                      color=self.summary_text_color,
                                      selectable=True)
        self.summary_rag = ft.Text("С RAG: нет оценок", size=13,
                                   weight=ft.FontWeight.BOLD,
                                   color=self.summary_text_color,
                                   selectable=True)
        self.summary_container = ft.Container(
            ft.Column([self.summary_no_rag, self.summary_rag], spacing=4),
            padding=10,
            bgcolor=self.summary_bgcolor,
            border_radius=8,
        )
        self._apply_saved_summaries()
        self._questions_column = questions_column

    # ------------------------------------------------------------------
    # Компоновка
    # ------------------------------------------------------------------
    def build(self) -> ft.Column:
        """Собирает вкладку: кнопка прогона, таблица вопросов, сводка."""
        return ft.Column([
            ft.Row([self.run_button, self.progress_bar, self.progress_text],
                   spacing=12),
            self._questions_column,
            self.summary_container,
        ], expand=True, spacing=10)

    def on_show(self) -> None:
        """Ничего не требует при открытии (заготовка под refresh)."""

    # ------------------------------------------------------------------
    # Строка вопроса
    # ------------------------------------------------------------------
    def _build_row(self, question: Dict[str, Any], saved: Dict[str, Any]) -> Dict[str, Any]:
        answer_no_rag = components.markdown_text(
            str(saved.get("answer_no_rag") or "—"))
        answer_rag = components.markdown_text(
            str(saved.get("answer_rag") or "—"))
        sources_text = ft.Text(
            str(saved.get("sources_text") or "—"), size=11,
            color=ft.Colors.BLUE_GREY_600, selectable=True)
        dropdown_no_rag = components.assessment_dropdown(
            normalize_assessment(saved.get("assessment_no_rag")) or "",
            on_select=self._on_assessment_change)
        dropdown_rag = components.assessment_dropdown(
            normalize_assessment(saved.get("assessment_rag")) or "",
            on_select=self._on_assessment_change)
        header = ft.Row([
            ft.Text(f"{question['id']}.", size=13, weight=ft.FontWeight.BOLD,
                    color=ft.Colors.BLUE_700),
            ft.Text(question["question"], size=13, weight=ft.FontWeight.BOLD,
                    expand=True, selectable=True),
        ], spacing=6)
        expected = ft.ExpansionTile(
            title=ft.Text("Ожидаемый ответ / источники", size=11),
            controls=[
                ft.Text(question["expected"], size=11, selectable=True),
                ft.Text("Источники: " + ", ".join(question["sources"]),
                        size=11, color=ft.Colors.BLUE_GREY_600),
            ],
            tile_padding=4,
        )
        card = ft.Card(ft.Container(ft.Column([
            header,
            expected,
            ft.Text("Ответ без RAG:", size=11, weight=ft.FontWeight.BOLD),
            answer_no_rag,
            ft.Text("Ответ с RAG:", size=11, weight=ft.FontWeight.BOLD),
            answer_rag,
            ft.Text("Источники:", size=11, weight=ft.FontWeight.BOLD),
            sources_text,
            ft.Row([
                ft.Text("Оценка без RAG:", size=12), dropdown_no_rag,
                ft.Text("Оценка с RAG:", size=12), dropdown_rag,
            ], spacing=8, wrap=True),
        ], spacing=4, tight=True), padding=10))
        return {
            "question": question,
            "card": card,
            "answer_no_rag": answer_no_rag,
            "answer_rag": answer_rag,
            "sources_text": sources_text,
            "dropdown_no_rag": dropdown_no_rag,
            "dropdown_rag": dropdown_rag,
        }

    # ------------------------------------------------------------------
    # Прогон
    # ------------------------------------------------------------------
    def _on_run(self, event=None) -> None:
        self._schedule(self._run_async)

    def _schedule(self, handler, *args) -> None:
        """Запускает async-задачу: КОРУТИННУЮ ФУНКЦИЮ (не корутину!) в
        page.run_task — иначе TypeError и error-overlay Flet. Гард
        (components.guarded_task) пишет traceback в лог и показывает
        SnackBar вместо падения интерфейса. Без page (тесты) — no-op."""
        if self.app.page is None:
            return
        self.app.page.run_task(
            components.guarded_task(self.app, handler), *args)

    async def _run_async(self) -> None:
        """Прогоняет все вопросы в двух режимах последовательно."""
        if self._running:
            return
        self._running = True
        self.run_button.disabled = True
        self.progress_bar.visible = True
        self.progress_bar.value = 0
        try:
            total = len(self.rows)
            for number, row in enumerate(self.rows, start=1):
                question_text = row["question"]["question"]
                self.progress_text.value = (
                    f"Вопрос {number}/{total}: {question_text[:60]}…")
                self.update()
                # Два изолированных запроса: история чата не меняется
                result_no_rag = await self.agent.answer_once(
                    question_text, rag_enabled=False)
                result_rag = await self.agent.answer_once(
                    question_text, rag_enabled=True)
                row["answer_no_rag"].value = self._answer_text(result_no_rag)
                row["answer_rag"].value = self._answer_text(result_rag)
                row["sources_text"].value = self._sources_text(result_rag)
                self.progress_bar.value = number / total
                self.update()
            self._persist_results()
            self.refresh_summaries()
            self.progress_text.value = "Прогон завершён — выставьте оценки"
        except Exception as exc:
            self.progress_text.value = f"Ошибка прогона: {exc}"
        finally:
            self._running = False
            self.run_button.disabled = False
            self.progress_bar.visible = False
            self.update()

    @staticmethod
    def _answer_text(result: Dict[str, Any]) -> str:
        if result.get("error"):
            return f"⚠️ Ошибка: {result['error']}"
        text = str(result.get("text") or "(пустой ответ)")
        rag_error = result.get("rag_error")
        if rag_error:
            text += f"\n\n⚠️ RAG: {rag_error}"
        return text

    @staticmethod
    def _sources_text(result: Dict[str, Any]) -> str:
        sources = result.get("sources") or []
        if not sources:
            return "—"
        lines = []
        for source in sources:
            section = source.get("section") or "—"
            strategy = source.get("strategy") or "fixed"
            lines.append(
                f"📄 {source.get('source')} → {section} ({strategy})")
        return "\n".join(lines)

    def _persist_results(self) -> None:
        """Сохраняет ответы и оценки (переживают перезапуск приложения)."""
        results: Dict[int, Dict[str, Any]] = {}
        for row in self.rows:
            results[int(row["question"]["id"])] = {
                "answer_no_rag": row["answer_no_rag"].value,
                "answer_rag": row["answer_rag"].value,
                "sources_text": row["sources_text"].value,
                "assessment_no_rag": normalize_assessment(
                    row["dropdown_no_rag"].value),
                "assessment_rag": normalize_assessment(
                    row["dropdown_rag"].value),
            }
        try:
            save_results(results, self.results_path)
        except OSError as exc:
            self.progress_text.value = f"Не сохранить результаты: {exc}"

    # ------------------------------------------------------------------
    # Оценки и сводка
    # ------------------------------------------------------------------
    def _on_assessment_change(self, event=None) -> None:
        self._persist_results()
        self.refresh_summaries()
        self.update()

    def assessments(self) -> tuple:
        """Текущие оценки двумя списками (для тестов и сводки)."""
        no_rag = [normalize_assessment(row["dropdown_no_rag"].value)
                  for row in self.rows]
        rag = [normalize_assessment(row["dropdown_rag"].value)
               for row in self.rows]
        return no_rag, rag

    def refresh_summaries(self) -> None:
        """Пересчитывает агрегированную статистику (формат ТЗ)."""
        no_rag, rag = self.assessments()
        self.summary_no_rag.value = components.format_summary(
            "Без RAG", components.aggregate_assessments(no_rag))
        self.summary_rag.value = components.format_summary(
            "С RAG", components.aggregate_assessments(rag))

    def _apply_saved_summaries(self) -> None:
        self.refresh_summaries()

    def update(self) -> None:
        if self.app.page is not None:
            self.app.page.update()
