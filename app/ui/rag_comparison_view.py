"""Вкладка «Сравнение RAG-режимов»: 5 режимов × вопросы + метрики.

В отличие от вкладки «Сравнение» (оценки выставляет пользователь) здесь
метрики полностью автоматические:

* количество чанков после фильтрации/реранка;
* средний similarity score (= 1 − cosine distance);
* покрытие ключевых слов из expected (✓/✗ — доля найденных в ответе
  и источниках).

Чистая логика (ключевые слова, метрики, агрегация, сохранение прогона)
вынесена в функции модуля — покрывается офлайн-тестами без Flet.

Прогон: для каждого вопроса пять изолированных запросов на «чистом»
агенте (RAGChatAgent.answer_once с rag_mode=N) — история чата не
засоряется. Результаты хранятся в app/data/rag_mode_comparison.json.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import flet as ft

from app.rag.pipeline import (
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    DEFAULT_TOP_K_FINAL,
    RAG_MODES,
    RAG_MODE_LABELS,
)
from app.ui import components

DEFAULT_QUESTIONS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "test_questions.json"
)
DEFAULT_RESULTS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "rag_mode_comparison.json"
)

# Настройки поиска по умолчанию (ТЗ): top_k=20, top_k_final=3, порог 0.7
DEFAULT_TOP_K_TEXT = str(DEFAULT_TOP_K)
DEFAULT_TOP_K_FINAL_TEXT = str(DEFAULT_TOP_K_FINAL)
DEFAULT_THRESHOLD_TEXT = str(DEFAULT_THRESHOLD)

# Максимум ключевых слов для метрики покрытия
MAX_KEYWORDS = 8
MIN_KEYWORD_LEN = 5

# Фиксированная ширина колонки режима (5 колонок в горизонтальном скролле)
MODE_COLUMN_WIDTH = 300
ANSWER_AREA_HEIGHT = 320


def extract_keywords(question: str, expected: str = "") -> List[str]:
    """Ключевые слова вопроса/эталона для автоматической метрики.

    Токены (буквы/цифры) нижнего регистра длиной ≥ 5, без дублей,
    не более MAX_KEYWORDS — сначала из вопроса, затем из expected.
    """
    keywords: List[str] = []
    seen = set()
    for source in (question or "", expected or ""):
        for token in re.findall(r"[а-яёa-z0-9]+", source.lower()):
            if len(token) >= MIN_KEYWORD_LEN and token not in seen:
                seen.add(token)
                keywords.append(token)
            if len(keywords) >= MAX_KEYWORDS:
                return keywords
    return keywords


def keyword_coverage(keywords: List[str], text: str) -> Tuple[int, int]:
    """Сколько ключевых слов встречается в тексте: (найдено, всего)."""
    if not keywords:
        return 0, 0
    haystack = (text or "").lower()
    found = sum(1 for keyword in keywords if keyword in haystack)
    return found, len(keywords)


def mode_metrics(result: Dict[str, Any], keywords: List[str]) -> Dict[str, Any]:
    """Метрики режима по результату answer_once (без оценок человека).

    Returns:
        {"chunks": int, "avg_score": Optional[float],
        "kw_found": int, "kw_total": int,
        "rewrite_used": bool, "rerank_backend": Optional[str],
        "search_query": str} — avg_score None, если чанков нет.
    """
    retrieval = result.get("retrieval") or {}
    scores = [float(s) for s in retrieval.get("scores") or []]
    sources = result.get("sources") or []
    answer = str(result.get("text") or "")
    sources_text = " ".join(
        str(source.get("preview") or "") + " " +
        str(source.get("source") or "") for source in sources
    )
    found, total = keyword_coverage(keywords, answer + " " + sources_text)
    return {
        "chunks": len(sources),
        "avg_score": round(sum(scores) / len(scores), 4) if scores else None,
        "kw_found": found,
        "kw_total": total,
        "rewrite_used": bool(retrieval.get("rewrite_used")),
        "rerank_backend": retrieval.get("rerank_backend"),
        "search_query": str(retrieval.get("search_query") or ""),
    }


def aggregate_metrics(
    runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Средние метрики режима по нескольким вопросам.

    Args:
        runs: список mode_metrics (по вопросу на каждый).

    Returns:
        {"questions": int, "chunks": float, "avg_score": Optional[float],
        "kw_rate": Optional[float]} — проценты None при пустом списке.
    """
    if not runs:
        return {"questions": 0, "chunks": 0.0, "avg_score": None,
                "kw_rate": None}
    chunks = [run["chunks"] for run in runs]
    scores = [run["avg_score"] for run in runs if run["avg_score"] is not None]
    found = sum(run["kw_found"] for run in runs)
    total = sum(run["kw_total"] for run in runs)
    return {
        "questions": len(runs),
        "chunks": round(sum(chunks) / len(chunks), 2),
        "avg_score": round(sum(scores) / len(scores), 4) if scores else None,
        "kw_rate": round(100.0 * found / total, 1) if total else None,
    }


def save_run(
    payload: Dict[str, Any],
    path: Path = DEFAULT_RESULTS_PATH,
) -> None:
    """Сохраняет последний прогон в JSON (переживает перезапуск)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_run(path: Path = DEFAULT_RESULTS_PATH) -> Dict[str, Any]:
    """Читает сохранённый прогон (нет файла/битый JSON → пустой словарь)."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def format_metrics_line(metrics: Dict[str, Any]) -> str:
    """Строка метрик под ответом: «Чанков: 3 · Ср. score: 0.82 · Ключ.: 4/5»."""
    avg = metrics.get("avg_score")
    avg_text = f"{avg:.2f}" if avg is not None else "—"
    found = metrics.get("kw_found", 0)
    total = metrics.get("kw_total", 0)
    return (f"Чанков: {metrics.get('chunks', 0)} · "
            f"Ср. score: {avg_text} · Ключ.: {found}/{total}")


def format_aggregate_line(mode: str, stats: Dict[str, Any]) -> str:
    """Строка агрегата прогона: «Basic RAG: 10 вопр. · 3.0 чанк. · 0.82 · 78%»."""
    if not stats.get("questions"):
        return f"{RAG_MODE_LABELS.get(mode, mode)}: нет данных"
    score = stats.get("avg_score")
    score_text = f"{score:.2f}" if score is not None else "—"
    rate = stats.get("kw_rate")
    rate_text = f"{rate:.0f}%" if rate is not None else "—"
    return (f"{RAG_MODE_LABELS.get(mode, mode)}: {stats['questions']} вопр. · "
            f"{stats['chunks']} чанк. · score {score_text} · ключ. {rate_text}")


class RAGModesView:
    """Таблица 5 режимов: вопрос → колонка на режим (метрики + ответ)."""

    def __init__(
        self,
        app: Any,
        questions_path: Path = DEFAULT_QUESTIONS_PATH,
        results_path: Path = DEFAULT_RESULTS_PATH,
    ) -> None:
        self.app = app
        self.agent = app.agent
        self.results_path = results_path
        self.questions = load_questions(questions_path)
        self._running = False

        # Настройки поиска (ТЗ: только через UI, без хардкода порога)
        self.top_k_field = ft.TextField(
            value=DEFAULT_TOP_K_TEXT, label="Top-K (до)", width=110,
            input_filter=ft.NumbersOnlyInputFilter(),
        )
        self.top_k_final_field = ft.TextField(
            value=DEFAULT_TOP_K_FINAL_TEXT, label="Top-K (после)", width=110,
            input_filter=ft.NumbersOnlyInputFilter(),
        )
        self.threshold_field = ft.TextField(
            value=DEFAULT_THRESHOLD_TEXT, label="Порог similarity", width=130,
        )
        self.mode_dropdown = ft.Dropdown(
            label="Режим",
            width=200,
            value=self.agent.rag_mode,
            options=[
                ft.DropdownOption(key=mode, content=ft.Text(
                    RAG_MODE_LABELS[mode], size=13))
                for mode in RAG_MODES
            ],
            on_select=self._on_mode_change,
            tooltip="Активный RAG-режим чата",
        )

        self.question_field = ft.TextField(
            label="Вопрос",
            hint_text="Введите вопрос и нажмите «Найти»",
            expand=True,
            on_submit=self._on_search,
        )
        self.search_button = ft.FilledButton(
            "Найти", icon=ft.Icons.SEARCH, on_click=self._on_search)
        self.run_button = ft.FilledTonalButton(
            f"Прогнать все ({len(self.questions)})", icon=ft.Icons.PLAY_ARROW,
            on_click=self._on_run_all)
        self.progress_bar = ft.ProgressBar(value=0, visible=False)
        self.progress_text = ft.Text("", size=12, expand=True)

        # Колонки режимов: заголовок, метрики, диагностика, ответ, источники
        self.mode_columns: Dict[str, Dict[str, ft.Control]] = {}
        self._build_mode_columns()

        # Сводка прогона: тёмная панель с явным светлым текстом (контраст
        # не зависит от темы — как на вкладке «Сравнение»)
        self.summary_color = ft.Colors.GREY_50
        self.summary_bgcolor = ft.Colors.BLUE_GREY_900
        self.summary_lines = {
            mode: ft.Text(format_aggregate_line(mode, {}),
                          size=12, color=self.summary_color,
                          weight=ft.FontWeight.BOLD, selectable=True)
            for mode in RAG_MODES
        }
        self.summary_container = ft.Container(
            ft.Column(list(self.summary_lines.values()), spacing=3),
            padding=10,
            bgcolor=self.summary_bgcolor,
            border_radius=8,
        )

        # Восстановление последнего прогона
        self.aggregate: Dict[str, Dict[str, Any]] = {}
        self.current_question: str = ""
        self._apply_saved(load_run(results_path))

    # ------------------------------------------------------------------
    # Компоновка
    # ------------------------------------------------------------------
    def build(self) -> ft.Column:
        """Собирает вкладку: настройки, вопрос, колонки режимов, сводка."""
        return ft.Column([
            ft.Text("Сравнение RAG-режимов",
                    size=16, weight=ft.FontWeight.BOLD),
            ft.Row([
                self.top_k_field, self.top_k_final_field,
                self.threshold_field, self.mode_dropdown,
            ], spacing=10, wrap=True),
            ft.Row([self.question_field, self.search_button,
                    self.run_button], spacing=10, wrap=True),
            ft.Row([self.progress_bar, self.progress_text], spacing=12),
            ft.Row(
                [self._mode_column_controls(mode) for mode in RAG_MODES],
                spacing=8,
                scroll=ft.ScrollMode.AUTO,
            ),
            self.summary_container,
        ], expand=True, spacing=10)

    def _build_mode_columns(self) -> None:
        """Создаёт контролы колонки для каждого режима."""
        for mode in RAG_MODES:
            metrics_text = ft.Text("—", size=12,
                                   weight=ft.FontWeight.BOLD,
                                   selectable=True)
            diag_text = ft.Text("", size=11,
                                color=ft.Colors.BLUE_GREY_600,
                                selectable=True)
            answer_markdown = components.markdown_text("—")
            answer_area = ft.Container(
                content=ft.Column([answer_markdown],
                                  scroll=ft.ScrollMode.AUTO),
                height=ANSWER_AREA_HEIGHT,
            )
            sources_text = ft.Text("—", size=11,
                                   color=ft.Colors.BLUE_GREY_600,
                                   selectable=True)
            card = ft.Card(ft.Container(
                ft.Column([
                    ft.Text(RAG_MODE_LABELS[mode], size=13,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.BLUE_700),
                    metrics_text,
                    diag_text,
                    ft.Text("Ответ:", size=11, weight=ft.FontWeight.BOLD),
                    answer_area,
                    ft.Text("Источники:", size=11, weight=ft.FontWeight.BOLD),
                    sources_text,
                ], spacing=4, tight=True),
                width=MODE_COLUMN_WIDTH, padding=10,
            ))
            self.mode_columns[mode] = {
                "card": card,
                "metrics": metrics_text,
                "diag": diag_text,
                "answer": answer_markdown,
                "sources": sources_text,
            }

    def _mode_column_controls(self, mode: str) -> ft.Control:
        return self.mode_columns[mode]["card"]

    def on_show(self) -> None:
        """Ничего не требует при открытии (заготовка под refresh)."""

    # ------------------------------------------------------------------
    # Обработчики (schedule → run_task; логика — в *_async)
    # ------------------------------------------------------------------
    def _on_search(self, event=None) -> None:
        self._schedule(self._search_async)

    def _on_run_all(self, event=None) -> None:
        self._schedule(self._run_all_async)

    def _on_mode_change(self, event=None) -> None:
        try:
            self.agent.set_rag_mode(str(self.mode_dropdown.value or ""))
        except ValueError:
            self.mode_dropdown.value = self.agent.rag_mode
            self.update()

    def _schedule(self, handler, *args) -> None:
        """Запускает async-задачу: КОРУТИННУЮ ФУНКЦИЮ (не корутину!) в
        page.run_task — иначе TypeError и error-overlay Flet. Гард
        (components.guarded_task) пишет traceback в лог и показывает
        SnackBar вместо падения интерфейса. Без page (тесты) — no-op."""
        if self.app.page is None:
            return
        self.app.page.run_task(
            components.guarded_task(self.app, handler), *args)

    # ------------------------------------------------------------------
    # Параметры поиска из полей UI
    # ------------------------------------------------------------------
    def retrieval_params(self) -> Dict[str, Any]:
        """Читает top_k/top_k_final/threshold из полей (прогон).

        Некорректные значения заменяются текущими настройками агента
        (приложение не должно падать из-за опечатки в поле).
        """
        agent = self.agent

        def _int_or(field_value, fallback):
            try:
                return int(str(field_value).strip())
            except (TypeError, ValueError):
                return fallback

        def _float_or(field_value, fallback):
            try:
                return float(str(field_value).strip().replace(",", "."))
            except (TypeError, ValueError):
                return fallback

        top_k = _int_or(self.top_k_field.value, agent.top_k_search)
        top_k_final = _int_or(self.top_k_final_field.value, agent.top_k)
        threshold = _float_or(self.threshold_field.value,
                              agent.similarity_threshold)
        return {"top_k": max(1, top_k), "top_k_final": max(1, top_k_final),
                "threshold": min(1.0, max(0.0, threshold))}

    # ------------------------------------------------------------------
    # Прогон одного вопроса (кнопка «Найти»)
    # ------------------------------------------------------------------
    async def _search_async(self) -> None:
        """Прогоняет введённый вопрос через все 5 режимов."""
        question = (self.question_field.value or "").strip()
        if not question or self._running:
            return
        await self._run_modes({
            "id": "adhoc", "question": question, "expected": "",
            "sources": [],
        })

    # ------------------------------------------------------------------
    # Прогон всех контрольных вопросов
    # ------------------------------------------------------------------
    async def _run_all_async(self) -> None:
        """Прогоняет все вопросы × 5 режимов с агрегацией метрик."""
        if self._running:
            return
        self._running = True
        self.run_button.disabled = True
        self.search_button.disabled = True
        self.progress_bar.visible = True
        self.progress_bar.value = 0
        runs: Dict[str, List[Dict[str, Any]]] = {mode: [] for mode in RAG_MODES}
        try:
            total = len(self.questions)
            for number, question in enumerate(self.questions, start=1):
                self.progress_text.value = (
                    f"Вопрос {number}/{total}: "
                    f"{question['question'][:60]}…")
                self.update()
                await self._run_modes(question, runs=runs)
                self.progress_bar.value = number / total
            self._refresh_summary()
            self.progress_text.value = (
                f"Прогон завершён ({total} вопросов × {len(RAG_MODES)} режимов)"
            )
        except Exception as exc:
            self.progress_text.value = f"Ошибка прогона: {exc}"
        finally:
            self._running = False
            self.run_button.disabled = False
            self.search_button.disabled = False
            self.progress_bar.visible = False
            self.update()

    async def _run_modes(
        self,
        question: Dict[str, Any],
        runs: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> None:
        """Один вопрос через все режимы: колонки + агрегация + сохранение."""
        keywords = extract_keywords(
            str(question.get("question") or ""),
            str(question.get("expected") or ""),
        )
        self.current_question = str(question.get("question") or "")
        params = self.retrieval_params()
        saved_payload = {
            "question": self.current_question,
            "params": params,
            "results": {},
            "aggregate": self.aggregate,
        }
        for mode in RAG_MODES:
            result = await self.agent.answer_once(
                str(question.get("question") or ""),
                rag_enabled=True,
                rag_mode=mode,
                retrieval_params=params,
            )
            metrics = mode_metrics(result, keywords)
            self._show_mode_result(mode, result, metrics)
            saved_payload["results"][mode] = {
                "answer": self._answer_text(result),
                "metrics": metrics,
            }
            if runs is not None:
                runs[mode].append(metrics)
        self.aggregate = {
            mode: aggregate_metrics(runs[mode])
            for mode in RAG_MODES
        } if runs is not None else self.aggregate
        saved_payload["aggregate"] = self.aggregate
        try:
            save_run(saved_payload, self.results_path)
        except OSError as exc:
            self.progress_text.value = f"Не сохранить результаты: {exc}"
        self.update()

    def _show_mode_result(self, mode: str, result: Dict[str, Any],
                          metrics: Dict[str, Any]) -> None:
        """Заполняет колонку режима: метрики, ответ, источники."""
        column = self.mode_columns[mode]
        column["metrics"].value = format_metrics_line(metrics)
        diagnostics: List[str] = []
        if metrics.get("rewrite_used"):
            diagnostics.append(f"✍ {metrics.get('search_query', '')}")
        if metrics.get("rerank_backend"):
            diagnostics.append(f"🔀 {metrics['rerank_backend']}")
        rag_error = result.get("rag_error")
        if rag_error:
            diagnostics.append(f"⚠ {rag_error}")
        if result.get("error"):
            diagnostics.append(f"⚠ {result['error']}")
        column["diag"].value = "\n".join(diagnostics)
        column["answer"].value = self._answer_text(result)
        column["sources"].value = self._sources_text(result)

    def _refresh_summary(self) -> None:
        """Пересчитывает агрегированные строки сводки."""
        for mode in RAG_MODES:
            self.summary_lines[mode].value = format_aggregate_line(
                mode, self.aggregate.get(mode) or {})

    def _apply_saved(self, payload: Dict[str, Any]) -> None:
        """Восстанавливает последний прогон из JSON."""
        if not payload:
            self._refresh_summary()
            return
        self.current_question = str(payload.get("question") or "")
        if self.current_question:
            self.question_field.value = self.current_question
        self.aggregate = {
            mode: stats for mode, stats in
            (payload.get("aggregate") or {}).items() if mode in RAG_MODES
        }
        saved_results = payload.get("results") or {}
        for mode in RAG_MODES:
            saved = saved_results.get(mode)
            if not isinstance(saved, dict):
                continue
            column = self.mode_columns[mode]
            metrics = saved.get("metrics") or {}
            column["metrics"].value = format_metrics_line(metrics)
            diagnostics: List[str] = []
            if metrics.get("rewrite_used"):
                diagnostics.append(f"✍ {metrics.get('search_query', '')}")
            if metrics.get("rerank_backend"):
                diagnostics.append(f"🔀 {metrics['rerank_backend']}")
            column["diag"].value = "\n".join(diagnostics)
            column["answer"].value = str(saved.get("answer") or "—")
        self._refresh_summary()

    # ------------------------------------------------------------------
    # Форматирование (как на вкладке «Сравнение»)
    # ------------------------------------------------------------------
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
            distance = source.get("distance")
            score_text = (
                f", s={1.0 - distance:.2f}" if distance is not None else ""
            )
            lines.append(
                f"📄 {source.get('source')} → {section}{score_text}")
        return "\n".join(lines)

    def update(self) -> None:
        if self.app.page is not None:
            self.app.page.update()


def load_questions(path: Path = DEFAULT_QUESTIONS_PATH) -> List[Dict[str, Any]]:
    """Читает контрольные вопросы (нет файла/битый JSON → пустой список).

    В отличие от вкладки «Сравнение» прогон не обязателен: вкладка
    работает и с ручным вопросом, поэтому отсутствие файла не ошибка.
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]
