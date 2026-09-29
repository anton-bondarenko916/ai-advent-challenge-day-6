"""UI Flet-приложения: вкладки чата, базы знаний, сравнения, настроек."""

from __future__ import annotations

from app.ui.chat_view import ChatView
from app.ui.comparison_view import (
    ComparisonView,
    load_results,
    load_test_questions,
    save_results,
)
from app.ui.components import (
    ASSESSMENT_OPTIONS,
    aggregate_assessments,
    format_summary,
    sources_panel,
)
from app.ui.knowledge_view import KnowledgeView
from app.ui.settings_view import SettingsView

__all__ = [
    "ChatView",
    "ComparisonView",
    "KnowledgeView",
    "SettingsView",
    "load_test_questions",
    "load_results",
    "save_results",
    "aggregate_assessments",
    "format_summary",
    "sources_panel",
]
