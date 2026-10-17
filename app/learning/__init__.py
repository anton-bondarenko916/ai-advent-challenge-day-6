"""Платформа обучения Android-разработке (ТЗ ai_1_36).

Модули:

* :mod:`app.learning.db` — SQLite (users/topics/subtopics/progress):
  мок-авторизация (SHA-256), план обучения, прогресс и streak;
* :mod:`app.learning.llm` — JSON-режим LLM (DeepSeek) + офлайн-заглушка;
* :mod:`app.learning.plan` — ленивая генерация темы (5 тем × 3–4
  подтемы, LLM вызывается при первом входе в тему) + напоминание
  через MCP reminder;
* :mod:`app.learning.review` — проверка задания: fs_server.py
  (MCP) читает файлы проекта пользователя, LLM оценивает по рубрике;
* :mod:`app.learning.pdf` — PDF-конспект темы через WeasyPrint.

UI — вкладка «🎓 Обучение» (app/ui/learning_view.py).
"""
