"""Общий Tk-root для GUI-тестов.

Особенность Tk на Windows: сгенерированные клавиатурные события
(``event_generate``) доставляются только ПЕРВОМУ окну процесса, а «второе»
окно их молча теряет. Поэтому все GUI-тесты используют одно общее окно,
которое создаётся один раз на процесс (лениво) и не уничтожается —
виджеты приложения между тестами пересоздаются.

Если графическая среда недоступна, :func:`get_root` возвращает None,
и тесты должны пропускаться через ``unittest.SkipTest``.
"""

from __future__ import annotations

from typing import Any, Optional

try:  # pragma: no cover — зависит от окружения
    import tkinter as tk
except ImportError:  # pragma: no cover
    tk = None  # type: ignore

_ROOT: Optional[Any] = None


def get_root() -> Optional[Any]:
    """Возвращает общий Tk-root (или None, если GUI недоступен)."""
    global _ROOT
    if tk is None:
        return None
    if _ROOT is None:
        try:
            root = tk.Tk()
        except Exception:  # нет дисплея / TclError
            return None
        # Окно должно быть отображено (mapped), но уводим его за экран
        root.geometry("640x480+-3000+-3000")
        root.update()
        _ROOT = root
    return _ROOT


def reset_children(root: Any) -> None:
    """Уничтожает виджеты предыдущего теста, оставляя само окно."""
    for child in root.winfo_children():
        try:
            child.destroy()
        except Exception:  # pragma: no cover
            pass
