"""Живой визуальный прогон MCP-панели (скриншоты, не unittest).

Скрипт запускает приложение в mock-режиме с локальным FastMCP-сервером
(без Node.js и сети), нажимает «Обновить», открывает модалку inputSchema
и снимает скриншоты в docs/screenshots/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["MOCK_MODE"] = "true"
os.environ["STORAGE_ENABLED"] = "false"
os.environ["MCP_ENABLED"] = "true"
os.environ["MCP_SERVER_COMMAND"] = (
    f'"{sys.executable}" -u "{PROJECT_ROOT / "tests/fixtures/mcp_echo_server.py"}"'
)
os.environ["MCP_TIMEOUT"] = "15"
os.environ["APP_STATE_PATH"] = str(PROJECT_ROOT / "tests/tmp_mcp_state/state.db")

import tkinter as tk

from src.config import load_config, validate_api_key
from src.agent import DeepSeekAgent
from src.tokens import HeuristicCounter
from src.ui import ChatApplication


def main() -> int:
    # Чистим сохранённое состояние панели (иначе прошлый прогон мог
    # оставить панель свёрнутой, и стартовый кадр будет не «Не подключён»).
    state_db = PROJECT_ROOT / "tests/tmp_mcp_state"
    if state_db.exists():
        import shutil

        shutil.rmtree(state_db, ignore_errors=True)

    config = load_config()
    validate_api_key(config)
    agent = DeepSeekAgent(
        api_key="", mock_mode=True, token_counter=HeuristicCounter()
    )
    root = tk.Tk()
    app = ChatApplication(root=root, agent=agent, config=config)
    root.geometry("1100x720+40+40")
    # Окно поверх остальных: ImageGrab снимает область экрана, и если
    # окно ушло на задний план, на скриншот попадёт чужое окно.
    root.attributes("-topmost", True)
    root.lift()

    shots = PROJECT_ROOT / "docs/screenshots"
    shots.mkdir(parents=True, exist_ok=True)

    from PIL import ImageGrab

    def grab(name: str) -> None:
        root.update_idletasks()
        root.update()
        x, y = root.winfo_rootx(), root.winfo_rooty()
        w, h = root.winfo_width(), root.winfo_height()
        ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(shots / name)
        print("saved", name)

    panel = app.mcp_panel
    assert panel is not None, "панель MCP не создана"

    # 1. Стартовое состояние: «Не подключён»
    root.after(300, lambda: grab("mcp_1_initial.png"))

    # 2. Нажимаем «Обновить», ждём реальный ответ сервера
    def do_refresh() -> None:
        panel.refresh()
        root.after(2500, after_refresh)

    def after_refresh() -> None:
        grab("mcp_2_connected.png")
        # 3. Модалка inputSchema по клику на инструмент
        tool = next(t for t in panel.tools if t.name == "wikipedia_summary")
        panel.open_schema_dialog(tool)
        root.after(400, after_dialog)

    def after_dialog() -> None:
        grab("mcp_3_schema_dialog.png")
        # 4. Закрываем модалку (она — ребёнок панели) и сворачиваем панель
        for w in panel.winfo_children():
            if isinstance(w, tk.Toplevel):
                w.destroy()
        panel.set_collapsed(True)
        root.after(300, lambda: grab("mcp_4_collapsed.png"))
        root.after(500, finish)

    def finish() -> None:
        root.quit()

    root.after(800, do_refresh)
    root.mainloop()
    root.destroy()
    print("status:", panel.get_status_text(), "| tools:", len(panel.tools))
    return 0


if __name__ == "__main__":
    sys.exit(main())
