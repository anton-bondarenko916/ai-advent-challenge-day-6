"""Доменные модели подсистемы MCP (Model Context Protocol).

Модуль не зависит от SDK ``mcp`` и от Tkinter: используется клиентом,
воркером и панелью UI. Ничего не знает о сети — только данные.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MCPTool:
    """Инструмент MCP-сервера (ответ на запрос ``tools/list``).

    Attributes:
        name: уникальное имя инструмента (например, ``wikipedia_summary``).
        description: человекочитаемое описание (что делает инструмент).
        input_schema: JSON Schema параметров вызова (поле ``inputSchema``
            протокола MCP); None — сервер не прислал схему.
    """

    name: str
    description: str = ""
    input_schema: Optional[Dict[str, Any]] = None

    def short_description(self, limit: int = 80) -> str:
        """Первая строка описания, обрезанная до ``limit`` символов.

        Используется в карточке панели: полное описание доступно в
        модальном окне inputSchema, чтобы не перегружать список.

        Args:
            limit: максимальная длина результата.

        Returns:
            Короткое описание ("" — описание отсутствует).
        """
        text = (self.description or "").strip().splitlines()
        first = text[0].strip() if text else ""
        if len(first) <= limit:
            return first
        return first[: limit - 1].rstrip() + "…"

    def schema_json(self, indent: int = 2) -> str:
        """Возвращает inputSchema как отформатированный JSON.

        Args:
            indent: отступ вложенных уровней.

        Returns:
            Строка JSON; ``"null"`` для отсутствующей схемы.
        """
        import json

        return json.dumps(self.input_schema, ensure_ascii=False, indent=indent)

    @staticmethod
    def sort_key(tool: "MCPTool") -> str:
        """Ключ сортировки списка инструментов по имени (регистронезависимо)."""
        return tool.name.lower()


def sort_tools(tools: List[MCPTool]) -> List[MCPTool]:
    """Возвращает копию списка инструментов, отсортированную по имени."""
    return sorted(tools, key=MCPTool.sort_key)
