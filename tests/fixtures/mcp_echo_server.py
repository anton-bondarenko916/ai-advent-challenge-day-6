"""Тестовый MCP-сервер для офлайн-проверок клиента (FastMCP, stdio).

Запускается тестами как подпроцесс: ``python -u mcp_echo_server.py``.
Сеть не нужна — транспорт stdio, сервер локальный. Эмулирует набор
инструментов в духе open-internet-mcp.

Важно: флаг ``-u`` обязателен — без него stdout подпроцесса буферизуется
и клиент не видит ответы JSON-RPC.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("test-server")


@mcp.tool()
def wikipedia_summary(title: str, sentences: int = 3) -> dict:
    """Возвращает краткую сводку статьи Википедии по названию."""
    return {"title": title, "summary": f"Статья про {title}", "sentences": sentences}


@mcp.tool()
def github_search(query: str) -> dict:
    """Ищет репозитории на GitHub по запросу."""
    return {"query": query, "total": 1, "items": ["octocat/Hello-World"]}


@mcp.tool()
def no_params() -> str:
    """Инструмент без параметров (проверка inputSchema без properties)."""
    return "ok"


if __name__ == "__main__":
    mcp.run()  # stdio-транспорт по умолчанию
