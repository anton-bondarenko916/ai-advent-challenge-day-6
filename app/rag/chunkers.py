"""Chunking MD-документов: две стратегии разбиения на чанки.

Стратегия A (``chunk_fixed``): фиксированный размер —
RecursiveCharacterTextSplitter по разделителям ["\\n\\n", "\\n", " ", ""].

Стратегия B (``chunk_structure``): structure-aware, двухэтапная — сначала
MarkdownHeaderTextSplitter режет документ по заголовкам #/##/###, затем
каждая секция догоняется до chunk_size тем же рекурсивным сплиттером.
Заголовки секций сохраняются в метаданных чанка (h1/h2/h3).

Валидация параметров и обогащение метаданных (chunk_id, section,
strategy, chunk_size, overlap) выполняются здесь, чтобы эмбеддер и
Chroma получали готовые Document'ы.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document

# Заголовки Markdown, по которым режет structure-aware стратегия
HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]

# Поддерживаемые стратегии (метаданное "strategy")
STRATEGY_FIXED = "fixed"
STRATEGY_STRUCTURE = "structure"

SEPARATORS = ["\n\n", "\n", " ", ""]

# Ленивый импорт langchain_text_splitters (кэш): пакет при импорте
# тащит sentence_transformers/transformers (~7 с), а сплиттеры нужны
# только в момент индексации — старт приложения не должен их ждать
_splitters = None


def _get_splitters():
    """MarkdownHeaderTextSplitter и RecursiveCharacterTextSplitter (лениво)."""
    global _splitters
    if _splitters is None:
        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )
        _splitters = (MarkdownHeaderTextSplitter,
                      RecursiveCharacterTextSplitter)
    return _splitters


def validate_chunking(chunk_size: int, overlap: int) -> None:
    """Проверяет параметры chunking.

    Args:
        chunk_size: максимальный размер чанка в символах (>= 50).
        overlap: перекрытие чанков в символах (>= 0, меньше chunk_size).

    Raises:
        ValueError: если параметры вне допустимых границ.
    """
    if not isinstance(chunk_size, int) or chunk_size < 50:
        raise ValueError(
            f"chunk_size должен быть целым >= 50, получено: {chunk_size!r}"
        )
    if not isinstance(overlap, int) or overlap < 0:
        raise ValueError(
            f"overlap должен быть целым >= 0, получено: {overlap!r}"
        )
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) должен быть меньше chunk_size ({chunk_size})"
        )


def chunk_fixed(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Document]:
    """Стратегия A: режет текст на чанки фиксированного размера.

    Args:
        text: исходный текст документа.
        chunk_size: максимальный размер чанка в символах.
        overlap: перекрытие соседних чанков в символах.

    Returns:
        Список langchain Document'ов (page_content + пустые метаданные).

    Raises:
        ValueError: некорректные chunk_size/overlap.
    """
    validate_chunking(chunk_size, overlap)
    _, recursive_cls = _get_splitters()
    splitter = recursive_cls(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=SEPARATORS,
    )
    return splitter.create_documents([text])


def chunk_structure(
    text: str, chunk_size: int = 500, overlap: int = 50
) -> List[Document]:
    """Стратегия B: structure-aware разбиение (двухэтапное).

    Этап 1 — MarkdownHeaderTextSplitter режет документ по заголовкам
    #/##/###, заголовки попадают в метаданные (h1/h2/h3).
    Этап 2 — каждая секция догоняется до chunk_size рекурсивным
    сплиттером с сохранением метаданных заголовков.

    Args:
        text: исходный MD-текст документа.
        chunk_size: максимальный размер чанка в символах.
        overlap: перекрытие соседних чанков в символах.

    Returns:
        Список Document'ов с метаданными заголовков (h1/h2/h3).

    Raises:
        ValueError: некорректные chunk_size/overlap.
    """
    validate_chunking(chunk_size, overlap)
    header_cls, recursive_cls = _get_splitters()
    header_splitter = header_cls(
        headers_to_split_on=HEADERS_TO_SPLIT_ON
    )
    header_chunks = header_splitter.split_text(text)
    text_splitter = recursive_cls(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return text_splitter.split_documents(header_chunks)


def chunk_by_strategy(
    text: str,
    strategy: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Document]:
    """Режет текст выбранной стратегией.

    Args:
        text: исходный MD-текст.
        strategy: "fixed" или "structure".
        chunk_size: максимальный размер чанка.
        overlap: перекрытие чанков.

    Returns:
        Список Document'ов.

    Raises:
        ValueError: неизвестная стратегия или некорректные параметры.
    """
    if strategy == STRATEGY_FIXED:
        return chunk_fixed(text, chunk_size, overlap)
    if strategy == STRATEGY_STRUCTURE:
        return chunk_structure(text, chunk_size, overlap)
    raise ValueError(
        f"Неизвестная стратегия chunking: {strategy!r} "
        f"(ожидалось '{STRATEGY_FIXED}' или '{STRATEGY_STRUCTURE}')"
    )


def section_from_metadata(metadata: Dict[str, Any]) -> str:
    """Собирает строку секции из метаданных заголовков.

    Берётся самый глубокий заголовок: h3 → "### Подзаголовок",
    h2 → "## Windows", h1 → "## Введение" (одинарный "# " убирается
    для h1, чтобы строка секции не путалась с title документа).

    Args:
        metadata: метаданные чанка (возможны ключи h1/h2/h3).

    Returns:
        Строка секции ("" если заголовков нет).
    """
    for key, level in (("h3", 3), ("h2", 2), ("h1", 1)):
        value = metadata.get(key)
        if value:
            prefix = "#" * level if level > 1 else "#"
            return f"{prefix} {value}"
    return ""


def enrich_metadata(
    documents: List[Document],
    source: str,
    title: str,
    strategy: str,
    chunk_size: int,
    overlap: int,
    first_chunk_id: int = 0,
) -> List[Document]:
    """Приводит метаданные чанков к формату панели «Источники».

    Формат (по ТЗ): source (имя файла), title (заголовок документа),
    section ("## Windows" — самый глубокий заголовок), chunk_id
    (сквозной номер), strategy, chunk_size, overlap. Сырые ключи
    h1/h2/h3 из header-сплиттера в итоговые метаданные не попадают.

    Args:
        documents: чанки одной стратегии одного файла.
        source: имя файла-источника (например, "installation.md").
        title: заголовок документа (первая "# " строка или имя файла).
        strategy: "fixed" | "structure".
        chunk_size: использованный размер чанка.
        overlap: использованное перекрытие.
        first_chunk_id: стартовый сквозной номер чанка.

    Returns:
        Тот же список Document'ов с перезаписанными метаданными.
    """
    enriched: List[Document] = []
    for offset, doc in enumerate(documents):
        doc.metadata = {
            "source": source,
            "title": title,
            "section": section_from_metadata(doc.metadata),
            "chunk_id": first_chunk_id + offset,
            "strategy": strategy,
            "chunk_size": chunk_size,
            "overlap": overlap,
        }
        enriched.append(doc)
    return enriched
