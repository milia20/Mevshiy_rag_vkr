"""
Вспомогательные утилиты для плагина MkDocs RAG.
"""

from mkdocs_plugin.utils.helpers import (
    extract_text_from_html,
    normalize_text,
    split_text_into_chunks,
)

__all__ = [
    "extract_text_from_html",
    "normalize_text",
    "split_text_into_chunks",
]
