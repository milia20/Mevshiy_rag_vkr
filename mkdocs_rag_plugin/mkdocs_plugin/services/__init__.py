"""
Services package for MkDocs RAG Plugin.

Модуль содержит сервисы для обработки документов и чанкования.
"""

from mkdocs_plugin.services.chunker import Chunk, Chunker
from mkdocs_plugin.services.doc_processor import DocumentProcessor

__all__ = ["Chunk", "Chunker", "DocumentProcessor"]
