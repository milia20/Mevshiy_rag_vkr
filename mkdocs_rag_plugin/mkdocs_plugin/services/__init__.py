"""
Services package for MkDocs RAG Plugin.

Модуль содержит сервисы для обработки документов и чанкования.
"""

from mkdocs_plugin.services.doc_processor import DocumentProcessor
from mkdocs_plugin.services.chunker import Chunker, Chunk

__all__ = ["DocumentProcessor", "Chunker", "Chunk"]
