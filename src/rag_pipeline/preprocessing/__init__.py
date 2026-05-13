"""
Document preprocessing and chunking module.

Handles:
- Loading documents from DataFrame
- Text preprocessing
- Chunking with metadata preservation
- Source tracking
"""

from .chunker import DocumentChunker

__all__ = ["DocumentChunker"]
