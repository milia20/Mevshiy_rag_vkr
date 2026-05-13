"""
Indexing module for Qdrant.

Handles:
- Collection creation for different vector types
- Dense, sparse, hybrid, and ColBERT indexing
- Batch upsert operations
"""

from .qdrant_indexer import QdrantIndexer

__all__ = ["QdrantIndexer"]
