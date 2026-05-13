"""
Retrieval module with multiple search strategies.

Supports:
- Dense search
- Sparse search
- Hybrid fusion (RRF, DBSF)
- ColBERT late interaction
- Experimental native Qdrant methods
- Code search with dual encoders
"""

from .code_searcher import CodeSearcher
from .experimental_searchers import (
    AdvancedColBERTSearcher,
    DocumentDenseSearcher,
    DocumentSparseSearcher,
    ExperimentalSearcherFactory,
    NativeDbsfHybridSearcher,
    NativeRrfHybridSearcher,
)
from .searchers import (
    ColBERTSearcher,
    DenseSearcher,
    HybridSearcher,
    SparseSearcher,
)

__all__ = [
    "DenseSearcher",
    "SparseSearcher",
    "HybridSearcher",
    "ColBERTSearcher",
    "CodeSearcher",
    "DocumentDenseSearcher",
    "DocumentSparseSearcher",
    "NativeRrfHybridSearcher",
    "NativeDbsfHybridSearcher",
    "AdvancedColBERTSearcher",
    "ExperimentalSearcherFactory",
]
