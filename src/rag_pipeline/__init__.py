"""
RAG Pipeline Module

Complete RAG pipeline with:
- Document preprocessing and chunking
- Multi-strategy indexing (dense, sparse, hybrid, ColBERT)
- Retrieval with fusion strategies (RRF, DBSF)
- Reranking
- Answer generation with source citations
- IR metrics evaluation (Hit@k, MRR@k, latency)
"""

from .evaluation.metrics import RetrievalEvaluator
from .generation.generator import AnswerGenerator
from .indexing.qdrant_indexer import QdrantIndexer
from .preprocessing.chunker import DocumentChunker
from .reranking.reranker import Reranker
from .retrieval.searchers import (
    ColBERTSearcher,
    DenseSearcher,
    HybridSearcher,
    SparseSearcher,
)

__all__ = [
    "AnswerGenerator",
    "ColBERTSearcher",
    "DenseSearcher",
    "DocumentChunker",
    "HybridSearcher",
    "QdrantIndexer",
    "Reranker",
    "RetrievalEvaluator",
    "SparseSearcher",
]
