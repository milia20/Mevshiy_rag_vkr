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

from .preprocessing.chunker import DocumentChunker
from .indexing.qdrant_indexer import QdrantIndexer
from .retrieval.searchers import (
    DenseSearcher,
    SparseSearcher,
    HybridSearcher,
    ColBERTSearcher,
)
from .reranking.reranker import Reranker
from .generation.generator import AnswerGenerator
from .evaluation.metrics import RetrievalEvaluator

__all__ = [
    "DocumentChunker",
    "QdrantIndexer",
    "DenseSearcher",
    "SparseSearcher",
    "HybridSearcher",
    "ColBERTSearcher",
    "Reranker",
    "AnswerGenerator",
    "RetrievalEvaluator",
]
