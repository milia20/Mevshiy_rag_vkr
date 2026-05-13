"""
Experimental Qdrant searchers using native Qdrant models.

This module implements experimental searchers using:
- models.Document for automatic text encoding
- models.RrfQuery for native RRF fusion
- models.FusionQuery for native DBSF fusion
- Advanced prefetch strategies

These are experimental and should be used separately from the main pipeline.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models


@dataclass
class ExperimentalSearchResult:
    """Represents a single search result from experimental searchers."""

    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.chunk_id,
            "score": self.score,
            "text": self.text,
            "metadata": self.metadata or {},
        }


@dataclass
class ExperimentalSearchResponse:
    """Complete search response with results and metadata."""

    results: list[ExperimentalSearchResult]
    latency_ms: float
    strategy: str
    total_hits: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "latency_ms": self.latency_ms,
            "strategy": self.strategy,
            "total_hits": self.total_hits,
        }


class BaseExperimentalSearcher(ABC):
    """Abstract base class for experimental searchers."""

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> ExperimentalSearchResponse:
        """Perform search and return results."""
        pass


class DocumentDenseSearcher(BaseExperimentalSearcher):
    """
    Dense search using models.Document for automatic encoding.

    Uses Qdrant's native Document model for on-the-fly encoding.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "dense_collection",
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_field: str = "dense",
    ):
        self.client = client
        self.collection_name = collection_name
        self.vector_field = vector_field
        self.model = model

    def search(self, query: str, top_k: int = 10) -> ExperimentalSearchResponse:
        start_time = time.perf_counter()

        # Use models.Document for automatic encoding
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=models.Document(text=query, model=self.model),
            using=self.vector_field,
            limit=top_k,
            with_payload=True,
        )

        # Format results
        results = []
        for hit in hits.points:
            payload = hit.payload or {}
            results.append(
                ExperimentalSearchResult(
                    chunk_id=str(hit.id),
                    score=float(hit.score),
                    text=payload.get("text", ""),
                    metadata=payload,
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return ExperimentalSearchResponse(
            results=results,
            latency_ms=latency_ms,
            strategy="document_dense",
            total_hits=len(results),
        )


class DocumentSparseSearcher(BaseExperimentalSearcher):
    """
    Sparse search using models.Document for automatic encoding.

    Uses Qdrant's native Document model for BM25 encoding.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "sparse_collection",
        model: str = "Qdrant/bm25",
        vector_field: str = "bm25",
    ):
        self.client = client
        self.collection_name = collection_name
        self.vector_field = vector_field
        self.model = model

    def search(self, query: str, top_k: int = 10) -> ExperimentalSearchResponse:
        start_time = time.perf_counter()

        # Use models.Document for automatic sparse encoding
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=models.Document(text=query, model=self.model),
            using=self.vector_field,
            limit=top_k,
            with_payload=True,
        )

        # Format results
        results = []
        for hit in hits.points:
            payload = hit.payload or {}
            results.append(
                ExperimentalSearchResult(
                    chunk_id=str(hit.id),
                    score=float(hit.score),
                    text=payload.get("text", ""),
                    metadata=payload,
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return ExperimentalSearchResponse(
            results=results,
            latency_ms=latency_ms,
            strategy="document_sparse",
            total_hits=len(results),
        )


class NativeRrfHybridSearcher(BaseExperimentalSearcher):
    """
    Hybrid search using native models.RrfQuery for fusion.

    Uses Qdrant's native RRF implementation instead of custom fusion.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "hybrid_collection",
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        sparse_model: str = "Qdrant/bm25",
        dense_vector_name: str = "dense",
        sparse_vector_name: str = "bm25",
        rrf_k: int = 60,
    ):
        self.client = client
        self.collection_name = collection_name
        self.dense_model = dense_model
        self.sparse_model = sparse_model
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 10) -> ExperimentalSearchResponse:
        start_time = time.perf_counter()

        # Use native RRF query
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=models.RrfQuery(rrf=models.Rrf(k=self.rrf_k)),
            prefetch=[
                models.Prefetch(
                    query=models.Document(text=query, model=self.dense_model),
                    using=self.dense_vector_name,
                    limit=top_k * 2,
                ),
                models.Prefetch(
                    query=models.Document(text=query, model=self.sparse_model),
                    using=self.sparse_vector_name,
                    limit=top_k * 2,
                ),
            ],
            limit=top_k,
            with_payload=True,
        )

        # Format results
        results = []
        for hit in hits.points:
            payload = hit.payload or {}
            results.append(
                ExperimentalSearchResult(
                    chunk_id=str(hit.id),
                    score=float(hit.score),
                    text=payload.get("text", ""),
                    metadata=payload,
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return ExperimentalSearchResponse(
            results=results,
            latency_ms=latency_ms,
            strategy="native_rrf_hybrid",
            total_hits=len(results),
        )


class NativeDbsfHybridSearcher(BaseExperimentalSearcher):
    """
    Hybrid search using native models.FusionQuery for DBSF fusion.

    Uses Qdrant's native DBSF implementation instead of custom fusion.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "hybrid_collection",
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        sparse_model: str = "Qdrant/bm25",
        dense_vector_name: str = "dense",
        sparse_vector_name: str = "bm25",
    ):
        self.client = client
        self.collection_name = collection_name
        self.dense_model = dense_model
        self.sparse_model = sparse_model
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name

    def search(self, query: str, top_k: int = 10) -> ExperimentalSearchResponse:
        start_time = time.perf_counter()

        # Use native DBSF fusion query
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=models.FusionQuery(fusion=models.Fusion.DBSF),
            prefetch=[
                models.Prefetch(
                    query=models.Document(text=query, model=self.dense_model),
                    using=self.dense_vector_name,
                    limit=top_k * 2,
                ),
                models.Prefetch(
                    query=models.Document(text=query, model=self.sparse_model),
                    using=self.sparse_vector_name,
                    limit=top_k * 2,
                ),
            ],
            limit=top_k,
            with_payload=True,
        )

        # Format results
        results = []
        for hit in hits.points:
            payload = hit.payload or {}
            results.append(
                ExperimentalSearchResult(
                    chunk_id=str(hit.id),
                    score=float(hit.score),
                    text=payload.get("text", ""),
                    metadata=payload,
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return ExperimentalSearchResponse(
            results=results,
            latency_ms=latency_ms,
            strategy="native_dbsf_hybrid",
            total_hits=len(results),
        )


class AdvancedColBERTSearcher(BaseExperimentalSearcher):
    """
    ColBERT search with advanced prefetch strategies.

    Uses late interaction with prefetch for hybrid retrieval.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "hybrid_with_colbert",
        colbert_model: str = "colbert-ir/colbert-v2.0",
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        sparse_model: str = "Qdrant/bm25",
        colbert_vector_name: str = "late_interaction",
        dense_vector_name: str = "dense",
        sparse_vector_name: str = "bm25",
    ):
        self.client = client
        self.collection_name = collection_name
        self.colbert_model = colbert_model
        self.dense_model = dense_model
        self.sparse_model = sparse_model
        self.colbert_vector_name = colbert_vector_name
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name

    def search(self, query: str, top_k: int = 10) -> ExperimentalSearchResponse:
        start_time = time.perf_counter()

        # Use advanced prefetch with ColBERT late interaction
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=models.Document(text=query, model=self.colbert_model),
            using=self.colbert_vector_name,
            prefetch=[
                models.Prefetch(
                    query=models.Document(text=query, model=self.dense_model),
                    using=self.dense_vector_name,
                    limit=top_k * 2,
                ),
                models.Prefetch(
                    query=models.Document(text=query, model=self.sparse_model),
                    using=self.sparse_vector_name,
                    limit=top_k * 2,
                ),
            ],
            limit=top_k,
            with_payload=True,
        )

        # Format results
        results = []
        for hit in hits.points:
            payload = hit.payload or {}
            results.append(
                ExperimentalSearchResult(
                    chunk_id=str(hit.id),
                    score=float(hit.score),
                    text=payload.get("text", ""),
                    metadata=payload,
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return ExperimentalSearchResponse(
            results=results,
            latency_ms=latency_ms,
            strategy="advanced_colbert",
            total_hits=len(results),
        )


class ExperimentalSearcherFactory:
    """Factory for creating experimental searchers."""

    @staticmethod
    def create_document_dense_searcher(
        client: QdrantClient,
        collection_name: str = "dense_collection",
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> DocumentDenseSearcher:
        """Create a Document-based dense searcher."""
        return DocumentDenseSearcher(client, collection_name, model)

    @staticmethod
    def create_document_sparse_searcher(
        client: QdrantClient,
        collection_name: str = "sparse_collection",
        model: str = "Qdrant/bm25",
    ) -> DocumentSparseSearcher:
        """Create a Document-based sparse searcher."""
        return DocumentSparseSearcher(client, collection_name, model)

    @staticmethod
    def create_native_rrf_searcher(
        client: QdrantClient,
        collection_name: str = "hybrid_collection",
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        sparse_model: str = "Qdrant/bm25",
        rrf_k: int = 60,
    ) -> NativeRrfHybridSearcher:
        """Create a native RRF hybrid searcher."""
        return NativeRrfHybridSearcher(
            client,
            collection_name,
            dense_model,
            sparse_model,
            rrf_k=rrf_k,
        )

    @staticmethod
    def create_native_dbsf_searcher(
        client: QdrantClient,
        collection_name: str = "hybrid_collection",
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        sparse_model: str = "Qdrant/bm25",
    ) -> NativeDbsfHybridSearcher:
        """Create a native DBSF hybrid searcher."""
        return NativeDbsfHybridSearcher(
            client,
            collection_name,
            dense_model,
            sparse_model,
        )

    @staticmethod
    def create_advanced_colbert_searcher(
        client: QdrantClient,
        collection_name: str = "hybrid_with_colbert",
        colbert_model: str = "colbert-ir/colbert-v2.0",
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        sparse_model: str = "Qdrant/bm25",
    ) -> AdvancedColBERTSearcher:
        """Create an advanced ColBERT searcher."""
        return AdvancedColBERTSearcher(
            client,
            collection_name,
            colbert_model,
            dense_model,
            sparse_model,
        )
