"""
Search strategies for Qdrant retrieval.

Implements:
- Dense search with sentence-transformers
- Sparse search with BM25-style sparse vectors
- Hybrid fusion (RRF, DBSF)
- ColBERT late interaction
"""

import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from loguru import logger
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


@dataclass
class SearchResult:
    """Represents a single search result."""

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
class SearchResponse:
    """Complete search response with results and metadata."""

    results: list[SearchResult]
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


class BaseSearcher(ABC):
    """Abstract base class for all searchers."""

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> SearchResponse:
        """Perform search and return results."""
        pass


class DenseSearcher(BaseSearcher):
    """Dense vector search using sentence-transformers."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "dense_collection",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_field: str = "dense",
        vector_size: int = 384,
    ):
        self.client = client
        self.collection_name = collection_name
        self.vector_field = vector_field
        self.vector_size = vector_size
        self._encoder: SentenceTransformer | None = None
        self._model_name = model_name

    @property
    def encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            logger.info(
                f"Loading dense encoder: {self._model_name}",
            )
            self._encoder = SentenceTransformer(self._model_name)
        return self._encoder

    def encode_query(self, query: str) -> list[float]:
        """Encode query into dense vector."""
        embedding = self.encoder.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    def search(self, query: str, top_k: int = 10) -> SearchResponse:
        start_time = time.perf_counter()

        # Encode query
        query_vector = self.encode_query(query)

        # Search in Qdrant
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        # Format results
        results = []
        for hit in hits.points:
            payload = hit.payload or {}
            results.append(
                SearchResult(
                    chunk_id=str(hit.id),
                    score=float(hit.score),
                    text=payload.get("text", ""),
                    metadata=payload,
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return SearchResponse(
            results=results,
            latency_ms=latency_ms,
            strategy="dense",
            total_hits=len(results),
        )


class SparseSearcher(BaseSearcher):
    """Sparse vector search (BM25-style)."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "sparse_collection",
        sparse_vector_name: str = "bm25",
    ):
        self.client = client
        self.collection_name = collection_name
        self.sparse_vector_name = sparse_vector_name

    def _text_to_sparse_vector(self, text: str) -> models.SparseVector:
        """Convert text to sparse vector indices and values."""
        from collections import Counter

        tokens = text.lower().split()
        if not tokens:
            return models.SparseVector(indices=[], values=[])

        tf = Counter(tokens)
        indices = []
        values = []

        for token, freq in tf.items():
            idx = abs(hash(token)) % (10**9)
            indices.append(idx)
            values.append(float(freq))

        return models.SparseVector(indices=indices, values=values)

    def search(self, query: str, top_k: int = 10) -> SearchResponse:
        start_time = time.perf_counter()
        logger.info(f"Sparse search: query='{query}', top_k={top_k}")

        # Convert query to sparse vector
        sparse_query = self._text_to_sparse_vector(query)
        logger.debug(f"Sparse query vector: {len(sparse_query.indices)} indices")
        logger.debug(f"Using named vector: {self.sparse_vector_name}")

        # Search in Qdrant
        try:
            hits = self.client.query_points(
                collection_name=self.collection_name,
                using=self.sparse_vector_name,
                query=sparse_query,
                limit=top_k,
                with_payload=True,
            )
            logger.info(f"Sparse search found {len(hits.points)} results")
        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            logger.error(f"Query type: {type(sparse_query)}")
            logger.error(f"Collection: {self.collection_name}")
            logger.error(f"Named vector: {self.sparse_vector_name}")
            raise

        # Format results
        results = []
        for hit in hits.points:
            payload = hit.payload or {}
            results.append(
                SearchResult(
                    chunk_id=str(hit.id),
                    score=float(hit.score),
                    text=payload.get("text", ""),
                    metadata=payload,
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"Sparse search completed in {latency_ms:.2f}ms")

        return SearchResponse(
            results=results,
            latency_ms=latency_ms,
            strategy="sparse",
            total_hits=len(results),
        )


class HybridSearcher(BaseSearcher):
    """Hybrid search with fusion strategies (RRF, DBSF)."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "hybrid_collection",
        dense_vector_name: str = "dense",
        sparse_vector_name: str = "bm25",
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        rrf_k: int = 60,
        vector_size: int = 384,
    ):
        self.client = client
        self.collection_name = collection_name
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name
        self._encoder: SentenceTransformer | None = None
        self._model_name = dense_model
        self.rrf_k = rrf_k
        self.vector_size = vector_size

    @property
    def encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            self._encoder = SentenceTransformer(self._model_name)
        return self._encoder

    def _text_to_sparse_vector(self, text: str) -> models.SparseVector:
        """Convert text to sparse vector."""
        from collections import Counter

        tokens = text.lower().split()
        tf = Counter(tokens)
        indices = [abs(hash(t)) % (10**9) for t in tf]
        values = [float(tf[t]) for t in tf]
        return models.SparseVector(indices=indices, values=values)

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Apply Reciprocal Rank Fusion (RRF)."""
        fused_scores: dict[str, float] = defaultdict(float)

        # Add dense scores
        for rank, item in enumerate(dense_results, start=1):
            fused_scores[item["id"]] += 1.0 / (self.rrf_k + rank)

        # Add sparse scores
        for rank, item in enumerate(sparse_results, start=1):
            fused_scores[item["id"]] += 1.0 / (self.rrf_k + rank)

        # Sort by fused score
        sorted_items = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        return [{"id": did, "score": score} for did, score in sorted_items]

    def _dbsf_fusion(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Apply Distribution-Based Score Fusion (DBSF)."""

        # Normalize scores within each list
        def normalize_scores(results: list[dict]) -> dict[str, float]:
            if not results:
                return {}
            max_score = max(r["score"] for r in results)
            min_score = min(r["score"] for r in results)
            range_score = max_score - min_score if max_score != min_score else 1.0

            return {r["id"]: (r["score"] - min_score) / range_score for r in results}

        dense_normalized = normalize_scores(dense_results)
        sparse_normalized = normalize_scores(sparse_results)

        # Combine scores
        combined: dict[str, float] = defaultdict(float)
        for doc_id, score in dense_normalized.items():
            combined[doc_id] += score
        for doc_id, score in sparse_normalized.items():
            combined[doc_id] += score

        # Sort and return top-k
        sorted_items = sorted(
            combined.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        return [{"id": did, "score": score} for did, score in sorted_items]

    def search(
        self,
        query: str,
        top_k: int = 10,
        fusion_strategy: str = "rrf",
    ) -> SearchResponse:
        start_time = time.perf_counter()
        logger.info(f"Hybrid search: query='{query}', top_k={top_k}, strategy={fusion_strategy}")

        # Encode query for dense search
        query_embedding = self.encoder.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()
        logger.debug(f"Dense query embedding: {len(query_embedding)} dimensions")

        # Convert query to sparse vector
        sparse_query = self._text_to_sparse_vector(query)
        logger.debug(f"Sparse query vector: {len(sparse_query.indices)} indices")

        # Perform dense search
        try:
            dense_hits = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                using=self.dense_vector_name,
                limit=top_k * 2,
                with_payload=True,
            )
            logger.info(f"Dense search found {len(dense_hits.points)} results")
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            raise

        # Perform sparse search
        try:
            sparse_hits = self.client.query_points(
                collection_name=self.collection_name,
                using=self.sparse_vector_name,
                query=sparse_query,
                limit=top_k * 2,
                with_payload=True,
            )
            logger.info(f"Sparse search found {len(sparse_hits.points)} results")
        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            logger.error(f"Query type: {type(sparse_query)}")
            logger.error(f"Collection: {self.collection_name}")
            logger.error(f"Named vector: {self.sparse_vector_name}")
            raise

        # Format results for fusion
        dense_results = [{"id": str(hit.id), "score": hit.score, "payload": hit.payload} for hit in dense_hits.points]
        sparse_results = [{"id": str(hit.id), "score": hit.score, "payload": hit.payload} for hit in sparse_hits.points]

        # Apply fusion
        if fusion_strategy.lower() == "rrf":
            fused = self._reciprocal_rank_fusion(dense_results, sparse_results, top_k)
        elif fusion_strategy.lower() == "dbsf":
            fused = self._dbsf_fusion(dense_results, sparse_results, top_k)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")

        # Build final results with payloads
        payload_map = {}
        for item in dense_results + sparse_results:
            if item["id"] not in payload_map and item.get("payload"):
                payload_map[item["id"]] = item["payload"]

        results = [
            SearchResult(
                chunk_id=item["id"],
                score=item["score"],
                text=payload_map.get(item["id"], {}).get("text", ""),
                metadata=payload_map.get(item["id"], {}),
            )
            for item in fused
        ]

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"Hybrid search completed in {latency_ms:.2f}ms")

        return SearchResponse(
            results=results,
            latency_ms=latency_ms,
            strategy=f"hybrid_{fusion_strategy}",
            total_hits=len(results),
        )


class ColBERTSearcher(BaseSearcher):
    """ColBERT-style late interaction search."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "hybrid_with_colbert",
        dense_vector_name: str = "dense",
        colbert_vector_name: str = "late_interaction",
        sparse_vector_name: str = "bm25",
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        colbert_model: str = "jinaai/jina-colbert-v2",
        vector_size: int = 384,
    ):
        self.client = client
        self.collection_name = collection_name
        self.dense_vector_name = dense_vector_name
        self.colbert_vector_name = colbert_vector_name
        self.sparse_vector_name = sparse_vector_name
        self._encoder: SentenceTransformer | None = None
        self._model_name = dense_model
        self.colbert_model = colbert_model
        self.vector_size = vector_size

    @property
    def encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            self._encoder = SentenceTransformer(self._model_name)
        return self._encoder

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_prefetch: bool = True,
    ) -> SearchResponse:
        start_time = time.perf_counter()

        # Encode query for dense search
        query_embedding = self.encoder.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

        # Convert query to sparse vector
        from collections import Counter

        tokens = query.lower().split()
        tf = Counter(tokens)
        indices = [abs(hash(t)) % (10**9) for t in tf]
        values = [float(tf[t]) for t in tf]
        sparse_query = models.SparseVector(indices=indices, values=values)

        if use_prefetch:
            # Use prefetch for hybrid retrieval with ColBERT late interaction
            hits = self.client.query_points(
                collection_name=self.collection_name,
                query=models.NamedVector(
                    name=self.colbert_vector_name,
                    vector=query_embedding,
                ),
                prefetch=[
                    models.Prefetch(
                        query=query_embedding,
                        using=self.dense_vector_name,
                        limit=top_k * 2,
                    ),
                    models.Prefetch(
                        using=self.sparse_vector_name,
                        query=sparse_query,
                        limit=top_k * 2,
                    ),
                ],
                limit=top_k,
                with_payload=True,
            )
        else:
            # Direct ColBERT search
            hits = self.client.query_points(
                collection_name=self.collection_name,
                query=models.NamedVector(
                    name=self.colbert_vector_name,
                    vector=query_embedding,
                ),
                limit=top_k,
                with_payload=True,
            )

        # Format results
        results = []
        for hit in hits.points:
            payload = hit.payload or {}
            results.append(
                SearchResult(
                    chunk_id=str(hit.id),
                    score=float(hit.score),
                    text=payload.get("text", ""),
                    metadata=payload,
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return SearchResponse(
            results=results,
            latency_ms=latency_ms,
            strategy="colbert",
            total_hits=len(results),
        )
