"""
Code search implementation using dual encoders.

Implements code search with:
- General-purpose encoder (sentence-transformers)
- Code-specific encoder (jina-embeddings-v2-base-code)
- Hybrid fusion of both encoders
"""

import re
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


def textify_code_structure(chunk: dict[str, Any]) -> str:
    """
    Convert code structure into natural language representation.

    Args:
        chunk: Dictionary-like representation of code structure

    Returns:
        Natural language description of the code structure
    """
    try:
        import inflection
    except ImportError:
        logger.warning("inflection not installed, using simple text conversion")
        # Fallback without inflection
        name = chunk.get("name", "")
        signature = chunk.get("signature", "")
        docstring = chunk.get("docstring", "")
        context = chunk.get("context", {})
        return f"{chunk.get('code_type', '')} {name} {docstring} {signature} {context}"

    # Get rid of camel case / snake case
    name = inflection.humanize(inflection.underscore(chunk.get("name", "")))
    signature = inflection.humanize(inflection.underscore(chunk.get("signature", "")))

    # Check if docstring is provided
    docstring = ""
    if chunk.get("docstring"):
        docstring = f"that does {chunk['docstring']} "

    # Extract location
    context = chunk.get("context", {})
    context_str = f"module {context.get('module', '')} " f"file {context.get('file_name', '')}"
    if context.get("struct_name"):
        struct_name = inflection.humanize(inflection.underscore(context["struct_name"]))
        context_str = f"defined in struct {struct_name} {context_str}"

    # Combine
    text_representation = (
        f"{chunk.get('code_type', '')} {name} " f"{docstring}" f"defined as {signature} " f"{context_str}"
    )

    # Remove special characters and concatenate tokens
    tokens = re.split(r"\W", text_representation)
    tokens = filter(lambda x: x, tokens)
    return " ".join(tokens)


@dataclass
class CodeSearchResult:
    """Represents a code search result."""

    chunk_id: str
    score: float
    text: str
    code_snippet: str
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.chunk_id,
            "score": self.score,
            "text": self.text,
            "code_snippet": self.code_snippet,
            "metadata": self.metadata or {},
        }


@dataclass
class CodeSearchResponse:
    """Complete code search response."""

    results: list[CodeSearchResult]
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


class CodeSearcher:
    """
    Code searcher using dual encoders.

    Uses both general-purpose and code-specific encoders for better code search.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "code_collection",
        text_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        code_model: str = "jinaai/jina-embeddings-v2-base-code",
        text_vector_size: int = 384,
        code_vector_size: int = 768,
    ):
        """
        Initialize code searcher.

        Args:
            client: Qdrant client
            collection_name: Name of the code collection
            text_model: General-purpose encoder model
            code_model: Code-specific encoder model
            text_vector_size: Dimension of text vectors
            code_vector_size: Dimension of code vectors
        """
        self.client = client
        self.collection_name = collection_name
        self.text_model_name = text_model
        self.code_model_name = code_model
        self.text_vector_size = text_vector_size
        self.code_vector_size = code_vector_size
        self._text_encoder: SentenceTransformer | None = None
        self._code_encoder: SentenceTransformer | None = None

    @property
    def text_encoder(self) -> SentenceTransformer:
        """Lazy-load text encoder."""
        if self._text_encoder is None:
            logger.info(f"Loading text encoder: {self.text_model_name}")
            self._text_encoder = SentenceTransformer(self.text_model_name)
        return self._text_encoder

    @property
    def code_encoder(self) -> SentenceTransformer:
        """Lazy-load code encoder."""
        if self._code_encoder is None:
            logger.info(f"Loading code encoder: {self.code_model_name}")
            self._code_encoder = SentenceTransformer(self.code_model_name)
        return self._code_encoder

    def create_collection(self, recreate: bool = True) -> None:
        """
        Create collection for code search with named vectors.

        Args:
            recreate: Whether to recreate if collection exists
        """
        if self.client.collection_exists(self.collection_name):
            if recreate:
                self.client.delete_collection(self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
            else:
                logger.info(f"Collection already exists: {self.collection_name}")
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "text": models.VectorParams(
                    size=self.text_vector_size,
                    distance=models.Distance.COSINE,
                ),
                "code": models.VectorParams(
                    size=self.code_vector_size,
                    distance=models.Distance.COSINE,
                ),
            },
        )
        logger.info(f"Created code collection: {self.collection_name}")

    def index_code_structures(
        self,
        structures: list[dict[str, Any]],
        batch_size: int = 64,
    ) -> None:
        """
        Index code structures into Qdrant.

        Args:
            structures: List of code structure dictionaries
            batch_size: Batch size for upserting
        """
        logger.info(f"Indexing {len(structures)} code structures")

        # Convert structures to text and code representations
        text_representations = [textify_code_structure(s) for s in structures]
        code_snippets = [s.get("context", {}).get("snippet", "") for s in structures]

        # Encode in batches
        points = []
        for i in range(0, len(structures), batch_size):
            batch = structures[i : i + batch_size]
            text_batch = text_representations[i : i + batch_size]
            code_batch = code_snippets[i : i + batch_size]

            # Encode
            text_vectors = self.text_encoder.encode(
                text_batch,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).tolist()

            code_vectors = self.code_encoder.encode(
                code_batch,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).tolist()

            # Create points
            for j, (structure, text_vec, code_vec) in enumerate(zip(batch, text_vectors, code_vectors)):
                import uuid

                points.append(
                    models.PointStruct(
                        id=uuid.uuid4().hex,
                        vector={
                            "text": text_vec,
                            "code": code_vec,
                        },
                        payload=structure,
                    )
                )

            # Upsert batch
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            points = []

        logger.info("Code indexing complete")

    def search_with_text_encoder(
        self,
        query: str,
        top_k: int = 5,
    ) -> CodeSearchResponse:
        """
        Search using text encoder only.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            Code search response
        """
        start_time = time.perf_counter()

        # Encode query
        query_vector = self.text_encoder.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

        # Search
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="text",
            limit=top_k,
            with_payload=True,
        )

        # Format results
        results = []
        for hit in hits.points:
            payload = hit.payload or {}
            results.append(
                CodeSearchResult(
                    chunk_id=str(hit.id),
                    score=float(hit.score),
                    text=textify_code_structure(payload),
                    code_snippet=payload.get("context", {}).get("snippet", ""),
                    metadata=payload,
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return CodeSearchResponse(
            results=results,
            latency_ms=latency_ms,
            strategy="code_text_encoder",
            total_hits=len(results),
        )

    def search_with_code_encoder(
        self,
        query: str,
        top_k: int = 5,
    ) -> CodeSearchResponse:
        """
        Search using code encoder only.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            Code search response
        """
        start_time = time.perf_counter()

        # Encode query
        query_vector = self.code_encoder.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

        # Search
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="code",
            limit=top_k,
            with_payload=True,
        )

        # Format results
        results = []
        for hit in hits.points:
            payload = hit.payload or {}
            results.append(
                CodeSearchResult(
                    chunk_id=str(hit.id),
                    score=float(hit.score),
                    text=textify_code_structure(payload),
                    code_snippet=payload.get("context", {}).get("snippet", ""),
                    metadata=payload,
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return CodeSearchResponse(
            results=results,
            latency_ms=latency_ms,
            strategy="code_code_encoder",
            total_hits=len(results),
        )

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
    ) -> CodeSearchResponse:
        """
        Search using both encoders with hybrid fusion.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            Code search response
        """
        start_time = time.perf_counter()

        # Encode queries
        text_vector = self.text_encoder.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

        code_vector = self.code_encoder.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

        # Batch search with both encoders
        responses = self.client.query_batch_points(
            self.collection_name,
            requests=[
                models.QueryRequest(
                    query=text_vector,
                    using="text",
                    with_payload=True,
                    limit=top_k,
                ),
                models.QueryRequest(
                    query=code_vector,
                    using="code",
                    with_payload=True,
                    limit=top_k,
                ),
            ],
        )

        # Combine results (simple averaging for now)
        text_hits = responses[0].points
        code_hits = responses[1].points

        # Score combination
        combined_scores = {}
        for hit in text_hits:
            combined_scores[str(hit.id)] = hit.score * 0.5

        for hit in code_hits:
            if str(hit.id) in combined_scores:
                combined_scores[str(hit.id)] += hit.score * 0.5
            else:
                combined_scores[str(hit.id)] = hit.score * 0.5

        # Sort by combined score
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)

        # Get full results
        all_hits = {str(hit.id): hit for hit in text_hits + code_hits}
        results = []
        for chunk_id in sorted_ids[:top_k]:
            hit = all_hits[chunk_id]
            payload = hit.payload or {}
            results.append(
                CodeSearchResult(
                    chunk_id=chunk_id,
                    score=combined_scores[chunk_id],
                    text=textify_code_structure(payload),
                    code_snippet=payload.get("context", {}).get("snippet", ""),
                    metadata=payload,
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return CodeSearchResponse(
            results=results,
            latency_ms=latency_ms,
            strategy="code_hybrid",
            total_hits=len(results),
        )

    def search_grouped(
        self,
        query: str,
        top_k: int = 5,
        group_by: str = "context.module",
        group_size: int = 1,
    ) -> CodeSearchResponse:
        """
        Search with grouping by module for diversity.

        Args:
            query: Search query
            top_k: Number of groups to return
            group_by: Field to group by
            group_size: Number of results per group

        Returns:
            Code search response
        """
        start_time = time.perf_counter()

        # Encode query
        query_vector = self.code_encoder.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

        # Search with grouping
        results = self.client.query_points_groups(
            collection_name=self.collection_name,
            using="code",
            query=query_vector,
            group_by=group_by,
            limit=top_k,
            group_size=group_size,
            with_payload=True,
        )

        # Format results
        formatted_results = []
        for group in results.groups:
            for hit in group.hits:
                payload = hit.payload or {}
                formatted_results.append(
                    CodeSearchResult(
                        chunk_id=str(hit.id),
                        score=float(hit.score),
                        text=textify_code_structure(payload),
                        code_snippet=payload.get("context", {}).get("snippet", ""),
                        metadata=payload,
                    )
                )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return CodeSearchResponse(
            results=formatted_results,
            latency_ms=latency_ms,
            strategy="code_grouped",
            total_hits=len(formatted_results),
        )
