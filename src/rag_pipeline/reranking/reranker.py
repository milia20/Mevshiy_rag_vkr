"""
Reranking using cross-encoders.

Reranks retrieved documents based on query-document relevance scores.
"""

import logging
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from ..retrieval.searchers import SearchResponse, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RerankerConfig:
    """Configuration for reranker."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5


class Reranker:
    """
    Reranker using cross-encoder models.

    Takes retrieved results and reranks them based on
    query-document relevance scores.
    """

    def __init__(self, config: RerankerConfig | None = None):
        """
        Initialize reranker.

        Args:
            config: Reranker configuration
        """
        self.config = config or RerankerConfig()
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        """Lazy-load cross-encoder model."""
        if self._model is None:
            logger.info("Loading reranker model: %s", self.config.model_name)
            self._model = CrossEncoder(self.config.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """
        Rerank search results based on query-document relevance.

        Args:
            query: The search query
            results: List of retrieved search results
            top_k: Number of results to return after reranking

        Returns:
            Reranked list of SearchResult objects
        """
        if not results:
            return []

        top_k = top_k or self.config.top_k
        top_k = min(top_k, len(results))

        # Prepare pairs for cross-encoder
        pairs = [(query, result.text) for result in results]

        # Get relevance scores
        scores = self.model.predict(pairs)

        # Sort by score descending
        scored_results = list(zip(results, scores, strict=False))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Take top-k and update scores
        reranked = []
        for result, score in scored_results[:top_k]:
            reranked.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    score=float(score),
                    text=result.text,
                    metadata=result.metadata,
                )
            )

        logger.info("Reranked %d results, returning top %d", len(results), top_k)
        return reranked

    def rerank_response(
        self,
        query: str,
        response: SearchResponse,
        top_k: int | None = None,
    ) -> SearchResponse:
        """
        Rerank a complete search response.

        Args:
            query: The search query
            response: SearchResponse to rerank
            top_k: Number of results to return

        Returns:
            New SearchResponse with reranked results
        """
        import time

        start_time = time.perf_counter()

        reranked_results = self.rerank(query, response.results, top_k)

        latency_ms = (time.perf_counter() - start_time) * 1000

        return SearchResponse(
            results=reranked_results,
            latency_ms=response.latency_ms + latency_ms,
            strategy=f"{response.strategy}_reranked",
            total_hits=len(reranked_results),
        )
