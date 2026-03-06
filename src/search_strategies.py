"""
Search strategies for thesis experiments:
 - Dense (Qdrant HNSW)
 - Sparse (BM25 using rank_bm25)
 - Hybrid (Reciprocal Rank Fusion)
 - Filtered (Qdrant with payload filters)

Each searcher exposes a `.search(query_text, top_k)` method that returns:
    (results: List[Dict], meta: Dict)

Where results are ordered lists of dicts:
    {"id": "<chunk_id>", "score": float, "payload": {...}}

And meta contains timing and QPS info:
    {"time_s": 0.012, "qps": 83.3}
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# -------------------------
# Data classes for configs
# -------------------------

@dataclass
class DenseConfig:
    collection_name: str
    vector_field: str = "vector"  # default vector field name in Qdrant
    ef_search: int = 64
    with_payload: bool = True
    prefer_grpc: bool = False
    top_k: int = 10


@dataclass
class SparseConfig:
    tokenizer: Optional[Any] = None  # function(str)->List[str]
    top_k: int = 10


@dataclass
class HybridConfig:
    rrf_k: int = 60  # RRF constant
    top_k: int = 10


@dataclass
class FilterConfig:
    collection_name: str
    vector_field: str = "vector"
    ef_search: int = 64
    top_k: int = 10


# -------------------------
# Utilities
# -------------------------


def _now():
    return time.perf_counter()


# -------------------------
# Dense searcher (Qdrant)
# -------------------------


class DenseSearcher:
    """
    Dense search wrapper around a Qdrant collection (HNSW).

    Uses QdrantClient.search with search params to set ef_search.
    """

    def __init__(self, client: QdrantClient, cfg: DenseConfig):
        self.client = client
        self.cfg = cfg
        logger.info("DenseSearcher initialized (collection=%s, ef_search=%d)", cfg.collection_name, cfg.ef_search)

    def search(self, query_vector: Sequence[float], top_k: Optional[int] = None, filter: Optional[models.Filter] = None) -> Tuple[List[Dict], Dict]:
        """
        Perform vector search on Qdrant.

        Parameters
        ----------
        query_vector : Sequence[float]
            A single vector (list/np.ndarray)
        top_k : Optional[int]
            override cfg.top_k
        filter : Optional[models.Filter]
            Qdrant filter object for payload filtering

        Returns
        -------
        results : list of dicts {"id":..., "score":..., "payload":...}
        meta : dict with timing (time_s, qps)
        """
        k = top_k or self.cfg.top_k
        t0 = _now()

        params = {"hnsw_ef": self.cfg.ef_search}

        # QdrantClient.search signature: client.search(collection_name, query_vector, limit, filter=..., with_payload=...)
        try:
            hits = self.client.search(
                collection_name=self.cfg.collection_name,
                query_vector=list(map(float, query_vector)),
                limit=k,
                filter=filter,
                with_payload=self.cfg.with_payload,
                params=params,
            )
        except Exception as exc:
            logger.exception("Qdrant dense search failed: %s", exc)
            raise

        results = []
        for h in hits:
            # hit is a ScoredPoint / models.ScoredPoint
            results.append(
                {
                    "id": h.id,
                    "score": float(h.score) if h.score is not None else None,
                    "payload": h.payload,
                }
            )

        duration = _now() - t0
        meta = {"time_s": duration, "qps": 1.0 / duration if duration > 0 else float("inf"), "k": k}
        return results, meta


# -------------------------
# Sparse searcher (BM25)
# -------------------------


def _default_tokenizer(text: str) -> List[str]:
    # simple whitespace + lowercase tokenizer; you can replace with spaCy / nltk if needed
    return [t for t in text.lower().split() if t]


class SparseSearcher:
    """
    BM25-based lexical search built with rank_bm25 (Okapi BM25).

    The constructor expects a list of documents (each doc is dict with 'id' and 'text').
    """

    def __init__(self, docs: List[Dict[str, Any]], cfg: SparseConfig = SparseConfig()):
        """
        docs: List[{"id": <chunk_id>, "text": "<raw text>"}]
        """
        self.cfg = cfg
        self.tokenizer = cfg.tokenizer or _default_tokenizer

        # Build corpus
        self.ids = [d["id"] for d in docs]
        self.docs_text = [d["text"] for d in docs]
        self.tokenized_corpus = [self.tokenizer(t) for t in tqdm(self.docs_text, desc="Tokenizing corpus")]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info("BM25 index built (n_docs=%d)", len(self.docs_text))

    def search(self, query_text: str, top_k: Optional[int] = None) -> Tuple[List[Dict], Dict]:
        k = top_k or self.cfg.top_k
        t0 = _now()

        q_tokens = self.tokenizer(query_text)
        scores = self.bm25.get_scores(q_tokens)
        # get top indices
        top_idx = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_idx:
            results.append({"id": self.ids[int(idx)], "score": float(scores[int(idx)]), "payload": None})

        duration = _now() - t0
        meta = {"time_s": duration, "qps": 1.0 / duration if duration > 0 else float("inf"), "k": k}
        return results, meta


# -------------------------
# RRF Hybrid fusion
# -------------------------


def reciprocal_rank_fusion(result_lists: List[List[Dict]], rrf_k: int = 60, top_k: int = 10) -> List[Dict]:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    Each list in result_lists is an ordered list of dicts with at least 'id' and 'score'.
    The returned list is ordered by fused RRF score descending.

    RRF score:
        score(d) = sum_{list} 1 / (rrf_k + rank_list(d))

    rank is 1-based position in each list. If doc absent from a list, skip.

    Parameters
    ----------
    result_lists : List[List[Dict]]
    rrf_k : int
        constant (commonly 60)
    top_k : int
        number of fused results to return
    """
    from collections import defaultdict

    fused_scores = defaultdict(float)
    for res_list in result_lists:
        for rank, itm in enumerate(res_list, start=1):
            did = itm["id"]
            fused_scores[did] += 1.0 / (rrf_k + rank)

    # sort by fused score desc
    sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"id": did, "score": float(score)} for did, score in sorted_items]


class HybridSearcher:
    """
    Hybrid searcher that runs both BM25 and Dense (Qdrant) and fuses using RRF.
    """

    def __init__(self, dense_searcher: DenseSearcher, sparse_searcher: SparseSearcher, cfg: HybridConfig = HybridConfig()):
        self.dense = dense_searcher
        self.sparse = sparse_searcher
        self.cfg = cfg

    def search(self, query_text: str, query_vector: Optional[Sequence[float]] = None, top_k: Optional[int] = None) -> Tuple[List[Dict], Dict]:
        """
        Run sparse and dense retrieval and fuse with RRF.

        Parameters:
            query_text: textual query for BM25
            query_vector: vector for dense retriever (if None, dense call will fail)
        """
        k = top_k or self.cfg.top_k
        # run sparse
        sparse_res, sparse_meta = self.sparse.search(query_text, top_k=k)
        # run dense
        if query_vector is None:
            raise ValueError("query_vector is required for dense retrieval in HybridSearcher.")
        dense_res, dense_meta = self.dense.search(query_vector, top_k=k)

        # fuse
        t0 = _now()
        fused = reciprocal_rank_fusion([sparse_res, dense_res], rrf_k=self.cfg.rrf_k, top_k=k)
        duration = _now() - t0

        meta = {
            "time_s": sparse_meta["time_s"] + dense_meta["time_s"] + duration,
            "qps": 1.0 / (sparse_meta["time_s"] + dense_meta["time_s"] + duration) if (sparse_meta["time_s"] + dense_meta["time_s"] + duration) > 0 else float("inf"),
            "components": {"sparse": sparse_meta, "dense": dense_meta, "rrf_fuse_s": duration},
        }
        return fused, meta


# -------------------------
# Filtered search wrapper
# -------------------------


class FilteredDenseSearcher(DenseSearcher):
    """
    Dense searcher with payload filtering support.
    """

    def __init__(self, client: QdrantClient, cfg: DenseConfig):
        super().__init__(client, cfg)

    def search_with_filter(self, query_vector: Sequence[float], filter: Optional[models.Filter], top_k: Optional[int] = None) -> Tuple[List[Dict], Dict]:
        return super().search(query_vector=query_vector, top_k=top_k, filter=filter)