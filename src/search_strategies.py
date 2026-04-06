"""
Search strategies for thesis experiments:
 - Dense (Qdrant HNSW)
 - Sparse (BM25 using rank_bm25 OR Qdrant Sparse Vectors)
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
    # Общие параметры
    tokenizer: Optional[Any] = None
    top_k: int = 10

    # BM25 параметры (для in-memory SparseSearcher)
    k1: float = 1.5
    b: float = 0.75

    # Qdrant-specific параметры (для QdrantSparseSearcher)
    collection_name: Optional[str] = None
    qdrant_url: str = "http://localhost:6333"
    sparse_vector_field: str = "sparse_vector"
    vocabulary: Optional[Dict[str, int]] = None
    use_idf: bool = True
    on_disk: bool = False
    recreate_collection: bool = True


@dataclass
class QdrantSparseConfig:
    collection_name: str
    sparse_vector_field: str = "sparse_vector"
    top_k: int = 10
    with_payload: bool = True
    vocabulary: Optional[Dict[str, int]] = None # vocabulary for token->index mapping (if using BM25-style sparse encoding)
    use_idf: bool = False  # Use IDF weighting (requires pre-computed IDF values in vocabulary)


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



def _now():
    return time.perf_counter()


class DenseSearcher:
    """
    Dense search wrapper around a Qdrant collection (HNSW).

    Uses QdrantClient.search with search params to set ef_search.
    """

    def __init__(self, client: QdrantClient, cfg: DenseConfig):
        self.client = client
        self.cfg = cfg
        logger.info("DenseSearcher initialized (collection=%s, ef_search=%d)", cfg.collection_name, cfg.ef_search)

    def search(self, query_vector: List[float], top_k: int = None) -> Tuple[List[Dict], Dict]:
        """
        Search using dense vectors.
        """
        top_k = top_k or self.cfg.top_k
        t0 = time.perf_counter()

        try:
            hits = self.client.query_points(
                collection_name=self.cfg.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )

            results = []
            for hit in hits.points:
                results.append({
                    "id": str(hit.id),
                    "score": hit.score,
                    "text": hit.payload.get("text", "") if hit.payload else ""
                })

            elapsed = time.perf_counter() - t0

            return results, {"time_s": elapsed, "n_results": len(results)}

        except Exception as e:
            logger.error("Qdrant dense search failed: %s", e)
            return [], {"time_s": time.perf_counter() - t0, "n_results": 0, "error": str(e)}


# Sparse Searcher (BM25 - in-memory)
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


# Sparse Searcher (Qdrant Native Sparse Vectors)
class QdrantSparseSearcher:
    """
    Qdrant-based sparse search using native sparse vectors.

    Drop-in replacement for SparseSearcher with identical interface:
        __init__(docs: List[Dict], cfg: SparseConfig)
        search(query_text: str, top_k: int) -> Tuple[List[Dict], Dict]

    Automatically handles:
        - Qdrant client initialization
        - Collection creation with sparse vector config
        - Vocabulary building and document upsert
        - Query tokenization and sparse vector conversion
    """

    def __init__(self, docs: List[Dict[str, Any]], cfg: SparseConfig):
        """
        Initialize QdrantSparseSearcher.

        Args:
            docs: List[{"id": <chunk_id>, "text": "<raw text>"}]
            cfg: SparseConfig with Qdrant-specific parameters
        """
        self.cfg = cfg
        self.tokenizer = cfg.tokenizer or _default_tokenizer

        # Store docs metadata for vocabulary building
        self.docs = docs
        self.ids = [d["id"] for d in docs]

        # Initialize Qdrant client
        try:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(url=cfg.qdrant_url, prefer_grpc=False)
        except ImportError:
            raise ImportError("qdrant_client is required for QdrantSparseSearcher")

        # Validate collection name
        if not cfg.collection_name:
            raise ValueError("collection_name must be specified in SparseConfig for QdrantSparseSearcher")

        # Build vocabulary and initialize collection
        self._initialize(cfg)

        logger.info("QdrantSparseSearcher initialized (collection=%s, vocab_size=%d)",
                    cfg.collection_name, len(self.vocabulary) if self.vocabulary else 0)

    def _initialize(self, cfg: SparseConfig):
        """Build vocabulary, create collection, and upsert documents."""

        # Build vocabulary from docs
        self.vocabulary = cfg.vocabulary or self._build_vocabulary(self.docs, self.tokenizer)

        # Create collection if needed
        if cfg.recreate_collection or not self.client.collection_exists(cfg.collection_name):
            self._create_sparse_collection(cfg.collection_name, cfg.sparse_vector_field, cfg.on_disk)

        # Upsert documents as sparse vectors
        self._upsert_documents(self.docs, batch_size=100)

    def _build_vocabulary(self, docs: List[Dict[str, Any]], tokenizer) -> Dict[str, int]:
        """Build vocabulary with token->index mapping and optional IDF values."""
        import math
        from tqdm.auto import tqdm

        all_tokens = set()
        doc_token_counts = {}

        for doc in tqdm(docs, desc="Building vocabulary"):
            text = doc.get("text", "")
            tokens = tokenizer(text)
            unique_tokens = set(tokens)
            all_tokens.update(unique_tokens)
            for token in unique_tokens:
                doc_token_counts[token] = doc_token_counts.get(token, 0) + 1

        vocabulary = {token: idx for idx, token in enumerate(sorted(all_tokens))}

        # Compute IDF values (BM25-style)
        n_docs = len(docs)
        idf_values = {}
        for token, doc_count in doc_token_counts.items():
            # BM25 IDF formula with smoothing
            idf_values[token] = math.log((n_docs - doc_count + 0.5) / (doc_count + 0.5) + 1.0)

        vocabulary["_idf"] = idf_values
        logger.info("Vocabulary built: %d tokens", len(vocabulary) - 1)
        return vocabulary

    def _create_sparse_collection(self, collection_name: str, vector_field: str, on_disk: bool):
        """Create Qdrant collection with sparse vector support."""
        from qdrant_client import models

        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config={},  # Empty for sparse-only
            sparse_vectors_config={
                vector_field: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=on_disk)
                )
            }
        )
        logger.info("Sparse collection created: %s", collection_name)

    def _text_to_sparse_vector(self, text: str) -> models.SparseVector:
        """Convert text to sparse vector using vocabulary and BM25-style weighting."""
        from qdrant_client import models

        tokens = self.tokenizer(text)

        # Calculate TF
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        indices, values = [], []
        idf_map = self.vocabulary.get("_idf", {})

        for token, freq in tf.items():
            if token in self.vocabulary and token != "_idf":
                idx = self.vocabulary[token]
                # Apply BM25-style TF-IDF weighting (k1 and b are noted but Qdrant uses raw scores)
                idf_val = idf_map.get(token, 1.0)
                weight = freq * idf_val  # Simplified: Qdrant handles scoring internally
                indices.append(idx)
                values.append(weight)

        return models.SparseVector(indices=indices, values=values)

    def _upsert_documents(self, docs: List[Dict[str, Any]], batch_size: int = 100):
        """Index documents into Qdrant as sparse vectors."""
        from qdrant_client import models
        from tqdm.auto import tqdm

        points = []
        for doc in tqdm(docs, desc="Upserting sparse vectors"):
            sparse_vec = self._text_to_sparse_vector(doc["text"])

            if not sparse_vec.indices:  # Skip empty vectors
                continue

            points.append(
                models.PointStruct(
                    id=doc["id"],
                    vector={self.cfg.sparse_vector_field: sparse_vec},
                    payload={"text": doc["text"], "chunk_id": doc["id"]}
                )
            )

            if len(points) >= batch_size:
                self.client.upsert(collection_name=self.cfg.collection_name, points=points)
                points = []

        if points:
            self.client.upsert(collection_name=self.cfg.collection_name, points=points)

        logger.info("Upserted %d documents to sparse collection", len(docs))

    def search(self, query_text: str, top_k: Optional[int] = None) -> Tuple[List[Dict], Dict]:
        """
        Search using sparse vectors in Qdrant.

        Identical interface to SparseSearcher.search()
        """
        from qdrant_client import models

        k = top_k or self.cfg.top_k
        t0 = _now()

        try:
            # Convert query to sparse vector
            sparse_query = self._text_to_sparse_vector(query_text)

            # Search in Qdrant
            hits = self.client.query_points(
                collection_name=self.cfg.collection_name,
                query=models.NamedSparseVector(
                    name=self.cfg.sparse_vector_field,
                    vector=sparse_query
                ),
                limit=k,
                with_payload=True,
            )

            # Format results to match SparseSearcher output
            results = []
            for hit in hits.points:
                results.append({
                    "id": str(hit.id),
                    "score": float(hit.score),
                    "payload": hit.payload if hit.payload else {}
                })

            elapsed = _now() - t0
            meta = {
                "time_s": elapsed,
                "qps": 1.0 / elapsed if elapsed > 0 else float("inf"),
                "k": k,
                "searcher": "qdrant_sparse"
            }
            return results, meta

        except Exception as e:
            logger.error("Qdrant sparse search failed: %s", e)
            elapsed = _now() - t0
            return [], {
                "time_s": elapsed,
                "qps": 0,
                "k": k,
                "error": str(e),
                "searcher": "qdrant_sparse"
            }

    def cleanup(self):
        """Delete the collection (useful for testing)."""
        try:
            self.client.delete_collection(self.cfg.collection_name)
            logger.info("Deleted collection: %s", self.cfg.collection_name)
        except Exception as e:
            logger.warning("Failed to delete collection %s: %s", self.cfg.collection_name, e)


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
    Hybrid searcher that runs both Sparse and Dense and fuses using RRF.

    Supports two sparse backends:
      - BM25 (SparseSearcher, in-memory)
      - Qdrant Sparse Vectors (QdrantSparseSearcher)
    """

    def __init__(
        self,
        dense_searcher: DenseSearcher,
        sparse_searcher: Any,  # SparseSearcher OR QdrantSparseSearcher
        cfg: HybridConfig = HybridConfig()
    ):
        self.dense = dense_searcher
        self.sparse = sparse_searcher
        self.cfg = cfg

        # Detect sparse searcher type
        self.sparse_type = "bm25" if isinstance(sparse_searcher, SparseSearcher) else "qdrant_sparse"
        logger.info("HybridSearcher initialized (sparse_backend=%s)", self.sparse_type)

    def search(self, query_text: str, query_vector: Optional[Sequence[float]] = None, top_k: Optional[int] = None) -> Tuple[List[Dict], Dict]:
        """
        Run sparse and dense retrieval and fuse with RRF.

        Parameters:
            query_text: textual query for sparse retriever
            query_vector: vector for dense retriever (if None, dense call will fail)
        """
        k = top_k or self.cfg.top_k

        sparse_res, sparse_meta = self.sparse.search(query_text, top_k=k)

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
            "components": {
                "sparse": sparse_meta,
                "dense": dense_meta,
                "rrf_fuse_s": duration
            },
            "sparse_backend": self.sparse_type
        }
        return fused, meta


# Filtered search wrapper
class FilteredDenseSearcher(DenseSearcher):
    """
    Dense searcher with payload filtering support.
    """

    def __init__(self, client: QdrantClient, cfg: DenseConfig):
        super().__init__(client, cfg)

    def search_with_filter(self, query_vector: Sequence[float], filter: Optional[models.Filter], top_k: Optional[int] = None) -> Tuple[List[Dict], Dict]:
        top_k = top_k or self.cfg.top_k
        t0 = time.perf_counter()

        try:
            hits = self.client.query_points(
                collection_name=self.cfg.collection_name,
                query=query_vector,
                query_filter=filter,
                limit=top_k,
                with_payload=True,
            )

            results = []
            for hit in hits.points:
                results.append({
                    "id": str(hit.id),
                    "score": hit.score,
                    "text": hit.payload.get("text", "") if hit.payload else ""
                })

            elapsed = time.perf_counter() - t0
            return results, {"time_s": elapsed, "n_results": len(results)}

        except Exception as e:
            logger.error("Qdrant filtered dense search failed: %s", e)
            return [], {"time_s": time.perf_counter() - t0, "n_results": 0, "error": str(e)}