"""
Benchmarking framework to evaluate search strategies against ground truth.

Saves JSONL results with metrics per parameter setting.

Metrics:
 - Precision@1, @3, @5, @10
 - Recall@10
 - MRR@10
 - NDCG@10
 - total_search_time
 - QPS

Parameter Grids:
 - HNSW: m=[8,16,32,64], ef_construct=[100,200,300], ef_search=[50,100,200]
 - BM25: k1=[0.5,1.0,1.5,2.0], b=[0.5,0.75,1.0]
 - Hybrid RRF: k=[30,60,120,240]
"""

from __future__ import annotations

import inspect
import json
import logging
import random
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from src.search_strategies import (
    DenseConfig,
    DenseSearcher,
    HybridConfig,
    HybridSearcher,
    QdrantSparseSearcher,
    SparseConfig,
)

SparseSearcher = QdrantSparseSearcher
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass
class BenchConfig:
    """Benchmark configuration."""

    ground_truth_path: str = "../indexing/data/ground_truth_en.jsonl"
    chunks_path: str = "../indexing/processed/chunks_en.jsonl"
    embeddings_memmap: str = "../indexing/data/embeddings_en.memmap"
    embedding_dim: int = 384
    qdrant_url: str = "http://localhost:6333"
    sample_size: int = 500  # N random queries (0 for all)
    top_k: int = 10
    results_out: str = "../indexing/data/benchmarks/benchmark_results_en.jsonl"
    seed: int = 42

    # HNSW grid parameters (matching thesis requirements)
    hnsw_m_values: tuple[int, ...] = (8, 16, 32, 64)
    hnsw_ef_construct_values: tuple[int, ...] = (100, 200, 300)
    hnsw_ef_search_values: tuple[int, ...] = (50, 100, 200)

    # BM25 grid parameters
    bm25_k1_values: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    bm25_b_values: tuple[float, ...] = (0.5, 0.75, 1.0)

    # Hybrid RRF constants
    hybrid_rrf_constants: tuple[int, ...] = (30, 60, 120, 240)

    # Qdrant collection prefix
    collection_prefix: str = "thesis_bench"


# -------------------------
# Helpers: Loading data
# -------------------------


def load_ground_truth(path: str) -> dict[str, list[str]]:
    """
    Load ground truth from JSONL file.
    Expected format per line: {"query_id": ["relevant_chunk_id1", "relevant_chunk_id2", ...]}
    """
    gt: dict[str, list[str]] = {}
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                for k, v in obj.items():
                    if isinstance(v, list):
                        gt[k] = v
                    else:
                        logger.warning("Line %d: expected list of relevant IDs, got %s", line_num, type(v))
            except json.JSONDecodeError as e:
                logger.error("Line %d: JSON decode error: %s", line_num, e)
                continue
    return gt


def load_chunks(path: str) -> list[dict[str, Any]]:
    """Load chunks from JSONL file."""
    chunks: list[dict[str, Any]] = []
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.error("Line %d: JSON decode error: %s", line_num, e)
                continue
    return chunks


def load_embeddings_memmap(path: str, dim: int) -> np.ndarray:
    """Load embeddings from memory-mapped file."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Embeddings memmap not found: {path}")

    size = path_obj.stat().st_size
    n = size // (4 * dim)  # float32 = 4 bytes
    return np.memmap(path, dtype="float32", mode="r", shape=(n, dim))


def check_qdrant_availability(url: str, timeout: float = 5.0) -> bool:
    """Check if Qdrant server is available."""
    try:
        import requests

        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except Exception as e:
        logger.warning("Qdrant not available at %s: %s", url, e)
        return False


# -------------------------
# Metrics
# -------------------------


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Calculate Precision@K."""
    if k <= 0:
        return 0.0
    topk = retrieved[:k]
    if not topk:
        return 0.0
    return len(set(topk).intersection(set(relevant))) / float(k)


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Calculate Recall@K."""
    if len(relevant) == 0:
        return 0.0
    topk = retrieved[:k]
    return len(set(topk).intersection(set(relevant))) / float(len(relevant))


def mrr_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Calculate Mean Reciprocal Rank@K."""
    topk = retrieved[:k]
    for i, did in enumerate(topk, start=1):
        if did in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Calculate NDCG@K (Normalized Discounted Cumulative Gain).
    Assumes binary relevance (relevant=1, not relevant=0).
    """
    if not relevant:
        return 0.0

    topk = retrieved[:k]

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(topk):
        if doc_id in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because i starts at 0

    # IDCG (ideal DCG)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))

    return dcg / idcg if idcg > 0 else 0.0


# -------------------------
# Benchmark Runner
# -------------------------


def detect_search_signature(searcher: Any) -> str:
    """
    Detect searcher type by inspecting search method signature.
    Returns: 'hybrid', 'dense', 'sparse', or 'unknown'
    """
    if not hasattr(searcher, "search"):
        return "unknown"

    try:
        sig = inspect.signature(searcher.search)
        params = list(sig.parameters.keys())

        if "query_vector" in params and "query_text" in params:
            return "hybrid"
        elif "query_vector" in params:
            return "dense"
        elif "query_text" in params or "query" in params:
            return "sparse"
        else:
            return "unknown"
    except Exception as e:
        logger.warning("Could not inspect search signature: %s", e)
        return "unknown"


def run_benchmark_for_searcher(
    searcher: Any,
    queries: list[tuple[str, dict[str, Any]]],
    ground_truth: dict[str, list[str]],
    top_k: int = 10,
) -> dict[str, Any]:
    """
    Run benchmark using a searcher object.

    Args:
        searcher: Searcher object with .search() method
        queries: List of (query_id, {"text": ..., "vector": ...})
        ground_truth: Dict mapping query_id to list of relevant chunk_ids
        top_k: Number of results to retrieve

    Returns:
        Aggregated metrics dictionary
    """
    metrics_acc: dict[str, list[float]] = {
        "p1": [],
        "p3": [],
        "p5": [],
        "p10": [],
        "recall10": [],
        "mrr10": [],
        "ndcg10": [],
    }

    total_search_time = 0.0
    n_success = 0
    n_failed = 0

    sig_type = detect_search_signature(searcher)
    logger.info("Detected searcher type: %s", sig_type)

    for qid, qrec in tqdm(queries, desc="Benchmark queries", leave=False):
        text = qrec.get("text")
        vec = qrec.get("vector")

        t0 = time.perf_counter()

        try:
            if sig_type == "hybrid":
                res, meta = searcher.search(text, query_vector=vec, top_k=top_k)
            elif sig_type == "dense":
                res, meta = searcher.search(vec, top_k=top_k)
            elif sig_type == "sparse":
                res, meta = searcher.search(text, top_k=top_k)
            else:
                # Fallback: try different signatures
                try:
                    res, meta = searcher.search(text, top_k=top_k)
                except TypeError:
                    res, meta = searcher.search(vec, top_k=top_k)
        except Exception as exc:
            logger.debug("Search failed for query %s: %s", qid, exc)
            n_failed += 1
            continue

        elapsed = time.perf_counter() - t0
        total_search_time += meta.get("time_s", elapsed)

        retrieved_ids = [r["id"] for r in res] if isinstance(res, list) else []
        gt_ids = ground_truth.get(qid, [])

        if not gt_ids:
            # Skip queries without ground truth
            continue

        metrics_acc["p1"].append(precision_at_k(retrieved_ids, gt_ids, 1))
        metrics_acc["p3"].append(precision_at_k(retrieved_ids, gt_ids, 3))
        metrics_acc["p5"].append(precision_at_k(retrieved_ids, gt_ids, 5))
        metrics_acc["p10"].append(precision_at_k(retrieved_ids, gt_ids, 10))
        metrics_acc["recall10"].append(recall_at_k(retrieved_ids, gt_ids, 10))
        metrics_acc["mrr10"].append(mrr_at_k(retrieved_ids, gt_ids, 10))
        metrics_acc["ndcg10"].append(ndcg_at_k(retrieved_ids, gt_ids, 10))

        n_success += 1

    def mean(arr: list[float]) -> float:
        return float(np.mean(arr)) if arr else 0.0

    n_queries = n_success + n_failed

    aggregated = {
        "Precision@1": mean(metrics_acc["p1"]),
        "Precision@3": mean(metrics_acc["p3"]),
        "Precision@5": mean(metrics_acc["p5"]),
        "Precision@10": mean(metrics_acc["p10"]),
        "Recall@10": mean(metrics_acc["recall10"]),
        "MRR@10": mean(metrics_acc["mrr10"]),
        "NDCG@10": mean(metrics_acc["ndcg10"]),
        "total_search_time": round(total_search_time, 4),
        "QPS": round(n_success / total_search_time, 2) if total_search_time > 0 else float("inf"),
        "n_queries": n_queries,
        "n_success": n_success,
        "n_failed": n_failed,
    }

    return aggregated


# -------------------------
# HNSW Grid Search
# -------------------------


def run_hnsw_grid(
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
    ground_truth: dict[str, list[str]],
    cfg: BenchConfig,
) -> list[dict[str, Any]]:
    """
    Run HNSW parameter grid search.

    Parameters:
        m: Number of connections per node
        ef_construct: Size of dynamic candidate list during construction
        ef_search: Size of dynamic candidate list during search
    """
    try:
        from qdrant_client import QdrantClient, models
    except ImportError:
        logger.error("qdrant_client not installed. Skipping HNSW grid.")
        return []

    if not check_qdrant_availability(cfg.qdrant_url):
        logger.error("Qdrant server not available. Skipping HNSW grid.")
        return []

    # Prepare queries
    all_ids = [c["metadata"]["chunk_id"] for c in chunks if "metadata" in c and "chunk_id" in c["metadata"]]
    if not all_ids:
        # Fallback: use index as ID
        all_ids = [str(i) for i in range(len(chunks))]

    id_to_idx = {cid: idx for idx, cid in enumerate(all_ids)}
    id_to_idx = {cid: idx for idx, cid in enumerate(all_ids) if idx < len(chunks)}

    rng = random.Random(cfg.seed)
    n_total = len(all_ids)
    sample_n = cfg.sample_size if 0 < cfg.sample_size < n_total else n_total
    sampled_ids = rng.sample(all_ids, sample_n) if sample_n < n_total else all_ids

    queries = []
    for qid in sampled_ids:
        idx = id_to_idx.get(qid)
        if idx is None or idx >= len(chunks):
            continue
        queries.append(
            (
                qid,
                {
                    "text": chunks[idx].get("text", ""),
                    "vector": embeddings[idx].tolist() if idx < len(embeddings) else [0.0] * cfg.embedding_dim,
                },
            )
        )

    if not queries:
        logger.error("No queries prepared for HNSW grid.")
        return []

    # Initialize Qdrant client
    client = QdrantClient(url=cfg.qdrant_url)
    results = []

    for m, ef_c, ef_s in product(cfg.hnsw_m_values, cfg.hnsw_ef_construct_values, cfg.hnsw_ef_search_values):
        coll_name = f"{cfg.collection_prefix}_hnsw_m{m}_ec{ef_c}_es{ef_s}"

        logger.info("Testing HNSW: m=%d, ef_construct=%d, ef_search=%d", m, ef_c, ef_s)

        try:
            # Delete collection if exists (for clean test)
            if client.collection_exists(coll_name):
                client.delete_collection(coll_name)

            # Create collection with HNSW config
            client.create_collection(
                collection_name=coll_name,
                vectors_config=models.VectorParams(size=cfg.embedding_dim, distance=models.Distance.COSINE),
                hnsw_config=models.HnswConfigDiff(m=m, ef_construct=ef_c),
            )

            # Upload vectors in batches
            batch_size = 500
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i : i + batch_size]
                batch_embeddings = embeddings[i : i + batch_size]
                batch_ids = all_ids[i : i + batch_size]

                points = []
                for j, (chunk, emb, cid) in enumerate(zip(batch_chunks, batch_embeddings, batch_ids, strict=False)):
                    if i + j >= len(embeddings):
                        break
                    points.append(
                        models.PointStruct(
                            id=cid if isinstance(cid, (int, str)) else str(cid),
                            vector=emb.tolist() if hasattr(emb, "tolist") else list(emb),
                            payload={"text": chunk.get("text", "")},
                        )
                    )

                if points:
                    client.upsert(collection_name=coll_name, points=points)

            # Wait for indexing
            time.sleep(1)

            # Create dense searcher
            dense_cfg = DenseConfig(collection_name=coll_name, ef_search=ef_s, top_k=cfg.top_k)
            dense_searcher = DenseSearcher(client=client, cfg=dense_cfg)

            # Run benchmark
            agg = run_benchmark_for_searcher(dense_searcher, queries, ground_truth, top_k=cfg.top_k)

            entry = {
                "method": "hnsw",
                "m": m,
                "ef_construct": ef_c,
                "ef_search": ef_s,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                **agg,
            }
            results.append(entry)

            # Cleanup collection
            client.delete_collection(coll_name)

        except Exception as e:
            logger.error("HNSW grid failed for m=%d, ef_c=%d, ef_s=%d: %s", m, ef_c, ef_s, e)
            entry = {
                "method": "hnsw",
                "m": m,
                "ef_construct": ef_c,
                "ef_search": ef_s,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e),
                "Precision@1": 0.0,
                "Precision@3": 0.0,
                "Precision@5": 0.0,
                "Precision@10": 0.0,
                "Recall@10": 0.0,
                "MRR@10": 0.0,
                "NDCG@10": 0.0,
                "total_search_time": 0.0,
                "QPS": 0.0,
                "n_queries": 0,
                "n_success": 0,
                "n_failed": len(queries),
            }
            results.append(entry)

    return results


# -------------------------
# BM25 Grid Search
# -------------------------


def run_bm25_grid(
    chunks: list[dict[str, Any]],
    ground_truth: dict[str, list[str]],
    cfg: BenchConfig,
) -> list[dict[str, Any]]:
    """
    Run BM25 parameter grid search.

    Parameters:
        k1: Term frequency saturation parameter
        b: Length normalization parameter
    """
    # Prepare docs
    docs = []
    for i, c in enumerate(chunks):
        cid = c.get("metadata", {}).get("chunk_id", str(i))
        docs.append({"id": cid, "text": c.get("text", "")})

    if not docs:
        logger.error("No documents for BM25 grid.")
        return []

    # Prepare queries
    all_ids = [d["id"] for d in docs]
    rng = random.Random(cfg.seed)
    n_total = len(all_ids)
    sample_n = cfg.sample_size if 0 < cfg.sample_size < n_total else n_total
    sampled_ids = rng.sample(all_ids, sample_n) if sample_n < n_total else all_ids

    id_to_doc = {d["id"]: d for d in docs}
    queries = [(sid, {"text": id_to_doc[sid]["text"]}) for sid in sampled_ids if sid in id_to_doc]

    if not queries:
        logger.error("No queries prepared for BM25 grid.")
        return []

    results = []

    for k1, b in product(cfg.bm25_k1_values, cfg.bm25_b_values):
        logger.info("Testing BM25: k1=%.2f, b=%.2f", k1, b)
        use_qdrant_sparse = True
        try:
            sparse_cfg = SparseConfig(
                top_k=cfg.top_k,
                k1=k1,
                b=b,
                # Qdrant-specific params
                collection_name=f"{cfg.collection_prefix}_sparse_k1{k1}_b{b}",
                qdrant_url=cfg.qdrant_url,
            )

            def create_sparse_searcher(docs: list[dict], cfg: SparseConfig, use_qdrant: bool = False):
                if use_qdrant:
                    return QdrantSparseSearcher(docs=docs, cfg=cfg)
                else:
                    return SparseSearcher(docs=docs, cfg=cfg)

            sparse_searcher = create_sparse_searcher(docs=docs, cfg=sparse_cfg, use_qdrant=use_qdrant_sparse)

            agg = run_benchmark_for_searcher(sparse_searcher, queries, ground_truth, top_k=cfg.top_k)

            # Cleanup if using Qdrant
            if use_qdrant_sparse and hasattr(sparse_searcher, "cleanup"):
                sparse_searcher.cleanup()

            entry = {"method": "bm25", "k1": k1, "b": b, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **agg}
            results.append(entry)

        except Exception as e:
            logger.error("BM25 grid failed for k1=%.2f, b=%.2f: %s", k1, b, e)
            entry = {
                "method": "bm25",
                "k1": k1,
                "b": b,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e),
                "Precision@1": 0.0,
                "Precision@3": 0.0,
                "Precision@5": 0.0,
                "Precision@10": 0.0,
                "Recall@10": 0.0,
                "MRR@10": 0.0,
                "NDCG@10": 0.0,
                "total_search_time": 0.0,
                "QPS": 0.0,
                "n_queries": 0,
                "n_success": 0,
                "n_failed": len(queries),
            }
            results.append(entry)

    return results


# -------------------------
# Hybrid Grid Search
# -------------------------


def run_hybrid_grid(
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
    ground_truth: dict[str, list[str]],
    cfg: BenchConfig,
    hnsw_collection: str | None = None,
) -> list[dict[str, Any]]:
    """
    Run Hybrid (Dense + Sparse + RRF) parameter grid search.

    Parameters:
        rrf_k: RRF constant for rank fusion
    """
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        logger.error("qdrant_client not installed. Skipping Hybrid grid.")
        return []

    if not check_qdrant_availability(cfg.qdrant_url):
        logger.error("Qdrant server not available. Skipping Hybrid grid.")
        return []

    # Prepare docs for sparse
    docs = []
    for i, c in enumerate(chunks):
        cid = c.get("metadata", {}).get("chunk_id", str(i))
        docs.append({"id": cid, "text": c.get("text", "")})

    # Prepare queries
    all_ids = [d["id"] for d in docs]
    id_to_idx = {cid: idx for idx, cid in enumerate(all_ids)}

    rng = random.Random(cfg.seed)
    n_total = len(all_ids)
    sample_n = cfg.sample_size if 0 < cfg.sample_size < n_total else n_total
    sampled_ids = rng.sample(all_ids, sample_n) if sample_n < n_total else all_ids

    queries = []
    for qid in sampled_ids:
        idx = id_to_idx.get(qid)
        if idx is None or idx >= len(chunks):
            continue
        queries.append(
            (
                qid,
                {
                    "text": chunks[idx].get("text", ""),
                    "vector": embeddings[idx].tolist() if idx < len(embeddings) else [0.0] * cfg.embedding_dim,
                },
            )
        )

    if not queries:
        logger.error("No queries prepared for Hybrid grid.")
        return []

    # Initialize components
    client = QdrantClient(url=cfg.qdrant_url)
    sparse_searcher = SparseSearcher(docs=docs, cfg=SparseConfig(top_k=cfg.top_k))

    # Use provided collection or create default
    if hnsw_collection is None:
        hnsw_collection = f"{cfg.collection_prefix}_hnsw_default"

        # Create default HNSW collection if not exists
        try:
            from qdrant_client import models

            if not client.collection_exists(hnsw_collection):
                client.create_collection(
                    collection_name=hnsw_collection,
                    vectors_config=models.VectorParams(size=cfg.embedding_dim, distance=models.Distance.COSINE),
                    hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
                )

                # Upload vectors
                batch_size = 500
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i : i + batch_size]
                    batch_embeddings = embeddings[i : i + batch_size]
                    batch_ids = all_ids[i : i + batch_size]

                    points = []
                    for j, (chunk, emb, cid) in enumerate(zip(batch_chunks, batch_embeddings, batch_ids, strict=False)):
                        if i + j >= len(embeddings):
                            break
                        points.append(
                            models.PointStruct(
                                id=cid if isinstance(cid, (int, str)) else str(cid),
                                vector=emb.tolist() if hasattr(emb, "tolist") else list(emb),
                                payload={"text": chunk.get("text", "")},
                            )
                        )

                    if points:
                        client.upsert(collection_name=hnsw_collection, points=points)

                time.sleep(1)
        except Exception as e:
            logger.error("Failed to create default HNSW collection: %s", e)
            return []

    results = []

    for rrf_k in cfg.hybrid_rrf_constants:
        logger.info("Testing Hybrid: rrf_k=%d", rrf_k)

        try:
            # Create dense searcher
            dense_cfg = DenseConfig(collection_name=hnsw_collection, ef_search=64, top_k=cfg.top_k)
            dense_searcher = DenseSearcher(client=client, cfg=dense_cfg)

            # Create hybrid searcher
            hybrid_cfg = HybridConfig(rrf_k=rrf_k, top_k=cfg.top_k)
            hybrid_searcher = HybridSearcher(
                dense_searcher=dense_searcher, sparse_searcher=sparse_searcher, cfg=hybrid_cfg
            )

            # Run benchmark
            agg = run_benchmark_for_searcher(hybrid_searcher, queries, ground_truth, top_k=cfg.top_k)

            entry = {
                "method": "hybrid_rrf",
                "rrf_k": rrf_k,
                "hnsw_collection": hnsw_collection,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                **agg,
            }
            results.append(entry)

        except Exception as e:
            logger.error("Hybrid grid failed for rrf_k=%d: %s", rrf_k, e)
            entry = {
                "method": "hybrid_rrf",
                "rrf_k": rrf_k,
                "hnsw_collection": hnsw_collection,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e),
                "Precision@1": 0.0,
                "Precision@3": 0.0,
                "Precision@5": 0.0,
                "Precision@10": 0.0,
                "Recall@10": 0.0,
                "MRR@10": 0.0,
                "NDCG@10": 0.0,
                "total_search_time": 0.0,
                "QPS": 0.0,
                "n_queries": 0,
                "n_success": 0,
                "n_failed": len(queries),
            }
            results.append(entry)

    return results


# -------------------------
# Save Results
# -------------------------


def save_results(results: list[dict[str, Any]], output_path: str) -> None:
    """Save results to JSONL file."""
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with outp.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("Saved %d results to %s", len(results), output_path)


# -------------------------
# Main Experiment Runner
# -------------------------


def run_experiment(
    cfg: BenchConfig | None = None,
    run_hnsw: bool = True,
    run_bm25: bool = True,
    run_hybrid: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """
    Run complete benchmark experiment.

    Args:
        cfg: Benchmark configuration
        run_hnsw: Whether to run HNSW grid search
        run_bm25: Whether to run BM25 grid search
        run_hybrid: Whether to run Hybrid grid search

    Returns:
        Dictionary with results per method
    """
    if cfg is None:
        cfg = BenchConfig()

    # Set seeds for reproducibility
    random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    rng.normal()
    logger.info("=" * 60)
    logger.info("BENCHMARK EXPERIMENT STARTED")
    logger.info("=" * 60)
    logger.info("Configuration: %s", cfg)

    # Load data
    logger.info("Loading ground truth and chunks...")
    gt = load_ground_truth(cfg.ground_truth_path)
    chunks = load_chunks(cfg.chunks_path)
    embeddings = load_embeddings_memmap(cfg.embeddings_memmap, cfg.embedding_dim)

    logger.info(
        "Loaded %d ground truth entries, %d chunks, embeddings shape: %s", len(gt), len(chunks), embeddings.shape
    )

    all_results: dict[str, list[dict[str, Any]]] = {"hnsw": [], "bm25": [], "hybrid": []}

    # Check Qdrant availability
    qdrant_available = check_qdrant_availability(cfg.qdrant_url)
    logger.info("Qdrant available: %s", qdrant_available)

    # Run HNSW grid
    if run_hnsw:
        if qdrant_available:
            logger.info("Running HNSW grid search...")
            all_results["hnsw"] = run_hnsw_grid(chunks, embeddings, gt, cfg)
            logger.info("HNSW: %d experiments completed", len(all_results["hnsw"]))
        else:
            logger.warning("Skipping HNSW: Qdrant not available")

    # Run BM25 grid
    if run_bm25:
        logger.info("Running BM25 grid search...")
        all_results["bm25"] = run_bm25_grid(chunks, gt, cfg)
        logger.info("BM25: %d experiments completed", len(all_results["bm25"]))

    # Run Hybrid grid
    if run_hybrid:
        if qdrant_available:
            logger.info("Running Hybrid grid search...")
            all_results["hybrid"] = run_hybrid_grid(chunks, embeddings, gt, cfg)
            logger.info("Hybrid: %d experiments completed", len(all_results["hybrid"]))
        else:
            logger.warning("Skipping Hybrid: Qdrant not available")

    # Combine and save all results
    combined_results = all_results["hnsw"] + all_results["bm25"] + all_results["hybrid"]

    if combined_results:
        save_results(combined_results, cfg.results_out)

    # Print summary
    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)

    for method, results in all_results.items():
        if results:
            successful = sum(1 for r in results if "error" not in r)
            logger.info("✓ %s: %d/%d successful", method, successful, len(results))
        else:
            logger.info("○ %s: skipped", method)

    return all_results


# -------------------------
# Entry Point
# -------------------------


if __name__ == "__main__":
    cfg = BenchConfig()

    results = run_experiment(cfg=cfg, run_hnsw=True, run_bm25=True, run_hybrid=True)

    logger.info(
        f"Benchmark completed. Results saved to: {cfg.results_out}",
    )
