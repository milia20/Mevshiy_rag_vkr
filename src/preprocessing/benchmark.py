"""
Benchmarking framework to evaluate search strategies against ground truth.

Saves JSONL results with metrics per parameter setting.

Metrics:
 - Precision@1, @3, @5, @10
 - Recall@10
 - MRR@10
 - total_search_time
 - QPS

 - HNSW parameters (m, ef_construct, ef_search)
 - BM25 parameters (k1, b) by rebuilding BM25 if needed
 - Hybrid RRF constant grid
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from tqdm.auto import tqdm

# import search classes
from ..search_strategies import DenseSearcher, SparseSearcher, HybridSearcher, DenseConfig, SparseConfig, HybridConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# -------------------------
# Configs
# -------------------------


@dataclass
class BenchConfig:
    ground_truth_path: str = "data/ground_truth/ground_truth.jsonl"
    chunks_path: str = "data/processed/chunks.jsonl"
    embeddings_memmap: str = "data/ground_truth/embeddings.memmap"
    embedding_dim: int = 384
    qdrant_url: str = "http://localhost:6333"
    sample_size: int = 500  # N random queries (or use 0 for all)
    top_k: int = 10
    results_out: str = "data/benchmarks/benchmark_results.jsonl"
    seed: int = 42


# -------------------------
# Helpers: loading ground truth/chunks
# -------------------------


def load_ground_truth(path: str) -> Dict[str, List[str]]:
    gt: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            for k, v in obj.items():
                gt[k] = v
    return gt


def load_chunks(path: str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def load_embeddings_memmap(path: str, dim: int):
    import numpy as np
    size = Path(path).stat().st_size
    n = size // (4 * dim)
    return np.memmap(path, dtype="float32", mode="r", shape=(n, dim))


# -------------------------
# Metrics
# -------------------------


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    topk = retrieved[:k]
    if k == 0:
        return 0.0
    return len(set(topk).intersection(set(relevant))) / float(k)


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    topk = retrieved[:k]
    if len(relevant) == 0:
        return 0.0
    return len(set(topk).intersection(set(relevant))) / float(len(relevant))


def mrr_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    topk = retrieved[:k]
    for i, did in enumerate(topk, start=1):
        if did in relevant:
            return 1.0 / i
    return 0.0


# -------------------------
# Runner
# -------------------------


def run_benchmark_for_searcher(
    searcher,
    queries: List[Tuple[str, Dict[str, Any]]],  # list of (query_id, {"text":..., "vector":...})
    ground_truth: Dict[str, List[str]],
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Run benchmark using a searcher object exposing .search(query_text, query_vector?, top_k?)

    Returns aggregated metrics and timing.
    """
    metrics_acc = {
        "p1": [], "p3": [], "p5": [], "p10": [], "recall10": [], "mrr10": []
    }

    total_search_time = 0.0
    n_queries = len(queries)

    for qid, qrec in tqdm(queries, desc="Benchmark queries"):
        text = qrec.get("text")
        vec = qrec.get("vector")
        t0 = time.perf_counter()
        # choose search call dynamically (some searchers need vector+text)
        try:
            # detect whether searcher supports hybrid signature
            if hasattr(searcher, "search") and searcher.search.__code__.co_argcount >= 3:
                # HybridSearcher.search(query_text, query_vector,...)
                res, meta = searcher.search(text, query_vector=vec, top_k=top_k)
            else:
                # Sparse or Dense: try both call patterns
                try:
                    res, meta = searcher.search(text, top_k=top_k)  # sparse uses text
                except TypeError:
                    res, meta = searcher.search(vec, top_k=top_k)  # dense uses vector
        except Exception as exc:
            logger.exception("Search failed for query %s: %s", qid, exc)
            continue

        total_search_time += meta.get("time_s", 0.0)
        retrieved_ids = [r["id"] for r in res]

        gt_ids = ground_truth.get(qid, [])

        metrics_acc["p1"].append(precision_at_k(retrieved_ids, gt_ids, 1))
        metrics_acc["p3"].append(precision_at_k(retrieved_ids, gt_ids, 3))
        metrics_acc["p5"].append(precision_at_k(retrieved_ids, gt_ids, 5))
        metrics_acc["p10"].append(precision_at_k(retrieved_ids, gt_ids, 10))
        metrics_acc["recall10"].append(recall_at_k(retrieved_ids, gt_ids, 10))
        metrics_acc["mrr10"].append(mrr_at_k(retrieved_ids, gt_ids, 10))

    # aggregate
    def mean(arr):
        return float(np.mean(arr)) if arr else 0.0

    aggregated = {
        "Precision@1": mean(metrics_acc["p1"]),
        "Precision@3": mean(metrics_acc["p3"]),
        "Precision@5": mean(metrics_acc["p5"]),
        "Precision@10": mean(metrics_acc["p10"]),
        "Recall@10": mean(metrics_acc["recall10"]),
        "MRR@10": mean(metrics_acc["mrr10"]),
        "total_search_time": total_search_time,
        "QPS": (len(queries) / total_search_time) if total_search_time > 0 else float("inf"),
        "n_queries": len(queries),
    }

    return aggregated


# -------------------------
# Grid runner
# -------------------------


def run_hnsw_grid(
    client: Any,
    chunks: List[Dict[str, Any]],
    embeddings: Any,
    ground_truth: Dict[str, List[str]],
    cfg: BenchConfig,
    m_values=(8, 16, 32),
    ef_construct_values=(100, 200),
    ef_search_values=(50, 100),
):
    """
    For each HNSW parameter combination:
      - create collection (or assume exists)
      - instantiate DenseSearcher (with ef_search)
      - run benchmark over queries
      - output JSONL results
    """
    # prepare queries list: sample or all
    all_ids = [c["metadata"]["chunk_id"] for c in chunks]
    id_to_idx = {cid: idx for idx, cid in enumerate(all_ids)}

    # build queries (query_id -> {"text","vector"})
    queries = []
    n_total = len(all_ids)
    rng = random.Random(cfg.seed)
    sample_n = cfg.sample_size if cfg.sample_size > 0 and cfg.sample_size < n_total else n_total
    sampled_ids = rng.sample(all_ids, sample_n) if sample_n < n_total else all_ids

    for qid in sampled_ids:
        idx = id_to_idx[qid]
        queries.append((qid, {"text": chunks[idx]["text"], "vector": embeddings[idx].tolist()}))

    # iterate grid
    from qdrant_indexer import QdrantIndexer  # reuse helper naming conventions
    indexer = QdrantIndexer(url=cfg.qdrant_url, in_memory=False)  # assumes qdrant reachable

    results = []
    for m, ef_c, ef_s in product(m_values, ef_construct_values, ef_search_values):
        coll_name = f"hnsw_m{m}_ec{ef_c}_es{ef_s}"
        full_coll = f"{indexer.collection_prefix}_{coll_name}"

        # create collection (try to create; if exists continue)
        from qdrant_client import models
        try:
            indexer.create_collection(collection_name=coll_name, vector_size=cfg.embedding_dim, hnsw_config=models.HnswConfig(m=m, ef_construct=ef_c))
            # here: recommeded to index vectors into this collection if not present first
        except Exception:
            logger.warning("Collection create may have failed or exists: %s", coll_name)

        # instantiate dense searcher
        client = indexer.client
        dense_cfg = DenseConfig(collection_name=full_coll, ef_search=ef_s, top_k=cfg.top_k)
        dense_searcher = DenseSearcher(client=client, cfg=dense_cfg)

        # run benchmark
        agg = run_benchmark_for_searcher(dense_searcher, queries, ground_truth, top_k=cfg.top_k)

        # save metrics with grid params
        entry = {
            "method": "hnsw",
            "m": m,
            "ef_construct": ef_c,
            "ef_search": ef_s,
            **agg,
        }
        results.append(entry)

    # persist
    outp = Path(cfg.results_out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return results


def run_bm25_grid(
    chunks: List[Dict[str, Any]],
    ground_truth: Dict[str, List[str]],
    cfg: BenchConfig,
    k1_values=(1.2,),
    b_values=(0.75,),
):
    """
    Sweep BM25 parameters; rank_bm25 doesn't natively expose k1/b to set easily unless using custom BM25 implementation.
    This function shows where you'd vary k1/b if using a BM25 implementation that exposes them (or swap in a library).
    For rank_bm25 (simple) you may need to patch the object or use another BM25 library which supports k1/b at init.
    """
    # Prepare docs list for SparseSearcher: id + text
    docs = [{"id": c["metadata"]["chunk_id"], "text": c["text"]} for c in chunks]

    results = []
    for k1, b in product(k1_values, b_values):
        # rank_bm25's BM25Okapi uses default k1/b; to vary you'd need to modify internals.
        # For demonstration we rebuild searcher normally and annotate params (you may replace with a library that allows k1,b)
        sparse_searcher = SparseSearcher(docs=docs, cfg=SparseConfig(top_k=cfg.top_k))

        # build queries (sample)
        all_ids = [d["id"] for d in docs]
        rng = random.Random(cfg.seed)
        sample_n = cfg.sample_size if cfg.sample_size > 0 and cfg.sample_size < len(all_ids) else len(all_ids)
        sampled_ids = rng.sample(all_ids, sample_n) if sample_n < len(all_ids) else all_ids

        queries = [(sid, {"text": docs[[dd["id"] for dd in docs].index(sid)]["text"]}) for sid in sampled_ids]

        agg = run_benchmark_for_searcher(sparse_searcher, queries, ground_truth, top_k=cfg.top_k)

        entry = {"method": "bm25", "k1": k1, "b": b, **agg}
        results.append(entry)

    outp = Path(cfg.results_out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return results


def run_hybrid_grid(
    client: Any,
    chunks: List[Dict[str, Any]],
    embeddings: Any,
    ground_truth: Dict[str, List[str]],
    cfg: BenchConfig,
    rrf_constants=(30, 60, 120),
):
    """
    Run hybrid experiments sweeping RRF constants.
    """
    # build base components
    docs = [{"id": c["metadata"]["chunk_id"], "text": c["text"]} for c in chunks]
    sparse_searcher = SparseSearcher(docs=docs, cfg=SparseConfig(top_k=cfg.top_k))
    indexer = None
    try:
        from qdrant_indexer import QdrantIndexer
        indexer = QdrantIndexer(url=cfg.qdrant_url, in_memory=False)
    except Exception:
        logger.warning("QdrantIndexing helper not available; hybrid dense side may fail.")

    all_ids = [c["metadata"]["chunk_id"] for c in chunks]
    id_to_idx = {cid: idx for idx, cid in enumerate(all_ids)}
    rng = random.Random(cfg.seed)
    sample_n = cfg.sample_size if cfg.sample_size > 0 and cfg.sample_size < len(all_ids) else len(all_ids)
    sampled_ids = rng.sample(all_ids, sample_n) if sample_n < len(all_ids) else all_ids

    queries = []
    for qid in sampled_ids:
        idx = id_to_idx[qid]
        queries.append((qid, {"text": chunks[idx]["text"], "vector": embeddings[idx].tolist()}))

    results = []
    for rrf_k in rrf_constants:
        # dense searcher using existing/assumed collection (you will pick the collection tested)
        # Use previously created collection name (adjust to your setup)
        dense_cfg = DenseConfig(collection_name=f"{QdrantIndexer(collection_prefix='mkdocs')._full_collection_name('hnsw_default')}", ef_search=64, top_k=cfg.top_k)
        dense_searcher = DenseSearcher(client=indexer.client, cfg=dense_cfg)
        hybrid_searcher = HybridSearcher(dense_searcher=dense_searcher, sparse_searcher=sparse_searcher, cfg=HybridConfig(rrf_k=rrf_k, top_k=cfg.top_k))

        agg = run_benchmark_for_searcher(hybrid_searcher, queries, ground_truth, top_k=cfg.top_k)
        entry = {"method": "hybrid_rrf", "rrf_k": rrf_k, **agg}
        results.append(entry)

    outp = Path(cfg.results_out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return results


# -------------------------
# Main
# -------------------------


if __name__ == "__main__":
    cfg = BenchConfig()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    logger.info("Loading ground truth and chunks...")
    gt = load_ground_truth(cfg.ground_truth_path)
    chunks = load_chunks(cfg.chunks_path)
    embeddings = load_embeddings_memmap(cfg.embeddings_memmap, cfg.embedding_dim)

    # Example: run HNSW grid
    _ = run_hnsw_grid(
        client=None,
        chunks=chunks,
        embeddings=embeddings,
        ground_truth=gt,
        cfg=cfg,
        m_values=(8, 16, 32),
        ef_construct_values=(100, 200),
        ef_search_values=(50, 100),
    )

    # Example: run BM25 grid (simple)
    _ = run_bm25_grid(chunks=chunks, ground_truth=gt, cfg=cfg, k1_values=(1.2,), b_values=(0.75,))

    # Example: run hybrid grid
    _ = run_hybrid_grid(client=None, chunks=chunks, embeddings=embeddings, ground_truth=gt, cfg=cfg, rrf_constants=(30, 60))