"""
Evaluate ANN vector index quality against brute-force ground truth.

This script:
1. ground-truth nearest neighbors
2. embeddings (memmap)

3. ANN index (HNSW or IVF)
4. Searches ANN index

5. Compares ANN results with ground truth
6. metrics:
      - Recall@k
      - Precision@k
      - MRR@k
      - HitRate@k
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from tqdm.auto import tqdm


@dataclass
class EvalConfig:

    ground_truth_path: str = "data/ground_truth/ground_truth.jsonl"
    embeddings_path: str = "data/ground_truth/embeddings.memmap"

    embedding_dim: int = 384  # MiniLM dimension

    top_k: int = 10
    batch_size: int = 512

    # ANN parameters
    index_type: str = "hnsw"  # hnsw | ivf
    hnsw_m: int = 32
    hnsw_ef_search: int = 64

    ivf_nlist: int = 256
    ivf_nprobe: int = 10


def load_ground_truth(path: str) -> Dict[str, List[str]]:
    """
    Load ground truth mapping:
        chunk_id -> [neighbor_ids]
    """

    gt: Dict[str, List[str]] = {}

    with open(path, "r", encoding="utf-8") as f:

        for line in f:
            obj = json.loads(line)

            for k, v in obj.items():
                gt[k] = v

    return gt


def load_embeddings(path: str, dim: int) -> np.memmap:
    """
    Load embeddings memmap.
    """

    file_size = Path(path).stat().st_size
    n = file_size // (4 * dim)  # float32 = 4 bytes

    return np.memmap(path, dtype="float32", mode="r", shape=(n, dim))


def build_hnsw_index(embeddings: np.ndarray, m: int = 32) -> faiss.Index:

    dim = embeddings.shape[1]

    index = faiss.IndexHNSWFlat(dim, m)
    index.hnsw.efSearch = 64

    index.add(embeddings)

    return index


def build_ivf_index(
    embeddings: np.ndarray,
    nlist: int = 256,
    nprobe: int = 10
) -> faiss.Index:

    dim = embeddings.shape[1]

    quantizer = faiss.IndexFlatIP(dim)

    index = faiss.IndexIVFFlat(
        quantizer,
        dim,
        nlist,
        faiss.METRIC_INNER_PRODUCT
    )

    index.train(embeddings)
    index.add(embeddings)

    index.nprobe = nprobe

    return index


def compute_metrics(
    ann_results: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    k: int,
):

    recalls = []
    precisions = []
    mrrs = []
    hits = []

    for qid, ann_ids in ann_results.items():

        gt_ids = ground_truth.get(qid, [])

        if not gt_ids:
            continue

        gt_set = set(gt_ids)

        retrieved = ann_ids[:k]

        intersection = gt_set.intersection(retrieved)

        recall = len(intersection) / len(gt_ids)
        precision = len(intersection) / k

        recalls.append(recall)
        precisions.append(precision)

        hits.append(1 if intersection else 0)

        # MRR
        rank = None
        for i, doc in enumerate(retrieved, start=1):
            if doc in gt_set:
                rank = i
                break

        if rank:
            mrrs.append(1 / rank)
        else:
            mrrs.append(0)

    metrics = {

        "recall@k": float(np.mean(recalls)),
        "precision@k": float(np.mean(precisions)),
        "mrr@k": float(np.mean(mrrs)),
        "hit_rate@k": float(np.mean(hits)),
    }

    return metrics


def evaluate(cfg: EvalConfig):

    start = time.time()

    print("Loading ground truth...")
    gt = load_ground_truth(cfg.ground_truth_path)

    print("Loading embeddings...")
    embeddings = load_embeddings(cfg.embeddings_path, cfg.embedding_dim)

    n = embeddings.shape[0]

    print("Building ANN index:", cfg.index_type)

    if cfg.index_type == "hnsw":

        index = build_hnsw_index(embeddings, cfg.hnsw_m)

    elif cfg.index_type == "ivf":

        index = build_ivf_index(
            embeddings,
            cfg.ivf_nlist,
            cfg.ivf_nprobe
        )

    else:
        raise ValueError("Unknown index type")

    print("Running ANN search...")

    ann_results: Dict[str, List[str]] = {}

    chunk_ids = list(gt.keys())

    id_lookup = {i: cid for i, cid in enumerate(chunk_ids)}

    for i in tqdm(range(0, n, cfg.batch_size)):

        j = min(n, i + cfg.batch_size)

        queries = embeddings[i:j]

        distances, indices = index.search(
            queries,
            cfg.top_k + 1
        )

        for row_idx, idxs in enumerate(indices):

            query_idx = i + row_idx

            query_id = id_lookup.get(query_idx)

            neighbors = []

            for idx in idxs:

                if idx == query_idx:
                    continue

                if idx < 0:
                    continue

                neighbors.append(id_lookup.get(idx))

                if len(neighbors) >= cfg.top_k:
                    break

            ann_results[query_id] = neighbors

    print("Computing metrics...")

    metrics = compute_metrics(
        ann_results,
        gt,
        cfg.top_k
    )

    elapsed = time.time() - start

    print("\nEvaluation Results")
    print("-------------------")

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print(f"\nTotal time: {elapsed:.2f} sec")


if __name__ == "__main__":

    config = EvalConfig(
        index_type="hnsw",  # change to "ivf"
        top_k=10
    )

    evaluate(config)