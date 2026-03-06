"""
Ground truth nearest-neighbors generator for document chunks.

 - Loads processed chunks from /chunks.jsonl
 - Encodes each chunk with a SentenceTransformer model
 - Stores embeddings in a numpy.memmap
 - Normalizes embeddings (unit length) so IndexFlatIP computes cosine similarity
 - Builds a FAISS IndexFlatIP and performs exact (brute-force) search
 - Writes top-N neighbors (excluding the query itself) to data/ground_truth/ground_truth.jsonl
 - Writes timing metadata to data/ground_truth/ground_truth_meta.json
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# -------------------------
# Configuration dataclass
# -------------------------


@dataclass
class GTConfig:
    chunks_path: str = "data/processed/chunks.jsonl"
    output_path: str = "data/ground_truth/ground_truth.jsonl"
    meta_path: str = "data/ground_truth/ground_truth_meta.json"
    emb_memmap_path: str = "data/ground_truth/embeddings.memmap"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    top_k: int = 10
    dtype: np.dtype = np.float32
    use_gpu_faiss: bool = False  # optional: if faiss with GPU is set up externally


# -------------------------
# Utilities
# -------------------------


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_chunks(chunks_jsonl_path: str) -> Tuple[List[str], List[Dict]]:
    """
    Load chunks from JSONL file.

    Returns:
        texts: list of chunk texts (in order)
        metadata_list: list of metadata dicts (in same order)
    """
    path = Path(chunks_jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_jsonl_path}")

    texts: List[str] = []
    metadata_list: List[Dict] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                texts.append(obj.get("text", ""))
                metadata_list.append(obj.get("metadata", {}))
            except json.JSONDecodeError as exc:
                logging.warning("Skipping invalid JSON line: %s", exc)

    return texts, metadata_list


def count_lines(path: str) -> int:
    """Count non-empty lines in a file (fast)."""
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f if _.strip())



def create_embeddings_memmap(
    texts: List[str],
    model_name: str,
    memmap_path: str,
    batch_size: int = 64,
    dtype: np.dtype = np.float32,
) -> Tuple[np.memmap, int]:
    """
    Encode texts using SentenceTransformer and write to numpy.memmap.

    Returns:
        memmap: numpy.memmap (n, dim)
        dim: embedding dimension
    """
    n = len(texts)
    if n == 0:
        raise ValueError("No texts to embed.")

    logging.info("Loading model: %s", model_name)
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        logging.exception("Failed to load sentence transformer model.")
        raise

    # dimension
    try:
        dim = model.get_sentence_embedding_dimension()
    except Exception:
        # fallback: encode one example
        dim = model.encode([texts[0]], convert_to_numpy=True).shape[1]

    memmap_dir = Path(memmap_path).parent
    memmap_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Creating memmap (%d x %d) @ %s", n, dim, memmap_path)
    memmap = np.memmap(memmap_path, mode="w+", dtype=dtype, shape=(n, dim))

    # encode in batches and write to memmap
    for i in tqdm(range(0, n, batch_size), desc="Embedding batches", unit="batch"):
        j = min(n, i + batch_size)
        batch_texts = texts[i:j]
        try:
            embeddings = model.encode(
                batch_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,  # we'll normalize later explicitly
            )
        except Exception as e:
            logging.exception("Encoding failed on batch %d:%d", i, j)
            raise

        if embeddings.dtype != dtype:
            embeddings = embeddings.astype(dtype)

        memmap[i:j] = embeddings
        # ensure memmap flushed to disk for large datasets
        memmap.flush()

    logging.info("Finished embeddings memmap creation.")
    return memmap, dim


def normalize_inplace_memmap(memmap: np.memmap) -> None:
    """
    Normalize the memmap rows to unit length (L2 norm).
    This allows IndexFlatIP to behave as cosine similarity.
    """
    logging.info("Normalizing embeddings (inplace).")
    # Compute norms in batches to avoid temporarily allocating huge arrays
    n, dim = memmap.shape
    batch = 8192  # tune if needed
    for i in range(0, n, batch):
        j = min(n, i + batch)
        block = memmap[i:j]
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        block[:] = block / norms
        memmap.flush()
    logging.info("Normalization complete.")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build FAISS IndexFlatIP and add embeddings.

    embeddings: numpy array (n, dim) float32, already normalized.
    """
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    n, dim = embeddings.shape
    logging.info("Building FAISS IndexFlatIP (dim=%d, n=%d)", dim, n)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)  # exact index
    logging.info("FAISS index built and %d vectors added.", index.ntotal)
    return index


def generate_ground_truth(
    memmap: np.memmap,
    metadata_list: List[Dict],
    index: faiss.Index,
    output_path: str,
    top_k: int = 10,
    batch_size: int = 512,
) -> Dict[str, List[str]]:
    """
    For every vector (treated as a query) find top_k most similar OTHER vectors.

    Writes results to output_path (jsonl) as one JSON per line:
        {"<chunk_id>": ["id1", "id2", ...]}

    Returns:
        results_map: dict mapping chunk_id -> list of neighbor chunk_ids
    """
    n = memmap.shape[0]
    if n != len(metadata_list):
        raise ValueError("Embeddings count and metadata count mismatch.")

    # prepare id mapping: index position -> chunk_id
    chunk_ids = [md.get("chunk_id") or str(i) for i, md in enumerate(metadata_list)]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results_map: Dict[str, List[str]] = {}

    k_search = min(n, top_k + 1)  # search +1 to allow excluding self
    logging.info("Searching nearest neighbors: top_k=%d (search k=%d)", top_k, k_search)

    with output_file.open("w", encoding="utf-8") as outf:
        # search in batches to save memory for very large n
        for i in tqdm(range(0, n, batch_size), desc="Searching batches", unit="batch"):
            j = min(n, i + batch_size)
            queries = np.asarray(memmap[i:j], dtype=np.float32)

            # FAISS returns (distances, indices)
            distances, indices = index.search(queries, k_search)

            for row_idx_in_batch, (d_row, ind_row) in enumerate(zip(distances, indices)):
                global_query_idx = i + row_idx_in_batch
                query_chunk_id = chunk_ids[global_query_idx]

                # filter out the query itself
                neighbors: List[str] = []
                for idx in ind_row:
                    if int(idx) == int(global_query_idx):
                        continue
                    # if idx == -1 (faiss can return -1 when not enough vectors), skip
                    if int(idx) < 0:
                        continue
                    neighbors.append(chunk_ids[int(idx)])
                    if len(neighbors) >= top_k:
                        break

                # write single line mapping
                json_line = json.dumps({query_chunk_id: neighbors}, ensure_ascii=False)
                outf.write(json_line + "\n")
                results_map[query_chunk_id] = neighbors

    logging.info("Ground truth written to %s", output_file)
    return results_map



def build_ground_truth_pipeline(cfg: GTConfig) -> Dict[str, List[str]]:
    """
    Full pipeline orchestration.

    Returns:
        results_map
    """
    t0 = time.perf_counter()
    texts, metadata_list = load_chunks(cfg.chunks_path)
    n = len(texts)
    logging.info("Loaded %d chunks from %s", n, cfg.chunks_path)

    # Create embeddings memmap (or reuse if exists)
    memmap_path = cfg.emb_memmap_path
    dims = None

    # If memmap exists with correct shape, reuse it.
    if Path(memmap_path).exists():
        logging.info("Found existing memmap at %s, attempting to reuse.", memmap_path)
        # We need dimension to map shape. We'll try to infer by size.
        mem = np.memmap(memmap_path, mode="r+", dtype=cfg.dtype)
        # try to reshape: dims = mem.shape[1] if mem.ndim == 2 else unknown
        if mem.ndim == 2 and mem.shape[0] == n:
            memmap = mem.reshape(mem.shape)
            dims = memmap.shape[1]
            logging.info("Reused memmap with shape %s", memmap.shape)
        else:
            logging.warning("Existing memmap shape mismatch; recreating.")
            memmap, dims = create_embeddings_memmap(
                texts, cfg.model_name, memmap_path, cfg.batch_size, cfg.dtype
            )
    else:
        memmap, dims = create_embeddings_memmap(
            texts, cfg.model_name, memmap_path, cfg.batch_size, cfg.dtype
        )

    # Normalize embeddings to unit length for cosine via inner product
    normalize_inplace_memmap(memmap)

    # Build FAISS index
    index = build_faiss_index(np.asarray(memmap, dtype=np.float32))

    # Search and write results
    results = generate_ground_truth(
        memmap=memmap,
        metadata_list=metadata_list,
        index=index,
        output_path=cfg.output_path,
        top_k=cfg.top_k,
        batch_size=max(cfg.batch_size, 256),
    )

    duration = time.perf_counter() - t0
    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": duration,
        "n_chunks": n,
        "model_name": cfg.model_name,
        "top_k": cfg.top_k,
    }

    Path(cfg.meta_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, ensure_ascii=False, indent=2)

    logging.info("Ground-truth generation finished in %.2fs", duration)
    logging.info("Meta written to %s", cfg.meta_path)

    return results


if __name__ == "__main__":
    setup_logging()

    default_cfg = GTConfig(
        chunks_path="data/processed/chunks.jsonl",
        output_path="data/ground_truth/ground_truth.jsonl",
        meta_path="data/ground_truth/ground_truth_meta.json",
        emb_memmap_path="data/ground_truth/embeddings.memmap",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=64,
        top_k=10,
    )

    try:
        results = build_ground_truth_pipeline(default_cfg)
        logging.info("Generated ground truth for %d queries.", len(results))
    except Exception as exc:
        logging.exception("Ground-truth generation failed: %s", exc)
        raise