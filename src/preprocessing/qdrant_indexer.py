"""
Qdrant indexer for MkDocs chunks.

This module provides QdrantIndexer - a small wrapper around qdrant_client
to create collections (different HNSW configurations), upload points in batches,
and create payload indexes for filter experiments.

Usage :
    from src.indexing.qdrant_indexer import QdrantIndexer
    indexer = QdrantIndexer(in_memory=True)
    indexer.create_collections_for_experiments(vector_size=384)
    indexer.index_documents(chunks=chunks_list, vectors=vectors_array, batch_size=200)
    indexer.create_payload_index(collection_name="mkdocs_payloads", field_name="doc_title", field_type="keyword")
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from tqdm.auto import tqdm

try:
    # qdrant_client, models are required runtime dependencies
    from qdrant_client import QdrantClient
    from qdrant_client import models
except Exception as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "qdrant-client is required. Install with `pip install qdrant-client`."
    ) from exc

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class QdrantIndexer:
    """
    Wrapper for Qdrant client to manage experiment collections and batch uploads.

    Parameters
    ----------
    host:
        Hostname or base URL (used when in_memory is False). Default 'localhost'.
    port:
        Port for local Qdrant service (ignored if `url` provided or in_memory True).
    collection_prefix:
        Prefix added to collection names created by helper methods.
    in_memory:
        If True instantiate QdrantClient in-memory (suitable for unit tests).
    url:
        Full url to Qdrant (e.g. "http://localhost:6333"). If provided, overrides host/port.
    api_key:
        Optional API key for Qdrant cloud or secured instance.
    prefer_grpc:
        If True, tries gRPC (client handles automatically if available).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_prefix: str = "mkdocs",
        in_memory: bool = False,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
    ) -> None:
        self.collection_prefix = collection_prefix
        self.in_memory = in_memory

        if in_memory:
            logger.info("Initializing in-memory Qdrant client.")
            # QdrantClient(":memory:") creates in-memory storage (local client)
            self.client = QdrantClient(":memory:")
        else:
            if url:
                client_url = url
            else:
                client_url = f"http://{host}:{port}"
            logger.info("Initializing Qdrant client at %s", client_url)
            # api_key may be None for local docker instances
            self.client = QdrantClient(url=client_url, api_key=api_key, prefer_grpc=prefer_grpc)

    # -------------------------
    # Collection creation
    # -------------------------
    def _full_collection_name(self, name: str) -> str:
        """Return namespaced collection name."""
        return f"{self.collection_prefix}_{name}"

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: models.Distance = models.Distance.COSINE,
        hnsw_config: Optional[models.HnswConfig] = None,
        shards: Optional[int] = None,
        wait: bool = True,
    ) -> None:
        """
        Create a collection with given HNSW configuration.

        Parameters
        ----------
        collection_name:
            Name (without prefix) of collection.
        vector_size:
            Dimensionality of stored vectors.
        distance:
            models.Distance.COSINE / DOT / EUCLID
        hnsw_config:
            Optional models.HnswConfig instance (m, ef_construct, full_scan_threshold, etc.)
            If None, Qdrant will use defaults.
        shards:
            Optional shard_number for collection creation.
        wait:
            Whether to wait for operation completion (where supported).
        """
        full_name = self._full_collection_name(collection_name)
        logger.info("Creating collection '%s' (size=%d, distance=%s)", full_name, vector_size, distance)

        # Build VectorParams using HTTP models when available
        vector_params = models.VectorParams(
            size=vector_size,
            distance=distance,
            hnsw_config=hnsw_config
        )

        try:
            # If collection exists, create_collection raises; handle gracefully
            if shards is not None:
                self.client.create_collection(
                    collection_name=full_name,
                    vectors_config=vector_params,
                    shard_number=shards,
                    wait=wait,
                )
            else:
                self.client.create_collection(
                    collection_name=full_name, vectors_config=vector_params, wait=wait
                )
            logger.info("Collection '%s' created.", full_name)
        except Exception as exc:
            # If collection already exists, log and continue
            msg = str(exc).lower()
            if "already exists" in msg or "already exists" in repr(exc).lower():
                logger.warning("Collection %s already exists — skipping create.", full_name)
            else:
                logger.exception("Failed to create collection %s", full_name)
                raise

    def create_collections_for_experiments(self, vector_size: int) -> None:
        """
        Create three collections for experiments:
          - <prefix>_hnsw_default  (m=16, ef_construct=100)
          - <prefix>_hnsw_optimized (m=32, ef_construct=256)
          - <prefix>_payloads (default HNSW + payload indexes support)
        """
        # Import HnswConfig model factory
        # Note: models.HnswConfig accepts m, ef_construct, full_scan_threshold, etc.
        logger.info("Creating experimental collections (vector_size=%d)", vector_size)

        # Collection 1: Default HNSW (m=16, ef_construct=100)
        hnsw1 = models.HnswConfig(m=16, ef_construct=100)
        self.create_collection("hnsw_default", vector_size=vector_size, hnsw_config=hnsw1)

        # Collection 2: Optimized HNSW (m=32, ef_construct=256)
        hnsw2 = models.HnswConfig(m=32, ef_construct=256)
        self.create_collection("hnsw_optimized", vector_size=vector_size, hnsw_config=hnsw2)

        # Collection 3: With payload indexes for filtering
        hnsw3 = models.HnswConfig(m=16, ef_construct=100)
        self.create_collection("payloads", vector_size=vector_size, hnsw_config=hnsw3)

        logger.info("All experiment collections created (prefixed by '%s').", self.collection_prefix)

    # -------------------------
    # Indexing / upload
    # -------------------------
    def _point_id_from_metadata(self, metadata: Dict[str, Any], default_id: int) -> Union[int, str]:
        """
        Determine Qdrant point id from metadata if available.

        Uses metadata['chunk_id'] if present, otherwise returns default_id (int).
        """
        cid = metadata.get("chunk_id") if metadata else None
        return cid if cid is not None else int(default_id)

    def index_documents(
        self,
        collection_name: str,
        chunks: List[Dict[str, Any]],
        vectors: Optional[np.ndarray] = None,
        batch_size: int = 200,
        wait: bool = True,
    ) -> None:
        """
        Batch upload documents (points) to a qdrant collection.

        Parameters
        ----------
        collection_name:
            Name (without prefix) of experiment collection to upload into.
        chunks:
            List of chunk dicts, each must contain at least 'metadata' key.
            Example chunk:
                {
                    "text": "...",
                    "metadata": {"source": "...", "chunk_id": "...", "doc_title": "..."}
                }
        vectors:
            Optional numpy array of shape (n, dim). If provided, vectors[i] is assigned
            to chunks[i]. If omitted, chunk must contain 'vector' key (list/array).
        batch_size:
            Number of points to send per batch (100-500 recommended).
        wait:
            Wait for operation completion.
        """
        full_name = self._full_collection_name(collection_name)
        logger.info("Indexing %d chunks -> collection %s (batch_size=%d)", len(chunks), full_name, batch_size)

        n = len(chunks)
        if vectors is not None:
            if len(vectors) != n:
                raise ValueError("Length of vectors does not match number of chunks.")
            # Ensure numpy array dtype is python float lists for qdrant
        else:
            # verify each chunk has 'vector'
            missing_vector = [i for i, c in enumerate(chunks) if "vector" not in c]
            if missing_vector:
                raise ValueError(f"Vectors not provided and chunks missing 'vector' keys at indices: {missing_vector[:5]}...")

        # build an iterable generator of PointStruct objects in batches and upload
        from qdrant_client.models import PointStruct

        total_uploaded = 0
        for start in tqdm(range(0, n, batch_size), desc=f"Uploading to {full_name}", unit="batch"):
            end = min(n, start + batch_size)
            batch_chunks = chunks[start:end]

            points: List[models.PointStruct] = []
            for i, chunk in enumerate(batch_chunks):
                global_idx = start + i
                metadata = chunk.get("metadata", {}) if chunk else {}
                point_id = self._point_id_from_metadata(metadata, default_id=global_idx)

                # prefer vectors parameter when provided else chunk['vector']
                if vectors is not None:
                    vec = vectors[global_idx]
                else:
                    vec = chunk.get("vector")

                if vec is None:
                    raise ValueError(f"Missing vector for chunk at index {global_idx}")

                # Ensure vector is list[float]
                try:
                    vector_list = np.asarray(vec, dtype=float).tolist()
                except Exception:
                    raise ValueError(f"Invalid vector for chunk index {global_idx}")

                # Build payload: include entire metadata; optionally drop large fields (like 'text') if desired.
                payload = metadata.copy()
                # Optionally remove raw text from payload to keep payload small:
                # payload.pop("text", None)

                points.append(PointStruct(id=point_id, vector=vector_list, payload=payload))

            try:
                # upload_points is optimized for medium batches; will handle retries internally
                self.client.upload_points(collection_name=full_name, points=points, wait=wait)
                total_uploaded += len(points)
                logger.debug("Uploaded batch %d-%d to %s", start, end, full_name)
            except Exception as exc:
                logger.exception("Failed to upload batch %d-%d to %s", start, end, full_name)
                raise

        # Verification: count points
        try:
            count = self.client.count(collection_name=full_name).count
            logger.info("Upload complete. Collection '%s' contains %d points (expected >= %d).", full_name, count, total_uploaded)
        except Exception:
            logger.warning("Upload complete but failed to verify count for %s", full_name)

    # -------------------------
    # Payload indexing
    # -------------------------
    def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        field_type: str = "keyword",
        wait: bool = True,
    ) -> None:
        """
        Create a payload index for a given payload field.

        Parameters
        ----------
        collection_name:
            Name (without prefix) of collection.
        field_name:
            Field in payload to index, e.g. 'doc_title', 'headers'
        field_type:
            One of: 'keyword', 'text', 'integer', 'float', or raw models.PayloadSchemaType enums
        wait:
            Whether to wait for index creation.
        """
        full_name = self._full_collection_name(collection_name)

        # Map strings to the client's model types
        field_type_lower = field_type.lower()
        try:
            if field_type_lower in ("keyword", "kw"):
                schema = models.PayloadSchemaType.KEYWORD
            elif field_type_lower == "text":
                schema = models.PayloadSchemaType.TEXT
            elif field_type_lower in ("int", "integer"):
                schema = models.PayloadSchemaType.INTEGER
            elif field_type_lower in ("float",):
                schema = models.PayloadSchemaType.FLOAT
            else:
                # allow passing models.PayloadSchemaType directly
                schema = getattr(models.PayloadSchemaType, field_type_upper := field_type.upper())
        except Exception:
            # fallback: try to treat as keyword
            logger.warning("Unknown field_type '%s' - falling back to KEYWORD", field_type)
            schema = models.PayloadSchemaType.KEYWORD

        try:
            logger.info("Creating payload index on '%s' (field=%s, schema=%s)", full_name, field_name, schema)
            self.client.create_payload_index(collection_name=full_name, field_name=field_name, field_schema=schema, wait=wait)
            logger.info("Payload index created for %s.%s", full_name, field_name)
        except Exception as exc:
            msg = str(exc).lower()
            if "already exists" in msg:
                logger.warning("Payload index %s on %s already exists; skipping.", field_name, full_name)
            else:
                logger.exception("Failed to create payload index %s on %s", field_name, full_name)
                raise

    # -------------------------
    # Helpers
    # -------------------------
    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection (convenience)."""
        full_name = self._full_collection_name(collection_name)
        logger.info("Deleting collection %s", full_name)
        try:
            self.client.delete_collection(collection_name=full_name)
            logger.info("Collection %s deleted.", full_name)
        except Exception as exc:
            logger.exception("Failed to delete collection %s", full_name)
            raise

    def count_points(self, collection_name: str) -> int:
        """Return count of points in a collection (exact)."""
        full_name = self._full_collection_name(collection_name)
        try:
            result = self.client.count(collection_name=full_name)
            return int(result.count)
        except Exception as exc:
            logger.exception("Failed to count points in collection %s", full_name)
            raise

    def collection_exists(self, collection_name: str) -> bool:
        """Check whether a collection exists (simple wrapper)."""
        full_name = self._full_collection_name(collection_name)
        try:
            return self.client.collection_exists(collection_name=full_name)
        except Exception as exc:
            logger.exception("Failed to check collection existence for %s", full_name)
            raise


if __name__ == "__main__":
    """
    Quick demo:

    - Creates three collections (hnsw_default, hnsw_optimized, payloads) for vector_size=384
    - Generates small random vectors and fake metadata for 100 points
    - Uploads them to the 'payloads' collection in batches
    - Creates a payload index on 'doc_title' and verifies counts

    This demo is useful for running quick local tests (in-memory by default).
    """

    import os

    if 'VIRTUAL_ENV' in os.environ:
        print("Inside venv")
    else:
        print("Not in venv")

    import uuid
    import time

    logging.getLogger().setLevel(logging.INFO)

    # instantiate in-memory indexer for quick tests
    indexer = QdrantIndexer(in_memory=True, collection_prefix="mkdocs_test")

    VECTOR_DIM = 384
    N = 200
    BATCH = 100

    # create the collections
    indexer.create_collections_for_experiments(vector_size=VECTOR_DIM)

    # prepare fake data
    rng = np.random.RandomState(42)
    vectors = rng.normal(size=(N, VECTOR_DIM)).astype(np.float32)

    chunks = []
    for i in range(N):
        metadata = {
            "source": f"docs/page_{i}.md",
            "url": f"/page_{i}/",
            "headers": ["Page", f"Section {i%5}"],
            "chunk_id": str(uuid.uuid4()),
            "doc_title": f"Page {i%10}",
            # keep text out of payload for this demo; attach optionally
        }
        chunks.append({"text": f"Dummy text {i}", "metadata": metadata})

    # index into payloads collection
    COLLECTION = "payloads"
    start = time.time()
    indexer.index_documents(collection_name=COLLECTION, chunks=chunks, vectors=vectors, batch_size=BATCH)
    elapsed = time.time() - start
    logger.info("Indexing finished in %.2fs", elapsed)

    # create a payload index for doc_title (keyword)
    indexer.create_payload_index(collection_name=COLLECTION, field_name="doc_title", field_type="keyword")

    # verify counts
    count = indexer.count_points(collection_name=COLLECTION)
    logger.info("Verified point count for collection %s: %d", indexer._full_collection_name(COLLECTION), count)

    logger.info("Demo complete.")