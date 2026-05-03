"""
Qdrant Indexing Pipeline for RAG
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import logging as transformers_logging

logger = logging.getLogger(__name__)

_FIELD_TYPE_MAP: dict[str, PayloadSchemaType] = {
    "keyword": PayloadSchemaType.KEYWORD,
    "integer": PayloadSchemaType.INTEGER,
    "float": PayloadSchemaType.FLOAT,
    "bool": PayloadSchemaType.BOOL,
    "datetime": PayloadSchemaType.DATETIME,
    "text": PayloadSchemaType.TEXT,
    "geo": PayloadSchemaType.GEO,
}


@dataclass
class CollectionSpec:
    """Specification for experiment collection"""

    name: str
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    with_payload_indexes: bool = False
    description: str = ""


@dataclass
class HNSWConfig:
    """HNSW configuration for collection optimization experiments"""

    m: int = 16
    ef_construct: int = 100
    full_scan_threshold: int = 10000
    max_indexing_threads: int = 0

    def to_dict(self) -> dict:
        return {
            "m": self.m,
            "ef_construct": self.ef_construct,
            "full_scan_threshold": self.full_scan_threshold,
            "max_indexing_threads": self.max_indexing_threads,
        }


class QdrantIndexer:
    """
    Qdrant indexing pipeline for RAG thesis experiments.
    Combines production features with experimental multi-collection support.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "mkdocs_docs",
        *,
        url: str | None = None,
        in_memory: bool = False,
        local_path: str | None = None,
        api_key: str | None = None,
        https: bool = False,
        prefer_grpc: bool = True,
        timeout: float = 300.0,
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.in_memory = in_memory
        self.timeout = timeout
        self._collection_created = False
        self._indexed_count = 0

        # Initialize Qdrant client
        if in_memory:
            logger.info("Initializing Qdrant client in IN-MEMORY mode (testing)")
            self.client = QdrantClient(location=":memory:")
        elif local_path is not None:
            logger.info(f"Initializing Qdrant client with local path: {local_path}")
            self.client = QdrantClient(path=local_path)
        else:
            if url is None:
                url = f"http://{host}:{port}"
            logger.info(f"Initializing Qdrant client: {url}")
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
                https=https,
                prefer_grpc=prefer_grpc,
                timeout=timeout,
            )

    def create_collection(
        self,
        vector_size: int,
        hnsw_config: HNSWConfig | None = None,
        distance: Distance = Distance.COSINE,
        on_disk_payload: bool = False,
        force_recreate: bool = False,
        collection_name: str | None = None,
    ) -> str:
        """Create collection with specific HNSW parameters."""
        name = collection_name or self.collection_name

        if hnsw_config is None:
            hnsw_config = HNSWConfig()

        collection_exists = self.client.collection_exists(name)

        if collection_exists:
            if force_recreate:
                logger.warning(f"Deleting existing collection: {name}")
                self.client.delete_collection(name, timeout=120)
            else:
                logger.info(f"Collection {name} already exists")
                self._collection_created = True
                return name

        logger.info(
            f"Creating collection '{name}' with "
            f"vector_size={vector_size}, distance={distance.name}, "
            f"HNSW(m={hnsw_config.m}, ef_construct={hnsw_config.ef_construct})"
        )

        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance,
                on_disk=on_disk_payload,
            ),
            hnsw_config=HnswConfigDiff(
                m=hnsw_config.m,
                ef_construct=hnsw_config.ef_construct,
                full_scan_threshold=hnsw_config.full_scan_threshold,
                max_indexing_threads=hnsw_config.max_indexing_threads,
            ),
            on_disk_payload=on_disk_payload,
        )

        self._collection_created = True
        logger.info(f"Collection '{name}' created successfully")
        return name

    def setup_experiment_collections(
        self,
        vector_size: int,
        *,
        base_name: str | None = None,
        force_recreate: bool = False,
        specs: list[CollectionSpec] | None = None,
    ) -> dict[str, str]:
        """
        Create multiple collections for HNSW/payload experiments.

        Returns dict mapping spec name -> actual collection name
        """
        base = base_name or self.collection_name

        if specs is None:
            specs = [
                CollectionSpec(
                    name=f"{base}_hnsw_default",
                    hnsw_m=16,
                    hnsw_ef_construct=100,
                    description="Default HNSW configuration",
                ),
                CollectionSpec(
                    name=f"{base}_hnsw_optimized",
                    hnsw_m=32,
                    hnsw_ef_construct=256,
                    description="Optimized HNSW for better recall",
                ),
                CollectionSpec(
                    name=f"{base}_payload_indexed",
                    hnsw_m=16,
                    hnsw_ef_construct=100,
                    with_payload_indexes=True,
                    description="With payload indexes for filtering",
                ),
            ]

        created: dict[str, str] = {}
        for spec in specs:
            logger.info(f"Creating experiment collection: {spec.name} ({spec.description})")

            self.create_collection(
                vector_size=vector_size,
                hnsw_config=HNSWConfig(m=spec.hnsw_m, ef_construct=spec.hnsw_ef_construct),
                collection_name=spec.name,
                force_recreate=force_recreate,
            )
            created[spec.name] = spec.name

            if spec.with_payload_indexes:
                self.create_default_payload_indexes(collection_name=spec.name)

        logger.info(f"Created {len(created)} experiment collections")
        return created

    def add_embeddings_to_chunks(
        self,
        chunks: list[dict[str, Any]],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        show_progress: bool = True,
        normalize_embeddings: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Add embeddings to chunks using SentenceTransformer model.

        Returns chunks with added 'vector' field.
        """
        logger.info(f"Loading embedding model: {model_name}")
        transformers_logging.set_verbosity_error()
        model = SentenceTransformer(model_name)

        texts = [chunk.get("text", "") for chunk in chunks]

        if show_progress:
            logger.info(f"Generating embeddings for {len(texts)} chunks...")

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize_embeddings,
        )

        chunks_with_embeddings = []
        for i, chunk in enumerate(chunks):
            chunk_copy = chunk.copy()
            chunk_copy["vector"] = embeddings[i].tolist()
            chunks_with_embeddings.append(chunk_copy)

        logger.info(f"Generated embeddings with dimension: {embeddings.shape[1]}")
        return chunks_with_embeddings, int(embeddings.shape[1])

    def create_payload_index(
        self,
        field_name: str,
        field_type: str = "keyword",
        *,
        collection_name: str | None = None,
    ) -> bool:
        """Create index for payload field to optimize filtered search."""
        name = collection_name or self.collection_name

        field_schema = _FIELD_TYPE_MAP.get(field_type.lower())
        if field_schema is None:
            raise ValueError(f"Unsupported field_type={field_type!r}. Supported: {sorted(_FIELD_TYPE_MAP.keys())}")

        logger.info(f"Creating payload index for field '{field_name}' with type '{field_type}'")

        try:
            self.client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )
            logger.info(f"Payload index created for '{field_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to create payload index: {e}")
            return False

    def create_default_payload_indexes(
        self,
        *,
        collection_name: str | None = None,
    ) -> dict[str, bool]:
        """Create default payload indexes for common filtering fields."""
        indexes = {
            "doc_title": "keyword",
            "source": "keyword",
            "headers": "keyword",
            "chunk_id": "keyword",
        }

        results = {}
        for field_name, field_type in indexes.items():
            results[field_name] = self.create_payload_index(field_name, field_type, collection_name=collection_name)

        return results

    def _generate_point_id(self, chunk_id: str) -> str:
        """Generate consistent point ID from chunk metadata"""
        try:
            uuid.UUID(chunk_id)
            return chunk_id
        except (ValueError, TypeError):
            return hashlib.md5(str(chunk_id).encode()).hexdigest()

    def _prepare_payload(self, chunk: dict) -> dict[str, Any]:
        """Prepare payload from chunk metadata."""
        metadata = chunk.get("metadata", {}) or {}
        payload = {
            "text": chunk.get("text", ""),
            "source": metadata.get("source", "unknown"),
            "url": metadata.get("url", ""),
            "doc_title": metadata.get("doc_title", ""),
            "headers": metadata.get("headers", []),
            "chunk_id": metadata.get("chunk_id", ""),
        }

        for key, value in metadata.items():
            if key not in payload:
                payload[key] = value

        # Add top-level fields (except vector/embedding/metadata)
        for k, v in chunk.items():
            if k not in {"vector", "embedding", "metadata", "text"} and k not in payload:
                payload[k] = v

        return payload

    def index_documents(
        self,
        chunks: list[dict[str, Any]],
        batch_size: int = 256,
        vector_field: str = "vector",
        show_progress: bool = True,
        *,
        collection_name: str | None = None,
    ) -> dict[str, int]:
        """Upload chunks to Qdrant with batching and progress tracking."""
        name = collection_name or self.collection_name

        if not self.client.collection_exists(name):
            raise RuntimeError(f"Collection '{name}' does not exist. Call create_collection() first.")

        if not chunks:
            logger.warning("No chunks to index")
            return {"uploaded": 0, "failed": 0}

        logger.info(f"Indexing {len(chunks)} chunks to '{name}' with batch_size={batch_size}")

        uploaded = 0
        failed = 0

        iterator = tqdm(
            range(0, len(chunks), batch_size),
            desc=f"Uploading to {name}",
            unit="batch",
            disable=not show_progress,
        )

        for batch_start in iterator:
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]

            points = []
            for chunk in batch_chunks:
                try:
                    vector = chunk.get(vector_field) or chunk.get("embedding")
                    if vector is None:
                        logger.warning(f"Chunk missing vector: {chunk.get('metadata', {}).get('chunk_id', 'unknown')}")
                        failed += 1
                        continue

                    chunk_id = chunk.get("metadata", {}).get("chunk_id", str(uuid.uuid4()))
                    point = PointStruct(
                        id=self._generate_point_id(chunk_id),
                        vector=list(map(float, vector)),
                        payload=self._prepare_payload(chunk),
                    )
                    points.append(point)
                except Exception as e:
                    logger.error(f"Error preparing point: {e}")
                    failed += 1

            if points:
                try:
                    self.client.upload_points(
                        collection_name=name,
                        points=points,
                        wait=True,
                    )
                    uploaded += len(points)

                    if show_progress:
                        iterator.set_postfix({"uploaded": uploaded, "failed": failed})

                except Exception as e:
                    logger.error(f"Batch upload failed: {e}")
                    error_text = str(e)
                    if "Vector dimension error" in error_text or "expected dim" in error_text:
                        raise RuntimeError(
                            "Vector dimension mismatch. "
                            "Recreate the collection with the same dimension as your embedding model. "
                            f"Underlying error: {error_text}"
                        ) from e
                    failed += len(points)

        self._indexed_count = uploaded
        logger.info(f"Indexing complete for '{name}': {uploaded} uploaded, {failed} failed")

        return {"uploaded": uploaded, "failed": failed}

    def index_documents_from_file(
        self,
        file_path: str,
        batch_size: int = 256,
        vector_field: str = "vector",
        show_progress: bool = True,
        *,
        collection_name: str | None = None,
        add_embeddings: bool = True,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> tuple[dict[str, int], int | None]:
        """
        Index documents from JSONL file.

        Returns: (upload_stats, vector_dimension)
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Loading chunks from {file_path}")

        with open(file_path, encoding="utf-8") as f:
            chunks = [
                json.loads(line) for line in tqdm(f, desc="Loading chunks", disable=not show_progress) if line.strip()
            ]

        logger.info(f"Loaded {len(chunks)} chunks from file")

        vector_dim = None
        if add_embeddings and chunks:
            chunks, vector_dim = self.add_embeddings_to_chunks(
                chunks,
                model_name=embedding_model,
                show_progress=show_progress,
            )

        stats = self.index_documents(
            chunks,
            batch_size=batch_size,
            vector_field=vector_field,
            show_progress=show_progress,
            collection_name=collection_name,
        )

        return stats, vector_dim

    def verify_index(self, *, collection_name: str | None = None) -> dict[str, Any]:
        """Verify indexing by counting points and checking collection info."""
        name = collection_name or self.collection_name

        logger.info(f"Verifying index for collection: {name}")

        try:
            collection_info = self.client.get_collection(name)

            params = getattr(getattr(collection_info, "config", None), "params", None)
            vectors_cfg = None
            if params is not None:
                vectors_cfg = getattr(params, "vectors_config", None)
                if vectors_cfg is None:
                    vectors_cfg = getattr(params, "vectors", None)

            vector_size = None
            distance_metric = None
            if vectors_cfg is not None:
                # Most common case: single unnamed vector config
                if hasattr(vectors_cfg, "size"):
                    vector_size = vectors_cfg.size
                    distance_metric = str(getattr(vectors_cfg, "distance", None))
                # Some versions store vectors as a dict of named configs
                elif isinstance(vectors_cfg, dict) and vectors_cfg:
                    first_cfg = next(iter(vectors_cfg.values()))
                    vector_size = getattr(first_cfg, "size", None)
                    distance_metric = str(getattr(first_cfg, "distance", None))

            hnsw_cfg = getattr(params, "hnsw_config", None) if params is not None else None
            hnsw_m = getattr(hnsw_cfg, "m", None)
            hnsw_ef = getattr(hnsw_cfg, "ef_construct", None)

            verification = {
                "collection_name": name,
                "points_count": collection_info.points_count,
                "indexed_vectors": getattr(
                    collection_info,
                    "vectors_count",
                    getattr(collection_info, "indexed_vectors_count", None),
                ),
                "vector_size": vector_size,
                "distance_metric": distance_metric,
                "hnsw_config": {
                    "m": hnsw_m,
                    "ef_construct": hnsw_ef,
                },
                "payload_schema": (
                    {k: str(v) for k, v in collection_info.payload_schema.items()}
                    if collection_info.payload_schema
                    else {}
                ),
                "status": "verified" if collection_info.points_count > 0 else "empty",
            }

            logger.info(
                f"Verification complete: {verification['points_count']} points, "
                f"HNSW(m={verification['hnsw_config']['m']}, "
                f"ef={verification['hnsw_config']['ef_construct']})"
            )

            return verification

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {"status": "error", "error": str(e), "collection_name": name}

    def get_collection_stats(self, *, collection_name: str | None = None) -> dict[str, Any]:
        """Get detailed collection statistics"""
        name = collection_name or self.collection_name
        info = self.client.get_collection(name)
        return {
            "name": name,
            "points_count": info.points_count,
            "vectors_count": getattr(info, "vectors_count", None),
            "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
            "payload_schema": dict(info.payload_schema) if info.payload_schema else {},
        }

    def search(
        self,
        query_text: str | None = None,
        query_vector: list[float] | None = None,
        limit: int = 10,
        filter_dict: dict | None = None,
        score_threshold: float | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        collection_name: str | None = None,
    ) -> list[dict]:
        """Search for similar documents."""
        name = collection_name or self.collection_name

        # Generate query vector if text provided
        if query_vector is None:
            if query_text is None:
                raise ValueError("Either query_text or query_vector must be provided")
            logger.info(f"Generating embedding for query: {query_text[:50]}...")
            model = SentenceTransformer(model_name)
            query_vector = model.encode([query_text], normalize_embeddings=True)[0].tolist()

        # Build filter
        search_filter = None
        if filter_dict:
            conditions = []
            for key, value in filter_dict.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )
            search_filter = Filter(must=conditions)

        # Qdrant client API compatibility:
        # - Some versions use client.search(query_vector=...)
        # - Some versions use client.query_points(query=<vector>)
        try:
            if hasattr(self.client, "search"):
                try:
                    results = self.client.search(
                        collection_name=name,
                        query_vector=query_vector,
                        query_filter=search_filter,
                        limit=limit,
                        score_threshold=score_threshold,
                        with_payload=True,
                        with_vectors=False,
                    )
                except TypeError:
                    results = self.client.search(
                        collection_name=name,
                        query_vector=query_vector,
                        filter=search_filter,
                        limit=limit,
                        score_threshold=score_threshold,
                        with_payload=True,
                        with_vectors=False,
                    )
            else:
                try:
                    results = self.client.query_points(
                        collection_name=name,
                        query=query_vector,
                        query_filter=search_filter,
                        limit=limit,
                        score_threshold=score_threshold,
                        with_payload=True,
                        with_vectors=False,
                    )
                except TypeError:
                    results = self.client.query_points(
                        collection_name=name,
                        query=query_vector,
                        filter=search_filter,
                        limit=limit,
                        score_threshold=score_threshold,
                        with_payload=True,
                        with_vectors=False,
                    )
        except Exception as e:
            raise RuntimeError(f"Qdrant search failed for collection '{name}': {e}") from e

        points = results.points if hasattr(results, "points") else results

        return [
            {
                "id": r.id,
                "score": r.score,
                "payload": r.payload,
            }
            for r in points
        ]

    def delete_collection(self, collection_name: str | None = None) -> bool:
        """Delete the collection"""
        name = collection_name or self.collection_name
        try:
            self.client.delete_collection(name)
            logger.info(f"Collection '{name}' deleted")
            self._collection_created = False
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False

    def close(self):
        """Close client connection"""
        if hasattr(self.client, "close"):
            self.client.close()
        logger.info("Qdrant client closed")


def create_cli() -> argparse.ArgumentParser:
    """Create CLI argument parser"""
    parser = argparse.ArgumentParser(
        description="Qdrant Indexing Pipeline for RAG Thesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default HNSW configuration with embeddings
  python qdrant_indexer.py --input chunks.jsonl --add-embeddings

  # Optimized HNSW configuration
  python qdrant_indexer.py --config optimized --input chunks.jsonl --collection docs_optimized

  # Multi-collection experiments
  python qdrant_indexer.py --experiments --input chunks.jsonl --base-name mkdocs_docs

  # In-memory mode for testing
  python qdrant_indexer.py --in-memory --input chunks.jsonl

  # Search test
  python qdrant_indexer.py --search "your query text" --collection docs_default
        """,
    )

    # Connection arguments
    parser.add_argument("--host", type=str, default="localhost", help="Qdrant host (default: localhost)")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port (default: 6333)")
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Use in-memory storage (testing mode)",
    )

    # Collection arguments
    parser.add_argument(
        "--collection",
        type=str,
        default="mkdocs_docs",
        help="Collection name (default: mkdocs_docs)",
    )
    parser.add_argument(
        "--base-name",
        type=str,
        default=None,
        help="Base name for experiment collections",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=384,
        help="Vector dimension (default: 384 for all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=["default", "optimized", "custom"],
        default="default",
        help="HNSW configuration preset",
    )
    parser.add_argument(
        "--hnsw-m",
        type=int,
        default=16,
        help="HNSW m parameter (default: 16)",
    )
    parser.add_argument(
        "--hnsw-ef",
        type=int,
        default=100,
        help="HNSW ef_construct parameter (default: 100)",
    )

    # Indexing arguments
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to JSONL file with chunks",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for uploading (default: 256)",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Delete and recreate collection if exists",
    )
    parser.add_argument(
        "--experiments",
        action="store_true",
        help="Create multiple collections for experiments",
    )

    # Embedding arguments
    embedding_group = parser.add_mutually_exclusive_group()
    embedding_group.add_argument(
        "--add-embeddings",
        dest="add_embeddings",
        action="store_true",
        help="Generate embeddings using SentenceTransformer (default)",
    )
    embedding_group.add_argument(
        "--no-embeddings",
        dest="add_embeddings",
        action="store_false",
        help="Do not generate embeddings; expects vectors already present in input file",
    )
    parser.set_defaults(add_embeddings=True)
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )

    # Payload index arguments
    parser.add_argument(
        "--create-indexes",
        action="store_true",
        help="Create default payload indexes",
    )
    parser.add_argument(
        "--index-field",
        type=str,
        action="append",
        help="Create index for specific field (can be used multiple times)",
    )

    # Verification arguments
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify index after upload",
    )
    parser.add_argument(
        "--output-stats",
        type=str,
        help="Path to save indexing statistics (JSON)",
    )

    # Search arguments
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="Search query text",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=5,
        help="Number of search results (default: 5)",
    )

    # Logging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser


def main():
    parser = create_cli()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    try:
        indexer = QdrantIndexer(
            host=args.host,
            port=args.port,
            collection_name=args.collection,
            in_memory=args.in_memory,
        )

        if args.config == "default":
            hnsw_config = HNSWConfig(m=16, ef_construct=100)
        elif args.config == "optimized":
            hnsw_config = HNSWConfig(m=32, ef_construct=256)
        else:  # custom
            hnsw_config = HNSWConfig(m=args.hnsw_m, ef_construct=args.hnsw_ef)

        if args.experiments:
            logger.info("EXPERIMENT Creating multiple collections")

            # Load and embed chunks first to get vector size
            with open(args.input, encoding="utf-8") as f:
                chunks = [json.loads(line) for line in f if line.strip()]

            vector_size = args.vector_size
            chunks, vector_size = indexer.add_embeddings_to_chunks(
                chunks,
                model_name=args.embedding_model,
                show_progress=True,
            )

            # Create experiment collections
            created = indexer.setup_experiment_collections(
                vector_size=vector_size,
                base_name=args.base_name,
                force_recreate=args.force_recreate,
            )

            # Index documents to all collections
            for coll_name in created.values():
                stats = indexer.index_documents(
                    chunks,
                    batch_size=args.batch_size,
                    collection_name=coll_name,
                )
                logger.info(f"Uploaded {stats['uploaded']} points to {coll_name}")

        else:
            # Single collection mode
            if args.force_recreate or not indexer.client.collection_exists(args.collection):
                if args.add_embeddings:
                    model = SentenceTransformer(args.embedding_model)
                    vector_size = int(model.get_sentence_embedding_dimension())
                else:
                    vector_size = args.vector_size

                indexer.create_collection(
                    vector_size=vector_size,
                    hnsw_config=hnsw_config,
                    force_recreate=args.force_recreate,
                )

            # Index documents from file
            _stats, _vector_dim = indexer.index_documents_from_file(
                file_path=args.input,
                batch_size=args.batch_size,
                add_embeddings=args.add_embeddings,
                embedding_model=args.embedding_model,
            )

            # Create payload indexes
            if args.create_indexes:
                indexer.create_default_payload_indexes()

            if args.index_field:
                for field in args.index_field:
                    indexer.create_payload_index(field, "keyword")

        # Verify index
        if args.verify:
            verification = indexer.verify_index()
            print("INDEX VERIFICATION")
            for key, value in verification.items():
                print(f"{key}: {value}")

            # Save stats if requested
            if args.output_stats:
                output_path = Path(args.output_stats)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "hnsw_config": hnsw_config.to_dict(),
                            "verification": verification,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
                logger.info(f"Statistics saved to {args.output_stats}")

        # Search test
        if args.search:
            print(f"SEARCH TEST: {args.search}")
            results = indexer.search(
                query_text=args.search,
                limit=args.search_limit,
                model_name=args.embedding_model,
            )
            for i, r in enumerate(results, 1):
                print(f"\n{i}. Score: {r['score']:.4f}")
                print(f"   ID: {r['id']}")
                print(f"   Text: {r['payload'].get('text', '')[:200]}...")
            print("=" * 60)

        # Summary
        print("INDEXING COMPLETE")
        print(f"Collection(s): {args.collection}")
        print(f"HNSW Config: m={hnsw_config.m}, ef_construct={hnsw_config.ef_construct}")

        indexer.close()

    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    if len(sys.argv) == 1:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )

        CHUNKS_FILE = r"D:\P_work\Rag-VKR\src\indexing\processed\chunks_en.jsonl"
        CHUNKS_FILE = r"D:\P_work\Rag-VKR\src\indexing\processed\agent-framework.jsonl"
        EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
        BATCH_SIZE = 256

        TEST_CONFIGS = {
            "hnsw_default": HNSWConfig(m=16, ef_construct=100),
            "hnsw_optimized": HNSWConfig(m=32, ef_construct=256),
        }

        try:
            print("\n[1/5] Checking input file...")
            if not Path(CHUNKS_FILE).exists():
                raise FileNotFoundError(f"Chunks file not found: {CHUNKS_FILE}")

            # Count lines
            with open(CHUNKS_FILE, encoding="utf-8") as f:
                chunk_count = sum(1 for line in f if line.strip())
            print(f"✓ Found {chunk_count} chunks in {CHUNKS_FILE}")

            print("\n[2/5] Initializing Qdrant client...")
            indexer = QdrantIndexer(
                host="localhost",
                port=6333,
                collection_name="test_collection",
                in_memory=False,  # Set to True for memory only testing
            )
            print("✓ Qdrant client initialized")

            print("\n[3/5] Loading chunks and generating embeddings...")
            with open(CHUNKS_FILE, encoding="utf-8") as f:
                chunks = [json.loads(line) for line in tqdm(f, desc="Loading") if line.strip()]

            print(f"✓ Loaded {len(chunks)} chunks")

            chunks, vector_size = indexer.add_embeddings_to_chunks(
                chunks,
                model_name=EMBEDDING_MODEL,
                show_progress=True,
            )
            print(f"✓ Generated embeddings (dimension: {vector_size})")

            print("\n[4/5] Creating collections and indexing...")
            for config_name, hnsw_config in TEST_CONFIGS.items():
                coll_name = f"test_{config_name}"

                # Create collection
                indexer.create_collection(
                    vector_size=vector_size,
                    hnsw_config=hnsw_config,
                    collection_name=coll_name,
                    force_recreate=True,
                )

                # Index documents
                stats = indexer.index_documents(
                    chunks,
                    batch_size=BATCH_SIZE,
                    collection_name=coll_name,
                    show_progress=True,
                )

                # Create payload indexes
                indexer.create_default_payload_indexes(collection_name=coll_name)

                print(f"✓ {coll_name}: {stats['uploaded']} uploaded, {stats['failed']} failed")

            print("\n[5/5] Verifying and testing search...")
            for config_name in TEST_CONFIGS.keys():
                coll_name = f"test_{config_name}"
                verification = indexer.verify_index(collection_name=coll_name)
                print(f"\n{coll_name}:")
                print(f"  Points: {verification.get('points_count', 'N/A')}")
                print(
                    f"  HNSW: m={verification.get('hnsw_config', {}).get('m', 'N/A')}, "
                    f"ef={verification.get('hnsw_config', {}).get('ef_construct', 'N/A')}"
                )

            print("\n" + "-" * 70)
            print("SEARCH TEST")
            print("-" * 70)
            test_queries = [
                "How to install Python?",
                "What is RAG?",
                "Documentation best practices",
            ]

            for query in test_queries:
                print(f"\nQuery: {query}")
                results = indexer.search(
                    query_text=query,
                    limit=3,
                    model_name=EMBEDDING_MODEL,
                    collection_name="test_hnsw_default",
                )
                for i, r in enumerate(results, 1):
                    print(f"  {i}. Score: {r['score']:.4f} | {r['payload'].get('text', '')[:100]}...")

        except FileNotFoundError as e:
            logger.error(f"File error: {e}")
        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
            print(f"Test failed: {e}")
        finally:
            # Cleanup test collections
            # for config_name in TEST_CONFIGS.keys():
            #     indexer.delete_collection(f"test_{config_name}")
            indexer.close()
            print("✓ Qdrant client closed")
    else:
        main()
