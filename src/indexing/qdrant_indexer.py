from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from tqdm.auto import tqdm

from qdrant_client import QdrantClient
from qdrant_client import models


_FIELD_TYPE_MAP: Dict[str, models.PayloadSchemaType] = {
    "keyword": models.PayloadSchemaType.KEYWORD,
    "integer": models.PayloadSchemaType.INTEGER,
    "float": models.PayloadSchemaType.FLOAT,
    "bool": models.PayloadSchemaType.BOOL,
    "datetime": models.PayloadSchemaType.DATETIME,
    "text": models.PayloadSchemaType.TEXT,
}


@dataclass(frozen=True)
class CollectionSpec:
    name: str
    hnsw_m: int
    hnsw_ef_construct: int
    with_payload_indexes: bool = False


class QdrantIndexer:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "mkdocs_docs",
        *,
        url: Optional[str] = None,
        in_memory: bool = False,
        local_path: Optional[str] = None,
        prefer_grpc: bool = True,
        timeout: Optional[float] = 60.0,
    ):
        self.collection_name = collection_name

        if in_memory:
            self.client = QdrantClient(path=":memory:")
        elif local_path is not None:
            self.client = QdrantClient(path=local_path)
        else:
            if url is None:
                url = f"http://{host}:{port}"
            self.client = QdrantClient(url=url, prefer_grpc=prefer_grpc, timeout=timeout)

    def create_collection(
        self,
        vector_size: int,
        hnsw_config: Dict[str, Any],
        *,
        collection_name: Optional[str] = None,
        recreate: bool = False,
    ) -> str:
        name = collection_name or self.collection_name

        if recreate and self.client.collection_exists(name):
            self.client.delete_collection(collection_name=name, timeout=120)

        if not self.client.collection_exists(name):
            m = int(hnsw_config.get("m", 16))
            ef_construct = int(hnsw_config.get("ef_construct", 100))

            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=int(vector_size),
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(m=m, ef_construct=ef_construct),
                ),
            )

        return name

    def create_payload_index(self, field_name: str, field_type: str = "keyword", *, collection_name: Optional[str] = None) -> None:
        name = collection_name or self.collection_name
        ft = _FIELD_TYPE_MAP.get(field_type.lower())
        if ft is None:
            raise ValueError(f"Unsupported field_type={field_type!r}. Supported: {sorted(_FIELD_TYPE_MAP.keys())}")

        self.client.create_payload_index(
            collection_name=name,
            field_name=field_name,
            field_schema=ft,
        )

    def index_documents(self, chunks: List[Dict[str, Any]], batch_size: int = 200, *, collection_name: Optional[str] = None, show_progress: bool = True) -> int:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if batch_size < 100 or batch_size > 500:
            raise ValueError("batch_size should be in range [100, 500] for this pipeline")

        name = collection_name or self.collection_name

        total_uploaded = 0
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        batch_iter = range(total_batches)
        if show_progress:
            batch_iter = tqdm(batch_iter, desc=f"Uploading points -> {name}", unit="batch")

        for b_idx in batch_iter:
            start = b_idx * batch_size
            end = min(start + batch_size, len(chunks))
            batch = chunks[start:end]
            ids: List[str] = []
            vectors: List[Sequence[float]] = []
            payloads: List[Dict[str, Any]] = []

            for i, ch in enumerate(batch):
                md = ch.get("metadata") or {}
                cid = md.get("chunk_id") or ch.get("chunk_id")
                if cid is None:
                    cid = str(total_uploaded + i)

                vec = ch.get("vector")
                if vec is None:
                    vec = ch.get("embedding")
                if vec is None:
                    raise ValueError("Each chunk must include a vector under key 'vector' or 'embedding'.")

                payload: Dict[str, Any] = {}
                payload.update(md if isinstance(md, dict) else {})
                for k, v in ch.items():
                    if k in {"vector", "embedding", "metadata"}:
                        continue
                    payload[k] = v

                ids.append(str(cid))
                vectors.append(list(map(float, vec)))
                payloads.append(payload)

            self.client.upload_points(
                collection_name=name,
                points=models.Batch(ids=ids, vectors=vectors, payloads=payloads),
                wait=True,
            )
            total_uploaded += len(batch)

        count = self.client.count(collection_name=name, exact=True).count
        if int(count) < len(chunks):
            raise RuntimeError(
                f"Point count verification failed for collection={name!r}: expected>={len(chunks)}, got={count}"
            )

        return int(count)

    def setup_experiment_collections(self, vector_size: int, *, base_name: Optional[str] = None, recreate: bool = False) -> Dict[str, str]:
        base = base_name or self.collection_name

        specs = [
            CollectionSpec(name=f"{base}_hnsw_default", hnsw_m=16, hnsw_ef_construct=100),
            CollectionSpec(name=f"{base}_hnsw_optimized", hnsw_m=32, hnsw_ef_construct=256),
            CollectionSpec(
                name=f"{base}_payload_indexed",
                hnsw_m=16,
                hnsw_ef_construct=100,
                with_payload_indexes=True,
            ),
        ]

        created: Dict[str, str] = {}
        for spec in specs:
            self.create_collection(
                vector_size=vector_size,
                hnsw_config={"m": spec.hnsw_m, "ef_construct": spec.hnsw_ef_construct},
                collection_name=spec.name,
                recreate=recreate,
            )
            created[spec.name] = spec.name

            if spec.with_payload_indexes:
                self.create_payload_index("doc_title", field_type="keyword", collection_name=spec.name)
                self.create_payload_index("headers", field_type="keyword", collection_name=spec.name)

        return created




def main():
    # Load pre-computed chunks with embeddings
    chunks_path = "../preprocessing/data/processed/chunks_en.jsonl"
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    # Initialize indexer (local Qdrant server)
    indexer = QdrantIndexer(host="localhost", port=6333)

    # Setup experiment collections
    vector_size = len(chunks[0]["metadata"])
    created = indexer.setup_experiment_collections(vector_size, recreate=True)
    print("Created collections:", created)

    # Upload to each collection
    for coll_name in created.values():
        count = indexer.index_documents(chunks, batch_size=200, collection_name=coll_name)
        print(f"Uploaded {count} points to {coll_name}")

if __name__ == "__main__":
    main()