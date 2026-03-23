from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from sentence_transformers import SentenceTransformer
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

    def add_embeddings_to_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Add embeddings to chunks using SentenceTransformer model.
        
        Parameters:
        -----------
        chunks: List[Dict[str, Any]]
            List of chunks with 'text' field
        model_name: str
            Name of the sentence transformer model
        batch_size: int
            Batch size for embedding generation
        show_progress: bool
            Whether to show progress bar
            
        Returns:
        --------
        List[Dict[str, Any]]
            Chunks with added 'vector' field containing embeddings
        """
        logging.info(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name)
        
        texts = [chunk.get("text", "") for chunk in chunks]
        
        if show_progress:
            logging.info(f"Generating embeddings for {len(texts)} chunks...")
        
        # Generate embeddings in batches
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        # Add embeddings to chunks
        chunks_with_embeddings = []
        for i, chunk in enumerate(chunks):
            chunk_copy = chunk.copy()
            chunk_copy["vector"] = embeddings[i].tolist()
            chunks_with_embeddings.append(chunk_copy)
        
        logging.info(f"Generated embeddings with dimension: {embeddings.shape[1]}")
        return chunks_with_embeddings
    
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




def load_chunks_with_embeddings(chunks_path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> tuple[List[Dict[str, Any]], int]:
    """
    Load chunks from JSONL file and add embeddings.
    
    Parameters:
    -----------
    chunks_path: str
        Path to chunks JSONL file
    model_name: str
        Name of sentence transformer model
        
    Returns:
    --------
    tuple[List[Dict[str, Any]], int]
        Chunks with embeddings and embedding dimension
    """
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    
    # Create indexer to add embeddings
    indexer = QdrantIndexer()
    chunks_with_embeddings = indexer.add_embeddings_to_chunks(
        chunks, 
        model_name=model_name,
        show_progress=True
    )
    
    # Get embedding dimension from first chunk
    vector_size = len(chunks_with_embeddings[0]["vector"])
    
    return chunks_with_embeddings, vector_size


def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    
    # Load chunks for English documentation
    chunks_path_en = "./processed/chunks_en.jsonl"
    chunks_en, vector_size = load_chunks_with_embeddings(chunks_path_en)
    
    # Load chunks for Russian documentation  
    chunks_path_ru = "./processed/chunks_ru.jsonl"
    chunks_ru, _ = load_chunks_with_embeddings(chunks_path_ru)
    
    indexer = QdrantIndexer(host="localhost", port=6333)
    
    # Create collections for English docs
    created_en = indexer.setup_experiment_collections(vector_size, recreate=True, base_name="mkdocs_docs_en")
    logging.info(f"Created English collections: {created_en}")
    
    # Create collections for Russian docs
    created_ru = indexer.setup_experiment_collections(vector_size, recreate=True, base_name="mkdocs_docs_ru")
    logging.info(f"Created Russian collections: {created_ru}")
    
    # Index English documents
    for coll_name in created_en.values():
        count = indexer.index_documents(chunks_en, batch_size=200, collection_name=coll_name)
        logging.info(f"Uploaded {count} English points to {coll_name}")
    
    # Index Russian documents
    for coll_name in created_ru.values():
        count = indexer.index_documents(chunks_ru, batch_size=200, collection_name=coll_name)
        logging.info(f"Uploaded {count} Russian points to {coll_name}")
    
    logging.info("Indexing completed successfully!")

if __name__ == "__main__":
    main()