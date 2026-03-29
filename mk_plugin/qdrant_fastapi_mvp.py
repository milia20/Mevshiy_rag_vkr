from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from mk_plugin.document_processor import process_docs

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_INDEXING_DIR = _REPO_ROOT / "src" / "indexing"
if str(_SRC_INDEXING_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_INDEXING_DIR))

from src.indexing.qdrant_uploader import HNSWConfig, QdrantIndexer

app = FastAPI(title="mk_plugin Qdrant MVP")


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    with open("main.html", "r") as f:
        res = f.read()
    return res


class IngestRequest(BaseModel):
    docs_dir: str
    output_path: str
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_url: Optional[str] = None
    collection_name: str ="test_hnsw_default" #"mkdocs_docs"
    force_recreate: bool = False
    chunk_size: int = 512
    chunk_overlap: int = 50
    add_embeddings: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 256
    create_payload_indexes: bool = True


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_url: Optional[str] = None
    collection_name: str = "test_hnsw_default"
    limit: int = 10
    score_threshold: Optional[float] = None
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    filter_dict: Optional[Dict[str, Any]] = None


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest) -> Dict[str, Any]:
    process_docs(
        docs_dir=req.docs_dir,
        output_path=req.output_path,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
    )

    indexer = QdrantIndexer(
        host=req.qdrant_host,
        port=req.qdrant_port,
        url=req.qdrant_url,
        collection_name=req.collection_name,
    )

    if req.force_recreate or not indexer.client.collection_exists(req.collection_name):
        vector_size = 384
        if req.add_embeddings:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(req.embedding_model)
            vector_size = int(model.get_sentence_embedding_dimension())

        indexer.create_collection(
            vector_size=vector_size,
            hnsw_config=HNSWConfig(m=16, ef_construct=100),
            force_recreate=req.force_recreate,
        )

    stats, vector_dim = indexer.index_documents_from_file(
        file_path=req.output_path,
        batch_size=req.batch_size,
        add_embeddings=req.add_embeddings,
        embedding_model=req.embedding_model,
    )

    if req.create_payload_indexes:
        indexer.create_default_payload_indexes()

    verification = indexer.verify_index()
    indexer.close()

    return {
        "collection": req.collection_name,
        "vector_dim": vector_dim,
        "upload_stats": stats,
        "verification": verification,
        "output_path": req.output_path,
    }


@app.post("/search")
def search(req: SearchRequest) -> Dict[str, Any]:
    indexer = QdrantIndexer(
        host=req.qdrant_host,
        port=req.qdrant_port,
        url=req.qdrant_url,
        collection_name=req.collection_name,
    )

    results = indexer.search(
        query_text=req.query,
        limit=req.limit,
        filter_dict=req.filter_dict,
        score_threshold=req.score_threshold,
        model_name=req.embedding_model,
        collection_name=req.collection_name,
    )
    indexer.close()

    return {"collection": req.collection_name, "results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)