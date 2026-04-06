import argparse
import hashlib
from collections import Counter

try:
    from qdrant_client import QdrantClient, models
except ImportError as e:
    raise SystemExit(
        "Missing dependency: qdrant-client.\n"
        "Install project dependencies, for example:\n"
        "  - pip install -e .\n"
        "  - or: pip install qdrant-client\n"
        "Then re-run: python src/test.py\n"
    ) from e


def _fake_dense_embedding(text: str, dim: int) -> list[float]:
    """Deterministic local embedding.

    Why: Qdrant's `models.Document(...)` requires server-side inference, which is
    usually not enabled. This function keeps the example runnable everywhere.
    """
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=dim)
    # map bytes [0..255] -> floats [-1..1]
    return [((b / 255.0) * 2.0 - 1.0) for b in h.digest()]


def _fake_sparse_vector(text: str, vocab_size: int = 2048) -> models.SparseVector:
    tokens = [t for t in text.lower().split() if t]
    counts = Counter(tokens)
    indices: list[int] = []
    values: list[float] = []
    for tok, tf in counts.items():
        idx = int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=2).digest(), "big")
        idx %= vocab_size
        indices.append(idx)
        values.append(float(tf))
    return models.SparseVector(indices=indices, values=values)


def _recreate_collection(client: QdrantClient, name: str, **kwargs) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        client.delete_collection(name)
    client.create_collection(name, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Qdrant dense/sparse/hybrid demo (self-contained).")
    parser.add_argument("--dim", type=int, default=64, help="Dense vector size")
    parser.add_argument("--top-k", type=int, default=5, help="How many results to return")
    args = parser.parse_args()

    client = QdrantClient(":memory:")

    dense_collection = "dense_collection"
    sparse_collection = "sparse_collection"
    hybrid_collection = "hybrid_collection"

    _recreate_collection(
        client,
        dense_collection,
        vectors_config=models.VectorParams(size=args.dim, distance=models.Distance.COSINE),
    )

    _recreate_collection(
        client,
        sparse_collection,
        sparse_vector_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )

    _recreate_collection(
        client,
        hybrid_collection,
        vectors_config={"dense": models.VectorParams(size=args.dim, distance=models.Distance.COSINE)},
        sparse_vector_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )

    texts = [
        "qdrant is a vector database",
        "bm25 is a sparse retrieval model",
        "hybrid search combines dense and sparse",
        "python fastapi and qdrant client",
        "vector similarity search with cosine distance",
    ]

    dense_points: list[models.PointStruct] = []
    sparse_points: list[models.PointStruct] = []
    hybrid_points: list[models.PointStruct] = []
    for i, text in enumerate(texts, start=1):
        payload = {"text": text}
        dense_vec = _fake_dense_embedding(text, args.dim)
        sparse_vec = _fake_sparse_vector(text)
        dense_points.append(models.PointStruct(id=i, vector=dense_vec, payload=payload))
        sparse_points.append(models.PointStruct(id=i, vector={"bm25": sparse_vec}, payload=payload))
        hybrid_points.append(
            models.PointStruct(id=i, vector={"dense": dense_vec, "bm25": sparse_vec}, payload=payload)
        )

    client.upsert(dense_collection, points=dense_points)
    client.upsert(sparse_collection, points=sparse_points)
    client.upsert(hybrid_collection, points=hybrid_points)

    query_text = "qdrant hybrid search"
    dense_query = _fake_dense_embedding(query_text, args.dim)
    sparse_query = _fake_sparse_vector(query_text)

    dense_res = client.query_points(
        dense_collection,
        query=dense_query,
        limit=args.top_k,
        with_payload=True,
    )
    print("\nDENSE results:")
    for p in dense_res.points:
        print(f"- id={p.id} score={p.score:.4f} text={p.payload.get('text')}")

    sparse_res = client.query_points(
        sparse_collection,
        query=sparse_query,
        using="bm25",
        limit=args.top_k,
        with_payload=True,
    )
    print("\nSPARSE results:")
    for p in sparse_res.points:
        print(f"- id={p.id} score={p.score:.4f} text={p.payload.get('text')}")

    hybrid_res = client.query_points(
        hybrid_collection,
        query=models.FusionQuery(fusion=models.Fusion.DBSF),
        prefetch=[
            models.Prefetch(query=dense_query, using="dense", limit=args.top_k),
            models.Prefetch(query=sparse_query, using="bm25", limit=args.top_k),
        ],
        limit=args.top_k,
        with_payload=True,
    )
    print("\nHYBRID (Fusion.DBSF) results:")
    for p in hybrid_res.points:
        print(f"- id={p.id} score={p.score:.4f} text={p.payload.get('text')}")


if __name__ == "__main__":
    main()
