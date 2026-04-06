import json

from src.indexing.qdrant_uploader import HNSWConfig, QdrantIndexer

if __name__ == "__main__":

    CHUNKS_FILE = r"D:\P_work\Rag-VKR\src\custom_dataset\ru_rag\chunks.jsonl"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    BATCH_SIZE = 256

    TEST_CONFIGS = {
        "hnsw_default": HNSWConfig(m=16, ef_construct=100),
        "hnsw_optimized": HNSWConfig(m=32, ef_construct=256),
    }

    print("\n[1/5] Checking input file")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunk_count = sum(1 for line in f if line.strip())
    print(f"✓ Found {chunk_count} chunks in {CHUNKS_FILE}")

    print("\n[2/5] Initializing Qdrant client")
    indexer = QdrantIndexer(
        host="localhost",
        port=6333,
        collection_name="test_collection",
        in_memory=False,  # Set to True for memory only testing
    )
    print("✓ Qdrant client initialized")

    print("\n[3/5] Loading chunks and generating embeddings")
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 1000:
                break
            if line.strip():
                chunks.append(json.loads(line))

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

    print("\n[5/5] Verifying and testing search")
    for config_name in TEST_CONFIGS.keys():
        coll_name = f"test_{config_name}"
        verification = indexer.verify_index(collection_name=coll_name)
        print(f"\n{coll_name}:")
        print(f"  Points: {verification.get('points_count', 'N/A')}")
        print(f"  HNSW: m={verification.get('hnsw_config', {}).get('m', 'N/A')}, "
              f"ef={verification.get('hnsw_config', {}).get('ef_construct', 'N/A')}")

    print("SEARCH TEST")
    test_queries = [
        "Петр 1",
        "Зимний дворец",
        "Documentation best practices",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        for config_name in TEST_CONFIGS.keys():
            coll_name = f"test_{config_name}"
            results = indexer.search(
                query_text=query,
                limit=3,
                model_name=EMBEDDING_MODEL,
                collection_name=coll_name,
            )
            for i, r in enumerate(results, 1):
                print(f"  {i}. Score: {r['score']:.4f} | {r['payload'].get('text', '')[:100]}...")
