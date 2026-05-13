# Qdrant Indexing Pipeline

# Default configuration

python qdrant_indexer.py --input chunks_en.jsonl --collection docs_default

# Optimized HNSW

python qdrant_indexer.py --input chunks_en.jsonl --config optimized --collection docs_optimized

# With payload indexes and verification

python qdrant_indexer.py --input chunks_en.jsonl --create-indexes --verify --output-stats results.json

# In-memory testing

python qdrant_indexer.py --input chunks_en.jsonl --in-memory --collection test_docs

# Run all experiments

python experiment_config.py --input chunks_en.jsonl
