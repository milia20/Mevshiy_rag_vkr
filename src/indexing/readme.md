D:\P_work\Rag-VKR\.venv\Scripts\python.exe D:\P_work\Rag-VKR\src\indexing\qdrant_uploader.py

[1/5] Checking input file...
✓ Found 3346 chunks in D:\P_work\Rag-VKR\src\indexing\processed\chunks_en.jsonl

[2/5] Initializing Qdrant client...
2026-04-06 17:45:15,951 | INFO | Initializing Qdrant client: http://localhost:6333
✓ Qdrant client initialized

[3/5] Loading chunks and generating embeddings...
Loading: 3346it [00:00, 142096.30it/s]
✓ Loaded 3346 chunks
2026-04-06 17:45:16,177 | INFO | Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
2026-04-06 17:45:16,180 | INFO | Use pytorch device_name: cpu
2026-04-06 17:45:16,180 | INFO | Load pretrained SentenceTransformer: sentence-transformers/all-MiniLM-L6-v2
2026-04-06 17:45:16,413 | INFO | HTTP Request: GET http://localhost:6333 "HTTP/1.1 200 OK"
2026-04-06 17:45:16,689 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/modules.json "HTTP/1.1 307 Temporary
Redirect"
2026-04-06 17:45:16,731 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/modules.json "
HTTP/1.1 200 OK"
2026-04-06 17:45:16,882 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config_sentence_transformers.json "
HTTP/1.1 307 Temporary Redirect"
2026-04-06 17:45:16,923 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config_sentence_transformers.json "
HTTP/1.1 200 OK"
2026-04-06 17:45:17,074 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config_sentence_transformers.json "
HTTP/1.1 307 Temporary Redirect"
2026-04-06 17:45:17,116 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config_sentence_transformers.json "
HTTP/1.1 200 OK"
2026-04-06 17:45:17,263 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/README.md "HTTP/1.1 307 Temporary
Redirect"
2026-04-06 17:45:17,304 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/README.md "
HTTP/1.1 200 OK"
2026-04-06 17:45:17,454 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/modules.json "HTTP/1.1 307 Temporary
Redirect"
2026-04-06 17:45:17,496 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/modules.json "
HTTP/1.1 200 OK"
2026-04-06 17:45:17,707 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/sentence_bert_config.json "HTTP/1.1 307
Temporary Redirect"
2026-04-06 17:45:17,748 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/sentence_bert_config.json "
HTTP/1.1 200 OK"
2026-04-06 17:45:17,903 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/adapter_config.json "HTTP/1.1 404 Not
Found"
2026-04-06 17:45:18,053 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json "HTTP/1.1 307 Temporary
Redirect"
2026-04-06 17:45:18,095 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config.json "
HTTP/1.1 200 OK"
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 8749.99it/s]
2026-04-06 17:45:18,326 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json "HTTP/1.1 307 Temporary
Redirect"
2026-04-06 17:45:18,369 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config.json "
HTTP/1.1 200 OK"
2026-04-06 17:45:18,526 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer_config.json "HTTP/1.1 307
Temporary Redirect"
2026-04-06 17:45:18,567 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/tokenizer_config.json "
HTTP/1.1 200 OK"
2026-04-06 17:45:18,725 | INFO | HTTP Request:
GET https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2/tree/main/additional_chat_templates?recursive=false&expand=false "
HTTP/1.1 404 Not Found"
2026-04-06 17:45:18,877 | INFO | HTTP Request:
GET https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2/tree/main?recursive=true&expand=false "
HTTP/1.1 200 OK"
2026-04-06 17:45:19,061 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/1_Pooling/config.json "HTTP/1.1 307
Temporary Redirect"
2026-04-06 17:45:19,103 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/1_Pooling%2Fconfig.json "
HTTP/1.1 200 OK"
2026-04-06 17:45:19,254 | INFO | HTTP Request:
GET https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2 "HTTP/1.1 200 OK"
2026-04-06 17:45:19,258 | INFO | Generating embeddings for 3346 chunks...
Batches: 100%|██████████| 105/105 [00:34<00:00, 3.00it/s]
2026-04-06 17:45:54,285 | INFO | Generated embeddings with dimension: 384
✓ Generated embeddings (dimension: 384)

[4/5] Creating collections and indexing...
2026-04-06 17:45:54,338 | WARNING | Deleting existing collection: test_hnsw_default
2026-04-06 17:45:54,554 | INFO | Creating collection 'test_hnsw_default' with vector_size=384, distance=COSINE, HNSW(
m=16, ef_construct=100)
2026-04-06 17:45:57,689 | INFO | Collection 'test_hnsw_default' created successfully
2026-04-06 17:45:57,690 | INFO | Indexing 3346 chunks to 'test_hnsw_default' with batch_size=256
Uploading to test_hnsw_default: 100%|██████████| 14/14 [00:02<00:00, 4.72batch/s, uploaded=3346, failed=0]
2026-04-06 17:46:00,656 | INFO | Indexing complete for 'test_hnsw_default': 3346 uploaded, 0 failed
2026-04-06 17:46:00,656 | INFO | Creating payload index for field 'doc_title' with type 'keyword'
2026-04-06 17:46:05,888 | INFO | Payload index created for 'doc_title'
2026-04-06 17:46:05,888 | INFO | Creating payload index for field 'source' with type 'keyword'
2026-04-06 17:46:11,027 | INFO | Payload index created for 'source'
2026-04-06 17:46:11,027 | INFO | Creating payload index for field 'headers' with type 'keyword'
2026-04-06 17:46:16,051 | INFO | Payload index created for 'headers'
2026-04-06 17:46:16,051 | INFO | Creating payload index for field 'chunk_id' with type 'keyword'
2026-04-06 17:46:20,841 | INFO | Payload index created for 'chunk_id'
2026-04-06 17:46:20,842 | WARNING | Deleting existing collection: test_hnsw_optimized
✓ test_hnsw_default: 3346 uploaded, 0 failed
2026-04-06 17:46:21,038 | INFO | Creating collection 'test_hnsw_optimized' with vector_size=384, distance=COSINE, HNSW(
m=32, ef_construct=256)
2026-04-06 17:46:24,923 | INFO | Collection 'test_hnsw_optimized' created successfully
2026-04-06 17:46:24,923 | INFO | Indexing 3346 chunks to 'test_hnsw_optimized' with batch_size=256
Uploading to test_hnsw_optimized: 100%|██████████| 14/14 [00:03<00:00, 4.05batch/s, uploaded=3346, failed=0]
2026-04-06 17:46:28,383 | INFO | Indexing complete for 'test_hnsw_optimized': 3346 uploaded, 0 failed
2026-04-06 17:46:28,383 | INFO | Creating payload index for field 'doc_title' with type 'keyword'
2026-04-06 17:46:34,216 | INFO | Payload index created for 'doc_title'
2026-04-06 17:46:34,216 | INFO | Creating payload index for field 'source' with type 'keyword'
2026-04-06 17:46:39,260 | INFO | Payload index created for 'source'
2026-04-06 17:46:39,260 | INFO | Creating payload index for field 'headers' with type 'keyword'
2026-04-06 17:46:44,434 | INFO | Payload index created for 'headers'
2026-04-06 17:46:44,434 | INFO | Creating payload index for field 'chunk_id' with type 'keyword'
2026-04-06 17:46:49,616 | INFO | Payload index created for 'chunk_id'
2026-04-06 17:46:49,616 | INFO | Verifying index for collection: test_hnsw_default
2026-04-06 17:46:49,618 | INFO | Verification complete: 3346 points, HNSW(m=None, ef=None)
2026-04-06 17:46:49,618 | INFO | Verifying index for collection: test_hnsw_optimized
2026-04-06 17:46:49,619 | INFO | Verification complete: 3346 points, HNSW(m=None, ef=None)
2026-04-06 17:46:49,619 | INFO | Generating embedding for query: How to install Python?...
2026-04-06 17:46:49,622 | INFO | Use pytorch device_name: cpu
2026-04-06 17:46:49,622 | INFO | Load pretrained SentenceTransformer: sentence-transformers/all-MiniLM-L6-v2
✓ test_hnsw_optimized: 3346 uploaded, 0 failed

[5/5] Verifying and testing search...

test_hnsw_default:
Points: 3346
HNSW: m=None, ef=None

test_hnsw_optimized:
Points: 3346
HNSW: m=None, ef=None

----------------------------------------------------------------------
SEARCH TEST
----------------------------------------------------------------------

Query: How to install Python?
2026-04-06 17:46:50,150 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/modules.json "HTTP/1.1 307 Temporary
Redirect"
2026-04-06 17:46:50,185 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/modules.json "
HTTP/1.1 200 OK"
2026-04-06 17:46:50,345 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config_sentence_transformers.json "
HTTP/1.1 307 Temporary Redirect"
2026-04-06 17:46:50,380 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config_sentence_transformers.json "
HTTP/1.1 200 OK"
2026-04-06 17:46:50,543 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config_sentence_transformers.json "
HTTP/1.1 307 Temporary Redirect"
2026-04-06 17:46:50,578 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config_sentence_transformers.json "
HTTP/1.1 200 OK"
2026-04-06 17:46:50,738 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/README.md "HTTP/1.1 307 Temporary
Redirect"
2026-04-06 17:46:50,773 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/README.md "
HTTP/1.1 200 OK"
2026-04-06 17:46:50,935 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/modules.json "HTTP/1.1 307 Temporary
Redirect"
2026-04-06 17:46:50,967 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/modules.json "
HTTP/1.1 200 OK"
2026-04-06 17:46:51,126 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/sentence_bert_config.json "HTTP/1.1 307
Temporary Redirect"
2026-04-06 17:46:51,160 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/sentence_bert_config.json "
HTTP/1.1 200 OK"
2026-04-06 17:46:51,325 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/adapter_config.json "HTTP/1.1 404 Not
Found"
2026-04-06 17:46:51,488 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json "HTTP/1.1 307 Temporary
Redirect"
2026-04-06 17:46:51,525 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config.json "
HTTP/1.1 200 OK"
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 8027.90it/s]
2026-04-06 17:46:51,722 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json "HTTP/1.1 307 Temporary
Redirect"
2026-04-06 17:46:51,754 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config.json "
HTTP/1.1 200 OK"
2026-04-06 17:46:51,921 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer_config.json "HTTP/1.1 307
Temporary Redirect"
2026-04-06 17:46:51,955 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/tokenizer_config.json "
HTTP/1.1 200 OK"
2026-04-06 17:46:52,118 | INFO | HTTP Request:
GET https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2/tree/main/additional_chat_templates?recursive=false&expand=false "
HTTP/1.1 404 Not Found"
2026-04-06 17:46:52,284 | INFO | HTTP Request:
GET https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2/tree/main?recursive=true&expand=false "
HTTP/1.1 200 OK"
2026-04-06 17:46:52,479 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/1_Pooling/config.json "HTTP/1.1 307
Temporary Redirect"
2026-04-06 17:46:52,513 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/1_Pooling%2Fconfig.json "
HTTP/1.1 200 OK"
2026-04-06 17:46:52,686 | INFO | HTTP Request:
GET https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2 "HTTP/1.1 200 OK"
Batches: 100%|██████████| 1/1 [00:00<00:00, 112.85it/s]
2026-04-06 17:46:52,702 | INFO | Generating embedding for query: What is RAG?...
2026-04-06 17:46:52,706 | INFO | Use pytorch device_name: cpu
2026-04-06 17:46:52,706 | INFO | Load pretrained SentenceTransformer: sentence-transformers/all-MiniLM-L6-v2

1. Score: 0.5874 | * **Install Python** for you, including different versions

* Manage the **virtual environment** for ...
    2. Score: 0.5584 | Also, depending on your operating system (e.g. Linux, Windows, macOS), it could have come with
       Pytho...
    3. Score: 0.5550 | ## Run Your Program { #run-your-program }

After you activated the virtual environment, you can run ...

Query: What is RAG?
2026-04-06 17:46:52,864 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/modules.json "HTTP/1.1 307 Temporary
Redirect"
2026-04-06 17:46:52,907 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/modules.json "
HTTP/1.1 200 OK"
2026-04-06 17:46:53,075 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config_sentence_transformers.json "
HTTP/1.1 307 Temporary Redirect"
2026-04-06 17:46:53,111 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config_sentence_transformers.json "
HTTP/1.1 200 OK"
2026-04-06 17:46:53,280 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config_sentence_transformers.json "
HTTP/1.1 307 Temporary Redirect"
2026-04-06 17:46:53,324 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config_sentence_transformers.json "
HTTP/1.1 200 OK"
2026-04-06 17:46:53,510 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/README.md "HTTP/1.1 307 Temporary
Redirect"
2026-04-06 17:46:53,547 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/README.md "
HTTP/1.1 200 OK"
2026-04-06 17:46:53,720 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/modules.json "HTTP/1.1 307 Temporary
Redirect"
2026-04-06 17:46:53,760 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/modules.json "
HTTP/1.1 200 OK"
2026-04-06 17:46:53,933 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/sentence_bert_config.json "HTTP/1.1 307
Temporary Redirect"
2026-04-06 17:46:53,969 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/sentence_bert_config.json "
HTTP/1.1 200 OK"
2026-04-06 17:46:54,130 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/adapter_config.json "HTTP/1.1 404 Not
Found"
2026-04-06 17:46:54,290 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json "HTTP/1.1 307 Temporary
Redirect"
2026-04-06 17:46:54,323 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config.json "
HTTP/1.1 200 OK"
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 10037.02it/s]
2026-04-06 17:46:54,524 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json "HTTP/1.1 307 Temporary
Redirect"
2026-04-06 17:46:54,556 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config.json "
HTTP/1.1 200 OK"
2026-04-06 17:46:54,719 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer_config.json "HTTP/1.1 307
Temporary Redirect"
2026-04-06 17:46:54,754 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/tokenizer_config.json "
HTTP/1.1 200 OK"
2026-04-06 17:46:54,927 | INFO | HTTP Request:
GET https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2/tree/main/additional_chat_templates?recursive=false&expand=false "
HTTP/1.1 404 Not Found"
2026-04-06 17:46:55,120 | INFO | HTTP Request:
GET https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2/tree/main?recursive=true&expand=false "
HTTP/1.1 200 OK"
2026-04-06 17:46:55,316 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/1_Pooling/config.json "HTTP/1.1 307
Temporary Redirect"
2026-04-06 17:46:55,351 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/1_Pooling%2Fconfig.json "
HTTP/1.1 200 OK"
2026-04-06 17:46:55,513 | INFO | HTTP Request:
GET https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2 "HTTP/1.1 200 OK"
Batches: 100%|██████████| 1/1 [00:00<00:00, 103.36it/s]
2026-04-06 17:46:55,530 | INFO | Generating embedding for query: Documentation best practices...
2026-04-06 17:46:55,533 | INFO | Use pytorch device_name: cpu
2026-04-06 17:46:55,533 | INFO | Load pretrained SentenceTransformer: sentence-transformers/all-MiniLM-L6-v2

1. Score: 0.2486 | ### Tested { #tested }

* 100% test coverage.
* 100% type annotated code base.
* Used in production ...
    2. Score: 0.2450 | It means that **FastAPI** was specifically tested with the editors used by 80% of the Python
       develop...
    3. Score: 0.2364 | It is the "**path operation decorator**".

///

You can also use the other operations:

* `@app.post...

Query: Documentation best practices
2026-04-06 17:46:55,696 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/modules.json "HTTP/1.1 307 Temporary
Redirect"
2026-04-06 17:46:55,730 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/modules.json "
HTTP/1.1 200 OK"
2026-04-06 17:46:55,890 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config_sentence_transformers.json "
HTTP/1.1 307 Temporary Redirect"
2026-04-06 17:46:55,922 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config_sentence_transformers.json "
HTTP/1.1 200 OK"
2026-04-06 17:46:56,089 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config_sentence_transformers.json "
HTTP/1.1 307 Temporary Redirect"
2026-04-06 17:46:56,123 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config_sentence_transformers.json "
HTTP/1.1 200 OK"
2026-04-06 17:46:56,288 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/README.md "HTTP/1.1 307 Temporary
Redirect"
2026-04-06 17:46:56,327 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/README.md "
HTTP/1.1 200 OK"
2026-04-06 17:46:56,490 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/modules.json "HTTP/1.1 307 Temporary
Redirect"
2026-04-06 17:46:56,525 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/modules.json "
HTTP/1.1 200 OK"
2026-04-06 17:46:56,686 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/sentence_bert_config.json "HTTP/1.1 307
Temporary Redirect"
2026-04-06 17:46:56,721 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/sentence_bert_config.json "
HTTP/1.1 200 OK"
2026-04-06 17:46:56,889 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/adapter_config.json "HTTP/1.1 404 Not
Found"
2026-04-06 17:46:57,052 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json "HTTP/1.1 307 Temporary
Redirect"
2026-04-06 17:46:57,087 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config.json "
HTTP/1.1 200 OK"
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 8586.00it/s]
2026-04-06 17:46:57,283 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json "HTTP/1.1 307 Temporary
Redirect"
2026-04-06 17:46:57,317 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config.json "
HTTP/1.1 200 OK"
2026-04-06 17:46:57,491 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer_config.json "HTTP/1.1 307
Temporary Redirect"
2026-04-06 17:46:57,525 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/tokenizer_config.json "
HTTP/1.1 200 OK"
2026-04-06 17:46:57,693 | INFO | HTTP Request:
GET https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2/tree/main/additional_chat_templates?recursive=false&expand=false "
HTTP/1.1 404 Not Found"
2026-04-06 17:46:57,857 | INFO | HTTP Request:
GET https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2/tree/main?recursive=true&expand=false "
HTTP/1.1 200 OK"
2026-04-06 17:46:58,058 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/1_Pooling/config.json "HTTP/1.1 307
Temporary Redirect"
2026-04-06 17:46:58,092 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/1_Pooling%2Fconfig.json "
HTTP/1.1 200 OK"
2026-04-06 17:46:58,258 | INFO | HTTP Request:
GET https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2 "HTTP/1.1 200 OK"
Batches: 100%|██████████| 1/1 [00:00<00:00, 79.86it/s]
2026-04-06 17:46:58,277 | INFO | Qdrant client closed

1. Score: 0.6118 | * To propose new documentation sections.

* To fix an existing issue/bug.
    * Make sure to add test...

    2. Score: 0.5454 | * If the PR is for a feature, it should have docs.

    * Unless it's a feature we want to discourage...

    3. Score: 0.4650 | Each dictionary can contain:

* `name` (**required**): a `str` with the same tag name you use in the...
  ✓ Qdrant client closed

Process finished with exit code 0

D:\P_work\Rag-VKR\.venv\Scripts\python.exe D:\P_work\Rag-VKR\src\indexing\test_search.py
2026-04-06 17:48:55,276 | INFO | Initializing Qdrant client: http://localhost:6333
2026-04-06 17:48:55,533 | INFO | Generating embedding for query: app.websockets params...
2026-04-06 17:48:55,536 | INFO | Use pytorch device_name: cpu
2026-04-06 17:48:55,536 | INFO | Load pretrained SentenceTransformer: sentence-transformers/all-MiniLM-L6-v2

Поиск по вопросу: app.websockets params

2026-04-06 17:48:55,711 | INFO | HTTP Request: GET http://localhost:6333 "HTTP/1.1 200 OK"
2026-04-06 17:48:56,069 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/modules.json "HTTP/1.1 307 Temporary
Redirect"
2026-04-06 17:48:56,101 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/modules.json "
HTTP/1.1 200 OK"
2026-04-06 17:48:56,261 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config_sentence_transformers.json "
HTTP/1.1 307 Temporary Redirect"
2026-04-06 17:48:56,293 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config_sentence_transformers.json "
HTTP/1.1 200 OK"
2026-04-06 17:48:56,452 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config_sentence_transformers.json "
HTTP/1.1 307 Temporary Redirect"
2026-04-06 17:48:56,484 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config_sentence_transformers.json "
HTTP/1.1 200 OK"
2026-04-06 17:48:56,644 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/README.md "HTTP/1.1 307 Temporary
Redirect"
2026-04-06 17:48:56,677 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/README.md "
HTTP/1.1 200 OK"
2026-04-06 17:48:56,843 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/modules.json "HTTP/1.1 307 Temporary
Redirect"
2026-04-06 17:48:56,878 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/modules.json "
HTTP/1.1 200 OK"
2026-04-06 17:48:57,038 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/sentence_bert_config.json "HTTP/1.1 307
Temporary Redirect"
2026-04-06 17:48:57,071 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/sentence_bert_config.json "
HTTP/1.1 200 OK"
2026-04-06 17:48:57,230 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/adapter_config.json "HTTP/1.1 404 Not
Found"
2026-04-06 17:48:57,393 | INFO | HTTP Request:
HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json "HTTP/1.1 307 Temporary
Redirect"
2026-04-06 17:48:57,426 | INFO | HTTP Request:
HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config.json "
HTTP/1.1 200 OK"
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 8596.08it/s]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key | Status | |
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED | |

Notes:

- UNEXPECTED:    can be ignored when loading from different task/architecture; not ok if you expect identical arch.
  2026-04-06 17:48:57,673 | INFO | HTTP Request:
  HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json "HTTP/1.1 307 Temporary
  Redirect"
  2026-04-06 17:48:57,708 | INFO | HTTP Request:
  HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config.json "
  HTTP/1.1 200 OK"
  2026-04-06 17:48:57,873 | INFO | HTTP Request:
  HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer_config.json "HTTP/1.1 307
  Temporary Redirect"
  2026-04-06 17:48:57,906 | INFO | HTTP Request:
  HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/tokenizer_config.json "
  HTTP/1.1 200 OK"
  2026-04-06 17:48:58,073 | INFO | HTTP Request:
  GET https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2/tree/main/additional_chat_templates?recursive=false&expand=false "
  HTTP/1.1 404 Not Found"
  2026-04-06 17:48:58,238 | INFO | HTTP Request:
  GET https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2/tree/main?recursive=true&expand=false "
  HTTP/1.1 200 OK"
  2026-04-06 17:48:58,433 | INFO | HTTP Request:
  HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/1_Pooling/config.json "HTTP/1.1 307
  Temporary Redirect"
  2026-04-06 17:48:58,465 | INFO | HTTP Request:
  HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/1_Pooling%2Fconfig.json "
  HTTP/1.1 200 OK"
  2026-04-06 17:48:58,629 | INFO | HTTP Request:
  GET https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2 "HTTP/1.1 200 OK"
  Batches: 100%|██████████| 1/1 [00:00<00:00, 66.13it/s]
  2026-04-06 17:48:58,655 | INFO | Qdrant client closed
  Найдено 3 фрагментов:

--- Результат 1 (score = 0.6599) ---
Текст: ## WebSockets client { #websockets-client }
Источник: WebSockets { #websockets } (advanced\websockets.md)

--- Результат 2 (score = 0.6510) ---
Текст: ///

With that you can connect the WebSocket and then send and receive messages:
Источник: WebSockets { #websockets } (advanced\websockets.md)

--- Результат 3 (score = 0.6255) ---
Текст: /// tip

When you want to define dependencies that should be compatible with both HTTP and WebSockets, you can define a parameter
that takes an `HTTPConnection` instead of a `Request` or a `WebSocket`.

///
Источник: WebSockets (reference\websockets.md)

Process finished with exit code 0
