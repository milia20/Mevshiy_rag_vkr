D:\P_work\Rag-VKR\.venv\Scripts\python.exe D:\P_work\Rag-VKR\src\custom_dataset\test_ru.py

[1/5] Checking input file
✓ Found 80705 chunks in D:\P_work\Rag-VKR\src\custom_dataset\ru_rag\chunks.jsonl

[2/5] Initializing Qdrant client
✓ Qdrant client initialized

[3/5] Loading chunks and generating embeddings
✓ Loaded 1000 chunks
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 8188.27it/s]
Batches: 100%|██████████| 32/32 [00:14<00:00, 2.20it/s]
✓ Generated embeddings (dimension: 384)

[4/5] Creating collections and indexing...
Deleting existing collection: test_hnsw_default
Uploading to test_hnsw_default: 100%|██████████| 4/4 [00:01<00:00, 3.36batch/s, uploaded=1000, failed=0]
✓ test_hnsw_default: 1000 uploaded, 0 failed
Deleting existing collection: test_hnsw_optimized
Uploading to test_hnsw_optimized: 100%|██████████| 4/4 [00:01<00:00, 2.97batch/s, uploaded=1000, failed=0]
✓ test_hnsw_optimized: 1000 uploaded, 0 failed

[5/5] Verifying and testing search

test_hnsw_default:
Points: 1000
HNSW: m=None, ef=None

test_hnsw_optimized:
Points: 1000
HNSW: m=None, ef=None
SEARCH TEST

Query: Петр 1
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 7632.47it/s]

1. Score: 0.6099 | == Персонажи ==...
2. Score: 0.6099 | == Персонажи ==...
3. Score: 0.5908 | === «Гарри Поттер и Дары Смерти» ===...
   Loading weights: 100%|██████████| 103/103 [00:00<00:00, 8443.53it/s]
1. Score: 0.6099 | == Персонажи ==...
2. Score: 0.6099 | == Персонажи ==...
3. Score: 0.5908 | === «Гарри Поттер и Дары Смерти» ===...

Query: Зимний дворец
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 8672.88it/s]

1. Score: 0.8620 | ==== Третий Зимний дворец — дворец Анны Иоанновны ====...
2. Score: 0.7792 | ==== Зимний дворец при Екатерине II ====...
3. Score: 0.7688 | Зимний дворец и Дворцовая площадь образуют архитектурный ансамбль, ставший одним из главных
   объектов...
   Loading weights: 100%|██████████| 103/103 [00:00<00:00, 8649.61it/s]
1. Score: 0.8620 | ==== Третий Зимний дворец — дворец Анны Иоанновны ====...
2. Score: 0.7792 | ==== Зимний дворец при Екатерине II ====...
3. Score: 0.7688 | Зимний дворец и Дворцовая площадь образуют архитектурный ансамбль, ставший одним из главных
   объектов...

Query: Documentation best practices
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 7954.58it/s]

1. Score: 0.0818 | == Примечания ==

== Ссылки ==
Mammoth Cave National Park website (англ.). www.nps.gov. Дата обращен...

2. Score: 0.0801 | CyArk Digital Nineveh Archives, общедоступное бесплатное хранилище данных из ранее связанного
   проект...
3. Score: 0.0691 | в полуфинал Кафельникова переиграл победитель того розыгрыша турнира Томас Мустер....
   Loading weights: 100%|██████████| 103/103 [00:00<00:00, 8650.30it/s]
1. Score: 0.0818 | == Примечания ==

== Ссылки ==
Mammoth Cave National Park website (англ.). www.nps.gov. Дата обращен...

2. Score: 0.0801 | CyArk Digital Nineveh Archives, общедоступное бесплатное хранилище данных из ранее связанного
   проект...
3. Score: 0.0691 | в полуфинал Кафельникова переиграл победитель того розыгрыша турнира Томас Мустер....

Process finished with exit code 0
