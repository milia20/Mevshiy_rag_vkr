# Использование MkDocs RAG Plugin

## Веб-интерфейс

Плагин предоставляет минималистичный веб-интерфейс для взаимодействия с RAG системой.

### Запуск frontend

```bash
# Вариант 1: Простой HTTP сервер
cd frontend
python -m http.server 3000

# Вариант 2: Nginx (в Docker)
docker-compose up -d frontend
```

Откройте браузер: http://localhost:3000

### Элементы интерфейса

1. **Поле вопроса** — введите ваш вопрос по документации
2. **Режим поиска** — выберите стратегию:
    - Гибридный (Dense + BM25) — рекомендуется
    - Только Dense — семантический поиск
    - Только Sparse — поиск по ключевым словам
3. **Количество источников** — сколько документов показать (1-20)
4. **Потоковый ответ** — показывать ответ по мере генерации

### Пример запроса

**Вопрос:** "Как настроить аутентификацию в приложении?"

**Ответ:**

```
Для настройки аутентификации выполните следующие шаги:

1. Установите пакет аутентификации:
   pip install auth-package

2. Создайте файл конфигурации config.yaml:
   ```yaml
   auth:
     provider: jwt
     secret_key: your-secret-key
     token_expiry: 3600
   ```

3. Инициализируйте middleware в приложении...

```

**Источники:**
- `docs/auth/setup.md` — Релевантность: 95%
- `docs/auth/config.md` — Релевантность: 87%
- `docs/security/best-practices.md` — Релевантность: 72%

## API Endpoints

### POST /api/v1/query

Основной endpoint для получения ответов на вопросы.

**Запрос:**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Как установить плагин?",
    "top_k": 5,
    "mode": "hybrid",
    "stream": false
  }'
```

**Ответ:**

```json
{
  "answer": "Для установки плагина используйте команду pip install...",
  "sources": [
    {
      "url": "/installation/",
      "title": "Установка",
      "snippet": "Для установки выполните pip install mkdocs-rag-plugin",
      "score": 0.95
    }
  ],
  "metadata": {
    "retrieval_time_ms": 45,
    "generation_time_ms": 1250,
    "model": "qwen-2.5-7b"
  }
}
```

### POST /api/v1/query (Streaming)

**Запрос с потоковым ответом:**

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Как настроить конфигурацию?",
    "stream": true
  }' \
  --no-buffer
```

**Формат SSE ответа:**

```
data: {"answer": "Для настройки "}
data: {"answer": "Для настройки конфигурации "}
data: {"answer": "Для настройки конфигурации используйте "}
...
data: {"sources": [...], "metadata": {...}}
data: [DONE]
```

### POST /api/v1/index

Запустить переиндексацию документов.

```bash
curl -X POST http://localhost:8000/api/v1/index
```

**Ответ:**

```json
{
  "status": "started",
  "message": "Индексация запущена",
  "total_docs": 15,
  "total_chunks": 127
}
```

### GET /api/v1/index/status

Получить статус индексации.

```bash
curl http://localhost:8000/api/v1/index/status
```

**Ответ:**

```json
{
  "status": "completed",
  "progress": 100,
  "indexed_count": 127,
  "errors": []
}
```

### DELETE /api/v1/index

Очистить индекс.

```bash
curl -X DELETE http://localhost:8000/api/v1/index
```

### GET /api/v1/models

Получить список доступных моделей.

```bash
curl http://localhost:8000/api/v1/models
```

**Ответ:**

```json
{
  "embedding_models": [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-large"
  ],
  "llm_models": [
    "qwen-2.5-7b",
    "llama-3-8b"
  ]
}
```

### GET /api/v1/models/stats

Статистика коллекции.

```bash
curl http://localhost:8000/api/v1/models/stats
```

**Ответ:**

```json
{
  "collection_name": "docs_index",
  "documents_count": 15,
  "chunks_count": 127,
  "vector_size": 384,
  "memory_usage_mb": 245
}
```

### GET /health

Проверка здоровья сервиса.

```bash
curl http://localhost:8000/health
```

**Ответ:**

```json
{
  "status": "healthy",
  "qdrant": "connected",
  "embedding_service": "ready",
  "llm_client": "ready",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Python SDK

### Базовое использование

```python
import requests

class RAGClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def query(self, question: str, top_k: int = 5, mode: str = "hybrid") -> dict:
        response = requests.post(
            f"{self.base_url}/api/v1/query",
            json={"question": question, "top_k": top_k, "mode": mode}
        )
        return response.json()

    def index(self) -> dict:
        response = requests.post(f"{self.base_url}/api/v1/index}")
        return response.json()

    def health(self) -> dict:
        response = requests.get(f"{self.base_url}/health")
        return response.json()

# Использование
client = RAGClient()

# Проверка здоровья
print(client.health())

# Запрос
result = client.query("Как настроить аутентификацию?")
print(f"Ответ: {result['answer']}")
print(f"Источники: {len(result['sources'])}")
```

### Асинхронное использование с streaming

```python
import aiohttp
import asyncio

async def stream_query(question: str):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/api/v1/query",
            json={"question": question, "stream": True}
        ) as response:
            async for line in response.content:
                if line.startswith(b"data: "):
                    data = line[6:].decode()
                    if data == "[DONE]":
                        break
                    print(data, end="", flush=True)

asyncio.run(stream_query("Как использовать плагин?"))
```

## Оценка качества

### Запуск оценки

```bash
# Создание тестового датасета
cat > test_queries.json << 'EOF'
[
  {
    "question": "Как установить плагин?",
    "expected_answer": "pip install mkdocs-rag-plugin",
    "relevant_docs": ["installation.md"]
  },
  {
    "question": "Какие режимы поиска доступны?",
    "expected_answer": "dense, sparse, hybrid",
    "relevant_docs": ["configuration.md"]
  }
]
EOF

# Запуск оценки
python scripts/evaluate.py \
  --dataset test_queries.json \
  --mode hybrid \
  --top-k 5 \
  --format both
```

### Интерпретация метрик

| Метрика      | Диапазон | Описание                 | Хорошо |
|--------------|----------|--------------------------|--------|
| NDCG@k       | 0-1      | Качество ранжирования    | > 0.7  |
| MRR          | 0-1      | Средняя обратная позиция | > 0.6  |
| Hit@k        | 0-1      | Попадание релевантного   | > 0.8  |
| Faithfulness | 0-1      | Соответствие контексту   | > 0.8  |

## Интеграция с MkDocs

### Автоматическая индексация при сборке

Плагин автоматически индексирует документы при выполнении `mkdocs build`:

```bash
mkdocs build
# Документы автоматически отправляются в Qdrant
```

### Принудительная переиндексация

```bash
# Через API
curl -X POST http://localhost:8000/api/v1/index

# Или через плагин (в коде)
from mkdocs_plugin.plugin import RAGPlugin
plugin = RAGPlugin()
plugin.reindex()
```

## Лучшие практики

### 1. Оптимизация чанкования

Для технической документации:

```env
CHUNK_MAX_TOKENS=500
CHUNK_OVERLAP=50
CHUNK_STRATEGY=structural
```

Для длинных документов:

```env
CHUNK_MAX_TOKENS=768
CHUNK_OVERLAP=100
CHUNK_STRATEGY=semantic
```

### 2. Настройка гибридного поиска

Для точных запросов (номера ошибок, команды):

```env
HYBRID_DENSE_WEIGHT=0.5
HYBRID_SPARSE_WEIGHT=0.5
```

Для концептуальных вопросов:

```env
HYBRID_DENSE_WEIGHT=0.8
HYBRID_SPARSE_WEIGHT=0.2
```

### 3. Кэширование частых запросов

Добавьте кэширование на уровне приложения:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_query(question_hash: str) -> dict:
    # ...
```

## Следующие шаги

- Изучите [архитектуру](../README.md#архитектура) системы
- Настройте [оценку качества](../scripts/evaluate.py)
- Внесите вклад в [разработку](../CONTRIBUTING.md)
