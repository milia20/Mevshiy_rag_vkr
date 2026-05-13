# Конфигурация MkDocs RAG Plugin

## Файл конфигурации .env

Все настройки задаются через переменные окружения в файле `.env`.

### Основные параметры

| Переменная          | По умолчанию          | Описание                      |
|---------------------|-----------------------|-------------------------------|
| `QDRANT_URL`        | http://localhost:6333 | URL подключения к Qdrant      |
| `QDRANT_COLLECTION` | docs_index            | Имя коллекции в Qdrant        |
| `QDRANT_API_KEY`    | -                     | API ключ для облачного Qdrant |

### Настройки эмбеддингов

| Переменная        | По умолчанию                                                | Описание               |
|-------------------|-------------------------------------------------------------|------------------------|
| `EMBEDDING_MODEL` | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Модель для эмбеддингов |

**Поддерживаемые модели для русского языка:**

- `GigaEmbeddings` — модель от GigaChat (требуется API ключ)
- `BGE-M3` — multilingual модель от BAAI
- `mE5-large` — multilingual модель от Microsoft
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` — легкая multilingual модель

### Настройки LLM

| Переменная       | По умолчанию             | Описание                                 |
|------------------|--------------------------|------------------------------------------|
| `LLM_PROVIDER`   | lmstudio                 | Провайдер LLM (lmstudio, ollama, openai) |
| `LLM_MODEL_NAME` | qwen-2.5-7b              | Название модели                          |
| `LLM_BASE_URL`   | http://localhost:1234/v1 | URL API провайдера                       |
| `LLM_TIMEOUT`    | 60                       | Таймаут запроса в секундах               |

**Примеры настроек для разных провайдеров:**

#### LM Studio

```env
LLM_PROVIDER=lmstudio
LLM_MODEL_NAME=qwen-2.5-7b
LLM_BASE_URL=http://localhost:1234/v1
```

#### Ollama

```env
LLM_PROVIDER=ollama
LLM_MODEL_NAME=qwen2.5:7b
LLM_BASE_URL=http://localhost:11434
```

#### OpenAI-compatible API

```env
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-4
LLM_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
```

### Настройки чанкования

| Переменная         | По умолчанию | Описание                                |
|--------------------|--------------|-----------------------------------------|
| `CHUNK_MAX_TOKENS` | 500          | Максимальное количество токенов в чанке |
| `CHUNK_OVERLAP`    | 50           | Перекрытие между чанками                |
| `CHUNK_STRATEGY`   | structural   | Стратегия чанкования                    |

**Стратегии чанкования:**

- `structural` — разделение по заголовкам Markdown
- `semantic` — разделение по смысловым границам
- `recursive` — рекурсивное разделение по символам

### Настройки гибридного поиска

| Переменная             | По умолчанию | Описание                     |
|------------------------|--------------|------------------------------|
| `HYBRID_DENSE_WEIGHT`  | 0.7          | Вес для dense поиска         |
| `HYBRID_SPARSE_WEIGHT` | 0.3          | Вес для sparse (BM25) поиска |

**Режимы поиска:**

- `dense` — только семантический поиск (эмбеддинги)
- `sparse` — только BM25 (ключевые слова)
- `hybrid` — комбинация обоих методов

### Настройки API сервера

| Переменная     | По умолчанию              | Описание                |
|----------------|---------------------------|-------------------------|
| `API_HOST`     | 0.0.0.0                   | Хост API сервера        |
| `API_PORT`     | 8000                      | Порт API сервера        |
| `DEBUG`        | false                     | Режим отладки           |
| `CORS_ORIGINS` | ["http://localhost:3000"] | Разрешенные CORS origin |

### Продвинутые настройки

| Переменная        | По умолчанию | Описание                     |
|-------------------|--------------|------------------------------|
| `MAX_RETRIES`     | 3            | Количество повторных попыток |
| `REQUEST_TIMEOUT` | 30           | Общий таймаут запросов       |
| `LOG_LEVEL`       | INFO         | Уровень логирования          |

## Конфигурация в mkdocs.yml

```yaml
plugins:
  - search:
      enabled: false
  - rag_plugin:
      # Основные настройки
      enabled: true
      qdrant_url: http://localhost:6333
      collection_name: docs_index

      # Эмбеддинги
      embedding_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

      # LLM
      llm_provider: lmstudio
      llm_model: qwen-2.5-7b

      # API
      api_host: localhost
      api_port: 8000

      # Чанкование
      chunk_size: 500
      chunk_overlap: 50
      chunk_strategy: structural

      # UI
      enable_chat_panel: true
      enable_search_ui: true
```

## Примеры конфигураций

### Минимальная конфигурация

```env
QDRANT_URL=http://localhost:6333
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
LLM_BASE_URL=http://localhost:1234/v1
LLM_MODEL_NAME=qwen-2.5-7b
```

### Production конфигурация

```env
# Qdrant Cloud
QDRANT_URL=https://your-cluster.qdrant.tech
QDRANT_API_KEY=your-api-key
QDRANT_COLLECTION=production_docs

# Embeddings
EMBEDDING_MODEL=BAAI/bge-m3

# LLM (Ollama с GPU)
LLM_PROVIDER=ollama
LLM_MODEL_NAME=qwen2.5:14b
LLM_BASE_URL=http://ollama:11434
LLM_TIMEOUT=120

# Chunking
CHUNK_MAX_TOKENS=768
CHUNK_OVERLAP=100
CHUNK_STRATEGY=semantic

# Hybrid search
HYBRID_DENSE_WEIGHT=0.8
HYBRID_SPARSE_WEIGHT=0.2

# API
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=["https://your-domain.com"]

# Advanced
MAX_RETRIES=5
REQUEST_TIMEOUT=60
LOG_LEVEL=WARNING
```

### Конфигурация для тестирования

```env
# Локальный Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=test_index

# Быстрая модель эмбеддингов
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Локальная LLM
LLM_PROVIDER=ollama
LLM_MODEL_NAME=qwen2.5:1.5b
LLM_BASE_URL=http://localhost:11434

# Маленькие чанки для тестов
CHUNK_MAX_TOKENS=256
CHUNK_OVERLAP=25

# Debug режим
DEBUG=true
LOG_LEVEL=DEBUG
```

## Валидация конфигурации

Проверьте корректность конфигурации:

```bash
python -c "from src.core.config import settings; print(settings.model_dump_json(indent=2))"
```

Или через API:

```bash
curl http://localhost:8000/api/v1/models/stats
```

## Переопределение настроек

Настройки можно переопределять через переменные окружения:

```bash
export QDRANT_URL=http://custom-qdrant:6333
export LLM_MODEL_NAME=llama-3-8b
uvicorn src.main:app --host 0.0.0.0 --port 8000
```
