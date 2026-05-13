# MkDocs RAG Plugin

**MkDocs RAG Plugin** — плагин для интеграции RAG (Retrieval-Augmented Generation) системы с MkDocs документацией.
Обеспечивает семантический поиск по документации с генерацией ответов на основе языковых моделей.

## 📋 Оглавление

- [Возможности](#возможности)
- [Архитектура](#архитектура)
- [Быстрый старт](#быстрый-старт)
- [Установка](#установка)
- [Конфигурация](#конфигурация)
- [Использование](#использование)
- [Оценка качества](#оценка-качества)
- [API Reference](#api-reference)
- [Разработка и тесты](#разработка-и-тесты)
- [Troubleshooting](#troubleshooting)

## ✨ Возможности

| Функция                | Описание                                                                               |
|------------------------|----------------------------------------------------------------------------------------|
| 🔍 **Гибридный поиск** | Комбинация dense (семантического) и sparse (BM25) поиска для максимальной точности     |
| 🤖 **LLM интеграция**  | Поддержка LM Studio, Ollama, OpenAI-compatible API                                     |
| 📚 **Источники**       | Каждый ответ сопровождается ссылками на оригинальные документы с оценкой релевантности |
| 🇷🇺 **Русский язык**  | Полная поддержка русского языка во всех компонентах                                    |
| ⚡ **Streaming**        | Потоковая передача токенов ответа в реальном времени (SSE)                             |
| 🐳 **Docker**          | Готовая конфигурация для развертывания через Docker Compose                            |
| 📊 **Метрики**         | Встроенные метрики оценки качества: NDCG, MRR, Hit@k, Faithfulness                     |
| 🔧 **Гибкость**        | Настройка параметров чанкования, поиска и генерации                                    |

## 🏗️ Архитектура

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   MkDocs        │────▶│  FastAPI         │────▶│   Qdrant        │
│   Plugin        │     │  Backend         │     │   Vector DB     │
│   (indexing)    │     │  (orchestration) │     │   (storage)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                         ┌──────────────────┐
                         │  LLM Client      │
                         │  (Ollama/LM      │
                         │   Studio)        │
                         └──────────────────┘
                               ▲
                               │
                         ┌──────────────────┐
                         │  Frontend        │
                         │  (Vanilla JS)    │
                         └──────────────────┘
```

**Компоненты:**

1. **MkDocs Plugin** — обрабатывает страницы документации, создает чанки
2. **FastAPI Backend** — оркестрирует поиск, генерацию, управление индексом
3. **Qdrant** — хранит векторные эмбеддинги с поддержкой гибридного поиска
4. **LLM Client** — генерирует ответы на основе контекста
5. **Frontend** — веб-интерфейс для взаимодействия с системой

## 🚀 Быстрый старт

### 1. Установка

```bash
cd mkdocs_rag_plugin
pip install -e ".[dev]"
```

### 2. Настройка окружения

```bash
cp .env.example .env
# Отредактируйте .env при необходимости
```

### 3. Запуск сервисов (Docker)

```bash
cd docker
docker-compose up -d
```

### 4. Проверка

```bash
# Health check
curl http://localhost:8000/health

# Тестовый запрос
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Как установить плагин?", "top_k": 3}'
```

### 5. Веб-интерфейс

Откройте `frontend/index.html` в браузере или запустите:

```bash
cd frontend
python -m http.server 3000
# http://localhost:3000
```

## 📦 Установка

### Вариант A: pip installation

```bash
pip install mkdocs-rag-plugin
```

### Вариант B: Docker Compose

```bash
cd docker
docker-compose up -d
```

### Требования

- Python 3.13+
- Docker и Docker Compose (рекомендуется)
- LLM сервер (LM Studio или Ollama)

## ⚙️ Конфигурация

### Файл .env

Основные переменные окружения:

```env
# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=docs_index

# Embeddings
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# LLM
LLM_PROVIDER=lmstudio
LLM_MODEL_NAME=qwen-2.5-7b
LLM_BASE_URL=http://localhost:1234/v1

# Chunking
CHUNK_MAX_TOKENS=500
CHUNK_OVERLAP=50

# Hybrid search
HYBRID_DENSE_WEIGHT=0.7
HYBRID_SPARSE_WEIGHT=0.3

# API
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=["http://localhost:3000"]
```

Полный список параметров см. в `.env.example`.

### Конфигурация в mkdocs.yml

```yaml
plugins:
    -   search:
            enabled: false
    -   rag_plugin:
            qdrant_url: http://localhost:6333
            collection_name: docs_index
            embedding_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
            llm_provider: lmstudio
            llm_model: qwen-2.5-7b
            api_host: localhost
            api_port: 8000
            enable_chat_panel: true
```

## 💻 Использование

### Через веб-интерфейс

1. Откройте `frontend/index.html`
2. Введите вопрос
3. Выберите режим поиска (hybrid/dense/sparse)
4. Получите ответ с источниками

### Через API

```bash
# Query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Как настроить аутентификацию?",
    "top_k": 5,
    "mode": "hybrid",
    "stream": false
  }'

# Index
curl -X POST http://localhost:8000/api/v1/index

# Health
curl http://localhost:8000/health
```

### Через Python

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={"question": "Как установить?", "top_k": 5}
)
data = response.json()
print(data["answer"])
```

## 📊 Оценка качества

### Запуск оценки

```bash
# Создание тестового датасета
cat > test_queries.json << 'EOF'
[
  {
    "question": "Как установить плагин?",
    "expected_answer": "pip install mkdocs-rag-plugin",
    "relevant_docs": ["installation.md"]
  }
]
EOF

# Запуск скрипта оценки
python scripts/evaluate.py \
  --dataset test_queries.json \
  --mode hybrid \
  --top-k 5 \
  --format both
```

### Метрики

| Метрика          | Описание                                      | Хорошо |
|------------------|-----------------------------------------------|--------|
| **NDCG@k**       | Качество ранжирования                         | > 0.7  |
| **MRR**          | Средняя обратная позиция первого релевантного | > 0.6  |
| **Hit@k**        | Доля запросов с релевантным в top-k           | > 0.8  |
| **Faithfulness** | Соответствие ответа контексту                 | > 0.8  |

### Интерпретация результатов

Скрипт генерирует отчеты в форматах JSON и Markdown с детальными результатами по каждому запросу.

## 📡 API Reference

### Endpoints

| Метод    | Endpoint               | Описание             |
|----------|------------------------|----------------------|
| `GET`    | `/`                    | Информация об API    |
| `GET`    | `/health`              | Проверка здоровья    |
| `POST`   | `/api/v1/query`        | RAG запрос           |
| `POST`   | `/api/v1/index`        | Запустить индексацию |
| `GET`    | `/api/v1/index/status` | Статус индексации    |
| `DELETE` | `/api/v1/index`        | Очистить индекс      |
| `GET`    | `/api/v1/models`       | Список моделей       |
| `GET`    | `/api/v1/models/stats` | Статистика коллекции |

### Формат запроса query

```json
{
    "question": "string",
    "top_k": 5,
    "mode": "hybrid",
    "stream": true
}
```

### Формат ответа query

```json
{
    "answer": "string",
    "sources": [
        {
            "url": "string",
            "title": "string",
            "snippet": "string",
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

Полная документация API доступна по адресу `/docs` после запуска сервера.

## 🧪 Разработка и тесты

### Установка dev зависимостей

```bash
pip install -e ".[dev]"
```

### Запуск тестов

```bash
# Все тесты
pytest tests/ -v

# С покрытием
pytest --cov=src --cov-report=html

# Только интеграционные
pytest tests/test_integration.py -v

# E2E тесты frontend
playwright install
pytest tests/test_frontend.py -v
```

### Линтинг и форматирование

```bash
# Форматирование
black mkdocs_plugin/ src/ tests/

# Линтинг
ruff check mkdocs_plugin/ src/ tests/

# Типы
mypy mkdocs_plugin/ src/ tests/
```

### LLM timeout

1. Убедитесь, что LLM сервер запущен
2. Проверьте настройки в `.env`:
   ```env
   LLM_BASE_URL=http://localhost:1234/v1
   LLM_TIMEOUT=60
   ```
3. Для больших моделей увеличьте таймаут

### Ошибки при индексации

```bash
#Verbose логирование
mkdocs build --verbose

# Проверка плагина
python -c "from mkdocs_plugin.plugin import RAGPlugin; print(RAGPlugin())"
```

### Ошибка CORS

Добавьте origin в `CORS_ORIGINS`:

```env
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]
```

### CUDA out of memory

Для GPU используйте меньшую модель или увеличьте память в `docker-compose.yml`.

## 📁 Список файлов проекта

### Основные файлы плагина

| Файл                                      | Назначение             |
|-------------------------------------------|------------------------|
| `mkdocs_plugin/__init__.py`               | Инициализация пакета   |
| `mkdocs_plugin/plugin.py`                 | Основной класс плагина |
| `mkdocs_plugin/config.py`                 | Схема конфигурации     |
| `mkdocs_plugin/models.py`                 | Модели данных          |
| `mkdocs_plugin/services/doc_processor.py` | Обработка страниц      |
| `mkdocs_plugin/services/chunker.py`       | Чанкование текста      |

### Backend файлы (не используются плагином напрямую)

| Файл                                | Назначение                  |
|-------------------------------------|-----------------------------|
| `src/main.py`                       | FastAPI приложение          |
| `src/api/query.py`                  | Query endpoint              |
| `src/api/index.py`                  | Index endpoints             |
| `src/api/models.py`                 | Models endpoints            |
| `src/api/health.py`                 | Health endpoint             |
| `src/core/config.py`                | Настройки приложения        |
| `src/services/qdrant_client.py`     | Qdrant сервис               |
| `src/services/indexer.py`           | Индексация документов       |
| `src/services/retriever.py`         | Поиск (dense/sparse/hybrid) |
| `src/services/search_pipeline.py`   | Поисковый пайплайн          |
| `src/services/llm_client.py`        | LLM клиент                  |
| `src/services/embedding_service.py` | Эмбеддинги                  |
| `src/services/evaluation.py`        | Метрики оценки              |

### Frontend файлы

| Файл                  | Назначение      |
|-----------------------|-----------------|
| `frontend/index.html` | Веб-интерфейс   |
| `frontend/styles.css` | Стили           |
| `frontend/app.js`     | Логика frontend |

### DevOps файлы

| Файл                        | Назначение       |
|-----------------------------|------------------|
| `docker/Dockerfile`         | Docker образ     |
| `docker/docker-compose.yml` | Оркестрация      |
| `.env.example`              | Шаблон окружения |
| `scripts/install.sh`        | Скрипт установки |
| `scripts/evaluate.py`       | Оценка качества  |

### Тесты

| Файл                        | Назначение           |
|-----------------------------|----------------------|
| `tests/conftest.py`         | Фикстуры pytest      |
| `tests/test_api.py`         | API тесты            |
| `tests/test_retrieval.py`   | Тесты поиска         |
| `tests/test_integration.py` | Интеграционные тесты |
| `tests/test_frontend.py`    | E2E тесты            |

### Примеры

| Файл                            | Назначение          |
|---------------------------------|---------------------|
| `examples/test-site/mkdocs.yml` | Пример конфигурации |
| `examples/test-site/docs/*.md`  | Пример документации |
| `examples/test-site/README.md`  | Инструкция          |

### Файлы, НЕ участвующие в работе плагина

Следующие файлы находятся в директории проекта, но не используются плагином напрямую:

| Файл/Директория              | Назначение                                                        |
|------------------------------|-------------------------------------------------------------------|
| `src/` (вся директория)      | Backend для standalone режима, плагин использует только через API |
| `frontend/` (вся директория) | Отдельный веб-интерфейс, не требуется для работы плагина в MkDocs |
| `scripts/`                   | Утилиты для разработки и оценки                                   |
| `tests/`                     | Тесты для разработки                                              |
| `docker/`                    | Конфигурация для deployment                                       |
| `examples/`                  | Примеры использования                                             |
| `notebooks/`                 | Jupyter ноутбуки для экспериментов                                |
| `experiments/`               | Экспериментальный код                                             |
| `research/`                  | Исследовательские материалы                                       |
| `docs/` (в корне)            | Дополнительная документация                                       |
| `*.md` (корневые)            | Документация проекта (plan.md, research.md, etc.)                 |
| `pyproject.toml` (корневой)  | Конфигурация всего проекта, плагин использует свой                |

