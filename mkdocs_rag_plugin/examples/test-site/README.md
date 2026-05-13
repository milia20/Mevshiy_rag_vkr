# MkDocs RAG Plugin - Пример тестового сайта

Этот пример демонстрирует настройку MkDocs проекта с RAG плагином.

## Структура проекта

```
test-site/
├── mkdocs.yml          # Конфигурация MkDocs
├── docs/
│   ├── index.md        # Главная страница
│   ├── installation.md # Страница установки
│   ├── configuration.md # Страница конфигурации
│   └── usage.md        # Страница использования
└── ...
```

## Быстрый старт

### 1. Установка зависимостей

```bash
cd examples/test-site
pip install -e ../..
```

### 2. Настройка mkdocs.yml

Файл `mkdocs.yml` уже настроен. При необходимости измените параметры:

```yaml
plugins:
  - search:
      enabled: false
  - rag_plugin:
      qdrant_url: http://localhost:6333
      collection_name: docs_index
      embedding_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
      llm_provider: lmstudio
      llm_model: qwen-2.5-7b
      api_host: localhost
      api_port: 8000
      enable_chat_panel: true
```

### 3. Запуск сервисов

#### Вариант A: Docker (рекомендуется)

```bash
cd ../../docker
docker-compose up -d
```

#### Вариант B: Локальный запуск

1. Запустите Qdrant:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

2. Запустите API сервер:

```bash
cd ../..
source .venv/bin/activate  # или .venv\Scripts\activate на Windows
uvicorn src.main:app --host localhost --port 8000
```

3. Запустите LLM (например, LM Studio):

- Откройте LM Studio
- Загрузите модель (например, Qwen 2.5 7B)
- Запустите локальный сервер на порту 1234

### 4. Сборка документации

```bash
mkdocs build
```

### 5. Запуск dev сервера

```bash
mkdocs serve
```

Откройте браузер:

- Документация: http://localhost:8000
- RAG Frontend: http://localhost:3000/frontend/index.html
- API Docs: http://localhost:8000/docs

## Тестирование RAG

### Через веб-интерфейс

1. Откройте `frontend/index.html` в браузере
2. Введите вопрос по документации
3. Выберите режим поиска (hybrid/dense/sparse)
4. Нажмите "Задать вопрос"

### Через API

```bash
# Query endpoint
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Как установить плагин?",
    "top_k": 5,
    "mode": "hybrid",
    "stream": false
  }'

# Index endpoint
curl -X POST http://localhost:8000/api/v1/index \
  -H "Content-Type: application/json"

# Health check
curl http://localhost:8000/health
```

### Через Python

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={
        "question": "Как настроить аутентификацию?",
        "top_k": 5,
        "mode": "hybrid"
    }
)

data = response.json()
print("Ответ:", data["answer"])
print("Источники:", data["sources"])
```

## Оценка качества

Для оценки качества RAG системы используйте скрипт evaluation:

```bash
# Создайте тестовый датасет
cat > test_queries.json << 'EOF'
[
  {
    "question": "Как установить плагин?",
    "expected_answer": "Для установки используйте pip install mkdocs-rag-plugin",
    "relevant_docs": ["installation.md"]
  },
  {
    "question": "Какие параметры конфигурации доступны?",
    "expected_answer": "Доступны параметры: qdrant_url, embedding_model, llm_model...",
    "relevant_docs": ["configuration.md"]
  }
]
EOF

# Запустите оценку
python ../../scripts/evaluate.py \
  --dataset test_queries.json \
  --mode hybrid \
  --top-k 5 \
  --format both
```

## Очистка

```bash
# Удалить коллекцию Qdrant
curl -X DELETE http://localhost:8000/api/v1/index

# Остановить Docker контейнеры
cd ../../docker
docker-compose down
```

## Troubleshooting

### Qdrant не подключается

Проверьте, что Qdrant запущен:

```bash
curl http://localhost:6333
```

### LLM не отвечает

Проверьте настройки в `.env`:

```
LLM_BASE_URL=http://localhost:1234/v1
LLM_MODEL_NAME=qwen-2.5-7b
```

### Ошибки при индексации

Убедитесь, что все страницы имеют контент:

```bash
mkdocs build --verbose
```
