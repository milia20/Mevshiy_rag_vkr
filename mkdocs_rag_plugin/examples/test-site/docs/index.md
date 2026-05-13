# MkDocs RAG Plugin

Добро пожаловать в документацию **MkDocs RAG Plugin** — плагина для интеграции RAG (Retrieval-Augmented Generation)
системы с MkDocs документацией.

## Что такое RAG?

RAG (Retrieval-Augmented Generation) — это подход, сочетающий поиск информации с генерацией ответов на основе языковых
моделей. Плагин позволяет:

- 🔍 **Семантический поиск** по документации с использованием векторных эмбеддингов
- 🤖 **Генерация ответов** на вопросы с помощью LLM (Qwen, Llama, etc.)
- 📚 **Источники** — каждый ответ сопровождается ссылками на оригинальные документы
- 🇷🇺 **Поддержка русского языка** во всех компонентах системы

## Возможности

| Функция           | Описание                                                 |
|-------------------|----------------------------------------------------------|
| Гибридный поиск   | Комбинация dense (семантического) и sparse (BM25) поиска |
| Streaming ответов | Потоковая передача токенов ответа в реальном времени     |
| Мультиязычность   | Поддержка русских и английских документов                |
| Гибкая настройка  | Настройка параметров чанкования, поиска и генерации      |
| Docker поддержка  | Готовая конфигурация для развертывания                   |

## Архитектура

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   MkDocs    │────▶│  FastAPI     │────▶│   Qdrant    │
│   Plugin    │     │   Backend    │     │   Vector DB │
└─────────────┘     └──────────────┘     └─────────────┘
                          │
                          ▼
                    ┌──────────────┐
                    │  LLM Client  │
                    │ (Ollama/LM   │
                    │   Studio)    │
                    └──────────────┘
```

## Быстрый старт

### 1. Установка

```bash
pip install mkdocs-rag-plugin
```

### 2. Настройка mkdocs.yml

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
```

### 3. Запуск сервисов

```bash
# Docker
docker-compose up -d

# API сервер
uvicorn src.main:app --host localhost --port 8000

# MkDocs
mkdocs serve
```

## Документация

- [Установка](installation.md) — подробная инструкция по установке
- [Конфигурация](configuration.md) — все параметры настройки
- [Использование](usage.md) — примеры использования API и веб-интерфейса

## Оценка качества

Для оценки качества RAG системы используйте скрипт `evaluate.py`:

```bash
python scripts/evaluate.py \
  --dataset test_queries.json \
  --mode hybrid \
  --top-k 5
```

Метрики: NDCG@k, MRR, Hit@k, Faithfulness

## Поддержка

- GitHub Issues: [Сообщить об ошибке](https://github.com/your-username/mkdocs-rag-plugin/issues)
- Документация: [Полная документация](https://your-username.github.io/mkdocs-rag-plugin)

## Лицензия

MIT License
