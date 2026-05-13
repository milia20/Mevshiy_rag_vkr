# Установка MkDocs RAG Plugin

## Требования

- Python 3.13+
- Docker и Docker Compose (рекомендуется)
- LLM сервер (LM Studio или Ollama)

## Вариант 1: Установка через pip

### Шаг 1: Установка пакета

```bash
pip install mkdocs-rag-plugin
```

Или из исходного кода:

```bash
git clone https://github.com/your-username/mkdocs-rag-plugin.git
cd mkdocs-rag-plugin
pip install -e ".[dev]"
```

### Шаг 2: Настройка .env файла

Скопируйте `.env.example` в `.env` и настройте параметры:

```bash
cp .env.example .env
```

### Шаг 3: Запуск сервисов

#### Qdrant (векторная база данных)

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  --name qdrant \
  qdrant/qdrant
```

#### LLM сервер

**Вариант A: LM Studio**

1. Скачайте с [lmstudio.ai](https://lmstudio.ai/)
2. Загрузите модель (например, Qwen 2.5 7B)
3. Запустите локальный сервер на порту 1234

**Вариант B: Ollama**

```bash
docker run -d -p 11434:11434 \
  --name ollama \
  ollama/ollama

docker exec ollama ollama pull qwen2.5:7b
```

## Вариант 2: Docker Compose (рекомендуется)

### Шаг 1: Клонирование репозитория

```bash
git clone https://github.com/your-username/mkdocs-rag-plugin.git
cd mkdocs-rag-plugin/docker
```

### Шаг 2: Настройка окружения

```bash
cp ../.env.example .env
```

### Шаг 3: Запуск всех сервисов

```bash
docker-compose up -d
```

Проверка статуса:

```bash
docker-compose ps
```

Ожидаемый вывод:

```
NAME                STATUS              PORTS
mkdocs_api          Up                  0.0.0.0:8000->8000/tcp
qdrant              Up                  0.0.0.0:6333->6333/tcp
ollama              Up                  0.0.0.0:11434->11434/tcp
```

## Проверка установки

### 1. Проверка Qdrant

```bash
curl http://localhost:6333
```

Ответ: `{"title":"qdrant - vector search engine","version":"..."}`

### 2. Проверка API

```bash
curl http://localhost:8000/health
```

Ответ: `{"status":"healthy","qdrant":"connected",...}`

### 3. Проверка LLM

**Для LM Studio:**

```bash
curl http://localhost:1234/v1/models
```

**Для Ollama:**

```bash
curl http://localhost:11434/api/tags
```

## Автоматическая установка

Используйте скрипт автоматической установки:

```bash
chmod +x scripts/install.sh
./scripts/install.sh
```

Скрипт выполнит:

- Проверку Python
- Создание виртуального окружения
- Установку зависимостей
- Копирование .env файла
- Опциональный запуск Docker контейнеров

## Troubleshooting

### Ошибка: "No module named 'mkdocs_plugin'"

Убедитесь, что пакет установлен в режиме editable:

```bash
pip install -e .
```

### Ошибка: "Qdrant connection failed"

Проверьте, что Qdrant запущен:

```bash
docker ps | grep qdrant
```

Перезапустите при необходимости:

```bash
docker restart qdrant
```

### Ошибка: "LLM timeout"

1. Убедитесь, что LLM сервер запущен
2. Проверьте настройки в `.env`:
   ```
   LLM_BASE_URL=http://localhost:1234/v1
   LLM_TIMEOUT=60
   ```
3. Для больших моделей увеличьте таймаут

### Ошибка: "CUDA out of memory"

Для GPU используйте меньшую модель или увеличьте память:

```yaml
# В docker-compose.yml раскомментируйте:
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

## Следующие шаги

После успешной установки:

1. Настройте [конфигурацию](configuration.md)
2. Изучите примеры [использования](usage.md)
3. Запустите [оценку качества](../examples/test-site/README.md)
