# Этап сборки зависимостей
FROM python:3.13-slim as builder

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Установка uv для быстрого управления зависимостями
RUN pip install uv

# Копирование файлов проекта
COPY pyproject.toml .
COPY mkdocs_plugin/ ./mkdocs_plugin/
COPY src/ ./src/

# Установка зависимостей
RUN uv pip install --system -e ".[dev]"

# Финальный этап
FROM python:3.13-slim as runtime

WORKDIR /app

# Создание пользователя для безопасности
RUN useradd --create-home --shell /bin/bash appuser

# Копирование установленных пакетов
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Копирование кода приложения
COPY mkdocs_plugin/ ./mkdocs_plugin/
COPY src/ ./src/
COPY pyproject.toml .

# Смена владельца
RUN chown -R appuser:appuser /app

USER appuser

# Переменные окружения
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV RAG_API_HOST=0.0.0.0
ENV RAG_API_PORT=8000

# Экспорт порта
EXPOSE 8000

# Запуск приложения
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
