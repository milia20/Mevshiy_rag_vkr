#!/bin/bash
# MkDocs RAG Plugin Installation Script
# Скрипт автоматической установки и настройки плагина

set -e

echo "🚀 Установка MkDocs RAG Plugin..."

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 не найден. Пожалуйста, установите Python 3.13+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Найдена Python версии: $PYTHON_VERSION"

# Создание виртуального окружения
if [ ! -d ".venv" ]; then
    echo "📦 Создание виртуального окружения..."
    python3 -m venv .venv
else
    echo "✅ Виртуальное окружение уже существует"
fi

# Активация виртуального окружения
source .venv/bin/activate

# Установка зависимостей
echo "📥 Установка зависимостей..."
pip install --upgrade pip
pip install -e ".[dev]"

# Копирование .env файла
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "📝 Создание файла .env из .env.example..."
    cp .env.example .env
    echo "⚠️  Не забудьте настроить параметры в файле .env!"
fi

# Проверка Docker
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo "✅ Docker и Docker Compose найдены"

    read -p "🐳 Запустить Docker контейнеры? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🚀 Запуск Docker контейнеров..."
        cd docker
        docker-compose up -d

        echo "⏳ Ожидание запуска сервисов..."
        sleep 10

        # Проверка здоровья сервисов
        if curl -s http://localhost:6333 > /dev/null; then
            echo "✅ Qdrant запущен"
        else
            echo "⚠️  Qdrant не отвечает на localhost:6333"
        fi

        cd ..
    fi
else
    echo "⚠️  Docker не найден. Пропускаем запуск контейнеров."
    echo "   Для использования Docker установите Docker Desktop или Docker Engine"
fi

echo ""
echo "============================================"
echo "✅ Установка завершена!"
echo ""
echo "Следующие шаги:"
echo "1. Настройте параметры в файле .env"
echo "2. Запустите API сервер:"
echo "   source .venv/bin/activate"
echo "   uvicorn src.main:app --host localhost --port 8000"
echo ""
echo "3. Запустите MkDocs:"
echo "   mkdocs serve"
echo ""
echo "4. Откройте браузер:"
echo "   Frontend: http://localhost:3000"
echo "   API Docs: http://localhost:8000/docs"
echo "============================================"
