#!/bin/bash
# Script for publishing MkDocs RAG Plugin to PyPI
# Скрипт для публикации пакета в PyPI

set -e

echo "🚀 Публикация MkDocs RAG Plugin на PyPI..."

# Проверка зависимостей
if ! command -v build &> /dev/null; then
    echo "❌ twine не найден. Установка..."
    pip install build twine
fi

# Очистка предыдущих сборок
echo "🧹 Очистка предыдущих сборок..."
rm -rf dist/ build/ *.egg-info

# Сборка пакета
echo "📦 Сборка пакета..."
python -m build

# Проверка пакета
echo "🔍 Проверка пакета..."
twine check dist/*

# Загрузка на PyPI
echo "📤 Загрузка на PyPI..."
twine upload dist/*

echo ""
echo "============================================"
echo "✅ Публикация завершена!"
echo ""
echo "Пакет доступен на: https://pypi.org/project/mkdocs-rag-plugin/"
echo ""
echo "Для установки:"
echo "  pip install mkdocs-rag-plugin"
echo "============================================"
