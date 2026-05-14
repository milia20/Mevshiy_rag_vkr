"""
Фикстуры для тестирования.

Модуль содержит общие фикстуры pytest для тестов.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def test_config() -> dict[str, Any]:
    """
    Конфигурация для тестов.

    Returns:
        Словарь с тестовой конфигурацией.
    """
    return {
        "qdrant_url": "http://localhost:6333",
        "qdrant_collection": "test_docs_index",
        "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "llm_provider": "ollama",
        "llm_model": "qwen2.5:7b",
        "api_host": "localhost",
        "api_port": 8001,  # Отличный от основного порт для тестов
        "enable_chat_panel": True,
        "api_password": "",
    }


@pytest.fixture(scope="session")
def sample_html() -> str:
    """
    Пример HTML-контента для тестов.

    Returns:
        Строка с HTML-разметкой.
    """
    return """
    <html>
        <head><title>Тестовая страница</title></head>
        <body>
            <h1>Заголовок</h1>
            <p>Это тестовый абзац с <b>жирным</b> текстом.</p>
            <script>alert('should be removed');</script>
            <style>.hidden { display: none; }</style>
            <a href="https://example.com">Ссылка</a>
        </body>
    </html>
    """


@pytest.fixture(scope="session")
def sample_markdown() -> str:
    """
    Пример Markdown-контента для тестов.

    Returns:
        Строка с Markdown-разметкой.
    """
    return """
# Заголовок первого уровня

## Заголовок второго уровня

Это обычный текст с **жирным** и *курсивным* форматированием.

[Ссылка](https://example.com)

```python
def hello():
    print("Привет, мир!")
```

- Список 1
- Список 2
- Список 3
"""


@pytest.fixture
def temp_dir(tmp_path: Path) -> Generator[Path]:
    """
    Временная директория для тестов.

    Args:
        tmp_path: Фикстура pytest для временных путей.

    Yields:
        Путь к временной директории.
    """
    test_dir = tmp_path / "rag_test"
    test_dir.mkdir()
    yield test_dir


@pytest.fixture
def mock_qdrant_response() -> list[dict[str, Any]]:
    """
    Мок ответа от Qdrant.

    Returns:
        Список мок-результатов поиска.
    """
    return [
        {
            "id": "doc_1_chunk_0",
            "score": 0.95,
            "payload": {
                "text": "Тестовый документ 1",
                "source": "test_doc.md",
                "page": "index",
            },
        },
        {
            "id": "doc_2_chunk_0",
            "score": 0.87,
            "payload": {
                "text": "Тестовый документ 2",
                "source": "another_doc.md",
                "page": "about",
            },
        },
    ]


@pytest.fixture
def mock_embedding_vector() -> list[float]:
    """
    Мок вектора эмбеддинга.

    Returns:
        Список float чисел, имитирующий вектор.
    """
    return [0.1] * 384  # Стандартная размерность для MiniLM


@pytest.fixture
def app_client() -> Generator[TestClient]:
    """
    Тестовый клиент FastAPI приложения.

    Yields:
        TestClient для тестирования API.
    """
    from src.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture(autouse=True)
def set_log_level() -> None:
    """
    Установить уровень логирования для тестов.

    Автоматически применяется ко всем тестам.
    """
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("mkdocs.plugins.rag_plugin").setLevel(logging.DEBUG)
    logging.getLogger("src").setLevel(logging.DEBUG)
