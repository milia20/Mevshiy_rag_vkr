"""
Тесты для API эндпоинтов.

Модуль содержит тесты для FastAPI маршрутов.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Тесты для эндпоинтов здоровья."""

    def test_health_check(self, app_client: TestClient) -> None:
        """Тест проверки здоровья сервиса."""
        response = app_client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["service"] == "mkdocs-rag-api"
        assert data["version"] == "0.1.0"

    def test_readiness_check(self, app_client: TestClient) -> None:
        """Тест проверки готовности сервиса."""
        response = app_client.get("/api/v1/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"

    def test_liveness_check(self, app_client: TestClient) -> None:
        """Тест проверки жизнеспособности сервиса."""
        response = app_client.get("/api/v1/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"


class TestSearchEndpoints:
    """Тесты для поисковых эндпоинтов."""

    def test_search_get_empty_query(self, app_client: TestClient) -> None:
        """Тест поиска с пустым запросом (должен вернуть ошибку)."""
        response = app_client.get("/api/v1/search")
        assert response.status_code == 422  # Validation error

    def test_search_get_short_query(self, app_client: TestClient) -> None:
        """Тест поиска с коротким запросом."""
        response = app_client.get("/api/v1/search", params={"q": ""})
        assert response.status_code == 422

    def test_search_get_valid(self, app_client: TestClient) -> None:
        """Тест валидного GET запроса поиска."""
        response = app_client.get("/api/v1/search", params={"q": "тест"})
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert data["query"] == "тест"
        assert "results" in data
        assert "total" in data

    def test_search_get_with_limit(self, app_client: TestClient) -> None:
        """Тест поиска с ограничением результатов."""
        response = app_client.get("/api/v1/search", params={"q": "тест", "limit": 5})
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) <= 5

    def test_search_post_valid(self, app_client: TestClient) -> None:
        """Тест валидного POST запроса поиска."""
        response = app_client.post("/api/v1/search", json={"query": "тест", "limit": 10})
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "тест"
        assert "results" in data


class TestChatEndpoints:
    """Тесты для чат эндпоинтов."""

    def test_chat_message_valid(self, app_client: TestClient) -> None:
        """Тест отправки сообщения в чат."""
        payload = {
            "messages": [
                {"role": "user", "content": "Привет!"},
            ]
        }
        response = app_client.post("/api/v1/chat/message", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "role" in data
        assert "content" in data

    def test_chat_message_multiple(self, app_client: TestClient) -> None:
        """Тест отправки нескольких сообщений в чат."""
        payload = {
            "messages": [
                {"role": "user", "content": "Вопрос 1"},
                {"role": "assistant", "content": "Ответ 1"},
                {"role": "user", "content": "Вопрос 2"},
            ]
        }
        response = app_client.post("/api/v1/chat/message", json=payload)
        assert response.status_code == 200

    def test_chat_file_upload(self, app_client: TestClient, tmp_path: Any) -> None:
        """Тест загрузки файлов в чат."""
        import tempfile

        # Создаем временный файл
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Тестовое содержимое файла")
            temp_file = f.name

        try:
            with open(temp_file, "rb") as fh:
                response = app_client.post(
                    "/api/v1/chat",
                    data={"history": "[]"},
                    files={"files": ("test.txt", fh, "text/plain")},
                )
            assert response.status_code == 200
        finally:
            import os

            os.unlink(temp_file)


class TestRootEndpoint:
    """Тесты для корневого эндпоинта."""

    def test_root(self, app_client: TestClient) -> None:
        """Тест корневого эндпоинта."""
        response = app_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data
        assert "health" in data
