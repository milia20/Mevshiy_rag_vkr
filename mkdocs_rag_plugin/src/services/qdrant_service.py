"""
Сервис для работы с векторной базой данных Qdrant.

Модуль предоставляет класс для подключения, индексации и поиска
в векторной базе данных Qdrant.
"""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.core.config import settings
from src.core.exceptions import QdrantError

logger = logging.getLogger(__name__)


class QdrantService:
    """
    Сервис для взаимодействия с Qdrant.

    Предоставляет методы для создания коллекций, индексации документов
    и выполнения семантического поиска.

    Атрибуты:
        client: Клиент Qdrant.
        collection_name: Имя коллекции для работы.
    """

    def __init__(
        self,
        url: str | None = None,
        collection_name: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """
        Инициализация сервиса Qdrant.

        Args:
            url: URL подключения к Qdrant. Если не указан, используется из настроек.
            collection_name: Имя коллекции. Если не указано, используется из настроек.
            api_key: API ключ для подключения. Если не указан, используется из настроек.
        """
        self.url = url or settings.qdrant_url
        self.collection_name = collection_name or settings.qdrant_collection
        self.api_key = api_key or settings.qdrant_api_key

        logger.info(
            f"Инициализация QdrantService: url={self.url}, collection={self.collection_name}"
        )

        self.client: QdrantClient | None = None

    def connect(self) -> None:
        """
        Установить подключение к Qdrant.

        Raises:
            QdrantError: Если не удалось подключиться.
        """
        try:
            self.client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
            )
            # Проверка подключения
            self.client.get_collections()
            logger.info("Успешное подключение к Qdrant")
        except Exception as e:
            msg = f"Не удалось подключиться к Qdrant: {e}"
            logger.error(msg)
            raise QdrantError(msg, str(e)) from e

    def create_collection(
        self,
        vector_size: int = 384,
        distance: str = "Cosine",
    ) -> bool:
        """
        Создать коллекцию в Qdrant.

        Args:
            vector_size: Размерность векторов.
            distance: Функция расстояния (Cosine, Euclid, Dot).

        Returns:
            True если коллекция создана или уже существует.

        Raises:
            QdrantError: Если произошла ошибка при создании.
        """
        if not self.client:
            self.connect()

        try:
            assert self.client is not None  # Для type checker

            collections = self.client.get_collections().collections
            existing = any(c.name == self.collection_name for c in collections)

            if existing:
                logger.info(f"Коллекция '{self.collection_name}' уже существует")
                return True

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance[distance.upper()],
                ),
            )
            logger.info(f"Коллекция '{self.collection_name}' успешно создана")
            return True

        except Exception as e:
            msg = f"Ошибка создания коллекции: {e}"
            logger.error(msg)
            raise QdrantError(msg, str(e)) from e

    def upsert_points(
        self,
        points: list[PointStruct],
    ) -> bool:
        """
        Добавить или обновить точки в коллекции.

        Args:
            points: Список точек для добавления.

        Returns:
            True если операция успешна.

        Raises:
            QdrantError: Если произошла ошибка при добавлении.
        """
        if not self.client:
            self.connect()

        try:
            assert self.client is not None  # Для type checker

            result = self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            logger.info(f"Добавлено {len(points)} точек в коллекцию")
            return result.status == "completed"

        except Exception as e:
            msg = f"Ошибка добавления точек: {e}"
            logger.error(msg)
            raise QdrantError(msg, str(e)) from e

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Выполнить поиск похожих векторов.

        Args:
            query_vector: Вектор запроса.
            limit: Максимальное количество результатов.
            filter_dict: Фильтр для поиска (опционально).

        Returns:
            Список результатов поиска.

        Raises:
            QdrantError: Если произошла ошибка при поиске.
        """
        if not self.client:
            self.connect()

        try:
            assert self.client is not None  # Для type checker

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=filter_dict,
                limit=limit,
            )

            return [
                {
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload,
                }
                for point in results
            ]

        except Exception as e:
            msg = f"Ошибка поиска: {e}"
            logger.error(msg)
            raise QdrantError(msg, str(e)) from e

    def delete_collection(self) -> bool:
        """
        Удалить коллекцию.

        Returns:
            True если коллекция удалена.

        Raises:
            QdrantError: Если произошла ошибка при удалении.
        """
        if not self.client:
            self.connect()

        try:
            assert self.client is not None  # Для type checker

            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Коллекция '{self.collection_name}' удалена")
            return True

        except Exception as e:
            msg = f"Ошибка удаления коллекции: {e}"
            logger.error(msg)
            raise QdrantError(msg, str(e)) from e

    def close(self) -> None:
        """Закрыть подключение к Qdrant."""
        if self.client:
            self.client.close()
            logger.info("Подключение к Qdrant закрыто")


# Singleton instance
_qdrant_service: QdrantService | None = None


def get_qdrant_service() -> QdrantService:
    """
    Получить экземпляр сервиса Qdrant (singleton).

    Returns:
        Экземпляр QdrantService.
    """
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService()
    return _qdrant_service
