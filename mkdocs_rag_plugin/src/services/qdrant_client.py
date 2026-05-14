"""
Асинхронный клиент Qdrant с поддержкой гибридного поиска.

Модуль предоставляет класс для управления коллекциями, индексации
и выполнения гибридного поиска (dense + sparse) в Qdrant.
"""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.core.config import settings
from src.core.exceptions import QdrantError

logger = logging.getLogger(__name__)


class QdrantService:
    """
    Асинхронный сервис для работы с Qdrant.

    Поддерживает:
    - Dense векторы (для семантического поиска)
    - Sparse векторы (для BM25 поиска)
    - Гибридный поиск с взвешенной фузией
    - Payload фильтры

    Атрибуты:
        url: URL подключения к Qdrant.
        collection_name: Имя коллекции.
        api_key: API ключ для аутентификации.
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

        logger.info(f"Инициализация QdrantService: url={self.url}, collection={self.collection_name}")

        self._sync_client: QdrantClient | None = None
        self._async_client: AsyncQdrantClient | None = None

    @property
    def client(self) -> QdrantClient:
        """
        Получить синхронный клиент (ленивая инициализация).

        Returns:
            Экземпляр QdrantClient.
        """
        if self._sync_client is None:
            self.connect()
        assert self._sync_client is not None
        return self._sync_client

    async def get_async_client(self) -> AsyncQdrantClient:
        """
        Получить асинхронный клиент (ленивая инициализация).

        Returns:
            Экземпляр AsyncQdrantClient.
        """
        if self._async_client is None:
            self._async_client = AsyncQdrantClient(
                url=self.url,
                api_key=self.api_key,
            )
            # Проверка подключения
            await self._async_client.get_collections()
            logger.info("Успешное асинхронное подключение к Qdrant")
        return self._async_client

    def connect(self) -> None:
        """
        Установить подключение к Qdrant.

        Raises:
            QdrantError: Если не удалось подключиться.
        """
        try:
            self._sync_client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
            )
            # Проверка подключения
            self._sync_client.get_collections()
            logger.info("Успешное подключение к Qdrant")
        except Exception as e:
            msg = f"Не удалось подключиться к Qdrant: {e}"
            logger.error(msg)
            raise QdrantError(msg, str(e)) from e

    async def create_collection(
        self,
        vector_size: int = 1024,
        distance: str = "Cosine",
        use_sparse: bool = True,
    ) -> bool:
        """
        Создать коллекцию в Qdrant с поддержкой sparse векторов.

        Args:
            vector_size: Размерность плотных векторов.
            distance: Функция расстояния (Cosine, Euclid, Dot).
            use_sparse: Использовать ли sparse векторы для BM25.

        Returns:
            True если коллекция создана или уже существует.

        Raises:
            QdrantError: Если произошла ошибка при создании.
        """
        client = await self.get_async_client()

        try:
            collections = await client.get_collections()
            existing = any(c.name == self.collection_name for c in collections.collections)

            if existing:
                logger.info(f"Коллекция '{self.collection_name}' уже существует")
                return True

            # Конфигурация векторов
            vectors_config = {
                "dense": VectorParams(
                    size=vector_size,
                    distance=Distance[distance.upper()],
                )
            }

            # Добавляем sparse вектор для BM25
            if use_sparse:
                vectors_config["text"] = SparseVectorParams()

            await client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
            )

            # Создаем индексы для payload полей
            await client.create_payload_index(
                collection_name=self.collection_name,
                field_name="url",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            await client.create_payload_index(
                collection_name=self.collection_name,
                field_name="header_path",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            await client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source_file",
                field_schema=PayloadSchemaType.KEYWORD,
            )

            logger.info(f"Коллекция '{self.collection_name}' успешно создана")
            return True

        except Exception as e:
            msg = f"Ошибка создания коллекции: {e}"
            logger.error(msg)
            raise QdrantError(msg, str(e)) from e

    async def upsert_points(
        self,
        points: list[PointStruct],
        batch_size: int = 100,
    ) -> bool:
        """
        Добавить или обновить точки в коллекции (асинхронно).

        Args:
            points: Список точек для добавления.
            batch_size: Размер батча для обработки.

        Returns:
            True если операция успешна.

        Raises:
            QdrantError: Если произошла ошибка при добавлении.
        """
        client = await self.get_async_client()

        try:
            # Разбиваем на батчи
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                result = await client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                )
                logger.info(f"Добавлен батч {i // batch_size + 1}: {len(batch)} точек")

            logger.info(f"Всего добавлено {len(points)} точек в коллекцию")
            return True

        except Exception as e:
            msg = f"Ошибка добавления точек: {e}"
            logger.error(msg)
            raise QdrantError(msg, str(e)) from e

    def upsert_points_sync(
        self,
        points: list[PointStruct],
        batch_size: int = 100,
    ) -> bool:
        """
        Добавить или обновить точки в коллекции (синхронно).

        Args:
            points: Список точек для добавления.
            batch_size: Размер батча для обработки.

        Returns:
            True если операция успешна.

        Raises:
            QdrantError: Если произошла ошибка при добавлении.
        """
        try:
            # Разбиваем на батчи
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                result = self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                )
                logger.info(f"Добавлен батч {i // batch_size + 1}: {len(batch)} точек")

            logger.info(f"Всего добавлено {len(points)} точек в коллекцию")
            return True

        except Exception as e:
            msg = f"Ошибка добавления точек: {e}"
            logger.error(msg)
            raise QdrantError(msg, str(e)) from e

    async def dense_search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Выполнить плотный семантический поиск.

        Args:
            query_vector: Вектор запроса.
            limit: Максимальное количество результатов.
            filter_dict: Фильтр для поиска (опционально).

        Returns:
            Список результатов поиска.

        Raises:
            QdrantError: Если произошла ошибка при поиске.
        """
        client = await self.get_async_client()

        try:
            qdrant_filter = None
            if filter_dict:
                qdrant_filter = self._build_filter(filter_dict)

            results = await client.search(
                collection_name=self.collection_name,
                query_vector=("dense", query_vector),
                query_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
            )

            return [
                {
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload,
                    "search_type": "dense",
                }
                for point in results
            ]

        except Exception as e:
            msg = f"Ошибка плотного поиска: {e}"
            logger.error(msg)
            raise QdrantError(msg, str(e)) from e

    async def sparse_search(
        self,
        query_text: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Выполнить разреженный поиск (BM25).

        Args:
            query_text: Текстовый запрос для токенизации.
            limit: Максимальное количество результатов.

        Returns:
            Список результатов поиска.

        Raises:
            QdrantError: Если произошла ошибка при поиске.
        """
        client = await self.get_async_client()

        try:
            # Токенизация запроса для sparse вектора
            tokens = self._tokenize_text(query_text)
            sparse_vector = SparseVector(
                indices=list(range(len(tokens))),  # Упрощенная версия
                values=[1.0] * len(tokens),
            )

            # В реальной реализации здесь будет TF-IDF или BM25
            # Для простоты используем единичные веса
            sparse_indices = []
            sparse_values = []
            token_to_idx = {}

            for token in tokens:
                if token not in token_to_idx:
                    idx = hash(token) % 100000  # Простой хэш для индекса
                    token_to_idx[token] = idx
                sparse_indices.append(token_to_idx[token])
                sparse_values.append(1.0)

            sparse_vector = SparseVector(
                indices=sparse_indices,
                values=sparse_values,
            )

            results = await client.search(
                collection_name=self.collection_name,
                query_vector=("text", sparse_vector),
                limit=limit,
                with_payload=True,
            )

            return [
                {
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload,
                    "search_type": "sparse",
                }
                for point in results
            ]

        except Exception as e:
            msg = f"Ошибка разреженного поиска: {e}"
            logger.error(msg)
            raise QdrantError(msg, str(e)) from e

    async def hybrid_search(
        self,
        query_vector: list[float],
        query_text: str,
        limit: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Выполнить гибридный поиск с взвешенной фузией.

        Комбинирует результаты плотного и разреженного поиска
        используя взвешенную сумму нормализованных скорингов.

        Args:
            query_vector: Плотный вектор запроса.
            query_text: Текстовый запрос для sparse поиска.
            limit: Максимальное количество результатов.
            dense_weight: Вес плотного поиска (0.0-1.0).
            sparse_weight: Вес разреженного поиска (0.0-1.0).
            filter_dict: Фильтр для поиска (опционально).

        Returns:
            Список результатов с комбинированными скорингами.

        Raises:
            QdrantError: Если произошла ошибка при поиске.
        """
        # Выполняем оба поиска параллельно
        dense_results, sparse_results = await asyncio.gather(
            self.dense_search(query_vector, limit=limit * 2, filter_dict=filter_dict),
            self.sparse_search(query_text, limit=limit * 2),
            return_exceptions=False,
        )

        # Нормализуем и комбинируем скоринги
        combined_scores: dict[int, dict[str, Any]] = {}

        # Находим макс скоринги для нормализации
        max_dense = max((r["score"] for r in dense_results), default=1.0) or 1.0
        max_sparse = max((r["score"] for r in sparse_results), default=1.0) or 1.0

        # Обрабатываем dense результаты
        for result in dense_results:
            doc_id = result["id"] if isinstance(result["id"], int) else hash(str(result["id"]))
            normalized_score = result["score"] / max_dense
            combined_scores[doc_id] = {
                "id": result["id"],
                "payload": result["payload"],
                "combined_score": normalized_score * dense_weight,
                "dense_score": normalized_score,
                "sparse_score": 0.0,
            }

        # Обрабатываем sparse результаты
        for result in sparse_results:
            doc_id = result["id"] if isinstance(result["id"], int) else hash(str(result["id"]))
            normalized_score = result["score"] / max_sparse

            if doc_id in combined_scores:
                combined_scores[doc_id]["sparse_score"] = normalized_score
                combined_scores[doc_id]["combined_score"] += normalized_score * sparse_weight
            else:
                combined_scores[doc_id] = {
                    "id": result["id"],
                    "payload": result["payload"],
                    "combined_score": normalized_score * sparse_weight,
                    "dense_score": 0.0,
                    "sparse_score": normalized_score,
                }

        # Сортируем по комбинированному скорингу
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x["combined_score"],
            reverse=True,
        )[:limit]

        # Добавляем метаданные
        for result in sorted_results:
            result["search_type"] = "hybrid"

        logger.info(
            f"Гибридный поиск: dense_weight={dense_weight}, sparse_weight={sparse_weight}, "
            f"результатов={len(sorted_results)}"
        )

        return sorted_results

    def _build_filter(self, filter_dict: dict[str, Any]) -> Filter:
        """
        Построить фильтр Qdrant из словаря.

        Args:
            filter_dict: Словарь с условиями фильтрации.

        Returns:
            Объект Filter для Qdrant.
        """
        must_conditions = []

        for condition in filter_dict.get("must", []):
            key = condition.get("key")
            match = condition.get("match", {})
            value = match.get("value")

            if key and value is not None:
                must_conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )

        return Filter(must=must_conditions) if must_conditions else None

    def _tokenize_text(self, text: str) -> list[str]:
        """
        Токенизировать текст для sparse поиска.

        Поддерживает русский и английский языки.

        Args:
            text: Текст для токенизации.

        Returns:
            Список токенов.
        """
        import re

        # Приводим к нижнему регистру
        text = text.lower()

        # Удаляем пунктуацию (поддержка кириллицы)
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

        # Разбиваем на токены
        tokens = text.split()

        # Удаляем стоп-слова (упрощенный список)
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "in",
            "on",
            "at",
            "to",
            "for",
            "и",
            "в",
            "на",
            "с",
            "по",
            "для",
            "или",
            "но",
            "а",
        }

        return [t for t in tokens if t not in stop_words and len(t) > 1]

    async def delete_by_url(self, url: str) -> int:
        """
        Удалить документы по URL.

        Args:
            url: URL документа для удаления.

        Returns:
            Количество удаленных документов.

        Raises:
            QdrantError: Если произошла ошибка при удалении.
        """
        client = await self.get_async_client()

        try:
            # Используем фильтр по URL
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="url",
                        match=MatchValue(value=url),
                    )
                ]
            )

            result = await client.delete(
                collection_name=self.collection_name,
                points_selector=filter_condition,
            )

            logger.info(f"Удалено документов по URL {url}: {result}")
            return result.status == "completed"

        except Exception as e:
            msg = f"Ошибка удаления по URL: {e}"
            logger.error(msg)
            raise QdrantError(msg, str(e)) from e

    async def clear_all(self) -> bool:
        """
        Очистить всю коллекцию.

        Returns:
            True если коллекция очищена.

        Raises:
            QdrantError: Если произошла ошибка при очистке.
        """
        client = await self.get_async_client()

        try:
            # Удаляем все точки
            result = await client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Коллекция '{self.collection_name}' полностью очищена")
            return True

        except Exception as e:
            msg = f"Ошибка очистки коллекции: {e}"
            logger.error(msg)
            raise QdrantError(msg, str(e)) from e

    async def get_collection_stats(self) -> dict[str, Any]:
        """
        Получить статистику коллекции.

        Returns:
            Словарь со статистикой коллекции.

        Raises:
            QdrantError: Если произошла ошибка при получении статистики.
        """
        client = await self.get_async_client()

        try:
            collection_info = await client.get_collection(self.collection_name)

            return {
                "collection_name": self.collection_name,
                "documents_count": collection_info.points_count or 0,
                "vectors_count": collection_info.vectors_count or 0,
                "vector_size": (
                    collection_info.config.params.vectors.size
                    if hasattr(collection_info.config.params, "vectors")
                    else 0
                ),
                "status": collection_info.status,
            }

        except Exception as e:
            msg = f"Ошибка получения статистики: {e}"
            logger.error(msg)
            raise QdrantError(msg, str(e)) from e

    def close(self) -> None:
        """Закрыть подключения к Qdrant."""
        if self._sync_client:
            self._sync_client.close()
            logger.info("Синхронное подключение к Qdrant закрыто")

        if self._async_client:
            import asyncio

            asyncio.run(self._async_client.close())
            logger.info("Асинхронное подключение к Qdrant закрыто")


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
