"""
Сервис для индексации документов в Qdrant.

Модуль предоставляет класс для обработки и индексации документов,
генерации эмбеддингов и управления точками в векторной базе данных.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from qdrant_client.models import PointStruct

from src.core.exceptions import DocumentProcessingError, EmbeddingError, QdrantError
from src.services.embedding_service import get_embedding_service
from src.services.qdrant_client import get_qdrant_service

logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """
    Обработанный документ для индексации.

    Атрибуты:
        url: URL документа.
        title: Заголовок документа.
        content: Содержимое чанка.
        header_path: Путь заголовка (иерархия).
        source_file: Исходный файл.
        chunk_id: Идентификатор чанка.
        metadata: Дополнительные метаданные.
    """

    url: str
    title: str
    content: str
    header_path: str
    source_file: str
    chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexingResult:
    """
    Результат индексации.

    Атрибуты:
        total_documents: Общее количество документов.
        indexed_documents: Количество успешно индексированных.
        failed_documents: Количество неудачных попыток.
        errors: Список ошибок.
        elapsed_time_ms: Время выполнения в миллисекундах.
    """

    total_documents: int
    indexed_documents: int
    failed_documents: int
    errors: list[str] = field(default_factory=list)
    elapsed_time_ms: int = 0


class Indexer:
    """
    Сервис для индексации документов в Qdrant.

    Поддерживает:
    - Пакетную индексацию с генерацией эмбеддингов
    - Удаление документов по URL
    - Очистку всего индекса
    - Отслеживание прогресса

    Атрибуты:
        collection_name: Имя коллекции для индексации.
        batch_size: Размер батча для обработки.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        batch_size: int = 100,
    ) -> None:
        """
        Инициализация индексера.

        Args:
            collection_name: Имя коллекции. Если не указано, используется из настроек.
            batch_size: Размер батча для пакетной обработки.
        """
        from src.core.config import settings

        self.collection_name = collection_name or settings.qdrant_collection
        self.batch_size = batch_size

        self._qdrant_service = get_qdrant_service()
        self._embedding_service = get_embedding_service()

        logger.info(f"Инициализация Indexer: collection={self.collection_name}, batch_size={self.batch_size}")

    async def index_documents(
        self,
        documents: list[ProcessedDocument],
        create_collection_if_not_exists: bool = True,
    ) -> IndexingResult:
        """
        Индексировать список документов.

        Генерирует эмбеддинги для каждого чанка и добавляет точки в Qdrant.

        Args:
            documents: Список обработанных документов для индексации.
            create_collection_if_not_exists: Создать коллекцию если не существует.

        Returns:
            IndexingResult с результатами индексации.

        Raises:
            DocumentProcessingError: Если произошла ошибка при обработке.
        """
        import time

        start_time = time.time()
        errors: list[str] = []
        indexed_count = 0
        failed_count = 0

        logger.info(f"Начало индексации {len(documents)} документов")

        try:
            # Создаем коллекцию если нужно
            if create_collection_if_not_exists:
                vector_size = self._embedding_service.get_dimension()
                await self._qdrant_service.create_collection(
                    vector_size=vector_size,
                    use_sparse=True,
                )

            # Генерируем эмбеддинги батчами
            contents = [doc.content for doc in documents]
            embeddings = await self._generate_embeddings_batch(contents)

            if len(embeddings) != len(documents):
                msg = f"Несовпадение количества эмбеддингов: {len(embeddings)} vs {len(documents)}"
                raise DocumentProcessingError(msg)

            # Создаем точки для Qdrant
            points: list[PointStruct] = []

            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                point_id = self._generate_point_id(doc)
                payload = self._build_payload(doc)

                point = PointStruct(
                    id=point_id,
                    vector={"dense": embedding, "text": self._build_sparse_vector(doc.content)},
                    payload=payload,
                )
                points.append(point)

            # Добавляем точки в Qdrant
            if points:
                success = await self._qdrant_service.upsert_points(
                    points,
                    batch_size=self.batch_size,
                )

                if success:
                    indexed_count = len(points)
                else:
                    failed_count = len(points)
                    errors.append("Ошибка добавления точек в Qdrant")

            elapsed_time = int((time.time() - start_time) * 1000)

            result = IndexingResult(
                total_documents=len(documents),
                indexed_documents=indexed_count,
                failed_documents=failed_count,
                errors=errors,
                elapsed_time_ms=elapsed_time,
            )

            logger.info(f"Индексация завершена: {indexed_count} успешно, {failed_count} ошибок, " f"{elapsed_time}мс")

            return result

        except Exception as e:
            msg = f"Ошибка индексации: {e}"
            logger.error(msg)
            errors.append(msg)

            elapsed_time = int((time.time() - start_time) * 1000)
            return IndexingResult(
                total_documents=len(documents),
                indexed_documents=indexed_count,
                failed_documents=failed_count + len(documents),
                errors=errors,
                elapsed_time_ms=elapsed_time,
            )

    async def _generate_embeddings_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Сгенерировать эмбеддинги для списка текстов.

        Args:
            texts: Список текстов для кодирования.

        Returns:
            Список векторов эмбеддингов.

        Raises:
            EmbeddingError: Если произошла ошибка при генерации.
        """
        try:
            # Используем синхронный метод для совместимости
            embeddings = self._embedding_service.encode_batch(
                texts,
                batch_size=self.batch_size,
                normalize=True,
                show_progress=True,
            )
            return embeddings

        except Exception as e:
            msg = f"Ошибка генерации эмбеддингов: {e}"
            logger.error(msg)
            raise EmbeddingError(msg, str(e)) from e

    def _generate_point_id(self, doc: ProcessedDocument) -> str:
        """
        Сгенерировать уникальный ID для точки.

        Args:
            doc: Обработанный документ.

        Returns:
            Строковый идентификатор точки.
        """
        import hashlib

        # Используем хэш от URL и chunk_id для уникальности
        key = f"{doc.url}:{doc.chunk_id}"
        return hashlib.md5(key.encode()).hexdigest()

    def _build_payload(self, doc: ProcessedDocument) -> dict[str, Any]:
        """
        Построить payload для точки.

        Args:
            doc: Обработанный документ.

        Returns:
            Словарь с метаданными для Qdrant.
        """
        return {
            "url": doc.url,
            "title": doc.title,
            "content": doc.content,
            "header_path": doc.header_path,
            "source_file": doc.source_file,
            "chunk_id": doc.chunk_id,
            **doc.metadata,
        }

    def _build_sparse_vector(self, text: str) -> dict[str, Any]:
        """
        Построить sparse вектор для BM25 поиска.

        Args:
            text: Текст для токенизации.

        Returns:
            Словарь с indices и values для sparse вектора.
        """
        tokens = self._qdrant_service._tokenize_text(text)

        # Создаем простой bag-of-words representation
        token_counts: dict[str, int] = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        # Преобразуем в формат sparse вектора
        indices = []
        values = []

        for token, count in token_counts.items():
            idx = hash(token) % 100000  # Простой хэш для индекса
            indices.append(idx)
            values.append(float(count))  # TF weight

        return {"indices": indices, "values": values}

    async def delete_by_url(self, url: str) -> bool:
        """
        Удалить документ по URL.

        Args:
            url: URL документа для удаления.

        Returns:
            True если документ удален.

        Raises:
            QdrantError: Если произошла ошибка при удалении.
        """
        logger.info(f"Удаление документа по URL: {url}")

        try:
            result = await self._qdrant_service.delete_by_url(url)
            return result

        except Exception as e:
            msg = f"Ошибка удаления по URL: {e}"
            logger.error(msg)
            raise QdrantError(msg, str(e)) from e

    async def clear_all(self) -> bool:
        """
        Очистить весь индекс.

        Returns:
            True если индекс очищен.

        Raises:
            QdrantError: Если произошла ошибка при очистке.
        """
        logger.info("Очистка всего индекса")

        try:
            result = await self._qdrant_service.clear_all()
            return result

        except Exception as e:
            msg = f"Ошибка очистки индекса: {e}"
            logger.error(msg)
            raise QdrantError(msg, str(e)) from e


# Singleton instance
_indexer: Indexer | None = None


def get_indexer(collection_name: str | None = None, batch_size: int = 100) -> Indexer:
    """
    Получить экземпляр индексера (singleton).

    Args:
        collection_name: Имя коллекции (используется только при первой инициализации).
        batch_size: Размер батча (используется только при первой инициализации).

    Returns:
        Экземпляр Indexer.
    """
    global _indexer
    if _indexer is None:
        _indexer = Indexer(collection_name=collection_name, batch_size=batch_size)
    return _indexer
