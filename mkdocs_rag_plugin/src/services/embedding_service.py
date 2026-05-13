"""
Сервис для генерации эмбеддингов.

Модуль предоставляет класс для создания векторных представлений текста
с использованием модели sentence-transformers.
"""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer
from src.core.config import settings
from src.core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Сервис для генерации эмбеддингов.

    Использует модель sentence-transformers для создания векторных
    представлений текста. Поддерживает многозадачность и кэширование.

    Атрибуты:
        model_name: Название модели для эмбеддингов.
        model: Экземпляр модели SentenceTransformer.
    """

    def __init__(self, model_name: str | None = None) -> None:
        """
        Инициализация сервиса эмбеддингов.

        Args:
            model_name: Название модели. Если не указано, используется из настроек.
        """
        self.model_name = model_name or settings.embedding_model
        logger.info(f"Инициализация EmbeddingService: model={self.model_name}")

        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """
        Получить экземпляр модели (ленивая загрузка).

        Returns:
            Экземпляр SentenceTransformer.
        """
        if self._model is None:
            logger.info(f"Загрузка модели {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Модель успешно загружена")
        return self._model

    def encode(
        self,
        text: str,
        normalize: bool = True,
    ) -> list[float]:
        """
        Создать эмбеддинг для одного текста.

        Args:
            text: Текст для кодирования.
            normalize: Нормализовать ли вектор (единичная длина).

        Returns:
            Вектор эмбеддинга.

        Raises:
            EmbeddingError: Если произошла ошибка при кодировании.
        """
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
            )
            return embedding.tolist()
        except Exception as e:
            msg = f"Ошибка кодирования текста: {e}"
            logger.error(msg)
            raise EmbeddingError(msg, str(e)) from e

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Создать эмбеддинги для списка текстов.

        Args:
            texts: Список текстов для кодирования.
            batch_size: Размер батча для обработки.
            normalize: Нормализовать ли векторы.
            show_progress: Показывать ли прогресс-бар.

        Returns:
            Список векторов эмбеддингов.

        Raises:
            EmbeddingError: Если произошла ошибка при кодировании.
        """
        if not texts:
            return []

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
            )
            return embeddings.tolist()
        except Exception as e:
            msg = f"Ошибка пакетного кодирования: {e}"
            logger.error(msg)
            raise EmbeddingError(msg, str(e)) from e

    def get_dimension(self) -> int:
        """
        Получить размерность векторов эмбеддинга.

        Returns:
            Размерность вектора.
        """
        return self.model.get_sentence_embedding_dimension()

    def encode_query(self, query: str) -> list[float]:
        """
        Создать эмбеддинг для поискового запроса.

        Этот метод может быть переопределен для моделей, поддерживающих
        разные промпты для запросов и документов.

        Args:
            query: Поисковый запрос.

        Returns:
            Вектор эмбеддинга запроса.
        """
        # Для большинства моделей просто кодируем текст
        return self.encode(query)

    def encode_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Создать эмбеддинги для документов.

        Этот метод может быть переопределен для моделей, поддерживающих
        разные промпты для запросов и документов.

        Args:
            documents: Список текстов документов.

        Returns:
            Список векторов эмбеддингов.
        """
        return self.encode_batch(documents)


# Singleton instance
_embedding_service: EmbeddingService | None = None


def get_embedding_service(model_name: str | None = None) -> EmbeddingService:
    """
    Получить экземпляр сервиса эмбеддингов (singleton).

    Args:
        model_name: Название модели (опционально, используется только при первой инициализации).

    Returns:
        Экземпляр EmbeddingService.
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(model_name=model_name)
    return _embedding_service
