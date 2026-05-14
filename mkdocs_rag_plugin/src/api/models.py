"""
Эндпоинты для управления моделями.

Модуль содержит маршруты для получения информации о доступных
моделях эмбеддингов и LLM, а также статистики коллекции.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


class ModelInfo(BaseModel):
    """
    Информация о модели.

    Атрибуты:
        name: Название модели.
        type: Тип модели (embedding, llm).
        provider: Провайдер модели.
        dimension: Размерность вектора (для embedding моделей).
        max_tokens: Максимальное количество токенов (для LLM).
        supported_languages: Список поддерживаемых языков.
    """

    name: str
    type: str
    provider: str
    dimension: int | None = None
    max_tokens: int | None = None
    supported_languages: list[str] = Field(default_factory=list)


class ModelsListResponse(BaseModel):
    """
    Ответ со списком моделей.

    Атрибуты:
        embedding_models: Список моделей эмбеддингов.
        llm_models: Список LLM моделей.
    """

    embedding_models: list[ModelInfo]
    llm_models: list[ModelInfo]


class CollectionStats(BaseModel):
    """
    Статистика коллекции Qdrant.

    Атрибуты:
        collection_name: Имя коллекции.
        documents_count: Количество документов.
        vectors_count: Количество векторов.
        vector_size: Размерность векторов.
        status: Статус коллекции.
        index_type: Тип индекса.
    """

    collection_name: str
    documents_count: int
    vectors_count: int
    vector_size: int
    status: str
    index_type: str | None = None


class ModelsStatsResponse(BaseModel):
    """
    Ответ со статистикой моделей и коллекции.

    Атрибуты:
        collection: Статистика коллекции.
        current_embedding_model: Текущая модель эмбеддингов.
        current_llm_model: Текущая LLM модель.
    """

    collection: CollectionStats | None
    current_embedding_model: str
    current_llm_model: str


@router.get("/models", response_model=ModelsListResponse)
async def list_models() -> ModelsListResponse:
    """
    Получить список доступных моделей.

    Возвращает информацию о доступных моделях эмбеддингов и LLM,
    включая их параметры и поддерживаемые языки.

    Returns:
        ModelsListResponse со списками моделей.
    """
    from src.core.config import settings

    # Модели эмбеддингов с поддержкой русского языка
    embedding_models = [
        ModelInfo(
            name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            type="embedding",
            provider="huggingface",
            dimension=384,
            supported_languages=["ru", "en", "de", "fr", "es", "it", "zh", "ar"],
        ),
        ModelInfo(
            name="GigaEmbeddings",
            type="embedding",
            provider="sberbank",
            dimension=768,
            supported_languages=["ru"],
        ),
        ModelInfo(
            name="BGE-M3",
            type="embedding",
            provider="bge",
            dimension=1024,
            supported_languages=["ru", "en", "zh", "fr", "es", "ja"],
        ),
        ModelInfo(
            name="mE5-large",
            type="embedding",
            provider="microsoft",
            dimension=1024,
            supported_languages=[
                "ru",
                "en",
                "ar",
                "de",
                "es",
                "fr",
                "hi",
                "ja",
                "ko",
                "pt",
                "th",
                "tr",
                "vi",
                "zh",
            ],
        ),
        ModelInfo(
            name=settings.embedding_model,
            type="embedding",
            provider="current",
            dimension=settings.embedding_dimension,
            supported_languages=["ru", "en"],
        ),
    ]

    # LLM модели
    llm_models = [
        ModelInfo(
            name="qwen2.5:7b",
            type="llm",
            provider="ollama",
            max_tokens=32768,
            supported_languages=["ru", "en", "zh"],
        ),
        ModelInfo(
            name="llama3:8b",
            type="llm",
            provider="ollama",
            max_tokens=8192,
            supported_languages=["ru", "en"],
        ),
        ModelInfo(
            name="mistral:7b",
            type="llm",
            provider="ollama",
            max_tokens=32768,
            supported_languages=["ru", "en", "fr", "de", "es"],
        ),
        ModelInfo(
            name=settings.llm_model,
            type="llm",
            provider=settings.llm_provider,
            max_tokens=4096,
            supported_languages=["ru", "en"],
        ),
    ]

    return ModelsListResponse(
        embedding_models=embedding_models,
        llm_models=llm_models,
    )


@router.get("/models/stats", response_model=ModelsStatsResponse)
async def get_models_stats() -> ModelsStatsResponse:
    """
    Получить статистику коллекции и текущих моделей.

    Возвращает информацию о количестве документов в индексе,
    размерности векторов и текущих используемых моделях.

    Returns:
        ModelsStatsResponse со статистикой.

    Raises:
        HTTPException: Если произошла ошибка при получении статистики.
    """
    from src.core.config import settings
    from src.services.qdrant_service import get_qdrant_service

    collection_stats: CollectionStats | None = None

    try:
        qdrant = get_qdrant_service()

        # Получаем информацию о коллекции
        try:
            if qdrant.client is None:
                qdrant.connect()

            collection_info = qdrant.client.get_collection(qdrant.collection_name)

            collection_stats = CollectionStats(
                collection_name=qdrant.collection_name,
                documents_count=collection_info.points_count or 0,
                vectors_count=collection_info.vectors_count or 0,
                vector_size=collection_info.config.params.vectors.size,
                status=collection_info.status,
                index_type=getattr(collection_info.config.params, "shard_number", None),
            )
        except Exception as e:
            logger.warning(f"Не удалось получить статистику коллекции: {e}")
            collection_stats = None

    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "InternalError", "message": f"Ошибка получения статистики: {e}"},
        ) from e

    return ModelsStatsResponse(
        collection=collection_stats,
        current_embedding_model=settings.embedding_model,
        current_llm_model=settings.llm_model,
    )
