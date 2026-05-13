"""
Эндпоинты для RAG запросов.

Модуль содержит маршруты для обработки вопросов пользователей
с использованием RAG системы с поддержкой стриминга.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from src.core.exceptions import LLMError, QdrantError, SearchError
from src.services.embedding_service import get_embedding_service
from src.services.llm_client import get_llm_client
from src.services.qdrant_service import get_qdrant_service

logger = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    """
    Модель запроса к RAG системе.

    Атрибуты:
        question: Вопрос пользователя.
        top_k: Количество результатов для поиска.
        country: Фильтр по стране (опционально).
        stream: Включить ли потоковую передачу.
    """

    question: str = Field(..., min_length=1, description="Вопрос пользователя")
    top_k: int = Field(default=5, ge=1, le=50, description="Количество результатов поиска")
    country: str | None = Field(default=None, description="Фильтр по стране")
    stream: bool = Field(default=True, description="Включить потоковую передачу ответа")


class SourceDocument(BaseModel):
    """
    Модель документа-источника.

    Атрибуты:
        url: URL документа.
        title: Заголовок документа.
        snippet: Фрагмент текста.
        score: Оценка релевантности.
    """

    url: str
    title: str
    snippet: str
    score: float


class ResponseMetadata(BaseModel):
    """
    Модель метаданных ответа.

    Атрибуты:
        retrieval_time_ms: Время поиска в миллисекундах.
        generation_time_ms: Время генерации ответа в миллисекундах.
        model: Название использованной модели.
    """

    retrieval_time_ms: int
    generation_time_ms: int
    model: str


class QueryResponse(BaseModel):
    """
    Модель ответа RAG системы.

    Атрибуты:
        answer: Сгенерированный ответ.
        sources: Список документов-источников.
        metadata: Метаданные выполнения запроса.
    """

    answer: str
    sources: list[SourceDocument]
    metadata: ResponseMetadata


class ErrorResponse(BaseModel):
    """
    Модель ошибки.

    Атрибуты:
        error: Тип ошибки.
        message: Сообщение об ошибке.
        details: Детали ошибки (опционально).
    """

    error: str
    message: str
    details: str | None = None


def _format_sources(search_results: list[dict[str, Any]]) -> list[SourceDocument]:
    """
    Преобразовать результаты поиска в список источников.

    Args:
        search_results: Результаты поиска из Qdrant.

    Returns:
        Список объектов SourceDocument.
    """
    sources = []
    for result in search_results:
        payload = result.get("payload", {})
        sources.append(
            SourceDocument(
                url=payload.get("url", "unknown"),
                title=payload.get("title", "Без названия"),
                snippet=payload.get("content", "")[:500],  # Ограничиваем длину
                score=result.get("score", 0.0),
            )
        )
    return sources


async def _generate_sse_response(
    question: str,
    top_k: int,
    country: str | None,
) -> AsyncGenerator[str, None]:
    """
    Генерировать SSE-ответ для потоковой передачи.

    Args:
        question: Вопрос пользователя.
        top_k: Количество результатов поиска.
        country: Фильтр по стране.

    Yields:
        SSE-события с частями ответа.
    """
    retrieval_start = time.time()

    try:
        # Поиск документов
        qdrant_service = get_qdrant_service()
        embedder = get_embedding_service()

        # Генерируем эмбеддинг запроса
        query_vector = embedder.encode_query(question)

        # Выполняем поиск
        filter_dict = None
        if country:
            filter_dict = {"must": [{"key": "country", "match": {"value": country}}]}

        search_results = qdrant_service.search(
            query_vector=query_vector,
            limit=top_k,
            filter_dict=filter_dict,
        )

        retrieval_time = int((time.time() - retrieval_start) * 1000)

        # Форматируем контекст для LLM
        contexts = [
            {
                "content": r["payload"].get("content", ""),
                "url": r["payload"].get("url", ""),
            }
            for r in search_results
        ]

        # Отправляем источники
        sources = _format_sources(search_results)
        yield f"data: {json.dumps({'type': 'sources', 'data': [s.model_dump() for s in sources]})}\n\n"

        # Генерируем ответ через LLM
        llm_client = get_llm_client()
        generation_start = time.time()

        full_answer = ""
        async for chunk in llm_client.generate_stream(question, contexts):
            full_answer += chunk
            yield f"data: {json.dumps({'type': 'chunk', 'data': chunk})}\n\n"

        generation_time = int((time.time() - generation_start) * 1000)

        # Отправляем метаданные
        metadata = {
            "type": "metadata",
            "data": {
                "retrieval_time_ms": retrieval_time,
                "generation_time_ms": generation_time,
                "model": llm_client.model_name,
            },
        }
        yield f"data: {json.dumps(metadata)}\n\n"

        # Завершающее событие
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except QdrantError as e:
        logger.error(f"Ошибка поиска: {e}")
        yield f"data: {json.dumps({'type': 'error', 'error': 'QdrantError', 'message': str(e)})}\n\n"
    except LLMError as e:
        logger.error(f"Ошибка LLM: {e}")
        yield f"data: {json.dumps({'type': 'error', 'error': 'LLMError', 'message': str(e)})}\n\n"
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        yield f"data: {json.dumps({'type': 'error', 'error': 'InternalError', 'message': str(e)})}\n\n"


@router.post("/query", response_model=QueryResponse, responses={500: {"model": ErrorResponse}})
async def query_rag(request: QueryRequest) -> QueryResponse | StreamingResponse:
    """
    Обработать вопрос пользователя с использованием RAG.

    Эндпоинт выполняет семантический поиск по документации и генерирует
    ответ на основе найденных контекстов с использованием LLM.

    Args:
        request: Запрос с вопросом и параметрами.

    Returns:
        QueryResponse с ответом, источниками и метаданными,
        или StreamingResponse для потоковой передачи.

    Raises:
        HTTPException: Если произошла ошибка при обработке запроса.
    """
    logger.info(f"RAG запрос: вопрос='{request.question[:50]}...', top_k={request.top_k}")

    retrieval_start = time.time()

    try:
        # Инициализация сервисов
        qdrant_service = get_qdrant_service()
        embedder = get_embedding_service()

        # Генерируем эмбеддинг запроса
        query_vector = embedder.encode_query(request.question)

        # Выполняем поиск
        filter_dict = None
        if request.country:
            filter_dict = {"must": [{"key": "country", "match": {"value": request.country}}]}

        search_results = qdrant_service.search(
            query_vector=query_vector,
            limit=request.top_k,
            filter_dict=filter_dict,
        )

        retrieval_time = int((time.time() - retrieval_start) * 1000)
        logger.info(
            f"Поиск завершен за {retrieval_time}мс, найдено {len(search_results)} результатов"
        )

        if not search_results:
            raise SearchError("Не найдено релевантных документов")

        # Форматируем контекст для LLM
        contexts = [
            {
                "content": r["payload"].get("content", ""),
                "url": r["payload"].get("url", ""),
            }
            for r in search_results
        ]

        # Потоковая передача
        if request.stream:
            return StreamingResponse(
                _generate_sse_response(request.question, request.top_k, request.country),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Обычный ответ
        generation_start = time.time()
        llm_client = get_llm_client()
        answer = await llm_client.generate(request.question, contexts)
        generation_time = int((time.time() - generation_start) * 1000)

        # Форматируем источники
        sources = _format_sources(search_results)

        logger.info(f"Ответ сгенерирован за {generation_time}мс")

        return QueryResponse(
            answer=answer,
            sources=sources,
            metadata=ResponseMetadata(
                retrieval_time_ms=retrieval_time,
                generation_time_ms=generation_time,
                model=llm_client.model_name,
            ),
        )

    except QdrantError as e:
        logger.error(f"Ошибка Qdrant: {e}")
        raise HTTPException(status_code=503, detail=e.to_dict()) from e
    except EmbeddingError as e:
        logger.error(f"Ошибка эмбеддинга: {e}")
        raise HTTPException(status_code=503, detail=e.to_dict()) from e
    except LLMError as e:
        logger.error(f"Ошибка LLM: {e}")
        raise HTTPException(status_code=503, detail=e.to_dict()) from e
    except SearchError as e:
        logger.error(f"Ошибка поиска: {e}")
        raise HTTPException(status_code=404, detail=e.to_dict()) from e
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "InternalError", "message": str(e)},
        ) from e
