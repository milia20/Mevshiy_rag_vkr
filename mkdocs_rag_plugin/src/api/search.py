"""
Эндпоинты для поиска.

Модуль содержит маршруты для семантического поиска по документации.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query

from src.services.search_pipeline import get_search_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/search")
async def search_documents(
    q: str = Query(..., min_length=1, description="Поисковый запрос"),
    limit: int = Query(default=10, ge=1, le=100, description="Максимальное количество результатов"),
) -> dict[str, Any]:
    """
    Выполнить семантический поиск по документации.

    Args:
        q: Поисковый запрос.
        limit: Максимальное количество результатов для возврата.

    Returns:
        Словарь с результатами поиска.

    Raises:
        HTTPException: Если произошла ошибка при поиске.
    """
    logger.info(f"Поисковый запрос: {q}, limit: {limit}")

    try:
        search_pipeline = get_search_pipeline()
        search_result = await search_pipeline.search(query=q, top_k=limit)

        results = [
            {
                "content": ctx.content,
                "url": ctx.url,
                "title": ctx.title,
                "score": ctx.score,
                "metadata": ctx.metadata,
            }
            for ctx in search_result.contexts
        ]

        return {
            "query": q,
            "results": results,
            "total": len(results),
            "retrieval_time_ms": search_result.retrieval_time_ms,
            "total_time_ms": search_result.total_time_ms,
            "strategy": search_result.strategy,
        }

    except Exception as e:
        logger.error(f"Ошибка поиска: {e}")
        from fastapi import HTTPException

        raise HTTPException(
            status_code=500,
            detail={"error": "SearchError", "message": str(e)},
        ) from e


@router.post("/search")
async def search_documents_post(
    query: str,
    limit: int = 10,
) -> dict[str, Any]:
    """
    Выполнить семантический поиск через POST запрос.

    Альтернативный эндпоинт для поиска, поддерживающий более сложные запросы.

    Args:
        query: Поисковый запрос.
        limit: Максимальное количество результатов.

    Returns:
        Словарь с результатами поиска.
    """
    logger.info(f"POST поисковый запрос: {query}, limit: {limit}")

    try:
        search_pipeline = get_search_pipeline()
        search_result = await search_pipeline.search(query=query, top_k=limit)

        results = [
            {
                "content": ctx.content,
                "url": ctx.url,
                "title": ctx.title,
                "score": ctx.score,
                "metadata": ctx.metadata,
            }
            for ctx in search_result.contexts
        ]

        return {
            "query": query,
            "results": results,
            "total": len(results),
            "retrieval_time_ms": search_result.retrieval_time_ms,
            "total_time_ms": search_result.total_time_ms,
            "strategy": search_result.strategy,
        }

    except Exception as e:
        logger.error(f"Ошибка поиска: {e}")
        from fastapi import HTTPException

        raise HTTPException(
            status_code=500,
            detail={"error": "SearchError", "message": str(e)},
        ) from e
