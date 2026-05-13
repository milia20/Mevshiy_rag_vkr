"""
Эндпоинты для проверки здоровья сервиса.

Модуль содержит маршруты для мониторинга состояния приложения.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter

from src.services.qdrant_service import get_qdrant_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """
    Проверить состояние сервиса.

    Returns:
        Словарь со статусом сервиса.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "mkdocs-rag-api",
        "version": "0.1.0",
    }


@router.get("/ready")
async def readiness_check() -> dict[str, Any]:
    """
    Проверить готовность сервиса к обработке запросов.

    Returns:
        Словарь со статусом готовности.
    """
    checks = {
        "qdrant": False,
        "status": "not_ready",
    }

    try:
        qdrant = get_qdrant_service()
        qdrant.connect()
        checks["qdrant"] = True
        logger.info("Qdrant подключение успешно")
    except Exception as e:
        logger.error(f"Ошибка подключения к Qdrant: {e}")

    checks["status"] = "ready" if checks["qdrant"] else "not_ready"
    return checks


@router.get("/live")
async def liveness_check() -> dict[str, str]:
    """
    Проверить жизнеспособность сервиса (liveness probe).

    Returns:
        Словарь со статусом жизнеспособности.
    """
    return {"status": "alive"}
