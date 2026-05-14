"""
FastAPI приложение для RAG системы.

Основной модуль приложения, создающий экземпляр FastAPI
и регистрирующий все маршруты.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import chat, health, index, models, query
from src.core.config import settings
from src.services.embedding_service import get_embedding_service
from src.services.llm_client import get_llm_client
from src.services.qdrant_service import get_qdrant_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Контекстный менеджер жизненного цикла приложения.

    Выполняет инициализацию при запуске и очистку при остановке.

    Args:
        app: Экземпляр FastAPI приложения.

    Yields:
        None
    """
    # Инициализация при запуске
    logger.info("RAG API: запуск приложения")
    logger.info("RAG API: инициализация сервисов...")

    try:
        # Инициализация сервисов
        qdrant_service = get_qdrant_service()
        qdrant_service.connect()
        logger.info("Qdrant сервис инициализирован")

        embedder = get_embedding_service()
        logger.info(f"Embedding сервис инициализирован: {settings.embedding_model}")

        llm_client = get_llm_client()
        logger.info(f"LLM клиент инициализирован: {settings.llm_model}")

        yield

    finally:
        # Очистка при остановке
        logger.info("RAG API: остановка приложения")

        # Закрываем подключения
        try:
            qdrant_service.close()
        except Exception as e:
            logger.error(f"Ошибка закрытия Qdrant подключения: {e}")

        try:
            await llm_client.close()
        except Exception as e:
            logger.error(f"Ошибка закрытия LLM клиента: {e}")


def create_app() -> FastAPI:
    """
    Создать и настроить экземпляр FastAPI приложения.

    Returns:
        Настроенный экземпляр FastAPI.
    """
    app = FastAPI(
        title="MkDocs RAG API",
        description="API для семантического поиска по документации MkDocs с использованием RAG",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Настройка CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Регистрация роутеров
    app.include_router(query.router, prefix="/api/v1", tags=["query"])
    app.include_router(index.router, prefix="/api/v1", tags=["index"])
    app.include_router(models.router, prefix="/api/v1", tags=["models"])
    app.include_router(chat.router, prefix="", tags=["chat"])
    app.include_router(health.router, prefix="", tags=["health"])

    @app.get("/")
    async def root() -> dict[str, str]:
        """
        Корневой эндпоинт.

        Returns:
            Приветственное сообщение.
        """
        return {
            "message": "Добро пожаловать в MkDocs RAG API",
            "docs": "/docs",
            "health": "/health",
            "query": "/api/v1/query",
            "index": "/api/v1/index",
            "models": "/api/v1/models",
        }

    return app


# Создаем экземпляр приложения
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
