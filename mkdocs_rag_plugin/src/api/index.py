"""
Эндпоинты для индексации документов.

Модуль содержит маршруты для управления процессом индексации:
запуск, отслеживание статуса и очистка индекса.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


def _split_into_chunks(text: str, chunk_size: int = 500) -> list[str]:
    """
    Разбить текст на чанки по предложениям.

    Args:
        text: Исходный текст.
        chunk_size: Максимальный размер чанка в символах.

    Returns:
        Список чанков.
    """
    import re

    # Разбиваем на предложения
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# Глобальное состояние индексации (в production использовать Redis)
_indexing_state: dict[str, Any] = {
    "status": "idle",  # idle, running, completed, failed
    "progress": 0,  # 0-100
    "total_documents": 0,
    "processed_documents": 0,
    "error_message": None,
    "started_at": None,
    "completed_at": None,
}


class IndexRequest(BaseModel):
    """
    Модель запроса на индексацию.

    Атрибуты:
        docs_path: Путь к документам для индексации.
        force: Принудительная переиндексация.
    """

    docs_path: str = Field(default="docs", description="Путь к документам")
    force: bool = Field(default=False, description="Принудительная переиндексация")


class IndexStatusResponse(BaseModel):
    """
    Модель статуса индексации.

    Атрибуты:
        status: Текущий статус (idle, running, completed, failed).
        progress: Прогресс в процентах (0-100).
        total_documents: Общее количество документов.
        processed_documents: Количество обработанных документов.
        error_message: Сообщение об ошибке (если есть).
        started_at: Время начала индексации.
        completed_at: Время завершения индексации.
    """

    status: str
    progress: int
    total_documents: int
    processed_documents: int
    error_message: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


class IndexResponse(BaseModel):
    """
    Модель ответа после запуска индексации.

    Атрибуты:
        message: Сообщение о запуске.
        task_id: Идентификатор задачи (опционально).
    """

    message: str
    task_id: str | None = None


class ClearIndexResponse(BaseModel):
    """
    Модель ответа после очистки индекса.

    Атрибуты:
        message: Сообщение о результате.
        deleted_count: Количество удаленных документов.
    """

    message: str
    deleted_count: int


async def _run_indexing(docs_path: str, force: bool) -> None:
    """
    Выполнить процесс индексации в фоне.

    Args:
        docs_path: Путь к документам.
        force: Принудительная переиндексация.
    """
    global _indexing_state

    try:
        _indexing_state["status"] = "running"
        _indexing_state["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _indexing_state["error_message"] = None

        logger.info(f"Запуск индексации: docs_path={docs_path}, force={force}")

        # Реальная логика индексации
        import os

        from src.services.indexer import ProcessedDocument, get_indexer

        indexer = get_indexer()

        # Загружаем документы из указанной директории
        documents = []
        docs_dir = os.path.abspath(docs_path)

        if not os.path.exists(docs_dir):
            raise FileNotFoundError(f"Директория не существует: {docs_dir}")

        # Рекурсивно обходим все файлы
        for root, dirs, files in os.walk(docs_dir):
            for file in files:
                if file.endswith((".md", ".txt", ".rst")):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            content = f.read()

                        # Разбиваем на чанки
                        chunks = _split_into_chunks(content, chunk_size=500)

                        for i, chunk in enumerate(chunks):
                            relative_path = os.path.relpath(file_path, docs_dir)
                            doc = ProcessedDocument(
                                url=f"/{relative_path}",
                                title=file,
                                content=chunk,
                                header_path=relative_path,
                                source_file=file_path,
                                chunk_id=f"{file_path}_{i}",
                            )
                            documents.append(doc)

                    except Exception as e:
                        logger.warning(f"Ошибка чтения файла {file_path}: {e}")

        _indexing_state["total_documents"] = len(documents)
        _indexing_state["processed_documents"] = 0

        # Индексируем документы батчами
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            result = await indexer.index_documents(batch, create_collection_if_not_exists=(i == 0))

            _indexing_state["processed_documents"] += result.indexed_documents
            _indexing_state["progress"] = int(
                (_indexing_state["processed_documents"] / _indexing_state["total_documents"]) * 100
            )

            logger.info(f"Прогресс индексации: {_indexing_state['progress']}%")

        _indexing_state["status"] = "completed"
        _indexing_state["progress"] = 100
        _indexing_state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        logger.info("Индексация успешно завершена")

    except Exception as e:
        logger.error(f"Ошибка индексации: {e}")
        _indexing_state["status"] = "failed"
        _indexing_state["error_message"] = str(e)
        _indexing_state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@router.post("/index", response_model=IndexResponse)
async def start_indexing(
    request: IndexRequest,
    background_tasks: BackgroundTasks,
) -> IndexResponse:
    """
    Запустить процесс индексации документов.

    Эндпоинт инициирует фоновую задачу по обработке и индексации
    документов из указанной директории в векторную базу Qdrant.

    Args:
        request: Параметры индексации.
        background_tasks: Менеджер фоновых задач FastAPI.

    Returns:
        IndexResponse с информацией о запущенной задаче.

    Raises:
        HTTPException: Если индексация уже выполняется.
    """
    global _indexing_state

    if _indexing_state["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail={"error": "ConflictError", "message": "Индексация уже выполняется"},
        )

    logger.info(f"Запрос на индексацию: {request}")

    # Сбрасываем состояние
    _indexing_state = {
        "status": "pending",
        "progress": 0,
        "total_documents": 0,
        "processed_documents": 0,
        "error_message": None,
        "started_at": None,
        "completed_at": None,
    }

    # Запускаем в фоне
    background_tasks.add_task(_run_indexing, request.docs_path, request.force)

    return IndexResponse(
        message="Индексация запущена",
        task_id=None,
    )


@router.get("/index/status", response_model=IndexStatusResponse)
async def get_indexing_status() -> IndexStatusResponse:
    """
    Получить текущий статус индексации.

    Returns:
        IndexStatusResponse с информацией о прогрессе.
    """
    global _indexing_state

    return IndexStatusResponse(
        status=_indexing_state["status"],
        progress=_indexing_state["progress"],
        total_documents=_indexing_state["total_documents"],
        processed_documents=_indexing_state["processed_documents"],
        error_message=_indexing_state["error_message"],
        started_at=_indexing_state["started_at"],
        completed_at=_indexing_state["completed_at"],
    )


@router.delete("/index", response_model=ClearIndexResponse)
async def clear_index() -> ClearIndexResponse:
    """
    Очистить индекс документов.

    Удаляет все документы из коллекции Qdrant.

    Returns:
        ClearIndexResponse с информацией о результате.

    Raises:
        HTTPException: Если произошла ошибка при очистке.
    """
    global _indexing_state

    if _indexing_state["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "ConflictError",
                "message": "Нельзя очистить индекс во время индексации",
            },
        )

    try:
        from src.services.qdrant_service import get_qdrant_service

        qdrant = get_qdrant_service()

        # Получаем статистику перед удалением
        try:
            collection_info = qdrant.client.get_collection(qdrant.collection_name)
            deleted_count = collection_info.points_count
        except Exception:
            deleted_count = 0

        # Очищаем коллекцию
        qdrant.delete_collection()

        # Сбрасываем состояние
        _indexing_state = {
            "status": "idle",
            "progress": 0,
            "total_documents": 0,
            "processed_documents": 0,
            "error_message": None,
            "started_at": None,
            "completed_at": None,
        }

        logger.info(f"Индекс очищен, удалено документов: {deleted_count}")

        return ClearIndexResponse(
            message="Индекс успешно очищен",
            deleted_count=deleted_count,
        )

    except Exception as e:
        logger.error(f"Ошибка очистки индекса: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "InternalError", "message": str(e)},
        ) from e
