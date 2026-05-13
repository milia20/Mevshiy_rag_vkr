"""
Эндпоинты для чата.

Модуль содержит маршруты для взаимодействия с RAG системой через чат.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel

from src.services.llm_client import get_llm_client
from src.services.search_pipeline import get_search_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatMessage(BaseModel):
    """
    Модель сообщения чата.

    Атрибуты:
        role: Роль отправителя (user, assistant, system).
        content: Содержание сообщения.
    """

    role: str
    content: str


class ChatRequest(BaseModel):
    """
    Модель запроса к чату.

    Атрибуты:
        messages: Список сообщений в диалоге.
    """

    messages: list[ChatMessage]


@router.post("/chat")
async def chat_with_docs(
    history: str = Form(default="[]", description="История чата в формате JSON"),
    files: list[UploadFile] | None = File(default=None, description="Файлы для анализа"),
) -> dict[str, Any]:
    """
    Получить ответ от RAG системы на основе истории чата и файлов.

    Args:
        history: История чата в формате JSON.
        files: Опциональные файлы для анализа (.txt, .md).

    Returns:
        Словарь с ответом системы.
    """
    logger.info(f"Запрос к чату, история: {history}, файлов: {len(files) if files else 0}")

    try:
        # Парсим историю чата
        chat_history = json.loads(history) if history else []

        # Получаем последнее сообщение пользователя
        last_message = ""
        if chat_history:
            last_message = (
                chat_history[-1].get("content", "") if isinstance(chat_history[-1], dict) else ""
            )

        # Обрабатываем файлы если есть
        file_contents = []
        if files:
            for file in files:
                try:
                    content = await file.read()
                    file_contents.append(content.decode("utf-8"))
                    logger.info(f"Обработан файл: {file.filename}")
                except Exception as e:
                    logger.error(f"Ошибка чтения файла {file.filename}: {e}")

        # Поиск релевантного контекста
        search_pipeline = get_search_pipeline()
        contexts = []
        sources = []

        if last_message:
            search_result = await search_pipeline.search(query=last_message, top_k=5)
            contexts = [
                {"content": ctx.content, "url": ctx.url, "title": ctx.title}
                for ctx in search_result.contexts
            ]
            sources = [
                {"url": ctx.url, "title": ctx.title, "score": ctx.score}
                for ctx in search_result.contexts
            ]

        # Генерация ответа через LLM
        llm_client = get_llm_client()
        response = await llm_client.generate(query=last_message, contexts=contexts)

        return {
            "response": response,
            "sources": sources,
        }

    except Exception as e:
        logger.error(f"Ошибка в чате: {e}")
        return {
            "response": f"Произошла ошибка: {str(e)}",
            "sources": [],
        }


@router.post("/chat/message")
async def send_message(request: ChatRequest) -> dict[str, Any]:
    """
    Отправить сообщение в чат и получить ответ.

    Args:
        request: Запрос с историей сообщений.

    Returns:
        Словарь с ответом ассистента.
    """
    logger.info(f"Сообщение в чат: {len(request.messages)} сообщений")

    try:
        # Получаем последнее сообщение пользователя
        last_message = ""
        if request.messages:
            last_message = request.messages[-1].content

        # Поиск релевантного контекста
        search_pipeline = get_search_pipeline()
        contexts = []
        sources = []

        if last_message:
            search_result = await search_pipeline.search(query=last_message, top_k=5)
            contexts = [
                {"content": ctx.content, "url": ctx.url, "title": ctx.title}
                for ctx in search_result.contexts
            ]
            sources = [
                {"url": ctx.url, "title": ctx.title, "score": ctx.score}
                for ctx in search_result.contexts
            ]

        # Генерация ответа через LLM
        llm_client = get_llm_client()
        response = await llm_client.generate(query=last_message, contexts=contexts)

        return {
            "role": "assistant",
            "content": response,
            "sources": sources,
        }

    except Exception as e:
        logger.error(f"Ошибка в отправке сообщения: {e}")
        return {
            "role": "assistant",
            "content": f"Произошла ошибка: {str(e)}",
            "sources": [],
        }
