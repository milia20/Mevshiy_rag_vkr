"""
Конфигурация плагина MkDocs RAG.

Модуль определяет схему конфигурации плагина с использованием Pydantic v2
и mkdocs.config.config_options для валидации параметров.
"""

from __future__ import annotations

import logging
from typing import Any

from mkdocs.config import Config, config_options
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("mkdocs.plugins.rag_plugin")


class RAGPluginConfig(BaseModel):
    """
    Схема конфигурации плагина RAG.

    Атрибуты:
        enabled: Включен ли плагин.
        qdrant_url: URL подключения к Qdrant серверу.
        collection_name: Имя коллекции в Qdrant для хранения эмбеддингов.
        embedding_model: Название модели для генерации эмбеддингов.
        chunk_strategy: Стратегия чанкования (structural, fixed, recursive).
        chunk_size: Размер чанка в токенах/символах.
        chunk_overlap: Перекрытие между чанками.
        api_host: Хост для FastAPI сервера.
        api_port: Порт для FastAPI сервера.
        enable_search_ui: Включить ли поисковый UI.
    """

    enabled: bool = Field(
        default=True,
        description="Включен ли плагин",
    )
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="URL подключения к Qdrant серверу",
    )
    collection_name: str = Field(
        default="mkdocs_docs",
        description="Имя коллекции в Qdrant для хранения эмбеддингов",
    )
    embedding_model: str = Field(
        default="gigachat-embeddings",
        description="Название модели для генерации эмбеддингов",
    )
    chunk_strategy: str = Field(
        default="structural",
        description="Стратегия чанкования (structural, fixed, recursive)",
    )
    chunk_size: int = Field(
        default=500,
        ge=100,
        description="Размер чанка в токенах/символах",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Перекрытие между чанками",
    )
    api_host: str = Field(
        default="0.0.0.0",
        description="Хост для FastAPI сервера",
    )
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Порт для FastAPI сервера",
    )
    enable_search_ui: bool = Field(
        default=True,
        description="Включить ли поисковый UI",
    )
    llm_provider: str = Field(
        default="openai",
        description="Провайдер LLM (openai, lmstudio, etc.)",
    )
    llm_model: str = Field(
        default="qwen/qwen3.5-9b",
        description="Название модели LLM",
    )
    enable_chat_panel: bool = Field(
        default=False,
        description="Включить ли чат-панель",
    )
    chatbot_url: str = Field(
        default="http://localhost:8000",
        description="URL чат-бот API сервера",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "title": "RAG Plugin Configuration",
            "description": "Конфигурация плагина RAG для MkDocs",
        },
    )


class PluginConfig(Config):
    """
    Конфигурация плагина MkDocs с использованием config_options.

    Этот класс используется MkDocs для валидации конфигурации плагина.
    """

    enabled = config_options.Type(bool, default=True)
    qdrant_url = config_options.Type(str, default="http://localhost:6333")
    collection_name = config_options.Type(str, default="mkdocs_docs")
    embedding_model = config_options.Type(str, default="gigachat-embeddings")
    chunk_strategy = config_options.Choice(
        choices=["structural", "fixed", "recursive"],
        default="structural",
    )
    chunk_size = config_options.Type(int, default=500)
    chunk_overlap = config_options.Type(int, default=50)
    api_host = config_options.Type(str, default="0.0.0.0")
    api_port = config_options.Type(int, default=8000)
    enable_search_ui = config_options.Type(bool, default=True)
    llm_provider = config_options.Type(str, default="openai")
    llm_model = config_options.Type(str, default="qwen/qwen3.5-9b")
    enable_chat_panel = config_options.Type(bool, default=False)
    chatbot_url = config_options.Type(str, default="http://localhost:8000")


def get_config_schema() -> tuple[tuple[str, Any], ...]:
    """
    Получить схему конфигурации для использования в плагине.

    Returns:
        Кортеж кортежей с именем опции и её конфигурацией.
    """
    return (
        ("enabled", config_options.Type(bool, default=True)),
        ("qdrant_url", config_options.Type(str, default="http://localhost:6333")),
        ("collection_name", config_options.Type(str, default="mkdocs_docs")),
        ("embedding_model", config_options.Type(str, default="gigachat-embeddings")),
        (
            "chunk_strategy",
            config_options.Choice(
                choices=["structural", "fixed", "recursive"],
                default="structural",
            ),
        ),
        ("chunk_size", config_options.Type(int, default=500)),
        ("chunk_overlap", config_options.Type(int, default=50)),
        ("api_host", config_options.Type(str, default="0.0.0.0")),
        ("api_port", config_options.Type(int, default=8000)),
        ("enable_search_ui", config_options.Type(bool, default=True)),
        ("llm_provider", config_options.Type(str, default="openai")),
        ("llm_model", config_options.Type(str, default="qwen/qwen3.5-9b")),
        ("enable_chat_panel", config_options.Type(bool, default=False)),
        ("chatbot_url", config_options.Type(str, default="http://localhost:8000")),
    )
