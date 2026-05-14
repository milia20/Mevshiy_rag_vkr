"""
Data models for MkDocs RAG Plugin.

Модуль содержит Pydantic модели для представления данных плагина:
- Chunk: чанк документа с метаданными
- ProcessedDocument: обработанный документ
- IndexingResult: результат индексации
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """
    Модель чанка документа.

    Атрибуты:
        id: Уникальный идентификатор чанка.
        content: Текст чанка.
        metadata: Метаданные чанка (url, title, header_path, source_file).
        embeddings: Векторные эмбеддинги (опционально).
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(..., description="Текст чанка")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Метаданные чанка",
    )
    embeddings: list[float] | None = Field(
        default=None,
        description="Векторные эмбеддинги",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "title": "Chunk",
            "description": "Чанк документа с метаданными",
        },
    )

    @property
    def url(self) -> str:
        """Получить URL из метаданных."""
        return self.metadata.get("url", "")

    @property
    def title(self) -> str:
        """Получить заголовок из метаданных."""
        return self.metadata.get("title", "")

    @property
    def header_path(self) -> list[str]:
        """Получить путь заголовков из метаданных."""
        return self.metadata.get("header_path", [])

    @property
    def source_file(self) -> str:
        """Получить исходный файл из метаданных."""
        return self.metadata.get("source_file", "")


class ProcessedDocument(BaseModel):
    """
    Модель обработанного документа.

    Атрибуты:
        url: URL документа.
        title: Заголовок документа.
        content: Полное содержимое документа.
        chunks: Список чанков документа.
        metadata: Дополнительные метаданные.
    """

    url: str = Field(..., description="URL документа")
    title: str = Field(..., description="Заголовок документа")
    content: str = Field(..., description="Полное содержимое документа")
    chunks: list[Chunk] = Field(
        default_factory=list,
        description="Список чанков документа",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Дополнительные метаданные",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "title": "ProcessedDocument",
            "description": "Обработанный документ с чанками",
        },
    )

    @property
    def total_chunks(self) -> int:
        """Получить общее количество чанков."""
        return len(self.chunks)

    def add_chunk(self, chunk: Chunk) -> None:
        """Добавить чанк к документу."""
        self.chunks.append(chunk)

    def get_all_content(self) -> str:
        """Получить весь текст из всех чанков."""
        return "\n\n".join(chunk.content for chunk in self.chunks)


class IndexingResult(BaseModel):
    """
    Модель результата индексации.

    Атрибуты:
        total_docs: Общее количество документов.
        total_chunks: Общее количество чанков.
        indexed_count: Количество успешно проиндексированных чанков.
        errors: Список ошибок при индексации.
        timestamp: Время завершения индексации.
    """

    total_docs: int = Field(..., description="Общее количество документов")
    total_chunks: int = Field(..., description="Общее количество чанков")
    indexed_count: int = Field(
        ...,
        description="Количество успешно проиндексированных чанков",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Список ошибок при индексации",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Время завершения индексации",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "title": "IndexingResult",
            "description": "Результат индексации документов",
        },
    )

    @property
    def success_rate(self) -> float:
        """Получить процент успешной индексации."""
        if self.total_chunks == 0:
            return 0.0
        return (self.indexed_count / self.total_chunks) * 100

    @property
    def has_errors(self) -> bool:
        """Проверить наличие ошибок."""
        return len(self.errors) > 0

    def add_error(self, error: str) -> None:
        """Добавить ошибку в список."""
        self.errors.append(error)


class DocumentMetadata(BaseModel):
    """
    Модель метаданных документа.

    Атрибуты:
        url: URL документа.
        title: Заголовок документа.
        hierarchical_path: Иерархический путь документа.
        source_file: Исходный файл.
        last_modified: Дата последнего изменения.
    """

    url: str = Field(..., description="URL документа")
    title: str = Field(..., description="Заголовок документа")
    hierarchical_path: list[str] = Field(
        default_factory=list,
        description="Иерархический путь документа",
    )
    source_file: str = Field(..., description="Исходный файл")
    last_modified: datetime | None = Field(
        default=None,
        description="Дата последнего изменения",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "title": "DocumentMetadata",
            "description": "Метаданные документа",
        },
    )


class ChunkingStrategy(BaseModel):
    """
    Модель стратегии чанкования.

    Атрибуты:
        strategy_type: Тип стратегии (structural, fixed, recursive).
        chunk_size: Размер чанка.
        chunk_overlap: Перекрытие между чанками.
        separators: Разделители для recursive стратегии.
    """

    strategy_type: str = Field(
        ...,
        description="Тип стратегии (structural, fixed, recursive)",
    )
    chunk_size: int = Field(..., description="Размер чанка")
    chunk_overlap: int = Field(default=0, description="Перекрытие между чанками")
    separators: list[str] = Field(
        default_factory=lambda: ["\n\n", "\n", ". ", " "],
        description="Разделители для recursive стратегии",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "title": "ChunkingStrategy",
            "description": "Стратегия чанкования документов",
        },
    )
