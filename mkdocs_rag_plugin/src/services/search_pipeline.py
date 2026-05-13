"""
Поисковый конвейер для RAG системы.

Модуль предоставляет класс для оркестрации всего процесса поиска:
- Предобработка запроса
- Выбор стратегии поиска
- Re-ranking
- Форматирование контекста
- Сбор метаданных
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from src.core.config import settings
from src.services.embedding_service import get_embedding_service
from src.services.retriever import ScoredChunk, get_retriever

logger = logging.getLogger(__name__)


@dataclass
class SearchContext:
    """
    Контекст для генерации ответа.

    Атрибуты:
        content: Содержимое чанка.
        url: URL источника.
        title: Заголовок документа.
        score: Оценка релевантности.
        metadata: Дополнительные метаданные.
    """

    content: str
    url: str
    title: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def format_for_llm(self) -> str:
        """
        Форматировать контекст для LLM.

        Returns:
            Строка с форматированным контекстом.
        """
        return f"[Источник: {self.title}]({self.url})\n{self.content}"


@dataclass
class SearchResult:
    """
    Результат поискового конвейера.

    Атрибуты:
        query: Исходный запрос.
        contexts: Список контекстов для LLM.
        chunks: Оригинальные чанки.
        retrieval_time_ms: Время поиска в миллисекундах.
        rerank_time_ms: Время re-ranking в миллисекундах.
        total_time_ms: Общее время выполнения.
        strategy: Использованная стратегия поиска.
        metadata: Дополнительные метаданные.
    """

    query: str
    contexts: list[SearchContext]
    chunks: list[ScoredChunk]
    retrieval_time_ms: int
    rerank_time_ms: int = 0
    total_time_ms: int = 0
    strategy: str = "dense"
    metadata: dict[str, Any] = field(default_factory=dict)


class SearchPipeline:
    """
    Конвейер для обработки поисковых запросов.

    Оркестрирует:
    - Предобработку запроса (нормализация, токенизация)
    - Выбор стратегии поиска (dense/sparse/hybrid)
    - Re-ranking результатов
    - Форматирование контекста для LLM
    - Сбор метаданных для логирования

    Атрибуты:
        collection_name: Имя коллекции для поиска.
        default_strategy: Стратегия поиска по умолчанию.
        use_reranking: Использовать ли re-ranking.
        rerank_top_k: Количество результатов после re-ranking.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        default_strategy: str = "hybrid",
        use_reranking: bool = False,
        rerank_top_k: int = 5,
    ) -> None:
        """
        Инициализация поискового конвейера.

        Args:
            collection_name: Имя коллекции. Если не указано, используется из настроек.
            default_strategy: Стратегия поиска по умолчанию (dense/sparse/hybrid).
            use_reranking: Использовать ли re-ranking.
            rerank_top_k: Количество результатов после re-ranking.
        """
        self.collection_name = collection_name or settings.qdrant_collection
        self.default_strategy = default_strategy
        self.use_reranking = use_reranking
        self.rerank_top_k = rerank_top_k

        self._retriever = get_retriever(collection_name=self.collection_name)
        self._embedder = get_embedding_service()

        logger.info(
            f"Инициализация SearchPipeline: collection={self.collection_name}, "
            f"strategy={default_strategy}, reranking={use_reranking}"
        )

    async def search(
        self,
        query: str,
        top_k: int = 10,
        strategy: str | None = None,
        filter_dict: dict[str, Any] | None = None,
        use_reranking: bool | None = None,
    ) -> SearchResult:
        """
        Выполнить полный цикл поиска.

        Args:
            query: Поисковый запрос.
            top_k: Количество результатов.
            strategy: Стратегия поиска (переопределяет default).
            filter_dict: Фильтр для поиска.
            use_reranking: Переопределение настройки re-ranking.

        Returns:
            SearchResult с контекстами и метаданными.

        Raises:
            ValueError: Если стратегия поиска некорректна.
        """
        start_time = time.time()
        strategy = strategy or self.default_strategy
        use_rerank = use_reranking if use_reranking is not None else self.use_reranking

        logger.info(f"Поисковый запрос: '{query[:50]}...', strategy={strategy}")

        # Предобработка запроса
        processed_query = self._preprocess_query(query)

        # Генерируем эмбеддинг
        retrieval_start = time.time()
        query_vector = self._embedder.encode_query(processed_query)

        # Выбираем стратегию поиска
        if strategy == "dense":
            chunks = await self._retriever.dense_search(
                query_vector=query_vector,
                top_k=top_k,
                filter_dict=filter_dict,
            )
        elif strategy == "sparse":
            chunks = await self._retriever.sparse_search(
                query_text=processed_query,
                top_k=top_k,
            )
        elif strategy == "hybrid":
            chunks = await self._retriever.hybrid_search(
                query_vector=query_vector,
                query_text=processed_query,
                top_k=top_k,
                filter_dict=filter_dict,
            )
        else:
            msg = f"Некорректная стратегия поиска: {strategy}"
            raise ValueError(msg)

        retrieval_time = int((time.time() - retrieval_start) * 1000)

        # Re-ranking если включен
        rerank_time = 0
        if use_rerank and chunks:
            rerank_start = time.time()
            chunks = await self._retriever.rerank(
                query=processed_query,
                chunks=chunks,
                top_k=min(top_k, self.rerank_top_k),
            )
            rerank_time = int((time.time() - rerank_start) * 1000)

        # Форматируем контексты
        contexts = [
            SearchContext(
                content=chunk.content,
                url=chunk.url,
                title=chunk.title,
                score=chunk.score,
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ]

        total_time = int((time.time() - start_time) * 1000)

        result = SearchResult(
            query=query,
            contexts=contexts,
            chunks=chunks,
            retrieval_time_ms=retrieval_time,
            rerank_time_ms=rerank_time,
            total_time_ms=total_time,
            strategy=strategy,
            metadata={
                "original_query": query,
                "processed_query": processed_query,
                "filter_used": filter_dict is not None,
            },
        )

        logger.info(
            f"Поиск завершен: {len(contexts)} результатов, "
            f"retrieval={retrieval_time}мс, rerank={rerank_time}мс, total={total_time}мс"
        )

        return result

    def _preprocess_query(self, query: str) -> str:
        """
        Предобработать поисковый запрос.

        Выполняет:
        - Приведение к нижнему регистру
        - Удаление лишней пунктуации
        - Нормализацию пробелов

        Args:
            query: Исходный запрос.

        Returns:
            Обработанный запрос.
        """
        # Сохраняем регистр для кириллицы (важно для некоторых моделей)
        # Удаляем только лишнюю пунктуацию
        processed = re.sub(r"[^\w\s\-\?]", " ", query, flags=re.UNICODE)

        # Нормализуем пробелы
        processed = " ".join(processed.split())

        logger.debug(f"Предобработка запроса: '{query}' -> '{processed}'")

        return processed

    def format_contexts(
        self,
        contexts: list[SearchContext],
        max_tokens: int | None = None,
    ) -> str:
        """
        Форматировать контексты для передачи в LLM.

        Args:
            contexts: Список контекстов.
            max_tokens: Максимальное количество токенов (опционально).

        Returns:
            Строка с форматированными контекстами.
        """
        if not contexts:
            return ""

        formatted = []
        total_length = 0

        for i, ctx in enumerate(contexts, start=1):
            ctx_str = ctx.format_for_llm()
            total_length += len(ctx_str)

            # Простая оценка токенов (1 токен ≈ 4 символа)
            if max_tokens and total_length > max_tokens * 4:
                logger.info(f"Достигнут лимит токенов: {max_tokens}")
                break

            formatted.append(f"=== Контекст {i} ===\n{ctx_str}")

        result = "\n\n".join(formatted)
        logger.debug(f"Форматировано {len(formatted)} контекстов, длина={len(result)}")

        return result

    def get_metadata(self, result: SearchResult) -> dict[str, Any]:
        """
        Извлечь метаданные из результата поиска.

        Args:
            result: Результат поиска.

        Returns:
            Словарь с метаданными для логирования/оценки.
        """
        return {
            "query": result.query,
            "num_results": len(result.contexts),
            "retrieval_time_ms": result.retrieval_time_ms,
            "rerank_time_ms": result.rerank_time_ms,
            "total_time_ms": result.total_time_ms,
            "strategy": result.strategy,
            "scores": [ctx.score for ctx in result.contexts],
            "urls": [ctx.url for ctx in result.contexts],
            "chunk_ids": [chunk.chunk_id for chunk in result.chunks],
        }


# Singleton instance
_search_pipeline: SearchPipeline | None = None


def get_search_pipeline(
    collection_name: str | None = None,
    default_strategy: str = "hybrid",
    use_reranking: bool = False,
    rerank_top_k: int = 5,
) -> SearchPipeline:
    """
    Получить экземпляр поискового конвейера (singleton).

    Args:
        collection_name: Имя коллекции.
        default_strategy: Стратегия поиска по умолчанию.
        use_reranking: Использовать ли re-ranking.
        rerank_top_k: Количество результатов после re-ranking.

    Returns:
        Экземпляр SearchPipeline.
    """
    global _search_pipeline
    if _search_pipeline is None:
        _search_pipeline = SearchPipeline(
            collection_name=collection_name,
            default_strategy=default_strategy,
            use_reranking=use_reranking,
            rerank_top_k=rerank_top_k,
        )
    return _search_pipeline
