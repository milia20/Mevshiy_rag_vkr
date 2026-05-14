"""
Сервис для поиска и извлечения документов.

Модуль предоставляет класс для выполнения различных стратегий поиска:
- Dense поиск (семантический)
- Sparse поиск (BM25)
- Hybrid поиск (комбинированный)
- Re-ranking (переупорядочивание)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from src.core.config import settings
from src.core.exceptions import QdrantError
from src.services.qdrant_client import get_qdrant_service

logger = logging.getLogger(__name__)


@dataclass
class ScoredChunk:
    """
    Чанк с оценкой релевантности.

    Атрибуты:
        chunk_id: Идентификатор чанка.
        url: URL документа.
        title: Заголовок документа.
        content: Содержимое чанка.
        header_path: Путь заголовка.
        source_file: Исходный файл.
        score: Оценка релевантности.
        search_type: Тип поиска (dense/sparse/hybrid).
        metadata: Дополнительные метаданные.
    """

    chunk_id: str
    url: str
    title: str
    content: str
    header_path: str
    source_file: str
    score: float
    search_type: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_search_result(cls, result: dict[str, Any]) -> ScoredChunk:
        """
        Создать ScoredChunk из результата поиска.

        Args:
            result: Словарь с результатом поиска из Qdrant.

        Returns:
            Экземпляр ScoredChunk.
        """
        payload = result.get("payload", {})

        return cls(
            chunk_id=payload.get("chunk_id", ""),
            url=payload.get("url", ""),
            title=payload.get("title", ""),
            content=payload.get("content", ""),
            header_path=payload.get("header_path", ""),
            source_file=payload.get("source_file", ""),
            score=result.get("score", 0.0),
            search_type=result.get("search_type", "unknown"),
            metadata={
                k: v
                for k, v in payload.items()
                if k not in {"chunk_id", "url", "title", "content", "header_path", "source_file"}
            },
        )


class Retriever:
    """
    Сервис для поиска документов с различными стратегиями.

    Поддерживает:
    - Dense поиск с использованием векторных эмбеддингов
    - Sparse поиск (BM25)
    - Hybrid поиск с взвешенной фузией
    - Re-ranking с использованием cross-encoder

    Атрибуты:
        collection_name: Имя коллекции для поиска.
    """

    def __init__(self, collection_name: str | None = None) -> None:
        """
        Инициализация retriever'а.

        Args:
            collection_name: Имя коллекции. Если не указано, используется из настроек.
        """
        self.collection_name = collection_name or settings.qdrant_collection
        self._qdrant_service = get_qdrant_service()

        logger.info(f"Инициализация Retriever: collection={self.collection_name}")

    async def dense_search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """
        Выполнить плотный семантический поиск.

        Использует dense эмбеддинги для поиска семантически похожих документов.

        Args:
            query_vector: Вектор запроса.
            top_k: Количество результатов.
            filter_dict: Фильтр для поиска (опционально).

        Returns:
            Список ScoredChunk отсортированных по убыванию score.

        Raises:
            QdrantError: Если произошла ошибка при поиске.
            SearchError: Если поиск не вернул результатов.
        """
        try:
            results = await self._qdrant_service.dense_search(
                query_vector=query_vector,
                limit=top_k,
                filter_dict=filter_dict,
            )

            if not results:
                logger.warning("Dense поиск не вернул результатов")
                return []

            chunks = [ScoredChunk.from_search_result(r) for r in results]
            logger.info(f"Dense поиск: найдено {len(chunks)} результатов")

            return chunks

        except QdrantError:
            raise
        except Exception as e:
            msg = f"Ошибка dense поиска: {e}"
            logger.error(msg)
            raise QdrantError(msg, str(e)) from e

    async def sparse_search(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> list[ScoredChunk]:
        """
        Выполнить разреженный поиск (BM25).

        Использует sparse векторы для полнотекстового поиска.

        Args:
            query_text: Текстовый запрос.
            top_k: Количество результатов.

        Returns:
            Список ScoredChunk отсортированных по убыванию score.

        Raises:
            QdrantError: Если произошла ошибка при поиске.
        """
        try:
            results = await self._qdrant_service.sparse_search(
                query_text=query_text,
                limit=top_k,
            )

            if not results:
                logger.warning("Sparse поиск не вернул результатов")
                return []

            chunks = [ScoredChunk.from_search_result(r) for r in results]
            logger.info(f"Sparse поиск: найдено {len(chunks)} результатов")

            return chunks

        except QdrantError:
            raise
        except Exception as e:
            msg = f"Ошибка sparse поиска: {e}"
            logger.error(msg)
            raise QdrantError(msg, str(e)) from e

    async def hybrid_search(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int = 10,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
        filter_dict: dict[str, Any] | None = None,
        use_rrf: bool = False,
    ) -> list[ScoredChunk]:
        """
        Выполнить гибридный поиск с комбинацией dense и sparse.

        Комбинирует результаты плотного и разреженного поиска используя:
        - Взвешенную сумму нормализованных скорингов (по умолчанию)
        - Reciprocal Rank Fusion (RRF) если use_rrf=True

        Args:
            query_vector: Плотный вектор запроса.
            query_text: Текстовый запрос для sparse поиска.
            top_k: Количество результатов.
            dense_weight: Вес плотного поиска (по умолчанию из настроек).
            sparse_weight: Вес разреженного поиска (по умолчанию из настроек).
            filter_dict: Фильтр для поиска (опционально).
            use_rrf: Использовать Reciprocal Rank Fusion.

        Returns:
            Список ScoredChunk отсортированных по комбинированному score.

        Raises:
            QdrantError: Если произошла ошибка при поиске.
        """
        # Получаем веса из настроек если не указаны
        if dense_weight is None or sparse_weight is None:
            weights = settings.hybrid_weights
            dense_weight = dense_weight or weights.get("dense", 0.7)
            sparse_weight = sparse_weight or weights.get("bm25", 0.3)

        try:
            if use_rrf:
                results = await self._hybrid_search_rrf(
                    query_vector=query_vector,
                    query_text=query_text,
                    top_k=top_k,
                    filter_dict=filter_dict,
                )
            else:
                results = await self._qdrant_service.hybrid_search(
                    query_vector=query_vector,
                    query_text=query_text,
                    limit=top_k,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight,
                    filter_dict=filter_dict,
                )

            if not results:
                logger.warning("Hybrid поиск не вернул результатов")
                return []

            chunks = [ScoredChunk.from_search_result(r) for r in results]
            logger.info(
                f"Hybrid поиск: найдено {len(chunks)} результатов, "
                f"dense_weight={dense_weight}, sparse_weight={sparse_weight}"
            )

            return chunks

        except QdrantError:
            raise
        except Exception as e:
            msg = f"Ошибка hybrid поиска: {e}"
            logger.error(msg)
            raise QdrantError(msg, str(e)) from e

    async def _hybrid_search_rrf(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Выполнить гибридный поиск с использованием Reciprocal Rank Fusion.

        Args:
            query_vector: Плотный вектор запроса.
            query_text: Текстовый запрос.
            top_k: Количество результатов.
            filter_dict: Фильтр для поиска.

        Returns:
            Список результатов с RRF скорингом.
        """
        # Выполняем оба поиска
        dense_results, sparse_results = await asyncio.gather(
            self._qdrant_service.dense_search(
                query_vector=query_vector,
                limit=top_k * 2,
                filter_dict=filter_dict,
            ),
            self._qdrant_service.sparse_search(
                query_text=query_text,
                limit=top_k * 2,
            ),
            return_exceptions=False,
        )

        # Вычисляем RRF scores
        rrf_scores: dict[int, dict[str, Any]] = {}
        k_rrf = 60  # Константа для RRF

        # Обрабатываем dense результаты
        for rank, result in enumerate(dense_results, start=1):
            doc_id = result["id"] if isinstance(result["id"], int) else hash(str(result["id"]))
            rrf_score = 1.0 / (k_rrf + rank)

            if doc_id in rrf_scores:
                rrf_scores[doc_id]["rrf_score"] += rrf_score
            else:
                rrf_scores[doc_id] = {
                    **result,
                    "rrf_score": rrf_score,
                }

        # Обрабатываем sparse результаты
        for rank, result in enumerate(sparse_results, start=1):
            doc_id = result["id"] if isinstance(result["id"], int) else hash(str(result["id"]))
            rrf_score = 1.0 / (k_rrf + rank)

            if doc_id in rrf_scores:
                rrf_scores[doc_id]["rrf_score"] += rrf_score
            else:
                rrf_scores[doc_id] = {
                    **result,
                    "rrf_score": rrf_score,
                }

        # Сортируем по RRF score
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )[:top_k]

        # Нормализуем score
        max_score = max((r["rrf_score"] for r in sorted_results), default=1.0) or 1.0
        for result in sorted_results:
            result["score"] = result["rrf_score"] / max_score
            result["search_type"] = "hybrid_rrf"

        return sorted_results

    async def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_k: int,
        model_name: str = "BAAI/bge-reranker-v2-m3",
    ) -> list[ScoredChunk]:
        """
        Переупорядочить чанки с использованием cross-encoder re-ranker.

        Использует модель cross-encoder для более точной оценки релевантности.

        Args:
            query: Поисковый запрос.
            chunks: Список чанков для переупорядочивания.
            top_k: Количество результатов после re-ranking.
            model_name: Название модели re-ranker.

        Returns:
            Список ScoredChunk переупорядоченных по релевантности.

        Raises:
            SearchError: Если произошла ошибка при re-ranking.
        """
        if not chunks:
            return []

        try:
            # Импортируем cross-encoder
            from sentence_transformers import CrossEncoder

            logger.info(f"Загрузка re-ranker модели: {model_name}")
            reranker = CrossEncoder(model_name)

            # Формируем пары (query, document)
            pairs = [(query, chunk.content) for chunk in chunks]

            # Получаем scores от cross-encoder
            scores = reranker.predict(pairs)

            # Обновляем scores в чанках
            for chunk, score in zip(chunks, scores):
                chunk.score = float(score)
                chunk.metadata["reranker_score"] = float(score)

            # Сортируем по новому score
            sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)[:top_k]

            logger.info(f"Re-ranking завершен: {len(sorted_chunks)} результатов из {len(chunks)}")

            return sorted_chunks

        except ImportError:
            logger.warning("CrossEncoder не доступен, пропускаем re-ranking")
            return chunks[:top_k]

        except Exception as e:
            msg = f"Ошибка re-ranking: {e}"
            logger.error(msg)
            # Возвращаем оригинальные чанки без re-ranking
            return chunks[:top_k]


# Singleton instance
_retriever: Retriever | None = None


def get_retriever(collection_name: str | None = None) -> Retriever:
    """
    Получить экземпляр retriever'а (singleton).

    Args:
        collection_name: Имя коллекции (используется только при первой инициализации).

    Returns:
        Экземпляр Retriever.
    """
    global _retriever
    if _retriever is None:
        _retriever = Retriever(collection_name=collection_name)
    return _retriever
