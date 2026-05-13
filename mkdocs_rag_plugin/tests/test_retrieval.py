"""
Тесты для модуля retrieval.

Тестирует:
- Dense поиск с mock Qdrant
- Sparse поиск с mock Qdrant
- Hybrid score fusion logic
- Re-ranking accuracy
- Evaluation metrics calculation
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.retriever import Retriever, ScoredChunk, get_retriever
from src.services.evaluation import (
    calculate_ndcg,
    calculate_mrr,
    calculate_hit_at_k,
    calculate_faithfulness,
    EvaluationResult,
    EvaluationLogger,
)


class TestDenseSearch:
    """Тесты для dense поиска."""

    @pytest.fixture
    def mock_qdrant_service(self):
        """Создать mock Qdrant service."""
        with patch("src.services.retriever.get_qdrant_service") as mock_getter:
            mock_service = MagicMock()
            mock_service.dense_search = AsyncMock(
                return_value=[
                    {
                        "id": 1,
                        "score": 0.95,
                        "payload": {
                            "chunk_id": "chunk_1",
                            "url": "https://example.com/doc1",
                            "title": "Документ 1",
                            "content": "Содержимое документа 1",
                            "header_path": "/doc1",
                            "source_file": "doc1.md",
                        },
                        "search_type": "dense",
                    },
                    {
                        "id": 2,
                        "score": 0.85,
                        "payload": {
                            "chunk_id": "chunk_2",
                            "url": "https://example.com/doc2",
                            "title": "Документ 2",
                            "content": "Содержимое документа 2",
                            "header_path": "/doc2",
                            "source_file": "doc2.md",
                        },
                        "search_type": "dense",
                    },
                ]
            )
            mock_getter.return_value = mock_service
            yield mock_service

    @pytest.fixture
    def retriever(self, mock_qdrant_service):
        """Создать retriever с mock сервисом."""
        return get_retriever(collection_name="test_collection")

    @pytest.mark.asyncio
    async def test_dense_search_returns_results(self, retriever):
        """Тест: dense поиск возвращает результаты."""
        query_vector = [0.1] * 384
        results = await retriever.dense_search(query_vector, top_k=5)

        assert len(results) == 2
        assert isinstance(results[0], ScoredChunk)
        assert results[0].score == 0.95
        assert results[0].url == "https://example.com/doc1"
        assert results[0].search_type == "dense"

    @pytest.mark.asyncio
    async def test_dense_search_with_filter(self, retriever, mock_qdrant_service):
        """Тест: dense поиск с фильтром."""
        query_vector = [0.1] * 384
        filter_dict = {"must": [{"key": "source_file", "match": {"value": "doc1.md"}}]}

        await retriever.dense_search(query_vector, top_k=5, filter_dict=filter_dict)

        mock_qdrant_service.dense_search.assert_called_once_with(
            query_vector=query_vector,
            limit=5,
            filter_dict=filter_dict,
        )

    @pytest.mark.asyncio
    async def test_dense_search_empty_results(self, retriever, mock_qdrant_service):
        """Тест: dense поиск без результатов."""
        mock_qdrant_service.dense_search.return_value = []

        results = await retriever.dense_search([0.1] * 384, top_k=5)

        assert len(results) == 0


class TestSparseSearch:
    """Тесты для sparse поиска (BM25)."""

    @pytest.fixture
    def mock_qdrant_service(self):
        """Создать mock Qdrant service."""
        with patch("src.services.retriever.get_qdrant_service") as mock_getter:
            mock_service = MagicMock()
            mock_service.sparse_search = AsyncMock(
                return_value=[
                    {
                        "id": 1,
                        "score": 0.75,
                        "payload": {
                            "chunk_id": "chunk_1",
                            "url": "https://example.com/doc1",
                            "title": "Документ 1",
                            "content": "Содержимое документа 1",
                            "header_path": "/doc1",
                            "source_file": "doc1.md",
                        },
                        "search_type": "sparse",
                    },
                ]
            )
            mock_getter.return_value = mock_service
            yield mock_service

    @pytest.fixture
    def retriever(self, mock_qdrant_service):
        """Создать retriever с mock сервисом."""
        return get_retriever(collection_name="test_collection")

    @pytest.mark.asyncio
    async def test_sparse_search_returns_results(self, retriever):
        """Тест: sparse поиск возвращает результаты."""
        results = await retriever.sparse_search("поисковый запрос", top_k=5)

        assert len(results) == 1
        assert results[0].score == 0.75
        assert results[0].search_type == "sparse"

    @pytest.mark.asyncio
    async def test_sparse_search_tokenization(self, retriever, mock_qdrant_service):
        """Тест: sparse поиск токенизирует запрос."""
        await retriever.sparse_search("Запрос с пунктуацией!", top_k=5)

        mock_qdrant_service.sparse_search.assert_called_once()


class TestHybridSearch:
    """Тесты для гибридного поиска."""

    @pytest.fixture
    def mock_qdrant_service(self):
        """Создать mock Qdrant service."""
        with patch("src.services.retriever.get_qdrant_service") as mock_getter:
            mock_service = MagicMock()
            mock_service.hybrid_search = AsyncMock(
                return_value=[
                    {
                        "id": 1,
                        "score": 0.90,
                        "payload": {
                            "chunk_id": "chunk_1",
                            "url": "https://example.com/doc1",
                            "title": "Документ 1",
                            "content": "Содержимое документа 1",
                            "header_path": "/doc1",
                            "source_file": "doc1.md",
                        },
                        "search_type": "hybrid",
                    },
                ]
            )
            mock_getter.return_value = mock_service
            yield mock_service

    @pytest.fixture
    def retriever(self, mock_qdrant_service):
        """Создать retriever с mock сервисом."""
        return get_retriever(collection_name="test_collection")

    @pytest.mark.asyncio
    async def test_hybrid_search_weighted_fusion(self, retriever):
        """Тест: гибридный поиск с взвешенной фузией."""
        query_vector = [0.1] * 384
        results = await retriever.hybrid_search(
            query_vector=query_vector,
            query_text="тестовый запрос",
            top_k=5,
            dense_weight=0.7,
            sparse_weight=0.3,
        )

        assert len(results) == 1
        assert results[0].search_type == "hybrid"

    @pytest.mark.asyncio
    async def test_hybrid_search_default_weights(self, retriever, mock_qdrant_service):
        """Тест: гибридный поиск с весами по умолчанию."""
        with patch("src.services.retriever.settings.hybrid_weights", {"dense": 0.7, "bm25": 0.3}):
            await retriever.hybrid_search(
                query_vector=[0.1] * 384,
                query_text="запрос",
                top_k=5,
            )

            mock_qdrant_service.hybrid_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_search_rrf(self, retriever, mock_qdrant_service):
        """Тест: гибридный поиск с RRF."""
        mock_qdrant_service.dense_search = AsyncMock(
            return_value=[
                {
                    "id": 1,
                    "score": 0.9,
                    "payload": {
                        "chunk_id": "c1",
                        "url": "u1",
                        "title": "t1",
                        "content": "c1",
                        "header_path": "h1",
                        "source_file": "s1",
                    },
                },
                {
                    "id": 2,
                    "score": 0.8,
                    "payload": {
                        "chunk_id": "c2",
                        "url": "u2",
                        "title": "t2",
                        "content": "c2",
                        "header_path": "h2",
                        "source_file": "s2",
                    },
                },
            ]
        )
        mock_qdrant_service.sparse_search = AsyncMock(
            return_value=[
                {
                    "id": 2,
                    "score": 0.95,
                    "payload": {
                        "chunk_id": "c2",
                        "url": "u2",
                        "title": "t2",
                        "content": "c2",
                        "header_path": "h2",
                        "source_file": "s2",
                    },
                },
                {
                    "id": 3,
                    "score": 0.7,
                    "payload": {
                        "chunk_id": "c3",
                        "url": "u3",
                        "title": "t3",
                        "content": "c3",
                        "header_path": "h3",
                        "source_file": "s3",
                    },
                },
            ]
        )

        results = await retriever.hybrid_search(
            query_vector=[0.1] * 384,
            query_text="запрос",
            top_k=5,
            use_rrf=True,
        )

        assert len(results) > 0
        assert results[0].search_type == "hybrid_rrf"


class TestReranking:
    """Тесты для re-ranking."""

    @pytest.fixture
    def sample_chunks(self):
        """Создать тестовые чанки."""
        return [
            ScoredChunk(
                chunk_id="chunk_1",
                url="https://example.com/doc1",
                title="Документ 1",
                content="Это первый документ с некоторым содержанием",
                header_path="/doc1",
                source_file="doc1.md",
                score=0.9,
            ),
            ScoredChunk(
                chunk_id="chunk_2",
                url="https://example.com/doc2",
                title="Документ 2",
                content="Это второй документ с другим содержанием",
                header_path="/doc2",
                source_file="doc2.md",
                score=0.8,
            ),
        ]

    @pytest.mark.asyncio
    async def test_rerank_with_cross_encoder(self, sample_chunks):
        """Тест: re-ranking с cross-encoder."""
        retriever = get_retriever()

        with patch("src.services.retriever.CrossEncoder") as MockCrossEncoder:
            mock_reranker = MagicMock()
            mock_reranker.predict.return_value = [0.95, 0.85]
            MockCrossEncoder.return_value = mock_reranker

            reranked = await retriever.rerank(
                query="тестовый запрос",
                chunks=sample_chunks,
                top_k=2,
            )

            assert len(reranked) == 2
            assert reranked[0].score == 0.95
            assert "reranker_score" in reranked[0].metadata

    @pytest.mark.asyncio
    async def test_rerank_empty_chunks(self, retriever):
        """Тест: re-ranking пустых чанков."""
        results = await retriever.rerank(query="запрос", chunks=[], top_k=5)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_rerank_fallback_on_error(self, sample_chunks):
        """Тест: re-ranking fallback при ошибке."""
        retriever = get_retriever()

        with patch("src.services.retriever.CrossEncoder") as MockCrossEncoder:
            MockCrossEncoder.side_effect = ImportError("CrossEncoder not available")

            reranked = await retriever.rerank(
                query="запрос",
                chunks=sample_chunks,
                top_k=2,
            )

            # Должен вернуть оригинальные чанки без re-ranking
            assert len(reranked) == 2
            assert reranked[0].score == 0.9  # Оригинальный score


class TestEvaluationMetrics:
    """Тесты для метрик оценки."""

    def test_calculate_ndcg_perfect_ranking(self):
        """Тест: NDCG для идеального ранжирования."""
        relevant = [3, 2, 1, 0, 0]
        retrieved = [3, 2, 1, 0, 0]

        ndcg = calculate_ndcg(relevant, retrieved, k=5)

        assert ndcg == pytest.approx(1.0, rel=1e-5)

    def test_calculate_ndcg_imperfect_ranking(self):
        """Тест: NDCG для неидеального ранжирования."""
        relevant = [3, 2, 1, 0, 0]
        retrieved = [1, 2, 3, 0, 0]  # Порядок изменен

        ndcg = calculate_ndcg(relevant, retrieved, k=3)

        assert 0.0 <= ndcg <= 1.0
        assert ndcg < 1.0  # Не идеальное ранжирование

    def test_calculate_ndcg_no_relevant(self):
        """Тест: NDCG когда нет релевантных документов."""
        relevant = [0, 0, 0]
        retrieved = [0, 0, 0]

        ndcg = calculate_ndcg(relevant, retrieved, k=3)

        assert ndcg == 1.0  # Нет релевантных, считаем идеальным

    def test_calculate_mrr_perfect(self):
        """Тест: MRR для идеальных позиций."""
        positions = [1, 1, 1]  # Все первые позиции

        mrr = calculate_mrr(positions)

        assert mrr == 1.0

    def test_calculate_mrr_mixed(self):
        """Тест: MRR для смешанных позиций."""
        positions = [1, 2, 4]  # Разные позиции

        mrr = calculate_mrr(positions)

        expected = (1 / 1 + 1 / 2 + 1 / 4) / 3
        assert mrr == pytest.approx(expected, rel=1e-5)

    def test_calculate_mrr_empty(self):
        """Тест: MRR для пустого списка."""
        mrr = calculate_mrr([])
        assert mrr == 0.0

    def test_calculate_hit_at_k_hit(self):
        """Тест: Hit@K когда есть попадание."""
        relevant = {"doc1", "doc2"}
        retrieved = ["doc3", "doc1", "doc4"]

        hit = calculate_hit_at_k(relevant, retrieved, k=3)

        assert hit == 1.0

    def test_calculate_hit_at_k_miss(self):
        """Тест: Hit@K когда нет попадания."""
        relevant = {"doc1", "doc2"}
        retrieved = ["doc3", "doc4", "doc5"]

        hit = calculate_hit_at_k(relevant, retrieved, k=3)

        assert hit == 0.0

    def test_calculate_hit_at_k_partial(self):
        """Тест: Hit@K с частичным попаданием."""
        relevant = {"doc1", "doc2"}
        retrieved = ["doc3", "doc4", "doc1"]

        # doc1 находится на позиции 3, так что Hit@2 = 0, Hit@3 = 1
        assert calculate_hit_at_k(relevant, retrieved, k=2) == 0.0
        assert calculate_hit_at_k(relevant, retrieved, k=3) == 1.0

    @pytest.mark.asyncio
    async def test_calculate_faithfulness_perfect(self):
        """Тест: Faithfulness для идеального ответа."""
        with patch("src.services.llm_client.get_llm_client") as mock_getter:
            mock_client = MagicMock()
            mock_client.generate = AsyncMock(return_value="1.0")
            mock_getter.return_value = mock_client

            faithfulness = await calculate_faithfulness(
                query="Тестовый вопрос",
                answer="Ответ основанный на контексте",
                contexts=["Контекст с информацией"],
            )

            assert faithfulness == 1.0

    @pytest.mark.asyncio
    async def test_calculate_faithfulness_empty_contexts(self):
        """Тест: Faithfulness для пустых контекстов."""
        faithfulness = await calculate_faithfulness(
            query="Вопрос",
            answer="Ответ",
            contexts=[],
        )

        assert faithfulness == 0.0


class TestEvaluationLogger:
    """Тесты для логгера оценки."""

    @pytest.fixture
    def temp_log_dir(self, tmp_path):
        """Создать временную директорию для логов."""
        return str(tmp_path / "eval_logs")

    def test_log_evaluation_creates_file(self, temp_log_dir):
        """Тест: log_evaluation создает файл."""
        logger = EvaluationLogger(log_dir=temp_log_dir)

        results = [
            EvaluationResult(metric_name="ndcg", score=0.85),
            EvaluationResult(metric_name="mrr", score=0.75),
        ]

        filepath = logger.log_evaluation(
            query="Тестовый запрос",
            results=results,
            metadata={"strategy": "hybrid"},
        )

        assert filepath.exists()
        assert filepath.suffix == ".json"

    def test_get_aggregate_stats(self, temp_log_dir):
        """Тест: get_aggregate_stats возвращает статистику."""
        logger = EvaluationLogger(log_dir=temp_log_dir)

        # Сохраняем несколько записей
        for i in range(3):
            results = [
                EvaluationResult(metric_name="ndcg", score=0.8 + i * 0.05),
            ]
            logger.log_evaluation(
                query=f"Запрос {i}",
                results=results,
            )

        stats = logger.get_aggregate_stats()

        assert stats["total_queries"] == 3
        assert "ndcg" in stats["metrics"]
        assert stats["metrics"]["ndcg"]["count"] == 3
