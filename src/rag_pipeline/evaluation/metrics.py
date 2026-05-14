"""
Retrieval evaluation metrics.

Implements standard IR metrics:
- Hit@k (Recall@k)
- MRR@k (Mean Reciprocal Rank)
- NDCG@k (Normalized Discounted Cumulative Gain)
- Latency measurement
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    hit_at_k: dict[int, float] = field(default_factory=dict)  # k -> score
    recall_at_k: dict[int, float] = field(default_factory=dict)  # k -> score
    precision_at_k: dict[int, float] = field(default_factory=dict)  # k -> score
    f1_at_k: dict[int, float] = field(default_factory=dict)  # k -> score
    mrr_at_k: dict[int, float] = field(default_factory=dict)
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    mean_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    total_queries: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hit_at_k": self.hit_at_k,
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "f1_at_k": self.f1_at_k,
            "mrr_at_k": self.mrr_at_k,
            "ndcg_at_k": self.ndcg_at_k,
            "mean_latency_ms": self.mean_latency_ms,
            "std_latency_ms": self.std_latency_ms,
            "total_queries": self.total_queries,
        }


class RetrievalEvaluator:
    """
    Evaluator for retrieval systems.

    Computes standard IR metrics comparing retrieved results
    against ground truth relevant documents.
    """

    def __init__(self, k_values: list[int] | None = None):
        """
        Initialize evaluator.

        Args:
            k_values: List of k values for metrics (default: [1, 3, 5, 10, 20])
        """
        self.k_values = k_values or [1, 3, 5, 10, 20]

    @staticmethod
    def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        """
        Calculate Hit@k (binary recall).

        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Cut-off rank

        Returns:
            1.0 if at least one relevant doc in top-k, else 0.0
        """
        if not relevant:
            return 0.0
        retrieved_set = set(retrieved[:k])
        return 1.0 if retrieved_set & relevant else 0.0

    @staticmethod
    def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        """
        Calculate Recall@k.

        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Cut-off rank

        Returns:
            Fraction of relevant docs found in top-k
        """
        if not relevant:
            return 0.0
        retrieved_set = set(retrieved[:k])
        return len(retrieved_set & relevant) / len(relevant)

    @staticmethod
    def mrr(retrieved: list[str], relevant: set[str]) -> float:
        """
        Calculate Mean Reciprocal Rank.

        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs

        Returns:
            1/rank of first relevant doc, or 0 if none found
        """
        for idx, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return 1.0 / idx
        return 0.0

    @staticmethod
    def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        """
        Calculate Precision@k.

        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Cut-off rank

        Returns:
            Fraction of retrieved docs in top-k that are relevant
        """
        if k == 0:
            return 0.0
        retrieved_set = set(retrieved[:k])
        if not retrieved_set:
            return 0.0
        return len(retrieved_set & relevant) / len(retrieved_set)

    @staticmethod
    def f1_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        """
        Calculate F1@k (harmonic mean of Precision@k and Recall@k).

        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Cut-off rank

        Returns:
            F1 score (0.0 to 1.0)
        """
        precision = RetrievalEvaluator.precision_at_k(retrieved, relevant, k)
        recall = RetrievalEvaluator.recall_at_k(retrieved, relevant, k)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        """
        Calculate NDCG@k.

        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Cut-off rank

        Returns:
            NDCG score (0.0 to 1.0)
        """
        if not relevant:
            return 0.0

        # DCG
        dcg = 0.0
        for idx, doc_id in enumerate(retrieved[:k]):
            if doc_id in relevant:
                dcg += 1.0 / np.log2(idx + 2)

        # IDCG (ideal DCG)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

        return dcg / idcg if idcg > 0 else 0.0

    def evaluate_query(
        self,
        retrieved_ids: list[str],
        relevant_ids: set[str],
        latency_ms: float = 0.0,
    ) -> dict[str, Any]:
        """
        Evaluate a single query.

        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: Set of relevant document IDs
            latency_ms: Query latency in milliseconds

        Returns:
            Dictionary with all metrics for this query
        """
        metrics = {"latency_ms": latency_ms}

        for k in self.k_values:
            metrics[f"hit@{k}"] = self.hit_at_k(retrieved_ids, relevant_ids, k)
            metrics[f"recall@{k}"] = self.recall_at_k(retrieved_ids, relevant_ids, k)
            metrics[f"precision@{k}"] = self.precision_at_k(retrieved_ids, relevant_ids, k)
            metrics[f"f1@{k}"] = self.f1_at_k(retrieved_ids, relevant_ids, k)
            metrics[f"mrr@{k}"] = self.mrr(retrieved_ids, relevant_ids)
            metrics[f"ndcg@{k}"] = self.ndcg_at_k(retrieved_ids, relevant_ids, k)

        return metrics

    def aggregate_metrics(
        self,
        query_results: list[dict[str, Any]],
    ) -> EvaluationMetrics:
        """
        Aggregate metrics across multiple queries.

        Args:
            query_results: List of per-query metric dictionaries

        Returns:
            Aggregated EvaluationMetrics
        """
        if not query_results:
            return EvaluationMetrics()

        n_queries = len(query_results)
        metrics = EvaluationMetrics(total_queries=n_queries)

        # Initialize accumulators
        hit_sums = dict.fromkeys(self.k_values, 0.0)
        recall_sums = dict.fromkeys(self.k_values, 0.0)
        precision_sums = dict.fromkeys(self.k_values, 0.0)
        f1_sums = dict.fromkeys(self.k_values, 0.0)
        mrr_sums = dict.fromkeys(self.k_values, 0.0)
        ndcg_sums = dict.fromkeys(self.k_values, 0.0)
        latencies = []

        # Accumulate
        for result in query_results:
            latencies.append(result.get("latency_ms", 0.0))
            for k in self.k_values:
                hit_sums[k] += result.get(f"hit@{k}", 0.0)
                recall_sums[k] += result.get(f"recall@{k}", 0.0)
                precision_sums[k] += result.get(f"precision@{k}", 0.0)
                f1_sums[k] += result.get(f"f1@{k}", 0.0)
                mrr_sums[k] += result.get(f"mrr@{k}", 0.0)
                ndcg_sums[k] += result.get(f"ndcg@{k}", 0.0)

        # Average
        for k in self.k_values:
            metrics.hit_at_k[k] = hit_sums[k] / n_queries
            metrics.recall_at_k[k] = recall_sums[k] / n_queries
            metrics.precision_at_k[k] = precision_sums[k] / n_queries
            metrics.f1_at_k[k] = f1_sums[k] / n_queries
            metrics.mrr_at_k[k] = mrr_sums[k] / n_queries
            metrics.ndcg_at_k[k] = ndcg_sums[k] / n_queries

        # Latency stats
        if latencies:
            metrics.mean_latency_ms = np.mean(latencies)
            metrics.std_latency_ms = np.std(latencies)

        return metrics

    def evaluate_strategy(
        self,
        queries: list[dict[str, Any]],
        search_fn,
        strategy_name: str = "unknown",
    ) -> dict[str, Any]:
        """
        Evaluate a complete retrieval strategy.

        Args:
            queries: List of query dicts with 'question' and 'relevant_chunk_ids'
            search_fn: Function that takes query string and returns (result_ids, latency_ms)
            strategy_name: Name of the strategy being evaluated

        Returns:
            Dictionary with aggregated metrics
        """
        logger.info(f"Evaluating strategy: {strategy_name} on {len(queries)} queries")

        query_results = []
        for idx, query_data in enumerate(queries):
            question = query_data.get("question", "")
            relevant_ids = set(query_data.get("relevant_chunk_ids", []))

            logger.debug(f"Query {idx+1}/{len(queries)}: '{question}'")
            logger.debug(f"Relevant chunk IDs: {relevant_ids}")

            # Perform search
            try:
                retrieved_ids, latency_ms = search_fn(question)
                logger.debug(f"Retrieved {len(retrieved_ids)} chunk IDs: {retrieved_ids[:5]}...")
                logger.debug(f"Latency: {latency_ms:.2f}ms")
            except Exception as e:
                logger.warning(f"Search failed for query: {e}")
                retrieved_ids, latency_ms = [], 0.0

            # Evaluate
            metrics = self.evaluate_query(retrieved_ids, relevant_ids, latency_ms)

            # Log per-query metrics for debugging
            if idx < 3:  # Log first 3 queries in detail
                logger.debug(
                    f"Query {idx+1} metrics: hit@1={metrics.get('hit@1', 0):.3f}, "
                    f"mrr@10={metrics.get('mrr@10', 0):.3f}, "
                    f"ndcg@10={metrics.get('ndcg@10', 0):.3f}"
                )

            query_results.append(metrics)

        # Aggregate
        aggregated = self.aggregate_metrics(query_results)

        logger.info(f"Aggregated metrics for {strategy_name}:")
        for k in self.k_values:
            logger.info(f"  Hit@{k}: {aggregated.hit_at_k.get(k, 0):.4f}")
            logger.info(f"  Recall@{k}: {aggregated.recall_at_k.get(k, 0):.4f}")
            logger.info(f"  Precision@{k}: {aggregated.precision_at_k.get(k, 0):.4f}")
            logger.info(f"  F1@{k}: {aggregated.f1_at_k.get(k, 0):.4f}")
            logger.info(f"  MRR@{k}: {aggregated.mrr_at_k.get(k, 0):.4f}")
            logger.info(f"  NDCG@{k}: {aggregated.ndcg_at_k.get(k, 0):.4f}")
        logger.info(f"  Mean latency: {aggregated.mean_latency_ms:.2f}ms")

        return {
            "strategy": strategy_name,
            "metrics": aggregated.to_dict(),
        }

    def evaluate_multi_dataset(
        self,
        datasets: dict[str, list[dict[str, Any]]],
        search_fn,
        strategy_name: str = "unknown",
    ) -> dict[str, Any]:
        """
        Evaluate a strategy across multiple datasets.

        Args:
            datasets: Dictionary mapping dataset names to query lists
            search_fn: Function that takes query string and returns (result_ids, latency_ms)
            strategy_name: Name of the strategy being evaluated

        Returns:
            Dictionary with per-dataset and combined metrics
        """
        logger.info(f"Evaluating strategy: {strategy_name} across {len(datasets)} datasets")

        results = {}
        all_query_results = []

        for dataset_name, queries in datasets.items():
            if not queries:
                logger.warning(f"Dataset {dataset_name} is empty, skipping")
                continue

            logger.info(f"Evaluating on dataset: {dataset_name} ({len(queries)} queries)")
            dataset_result = self.evaluate_strategy(queries, search_fn, f"{strategy_name}_{dataset_name}")
            results[dataset_name] = dataset_result

            # Collect for combined evaluation
            for query_data in queries:
                question = query_data.get("question", "")
                relevant_ids = set(query_data.get("relevant_chunk_ids", []))
                try:
                    retrieved_ids, latency_ms = search_fn(question)
                    metrics = self.evaluate_query(retrieved_ids, relevant_ids, latency_ms)
                    all_query_results.append(metrics)
                except Exception as e:
                    logger.warning(f"Search failed for query in {dataset_name}: {e}")

        # Combined metrics
        if all_query_results:
            combined_aggregated = self.aggregate_metrics(all_query_results)
            results["combined"] = {
                "strategy": f"{strategy_name}_combined",
                "metrics": combined_aggregated.to_dict(),
            }
            logger.info("Combined metrics across all datasets:")
            for k in self.k_values:
                logger.info(f"  Hit@{k}: {combined_aggregated.hit_at_k.get(k, 0):.4f}")
                logger.info(f"  Recall@{k}: {combined_aggregated.recall_at_k.get(k, 0):.4f}")
                logger.info(f"  Precision@{k}: {combined_aggregated.precision_at_k.get(k, 0):.4f}")
                logger.info(f"  F1@{k}: {combined_aggregated.f1_at_k.get(k, 0):.4f}")
                logger.info(f"  MRR@{k}: {combined_aggregated.mrr_at_k.get(k, 0):.4f}")
                logger.info(f"  NDCG@{k}: {combined_aggregated.ndcg_at_k.get(k, 0):.4f}")
            logger.info(f"  Mean latency: {combined_aggregated.mean_latency_ms:.2f}ms")

        return results
