"""
Модуль для оценки качества RAG системы.

Модуль предоставляет функции для расчета метрик:
- NDCG (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
- Hit@K
- Faithfulness (оценка правдоподобности через LLM)
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    Результат оценки.

    Атрибуты:
        metric_name: Название метрики.
        score: Значение метрики.
        query: Исходный запрос (опционально).
        details: Дополнительные детали.
    """

    metric_name: str
    score: float
    query: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


def calculate_ndcg(relevant: list[int], retrieved: list[int], k: int = 10) -> float:
    """
    Рассчитать Normalized Discounted Cumulative Gain (NDCG).

    Метрика оценивает качество ранжирования с учетом релевантности документов.

    Args:
        relevant: Список уровней релевантности для каждого документа в идеальном порядке.
        retrieved: Список уровней релевантности для retrieved документов.
        k: Количество топ результатов для учета.

    Returns:
        Значение NDCG в диапазоне [0, 1].

    Example:
        >>> relevant = [3, 2, 1, 0, 0]
        >>> retrieved = [2, 1, 3, 0, 0]
        >>> calculate_ndcg(relevant, retrieved, k=3)
        0.89...
    """

    def dcg_at_k(rels: list[int], k: int) -> float:
        """Рассчитать Discounted Cumulative Gain."""
        rels = rels[:k]
        gain = 0.0
        for i, rel in enumerate(rels, start=1):
            # Формула: gain / log2(i + 1)
            gain += rel / math.log2(i + 1)
        return gain

    # Идеальный DCG (сортируем по убыванию релевантности)
    ideal_relevant = sorted(relevant, reverse=True)[:k]
    dcg_ideal = dcg_at_k(ideal_relevant, k)

    if dcg_ideal == 0:
        return 1.0  # Нет релевантных документов

    # Реальный DCG
    dcg_actual = dcg_at_k(retrieved[:k], k)

    ndcg = dcg_actual / dcg_ideal
    logger.debug(f"NDCG@{k}: {ndcg:.4f} (dcg_actual={dcg_actual:.4f}, dcg_ideal={dcg_ideal:.4f})")

    return ndcg


def calculate_mrr(relevant_positions: list[int]) -> float:
    """
    Рассчитать Mean Reciprocal Rank (MRR).

    Метрика оценивает среднюю обратную позицию первого релевантного документа.

    Args:
        relevant_positions: Позиции релевантных документов (1-indexed).
                           Если пустой список, RR = 0.

    Returns:
        Значение MRR в диапазоне [0, 1].

    Example:
        >>> calculate_mrr([1, 3, 2])  # Три запроса с разными позициями
        0.61...
    """
    if not relevant_positions:
        return 0.0

    reciprocal_ranks = []
    for pos in relevant_positions:
        if pos > 0:
            reciprocal_ranks.append(1.0 / pos)
        else:
            reciprocal_ranks.append(0.0)

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    logger.debug(f"MRR: {mrr:.4f} (средний из {len(reciprocal_ranks)} значений)")

    return mrr


def calculate_hit_at_k(relevant: set, retrieved: list, k: int = 10) -> float:
    """
    Рассчитать Hit Rate @ K.

    Бинарная метрика: 1 если хотя бы один релевантный документ найден в топ-K.

    Args:
        relevant: Множество идентификаторов релевантных документов.
        retrieved: Список идентификаторов retrieved документов.
        k: Количество топ результатов для проверки.

    Returns:
        1.0 если найден хотя бы один релевантный документ, иначе 0.0.

    Example:
        >>> relevant = {"doc1", "doc2"}
        >>> retrieved = ["doc3", "doc1", "doc4"]
        >>> calculate_hit_at_k(relevant, retrieved, k=3)
        1.0
    """
    top_k = set(retrieved[:k])
    hit = len(relevant & top_k) > 0

    logger.debug(f"Hit@{k}: {'hit' if hit else 'miss'}")

    return 1.0 if hit else 0.0


async def calculate_faithfulness(
    query: str,
    answer: str,
    contexts: list[str],
    llm_client: Any | None = None,
) -> float:
    """
    Рассчитать Faithfulness (правдоподобность) ответа.

    Использует LLM-as-a-judge подход для оценки того, насколько ответ
    соответствует предоставленному контексту.

    Args:
        query: Исходный вопрос.
        answer: Сгенерированный ответ.
        contexts: Список контекстов использованных для генерации.
        llm_client: Клиент LLM для оценки (опционально).

    Returns:
        Значение faithfulness в диапазоне [0, 1].

    Raises:
        ValueError: Если LLM клиент не предоставлен.
    """
    if not contexts:
        return 0.0

    # Форматируем контекст
    formatted_context = "\n\n".join([f"[Контекст {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)])

    # Промпт для оценки faithfulness на русском
    evaluation_prompt = f"""Ты — оценщик качества RAG системы. Оцени правдоподобность ответа.

Вопрос: {query}

Контекст:
{formatted_context}

Ответ:
{answer}

Оцени, насколько ответ соответствует предоставленному контексту:
- 1.0: Ответ полностью основан на контексте, нет выдумок
- 0.5: Ответ частично основан на контексте, есть небольшие домыслы
- 0.0: Ответ не основан на контексте или противоречит ему

Верни ТОЛЬКО число от 0.0 до 1.0 без дополнительных объяснений.
Оценка:"""

    try:
        if llm_client is None:
            # Импортируем клиент если не предоставлен
            from src.services.llm_client import get_llm_client

            llm_client = get_llm_client()

        # Получаем оценку от LLM
        response = await llm_client.generate(
            query=evaluation_prompt,
            contexts=[],  # Не передаем контекст, он уже в промпте
            max_tokens=10,
        )

        # Парсим ответ
        score_str = response.strip()
        # Извлекаем число из ответа
        import re

        match = re.search(r"(\d\.?\d*)", score_str)
        if match:
            score = float(match.group(1))
            # Нормализуем к [0, 1]
            score = max(0.0, min(1.0, score))
            logger.info(f"Faithfulness оценка: {score:.4f}")
            return score
        else:
            logger.warning(f"Не удалось распарсить оценку faithfulness: {score_str}")
            return 0.5  # Возвращаем нейтральное значение

    except Exception as e:
        logger.error(f"Ошибка при оценке faithfulness: {e}")
        return 0.5  # Возвращаем нейтральное значение при ошибке


class EvaluationLogger:
    """
    Логгер для результатов оценки.

    Сохраняет результаты оценки в директорию eval_logs/
    """

    def __init__(self, log_dir: str = "eval_logs") -> None:
        """
        Инициализация логгера.

        Args:
            log_dir: Директория для сохранения логов.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"EvaluationLogger инициализирован: {self.log_dir.absolute()}")

    def log_evaluation(
        self,
        query: str,
        results: list[EvaluationResult],
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """
        Сохранить результаты оценки.

        Args:
            query: Исходный запрос.
            results: Список результатов оценки.
            metadata: Дополнительные метаданные.

        Returns:
            Путь к сохраненному файлу.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{timestamp}.json"
        filepath = self.log_dir / filename

        log_entry = {
            "timestamp": timestamp,
            "query": query,
            "results": [
                {
                    "metric": r.metric_name,
                    "score": r.score,
                    "details": r.details,
                }
                for r in results
            ],
            "metadata": metadata or {},
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)

        logger.info(f"Результаты оценки сохранены: {filepath}")

        return filepath

    def get_aggregate_stats(self) -> dict[str, Any]:
        """
        Получить агрегированную статистику по всем логам.

        Returns:
            Словарь со статистикой по метрикам.
        """
        metrics: dict[str, list[float]] = {}
        total_queries = 0

        for log_file in self.log_dir.glob("eval_*.json"):
            try:
                with open(log_file, encoding="utf-8") as f:
                    data = json.load(f)

                total_queries += 1

                for result in data.get("results", []):
                    metric_name = result["metric"]
                    score = result["score"]

                    if metric_name not in metrics:
                        metrics[metric_name] = []
                    metrics[metric_name].append(score)

            except Exception as e:
                logger.warning(f"Ошибка чтения лога {log_file}: {e}")

        # Вычисляем средние значения
        aggregate = {
            "total_queries": total_queries,
            "metrics": {},
        }

        for metric_name, scores in metrics.items():
            aggregate["metrics"][metric_name] = {
                "mean": sum(scores) / len(scores) if scores else 0.0,
                "min": min(scores) if scores else 0.0,
                "max": max(scores) if scores else 0.0,
                "count": len(scores),
            }

        return aggregate


# Singleton instance
_evaluation_logger: EvaluationLogger | None = None


def get_evaluation_logger(log_dir: str = "eval_logs") -> EvaluationLogger:
    """
    Получить экземпляр evaluation логгера (singleton).

    Args:
        log_dir: Директория для сохранения логов.

    Returns:
        Экземпляр EvaluationLogger.
    """
    global _evaluation_logger
    if _evaluation_logger is None:
        _evaluation_logger = EvaluationLogger(log_dir=log_dir)
    return _evaluation_logger
