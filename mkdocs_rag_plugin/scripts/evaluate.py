#!/usr/bin/env python3
"""
Скрипт для оценки качества RAG системы.

Запускает тестовые запросы через пайплайн и вычисляет метрики:
- NDCG@k
- MRR
- Hit@k
- Faithfulness (LLM-as-a-judge)

Пример использования:
    python scripts/evaluate.py --dataset data/test_queries.json --output reports/eval_results.md
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.services.evaluation import (
    EvaluationLogger,
    calculate_faithfulness,
    calculate_hit_at_k,
    calculate_mrr,
    calculate_ndcg,
)
from src.services.search_pipeline import SearchPipeline, SearchResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_dataset(dataset_path: str) -> list[dict[str, Any]]:
    """
    Загрузить тестовый датасет.

    Формат датасета (JSON):
    [
        {
            "question": "Как настроить аутентификацию?",
            "expected_answer": "Для настройки аутентификации используйте...",
            "relevant_docs": ["auth.md", "security.md"]
        },
        ...
    ]

    Args:
        dataset_path: Путь к файлу датасета.

    Returns:
        Список тестовых кейсов.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Датасет не найден: {dataset_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Загружено {len(data)} тестовых кейсов из {dataset_path}")
    return data


async def run_evaluation(
    dataset: list[dict[str, Any]],
    output_dir: str,
    top_k: int = 5,
    mode: str = "hybrid",
) -> dict[str, Any]:
    """
    Запустить оценку качества RAG системы.

    Args:
        dataset: Тестовый датасет.
        output_dir: Директория для вывода результатов.
        top_k: Количество retrieved документов.
        mode: Режим поиска (dense, sparse, hybrid).

    Returns:
        Словарь с результатами оценки.
    """
    # Инициализация пайплайна и логгера
    pipeline = SearchPipeline()
    eval_logger = EvaluationLogger(output_dir=output_dir)

    results = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": len(dataset),
        "mode": mode,
        "top_k": top_k,
        "metrics": {
            "ndcg_scores": [],
            "mrr_scores": [],
            "hit_at_k_scores": [],
            "faithfulness_scores": [],
        },
        "query_results": [],
    }

    logger.info(f"Начало оценки: {len(dataset)} запросов, режим={mode}, top_k={top_k}")

    for i, test_case in enumerate(dataset, 1):
        question = test_case.get("question", "")
        expected_answer = test_case.get("expected_answer", "")
        relevant_docs = set(test_case.get("relevant_docs", []))

        if not question:
            logger.warning(f"Пропуск кейса {i}: пустой вопрос")
            continue

        logger.info(f"[{i}/{len(dataset)}] Обработка: {question[:50]}...")

        try:
            # Выполнение поиска
            search_result: SearchResult = await pipeline.search(
                query=question,
                top_k=top_k,
                mode=mode,
            )

            # Извлечение URL retrieved документов
            retrieved_urls = [
                chunk.metadata.get("source_file", "").split("/")[-1]
                for chunk in search_result.chunks
            ]

            # Вычисление метрик retrieval
            ndcg = calculate_ndcg(relevant_docs, retrieved_urls, k=top_k)
            mrr = calculate_mrr(relevant_docs, retrieved_urls)
            hit_at_k = calculate_hit_at_k(relevant_docs, retrieved_urls, k=top_k)

            # Вычисление faithfulness (если есть ожидаемый ответ)
            faithfulness = 0.0
            if expected_answer and search_result.answer:
                faithfulness = await calculate_faithfulness(
                    query=question,
                    answer=search_result.answer,
                    contexts=[chunk.content for chunk in search_result.chunks],
                )

            # Сохранение результатов
            query_result = {
                "question": question,
                "answer": search_result.answer,
                "ndcg": ndcg,
                "mrr": mrr,
                "hit_at_k": hit_at_k,
                "faithfulness": faithfulness,
                "retrieval_time_ms": search_result.metadata.get("retrieval_time_ms", 0),
                "generation_time_ms": search_result.metadata.get("generation_time_ms", 0),
                "sources_count": len(search_result.chunks),
                "relevant_docs": list(relevant_docs),
                "retrieved_docs": retrieved_urls,
            }

            results["query_results"].append(query_result)
            results["metrics"]["ndcg_scores"].append(ndcg)
            results["metrics"]["mrr_scores"].append(mrr)
            results["metrics"]["hit_at_k_scores"].append(hit_at_k)
            results["metrics"]["faithfulness_scores"].append(faithfulness)

            # Логирование в файл
            eval_logger.log_query(
                query=question,
                answer=search_result.answer,
                chunks=search_result.chunks,
                relevance_labels=list(relevant_docs),
            )

        except Exception as e:
            logger.error(f"Ошибка при обработке кейса {i}: {e}")
            results["query_results"].append(
                {
                    "question": question,
                    "error": str(e),
                }
            )

    # Вычисление средних метрик
    ndcg_scores = results["metrics"]["ndcg_scores"]
    mrr_scores = results["metrics"]["mrr_scores"]
    hit_scores = results["metrics"]["hit_at_k_scores"]
    faith_scores = results["metrics"]["faithfulness_scores"]

    results["aggregate_metrics"] = {
        "ndcg_mean": sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0,
        "ndcg_std": _calculate_std(ndcg_scores),
        "mrr_mean": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0,
        "hit_at_k_mean": sum(hit_scores) / len(hit_scores) if hit_scores else 0,
        "faithfulness_mean": sum(faith_scores) / len(faith_scores) if faith_scores else 0,
    }

    logger.info("Оценка завершена")
    logger.info(f"NDCG@{top_k}: {results['aggregate_metrics']['ndcg_mean']:.3f}")
    logger.info(f"MRR: {results['aggregate_metrics']['mrr_mean']:.3f}")
    logger.info(f"Hit@{top_k}: {results['aggregate_metrics']['hit_at_k_mean']:.3f}")
    logger.info(f"Faithfulness: {results['aggregate_metrics']['faithfulness_mean']:.3f}")

    return results


def _calculate_std(values: list[float]) -> float:
    """Вычислить стандартное отклонение."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance**0.5


def save_results_json(results: dict[str, Any], output_path: str) -> None:
    """Сохранить результаты в JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Результаты сохранены в {output_path}")


def save_results_markdown(results: dict[str, Any], output_path: str) -> None:
    """Сохранить результаты в Markdown."""
    agg = results.get("aggregate_metrics", {})

    md_content = f"""# Отчет об оценке качества RAG системы

**Дата:** {results.get('timestamp', 'N/A')}
**Режим поиска:** {results.get('mode', 'N/A')}
**Количество запросов:** {results.get('total_queries', 0)}
**Top-K:** {results.get('top_k', 5)}

## Агрегированные метрики

| Метрика | Значение |
|---------|----------|
| NDCG@{results.get('top_k', 5)} | {agg.get('ndcg_mean', 0):.4f} (±{agg.get('ndcg_std', 0):.4f}) |
| MRR | {agg.get('mrr_mean', 0):.4f} |
| Hit@{results.get('top_k', 5)} | {agg.get('hit_at_k_mean', 0):.4f} |
| Faithfulness | {agg.get('faithfulness_mean', 0):.4f} |

## Детальные результаты по запросам

| # | Вопрос | NDCG | MRR | Hit@K | Faithfulness |
|---|--------|------|-----|-------|--------------|
"""

    for i, qr in enumerate(results.get("query_results", []), 1):
        if "error" in qr:
            md_content += f"| {i} | {qr['question'][:30]}... | - | - | - | ❌ Ошибка |\n"
        else:
            md_content += (
                f"| {i} | {qr['question'][:30]}... | "
                f"{qr.get('ndcg', 0):.3f} | "
                f"{qr.get('mrr', 0):.3f} | "
                f"{qr.get('hit_at_k', 0):.3f} | "
                f"{qr.get('faithfulness', 0):.3f} |\n"
            )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"Markdown отчет сохранен в {output_path}")


async def main() -> None:
    """Точка входа скрипта."""
    parser = argparse.ArgumentParser(
        description="Оценка качества RAG системы",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python scripts/evaluate.py --dataset data/test_queries.json
  python scripts/evaluate.py --dataset data/test_queries.json --mode dense --top-k 10
  python scripts/evaluate.py --dataset data/test_queries.json --output reports/
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Путь к JSON файлу с тестовыми запросами",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_logs",
        help="Директория для вывода результатов (по умолчанию: eval_logs)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dense", "sparse", "hybrid"],
        default="hybrid",
        help="Режим поиска (по умолчанию: hybrid)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Количество retrieved документов (по умолчанию: 5)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "markdown", "both"],
        default="both",
        help="Формат вывода результатов (по умолчанию: both)",
    )

    args = parser.parse_args()

    # Создание директории вывода
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Загрузка датасета
    dataset = load_dataset(args.dataset)

    # Запуск оценки
    results = await run_evaluation(
        dataset=dataset,
        output_dir=args.output,
        top_k=args.top_k,
        mode=args.mode,
    )

    # Сохранение результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.format in ["json", "both"]:
        json_path = output_path / f"eval_results_{timestamp}.json"
        save_results_json(results, str(json_path))

    if args.format in ["markdown", "both"]:
        md_path = output_path / f"eval_report_{timestamp}.md"
        save_results_markdown(results, str(md_path))

    print("\n" + "=" * 60)
    print("✅ Оценка завершена!")
    print("=" * 60)
    print(f"NDCG@{args.top_k}: {results['aggregate_metrics']['ndcg_mean']:.4f}")
    print(f"MRR: {results['aggregate_metrics']['mrr_mean']:.4f}")
    print(f"Hit@{args.top_k}: {results['aggregate_metrics']['hit_at_k_mean']:.4f}")
    print(f"Faithfulness: {results['aggregate_metrics']['faithfulness_mean']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
