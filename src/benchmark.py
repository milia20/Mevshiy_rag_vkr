import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import jsonlines
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models


@dataclass
class SearchConfig:
    """Конфигурация для тестирования"""

    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "prithivida/Splade_PP_en_v1"
    colbert_model: str = "jinaai/jina-colbert-v2"
    collection_prefix: str = "benchmark"
    collection_name: str = "documents"  # имя коллекции с уже загруженными данными
    top_k: int = 10
    queries_file: str = "queries.jsonl"  # Путь к файлу с тестовыми запросами
    max_queries: int | None = None  # Ограничить количество запросов для теста (None = все)


@dataclass
class TestQuery:
    """Структура тестового запроса из JSONL"""

    question: str
    answer: str
    expected_file: str
    metadata: dict = field(default_factory=dict)


def load_test_queries(filepath: str, max_queries: int | None = None) -> list[TestQuery]:
    """
    Загружает тестовые запросы из JSONL-файла.

    Формат строки:
    {"question": "...", "answer": "...", "file": "FAQS.md"}
    """
    queries = []
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Файл с запросами не найден: {filepath}")

    with jsonlines.open(path, mode="r") as reader:
        for i, line in enumerate(reader):
            if max_queries and i >= max_queries:
                break

            query = TestQuery(
                question=line.get("question", ""),
                answer=line.get("answer", ""),
                expected_file=line.get("file", ""),
                metadata={k: v for k, v in line.items() if k not in ["question", "answer", "file"]},
            )

            if query.question and query.expected_file:
                queries.append(query)

    print(f"Загружено {len(queries)} тестовых запросов из {filepath}")
    return queries


def load_sample_queries_from_collection(
    client: QdrantClient, collection_name: str, sample_size: int = 20
) -> list[TestQuery]:
    """
    Альтернативный метод: создаёт тестовые запросы из случайных документов в коллекции.
    Использует текст документа как "вопрос" и проверяет поиск по полю 'file'.
    """
    points, _ = client.scroll(
        collection_name=collection_name, limit=sample_size, with_payload=["text", "file", "title"]
    )

    queries = []
    for point in points:
        payload = point.payload or {}
        text = payload.get("text", "")
        file_name = payload.get("file", "")

        if text and file_name:
            # Используем первые 100 символов как "вопрос" для симуляции поиска
            query_text = text[:100].strip()
            queries.append(TestQuery(question=query_text, answer=text, expected_file=file_name))

    return queries


class SearchBenchmark:
    def __init__(self, config: SearchConfig):
        self.config = config
        self.client = QdrantClient("http://localhost:6333")
        self.queries: list[TestQuery] = []

        print("Загрузка моделей эмбеддингов...")
        self.dense_model = TextEmbedding(model_name=config.dense_model)
        self.sparse_model = SparseTextEmbedding(model_name=config.sparse_model)

        self.colbert_model = None  # опционально
        try:
            self.colbert_model = LateInteractionTextEmbedding(model_name=config.colbert_model)
            print("✓ ColBERT модель загружена")
        except Exception as e:
            print(f"⚠ ColBERT не доступен: {e}")

        self.results = {}
        self.collection_name = f"{config.collection_prefix}_{config.collection_name}"

    def _get_collection_name(self, suffix: str = "") -> str:
        """Формирует имя коллекции для конкретного метода поиска"""
        if suffix:
            return f"{self.config.collection_prefix}_{suffix}"
        return self.config.collection_name

    def search_dense(self, query: str, limit: int = 10) -> dict:
        """Поиск только по плотным векторам"""
        query_vec = next(iter(self.dense_model.embed([query])))

        start = time.time()
        result = self.client.query_points(
            self._get_collection_name("dense"),
            query=query_vec.tolist(),
            limit=limit,
        )
        latency = time.time() - start

        return {"method": "dense", "latency": latency, "points": result.points, "query": query}

    def search_sparse(self, query: str, limit: int = 10) -> dict:
        """Поиск только по разреженным векторам"""
        query_vec = next(iter(self.sparse_model.embed([query])))

        start = time.time()
        result = self.client.query_points(
            self._get_collection_name("sparse"),
            query=models.SparseVector(indices=query_vec.indices.tolist(), values=query_vec.values.tolist()),
            using="bm25",
            limit=limit,
        )
        latency = time.time() - start

        return {"method": "sparse", "latency": latency, "points": result.points, "query": query}

    def search_hybrid_rrf(self, query: str, limit: int = 10, rrf_k: int = 30) -> dict:
        """Гибридный поиск с RRF fusion"""
        query_dense = next(iter(self.dense_model.embed([query])))
        query_sparse = next(iter(self.sparse_model.embed([query])))

        start = time.time()
        result = self.client.query_points(
            self._get_collection_name("hybrid"),
            query=models.RrfQuery(rrf=models.Rrf(k=rrf_k)),
            prefetch=[
                models.Prefetch(query=query_dense.tolist(), using="dense", limit=limit * 2),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query_sparse.indices.tolist(), values=query_sparse.values.tolist()
                    ),
                    using="bm25",
                    limit=limit * 2,
                ),
            ],
            limit=limit,
        )
        latency = time.time() - start

        return {
            "method": f"hybrid_rrf_k{rrf_k}",
            "latency": latency,
            "points": result.points,
            "query": query,
            "params": {"rrf_k": rrf_k},
        }

    def search_hybrid_dbsf(self, query: str, limit: int = 10) -> dict:
        """Гибридный поиск с DBSF fusion"""
        query_dense = next(iter(self.dense_model.embed([query])))
        query_sparse = next(iter(self.sparse_model.embed([query])))

        start = time.time()
        result = self.client.query_points(
            self._get_collection_name("hybrid"),
            query=models.FusionQuery(fusion=models.Fusion.DBSF),
            prefetch=[
                models.Prefetch(query=query_dense.tolist(), using="dense", limit=limit * 2),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query_sparse.indices.tolist(), values=query_sparse.values.tolist()
                    ),
                    using="bm25",
                    limit=limit * 2,
                ),
            ],
            limit=limit,
        )
        latency = time.time() - start

        return {"method": "hybrid_dbsf", "latency": latency, "points": result.points, "query": query}

    def search_colbert(self, query: str, limit: int = 10) -> dict:
        """Поиск с ColBERT (late interaction)"""
        if not self.colbert_model:
            return None

        query_dense = next(iter(self.dense_model.embed([query])))
        query_sparse = next(iter(self.sparse_model.embed([query])))
        query_colbert = next(iter(self.colbert_model.embed([query])))

        start = time.time()
        result = self.client.query_points(
            self._get_collection_name("colbert"),
            query=[mv.tolist() for mv in query_colbert],
            using="late_interaction",
            prefetch=[
                models.Prefetch(query=query_dense.tolist(), using="dense", limit=limit * 3),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query_sparse.indices.tolist(), values=query_sparse.values.tolist()
                    ),
                    using="bm25",
                    limit=limit * 3,
                ),
            ],
            limit=limit,
        )
        latency = time.time() - start

        return {"method": "colbert", "latency": latency, "points": result.points, "query": query}

    def calculate_precision_by_file(self, search_result: dict, expected_file: str, top_k: int = 10) -> dict[str, float]:
        """
        Вычисляет метрики качества для результатов поиска.
        Проверяет, содержится ли ожидаемый файл в результатах.

        Returns:
            Dict с precision@k, recall@k, mrr
        """
        if not search_result or not search_result["points"]:
            return {"precision": 0.0, "recall": 0.0, "mrr": 0.0}

        points = search_result["points"][:top_k]
        relevant_found = False
        relevant_position = -1

        for idx, point in enumerate(points):
            payload = point.payload or {}
            doc_file = payload.get("file", "")

            # Сравниваем имя файла (регистрация не учитывается)
            if doc_file.lower() == expected_file.lower():
                relevant_found = True
                relevant_position = idx + 1  # 1-based позиция
                break

        precision = 1.0 / len(points) if relevant_found and len(points) > 0 else 0.0
        recall = 1.0 if relevant_found else 0.0  # Для одного релевантного документа
        mrr = 1.0 / relevant_position if relevant_position > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "mrr": mrr,
            "found_at_position": relevant_position if relevant_found else None,
        }

    def run_benchmark(self, queries: list[TestQuery], limit: int = 10):
        """Запускает полный бенчмарк всех методов поиска"""
        print(f"\n=== Запуск бенчмарка ({len(queries)} запросов) ===")

        methods = ["dense", "sparse", "hybrid_rrf", "hybrid_dbsf"]
        if self.colbert_model:
            methods.append("colbert")

        all_results = {method: [] for method in methods}

        for i, tq in enumerate(queries):
            print(f"\nЗапрос {i + 1}/{len(queries)}")
            print(f"  Вопрос: {tq.question[:100]}{'...' if len(tq.question) > 100 else ''}")
            print(f"  Ожидаемый файл: {tq.expected_file}")

            # Dense
            r = self.search_dense(tq.question, limit)
            metrics = self.calculate_precision_by_file(r, tq.expected_file, limit)
            r.update(metrics)
            all_results["dense"].append(r)
            print(f"  Dense: {r['latency']:.3f}s, P@{limit}: {r['precision']:.2f}, MRR: {r['mrr']:.2f}")

            # Sparse
            r = self.search_sparse(tq.question, limit)
            metrics = self.calculate_precision_by_file(r, tq.expected_file, limit)
            r.update(metrics)
            all_results["sparse"].append(r)
            print(f"  Sparse: {r['latency']:.3f}s, P@{limit}: {r['precision']:.2f}, MRR: {r['mrr']:.2f}")

            # Hybrid RRF
            r = self.search_hybrid_rrf(tq.question, limit, rrf_k=30)
            metrics = self.calculate_precision_by_file(r, tq.expected_file, limit)
            r.update(metrics)
            all_results["hybrid_rrf"].append(r)
            print(f"  Hybrid RRF: {r['latency']:.3f}s, P@{limit}: {r['precision']:.2f}, MRR: {r['mrr']:.2f}")

            # Hybrid DBSF
            r = self.search_hybrid_dbsf(tq.question, limit)
            metrics = self.calculate_precision_by_file(r, tq.expected_file, limit)
            r.update(metrics)
            all_results["hybrid_dbsf"].append(r)
            print(f"  Hybrid DBSF: {r['latency']:.3f}s, P@{limit}: {r['precision']:.2f}, MRR: {r['mrr']:.2f}")

            # ColBERT
            if self.colbert_model:
                r = self.search_colbert(tq.question, limit)
                if r:
                    metrics = self.calculate_precision_by_file(r, tq.expected_file, limit)
                    r.update(metrics)
                    all_results["colbert"].append(r)
                    print(f"  ColBERT: {r['latency']:.3f}s, P@{limit}: {r['precision']:.2f}, MRR: {r['mrr']:.2f}")

        self.results = all_results
        return all_results

    def tune_rrf_parameters(
        self, query: str, expected_file: str, k_values: list[int] | tuple[int] = (10, 20, 30, 50, 100), limit: int = 10
    ):
        """Подбирает оптимальный параметр k для RRF"""
        print("\n=== Подбор параметров RRF ===")
        print(f"Запрос: {query[:80]}{'...' if len(query) > 80 else ''}")

        results = []
        for k in k_values:
            r = self.search_hybrid_rrf(query, limit=limit, rrf_k=k)
            metrics = self.calculate_precision_by_file(r, expected_file, limit)
            r.update(metrics)
            r["params"] = {"rrf_k": k}
            results.append(r)
            print(f"  k={k}: latency={r['latency']:.3f}s, precision={r['precision']:.2f}, mrr={r['mrr']:.2f}")

        best = max(results, key=lambda x: x["mrr"])  # Оптимизируем по MRR
        print(f"\n✓ Оптимальный k={best['params']['rrf_k']} (MRR={best['mrr']:.2f})")

        return best

    def print_summary(self):
        """Выводит сводную таблицу результатов"""
        print("\n" + "=" * 90)
        print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
        print("=" * 90)

        print(f"\n{'Метод':<20} {'Ср. время (мс)':<18} {'P@10':<10} {'MRR':<10} {'Запросов':<10}")
        print("-" * 90)

        for method, results in self.results.items():
            if not results:
                continue

            avg_latency = statistics.mean([r["latency"] for r in results]) * 1000
            avg_precision = statistics.mean([r["precision"] for r in results])
            avg_mrr = statistics.mean([r["mrr"] for r in results])

            print(f"{method:<20} {avg_latency:<18.2f} {avg_precision:<10.2f} {avg_mrr:<10.2f} {len(results):<10}")

        print("=" * 90)

        # Рекомендации
        print("\nРЕКОМЕНДАЦИИ:")
        print("-" * 90)

        valid_results = [(m, r_list) for m, r_list in self.results.items() if r_list]

        if valid_results:
            best_mrr = max(
                [(m, statistics.mean([r["mrr"] for r in r_list])) for m, r_list in valid_results], key=lambda x: x[1]
            )
            best_latency = min(
                [(m, statistics.mean([r["latency"] for r in r_list]) * 1000) for m, r_list in valid_results],
                key=lambda x: x[1],
            )

            print(f"• Лучшее качество (MRR): {best_mrr[0]} (MRR={best_mrr[1]:.2f})")
            print(f"• Лучшая скорость: {best_latency[0]} (latency={best_latency[1]:.2f}ms)")

            if best_mrr[0] in ["hybrid_rrf", "hybrid_dbsf", "colbert"]:
                print("• Гибридный поиск рекомендуется для production RAG систем")
            if best_latency[0] == "dense":
                print("• Dense поиск подходит для высоконагруженных систем с ограниченным бюджетом")

        print("=" * 90)

    def export_results(self, output_path: str = "benchmark_results.json"):
        """Экспортирует результаты в JSON для дальнейшего анализа"""
        serializable = {}

        for method, results in self.results.items():
            serializable[method] = []
            for r in results:
                serializable[method].append(
                    {
                        "method": r["method"],
                        "query": r["query"][:200],  # Обрезаем для читаемости
                        "expected_file": next(
                            (q.expected_file for q in self.queries if q.question == r["query"]), None
                        ),
                        "latency": r["latency"],
                        "precision": r["precision"],
                        "recall": r["recall"],
                        "mrr": r["mrr"],
                        "found_at_position": r.get("found_at_position"),
                        "num_results": len(r["points"]),
                        "params": r.get("params", {}),
                    }
                )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Результаты сохранены в {output_path}")


if __name__ == "__main__":
    config = SearchConfig(
        collection_name="hnsw_optimized",  # Коллекция данными в Qdrant
        queries_file="./custom_dataset/datasets/qna_agent-framework_en.jsonl",  # файл с вопросами
        top_k=10,
        max_queries=50,  # для быстрого теста
    )

    benchmark = SearchBenchmark(config)

    # Загрузка тестовых запросов из JSONL
    try:
        queries = load_test_queries(config.queries_file, max_queries=config.max_queries)
    except FileNotFoundError:
        print(f"⚠ Файл {config.queries_file} не найден. Пробуем загрузить из коллекции...")
        queries = load_sample_queries_from_collection(
            benchmark.client, config.collection_name, sample_size=config.max_queries or 20
        )

    if not queries:
        raise ValueError("Не удалось загрузить тестовые запросы")

    benchmark.queries = queries

    benchmark.run_benchmark(queries, limit=config.top_k)

    # Подбор параметров RRF для первого запроса (опционально)
    if queries:
        benchmark.tune_rrf_parameters(
            query=queries[0].question,
            expected_file=queries[0].expected_file,
            k_values=[10, 20, 30, 50, 100],
            limit=config.top_k,
        )

    benchmark.print_summary()

    benchmark.export_results("benchmark_results.json")
