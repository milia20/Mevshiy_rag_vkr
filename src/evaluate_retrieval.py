"""
Retrieval evaluation script for RAG systems using Qdrant.
Supports dense, sparse, and hybrid search with metrics: Hit@k, Recall@k, MRR, NDCG@k.

USAGE (no arguments needed):
    python src/evaluate_retrieval.py

To customize paths/models: edit CONFIG section at the bottom of this file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


# ============================================================================
# CONFIGURATION — редактируйте здесь, а не в аргументах командной строки
# ============================================================================


@dataclass
class RunConfig:
    """Hardcoded configuration for immediate execution."""

    # === Пути к данным ===
    chunks_path: str = Path(__file__).parent / "custom_dataset/agent-framework.jsonl"
    gt_paths: dict[str, str] = field(
        default_factory=lambda: {
            "all_q": Path(__file__).parent.parent / "datasets/all_q_True.csv",
            "nq": Path(__file__).parent.parent / "datasets/nq_True.csv",
        }
    )
    output_dir: str = "results"

    # === Qdrant ===
    qdrant_url: str = "http://localhost:6333"
    collection_prefix: str = "eval"
    recreate_collection: bool = True

    # === Модели ===
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_vocab_size: int = 100_000  # для хеширования токенов

    # === Поиск ===
    model_types: list[str] = field(default_factory=lambda: ["dense", "hybrid"])
    top_k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    rrf_k: int = 60  # параметр RRF для гибридного поиска
    batch_size: int = 128

    # === Метрики ===
    metrics: list[str] = field(default_factory=lambda: ["hit", "recall", "mrr", "ndcg"])


# Глобальный конфиг — единственный источник правды
CFG = RunConfig()

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ============================================================================
# UTILITIES
# ============================================================================


def token_to_index(token: str, vocab_size: int) -> int:
    """Стабильное хеширование токена в индекс [0, vocab_size)."""
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % vocab_size


def ensure_list_of_strings(val: Any) -> list[str]:
    """Нормализует значение в список строк для relevant_chunk_ids."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, str):
        return [val] if val.strip() else []
    if isinstance(val, list):
        return [str(v).strip() for v in val if str(v).strip()]
    return [str(val).strip()]


# ============================================================================
# METRICS CALCULATOR
# ============================================================================


class MetricsCalculator:
    """Расчёт метрик retrieval: Hit@k, Recall@k, MRR, NDCG@k."""

    @staticmethod
    def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> int:
        return int(bool(set(retrieved[:k]) & relevant))

    @staticmethod
    def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        if not relevant:
            return 0.0
        return len(set(retrieved[:k]) & relevant) / len(relevant)

    @staticmethod
    def mrr(retrieved: list[str], relevant: set[str]) -> float:
        for idx, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return 1.0 / idx
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        if not relevant:
            return 0.0
        dcg = sum(1.0 / np.log2(idx + 2) for idx, doc_id in enumerate(retrieved[:k]) if doc_id in relevant)
        ideal = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal))
        return dcg / idcg if idcg > 0 else 0.0


# ============================================================================
# DENSE ENCODER
# ============================================================================


class DenseEncoder:
    """Wrapper для sentence-transformers с кэшированием модели."""

    _models: dict[str, SentenceTransformer] = {}

    def __init__(self, model_name: str):
        if model_name not in self._models:
            logger.info("Loading dense model: %s", model_name)
            self._models[model_name] = SentenceTransformer(model_name)
        self.model = self._models[model_name]
        self.dim = self.model.get_embedding_dimension()

    def encode(self, texts: str | list[str], batch_size: int = 32) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)


# ============================================================================
# SPARSE ENCODER (TOKEN HASHING)
# ============================================================================


class SparseEncoder:
    """Простой sparse encoder на основе хеширования токенов."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> models.SparseVector:
        tokens = text.lower().split()
        token_counts = Counter(tokens)
        if not token_counts:
            return models.SparseVector(indices=[], values=[])
        return models.SparseVector(
            indices=[token_to_index(t, self.vocab_size) for t in token_counts], values=list(token_counts.values())
        )


# ============================================================================
# GROUND TRUTH LOADER
# ============================================================================


def load_ground_truth(gt_path: str | Path) -> pd.DataFrame:
    """Загружает и нормализует ground truth из CSV/Parquet."""
    path = Path(gt_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth not found: {path}")

    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

    # Нормализация колонок
    df.columns = df.columns.str.lower().str.strip()

    # Обязательная колонка
    if "question" not in df.columns:
        raise ValueError(f"Ground truth must have 'question' column. Found: {list(df.columns)}")

    # Нормализация relevant_chunk_ids
    if "relevant_chunk_ids" in df.columns:
        df["relevant_chunk_ids"] = df["relevant_chunk_ids"].apply(ensure_list_of_strings)
    elif "chunk_id" in df.columns:
        df["relevant_chunk_ids"] = df["chunk_id"].apply(lambda x: ensure_list_of_strings(x) if pd.notna(x) else [])
    elif "question_id" in df.columns:
        df["relevant_chunk_ids"] = df["question_id"].astype(str).apply(ensure_list_of_strings)
    else:
        logger.warning("No chunk_id column found — using empty relevant sets")
        df["relevant_chunk_ids"] = [[] for _ in range(len(df))]

    # Уникальный ID для вопроса
    if "question_id" not in df.columns:
        df["question_id"] = df.index.astype(str)

    logger.info("Loaded %d queries from %s", len(df), path.name)
    return df


def convert_gt_to_parquet(csv_path: str, output_path: str) -> None:
    """Конвертирует CSV ground truth в parquet с нормализованной схемой."""
    df = load_ground_truth(csv_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Saved normalized ground truth to %s", output_path)


# ============================================================================
# CHUNKS LOADER
# ============================================================================


def load_chunks(chunks_path: str | Path) -> list[dict[str, Any]]:
    """Загружает чанки из JSONL."""
    path = Path(chunks_path)
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                # Гарантируем наличие chunk_id
                if "chunk_id" not in obj.get("metadata", {}):
                    obj.setdefault("metadata", {})["chunk_id"] = f"chunk_{line_num}"
                chunks.append(obj)
            except json.JSONDecodeError as e:
                logger.warning("Skipping invalid JSON at line %d: %s", line_num, e)

    logger.info("Loaded %d chunks from %s", len(chunks), path.name)
    return chunks


# ============================================================================
# QDRANT MANAGER
# ============================================================================


class QdrantManager:
    """Управление коллекциями Qdrant: создание, индексация, поиск."""

    def __init__(self, url: str, collection_name: str):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name

    def create_collection(self, vector_dim: int, use_sparse: bool = False, recreate: bool = True) -> None:
        """Создаёт коллекцию с корректной конфигурацией для dense/sparse/hybrid."""
        if recreate and self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
            logger.info("Deleted existing collection: %s", self.collection_name)

        if self.client.collection_exists(self.collection_name):
            logger.info("Collection already exists: %s", self.collection_name)
            return

        # 🔧 ВСЕГДА используем именованные векторы для консистентности
        # Даже для dense-only — это упрощает код и избегает ошибок конверсии
        if use_sparse:
            vectors_config = {
                "dense": models.VectorParams(size=vector_dim, distance=models.Distance.COSINE),
                "sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False)),
            }
        else:
            # Dense-only: всё равно используем именованный вектор "dense"
            vectors_config = {"dense": models.VectorParams(size=vector_dim, distance=models.Distance.COSINE)}

        self.client.create_collection(collection_name=self.collection_name, vectors_config=vectors_config)
        logger.info("Created collection: %s (vectors=%s)", self.collection_name, list(vectors_config.keys()))

    def index_chunks(
        self,
        chunks: list[dict[str, Any]],
        dense_encoder: DenseEncoder | None,
        sparse_encoder: SparseEncoder | None,
        batch_size: int = 128,
    ) -> None:
        """Индексирует чанки в Qdrant с поддержкой dense/sparse векторов."""
        points = []
        use_sparse = sparse_encoder is not None

        for chunk in tqdm(chunks, desc="Indexing chunks", unit="chunk"):
            chunk_id = chunk.get("metadata", {}).get("chunk_id")
            if not chunk_id:
                continue
            text = chunk.get("text", "")

            # Dense вектор
            vector = None
            if dense_encoder:
                vector = dense_encoder.encode(text).tolist()

            # Sparse вектор
            sparse_vector = None
            if sparse_encoder:
                sparse_vector = sparse_encoder.encode(text)

            # 🔧 Формируем dict с именованными векторами
            vector_dict: dict[str, list[float] | models.SparseVector] = {}
            if vector is not None:
                vector_dict["dense"] = vector
            if sparse_vector is not None:
                vector_dict["sparse"] = sparse_vector

            # 🔧 КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: используем vector=... (не vectors=...)
            # Qdrant принимает dict в поле vector для именованных векторов
            point = models.PointStruct(
                id=chunk_id,
                vector=vector_dict,  # ← dict, но параметр называется 'vector'
                payload={
                    "chunk_id": chunk_id,
                    "text": text[:500],
                    "url": chunk.get("metadata", {}).get("url", ""),
                    "section": (
                        chunk.get("metadata", {}).get("headers", [""])[0]
                        if chunk.get("metadata", {}).get("headers")
                        else ""
                    ),
                },
            )
            points.append(point)

            if len(points) >= batch_size:
                self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
                points = []

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        logger.info("Indexed %d chunks into %s", len(chunks), self.collection_name)

    def search_dense(self, query_vector: list[float], top_k: int, with_payload: bool = False) -> list[dict[str, Any]]:
        """Поиск по dense вектору."""
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,  # ← просто список для dense
            limit=top_k,
            with_payload=with_payload,
            using="dense",  # ← указываем имя вектора
        ).points
        return [{"id": r.id, "score": r.score, **(r.payload or {})} for r in results]

    def search_sparse(
        self, query_sparse: models.SparseVector, top_k: int, with_payload: bool = False
    ) -> list[dict[str, Any]]:
        """Поиск по sparse вектору."""
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_sparse,  # ← SparseVector объект
            limit=top_k,
            with_payload=with_payload,
            using="sparse",
        ).points
        return [{"id": r.id, "score": r.score, **(r.payload or {})} for r in results]

    def search_dense(self, query_vector: list[float], top_k: int, with_payload: bool = False) -> list[dict[str, Any]]:
        """Поиск по dense вектору."""
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,  # ← просто список для dense
            limit=top_k,
            with_payload=with_payload,
            using="dense",  # ← указываем имя вектора
        ).points
        return [{"id": r.id, "score": r.score, **(r.payload or {})} for r in results]

    def search_sparse(
        self, query_sparse: models.SparseVector, top_k: int, with_payload: bool = False
    ) -> list[dict[str, Any]]:
        """Поиск по sparse вектору."""
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_sparse,  # ← SparseVector объект
            limit=top_k,
            with_payload=with_payload,
            using="sparse",
        ).points
        return [{"id": r.id, "score": r.score, **(r.payload or {})} for r in results]

    def search_hybrid_rrf(
        self,
        query_vector: list[float],
        query_sparse: models.SparseVector,
        top_k: int,
        rrf_k: int = 60,
        with_payload: bool = False,
    ) -> list[dict[str, Any]]:
        """Гибридный поиск с Reciprocal Rank Fusion."""
        # Dense поиск
        dense_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k * 2,
            with_payload=with_payload,
            using="dense",
        ).points

        # Sparse поиск
        sparse_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_sparse,
            limit=top_k * 2,
            with_payload=with_payload,
            using="sparse",
        ).points

        # RRF scoring
        scores: dict[str, float] = {}
        payloads: dict[str, dict] = {}

        for rank, r in enumerate(dense_results, start=1):
            scores[r.id] = scores.get(r.id, 0) + 1.0 / (rank + rrf_k)
            if r.payload:
                payloads[r.id] = r.payload

        for rank, r in enumerate(sparse_results, start=1):
            scores[r.id] = scores.get(r.id, 0) + 1.0 / (rank + rrf_k)
            if r.payload:
                payloads[r.id] = r.payload

        # Сортировка и возврат топ-k
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
        return [{"id": doc_id, "score": scores[doc_id], **(payloads.get(doc_id) or {})} for doc_id in sorted_ids]


# ============================================================================
# EVALUATION ENGINE
# ============================================================================


def evaluate_model(
    qdrant_mgr: QdrantManager,
    gt_df: pd.DataFrame,
    dense_encoder: DenseEncoder | None,
    sparse_encoder: SparseEncoder | None,
    model_type: str,
    top_k_values: list[int],
    rrf_k: int = 60,
) -> dict[str, Any]:
    """Запускает оценку для одного типа модели."""
    metrics_calc = MetricsCalculator()
    results = {k: [] for k in top_k_values}
    mrr_scores = []
    times = []

    max_k = max(top_k_values)

    for _, row in tqdm(gt_df.iterrows(), total=len(gt_df), desc=f"Evaluating {model_type}"):
        question = row["question"]
        relevant_ids = set(row["relevant_chunk_ids"])

        t0 = time.perf_counter()

        # Кодирование запроса
        query_dense = dense_encoder.encode(question).tolist() if dense_encoder else None
        query_sparse = sparse_encoder.encode(question) if sparse_encoder else None

        # Поиск
        if model_type == "dense":
            retrieved = qdrant_mgr.search_dense(query_dense, max_k)
        elif model_type == "sparse":
            retrieved = qdrant_mgr.search_sparse(query_sparse, max_k)
        elif model_type == "hybrid":
            retrieved = qdrant_mgr.search_hybrid_rrf(query_dense, query_sparse, max_k, rrf_k=rrf_k)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        retrieved_ids = [r["id"] for r in retrieved]

        # Расчёт метрик
        for k in top_k_values:
            results[k].append(
                {
                    "hit": metrics_calc.hit_at_k(retrieved_ids, relevant_ids, k),
                    "recall": metrics_calc.recall_at_k(retrieved_ids, relevant_ids, k),
                    "ndcg": metrics_calc.ndcg_at_k(retrieved_ids, relevant_ids, k),
                }
            )
        mrr_scores.append(metrics_calc.mrr(retrieved_ids, relevant_ids))

    # Агрегация
    aggregated = {}
    for k in top_k_values:
        aggregated[f"hit@{k}"] = np.mean([r["hit"] for r in results[k]])
        aggregated[f"recall@{k}"] = np.mean([r["recall"] for r in results[k]])
        aggregated[f"ndcg@{k}"] = np.mean([r["ndcg"] for r in results[k]])

    aggregated["mrr"] = np.mean(mrr_scores)
    aggregated["time_per_query_ms"] = np.mean(times) * 1000

    return aggregated


# ============================================================================
# RESULTS SAVER
# ============================================================================


def save_results(all_results: dict[str, dict[str, Any]], output_dir: str, top_k_values: list[int]) -> None:
    """Сохраняет результаты в JSON и CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # JSON с полными результатами
    json_path = output_path / f"{timestamp}_full_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("Saved full results to %s", json_path)

    # CSV-таблица для сравнения моделей
    rows = []
    for model_name, metrics in all_results.items():
        for k in top_k_values:
            rows.append(
                {
                    "model": model_name,
                    "k": k,
                    "recall@k": metrics.get(f"recall@{k}", 0),
                    "hit@k": metrics.get(f"hit@{k}", 0),
                    "ndcg@k": metrics.get(f"ndcg@{k}", 0),
                    "mrr": metrics.get("mrr", 0),
                    "time_ms": metrics.get("time_per_query_ms", 0),
                }
            )

    csv_path = output_path / f"{timestamp}_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info("Saved summary table to %s", csv_path)

    # Печать в консоль
    print("\n" + "=" * 90)
    print(f"RETRIEVAL EVALUATION SUMMARY ({timestamp})")
    print("=" * 90)
    df_print = pd.DataFrame(rows)
    if not df_print.empty:
        print(df_print.to_string(index=False))
    print("=" * 90)


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main() -> None:
    """Точка входа: выполняет всю пайплайн оценки без аргументов."""

    logger.info("Starting RAG retrieval evaluation")
    logger.info("Config: %s", CFG)

    # === Шаг 0: Конвертация GT в parquet (опционально) ===
    for name, gt_path in CFG.gt_paths.items():
        if Path(gt_path).suffix == ".csv":
            parquet_path = Path(CFG.output_dir) / f"{name}_gt.parquet"
            if not parquet_path.exists():
                logger.info("Converting %s to parquet...", gt_path)
                convert_gt_to_parquet(gt_path, str(parquet_path))

    # === Шаг 1: Загрузка данных ===
    chunks = load_chunks(CFG.chunks_path)
    if not chunks:
        logger.error("No chunks loaded — exiting")
        return

    # === Шаг 2: Инициализация энкодеров ===
    dense_encoder = DenseEncoder(CFG.dense_model) if "dense" in CFG.model_types else None
    sparse_encoder = (
        SparseEncoder(CFG.sparse_vocab_size) if any(t in CFG.model_types for t in ["sparse", "hybrid"]) else None
    )

    # === Шаг 3: Оценка для каждой модели ===
    all_results = {}

    for model_type in CFG.model_types:
        logger.info("\n" + "=" * 60)
        logger.info("Evaluating model: %s", model_type.upper())
        logger.info("=" * 60)

        # Коллекция для этой модели
        collection_name = f"{CFG.collection_prefix}_{model_type}"
        qdrant_mgr = QdrantManager(CFG.qdrant_url, collection_name)

        # Создание коллекции
        use_sparse = model_type in ["sparse", "hybrid"]
        qdrant_mgr.create_collection(
            vector_dim=dense_encoder.dim if dense_encoder else 0,
            use_sparse=use_sparse,
            recreate=CFG.recreate_collection,
        )

        # Индексация (только один раз, можно оптимизировать кэшированием)
        qdrant_mgr.index_chunks(
            chunks,
            dense_encoder=dense_encoder if model_type in ["dense", "hybrid"] else None,
            sparse_encoder=sparse_encoder if model_type in ["sparse", "hybrid"] else None,
            batch_size=CFG.batch_size,
        )

        # Загрузка GT для оценки (берём первый доступный)
        gt_path = next(iter(CFG.gt_paths.values()))
        if Path(gt_path).suffix == ".parquet":
            gt_df = load_ground_truth(gt_path)
        else:
            # Конвертируем на лету
            parquet_path = Path(CFG.output_dir) / f"temp_gt.parquet"
            convert_gt_to_parquet(gt_path, str(parquet_path))
            gt_df = load_ground_truth(parquet_path)

        # Запуск оценки
        metrics = evaluate_model(
            qdrant_mgr=qdrant_mgr,
            gt_df=gt_df,
            dense_encoder=dense_encoder if model_type in ["dense", "hybrid"] else None,
            sparse_encoder=sparse_encoder if model_type in ["sparse", "hybrid"] else None,
            model_type=model_type,
            top_k_values=CFG.top_k_values,
            rrf_k=CFG.rrf_k,
        )

        all_results[model_type] = metrics
        logger.info("Completed %s: MRR=%.3f, Recall@5=%.3f", model_type, metrics["mrr"], metrics.get("recall@5", 0))

    # === Шаг 4: Сохранение результатов ===
    save_results(all_results, CFG.output_dir, CFG.top_k_values)

    logger.info("Evaluation complete. Results saved to %s/", CFG.output_dir)


if __name__ == "__main__":
    main()


from qdrant_client import QdrantClient, models

cl = QdrantClient()

cl.create_collection(
    "dense_collection",
    vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
)

cl.create_collection(
    "sparse_collection",
    sparse_vector_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
)

cl.create_collection(
    "hybrid_collection",
    vectors_config={"dense": models.VectorParams(size=512, distance=models.Distance.COSINE)},
    sparse_vector_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
)

cl.create_collection(
    "hybrid_with_colbert",
    vectors_config={
        "dense": models.VectorParams(size=512, distance=models.Distance.COSINE),
        "late_interaction": models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
            hnsw_config=models.HnswConfigDiff(m=0),
        ),
    },
    sparse_vector_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
)


# how to upsert


cl.upsert(
    "dense_collection",
    points=[
        models.PointStruct(
            id=1,
            vector=[models.Document(text="some text", model="sentence-transformers/all-minilm-l6-v2")],
        )
    ],
)

cl.upsert(
    "sparse_collection",
    points=[
        models.PointStruct(
            id=1,
            vector={"bm25": models.Document(text="some text", model="Qdrant/bm25")},
        )
    ],
)


cl.upsert(
    "hybrid_collection",
    points=[
        models.PointStruct(
            id=1,
            vector={
                "dense": models.Document(text="some text", model="sentence-transformers/all-minilm-l6-v2"),
                "bm25": models.Document(text="some text", model="Qdrant/bm25"),
            },
        )
    ],
)

cl.upsert(
    "hybrid_collection_with_colbert",
    points=[
        models.PointStruct(
            id=1,
            vector={
                "colbert": models.Document(text="some text", model="colbert-ir/colbert-v2.0"),
                "dense": models.Document(text="some text", model="sentence-transformers/all-minilm-l6-v2"),
                "bm25": models.Document(text="some text", model="Qdrant/bm25"),
            },
        )
    ],
)


cl.query_points(
    "dense_collection", query=models.Document(text="my text", model="sentence-transformers/all-minilm-l6-v2")
)

cl.query_points(
    "sparse_collection",
    query=models.Document(text="my text", model="Qdrant/bm25"),
    using="bm25",
)

cl.query_points(
    "hybrid_collection",
    query=models.RrfQuery(rrf=models.Rrf(k=30)),
    prefetch=[
        models.Prefetch(
            query=models.Document(text="my text", model="sentence-transformers/all-minilm-l6-v2"), using="dense"
        ),
        models.Prefetch(query=models.Document(text="my text", model="Qdrant/bm25"), using="bm25"),
    ],
)

cl.query_points(
    "hybrid_collection",
    query=models.FusionQuery(fusion=models.Fusion.DBSF),
    prefetch=[
        models.Prefetch(
            query=models.Document(text="my text", model="sentence-transformers/all-minilm-l6-v2"), using="dense"
        ),
        models.Prefetch(query=models.Document(text="my text", model="Qdrant/bm25"), using="bm25"),
    ],
)

cl.query_points(
    "hybrid_collection_with_colbert",
    query=models.Document(text="my text", model="colbert-ir/colbert-v2.0"),
    prefetch=[
        models.Prefetch(
            query=models.Document(text="my text", model="sentence-transformers/all-minilm-l6-v2"), using="dense"
        ),
        models.Prefetch(query=models.Document(text="my text", model="Qdrant/bm25"), using="bm25"),
    ],
    using="late_interaction",
)
