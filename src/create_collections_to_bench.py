"""
create_benchmark_collections.py
Создаёт коллекции для бенчмарка поиска из JSONL-файла
"""

import json
import logging
import uuid

from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from qdrant_client import QdrantClient, models
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Пути и конфиги
JSONL_PATH = r"D:\P_work\Rag-VKR\src\indexing\processed\agent-framework.jsonl"
COLLECTION_PREFIX = "benchmark"
DENSE_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dim
SPARSE_MODEL = "prithivida/Splade_PP_en_v1"
COLBERT_MODEL = "jinaai/jina-colbert-v2"  # 128 dim, multivector


def generate_point_id(chunk_id: str, fallback_idx: int) -> int:
    """
    Генерирует валидный integer ID для точки Qdrant.
    Если chunk_id не является валидным UUID/integer, хешируем его.
    """
    # Проверяем, является ли chunk_id целым числом
    try:
        return int(chunk_id)
    except (ValueError, TypeError):
        pass

    # Проверяем, является ли chunk_id валидным UUID
    try:
        import uuid
        uuid.UUID(chunk_id)
        # Если UUID валиден, используем его int-представление (первые 64 бита)
        return uuid.UUID(chunk_id).int >> 64
    except (ValueError, TypeError):
        pass

    # fallback: хешируем строку в integer (используем первые 8 байт)
    import hashlib
    hash_bytes = hashlib.md5(str(chunk_id).encode()).digest()[:8]
    return int.from_bytes(hash_bytes, byteorder='big')


def load_jsonl(filepath: str) -> list[dict]:
    """Загружает записи из JSONL-файла"""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def prepare_payload(record: dict, chunk_idx: int) -> dict:
    """
    Преобразует запись из JSONL в формат для Qdrant.
    Ожидает формат: {"question": "...", "answer": "...", "file": "FAQS.md"}
    """
    return {
        "text": f"{record.get('question', '')}\n\n{record.get('answer', '')}",
        "question": record.get("question", ""),
        "answer": record.get("answer", ""),
        "file": record.get("file", "unknown"),  # ← ключевое поле для бенчмарка
        "chunk_id": record.get("chunk_id", str(uuid.uuid4())),
        **{k: v for k, v in record.items() if k not in ["question", "answer", "file", "chunk_id"]}
    }


def create_benchmark_collections(client: QdrantClient, prefix: str, vector_size: int, colbert_size: int = 128):
    """Создаёт 4 коллекции для разных стратегий поиска"""

    # 1. Dense only
    if not client.collection_exists(f"{prefix}_dense"):
        client.create_collection(
            f"{prefix}_dense",
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        logger.info("✓ Создана коллекция: %s_dense", prefix)

    # 2. Sparse only
    if not client.collection_exists(f"{prefix}_sparse"):
        client.create_collection(
            f"{prefix}_sparse",
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        logger.info("✓ Создана коллекция: %s_sparse", prefix)

    # 3. Hybrid (dense + sparse)
    if not client.collection_exists(f"{prefix}_hybrid"):
        client.create_collection(
            f"{prefix}_hybrid",
            vectors_config={"dense": models.VectorParams(size=vector_size, distance=models.Distance.COSINE)},
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        logger.info("✓ Создана коллекция: %s_hybrid", prefix)

    # 4. ColBERT + hybrid (опционально)
    try:
        if not client.collection_exists(f"{prefix}_colbert"):
            client.create_collection(
                f"{prefix}_colbert",
                vectors_config={
                    "dense": models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                    "late_interaction": models.VectorParams(
                        size=colbert_size,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        ),
                    ),
                },
                sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
            logger.info("✓ Создана коллекция: %s_colbert", prefix)
    except Exception as e:
        logger.warning("⚠ Не удалось создать ColBERT коллекцию: %s", e)


def generate_vectors(texts: list[str], dense_model, sparse_model, colbert_model=None):
    """Генерирует все типы векторов для списка текстов"""
    dense_vecs = list(dense_model.embed(texts))
    sparse_vecs = list(sparse_model.embed(texts))
    colbert_vecs = list(colbert_model.embed(texts)) if colbert_model else None
    return dense_vecs, sparse_vecs, colbert_vecs


def main():
    # 1. Загрузка данных
    logger.info("Загрузка данных из %s", JSONL_PATH)
    records = load_jsonl(JSONL_PATH)
    logger.info("✓ Загружено %d записей", len(records))

    # 2. Инициализация клиентов и моделей
    qdrant_client = QdrantClient("http://localhost:6333")

    logger.info("Загрузка моделей эмбеддингов...")
    dense_model = TextEmbedding(model_name=DENSE_MODEL)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)

    colbert_model = None
    try:
        colbert_model = LateInteractionTextEmbedding(model_name=COLBERT_MODEL)
        logger.info("✓ ColBERT модель загружена")
    except Exception as e:
        logger.warning("⚠ ColBERT недоступен: %s", e)

    # 3. Получение размера вектора и создание коллекций
    sample_vec = list(dense_model.embed(["test"]))[0]
    vector_size = len(sample_vec)

    create_benchmark_collections(qdrant_client, COLLECTION_PREFIX, vector_size)

    # 4. Подготовка данных
    logger.info("Подготовка пейлоадов и текстов...")
    payloads = [prepare_payload(r, i) for i, r in enumerate(records)]
    texts = [p["text"] for p in payloads]

    # 5. Генерация векторов (батчами для экономии памяти)
    BATCH_SIZE = 32
    logger.info("Генерация векторов (батчами по %d)...", BATCH_SIZE)

    for batch_start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch_texts = texts[batch_start:batch_start + BATCH_SIZE]
        batch_payloads = payloads[batch_start:batch_start + BATCH_SIZE]

        dense_vecs, sparse_vecs, colbert_vecs = generate_vectors(
            batch_texts, dense_model, sparse_model, colbert_model
        )

        # --- Dense collection ---
        points = [
            models.PointStruct(
                id=p["chunk_id"],
                vector=v.tolist(),
                payload=p
            )
            for p, v in zip(batch_payloads, dense_vecs)
        ]
        qdrant_client.upsert(f"{COLLECTION_PREFIX}_dense", points)

        # --- Sparse collection ---
        points = [
            models.PointStruct(
                id=p["chunk_id"],
                vector={"bm25": models.SparseVector(
                    indices=v.indices.tolist(),
                    values=v.values.tolist()
                )},
                payload=p
            )
            for p, v in zip(batch_payloads, sparse_vecs)
        ]
        qdrant_client.upsert(f"{COLLECTION_PREFIX}_sparse", points)

        # --- Hybrid collection ---
        points = [
            models.PointStruct(
                id=p["chunk_id"],
                vector={
                    "dense": dv.tolist(),
                    "bm25": models.SparseVector(
                        indices=sv.indices.tolist(),
                        values=sv.values.tolist()
                    )
                },
                payload=p
            )
            for p, dv, sv in zip(batch_payloads, dense_vecs, sparse_vecs)
        ]
        qdrant_client.upsert(f"{COLLECTION_PREFIX}_hybrid", points)

        # --- ColBERT collection (если доступен) ---
        if colbert_vecs and colbert_model:
            points = [
                models.PointStruct(
                    id=p["chunk_id"],
                    vector={
                        "dense": dv.tolist(),
                        "late_interaction": [mv.tolist() for mv in cv],
                        "bm25": models.SparseVector(
                            indices=sv.indices.tolist(),
                            values=sv.values.tolist()
                        )
                    },
                    payload=p
                )
                for p, dv, sv, cv in zip(batch_payloads, dense_vecs, sparse_vecs, colbert_vecs)
            ]
            qdrant_client.upsert(f"{COLLECTION_PREFIX}_colbert", points)

    # 6. Создание индексов для пейлоада (для фильтрации в бенчмарке)
    logger.info("Создание индексов для пейлоада...")
    for field in ["file", "chunk_id"]:
        for suffix in ["_dense", "_sparse", "_hybrid", "_colbert"]:
            coll_name = f"{COLLECTION_PREFIX}{suffix}"
            if qdrant_client.collection_exists(coll_name):
                try:
                    qdrant_client.create_payload_index(
                        collection_name=coll_name,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass  # Индекс может уже существовать

    # 7. Финальная проверка
    logger.info("\n📊 Статистика коллекций:")
    for suffix in ["_dense", "_sparse", "_hybrid", "_colbert"]:
        name = f"{COLLECTION_PREFIX}{suffix}"
        if qdrant_client.collection_exists(name):
            info = qdrant_client.get_collection(name)
            logger.info("  %s: %d точек", name, info.points_count)

    logger.info("\n✅ Готово! Коллекции для бенчмарка созданы.")
    logger.info("Используйте их в бенчмарке с collection_prefix='%s'", COLLECTION_PREFIX)


if __name__ == "__main__":
    global_point_idx = 0
    qdrant_client = QdrantClient("http://localhost:6333")
    for suffix in ["_dense", "_sparse", "_hybrid", "_colbert"]:
        name = f"{COLLECTION_PREFIX}{suffix}"
        if qdrant_client.collection_exists(name):
            qdrant_client.delete_collection(name)
            logger.info("🗑 Удалена старая коллекция: %s", name)
    main()
