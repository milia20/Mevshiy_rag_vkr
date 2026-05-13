"""
Qdrant indexer for multiple retrieval strategies.

Supports:
- Dense vectors (sentence-transformers)
- Sparse vectors (BM25-style)
- Hybrid (dense + sparse)
- Multi-vector / ColBERT (late interaction)
"""

from dataclasses import dataclass

from loguru import logger
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from ..preprocessing.chunker import Chunk


@dataclass
class IndexingConfig:
    """Configuration for Qdrant indexing."""

    # Collection names
    dense_collection: str = "dense_collection"
    sparse_collection: str = "sparse_collection"
    hybrid_collection: str = "hybrid_collection"
    colbert_collection: str = "hybrid_with_colbert"

    # Vector configurations
    dense_vector_size: int = 384  # all-MiniLM-L6-v2
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dense_distance: str = "COSINE"

    # Alternative dense models
    bge_small_model: str = "BAAI/bge-small-en-v1.5"
    bge_small_vector_size: int = 512
    bge_m3_model: str = "BAAI/bge-m3"
    bge_m3_vector_size: int = 1024

    # Sparse configuration
    sparse_vector_name: str = "bm25"
    sparse_model: str = "prithivida/Splade_PP_en_v1"
    use_idf: bool = True

    # ColBERT configuration
    colbert_vector_size: int = 128
    colbert_vector_name: str = "late_interaction"
    colbert_model: str = "jinaai/jina-colbert-v2"

    # Qdrant settings
    qdrant_url: str = "http://localhost:6333"
    recreate_collections: bool = True
    batch_size: int = 128

    # HNSW config for ColBERT
    hnsw_m: int = 0  # For ColBERT, typically 0

    # Model selection
    use_bge_small: bool = False
    use_bge_m3: bool = False
    use_splade: bool = False


class QdrantIndexer:
    """
    Indexer for multiple retrieval strategies in Qdrant.

    Creates and manages collections for:
    - Dense-only search
    - Sparse-only search (BM25)
    - Hybrid search (dense + sparse)
    - ColBERT-style late interaction
    """

    def __init__(self, config: IndexingConfig | None = None):
        """
        Initialize the indexer.

        Args:
            config: Indexing configuration
        """
        self.config = config or IndexingConfig()
        self.client = QdrantClient(url=self.config.qdrant_url)
        self._dense_encoder: SentenceTransformer | None = None

    @property
    def dense_encoder(self) -> SentenceTransformer:
        """Lazy-load dense encoder model based on configuration."""
        if self._dense_encoder is None:
            if self.config.use_bge_small:
                model_name = self.config.bge_small_model
                logger.info(f"Loading BGE-small encoder: {model_name}")
            elif self.config.use_bge_m3:
                model_name = self.config.bge_m3_model
                logger.info(f"Loading BGE-m3 encoder: {model_name}")
            else:
                model_name = self.config.dense_model
                logger.info(f"Loading dense encoder: {model_name}")
            self._dense_encoder = SentenceTransformer(model_name)
        return self._dense_encoder

    def _get_distance_model(self, distance_str: str) -> models.Distance:
        """Convert distance string to Qdrant Distance enum."""
        return getattr(models.Distance, distance_str.upper())

    def create_dense_collection(self) -> None:
        """Create collection for dense-only search."""
        # Check if collection exists and handle based on recreate flag
        if self.client.collection_exists(self.config.dense_collection):
            if self.config.recreate_collections:
                self.client.delete_collection(self.config.dense_collection)
                logger.info("Deleted existing dense collection")
            else:
                logger.info("Dense collection already exists, skipping creation")
                return

        # Determine vector size based on selected model
        if self.config.use_bge_small:
            vector_size = self.config.bge_small_vector_size
        elif self.config.use_bge_m3:
            vector_size = self.config.bge_m3_vector_size
        else:
            vector_size = self.config.dense_vector_size

        self.client.create_collection(
            collection_name=self.config.dense_collection,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=self._get_distance_model(self.config.dense_distance),
            ),
        )
        logger.info(f"Created dense collection: {self.config.dense_collection} with vector size {vector_size}")

    def create_sparse_collection(self) -> None:
        """Create collection for sparse-only search."""
        # Check if collection exists and handle based on recreate flag
        if self.client.collection_exists(self.config.sparse_collection):
            if self.config.recreate_collections:
                self.client.delete_collection(self.config.sparse_collection)
                logger.info("Deleted existing sparse collection")
            else:
                logger.info("Sparse collection already exists, skipping creation")
                return

        self.client.create_collection(
            collection_name=self.config.sparse_collection,
            vectors_config={},  # No dense vectors
            sparse_vectors_config={
                self.config.sparse_vector_name: models.SparseVectorParams(
                    modifier=models.Modifier.IDF if self.config.use_idf else None
                )
            },
        )
        logger.info(f"Created sparse collection: {self.config.sparse_collection}")

    def create_hybrid_collection(self) -> None:
        """Create collection for hybrid search (dense + sparse)."""
        # Check if collection exists and handle based on recreate flag
        if self.client.collection_exists(self.config.hybrid_collection):
            if self.config.recreate_collections:
                self.client.delete_collection(self.config.hybrid_collection)
                logger.info("Deleted existing hybrid collection")
            else:
                logger.info("Hybrid collection already exists, skipping creation")
                return

        self.client.create_collection(
            collection_name=self.config.hybrid_collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=self.config.dense_vector_size,
                    distance=self._get_distance_model(self.config.dense_distance),
                )
            },
            sparse_vectors_config={
                self.config.sparse_vector_name: models.SparseVectorParams(
                    modifier=models.Modifier.IDF if self.config.use_idf else None
                )
            },
        )
        logger.info(f"Created hybrid collection: {self.config.hybrid_collection}")

    def create_colbert_collection(self) -> None:
        """Create collection for ColBERT-style late interaction."""
        # Check if collection exists and handle based on recreate flag
        if self.client.collection_exists(self.config.colbert_collection):
            if self.config.recreate_collections:
                self.client.delete_collection(self.config.colbert_collection)
                logger.info("Deleted existing ColBERT collection")
            else:
                logger.info("ColBERT collection already exists, skipping creation")
                return

        self.client.create_collection(
            collection_name=self.config.colbert_collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=self.config.dense_vector_size,
                    distance=self._get_distance_model(self.config.dense_distance),
                ),
                self.config.colbert_vector_name: models.VectorParams(
                    size=self.config.colbert_vector_size,
                    distance=self._get_distance_model(self.config.dense_distance),
                    multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
                    hnsw_config=models.HnswConfigDiff(m=self.config.hnsw_m),
                ),
            },
            sparse_vectors_config={
                self.config.sparse_vector_name: models.SparseVectorParams(
                    modifier=models.Modifier.IDF if self.config.use_idf else None
                )
            },
        )
        logger.info(f"Created ColBERT collection: {self.config.colbert_collection}")

    def create_all_collections(self) -> None:
        """Create all collections for different retrieval strategies."""
        self.create_dense_collection()
        self.create_sparse_collection()
        self.create_hybrid_collection()
        self.create_colbert_collection()
        logger.info("All collections created successfully")

    def encode_dense(self, texts: list[str]) -> list[list[float]]:
        """Encode texts into dense vectors."""
        embeddings = self.dense_encoder.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def _text_to_sparse_indices(self, text: str) -> tuple[list[int], list[float]]:
        """
        Convert text to sparse vector indices and values.

        Simple tokenization with term frequency weighting.
        In production, consider using a proper BM25 implementation.
        """
        from collections import Counter

        tokens = text.lower().split()
        if not tokens:
            return [], []

        # Count term frequencies
        tf = Counter(tokens)

        # Use hash-based indexing for vocabulary-free approach
        indices = []
        values = []
        for token, freq in tf.items():
            # Hash token to index (simple approach)
            idx = abs(hash(token)) % (10**9)
            indices.append(idx)
            values.append(float(freq))

        return indices, values

    def index_dense(self, chunks: list[Chunk]) -> None:
        """Index chunks into dense collection."""
        if not chunks:
            logger.warning("No chunks to index")
            return

        logger.info(f"Indexing {len(chunks)} chunks into dense collection")

        points = []
        for i in range(0, len(chunks), self.config.batch_size):
            batch = chunks[i : i + self.config.batch_size]
            texts = [chunk.text for chunk in batch]

            # Encode dense vectors
            vectors = self.encode_dense(texts)

            # Create points
            for chunk, vector in zip(batch, vectors):
                points.append(
                    models.PointStruct(
                        id=chunk.chunk_id,
                        vector=vector,
                        payload={
                            "text": chunk.text,
                            **chunk.metadata,
                        },
                    )
                )

            # Upsert batch
            self.client.upsert(
                collection_name=self.config.dense_collection,
                points=points,
            )
            points = []

        logger.info("Dense indexing complete")

    def index_sparse(self, chunks: list[Chunk]) -> None:
        """Index chunks into sparse collection."""
        if not chunks:
            logger.warning("No chunks to index")
            return

        logger.info(f"Indexing {len(chunks)} chunks into sparse collection")

        points = []
        for chunk in tqdm(chunks, desc="Indexing sparse vectors"):
            indices, values = self._text_to_sparse_indices(chunk.text)

            if not indices:
                continue

            sparse_vector = models.SparseVector(indices=indices, values=values)

            points.append(
                models.PointStruct(
                    id=chunk.chunk_id,
                    vector={self.config.sparse_vector_name: sparse_vector},
                    payload={
                        "text": chunk.text,
                        **chunk.metadata,
                    },
                )
            )

            if len(points) >= self.config.batch_size:
                self.client.upsert(
                    collection_name=self.config.sparse_collection,
                    points=points,
                )
                points = []

        if points:
            self.client.upsert(
                collection_name=self.config.sparse_collection,
                points=points,
            )

        logger.info("Sparse indexing complete")

    def index_hybrid(self, chunks: list[Chunk]) -> None:
        """Index chunks into hybrid collection (dense + sparse)."""
        if not chunks:
            logger.warning("No chunks to index")
            return

        logger.info(f"Indexing {len(chunks)} chunks into hybrid collection")

        points = []
        for i in range(0, len(chunks), self.config.batch_size):
            batch = chunks[i : i + self.config.batch_size]
            texts = [chunk.text for chunk in batch]

            # Encode dense vectors
            dense_vectors = self.encode_dense(texts)

            # Create points with both dense and sparse vectors
            for chunk, dense_vec in zip(batch, dense_vectors):
                indices, values = self._text_to_sparse_indices(chunk.text)

                if not indices:
                    continue

                sparse_vector = models.SparseVector(indices=indices, values=values)

                points.append(
                    models.PointStruct(
                        id=chunk.chunk_id,
                        vector={
                            "dense": dense_vec,
                            self.config.sparse_vector_name: sparse_vector,
                        },
                        payload={
                            "text": chunk.text,
                            **chunk.metadata,
                        },
                    )
                )

            if len(points) >= self.config.batch_size:
                self.client.upsert(
                    collection_name=self.config.hybrid_collection,
                    points=points,
                )
                points = []

        if points:
            self.client.upsert(
                collection_name=self.config.hybrid_collection,
                points=points,
            )

        logger.info("Hybrid indexing complete")

    def index_colbert(self, chunks: list[Chunk]) -> None:
        """
        Index chunks into ColBERT collection.

        Note: This is a simplified version. Full ColBERT requires
        token-level embeddings which need the actual ColBERT model.
        """
        if not chunks:
            logger.warning("No chunks to index")
            return

        logger.info(f"Indexing {len(chunks)} chunks into ColBERT collection")

        # For now, we'll use dense embeddings as a placeholder
        # In production, use actual ColBERT model for token-level embeddings
        points = []
        for i in range(0, len(chunks), self.config.batch_size):
            batch = chunks[i : i + self.config.batch_size]
            texts = [chunk.text for chunk in batch]

            # Encode dense vectors
            dense_vectors = self.encode_dense(texts)

            # Create points with dense, ColBERT, and sparse vectors
            for chunk, dense_vec in zip(batch, dense_vectors):
                indices, values = self._text_to_sparse_indices(chunk.text)

                # Placeholder for ColBERT vectors (in production, use actual ColBERT)
                # Using single vector as approximation
                colbert_vec = dense_vec[: self.config.colbert_vector_size]

                if not indices:
                    continue

                sparse_vector = models.SparseVector(indices=indices, values=values)

                points.append(
                    models.PointStruct(
                        id=chunk.chunk_id,
                        vector={
                            "dense": dense_vec,
                            self.config.colbert_vector_name: colbert_vec,
                            self.config.sparse_vector_name: sparse_vector,
                        },
                        payload={
                            "text": chunk.text,
                            **chunk.metadata,
                        },
                    )
                )

            if len(points) >= self.config.batch_size:
                self.client.upsert(
                    collection_name=self.config.colbert_collection,
                    points=points,
                )
                points = []

        if points:
            self.client.upsert(
                collection_name=self.config.colbert_collection,
                points=points,
            )

        logger.info("ColBERT indexing complete")

    def index_all(self, chunks: list[Chunk], resume_from: int = 0) -> None:
        """Index chunks into all collections with resume capability."""
        if resume_from > 0:
            logger.info(f"Resuming indexing from chunk {resume_from}")
            chunks = chunks[resume_from:]
            logger.info(f"Remaining chunks to index: {len(chunks)}")

        self.index_dense(chunks)
        self.index_sparse(chunks)
        self.index_hybrid(chunks)
        self.index_colbert(chunks)
        logger.info("All indexing complete")
