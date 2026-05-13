"""
Document chunking with metadata preservation.

This module provides functionality to:
- Load documents from pandas DataFrame
- Split text into chunks with configurable strategies
- Preserve source metadata (dataset, question_id, context, etc.)
- Generate unique chunk IDs compatible with Qdrant
- Optional text preprocessing for Russian language support
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =============================================================================
# OPTIONAL: Text Preprocessor (can be disabled by setting ENABLE_PREPROCESSING = False)
# =============================================================================
ENABLE_PREPROCESSING = True  # ← Toggle this to disable preprocessing


def preprocess_text_russian(text: str) -> str:
    """
    Basic preprocessing for Russian text.

    - Normalizes whitespace
    - Removes excessive newlines
    - Optional: integrate PyMorphy3 for lemmatization if needed

    Args:
        text: Raw input text

    Returns:
        Preprocessed text
    """
    if not ENABLE_PREPROCESSING:
        return text

    # Normalize whitespace: replace multiple spaces/tabs/newlines with single space
    text = re.sub(r"[\s\u00a0]+", " ", text)

    # Remove leading/trailing whitespace from lines, then rejoin
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = "\n".join(lines)

    # Optional: PyMorphy3 lemmatization (uncomment if needed)
    # from pymorphy3 import MorphAnalyzer
    # morph = MorphAnalyzer(lang='ru')
    # tokens = text.split()
    # lemmatized = [morph.parse(token)[0].normal_form for token in tokens]
    # text = ' '.join(lemmatized)

    return text.strip()


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Chunk:
    """Represents a single text chunk with metadata."""

    chunk_id: str  # UUID string compatible with Qdrant
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary format for indexing."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
        }

    def to_qdrant_point(self, vector: Optional[list[float]] = None) -> dict[str, Any]:
        """
        Convert chunk to Qdrant PointStruct format.

        Args:
            vector: Optional pre-computed embedding vector.
                   If None, caller should fill it before upsert.

        Returns:
            Dict compatible with qdrant-client PointStruct
        """
        return {
            "id": self.chunk_id,  # UUID string
            "vector": vector if vector is not None else [],  # Empty list or filled vector
            "payload": {
                "text": self.text,
                **{k: v for k, v in self.metadata.items() if k != "chunk_id"},  # Avoid duplication
            },
        }

    def to_batch_payload(self) -> tuple[str, dict[str, Any]]:
        """
        Return (id, payload) tuple for batch Qdrant upsert.
        Vector should be added separately by the indexer.
        """
        return (
            self.chunk_id,
            {
                "text": self.text,
                **{k: v for k, v in self.metadata.items() if k != "chunk_id"},
            },
        )


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""

    chunk_size: int = 512
    chunk_overlap: int = 50
    separator: str = "\n\n"
    add_start_index: bool = False
    strip_whitespace: bool = True

    # Metadata fields to preserve and optionally index in Qdrant
    metadata_fields: list[str] = field(
        default_factory=lambda: ["dataset", "question_id", "context", "correct_answer", "context_used"]
    )

    # Qdrant payload index configuration: field_name -> index_type
    # Supported types: "keyword", "text", "integer", "float", "geo", "bool"
    payload_indexes: dict[str, str] = field(
        default_factory=lambda: {
            "dataset": "keyword",
            "question_id": "keyword",
            "context_used": "text",
        }
    )


# =============================================================================
# Main Chunker Class
# =============================================================================


class DocumentChunker:
    """
    Document chunker with metadata preservation and Qdrant compatibility.

    Supports loading from pandas DataFrame and splitting text
    while preserving source metadata for each chunk.
    """

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        preprocessor: Callable[[str], str] | None = None,
    ):
        """
        Initialize the chunker.

        Args:
            config: Chunking configuration. Uses defaults if not provided.
            preprocessor: Optional text preprocessing function.
                         If None, uses default preprocess_text_russian if ENABLE_PREPROCESSING.
        """
        self.config = config or ChunkingConfig()
        self._preprocessor = preprocessor if preprocessor is not None else preprocess_text_russian

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=[self.config.separator, "\n", ". ", " ", ""],
            keep_separator=True,
            strip_whitespace=self.config.strip_whitespace,
        )

    def _generate_chunk_id(self, text: str, source_id: str, index: int) -> str:
        """
        Generate a unique UUID-based chunk ID compatible with Qdrant.

        Uses uuid.uuid4() for guaranteed uniqueness and Qdrant compatibility.
        For deterministic IDs (reproducibility), use the commented alternative.

        Args:
            text: The chunk text
            source_id: Source document identifier
            index: Chunk index within the document

        Returns:
            UUID string compatible with Qdrant point IDs
        """
        # Option 1: Random UUID (recommended - simple, collision-free)
        return str(uuid.uuid4())

        # Option 2: Deterministic UUID from content hash (uncomment if reproducibility needed)
        # content_hash = hashlib.md5(f"{source_id}:{index}:{text[:100]}".encode()).hexdigest()
        # return str(uuid.UUID(hex=content_hash[:32]))

    def _extract_metadata(self, row: pd.Series) -> dict[str, Any]:
        """
        Extract metadata from DataFrame row.

        Args:
            row: DataFrame row

        Returns:
            Dictionary with preserved metadata fields
        """
        metadata = {}
        for field_name in self.config.metadata_fields:
            if field_name in row.index:
                value = row[field_name]
                # Handle NaN values
                if pd.isna(value):
                    continue
                # Convert numpy types to native Python for JSON/Qdrant compatibility
                if isinstance(value, (pd.Series, pd.DataFrame)):
                    continue
                metadata[field_name] = value.item() if hasattr(value, "item") else value

        # Add source identifier for traceability
        if "question_id" in metadata:
            metadata["source_id"] = f"{metadata.get('dataset', 'unknown')}_{metadata['question_id']}"
        elif "dataset" in metadata:
            metadata["source_id"] = str(metadata["dataset"])

        return metadata

    def chunk_text(self, text: str, source_metadata: dict[str, Any]) -> list[Chunk]:
        """
        Split text into chunks with metadata.

        Args:
            text: Text to split
            source_metadata: Metadata to attach to each chunk

        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []

        # Apply optional preprocessing
        if ENABLE_PREPROCESSING and self._preprocessor:
            text = self._preprocessor(text)

        if not text.strip():
            return []

        # Split text using configured strategy
        texts = self._splitter.split_text(text)

        chunks = []
        source_id = source_metadata.get("source_id", "doc")

        for idx, chunk_text in enumerate(texts):
            if not chunk_text.strip():
                continue

            chunk = Chunk(
                chunk_id=self._generate_chunk_id(chunk_text, source_id, idx),
                text=chunk_text.strip(),
                metadata={
                    **source_metadata,
                    "chunk_index": idx,
                    "total_chunks": len(texts),
                },
            )
            chunks.append(chunk)

        return chunks

    def chunk_dataframe(self, df: pd.DataFrame) -> Iterator[Chunk]:
        """
        Process DataFrame and yield chunks.

        Each row is expected to have:
        - 'context' or 'text': The text to chunk
        - Optional metadata fields (dataset, question_id, etc.)

        Args:
            df: pandas DataFrame with documents

        Yields:
            Chunk objects
        """
        # Determine text column priority
        text_column = "context" if "context" in df.columns else "text"

        if text_column not in df.columns:
            raise ValueError(f"DataFrame must contain either 'context' or 'text' column. Found: {list(df.columns)}")

        for idx, row in df.iterrows():
            text = row.get(text_column, "")
            if not text or not isinstance(text, str):
                continue

            # Extract metadata
            source_metadata = self._extract_metadata(row)
            source_metadata["row_index"] = int(idx)  # Ensure native int type

            # Generate chunks
            chunks = self.chunk_text(text, source_metadata)
            yield from chunks

    def chunk_dataframe_batched(self, df: pd.DataFrame, batch_size: int = 100) -> Iterator[list[Chunk]]:
        """
        Process DataFrame and yield chunks in batches for efficient Qdrant upsert.

        Args:
            df: pandas DataFrame with documents
            batch_size: Number of chunks per batch

        Yields:
            Lists of Chunk objects
        """
        batch = []
        for chunk in self.chunk_dataframe(df):
            batch.append(chunk)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def load_and_chunk(
        self,
        data_path: str | None = None,
        df: pd.DataFrame | None = None,
    ) -> list[Chunk]:
        """
        Load documents and create chunks.

        Args:
            data_path: Path to CSV/Parquet file (optional if df provided)
            df: pandas DataFrame (optional if data_path provided)

        Returns:
            List of all chunks

        Raises:
            ValueError: If neither data_path nor df is provided
            FileNotFoundError: If data_path doesn't exist
        """
        if df is None:
            if data_path is None:
                raise ValueError("Either data_path or df must be provided")

            # Load from file with format detection
            path_lower = data_path.lower()
            if path_lower.endswith(".csv"):
                df = pd.read_csv(data_path)
            elif path_lower.endswith(".parquet"):
                df = pd.read_parquet(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}. Supported: .csv, .parquet")

        return list(self.chunk_dataframe(df))

    def count_chunks_estimate(self, df: pd.DataFrame) -> int:
        """
        Estimate total chunks that would be generated from DataFrame.

        Note: This is a rough estimate based on text length.
        For exact count, iterate through chunk_dataframe().

        Args:
            df: pandas DataFrame

        Returns:
            Estimated number of chunks (upper bound)
        """
        text_column = "context" if "context" in df.columns else "text"
        if text_column not in df.columns:
            return 0

        total = 0
        effective_size = self.config.chunk_size - self.config.chunk_overlap

        for _, row in df.iterrows():
            text = row.get(text_column, "")
            if text and isinstance(text, str):
                # Conservative estimate: assume worst-case splitting
                estimated = max(1, len(str(text)) // max(1, effective_size))
                total += estimated

        return total

    def get_payload_index_config(self) -> dict[str, dict[str, str]]:
        """
        Return Qdrant payload index configuration for setup.

        Returns:
            Dict suitable for qdrant_client.create_payload_index()
        """
        return {field_name: {"type": index_type} for field_name, index_type in self.config.payload_indexes.items()}
