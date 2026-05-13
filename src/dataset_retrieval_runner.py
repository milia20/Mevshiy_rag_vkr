"""
Dataset Runner with Qdrant Retrieval

This module extends the dataset runner to:
1. Load questions from datasets
2. Search for relevant chunks in Qdrant using configurable strategies
3. Save retrieved chunks and context separately
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient
from tqdm import tqdm

from src.logger import logger


@dataclass
class RetrievalConfig:
    """Configuration for retrieval runner."""

    # Data paths
    project_root: str = None
    output_dir: str = "test_datasets"

    # Qdrant connection
    qdrant_url: str = "http://localhost:6333"

    # Search parameters
    strategy: str = "dense"
    top_k: int = 10

    # Models
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    code_text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    code_model: str = "jinaai/jina-embeddings-v2-base-code"

    # Output
    save_chunks: bool = True
    save_context: bool = True
    encoding: str = "mbcs"

    def __post_init__(self):
        if self.project_root is None:
            self.project_root = str(Path(__file__).parent.parent)


class DatasetRetrievalRunner:
    """
    Runner for dataset retrieval with configurable search strategies.
    """

    AVAILABLE_STRATEGIES = [
        "dense",
        "sparse",
        "hybrid_rrf",
        "hybrid_dbsf",
        "colbert",
        "code_text",
        "code_code",
        "code_hybrid",
    ]

    def __init__(self, config: RetrievalConfig):
        """
        Initialize the retrieval runner.

        Args:
            config: Retrieval configuration
        """
        self.config = config
        self.client = QdrantClient(url=config.qdrant_url)
        self.project_root = Path(config.project_root)
        self.output_dir = self.project_root / config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize searcher based on strategy
        self._searcher = None

    def get_collection_name(self, strategy: str) -> str:
        """Get collection name for strategy."""
        collection_map = {
            "dense": "dense_collection",
            "sparse": "sparse_collection",
            "hybrid_rrf": "hybrid_collection",
            "hybrid_dbsf": "hybrid_collection",
            "colbert": "hybrid_with_colbert",
            "code_text": "code_collection",
            "code_code": "code_collection",
            "code_hybrid": "code_collection",
        }
        return collection_map.get(strategy, "dense_collection")

    def search_dense(self, query: str, top_k: int) -> list[dict]:
        """Dense search using sentence transformers."""
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.config.dense_model)
        query_embedding = model.encode(query, convert_to_numpy=True)

        collection_name = self.get_collection_name("dense")
        hits = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            with_payload=True,
        )

        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                {
                    "chunk_id": str(hit.id),
                    "score": float(hit.score),
                    "text": payload.get("text", ""),
                    "metadata": payload,
                }
            )
        return results

    def search_sparse(self, query: str, top_k: int) -> list[dict]:
        """Sparse search using BM25."""
        from qdrant_client.models import Document

        collection_name = self.get_collection_name("sparse")
        hits = self.client.query_points(
            collection_name=collection_name,
            query=Document(text=query, model="Qdrant/bm25"),
            using="bm25",
            limit=top_k,
            with_payload=True,
        )

        results = []
        for hit in hits.points:
            payload = hit.payload or {}
            results.append(
                {
                    "chunk_id": str(hit.id),
                    "score": float(hit.score),
                    "text": payload.get("text", ""),
                    "metadata": payload,
                }
            )
        return results

    def search_hybrid_rrf(self, query: str, top_k: int) -> list[dict]:
        """Hybrid search with RRF fusion."""
        from qdrant_client.models import Document, Prefetch, RrfQuery, Rrf

        collection_name = self.get_collection_name("hybrid_rrf")
        hits = self.client.query_points(
            collection_name=collection_name,
            query=RrfQuery(rrf=Rrf(k=60)),
            prefetch=[
                Prefetch(
                    query=Document(text=query, model=self.config.dense_model),
                    using="dense",
                    limit=top_k * 2,
                ),
                Prefetch(
                    query=Document(text=query, model="Qdrant/bm25"),
                    using="bm25",
                    limit=top_k * 2,
                ),
            ],
            limit=top_k,
            with_payload=True,
        )

        results = []
        for hit in hits.points:
            payload = hit.payload or {}
            results.append(
                {
                    "chunk_id": str(hit.id),
                    "score": float(hit.score),
                    "text": payload.get("text", ""),
                    "metadata": payload,
                }
            )
        return results

    def search_hybrid_dbsf(self, query: str, top_k: int) -> list[dict]:
        """Hybrid search with DBSF fusion."""
        from qdrant_client.models import Document, Fusion, FusionQuery, Prefetch

        collection_name = self.get_collection_name("hybrid_dbsf")
        hits = self.client.query_points(
            collection_name=collection_name,
            query=FusionQuery(fusion=Fusion.DBSF),
            prefetch=[
                Prefetch(
                    query=Document(text=query, model=self.config.dense_model),
                    using="dense",
                    limit=top_k * 2,
                ),
                Prefetch(
                    query=Document(text=query, model="Qdrant/bm25"),
                    using="bm25",
                    limit=top_k * 2,
                ),
            ],
            limit=top_k,
            with_payload=True,
        )

        results = []
        for hit in hits.points:
            payload = hit.payload or {}
            results.append(
                {
                    "chunk_id": str(hit.id),
                    "score": float(hit.score),
                    "text": payload.get("text", ""),
                    "metadata": payload,
                }
            )
        return results

    def search_colbert(self, query: str, top_k: int) -> list[dict]:
        """ColBERT search with late interaction."""
        from qdrant_client.models import Document, Prefetch

        collection_name = self.get_collection_name("colbert")
        hits = self.client.query_points(
            collection_name=collection_name,
            query=Document(text=query, model="colbert-ir/colbert-v2.0"),
            using="late_interaction",
            prefetch=[
                Prefetch(
                    query=Document(text=query, model=self.config.dense_model),
                    using="dense",
                    limit=top_k * 2,
                ),
                Prefetch(
                    query=Document(text=query, model="Qdrant/bm25"),
                    using="bm25",
                    limit=top_k * 2,
                ),
            ],
            limit=top_k,
            with_payload=True,
        )

        results = []
        for hit in hits.points:
            payload = hit.payload or {}
            results.append(
                {
                    "chunk_id": str(hit.id),
                    "score": float(hit.score),
                    "text": payload.get("text", ""),
                    "metadata": payload,
                }
            )
        return results

    def search_code_text(self, query: str, top_k: int) -> list[dict]:
        """Code search using text encoder."""
        from rag_pipeline.retrieval import CodeSearcher

        searcher = CodeSearcher(
            self.client,
            collection_name="code_collection",
            text_model=self.config.code_text_model,
            code_model=self.config.code_model,
        )
        response = searcher.search_with_text_encoder(query, top_k=top_k)

        results = []
        for r in response.results:
            results.append(
                {
                    "chunk_id": r.chunk_id,
                    "score": r.score,
                    "text": r.text,
                    "metadata": r.metadata,
                }
            )
        return results

    def search_code_code(self, query: str, top_k: int) -> list[dict]:
        """Code search using code encoder."""
        from rag_pipeline.retrieval import CodeSearcher

        searcher = CodeSearcher(
            self.client,
            collection_name="code_collection",
            text_model=self.config.code_text_model,
            code_model=self.config.code_model,
        )
        response = searcher.search_with_code_encoder(query, top_k=top_k)

        results = []
        for r in response.results:
            results.append(
                {
                    "chunk_id": r.chunk_id,
                    "score": r.score,
                    "text": r.text,
                    "metadata": r.metadata,
                }
            )
        return results

    def search_code_hybrid(self, query: str, top_k: int) -> list[dict]:
        """Code search using hybrid approach."""
        from rag_pipeline.retrieval import CodeSearcher

        searcher = CodeSearcher(
            self.client,
            collection_name="code_collection",
            text_model=self.config.code_text_model,
            code_model=self.config.code_model,
        )
        response = searcher.search_hybrid(query, top_k=top_k)

        results = []
        for r in response.results:
            results.append(
                {
                    "chunk_id": r.chunk_id,
                    "score": r.score,
                    "text": r.text,
                    "metadata": r.metadata,
                }
            )
        return results

    def search(self, query: str, strategy: str = None, top_k: int = None) -> list[dict]:
        """
        Perform search with specified strategy.

        Args:
            query: Search query
            strategy: Search strategy
            top_k: Number of results to return

        Returns:
            List of search results
        """
        strategy = strategy or self.config.strategy
        top_k = top_k or self.config.top_k

        if strategy not in self.AVAILABLE_STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. " f"Available: {', '.join(self.AVAILABLE_STRATEGIES)}")

        search_method = getattr(self, f"search_{strategy}")
        return search_method(query, top_k)

    def load_questions(self) -> pd.DataFrame:
        """Load all questions from datasets."""
        from load_dataets import load_all_questions

        questions = load_all_questions(self.project_root)
        logger.info(f"Loaded {len(questions)} questions")
        return questions

    def build_context(self, chunks: list[dict]) -> str:
        """
        Build context string from retrieved chunks.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Combined context string
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            context_parts.append(f"[Chunk {i}] {text}")

        return "\n\n".join(context_parts)

    def save_chunks(self, question_data: dict, chunks: list[dict], output_file: Path):
        """
        Save retrieved chunks to file.

        Args:
            question_data: Question metadata
            chunks: Retrieved chunks
            output_file: Output file path
        """
        output_data = {
            "question_id": question_data.get("question_id"),
            "dataset": question_data.get("dataset"),
            "question": question_data.get("question"),
            "strategy": self.config.strategy,
            "top_k": self.config.top_k,
            "num_chunks": len(chunks),
            "chunks": chunks,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    def save_context(self, question_data: dict, context: str, output_file: Path):
        """
        Save context string to file.

        Args:
            question_data: Question metadata
            context: Combined context string
            output_file: Output file path
        """
        output_data = {
            "question_id": question_data.get("question_id"),
            "dataset": question_data.get("dataset"),
            "question": question_data.get("question"),
            "strategy": self.config.strategy,
            "top_k": self.config.top_k,
            "context": context,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    def run(self, questions: pd.DataFrame = None, test_mode: bool = False):
        """
        Run retrieval for all questions.

        Args:
            questions: Questions DataFrame (if None, will load from datasets)
            test_mode: If True, only process first 2 questions
        """
        if questions is None:
            questions = self.load_questions()

        if test_mode:
            questions = questions.head(2)
            logger.info(f"Test mode: processing {len(questions)} questions")

        if questions.empty:
            logger.warning("No questions found")
            return

        # Create output directories
        chunks_dir = self.output_dir / "retrieved_chunks" / self.config.strategy
        context_dir = self.output_dir / "retrieved_contexts" / self.config.strategy
        chunks_dir.mkdir(parents=True, exist_ok=True)
        context_dir.mkdir(parents=True, exist_ok=True)

        # Process each question
        results = []
        for _, row in tqdm(questions.iterrows(), desc="Retrieving chunks", file=sys.stdout):
            question = row.get("question", "")
            question_id = row.get("question_id")
            dataset = row.get("dataset")

            question_data = {
                "question_id": question_id,
                "dataset": dataset,
                "question": question,
            }

            try:
                # Search for chunks
                chunks = self.search(question)

                # Build context
                context = self.build_context(chunks)

                # Save chunks
                if self.config.save_chunks:
                    chunks_file = chunks_dir / f"{dataset}_{question_id}_chunks.json"
                    self.save_chunks(question_data, chunks, chunks_file)

                # Save context
                if self.config.save_context:
                    context_file = context_dir / f"{dataset}_{question_id}_context.json"
                    self.save_context(question_data, context, context_file)

                results.append(
                    {
                        "question_id": question_id,
                        "dataset": dataset,
                        "num_chunks": len(chunks),
                        "success": True,
                    }
                )

            except Exception as e:
                logger.error(f"Failed to retrieve for question {question_id}: {e}")
                results.append(
                    {
                        "question_id": question_id,
                        "dataset": dataset,
                        "num_chunks": 0,
                        "success": False,
                        "error": str(e),
                    }
                )

        # Save summary
        summary_file = self.output_dir / f"retrieval_summary_{self.config.strategy}.csv"
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(summary_file, index=False, encoding=self.config.encoding)

        logger.info(f"Retrieval completed. Summary saved to: {summary_file}")
        logger.info(f"Successful: {sum(r['success'] for r in results)}/{len(results)}")


def main_run(
    strategy=None,
    top_k=None,
    output_dir=None,
    qdrant_url=None,
    test_mode=False,
    save_chunks=True,
    save_context=True,
):
    """
    Main entry point for dataset retrieval runner.

    Can accept parameters directly or parse command line arguments.
    """
    # Check if explicit arguments were provided
    explicit_args = any(
        [
            strategy is not None,
            top_k is not None,
            output_dir is not None,
            qdrant_url is not None,
        ]
    )

    if explicit_args:
        # Use provided arguments with defaults
        if strategy is None:
            strategy = "dense"
        if top_k is None:
            top_k = 10
        if output_dir is None:
            output_dir = "test_datasets"
        if qdrant_url is None:
            qdrant_url = "http://localhost:6333"
    else:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Dataset Retrieval Runner with Qdrant")
        parser.add_argument(
            "--strategy",
            type=str,
            choices=DatasetRetrievalRunner.AVAILABLE_STRATEGIES,
            default="dense",
            help="Search strategy to use",
        )
        parser.add_argument("--top-k", type=int, default=10, help="Number of chunks to retrieve")
        parser.add_argument("--output-dir", type=str, default="test_datasets", help="Output directory")
        parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333", help="Qdrant server URL")
        parser.add_argument("--test-mode", action="store_true", help="Run in test mode (2 questions only)")
        parser.add_argument("--no-chunks", action="store_true", help="Don't save individual chunks")
        parser.add_argument("--no-context", action="store_true", help="Don't save context files")

        args = parser.parse_args()

        strategy = args.strategy
        top_k = args.top_k
        output_dir = args.output_dir
        qdrant_url = args.qdrant_url
        test_mode = args.test_mode
        save_chunks = not args.no_chunks
        save_context = not args.no_context

    # Create configuration
    config = RetrievalConfig(
        output_dir=output_dir,
        qdrant_url=qdrant_url,
        strategy=strategy,
        top_k=top_k,
        save_chunks=save_chunks,
        save_context=save_context,
    )

    # Create runner and execute
    runner = DatasetRetrievalRunner(config)
    runner.run(test_mode=test_mode)


if __name__ == "__main__":
    main_run()
from src.dataset_retrieval_runner import main_run

main_run(strategy="hybrid_rrf", top_k=10, test_mode=True)
