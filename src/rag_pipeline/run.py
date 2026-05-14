"""
RAG Pipeline Runner

Complete pipeline for:
1. Loading and chunking documents from DataFrame
2. Indexing into Qdrant (dense, sparse, hybrid, ColBERT)
3. Retrieval with multiple strategies
4. Reranking
5. Answer generation
6. Evaluation with IR metrics
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from qdrant_client import QdrantClient

from rag_pipeline.retrieval import CodeSearcher

logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from rag_pipeline import (
    AnswerGenerator,
    ColBERTSearcher,
    DenseSearcher,
    HybridSearcher,
    Reranker,
    RetrievalEvaluator,
    SparseSearcher,
)
from rag_pipeline.experiments import ExperimentConfig, ParameterTuner
from rag_pipeline.generation.generator import GenerationConfig
from rag_pipeline.indexing.qdrant_indexer import IndexingConfig, QdrantIndexer
from rag_pipeline.preprocessing.chunker import ChunkingConfig, DocumentChunker
from rag_pipeline.reranking.reranker import RerankerConfig
from src.logger import logger


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    # Data paths
    data_path: str = "src/all_q_True.csv"
    output_dir: str = "results"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    recreate_collections: bool = True

    # Models
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Code search models
    code_text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    code_model: str = "jinaai/jina-embeddings-v2-base-code"

    # Search
    top_k: int = 10
    rerank_top_k: int = 5

    # Evaluation
    k_values: list = None

    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1, 3, 5, 10]


class RAGPipeline:
    """
    Complete RAG pipeline orchestrator.

    Coordinates all components from preprocessing to evaluation.
    """

    def __init__(self, config: PipelineConfig | None = None):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.client = QdrantClient(url=self.config.qdrant_url)

        # Initialize components
        self._chunker = None
        self._indexer = None
        self._searchers = {}
        self._reranker = None
        self._generator = None
        self._evaluator = None

    @property
    def chunker(self) -> DocumentChunker:
        if self._chunker is None:
            self._chunker = DocumentChunker(
                ChunkingConfig(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                )
            )
        return self._chunker

    @property
    def indexer(self) -> QdrantIndexer:
        if self._indexer is None:
            self._indexer = QdrantIndexer(
                IndexingConfig(
                    qdrant_url=self.config.qdrant_url,
                    recreate_collections=self.config.recreate_collections,
                    dense_model=self.config.dense_model,
                )
            )
        return self._indexer

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker(
                RerankerConfig(
                    model_name=self.config.rerank_model,
                    top_k=self.config.rerank_top_k,
                )
            )
        return self._reranker

    @property
    def generator(self) -> AnswerGenerator:
        if self._generator is None:
            self._generator = AnswerGenerator(GenerationConfig())
        return self._generator

    @property
    def evaluator(self) -> RetrievalEvaluator:
        if self._evaluator is None:
            self._evaluator = RetrievalEvaluator(k_values=self.config.k_values)
        return self._evaluator

    def get_searcher(self, strategy: str):
        """Get searcher for specified strategy."""
        if strategy not in self._searchers:
            if strategy == "dense":
                self._searchers[strategy] = DenseSearcher(
                    self.client,
                    collection_name="dense_collection",
                    model_name=self.config.dense_model,
                )
            elif strategy == "sparse":
                self._searchers[strategy] = SparseSearcher(
                    self.client,
                    collection_name="sparse_collection",
                )
            elif strategy == "hybrid_rrf":
                self._searchers[strategy] = HybridSearcher(
                    self.client,
                    collection_name="hybrid_collection",
                    dense_model=self.config.dense_model,
                )
            elif strategy == "hybrid_dbsf":
                self._searchers[strategy] = HybridSearcher(
                    self.client,
                    collection_name="hybrid_collection",
                    dense_model=self.config.dense_model,
                )
            elif strategy == "colbert":
                self._searchers[strategy] = ColBERTSearcher(
                    self.client,
                    collection_name="hybrid_with_colbert",
                    dense_model=self.config.dense_model,
                )
            elif strategy == "code_text":
                self._searchers[strategy] = CodeSearcher(
                    self.client,
                    collection_name="code_collection",
                    text_model=self.config.code_text_model,
                    code_model=self.config.code_model,
                )
            elif strategy == "code_code":
                self._searchers[strategy] = CodeSearcher(
                    self.client,
                    collection_name="code_collection",
                    text_model=self.config.code_text_model,
                    code_model=self.config.code_model,
                )
            elif strategy == "code_hybrid":
                self._searchers[strategy] = CodeSearcher(
                    self.client,
                    collection_name="code_collection",
                    text_model=self.config.code_text_model,
                    code_model=self.config.code_model,
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        return self._searchers[strategy]

    def load_data(self, data_path: str | None = None) -> pd.DataFrame:
        """Load dataset from CSV/Parquet."""
        path = data_path or self.config.data_path
        logger.info(f"Loading data from: {path}")

        if path.endswith(".csv"):
            df = pd.read_csv(path)
        elif path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {path}")

        logger.info(f"Loaded {len(df)} rows")
        return df

    def prepare_chunks(self, df: pd.DataFrame) -> list:
        """Prepare chunks from DataFrame."""
        logger.info("Creating chunks...")
        chunks = self.chunker.load_and_chunk(df=df)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def _prepare_evaluation_queries(self, df: pd.DataFrame, chunks: list) -> list[dict]:
        """
        Создаёт queries с корректными relevant_chunk_ids на основе сгенерированных чанков.

        Args:
            df: Исходный DataFrame с вопросами
            chunks: Список сгенерированных чанков

        Returns:
            Список словарей с вопросами и ground truth chunk IDs
        """
        # Строим маппинг question_id → list[chunk_id]
        question_id_to_chunk_ids = {}
        for chunk in chunks:
            qid = chunk.metadata.get("question_id")
            if qid is not None:
                question_id_to_chunk_ids.setdefault(qid, []).append(chunk.chunk_id)

        logger.info(f"Built mapping for {len(question_id_to_chunk_ids)} questions to chunk IDs")

        # Debug-логирование
        df_question_ids = set(df["question_id"].dropna().unique()) if "question_id" in df.columns else set()
        chunk_question_ids = set(question_id_to_chunk_ids.keys())
        logger.info(f"DataFrame question_ids count: {len(df_question_ids)}")
        logger.info(f"Chunk question_ids count: {len(chunk_question_ids)}")
        logger.info(f"Overlap: {len(df_question_ids & chunk_question_ids)}")

        queries = []
        for _, row in df.iterrows():
            question = row.get("question", "")
            question_id = row.get("question_id")
            relevant_ids = question_id_to_chunk_ids.get(question_id, [])

            if not relevant_ids and question_id is not None:
                logger.warning(f"No chunks found for question_id: {question_id} (type: {type(question_id)})")

            queries.append(
                {
                    "question": question,
                    "relevant_chunk_ids": relevant_ids,
                    "question_id": question_id,
                }
            )

        return queries

    def _validate_queries(self, queries: list[dict]) -> tuple[list[dict], list[str]]:
        """
        Проверяет, что у запросов есть ground truth для оценки.

        Returns:
            Tuple с валидными запросами и списком предупреждений
        """
        valid, warnings = [], []
        for i, q in enumerate(queries):
            if not q.get("relevant_chunk_ids"):
                warnings.append(f"Query {i}: empty relevant_chunk_ids for question_id={q.get('question_id')}")
            else:
                valid.append(q)
        return valid, warnings

    def create_collections(self):
        """Create all Qdrant collections."""
        logger.info("Creating collections...")
        self.indexer.create_all_collections()

    def index_documents(self, chunks: list, resume_from: int = 0) -> int:
        """
        Index chunks into all collections with error handling and resume capability.

        Returns:
            Number of already indexed chunks (for resume), or 0 on success
        """
        logger.info(f"Indexing {len(chunks)} chunks, resume_from={resume_from}")

        try:
            self.indexer.index_all(chunks, resume_from=resume_from)
            logger.info("Indexing completed successfully")
            return 0  # Success, no need to resume
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            logger.info("Checking indexed collections for resume capability...")

            # Get collection info to check progress - БЕЗ РЕКУРСИИ
            try:
                dense_info = self.client.get_collection(self.indexer.config.dense_collection)
                indexed_count = dense_info.points_count
                total_chunks = len(chunks)
                logger.info(f"Already indexed {indexed_count}/{total_chunks} chunks in dense collection")

                if indexed_count > 0:
                    logger.info(f"Can resume indexing from chunk {indexed_count}")
                    return indexed_count
                else:
                    logger.info("No chunks indexed yet, will need to restart")
                    return 0
            except Exception as check_e:
                logger.error(f"Failed to check collection status: {check_e}")

            raise

    def search(
        self,
        query: str,
        strategy: str = "dense",
        top_k: int | None = None,
        use_reranking: bool = False,
    ):
        """
        Perform search with specified strategy.

        Args:
            query: Search query
            strategy: Retrieval strategy (dense, sparse, hybrid_rrf, hybrid_dbsf, colbert, code_text, code_code, code_hybrid)
            top_k: Number of results
            use_reranking: Whether to apply reranking

        Returns:
            Search results
        """
        top_k = top_k or self.config.top_k
        searcher = self.get_searcher(strategy)

        # Handle code search strategies
        if strategy.startswith("code_"):
            if strategy == "code_text":
                response = searcher.search_with_text_encoder(query, top_k=top_k)
            elif strategy == "code_code":
                response = searcher.search_with_code_encoder(query, top_k=top_k)
            elif strategy == "code_hybrid":
                response = searcher.search_hybrid(query, top_k=top_k)
            else:
                raise ValueError(f"Unknown code strategy: {strategy}")
        else:
            # Determine fusion strategy for hybrid
            if strategy == "hybrid_dbsf":
                response = searcher.search(query, top_k=top_k, fusion_strategy="dbsf")
            else:
                response = searcher.search(query, top_k=top_k)

            # Apply reranking if requested
            if use_reranking and strategy != "sparse":
                response = self.reranker.rerank_response(query, response, top_k=self.config.rerank_top_k)

        return response

    def code_search(
        self,
        query: str,
        strategy: str = "hybrid",
        top_k: int = 5,
        group_by: str | None = None,
    ):
        """
        Perform code search with specified strategy.

        Args:
            query: Search query
            strategy: Code search strategy (text, code, hybrid, grouped)
            top_k: Number of results
            group_by: Field to group by (for grouped strategy)

        Returns:
            Code search results
        """
        searcher = CodeSearcher(
            self.client,
            collection_name="code_collection",
            text_model=self.config.code_text_model,
            code_model=self.config.code_model,
        )

        if strategy == "text":
            return searcher.search_with_text_encoder(query, top_k=top_k)
        elif strategy == "code":
            return searcher.search_with_code_encoder(query, top_k=top_k)
        elif strategy == "hybrid":
            return searcher.search_hybrid(query, top_k=top_k)
        elif strategy == "grouped":
            return searcher.search_grouped(query, top_k=top_k, group_by=group_by or "context.module")
        else:
            raise ValueError(f"Unknown code search strategy: {strategy}")

    def generate_answer(self, query: str, contexts: list[dict]) -> dict:
        """Generate answer from retrieved contexts."""
        result = self.generator.generate(query, contexts)
        return result.to_dict()

    def evaluate(
        self,
        queries: list[dict],
        strategies: list[str] | None = None,
    ) -> dict:
        """
        Evaluate retrieval strategies.

        Args:
            queries: List of query dicts with 'question' and 'relevant_chunk_ids'
            strategies: List of strategies to evaluate

        Returns:
            Evaluation results dictionary
        """
        if strategies is None:
            strategies = ["dense", "sparse", "hybrid_rrf"]

        # Валидация запросов
        valid_queries, warnings = self._validate_queries(queries)
        for warning in warnings:
            logger.warning(warning)

        if not valid_queries:
            logger.error("No valid queries with ground truth for evaluation")
            return {}

        results = {}

        for strategy in strategies:
            logger.info(f"Evaluating strategy: {strategy}")
            searcher = self.get_searcher(strategy)

            def search_fn(q):
                response = searcher.search(q, top_k=self.config.top_k)
                return [r.chunk_id for r in response.results], response.latency_ms

            eval_result = self.evaluator.evaluate_strategy(valid_queries, search_fn, strategy)
            results[strategy] = eval_result

        return results

    def run_full_pipeline(
        self,
        data_path: str | None = None,
        strategies: list[str] | None = None,
        sample_queries: int | None = None,
        resume_from: int = 0,
        sample_size: int | None = None,  # Новый параметр вместо хардкода
        run_by_dataset: bool = False,  # Новый параметр для запуска по каждому датасету
    ) -> dict:
        """
        Run complete RAG pipeline.

        Args:
            data_path: Path to dataset
            strategies: Strategies to evaluate
            sample_queries: Number of queries to evaluate (None = all)
            resume_from: Chunk index to resume indexing from
            sample_size: Number of rows to sample from dataset (None = all)
            run_by_dataset: If True, run pipeline for full dataset and each individual dataset

        Returns:
            Complete pipeline results
        """
        # Load data first to check for dataset column
        df = self.load_data(data_path)

        # Check if dataset column exists and run_by_dataset is True
        if run_by_dataset and "dataset" in df.columns:
            return self._run_pipeline_by_dataset(
                df=df,
                strategies=strategies,
                sample_queries=sample_queries,
                resume_from=resume_from,
                sample_size=sample_size,
            )

        # Original single-run logic
        start_time = time.perf_counter()
        logger.info(f"Starting pipeline with data_path: {data_path}")

        # Sample data if requested
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} rows from dataset")

        # Step 2: Create chunks
        chunks = self.prepare_chunks(df)

        # Step 3: Create collections and index
        self.create_collections()

        # Handle resume logic
        if resume_from > 0:
            logger.info(f"Resuming pipeline from chunk {resume_from}")

        try:
            self.index_documents(chunks, resume_from=resume_from)
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            # Get resume point without recursion
            try:
                dense_info = self.client.get_collection(self.indexer.config.dense_collection)
                resume_point = dense_info.points_count
            except:
                resume_point = 0
            return {
                "error": "Indexing failed",
                "resume_from": resume_point,
                "total_chunks": len(chunks),
                "message": f"Restart with --resume-from {resume_point}",
            }

        # Step 4: Prepare evaluation queries with CORRECT ground truth
        queries = self._prepare_evaluation_queries(df, chunks)

        if sample_queries:
            queries = queries[:sample_queries]

        # Step 5: Evaluate strategies
        if strategies is None:
            strategies = ["dense", "sparse", "hybrid_rrf"]

        eval_results = self.evaluate(queries, strategies)

        # Step 6: Demo query
        demo_query = "Что может вызвать цунами?"
        logger.info(f"Running demo query: {demo_query}")

        demo_results = {}
        for strategy in strategies:
            response = self.search(demo_query, strategy=strategy, top_k=3)
            demo_results[strategy] = {
                "results": [r.to_dict() for r in response.results],
                "latency_ms": response.latency_ms,
            }

        total_time = time.perf_counter() - start_time

        return {
            "evaluation": eval_results,
            "demo_results": demo_results,
            "total_time_seconds": total_time,
            "num_chunks": len(chunks),
            "num_queries_evaluated": len(queries),
        }

    def _run_pipeline_by_dataset(
        self,
        df: pd.DataFrame,
        strategies: list[str] | None = None,
        sample_queries: int | None = None,
        resume_from: int = 0,
        sample_size: int | None = None,
    ) -> dict:
        """
        Run pipeline for full dataset and each individual dataset.

        Args:
            df: Loaded DataFrame with dataset column
            strategies: Strategies to evaluate
            sample_queries: Number of queries to evaluate (None = all)
            resume_from: Chunk index to resume indexing from
            sample_size: Number of rows to sample from dataset (None = all)

        Returns:
            Dictionary with results for full dataset and each individual dataset
        """
        # Get unique datasets
        unique_datasets = df["dataset"].unique()
        logger.info(f"Found {len(unique_datasets)} unique datasets: {list(unique_datasets)}")

        # Sample data if requested (apply to full dataset first)
        if sample_size and sample_size < len(df):
            df_sampled = df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} rows from full dataset")
        else:
            df_sampled = df

        all_results = {}

        # Run for full dataset
        logger.info("=" * 60)
        logger.info("RUNNING PIPELINE FOR FULL DATASET")
        logger.info("=" * 60)
        full_result = self._run_single_pipeline(
            df=df_sampled,
            dataset_name="full",
            strategies=strategies,
            sample_queries=sample_queries,
            resume_from=resume_from,
        )
        all_results["full"] = full_result

        # Run for each individual dataset
        for dataset_name in unique_datasets:
            logger.info("=" * 60)
            logger.info(f"RUNNING PIPELINE FOR DATASET: {dataset_name}")
            logger.info("=" * 60)

            # Filter by dataset
            df_dataset = df_sampled[df_sampled["dataset"] == dataset_name].copy()
            logger.info(f"Filtered to {len(df_dataset)} rows for dataset {dataset_name}")

            if len(df_dataset) == 0:
                logger.warning(f"No data found for dataset {dataset_name}, skipping")
                all_results[dataset_name] = {"error": "No data found"}
                continue

            # Run pipeline for this dataset
            dataset_result = self._run_single_pipeline(
                df=df_dataset,
                dataset_name=dataset_name,
                strategies=strategies,
                sample_queries=sample_queries,
                resume_from=resume_from,
            )
            all_results[dataset_name] = dataset_result

        return all_results

    def _run_single_pipeline(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        strategies: list[str] | None = None,
        sample_queries: int | None = None,
        resume_from: int = 0,
    ) -> dict:
        """
        Run pipeline for a single dataset.

        Args:
            df: DataFrame for this dataset
            dataset_name: Name of the dataset
            strategies: Strategies to evaluate
            sample_queries: Number of queries to evaluate (None = all)
            resume_from: Chunk index to resume indexing from

        Returns:
            Pipeline results for this dataset
        """
        start_time = time.perf_counter()
        logger.info(f"Starting pipeline for dataset: {dataset_name} ({len(df)} rows)")

        # Step 1: Create chunks
        chunks = self.prepare_chunks(df)

        # Step 2: Create collections and index (recreate for each dataset)
        self.config.recreate_collections = True
        self.create_collections()

        # Handle resume logic
        if resume_from > 0:
            logger.info(f"Resuming pipeline from chunk {resume_from}")

        try:
            self.index_documents(chunks, resume_from=resume_from)
        except Exception as e:
            logger.error(f"Indexing failed for dataset {dataset_name}: {e}")
            # Get resume point without recursion
            try:
                dense_info = self.client.get_collection(self.indexer.config.dense_collection)
                resume_point = dense_info.points_count
            except:
                resume_point = 0
            return {
                "error": "Indexing failed",
                "resume_from": resume_point,
                "total_chunks": len(chunks),
                "message": f"Restart with --resume-from {resume_point}",
            }

        # Step 3: Prepare evaluation queries with CORRECT ground truth
        queries = self._prepare_evaluation_queries(df, chunks)

        if sample_queries:
            queries = queries[:sample_queries]

        # Step 4: Evaluate strategies
        if strategies is None:
            strategies = ["dense", "sparse", "hybrid_rrf"]

        eval_results = self.evaluate(queries, strategies)

        # Step 5: Demo query
        demo_query = "Что может вызвать цунами?"
        logger.info(f"Running demo query: {demo_query}")

        demo_results = {}
        for strategy in strategies:
            response = self.search(demo_query, strategy=strategy, top_k=3)
            demo_results[strategy] = {
                "results": [r.to_dict() for r in response.results],
                "latency_ms": response.latency_ms,
            }

        total_time = time.perf_counter() - start_time

        return {
            "evaluation": eval_results,
            "demo_results": demo_results,
            "total_time_seconds": total_time,
            "num_chunks": len(chunks),
            "num_queries_evaluated": len(queries),
        }

    def run_experiments(
        self,
        experiment_type: str = "chunk_size",
        data_path: str | None = None,
        queries: list[dict] | None = None,
        max_failures: int = 5,
        sample_size: int | None = 10,  # Параметр для сэмплирования данных
    ) -> dict:
        """
        Run experimental parameter tuning with error isolation.

        Args:
            experiment_type: Type of experiment (chunk_size, model, strategy, top_k, rrf_k)
            data_path: Path to dataset for indexing
            queries: Evaluation queries (if None, will use dataset queries)
            max_failures: Maximum consecutive failures before stopping
            sample_size: Number of rows to sample for experiments

        Returns:
            Experiment results summary
        """
        logger.info(f"Starting experiments: {experiment_type}")

        # Initialize parameter tuner
        tuner = ParameterTuner(
            output_dir="experiments/results",
            results_file=f"{experiment_type}_results.csv",
        )

        # Load data and prepare queries if not provided
        if queries is None and data_path:
            df = self.load_data(data_path)

            # Sample for experiments
            if sample_size and sample_size < len(df):
                df = df.sample(n=sample_size, random_state=42)
                logger.info(f"Sampled {sample_size} rows for experiments")

            # Generate chunks for ground truth mapping
            chunks = self.prepare_chunks(df)

            # Use unified method for query preparation
            queries = self._prepare_evaluation_queries(df, chunks)

            # Validate queries
            valid_queries, warnings = self._validate_queries(queries)
            for warning in warnings:
                logger.warning(warning)
            queries = valid_queries

        if not queries:
            logger.error("No valid queries available for experiments")
            return {"error": "No valid queries"}

        # Define experiment function with error isolation
        def experiment_fn(config: ExperimentConfig, queries: list[dict]) -> dict[str, Any]:
            """Run single experiment with given config and queries."""
            try:
                # Create temporary pipeline with experiment config
                temp_config = PipelineConfig(
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    dense_model=config.dense_model,
                    top_k=config.top_k,
                    k_values=config.k_values,
                    qdrant_url=self.config.qdrant_url,
                    recreate_collections=False,  # Don't recreate for experiments
                )

                # Get searcher for strategy
                searcher = self.get_searcher(config.strategy)

                # Evaluate
                evaluator = RetrievalEvaluator(k_values=config.k_values)

                def search_fn(q):
                    response = searcher.search(q, top_k=config.top_k)
                    return [r.chunk_id for r in response.results], response.latency_ms

                eval_result = evaluator.evaluate_strategy(queries, search_fn, config.experiment_id)

                return {
                    "metrics": eval_result["metrics"],
                    "total_queries": eval_result["metrics"].get("total_queries", len(queries)),
                }

            except Exception as e:
                logger.error(f"Experiment failed: {e}")
                raise

        # Generate experiment configs based on type
        base_config = ExperimentConfig(
            experiment_id="base",
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            dense_model=self.config.dense_model,
            strategy="dense",
            top_k=self.config.top_k,
            k_values=self.config.k_values,
        )

        if experiment_type == "chunk_size":
            logger.warning(
                "chunk_size experiments require re-indexing. Results may not reflect actual chunk size changes."
            )
            configs = tuner.generate_chunk_size_grid(
                chunk_sizes=[256, 512, 768, 1024],
                base_config=base_config,
            )
        elif experiment_type == "model":
            configs = tuner.generate_model_grid(
                models=[
                    ("sentence-transformers/all-MiniLM-L6-v2", False, False, False),
                    ("BAAI/bge-small-en-v1.5", True, False, False),
                    ("BAAI/bge-m3", False, True, False),
                ],
                base_config=base_config,
            )
        elif experiment_type == "strategy":
            configs = tuner.generate_strategy_grid(
                strategies=["dense", "sparse", "hybrid_rrf", "hybrid_dbsf", "colbert"],
                base_config=base_config,
            )
        elif experiment_type == "top_k":
            configs = tuner.generate_top_k_grid(
                top_k_values=[5, 10, 20, 50],
                base_config=base_config,
            )
        elif experiment_type == "rrf_k":
            configs = tuner.generate_rrf_k_grid(
                rrf_k_values=[30, 60, 100],
                base_config=base_config,
            )
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")

        # Run grid search with error isolation
        results = tuner.run_grid_search(
            configs=configs,
            experiment_fn=experiment_fn,
            queries=queries,
            max_failures=max_failures,
        )

        # Get best configuration
        best_config = tuner.get_best_config(metric=experiment_type, maximize=True)

        summary = {
            "experiment_type": experiment_type,
            "total_experiments": len(configs),
            "successful": sum(r.success for r in results),
            "failed": sum(not r.success for r in results),
            "best_config": best_config.to_dict() if best_config else None,
            "results_file": str(tuner.results_path),
        }

        logger.info(f"Experiments completed: {summary}")
        return summary


def main(
    data_path=None,
    output_dir=None,
    chunk_size=None,
    top_k=None,
    sample_queries=None,
    strategies=None,
    no_recreate=False,
    resume_from=None,
    experiment_type=None,
    sample_size=None,
    run_by_dataset=False,
):
    """
    Запуск полного пайплайна.

    Может принимать параметры напрямую (при вызове из Python-кода) или парсить
    аргументы командной строки, если они не переданы.
    """
    # Определяем, были ли переданы какие-либо аргументы
    explicit_args = any(
        [
            data_path is not None,
            output_dir is not None,
            chunk_size is not None,
            top_k is not None,
            sample_queries is not None,
            strategies is not None,
            resume_from is not None,
            experiment_type is not None,
            sample_size is not None,
            run_by_dataset is not None,
        ]
    )

    if explicit_args:
        # Используем переданные значения, подставляя стандартные defaults вместо None
        if data_path is None:
            data_path = (Path(__file__).parents[2] / "datasets" / "all_q_True.csv").as_posix()
        if output_dir is None:
            output_dir = "results"
        if chunk_size is None:
            chunk_size = 512
        if top_k is None:
            top_k = 10
        if strategies is None:
            strategies = ["dense", "sparse", "hybrid_rrf"]
        if resume_from is None:
            resume_from = 0
        if sample_size is None:
            sample_size = 1000
    else:
        # Стандартное поведение: парсим командную строку
        parser = argparse.ArgumentParser(description="RAG Pipeline Runner")
        parser.add_argument(
            "--data-path", type=str, default=(Path(__file__).parents[2] / "datasets" / "all_q_True.csv").as_posix()
        )
        parser.add_argument("--output-dir", type=str, default="results")
        parser.add_argument("--chunk-size", type=int, default=512)
        parser.add_argument("--top-k", type=int, default=10)
        parser.add_argument("--sample-queries", type=int, default=None)
        parser.add_argument("--sample-size", type=int, default=10, help="Number of rows to sample from dataset")
        parser.add_argument("--strategies", type=str, nargs="+", default=["dense", "sparse", "hybrid_rrf"])
        parser.add_argument("--no-recreate", action="store_true", help="Don't recreate collections")
        parser.add_argument("--resume-from", type=int, default=0, help="Resume indexing from chunk number")
        parser.add_argument(
            "--experiment-type",
            type=str,
            choices=["chunk_size", "model", "strategy", "top_k", "rrf_k", "all"],
            default=None,
            help="Run parameter tuning experiments",
        )
        parser.add_argument(
            "--run-by-dataset", action="store_true", help="Run pipeline for full dataset and each individual dataset"
        )
        args = parser.parse_args()

        data_path = args.data_path
        output_dir = args.output_dir
        chunk_size = args.chunk_size
        top_k = args.top_k
        sample_queries = args.sample_queries
        sample_size = args.sample_size
        strategies = args.strategies
        no_recreate = args.no_recreate
        resume_from = args.resume_from
        experiment_type = args.experiment_type
        run_by_dataset = args.run_by_dataset

    # Создаём конфигурацию пайплайна
    config = PipelineConfig(
        data_path=data_path,
        output_dir=output_dir,
        chunk_size=chunk_size,
        top_k=top_k,
        recreate_collections=not no_recreate,
    )

    # Запускаем пайплайн
    pipeline = RAGPipeline(config)

    # If experiment_type is provided, run experiments instead of full pipeline
    if experiment_type:
        if experiment_type == "all":
            # Запускаем все типы экспериментов последовательно
            for exp_type in ["chunk_size", "model", "strategy", "top_k", "rrf_k"]:
                logger.info(f"Running experiments: {exp_type}")
                try:
                    experiment_results = pipeline.run_experiments(
                        experiment_type=exp_type,
                        data_path=data_path,
                        sample_size=sample_size,
                        max_failures=5,
                    )
                    logger.info(f"Experiments [{exp_type}] completed successfully")
                    logger.info(f"Results saved to: {experiment_results.get('results_file', 'N/A')}")
                except Exception as e:
                    logger.error(f"Experiments [{exp_type}] failed: {e}")
                    logger.exception(e)
        else:
            # Запускаем один тип эксперимента
            logger.info(f"Running experiments: {experiment_type}")
            try:
                experiment_results = pipeline.run_experiments(
                    experiment_type=experiment_type,
                    data_path=data_path,
                    sample_size=sample_size,
                    max_failures=5,
                )
                logger.info("Experiments completed successfully")
                logger.info(f"Results saved to: {experiment_results['results_file']}")
            except Exception as e:
                logger.error(f"Experiments failed: {e}")
                logger.exception(e)

    # Otherwise run full pipeline
    results = pipeline.run_full_pipeline(
        data_path=data_path,
        strategies=strategies,
        sample_queries=sample_queries,
        resume_from=resume_from,
        sample_size=sample_size,
        run_by_dataset=run_by_dataset,
    )

    # Сохраняем результаты
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / "pipeline_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Pipeline completed in {results['total_time_seconds']:.2f} seconds")

    # Вывод сводки
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for strategy, eval_data in results["evaluation"].items():
        metrics = eval_data["metrics"]
        print(f"\n{strategy.upper()}:")
        print(f"  Hit@1: {metrics['hit_at_k'].get(1, 0):.4f}")
        print(f"  Hit@5: {metrics['hit_at_k'].get(5, 0):.4f}")
        print(f"  MRR@10: {metrics['mrr_at_k'].get(10, 0):.4f}")
        print(f"  NDCG@10: {metrics['ndcg_at_k'].get(10, 0):.4f}")
        print(f"  Mean Latency: {metrics['mean_latency_ms']:.2f}ms")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Пример вызова: python rag_pipeline.py --experiment-type strategy --sample-size 20
    main()
