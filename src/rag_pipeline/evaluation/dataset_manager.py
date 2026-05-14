"""
Dataset manager for multi-dataset evaluation.

Supports evaluation across multiple datasets:
- CoSQA
- RuBQ
- natural_questions
- ru_rag_test_dataset
- Combined (all datasets)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""

    # Dataset paths
    base_path: str = "datasets"
    cosqa_path: str = "datasets/CoSQA"
    rubq_path: str = "datasets/RuBQ"
    natural_questions_path: str = "datasets/natural_questions"
    ru_rag_path: str = "datasets/ru_rag_test_dataset"

    # Combined dataset paths
    all_q_true: str = "datasets/all_q_True.csv"
    all_q_false: str = "datasets/all_q_False.csv"
    nq_true: str = "datasets/nq_True.csv"
    nq_false: str = "datasets/nq_False.csv"


class DatasetManager:
    """
    Manager for loading and splitting datasets for evaluation.

    Supports individual dataset evaluation and combined evaluation.
    """

    def __init__(self, config: DatasetConfig | None = None):
        """
        Initialize dataset manager.

        Args:
            config: Dataset configuration
        """
        self.config = config or DatasetConfig()
        self._datasets = {}

    def load_cosqa(self) -> list[dict[str, Any]]:
        """Load CoSQA dataset."""
        logger.info("Loading CoSQA dataset")
        path = Path(self.config.cosqa_path)

        # Try to load parquet files
        train_path = path / "json-train.parquet"
        val_path = path / "json-validation.parquet"

        queries = []
        if train_path.exists():
            df = pd.read_parquet(train_path)
            queries.extend(self._parse_dataframe(df, "cosqa"))
            logger.info(f"Loaded {len(df)} samples from CoSQA train")

        if val_path.exists():
            df = pd.read_parquet(val_path)
            queries.extend(self._parse_dataframe(df, "cosqa"))
            logger.info(f"Loaded {len(df)} samples from CoSQA validation")

        self._datasets["cosqa"] = queries
        return queries

    def load_rubq(self) -> list[dict[str, Any]]:
        """Load RuBQ dataset."""
        logger.info("Loading RuBQ dataset")
        path = Path(self.config.rubq_path)

        queries = []
        test_path = path / "RuBQ_2.0_test.json"
        paragraphs_path = path / "RuBQ_2.0_paragraphs.json"

        if test_path.exists():
            import json

            with open(test_path, encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    queries.append(
                        {
                            "question": item.get("question", ""),
                            "relevant_chunk_ids": item.get("relevant_chunk_ids", []),
                            "dataset": "rubq",
                            "metadata": item,
                        }
                    )
            logger.info(f"Loaded {len(queries)} samples from RuBQ test")

        self._datasets["rubq"] = queries
        return queries

    def load_natural_questions(self) -> list[dict[str, Any]]:
        """Load Natural Questions dataset."""
        logger.info("Loading Natural Questions dataset")
        path = Path(self.config.natural_questions_path)

        queries = []
        train_path = path / "simplified-nq-train.jsonl.gz"
        dev_path = path / "v1.0-simplified_nq-dev-all.json"

        if dev_path.exists():
            import json

            with open(dev_path, encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    queries.append(
                        {
                            "question": item.get("question_text", ""),
                            "relevant_chunk_ids": item.get("relevant_chunk_ids", []),
                            "dataset": "natural_questions",
                            "metadata": item,
                        }
                    )
            logger.info(f"Loaded {len(queries)} samples from Natural Questions dev")

        self._datasets["natural_questions"] = queries
        return queries

    def load_ru_rag(self) -> list[dict[str, Any]]:
        """Load RuRAG test dataset."""
        logger.info("Loading RuRAG test dataset")
        path = Path(self.config.ru_rag_path)

        queries = []
        # Load all .txt files as chunks
        if path.exists():
            for txt_file in path.glob("*.txt"):
                with open(txt_file, encoding="utf-8") as f:
                    content = f.read()
                    # Assume each file is a chunk with its ID
                    chunk_id = txt_file.stem
                    queries.append(
                        {
                            "question": "",  # Will need separate questions file
                            "relevant_chunk_ids": [chunk_id],
                            "dataset": "ru_rag",
                            "metadata": {"file": str(txt_file)},
                        }
                    )
            logger.info(f"Loaded {len(queries)} chunks from RuRAG dataset")

        self._datasets["ru_rag"] = queries
        return queries

    def load_combined(self) -> list[dict[str, Any]]:
        """Load combined dataset from all_q_True.csv."""
        logger.info("Loading combined dataset from all_q_True.csv")
        path = Path(self.config.all_q_true)

        if not path.exists():
            logger.warning(f"Combined dataset file not found: {path}")
            return []

        df = pd.read_csv(path)
        queries = self._parse_dataframe(df, "combined")
        logger.info(f"Loaded {len(queries)} samples from combined dataset")

        self._datasets["combined"] = queries
        return queries

    def _parse_dataframe(self, df: pd.DataFrame, dataset_name: str) -> list[dict[str, Any]]:
        """
        Parse DataFrame into query format.

        Args:
            df: Input DataFrame
            dataset_name: Name of the dataset

        Returns:
            List of query dictionaries
        """
        queries = []
        for _, row in df.iterrows():
            query = {
                "question": row.get("question", row.get("query", "")),
                "relevant_chunk_ids": row.get("relevant_chunk_ids", row.get("chunk_ids", [])),
                "dataset": dataset_name,
                "metadata": row.to_dict(),
            }
            queries.append(query)
        return queries

    def get_dataset(self, dataset_name: str) -> list[dict[str, Any]]:
        """
        Get a specific dataset by name.

        Args:
            dataset_name: One of 'cosqa', 'rubq', 'natural_questions', 'ru_rag', 'combined'

        Returns:
            List of query dictionaries
        """
        if dataset_name in self._datasets:
            return self._datasets[dataset_name]

        # Load if not cached
        loaders = {
            "cosqa": self.load_cosqa,
            "rubq": self.load_rubq,
            "natural_questions": self.load_natural_questions,
            "ru_rag": self.load_ru_rag,
            "combined": self.load_combined,
        }

        if dataset_name not in loaders:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        return loaders[dataset_name]()

    def get_all_datasets(self) -> dict[str, list[dict[str, Any]]]:
        """
        Load all datasets.

        Returns:
            Dictionary mapping dataset names to query lists
        """
        datasets = {}
        for name in ["cosqa", "rubq", "natural_questions", "ru_rag", "combined"]:
            try:
                datasets[name] = self.get_dataset(name)
            except Exception as e:
                logger.warning(f"Failed to load dataset {name}: {e}")
                datasets[name] = []
        return datasets

    def get_dataset_stats(self) -> dict[str, dict[str, Any]]:
        """
        Get statistics for all datasets.

        Returns:
            Dictionary with dataset statistics
        """
        stats = {}
        for name, queries in self.get_all_datasets().items():
            stats[name] = {
                "num_queries": len(queries),
                "avg_relevant_chunks": (
                    sum(len(q.get("relevant_chunk_ids", [])) for q in queries) / len(queries) if queries else 0
                ),
            }
        return stats
