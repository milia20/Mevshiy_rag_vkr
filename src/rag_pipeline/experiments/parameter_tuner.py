"""
Experimental parameter tuning framework.

Runs experiments with different parameter combinations:
- Chunk size variations
- Model variations
- Strategy variations
- Other hyperparameters

Features:
- Error isolation (one experiment failure doesn't affect others)
- Metrics collection at different @k values
- CSV export of results
- Progress tracking
"""

import ast
import csv
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    # Experiment metadata
    experiment_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Chunking parameters
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Model parameters
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_bge_small: bool = False
    use_bge_m3: bool = False
    use_splade: bool = False

    # Search parameters
    top_k: int = 10
    rrf_k: int = 60

    # Strategy
    strategy: str = "dense"

    # Evaluation
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "dense_model": self.dense_model,
            "use_bge_small": self.use_bge_small,
            "use_bge_m3": self.use_bge_m3,
            "use_splade": self.use_splade,
            "top_k": self.top_k,
            "rrf_k": self.rrf_k,
            "strategy": self.strategy,
            "k_values": self.k_values,
        }


@dataclass
class ExperimentResult:
    """Result of a single experiment."""

    config: ExperimentConfig
    success: bool
    error_message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    total_queries: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = self.config.to_dict()
        result.update(
            {
                "success": self.success,
                "error_message": self.error_message,
                "latency_ms": self.latency_ms,
                "total_queries": self.total_queries,
            }
        )
        # Flatten metrics
        for metric_name, metric_value in self.metrics.items():
            if isinstance(metric_value, dict):
                for k, v in metric_value.items():
                    result[f"{metric_name}_{k}"] = v
            else:
                result[metric_name] = metric_value
        return result


class ParameterTuner:
    """
    Experimental parameter tuning framework.

    Runs experiments with different parameter combinations,
    collects metrics, and saves results to CSV.
    """

    def __init__(
        self,
        output_dir: str = "experiments/results",
        results_file: str = "experiments_results.csv",
    ):
        """
        Initialize parameter tuner.

        Args:
            output_dir: Directory to save results
            results_file: CSV file name for results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = results_file
        self.results_path = self.output_dir / self.results_file

        # Initialize CSV file with headers
        self._initialize_csv()

    def _initialize_csv(self):
        """Initialize CSV file with headers."""
        if not self.results_path.exists():
            # Create with sample config to get headers
            sample_config = ExperimentConfig(experiment_id="sample")
            sample_result = ExperimentResult(config=sample_config, success=False)
            headers = list(sample_result.to_dict().keys())

            with open(self.results_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
            logger.info(f"Initialized results CSV: {self.results_path}")

    def _save_result(self, result: ExperimentResult):
        """Save single experiment result to CSV."""
        result_dict = result.to_dict()

        # Check if file exists and has headers
        if self.results_path.exists():
            try:
                existing_df = pd.read_csv(self.results_path)
                existing_columns = set(existing_df.columns)
                new_columns = set(result_dict.keys())

                # Add new columns if needed
                if new_columns - existing_columns:
                    for col in new_columns - existing_columns:
                        existing_df[col] = None
                    existing_df.to_csv(self.results_path, index=False)
            except Exception as e:
                logger.warning(f"Could not check/expand CSV columns: {e}")

        # Append result - header=False при режиме 'a' (файл уже существует после _initialize_csv)
        df = pd.DataFrame([result_dict])
        df.to_csv(self.results_path, mode="a", header=False, index=False)
        logger.info(f"Saved result for experiment: {result.config.experiment_id}")

    def run_experiment(
        self,
        config: ExperimentConfig,
        experiment_fn: Callable[[ExperimentConfig, list[dict]], dict[str, Any]],
        queries: list[dict] | None = None,
    ) -> ExperimentResult:
        """
        Run a single experiment with error isolation.

        Args:
            config: Experiment configuration
            experiment_fn: Function to run the experiment

        Returns:
            Experiment result
        """
        logger.info(f"Running experiment: {config.experiment_id}")
        start_time = time.perf_counter()

        try:
            # Run experiment with queries
            if queries is None:
                queries = []
            metrics = experiment_fn(config, queries)
            success = True
            error_message = ""
            logger.info(f"Experiment {config.experiment_id} completed successfully")
        except Exception as e:
            success = False
            error_message = str(e)
            metrics = {}
            logger.error(f"Experiment {config.experiment_id} failed: {e}")
            logger.exception(e)

        latency_ms = (time.perf_counter() - start_time) * 1000

        result = ExperimentResult(
            config=config,
            success=success,
            error_message=error_message,
            metrics=metrics,
            latency_ms=latency_ms,
        )

        self._save_result(result)
        return result

    def generate_chunk_size_grid(
        self,
        chunk_sizes: list[int] = [256, 512, 768, 1024],
        base_config: ExperimentConfig | None = None,
    ) -> list[ExperimentConfig]:
        """
        Generate experiment configs for chunk size variations.

        Args:
            chunk_sizes: List of chunk sizes to try
            base_config: Base configuration to extend

        Returns:
            List of experiment configurations
        """
        base = base_config or ExperimentConfig(experiment_id="base")
        configs = []

        for i, chunk_size in enumerate(chunk_sizes):
            config = ExperimentConfig(
                experiment_id=f"chunk_size_{chunk_size}_{i}",
                chunk_size=chunk_size,
                chunk_overlap=base.chunk_overlap,
                dense_model=base.dense_model,
                strategy=base.strategy,
                top_k=base.top_k,
                k_values=base.k_values,
            )
            configs.append(config)

        return configs

    def generate_model_grid(
        self,
        models: list[tuple[str, bool, bool, bool]] = [
            ("sentence-transformers/all-MiniLM-L6-v2", False, False, False),
            ("BAAI/bge-small-en-v1.5", True, False, False),
            ("BAAI/bge-m3", False, True, False),
        ],
        base_config: ExperimentConfig | None = None,
    ) -> list[ExperimentConfig]:
        """
        Generate experiment configs for model variations.

        Args:
            models: List of (model_name, use_bge_small, use_bge_m3, use_splade) tuples
            base_config: Base configuration to extend

        Returns:
            List of experiment configurations
        """
        base = base_config or ExperimentConfig(experiment_id="base")
        configs = []

        for i, (model_name, use_bge_small, use_bge_m3, use_splade) in enumerate(models):
            config = ExperimentConfig(
                experiment_id=f"model_{model_name.replace('/', '_')}_{i}",
                chunk_size=base.chunk_size,
                chunk_overlap=base.chunk_overlap,
                dense_model=model_name,
                use_bge_small=use_bge_small,
                use_bge_m3=use_bge_m3,
                use_splade=use_splade,
                strategy=base.strategy,
                top_k=base.top_k,
                k_values=base.k_values,
            )
            configs.append(config)

        return configs

    def generate_strategy_grid(
        self,
        strategies: list[str] = ["dense", "sparse", "hybrid_rrf", "hybrid_dbsf", "colbert"],
        base_config: ExperimentConfig | None = None,
    ) -> list[ExperimentConfig]:
        """
        Generate experiment configs for strategy variations.

        Args:
            strategies: List of strategies to try
            base_config: Base configuration to extend

        Returns:
            List of experiment configurations
        """
        base = base_config or ExperimentConfig(experiment_id="base")
        configs = []

        for i, strategy in enumerate(strategies):
            config = ExperimentConfig(
                experiment_id=f"strategy_{strategy}_{i}",
                chunk_size=base.chunk_size,
                chunk_overlap=base.chunk_overlap,
                dense_model=base.dense_model,
                strategy=strategy,
                top_k=base.top_k,
                k_values=base.k_values,
            )
            configs.append(config)

        return configs

    def generate_top_k_grid(
        self,
        top_k_values: list[int] = [5, 10, 20, 50],
        base_config: ExperimentConfig | None = None,
    ) -> list[ExperimentConfig]:
        """
        Generate experiment configs for top_k variations.

        Args:
            top_k_values: List of top_k values to try
            base_config: Base configuration to extend

        Returns:
            List of experiment configurations
        """
        base = base_config or ExperimentConfig(experiment_id="base")
        configs = []

        for i, top_k in enumerate(top_k_values):
            config = ExperimentConfig(
                experiment_id=f"top_k_{top_k}_{i}",
                chunk_size=base.chunk_size,
                chunk_overlap=base.chunk_overlap,
                dense_model=base.dense_model,
                strategy=base.strategy,
                top_k=top_k,
                k_values=base.k_values,
            )
            configs.append(config)

        return configs

    def generate_rrf_k_grid(
        self,
        rrf_k_values: list[int] = [30, 60, 100],
        base_config: ExperimentConfig | None = None,
    ) -> list[ExperimentConfig]:
        """
        Generate experiment configs for RRF k variations.

        Args:
            rrf_k_values: List of RRF k values to try
            base_config: Base configuration to extend

        Returns:
            List of experiment configurations
        """
        base = base_config or ExperimentConfig(experiment_id="base")
        configs = []

        for i, rrf_k in enumerate(rrf_k_values):
            config = ExperimentConfig(
                experiment_id=f"rrf_k_{rrf_k}_{i}",
                chunk_size=base.chunk_size,
                chunk_overlap=base.chunk_overlap,
                dense_model=base.dense_model,
                strategy="hybrid_rrf",
                top_k=base.top_k,
                rrf_k=rrf_k,
                k_values=base.k_values,
            )
            configs.append(config)

        return configs

    def run_grid_search(
        self,
        configs: list[ExperimentConfig],
        experiment_fn: Callable[[ExperimentConfig, list[dict]], dict[str, Any]],
        queries: list[dict] | None = None,
        max_failures: int = 5,
    ) -> list[ExperimentResult]:
        """
        Run grid search over multiple experiment configs.

        Args:
            configs: List of experiment configurations
            experiment_fn: Function to run each experiment
            queries: Queries to use for evaluation
            max_failures: Maximum number of consecutive failures before stopping

        Returns:
            List of experiment results
        """
        logger.info(f"Starting grid search with {len(configs)} experiments")
        results = []
        consecutive_failures = 0

        for i, config in enumerate(configs, start=1):
            logger.info(f"Progress: {i}/{len(configs)}")

            result = self.run_experiment(config, experiment_fn, queries)
            results.append(result)

            if result.success:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                logger.warning(f"Consecutive failures: {consecutive_failures}/{max_failures}")

                if consecutive_failures >= max_failures:
                    logger.error(f"Stopping due to {max_failures} consecutive failures")
                    break

        logger.info(f"Grid search completed. Success: {sum(r.success for r in results)}/{len(results)}")
        return results

    def load_results(self) -> pd.DataFrame:
        """
        Load all experimental results from CSV.

        Returns:
            DataFrame with all results
        """
        if self.results_path.exists():
            df = pd.read_csv(self.results_path)
            logger.info(f"Loaded {len(df)} experimental results")
            return df
        else:
            logger.warning(f"No results file found: {self.results_path}")
            return pd.DataFrame()

    def get_best_config(
        self,
        metric: str = "hit_at_k_10",
        maximize: bool = True,
    ) -> ExperimentConfig | None:
        """
        Get best configuration based on a metric.

        Args:
            metric: Metric name to optimize (e.g., "hit_at_k_10", "recall_at_k_10")
            maximize: Whether to maximize or minimize the metric

        Returns:
            Best configuration or None if no results
        """
        df = self.load_results()
        if df.empty:
            return None

        # Filter successful experiments
        successful = df[df["success"] == True]

        if metric not in successful.columns:
            logger.warning(f"Metric {metric} not found in results. Available: {list(successful.columns)}")
            return None

        # Sort by metric
        if maximize:
            best_row = successful.nlargest(1, metric).iloc[0]
        else:
            best_row = successful.nsmallest(1, metric).iloc[0]

        # Reconstruct config - БЕЗОПАСНО: используем ast.literal_eval вместо eval()
        k_values_raw = best_row["k_values"]
        if isinstance(k_values_raw, str):
            try:
                k_values = ast.literal_eval(k_values_raw)
            except (ValueError, SyntaxError):
                logger.warning(f"Could not parse k_values: {k_values_raw}, using default")
                k_values = [1, 3, 5, 10]
        else:
            k_values = k_values_raw

        config = ExperimentConfig(
            experiment_id=best_row["experiment_id"],
            chunk_size=int(best_row["chunk_size"]),
            chunk_overlap=int(best_row["chunk_overlap"]),
            dense_model=best_row["dense_model"],
            use_bge_small=bool(best_row["use_bge_small"]),
            use_bge_m3=bool(best_row["use_bge_m3"]),
            use_splade=bool(best_row["use_splade"]),
            top_k=int(best_row["top_k"]),
            rrf_k=int(best_row.get("rrf_k", 60)),
            strategy=best_row["strategy"],
            k_values=k_values,
        )

        metric_value = best_row[metric]
        logger.info(f"Best config: {config.experiment_id} with {metric}={metric_value:.4f}")
        return config
