"""
Experiment Configuration for Qdrant Indexing Thesis

Define multiple collection configurations for comparative experiments.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from qdrant_uploader import QdrantIndexer, HNSWConfig

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single indexing experiment"""
    name: str
    collection_name: str
    hnsw_config: HNSWConfig
    description: str


# Define experiment configurations
EXPERIMENTS = [
    ExperimentConfig(
        name="baseline",
        collection_name="docs_baseline",
        hnsw_config=HNSWConfig(m=16, ef_construct=100),
        description="Default HNSW parameters (baseline)",
    ),
    ExperimentConfig(
        name="optimized_speed",
        collection_name="docs_optimized_speed",
        hnsw_config=HNSWConfig(m=32, ef_construct=256),
        description="Optimized for search speed (higher m, ef)",
    ),
    ExperimentConfig(
        name="optimized_memory",
        collection_name="docs_optimized_memory",
        hnsw_config=HNSWConfig(m=8, ef_construct=50),
        description="Optimized for memory efficiency (lower m, ef)",
    ),
    ExperimentConfig(
        name="high_precision",
        collection_name="docs_high_precision",
        hnsw_config=HNSWConfig(m=64, ef_construct=512),
        description="High precision indexing (maximum m, ef)",
    ),
]


def run_experiment(
    exp: ExperimentConfig,
    input_file: str,
    vector_size: int = 768,
    batch_size: int = 256,
) -> dict:
    """Run a single indexing experiment"""
    print(f"\n{'='*60}")
    print(f"Running Experiment: {exp.name}")
    print(f"{'='*60}")
    print(f"Collection: {exp.collection_name}")
    print(f"HNSW: m={exp.hnsw_config.m}, ef_construct={exp.hnsw_config.ef_construct}")
    print(f"Description: {exp.description}")

    indexer = None
    try:
        # Initialize indexer
        indexer = QdrantIndexer(
            collection_name=exp.collection_name,
            in_memory=False,
        )

        # Create collection with force_recreate=True
        indexer.create_collection(
            vector_size=vector_size,
            hnsw_config=exp.hnsw_config,
            force_recreate=True,  # This parameter is now properly defined
        )

        # Index documents
        stats = indexer.index_documents_from_file(
            file_path=input_file,
            batch_size=batch_size,
        )

        # Create payload indexes
        indexer.create_default_payload_indexes()

        # Verify
        verification = indexer.verify_index()

        result = {
            "experiment": exp.name,
            "collection": exp.collection_name,
            "hnsw_config": exp.hnsw_config.to_dict(),
            "upload_stats": stats,
            "verification": verification,
            "status": "success",
        }

        print(f"✓ Experiment {exp.name} completed successfully")
        print(f"  Points indexed: {verification.get('points_count', 'N/A')}")

        return result

    except Exception as e:
        logger.error(f"Experiment {exp.name} failed: {e}")
        return {
            "experiment": exp.name,
            "collection": exp.collection_name,
            "status": "failed",
            "error": str(e),
        }
    finally:
        if indexer:
            indexer.close()


def run_all_experiments(
    input_file: str,
    vector_size: int = 768,
    batch_size: int = 256,
    output_path: str = "experiments/indexing_results.json",
) -> List[dict]:
    """Run all indexing experiments"""
    results = []

    for exp in EXPERIMENTS:
        result = run_experiment(
            exp=exp,
            input_file=input_file,
            vector_size=vector_size,
            batch_size=batch_size,
        )
        results.append(result)

    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Experiment results saved to {output_file}")
    print(f"{'='*60}")

    # Print summary
    success_count = sum(1 for r in results if r.get("status") == "success")
    print(f"\nSummary: {success_count}/{len(results)} experiments succeeded")

    return results


def main():
    """CLI entry point for running experiments"""
    parser = argparse.ArgumentParser(
        description="Run Qdrant indexing experiments for thesis"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file with chunks"
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=768,
        help="Vector dimension (default: 768)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for uploading (default: 256)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=Path(__file__).parent / "experiments/indexing_results.json",
        help="Output path for results JSON"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["all", "baseline", "optimized_speed", "optimized_memory", "high_precision"],
        default="all",
        help="Which experiment(s) to run"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Filter experiments if specific one requested
    experiments_to_run = EXPERIMENTS
    if args.experiment != "all":
        experiments_to_run = [e for e in EXPERIMENTS if e.name == args.experiment]

    if not experiments_to_run:
        print(f"No experiments found for: {args.experiment}")
        return

    # Run experiments
    results = []
    for exp in experiments_to_run:
        result = run_experiment(
            exp=exp,
            input_file=args.input,
            vector_size=args.vector_size,
            batch_size=args.batch_size,
        )
        results.append(result)

    # Save results
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")

    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "✓" if r.get("status") == "success" else "✗"
        print(f"{status} {r['experiment']}: {r.get('status', 'unknown')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()