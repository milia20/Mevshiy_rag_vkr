"""
Evaluation module for retrieval metrics.

Provides IR metrics: Hit@k, MRR@k, NDCG@k, latency measurement.
Multi-dataset evaluation support.
"""

from .dataset_manager import DatasetManager, DatasetConfig
from .metrics import RetrievalEvaluator

__all__ = ["RetrievalEvaluator", "DatasetManager", "DatasetConfig"]
