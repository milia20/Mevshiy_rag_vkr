"""
Ядро приложения RAG.

Модуль содержит основные компоненты и конфигурацию ядра приложения.
"""

from src.core.config import settings
from src.core.exceptions import ConfigurationError, RAGException, SearchError

__all__ = [
    "ConfigurationError",
    "RAGException",
    "SearchError",
    "settings",
]
