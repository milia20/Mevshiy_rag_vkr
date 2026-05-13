"""
Ядро приложения RAG.

Модуль содержит основные компоненты и конфигурацию ядра приложения.
"""

from src.core.config import settings
from src.core.exceptions import RAGException, ConfigurationError, SearchError

__all__ = [
    "settings",
    "RAGException",
    "ConfigurationError",
    "SearchError",
]
