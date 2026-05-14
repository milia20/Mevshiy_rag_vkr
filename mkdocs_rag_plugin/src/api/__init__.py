"""
API модули для RAG системы.
"""

from src.api.health import router as health_router
from src.api.index import router as index_router
from src.api.models import router as models_router
from src.api.query import router as query_router

__all__ = ["health_router", "index_router", "models_router", "query_router"]
