"""
MkDocs RAG Plugin - плагин для интеграции RAG системы с MkDocs документацией.

Этот пакет предоставляет плагин для MkDocs, который позволяет осуществлять
семантический поиск по документации с использованием векторной базы данных
Qdrant и языковых моделей.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from mkdocs_plugin.plugin import RAGPlugin


def make_plugin() -> RAGPlugin:
    """
    Factory function for creating the RAG plugin instance.

    This function is used by MkDocs to instantiate the plugin.

    Returns:
        RAGPlugin instance.
    """
    return RAGPlugin()


__all__ = ["RAGPlugin", "make_plugin"]
