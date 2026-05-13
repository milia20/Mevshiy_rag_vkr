"""
Исключения приложения.

Модуль содержит кастомные исключения для обработки ошибок в RAG системе.
"""

from __future__ import annotations


class RAGException(Exception):
    """
    Базовое исключение для RAG системы.

    Все кастомные исключения наследуются от этого класса.
    """

    def __init__(self, message: str, details: str | None = None) -> None:
        """
        Инициализация исключения.

        Args:
            message: Основное сообщение об ошибке.
            details: Дополнительные детали (опционально).
        """
        super().__init__(message)
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, str | None]:
        """
        Преобразовать исключение в словарь.

        Returns:
            Словарь с информацией об ошибке.
        """
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class ConfigurationError(RAGException):
    """
    Исключение ошибки конфигурации.

    Вызывается при некорректной конфигурации приложения.
    """

    def __init__(self, message: str, details: str | None = None) -> None:
        """
        Инициализация исключения конфигурации.

        Args:
            message: Сообщение об ошибке.
            details: Детали ошибки.
        """
        super().__init__(
            message=f"Ошибка конфигурации: {message}",
            details=details,
        )


class SearchError(RAGException):
    """
    Исключение ошибки поиска.

    Вызывается при ошибках во время выполнения поиска.
    """

    def __init__(self, message: str, details: str | None = None) -> None:
        """
        Инициализация исключения поиска.

        Args:
            message: Сообщение об ошибке.
            details: Детали ошибки.
        """
        super().__init__(
            message=f"Ошибка поиска: {message}",
            details=details,
        )


class EmbeddingError(RAGException):
    """
    Исключение ошибки генерации эмбеддингов.

    Вызывается при ошибках во время создания векторных представлений.
    """

    def __init__(self, message: str, details: str | None = None) -> None:
        """
        Инициализация исключения эмбеддингов.

        Args:
            message: Сообщение об ошибке.
            details: Детали ошибки.
        """
        super().__init__(
            message=f"Ошибка эмбеддинга: {message}",
            details=details,
        )


class LLMError(RAGException):
    """
    Исключение ошибки LLM.

    Вызывается при ошибках взаимодействия с языковой моделью.
    """

    def __init__(self, message: str, details: str | None = None) -> None:
        """
        Инициализация исключения LLM.

        Args:
            message: Сообщение об ошибке.
            details: Детали ошибки.
        """
        super().__init__(
            message=f"Ошибка LLM: {message}",
            details=details,
        )


class QdrantError(RAGException):
    """
    Исключение ошибки Qdrant.

    Вызывается при ошибках взаимодействия с базой данных Qdrant.
    """

    def __init__(self, message: str, details: str | None = None) -> None:
        """
        Инициализация исключения Qdrant.

        Args:
            message: Сообщение об ошибке.
            details: Детали ошибки.
        """
        super().__init__(
            message=f"Ошибка Qdrant: {message}",
            details=details,
        )


class DocumentProcessingError(RAGException):
    """
    Исключение ошибки обработки документа.

    Вызывается при ошибках во время обработки документов.
    """

    def __init__(self, message: str, details: str | None = None) -> None:
        """
        Инициализация исключения обработки документа.

        Args:
            message: Сообщение об ошибке.
            details: Детали ошибки.
        """
        super().__init__(
            message=f"Ошибка обработки документа: {message}",
            details=details,
        )
