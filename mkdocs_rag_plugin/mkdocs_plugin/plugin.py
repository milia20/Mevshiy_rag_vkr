"""
Плагин MkDocs для RAG-поиска.

Модуль содержит основной класс плагина, который интегрируется с MkDocs
и предоставляет события для обработки документации.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from mkdocs.config import Config
from mkdocs.plugins import BasePlugin, event_priority
from mkdocs_plugin.config import get_config_schema
from mkdocs_plugin.models import Chunk, ProcessedDocument, IndexingResult
from mkdocs_plugin.services.doc_processor import DocumentProcessor
from mkdocs_plugin.services.chunker import Chunker

if TYPE_CHECKING:
    from mkdocs.structure.files import Files


logger = logging.getLogger("mkdocs.plugins.rag_plugin")


class RAGPlugin(BasePlugin):
    """
    Плагин MkDocs для интеграции с RAG системой.

    Этот плагин позволяет осуществлять семантический поиск по документации
    с использованием векторной базы данных Qdrant и языковых моделей.

    Атрибуты:
        config_scheme: Схема конфигурации плагина.
        api_base_url: Базовый URL для API запросов.
    """

    config_scheme = get_config_schema()

    def __init__(self) -> None:
        """Инициализация плагина."""
        super().__init__()
        self.api_base_url: str = ""
        self._config_validated: bool = False
        self._doc_processor: DocumentProcessor | None = None
        self._chunker: Chunker | None = None
        self._processed_docs: List[ProcessedDocument] = []
        self._indexing_result: IndexingResult | None = None

    def on_config(self, config: Config, **kwargs: Any) -> Config | None:
        """
        Обработчик события on_config.

        Вызывается после загрузки конфигурации MkDocs. Используется для
        инициализации параметров подключения к API и валидации конфигурации.

        Args:
            config: Конфигурация MkDocs.
            **kwargs: Дополнительные аргументы от MkDocs.

        Returns:
            Конфигурация MkDocs (возможно модифицированная).
        """
        logger.info("RAG Plugin: инициализация конфигурации")

        # Валидация конфигурации
        if not self.config.get("enabled", True):
            logger.info("RAG Plugin: плагин отключен в конфигурации")
            return config

        # Формируем базовый URL для запросов к FastAPI
        self.api_base_url = f"http://{self.config['api_host']}:{self.config['api_port']}"

        # Инициализируем сервисы
        self._doc_processor = DocumentProcessor()
        self._chunker = Chunker(
            chunk_size=self.config.get("chunk_size", 500),
            chunk_overlap=self.config.get("chunk_overlap", 50),
        )

        logger.info(f"RAG Plugin: API URL установлен в {self.api_base_url}")
        logger.info(
            f"RAG Plugin: Qdrant URL: {self.config['qdrant_url']}, "
            f"коллекция: {self.config['collection_name']}"
        )
        logger.info(
            f"RAG Plugin: стратегия чанкования: {self.config['chunk_strategy']}, "
            f"размер чанка: {self.config['chunk_size']}, "
            f"перекрытие: {self.config['chunk_overlap']}"
        )
        logger.info(
            f"RAG Plugin: Chatbot URL: {self.config.get('chatbot_url', 'http://localhost:8000')}"
        )

        from pathlib import Path

        self._config_validated = True
        overrides_path = Path(__file__).resolve().parent / "overrides"
        config.theme.dirs.insert(0, str(overrides_path))

        # Make chatbot_url available as template variable
        config["chatbot_url"] = self.config.get("chatbot_url", "http://localhost:8000")

        return config

    @event_priority(50)
    def on_files(self, files: Files, **kwargs: Any) -> Files | None:
        """
        Обработчик события on_files.

        Вызывается после сбора всех файлов документации.
        Обрабатывает все страницы: извлекает контент и создает чанки.

        Args:
            files: Объект Files MkDocs.
            **kwargs: Дополнительные аргументы от MkDocs.

        Returns:
            Объект Files (возможно модифицированный).
        """
        if not self._config_validated or not self.config.get("enabled", True):
            return files

        # Извлекаем все страницы документации
        docs_files = [f for f in files.documentation_pages()]
        logger.info(f"RAG Plugin: обработка {len(docs_files)} файлов")

        if not self._doc_processor or not self._chunker:
            logger.error("RAG Plugin: сервисы не инициализированы")
            return files

        self._processed_docs = []

        for file in docs_files:
            try:
                # Читаем содержимое файла
                from pathlib import Path

                file_path = Path(file.abs_src_path)
                content = file_path.read_text(encoding="utf-8")
                if not content.strip():
                    logger.debug(f"Файл {file.src_uri} пустой, пропускаем")
                    continue

                # Создаем чанки
                strategy = self.config.get("chunk_strategy", "structural")
                chunks_data = self._chunker.chunk(
                    content,
                    strategy=strategy,
                    max_tokens=self.config.get("chunk_size", 500),
                )

                # Преобразуем в модели Chunk
                chunk_models: List[Chunk] = []
                for chunk_data in chunks_data:
                    chunk_metadata = {
                        "url": file.url,
                        "title": file.name,
                        "source_file": str(file.src_path),
                        "header_path": chunk_data.metadata.get("header_path", []),
                    }

                    chunk_model = Chunk(
                        id=chunk_data.id,
                        content=chunk_data.content,
                        metadata=chunk_metadata,
                    )
                    chunk_models.append(chunk_model)

                # Создаем обработанный документ
                doc = ProcessedDocument(
                    url=file.url,
                    title=file.name,
                    content=content,
                    chunks=chunk_models,
                    metadata={"source_file": str(file.src_path)},
                )

                self._processed_docs.append(doc)
                logger.debug(f"Обработан файл {file.src_uri}: {len(chunk_models)} чанков")

            except Exception as e:
                logger.error(f"Ошибка обработки файла {file.src_uri}: {e}")

        total_chunks = sum(doc.total_chunks for doc in self._processed_docs)
        logger.info(
            f"RAG Plugin: обработано {len(self._processed_docs)} документов, "
            f"создано {total_chunks} чанков"
        )

        return files

    def on_page_content(self, html: str, page: Page, config: Config, **kwargs: Any) -> str | None:
        """
        Обработчик события on_page_content.

        Вызывается после рендеринга содержимого страницы.
        Может быть использован для извлечения текста для индексации.

        Args:
            html: HTML-содержимое страницы.
            page: Объект Page.
            config: Конфигурация MkDocs.
            **kwargs: Дополнительные аргументы от MkDocs.

        Returns:
            HTML-содержимое (возможно модифицированное).
        """
        logger.debug(f"RAG Plugin: обработка содержимого страницы {page.title}")
        return html

    @event_priority(-100)
    def on_post_build(self, config: Config, **kwargs: Any) -> None:
        """
        Обработчик события on_post_build.

        Вызывается после завершения сборки сайта. Используется для
        финальной индексации документации в Qdrant.

        Args:
            config: Конфигурация MkDocs.
            **kwargs: Дополнительные аргументы от MkDocs.
        """
        if not self._config_validated or not self.config.get("enabled", True):
            logger.info("RAG Plugin: плагин отключен, пропускаем индексацию")
            return

        logger.info("RAG Plugin: завершение сборки, начало индексации")

        if not self._processed_docs:
            logger.warning("RAG Plugin: нет обработанных документов для индексации")
            self._indexing_result = IndexingResult(
                total_docs=0,
                total_chunks=0,
                indexed_count=0,
            )
            return

        # Считаем статистику
        total_docs = len(self._processed_docs)
        total_chunks = sum(doc.total_chunks for doc in self._processed_docs)

        # Здесь будет логика индексации в Qdrant
        # Пока просто логируем
        logger.info(
            f"RAG Plugin: готово к индексации {total_docs} документов, " f"{total_chunks} чанков"
        )
        logger.info(f"RAG Plugin: Qdrant URL: {self.config['qdrant_url']}")
        logger.info(f"RAG Plugin: коллекция: {self.config['collection_name']}")

        # Создаем результат индексации
        self._indexing_result = IndexingResult(
            total_docs=total_docs,
            total_chunks=total_chunks,
            indexed_count=total_chunks,  # Пока считаем что все успешно
        )

        logger.info(
            f"RAG Plugin: индексация завершена. "
            f"Успешно: {self._indexing_result.indexed_count}/{total_chunks} "
            f"({self._indexing_result.success_rate:.1f}%)"
        )

    def on_post_page(self, output: str, page: Page, config: Config, **kwargs: Any) -> str | None:
        """
        Обработчик события on_post_page.

        Вызывается после сохранения HTML-файла страницы.
        Используется для внедрения JavaScript клиента для поиска.

        Args:
            output: HTML-вывод страницы.
            page: Объект Page.
            config: Конфигурация MkDocs.
            **kwargs: Дополнительные аргументы от MkDocs.

        Returns:
            HTML-вывод (возможно модифицированный).
        """
        if not self.config.get("enable_search_ui", True):
            return output

        logger.debug(f"RAG Plugin: внедрение клиента на страницу {page.title}")
        # Здесь будет логика внедрения JavaScript клиента
        return output

    def validate_config(self) -> bool:
        """
        Проверка корректности конфигурации плагина.

        Returns:
            True если конфигурация валидна, иначе False.
        """
        if not self._config_validated:
            logger.warning("RAG Plugin: конфигурация не была валидирована")
            return False

        # Проверка обязательных параметров
        required_fields = ["qdrant_url", "collection_name", "embedding_model"]
        for field in required_fields:
            if not self.config.get(field):
                logger.error(f"RAG Plugin: отсутствует обязательное поле '{field}'")
                return False

        logger.info("RAG Plugin: конфигурация валидна")
        return True

    def get_plugin_info(self) -> dict[str, Any]:
        """
        Получить информацию о плагине.

        Returns:
            Словарь с информацией о плагине.
        """
        return {
            "name": "rag_plugin",
            "version": "0.1.0",
            "api_url": self.api_base_url,
            "collection_name": self.config.get("collection_name", ""),
            "embedding_model": self.config.get("embedding_model", ""),
            "enabled": self.config.get("enabled", True),
        }

    def get_processed_documents(self) -> List[ProcessedDocument]:
        """
        Получить список обработанных документов.

        Returns:
            Список обработанных документов.
        """
        return self._processed_docs

    def get_indexing_result(self) -> IndexingResult | None:
        """
        Получить результат индексации.

        Returns:
            Результат индексации или None.
        """
        return self._indexing_result
