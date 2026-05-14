"""
Document Processor service for MkDocs RAG Plugin.

Модуль содержит класс DocumentProcessor для извлечения и обработки
контента из страниц MkDocs.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from mkdocs.structure.pages import Page

logger = logging.getLogger("mkdocs.plugins.rag_plugin")


class DocumentProcessor:
    """
    Сервис для обработки документов MkDocs.

    Извлекает контент из страниц MkDocs, очищает его и извлекает метаданные.
    Поддерживает русский (кириллический) текст.
    """

    def __init__(self) -> None:
        """Инициализация процессора документов."""
        self._header_pattern = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)
        self._code_block_pattern = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
        self._html_comment_pattern = re.compile(r"<!--.*?-->", re.DOTALL)
        self._nav_artifact_pattern = re.compile(r"¶|⚓|#.*", re.MULTILINE)

    def extract_markdown_content(self, page: Page) -> str:
        """
        Извлечь raw Markdown из объекта страницы MkDocs.

        Сохраняет иерархию заголовков (h1-h6), извлекает блоки кода с языком,
        таблицы и списки с правильным форматированием.

        Args:
            page: Объект страницы MkDocs.

        Returns:
            Raw Markdown контент страницы.
        """
        try:
            # Получаем raw markdown из page.markdown
            if hasattr(page, "markdown") and page.markdown:
                content = page.markdown
            elif hasattr(page, "content"):
                # Если markdown недоступен, пробуем получить content
                content = str(page.content)
            else:
                logger.warning(f"Страница {page.title} не имеет контента")
                return ""

            # Сохраняем заголовки и структуру
            content = self._preserve_structure(content)

            return content.strip()

        except Exception as e:
            logger.error(f"Ошибка извлечения контента со страницы {page.title}: {e}")
            return ""

    def _preserve_structure(self, content: str) -> str:
        """
        Сохранить структуру Markdown.

        Args:
            content: Raw Markdown контент.

        Returns:
            Контент с сохраненной структурой.
        """
        # Нормализуем заголовки - убеждаемся что есть пробел после #
        content = re.sub(r"^#{1,6}(?!\s)", r"\g<0> ", content, flags=re.MULTILINE)

        # Сохраняем блоки кода - убеждаемся что они правильно оформлены
        def normalize_code_block(match: re.Match) -> str:
            lang = match.group(1) or ""
            code = match.group(2).strip()
            return f"```{lang}\n{code}\n```"

        content = self._code_block_pattern.sub(normalize_code_block, content)

        return content

    def clean_content(self, markdown: str) -> str:
        """
        Очистить контент от артефактов.

        Удаляет:
        - Навигационные артефакты (символы ¶, ⚓)
        - HTML комментарии
        - Избыточные пробелы

        Args:
            markdown: Raw Markdown контент.

        Returns:
            Очищенный Markdown.
        """
        if not markdown:
            return ""

        text = markdown

        # Удаляем HTML комментарии
        text = self._html_comment_pattern.sub("", text)

        # Удаляем навигационные артефакты
        text = self._nav_artifact_pattern.sub("", text)

        # Удаляем избыточные переводы строк (более 2 подряд)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Нормализуем пробелы в начале/конце строк
        lines = text.split("\n")
        cleaned_lines = [line.rstrip() for line in lines]
        text = "\n".join(cleaned_lines)

        return text.strip()

    def extract_metadata(self, page: Page) -> dict[str, Any]:
        """
        Извлечь метаданные из страницы MkDocs.

        Извлекает:
        - url: URL страницы
        - title: Заголовок страницы
        - hierarchical_path: Иерархический путь в навигации
        - source_file: Исходный файл
        - last_modified: Дата последнего изменения

        Args:
            page: Объект страницы MkDocs.

        Returns:
            Словарь с метаданными.
        """
        metadata: dict[str, Any] = {}

        try:
            # URL страницы
            metadata["url"] = page.url if hasattr(page, "url") else ""

            # Заголовок
            metadata["title"] = page.title if hasattr(page, "title") else ""

            # Иерархический путь
            metadata["hierarchical_path"] = self._extract_hierarchical_path(page)

            # Исходный файл
            metadata["source_file"] = self._extract_source_file(page)

            # Дата последнего изменения
            metadata["last_modified"] = self._extract_last_modified(page)

            # Дополнительные метаданные из page.meta
            if hasattr(page, "meta") and page.meta:
                for key, value in page.meta.items():
                    if key not in metadata:
                        metadata[key] = value

        except Exception as e:
            logger.error(f"Ошибка извлечения метаданных со страницы {page.title}: {e}")

        return metadata

    def _extract_hierarchical_path(self, page: Page) -> list[str]:
        """
        Извлечь иерархический путь страницы.

        Args:
            page: Объект страницы MkDocs.

        Returns:
            Список заголовков пути.
        """
        path: list[str] = []

        try:
            # Пытаемся получить путь из parent
            current = page
            while hasattr(current, "parent") and current.parent:
                if hasattr(current.parent, "title") and current.parent.title:
                    path.insert(0, current.parent.title)
                current = current.parent

            # Добавляем текущий заголовок
            if hasattr(page, "title") and page.title:
                path.append(page.title)

        except Exception as e:
            logger.debug(f"Не удалось извлечь иерархический путь: {e}")

        return path

    def _extract_source_file(self, page: Page) -> str:
        """
        Извлечь путь к исходному файлу.

        Args:
            page: Объект страницы MkDocs.

        Returns:
            Путь к исходному файлу.
        """
        try:
            if hasattr(page, "file") and page.file:
                if hasattr(page.file, "src_path"):
                    return page.file.src_path
                elif hasattr(page.file, "abs_src_path") and page.file.abs_src_path:
                    return str(page.file.abs_src_path)
        except Exception as e:
            logger.debug(f"Не удалось извлечь исходный файл: {e}")

        return ""

    def _extract_last_modified(self, page: Page) -> datetime | None:
        """
        Извлечь дату последнего изменения.

        Args:
            page: Объект страницы MkDocs.

        Returns:
            Дата последнего изменения или None.
        """
        try:
            # Пытаемся получить из page.file
            if hasattr(page, "file") and page.file:
                if hasattr(page.file, "abs_src_path") and page.file.abs_src_path:
                    file_path = Path(page.file.abs_src_path)
                    if file_path.exists():
                        mtime = file_path.stat().st_mtime
                        return datetime.fromtimestamp(mtime)
        except Exception as e:
            logger.debug(f"Не удалось извлечь дату изменения: {e}")

        return None

    def process_page(self, page: Page) -> tuple[str, dict[str, Any]]:
        """
        Полная обработка страницы: извлечение + очистка + метаданные.

        Args:
            page: Объект страницы MkDocs.

        Returns:
            Кортеж (очищенный контент, метаданные).
        """
        raw_content = self.extract_markdown_content(page)
        cleaned_content = self.clean_content(raw_content)
        metadata = self.extract_metadata(page)

        return cleaned_content, metadata

    def extract_code_blocks(self, content: str) -> list[dict[str, str]]:
        """
        Извлечь блоки кода из контента.

        Args:
            content: Markdown контент.

        Returns:
            Список словарей с языком и кодом.
        """
        code_blocks = []

        for match in self._code_block_pattern.finditer(content):
            lang = match.group(1) or "text"
            code = match.group(2).strip()
            code_blocks.append({"language": lang, "code": code})

        return code_blocks

    def extract_tables(self, content: str) -> list[str]:
        """
        Извлечь таблицы из Markdown контента.

        Args:
            content: Markdown контент.

        Returns:
            Список таблиц в формате Markdown.
        """
        tables = []
        lines = content.split("\n")
        current_table: list[str] = []
        in_table = False

        for line in lines:
            # Проверяем является ли строка частью таблицы
            if re.match(r"^\s*\|.*\|\s*$", line):
                in_table = True
                current_table.append(line)
            elif re.match(r"^\s*\|?\s*[-:]+[-:|\s]*\|?\s*$", line):
                # Разделитель таблицы
                if current_table:
                    current_table.append(line)
                in_table = True
            else:
                if in_table and current_table:
                    tables.append("\n".join(current_table))
                    current_table = []
                in_table = False

        # Добавляем последнюю таблицу
        if current_table:
            tables.append("\n".join(current_table))

        return tables
