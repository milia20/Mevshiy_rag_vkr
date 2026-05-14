"""
Chunker service for MkDocs RAG Plugin.

Модуль содержит класс Chunker с различными стратегиями чанкования:
- structural: разбиение по заголовкам Markdown
- fixed: фиксированный размер чанков с перекрытием
- recursive: рекурсивное разбиение по разделителям
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("mkdocs.plugins.rag_plugin")


@dataclass
class Chunk:
    """
    Модель чанка документа.

    Атрибуты:
        id: Уникальный идентификатор чанка.
        content: Текст чанка.
        metadata: Метаданные чанка.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = field(default="")
    metadata: dict[str, Any] = field(default_factory=dict)


class Chunker:
    """
    Сервис для чанкования текста с различными стратегиями.

    Поддерживает три стратегии:
    - structural: разбиение по заголовкам Markdown
    - fixed: фиксированный размер чанков
    - recursive: рекурсивное разбиение по разделителям

    Все методы поддерживают русский (кириллический) текст.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        """
        Инициализация чанкера.

        Args:
            chunk_size: Размер чанка в символах/токенах.
            chunk_overlap: Перекрытие между чанками.
            separators: Разделители для recursive стратегии.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]
        self._header_pattern = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)
        self._code_block_pattern = re.compile(r"```[\w]*\n.*?```", re.DOTALL)

    def structural_chunk(self, text: str, max_tokens: int | None = None) -> list[Chunk]:
        """
        Разбить текст по заголовкам Markdown.

        Сохраняет иерархию заголовков в метаданных:
        ["Main", "Section", "Subsection"]

        Не разбивает блоки кода.

        Args:
            text: Текст для разбиения.
            max_tokens: Максимальный размер чанка (опционально).

        Returns:
            Список чанков с метаданными.
        """
        if not text.strip():
            return []

        chunks: list[Chunk] = []
        current_headers: list[str] = []
        current_content: list[str] = []

        lines = text.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            # Проверяем заголовок
            header_match = self._header_pattern.match(line)

            if header_match:
                # Если есть накопленный контент, создаем чанк
                if current_content:
                    content = "\n".join(current_content).strip()
                    if content:
                        chunks.append(
                            Chunk(
                                content=content,
                                metadata={
                                    "header_path": current_headers.copy(),
                                    "header_level": len(current_headers),
                                },
                            )
                        )
                    current_content = []

                # Обновляем иерархию заголовков
                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                # Обрезаем текущий путь до уровня заголовка
                current_headers = current_headers[: level - 1]
                current_headers.append(title)

                # Добавляем сам заголовок в контент
                current_content.append(line)

            else:
                # Проверяем блок кода
                code_block_match = self._code_block_pattern.match(line)

                if code_block_match or line.startswith("```"):
                    # Собираем весь блок кода
                    code_lines = [line]
                    i += 1
                    while i < len(lines):
                        code_lines.append(lines[i])
                        if lines[i].strip().startswith("```") and len(code_lines) > 1:
                            break
                        i += 1

                    current_content.extend(code_lines)
                else:
                    current_content.append(line)

            i += 1

        # Добавляем последний чанк
        if current_content:
            content = "\n".join(current_content).strip()
            if content:
                chunks.append(
                    Chunk(
                        content=content,
                        metadata={
                            "header_path": current_headers.copy(),
                            "header_level": len(current_headers),
                        },
                    )
                )

        # Применяем ограничение по размеру если указано
        if max_tokens:
            chunks = self._apply_max_tokens(chunks, max_tokens)

        logger.debug(f"Structural chunking: создано {len(chunks)} чанков")
        return chunks

    def _apply_max_tokens(self, chunks: list[Chunk], max_tokens: int) -> list[Chunk]:
        """
        Применить ограничение максимального размера к чанкам.

        Args:
            chunks: Список чанков.
            max_tokens: Максимальный размер.

        Returns:
            Список чанков с ограниченным размером.
        """
        result: list[Chunk] = []

        for chunk in chunks:
            content = chunk.content
            if len(content) <= max_tokens:
                result.append(chunk)
            else:
                # Разбиваем большой чанк на меньшие
                sub_chunks = self.fixed_size_chunk(
                    content,
                    chunk_size=max_tokens,
                    overlap=self.chunk_overlap,
                )
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata.update(chunk.metadata)
                    result.append(sub_chunk)

        return result

    def fixed_size_chunk(self, text: str, chunk_size: int, overlap: int) -> list[Chunk]:
        """
        Разбить текст на чанки фиксированного размера с перекрытием.

        Использует посимвольное разбиение с попыткой сохранения границ предложений.

        Args:
            text: Текст для разбиения.
            chunk_size: Размер чанка в символах.
            overlap: Перекрытие между чанками в символах.

        Returns:
            Список чанков.
        """
        if not text.strip():
            return []

        if chunk_size <= 0:
            raise ValueError("chunk_size должен быть положительным числом")

        if overlap < 0:
            raise ValueError("overlap должен быть неотрицательным числом")

        if overlap >= chunk_size:
            raise ValueError("overlap должен быть меньше chunk_size")

        chunks: list[Chunk] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Определяем конец текущего чанка
            end = start + chunk_size

            # Если это не последний чанк, пытаемся разбить по границе предложения
            if end < text_length:
                # Ищем последнюю точку, вопросительный или восклицательный знак
                sentence_end = max(
                    text.rfind(".", start, end),
                    text.rfind("?", start, end),
                    text.rfind("!", start, end),
                    text.rfind("\n", start, end),
                )

                # Если найдена граница предложения, используем её
                if sentence_end > start:
                    end = sentence_end + 1

            # Добавляем чанк
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        metadata={
                            "start_pos": start,
                            "end_pos": end,
                            "chunk_type": "fixed_size",
                        },
                    )
                )

            # Перемещаемся к следующему чанку с учетом перекрытия
            start = end - overlap

            # Предотвращаем бесконечный цикл
            if start >= text_length and end >= text_length:
                break

            # Защита от зависания
            if start < 0:
                start = end

        logger.debug(f"Fixed-size chunking: создано {len(chunks)} чанков")
        return chunks

    def recursive_chunk(self, text: str, separators: list[str] | None = None) -> list[Chunk]:
        """
        Рекурсивно разбить текст по иерархии разделителей.

        Использует иерархическое разбиение: ["\n\n", "\n", ". ", " "]

        Args:
            text: Текст для разбиения.
            separators: Список разделителей в порядке приоритета.

        Returns:
            Список чанков.
        """
        if not text.strip():
            return []

        seps = separators or self.separators
        chunks: list[Chunk] = []

        def split_recursive(text_to_split: str, sep_index: int) -> list[str]:
            """Рекурсивно разбивает текст по разделителям."""
            if sep_index >= len(seps):
                # Достигли последнего разделителя, возвращаем как есть
                return [text_to_split] if text_to_split.strip() else []

            separator = seps[sep_index]
            parts = text_to_split.split(separator)

            result: list[str] = []
            for part in parts:
                part = part.strip()
                if not part:
                    continue

                # Если часть больше chunk_size, продолжаем делить
                if len(part) > self.chunk_size:
                    sub_parts = split_recursive(part, sep_index + 1)
                    result.extend(sub_parts)
                else:
                    result.append(part)

            return result

        split_parts = split_recursive(text, 0)

        for part in split_parts:
            chunks.append(
                Chunk(
                    content=part,
                    metadata={
                        "chunk_type": "recursive",
                        "separators_used": seps,
                    },
                )
            )

        logger.debug(f"Recursive chunking: создано {len(chunks)} чанков")
        return chunks

    def chunk(
        self,
        text: str,
        strategy: str = "structural",
        max_tokens: int | None = None,
    ) -> list[Chunk]:
        """
        Разбить текст используя указанную стратегию.

        Args:
            text: Текст для разбиения.
            strategy: Стратегия чанкования (structural, fixed, recursive).
            max_tokens: Максимальный размер чанка.

        Returns:
            Список чанков.

        Raises:
            ValueError: Если указана неверная стратегия.
        """
        if strategy == "structural":
            return self.structural_chunk(text, max_tokens or self.chunk_size)
        elif strategy == "fixed":
            return self.fixed_size_chunk(text, self.chunk_size, self.chunk_overlap)
        elif strategy == "recursive":
            return self.recursive_chunk(text)
        else:
            raise ValueError(
                f"Неизвестная стратегия чанкования: {strategy}. " f"Доступные: structural, fixed, recursive"
            )

    def count_tokens(self, text: str) -> int:
        """
        Подсчитать количество токенов в тексте.

        Простая эвристика: средний русский токен ~4 символа.

        Args:
            text: Текст для подсчета.

        Returns:
            Приблизительное количество токенов.
        """
        # Для русского языка ~4 символа на токен
        # Для английского ~4-5 символов на токен
        return max(1, len(text) // 4)

    def add_metadata_to_chunks(self, chunks: list[Chunk], metadata: dict[str, Any]) -> None:
        """
        Добавить общие метаданные ко всем чанкам.

        Args:
            chunks: Список чанков.
            metadata: Метаданные для добавления.
        """
        for chunk in chunks:
            chunk.metadata.update(metadata)
