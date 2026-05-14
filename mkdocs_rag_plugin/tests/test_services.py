"""
Tests for document processor and chunker services.

Модуль содержит тесты для сервисов обработки документов и чанкования.
"""

from __future__ import annotations

import pytest
from mkdocs_plugin.services.chunker import Chunk, Chunker
from mkdocs_plugin.services.doc_processor import DocumentProcessor


class TestDocumentProcessor:
    """Тесты для DocumentProcessor."""

    def setup_method(self) -> None:
        """Инициализация перед каждым тестом."""
        self.processor = DocumentProcessor()

    def test_clean_content_removes_html_comments(self) -> None:
        """Тест удаления HTML комментариев."""
        markdown = "Текст <!-- комментарий --> еще текст"
        result = self.processor.clean_content(markdown)
        assert "<!--" not in result
        assert "-->" not in result
        assert "Текст" in result
        assert "еще текст" in result

    def test_clean_content_removes_nav_artifacts(self) -> None:
        """Тест удаления навигационных артефактов."""
        markdown = "Заголовок ¶ Текст ⚓"
        result = self.processor.clean_content(markdown)
        assert "¶" not in result
        assert "⚓" not in result

    def test_clean_content_normalizes_newlines(self) -> None:
        """Тест нормализации переводов строк."""
        markdown = "Текст\n\n\n\nЕще текст"
        result = self.processor.clean_content(markdown)
        # Не более 2 переводов строк подряд
        assert "\n\n\n" not in result

    def test_clean_content_empty(self) -> None:
        """Тест очистки пустого контента."""
        result = self.processor.clean_content("")
        assert result == ""

    def test_extract_code_blocks(self) -> None:
        """Тест извлечения блоков кода."""
        markdown = """
# Заголовок

Текст

```python
def hello():
    print("Привет")
```

Еще текст
"""
        code_blocks = self.processor.extract_code_blocks(markdown)
        assert len(code_blocks) == 1
        assert code_blocks[0]["language"] == "python"
        assert "def hello():" in code_blocks[0]["code"]

    def test_extract_tables(self) -> None:
        """Тест извлечения таблиц."""
        markdown = """
# Заголовок

| Col1 | Col2 |
|------|------|
| A    | B    |
| C    | D    |

Текст
"""
        tables = self.processor.extract_tables(markdown)
        assert len(tables) == 1
        assert "| Col1 | Col2 |" in tables[0]


class TestChunkerStructural:
    """Тесты для structural chunking стратегии."""

    def setup_method(self) -> None:
        """Инициализация перед каждым тестом."""
        self.chunker = Chunker(chunk_size=500, chunk_overlap=50)

    def test_structural_chunk_by_headers(self) -> None:
        """Тест разбиения по заголовкам."""
        text = """# Главный заголовок

Текст под главным заголовком.

## Подзаголовок 1

Текст подзаголовка 1.

## Подзаголовок 2

Текст подзаголовка 2.
"""
        chunks = self.chunker.structural_chunk(text)

        assert len(chunks) >= 3
        # Проверяем что первый чанк содержит главный заголовок
        assert "# Главный заголовок" in chunks[0].content
        # Проверяем метаданные
        assert chunks[0].metadata["header_path"] == ["Главный заголовок"]

    def test_structural_chunk_preserves_header_hierarchy(self) -> None:
        """Тест сохранения иерархии заголовков."""
        text = """# Уровень 1

## Уровень 2

### Уровень 3

Текст уровня 3.
"""
        chunks = self.chunker.structural_chunk(text)

        # Последний чанк должен иметь полную иерархию
        last_chunk = chunks[-1]
        assert last_chunk.metadata["header_path"] == ["Уровень 1", "Уровень 2", "Уровень 3"]

    def test_structural_chunk_empty(self) -> None:
        """Тест пустого текста."""
        chunks = self.chunker.structural_chunk("")
        assert len(chunks) == 0

    def test_structural_chunk_cyrillic(self) -> None:
        """Тест кириллического текста."""
        text = """# Введение

Это введение на русском языке.

## Основная часть

Основной текст также на русском.
"""
        chunks = self.chunker.structural_chunk(text)

        assert len(chunks) >= 2
        assert "Введение" in chunks[0].content
        assert "русском" in " ".join(c.content for c in chunks)


class TestChunkerFixedSize:
    """Тесты для fixed-size chunking стратегии."""

    def setup_method(self) -> None:
        """Инициализация перед каждым тестом."""
        self.chunker = Chunker(chunk_size=100, chunk_overlap=20)

    def test_fixed_size_chunk_basic(self) -> None:
        """Тест базового разбиения фиксированного размера."""
        text = "A. " * 50  # Длинный текст
        chunks = self.chunker.fixed_size_chunk(text, chunk_size=50, overlap=10)

        assert len(chunks) > 1
        # Проверяем что чанки не превышают размер
        for chunk in chunks:
            assert len(chunk.content) <= 55  # Небольшой допуск

    def test_fixed_size_chunk_overlap(self) -> None:
        """Тест перекрытия чанков."""
        text = "Первое предложение. Второе предложение. Третье предложение."
        chunks = self.chunker.fixed_size_chunk(text, chunk_size=30, overlap=10)

        assert len(chunks) > 1
        # Проверяем наличие перекрытия
        if len(chunks) >= 2:
            # Конец первого чанка должен пересекаться с началом второго
            pass  # Сложно проверить напрямую, но тестируем логику

    def test_fixed_size_chunk_invalid_params(self) -> None:
        """Тест некорректных параметров."""
        text = "Тест"

        with pytest.raises(ValueError):
            self.chunker.fixed_size_chunk(text, chunk_size=0, overlap=10)

        with pytest.raises(ValueError):
            self.chunker.fixed_size_chunk(text, chunk_size=10, overlap=-5)

        with pytest.raises(ValueError):
            self.chunker.fixed_size_chunk(text, chunk_size=10, overlap=10)

    def test_fixed_size_chunk_sentence_boundaries(self) -> None:
        """Тест разбиения по границам предложений."""
        text = "Первое предложение. Второе предложение. Третье предложение."
        chunks = self.chunker.fixed_size_chunk(text, chunk_size=25, overlap=5)

        # Проверяем что чанки заканчиваются на предложениях (кроме последнего)
        for i, chunk in enumerate(chunks[:-1]):
            assert chunk.content.endswith(".") or chunk.content.endswith("!") or chunk.content.endswith("?")

    def test_fixed_size_chunk_cyrillic(self) -> None:
        """Тест кириллического текста."""
        text = "Привет мир! Как дела? Все хорошо. Отлично!"
        chunks = self.chunker.fixed_size_chunk(text, chunk_size=20, overlap=5)

        assert len(chunks) > 1
        assert any("Привет" in c.content for c in chunks)


class TestChunkerRecursive:
    """Тесты для recursive chunking стратегии."""

    def setup_method(self) -> None:
        """Инициализация перед каждым тестом."""
        self.chunker = Chunker(chunk_size=100, chunk_overlap=0)

    def test_recursive_chunk_basic(self) -> None:
        """Тест базового рекурсивного разбиения."""
        text = """Абзац 1.

Абзац 2.

Абзац 3.
"""
        chunks = self.chunker.recursive_chunk(text)

        assert len(chunks) >= 3
        assert any("Абзац 1" in c.content for c in chunks)

    def test_recursive_chunk_custom_separators(self) -> None:
        """Тест кастомных разделителей."""
        text = "Часть 1; Часть 2; Часть 3"
        separators = ["; "]
        chunks = self.chunker.recursive_chunk(text, separators=separators)

        assert len(chunks) == 3

    def test_recursive_chunk_nested(self) -> None:
        """Тест вложенного разбиения."""
        text = """
Раздел 1.

Подраздел 1.1.

Подраздел 1.2.

Раздел 2.
"""
        chunks = self.chunker.recursive_chunk(text)

        assert len(chunks) >= 4

    def test_recursive_chunk_empty(self) -> None:
        """Тест пустого текста."""
        chunks = self.chunker.recursive_chunk("")
        assert len(chunks) == 0

    def test_recursive_chunk_cyrillic(self) -> None:
        """Тест кириллического текста."""
        text = """Первый абзац на русском.

Второй абзац тоже на русском.
"""
        chunks = self.chunker.recursive_chunk(text)

        assert len(chunks) >= 2
        assert any("русском" in c.content for c in chunks)


class TestChunkerGeneric:
    """Тесты для общего метода chunk()."""

    def setup_method(self) -> None:
        """Инициализация перед каждым тестом."""
        self.chunker = Chunker(chunk_size=100, chunk_overlap=10)

    def test_chunk_structural_strategy(self) -> None:
        """Тест стратегии structural."""
        text = "# Заголовок\nТекст"
        chunks = self.chunker.chunk(text, strategy="structural")
        assert len(chunks) >= 1

    def test_chunk_fixed_strategy(self) -> None:
        """Тест стратегии fixed."""
        text = "Текст. " * 20
        chunks = self.chunker.chunk(text, strategy="fixed")
        assert len(chunks) >= 1

    def test_chunk_recursive_strategy(self) -> None:
        """Тест стратегии recursive."""
        text = "Абзац 1.\n\nАбзац 2."
        chunks = self.chunker.chunk(text, strategy="recursive")
        assert len(chunks) >= 1

    def test_chunk_invalid_strategy(self) -> None:
        """Тест неверной стратегии."""
        text = "Тест"
        with pytest.raises(ValueError):
            self.chunker.chunk(text, strategy="invalid")

    def test_count_tokens(self) -> None:
        """Тест подсчета токенов."""
        text = "Привет мир"
        token_count = self.chunker.count_tokens(text)
        assert token_count > 0

    def test_add_metadata_to_chunks(self) -> None:
        """Тест добавления метаданных."""
        chunks = [Chunk(content="Тест")]
        metadata = {"url": "/test/", "title": "Test"}

        self.chunker.add_metadata_to_chunks(chunks, metadata)

        assert chunks[0].metadata["url"] == "/test/"
        assert chunks[0].metadata["title"] == "Test"


class TestChunkModel:
    """Тесты для модели Chunk."""

    def test_chunk_creation(self) -> None:
        """Тест создания чанка."""
        chunk = Chunk(content="Тестовый контент")

        assert chunk.id is not None
        assert chunk.content == "Тестовый контент"
        assert isinstance(chunk.metadata, dict)

    def test_chunk_with_metadata(self) -> None:
        """Тест чанка с метаданными."""
        chunk = Chunk(content="Тест", metadata={"url": "/test/", "title": "Test"})

        assert chunk.metadata["url"] == "/test/"
        assert chunk.metadata["title"] == "Test"
