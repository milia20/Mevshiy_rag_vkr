"""
Тесты для вспомогательных функций.

Модуль содержит тесты для утилит обработки текста.
"""

from __future__ import annotations

import pytest
from mkdocs_plugin.utils.helpers import (
    clean_markdown_links,
    extract_text_from_html,
    generate_chunk_id,
    iterate_paragraphs,
    normalize_text,
    split_text_into_chunks,
)


class TestExtractTextFromHTML:
    """Тесты для функции extract_text_from_html."""

    def test_basic_html(self) -> None:
        """Тест извлечения текста из простого HTML."""
        html = "<p>Привет мир!</p>"
        result = extract_text_from_html(html)
        assert result == "Привет мир!"

    def test_nested_tags(self) -> None:
        """Тест извлечения текста из HTML с вложенными тегами."""
        html = "<div><p>Текст с <b>жирным</b> и <i>курсивным</i></p></div>"
        result = extract_text_from_html(html)
        assert "Текст с" in result
        assert "жирным" in result
        assert "курсивным" in result

    def test_script_removal(self) -> None:
        """Тест удаления скриптов."""
        html = "<p>Текст</p><script>alert('xss');</script><p>еще текст</p>"
        result = extract_text_from_html(html)
        assert "alert" not in result
        assert "Текст" in result
        assert "еще текст" in result

    def test_style_removal(self) -> None:
        """Тест удаления стилей."""
        html = "<style>.hidden { display: none; }</style><p>Контент</p>"
        result = extract_text_from_html(html)
        assert "display" not in result
        assert "Контент" in result

    def test_html_entities(self) -> None:
        """Тест декодирования HTML-сущностей."""
        html = "<p>&lt;script&gt; &amp; &quot;test&quot;</p>"
        result = extract_text_from_html(html)
        assert "<script>" in result
        assert "&" in result
        assert '"test"' in result

    def test_empty_html(self) -> None:
        """Тест пустого HTML."""
        html = ""
        result = extract_text_from_html(html)
        assert result == ""

    def test_whitespace_normalization(self) -> None:
        """Тест нормализации пробелов."""
        html = "<p>Текст    с     множественными   пробелами</p>"
        result = extract_text_from_html(html)
        assert result == "Текст с множественными пробелами"


class TestNormalizeText:
    """Тесты для функции normalize_text."""

    def test_lowercase_conversion(self) -> None:
        """Тест приведения к нижнему регистру."""
        text = "ПРИВЕТ МИР"
        result = normalize_text(text)
        assert result == "привет мир"

    def test_whitespace_stripping(self) -> None:
        """Тест удаления лишних пробелов."""
        text = "  текст   с   пробелами  "
        result = normalize_text(text)
        assert result == "текст с пробелами"

    def test_special_chars_removal(self) -> None:
        """Тест удаления специальных символов."""
        text = "Текст@#$% со спецсимволами!&*()"
        result = normalize_text(text)
        assert "@" not in result
        assert "#" not in result
        assert "со" in result

    def test_punctuation_preservation(self) -> None:
        """Тест сохранения пунктуации."""
        text = "Привет, мир! Как дела?"
        result = normalize_text(text)
        assert "," in result
        assert "!" in result
        assert "?" in result

    def test_empty_string(self) -> None:
        """Тест пустой строки."""
        text = ""
        result = normalize_text(text)
        assert result == ""


class TestSplitTextIntoChunks:
    """Тесты для функции split_text_into_chunks."""

    def test_small_text(self) -> None:
        """Тест разбиения небольшого текста."""
        text = "Одно предложение."
        chunks = split_text_into_chunks(text, chunk_size=100, chunk_overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == "Одно предложение."

    def test_large_text(self) -> None:
        """Тест разбиения большого текста."""
        text = ". ".join([f"Предложение {i}" for i in range(20)]) + "."
        chunks = split_text_into_chunks(text, chunk_size=50, chunk_overlap=10)
        assert len(chunks) > 1
        # Проверяем что все чанки не превышают размер
        for chunk in chunks:
            assert len(chunk) <= 55  # Небольшой допуск

    def test_chunk_overlap(self) -> None:
        """Тест перекрытия чанков."""
        text = "A. B. C. D. E."
        chunks = split_text_into_chunks(text, chunk_size=10, chunk_overlap=5)
        assert len(chunks) > 1

    def test_invalid_chunk_size(self) -> None:
        """Тест некорректного размера чанка."""
        text = "Тест"
        with pytest.raises(ValueError):
            split_text_into_chunks(text, chunk_size=0)

        with pytest.raises(ValueError):
            split_text_into_chunks(text, chunk_size=-10)

    def test_invalid_overlap(self) -> None:
        """Тест некорректного перекрытия."""
        text = "Тест"
        with pytest.raises(ValueError):
            split_text_into_chunks(text, chunk_size=10, chunk_overlap=10)

        with pytest.raises(ValueError):
            split_text_into_chunks(text, chunk_size=10, chunk_overlap=15)

    def test_empty_text(self) -> None:
        """Тест пустого текста."""
        text = ""
        chunks = split_text_into_chunks(text, chunk_size=10)
        assert len(chunks) == 0

    def test_sentence_boundaries(self) -> None:
        """Тест разбиения по границам предложений."""
        text = "Первое предложение. Второе предложение. Третье предложение."
        chunks = split_text_into_chunks(text, chunk_size=25, chunk_overlap=5)
        # Проверяем что чанки заканчиваются на предложениях
        for chunk in chunks:
            if chunk != chunks[-1]:  # Кроме последнего
                assert chunk.endswith(".") or chunk.endswith("!") or chunk.endswith("?")


class TestCleanMarkdownLinks:
    """Тесты для функции clean_markdown_links."""

    def test_inline_link(self) -> None:
        """Тест очистки inline ссылок."""
        text = "[Google](https://google.com)"
        result = clean_markdown_links(text)
        assert result == "Google"

    def test_reference_link(self) -> None:
        """Тест очистки reference ссылок."""
        text = "[Example][ref]"
        result = clean_markdown_links(text)
        assert result == "Example"

    def test_multiple_links(self) -> None:
        """Тест множественных ссылок."""
        text = "[Link1](url1) and [Link2](url2)"
        result = clean_markdown_links(text)
        assert result == "Link1 and Link2"

    def test_no_links(self) -> None:
        """Тест текста без ссылок."""
        text = "Просто текст без ссылок"
        result = clean_markdown_links(text)
        assert result == text


class TestGenerateChunkId:
    """Тесты для функции generate_chunk_id."""

    def test_id_generation(self) -> None:
        """Тест генерации идентификатора."""
        chunk_id = generate_chunk_id("doc1", 0, "test text")
        assert chunk_id.startswith("doc1_chunk0_")
        assert len(chunk_id) == 24  # prefix + _chunk + index + _ + hash(8)

    def test_unique_ids(self) -> None:
        """Тест уникальности идентификаторов."""
        id1 = generate_chunk_id("doc", 0, "text1")
        id2 = generate_chunk_id("doc", 0, "text2")
        assert id1 != id2

    def test_same_text_same_id(self) -> None:
        """Тест одинаковых идентификаторов для одинакового текста."""
        id1 = generate_chunk_id("doc", 0, "same text")
        id2 = generate_chunk_id("doc", 0, "same text")
        assert id1 == id2


class TestIterateParagraphs:
    """Тесты для функции iterate_paragraphs."""

    def test_single_paragraph(self) -> None:
        """Тест одного абзаца."""
        text = "Один абзац"
        paragraphs = list(iterate_paragraphs(text))
        assert len(paragraphs) == 1
        assert paragraphs[0] == "Один абзац"

    def test_multiple_paragraphs(self) -> None:
        """Тест нескольких абзацев."""
        text = "Абзац 1\n\nАбзац 2\n\nАбзац 3"
        paragraphs = list(iterate_paragraphs(text))
        assert len(paragraphs) == 3

    def test_empty_paragraphs_skipped(self) -> None:
        """Тест пропуска пустых абзацев."""
        text = "Текст\n\n\n\nЕще текст"
        paragraphs = list(iterate_paragraphs(text))
        assert len(paragraphs) == 2

    def test_whitespace_handling(self) -> None:
        """Тест обработки пробелов."""
        text = "  Текст с пробелами  \n\n  Еще текст  "
        paragraphs = list(iterate_paragraphs(text))
        assert paragraphs[0] == "Текст с пробелами"
        assert paragraphs[1] == "Еще текст"
