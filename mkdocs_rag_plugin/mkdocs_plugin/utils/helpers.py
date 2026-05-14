"""
Вспомогательные функции для обработки текста и HTML.

Модуль содержит утилиты для извлечения текста из HTML,
нормализации текста и разбиения на чанки.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from html import unescape


def extract_text_from_html(html: str) -> str:
    """
    Извлечь текст из HTML-строки.

    Удаляет все HTML-теги, скрипты, стили и декодирует HTML-сущности.

    Args:
        html: HTML-строка для обработки.

    Returns:
        Очищенный текст без HTML-тегов.

    Example:
        >>> html = "<p>Привет <b>мир</b>!</p>"
        >>> extract_text_from_html(html)
        'Привет мир!'
    """
    # Удаляем содержимое тегов script и style
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Удаляем HTML-теги
    text = re.sub(r"<[^>]+>", " ", text)

    # Декодируем HTML-сущности
    text = unescape(text)

    # Нормализуем пробелы
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_text(text: str) -> str:
    """
    Нормализовать текст для индексации.

    Выполняет следующие преобразования:
    - Приведение к нижнему регистру
    - Удаление лишних пробелов
    - Удаление специальных символов (кроме базовой пунктуации)
    - Замена множественных пробелов на одиночные

    Args:
        text: Текст для нормализации.

    Returns:
        Нормализованный текст.

    Example:
        >>> normalize_text("  Привет   МИР!  ")
        'привет мир!'
    """
    # Приводим к нижнему регистру
    text = text.lower()

    # Удаляем лишние пробелы по краям
    text = text.strip()

    # Заменяем множественные пробелы на одиночные
    text = re.sub(r"\s+", " ", text)

    # Удаляем специальные символы, оставляя буквы, цифры и базовую пунктуацию
    text = re.sub(r"[^\w\s.,!?;:()\-\"]", "", text)

    return text


def split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """
    Разбить текст на перекрывающиеся чанки.

    Разбивает текст на части заданного размера с перекрытием для сохранения
    контекста между чанками. Границы чанков стараются проходить по предложениям.

    Args:
        text: Текст для разбиения.
        chunk_size: Максимальный размер чанка в символах.
        chunk_overlap: Размер перекрытия между чанками в символах.

    Returns:
        Список текстовых чанков.

    Raises:
        ValueError: Если chunk_overlap >= chunk_size или chunk_size <= 0.

    Example:
        >>> text = "Первое предложение. Второе предложение. Третье предложение."
        >>> chunks = split_text_into_chunks(text, chunk_size=30, chunk_overlap=10)
        >>> len(chunks) > 0
        True
    """
    if chunk_size <= 0:
        msg = f"chunk_size должен быть положительным числом, получено {chunk_size}"
        raise ValueError(msg)

    if chunk_overlap >= chunk_size:
        msg = f"chunk_overlap ({chunk_overlap}) должен быть меньше chunk_size " f"({chunk_size})"
        raise ValueError(msg)

    if not text:
        return []

    chunks: list[str] = []
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
            )

            # Если найдена граница предложения, используем её
            if sentence_end > start:
                end = sentence_end + 1

        # Добавляем чанк
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Перемещаемся к следующему чанку с учетом перекрытия
        start = end - chunk_overlap

        # Предотвращаем бесконечный цикл, если чанк не прогрессировал
        if start >= text_length and end >= text_length:
            break

    return chunks


def generate_chunk_id(prefix: str, index: int, text: str) -> str:
    """
    Сгенерировать уникальный идентификатор для чанка.

    Args:
        prefix: Префикс для идентификатора (например, имя файла).
        index: Индекс чанка.
        text: Текст чанка для создания хеша.

    Returns:
        Уникальный идентификатор чанка.
    """
    import hashlib

    # Создаем хеш от текста чанка
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]

    return f"{prefix}_chunk{index}_{text_hash}"


def clean_markdown_links(text: str) -> str:
    """
    Очистить текст от Markdown-ссылок.

    Преобразует ссылки вида [текст](url) в просто текст.

    Args:
        text: Текст с Markdown-разметкой.

    Returns:
        Текст без Markdown-ссылок.

    Example:
        >>> clean_markdown_links("[Google](https://google.com)")
        'Google'
    """
    # Заменяем [текст](url) на текст
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Заменяем [текст][ref] на текст
    text = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", text)

    return text


def iterate_paragraphs(text: str) -> Iterator[str]:
    """
    Итерировать по абзацам текста.

    Разбивает текст на абзацы по двойным переводам строки.

    Args:
        text: Исходный текст.

    Yields:
        Отдельные абзацы текста.
    """
    # Разбиваем по двойным переводам строки
    paragraphs = re.split(r"\n\s*\n", text)

    for paragraph in paragraphs:
        cleaned = paragraph.strip()
        if cleaned:
            yield cleaned
