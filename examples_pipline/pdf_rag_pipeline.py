import pickle
import re

import pdfplumber
import PyPDF2
from pdfminer.high_level import extract_text
from rank_bm25 import BM25Okapi


class PDFRAGPipeline:
    """
    Класс для обработки PDF-файлов для задач Retrieval-Augmented Generation (RAG).
    Основные возможности:
      - Извлечение текста из PDF с сохранением номера страницы.
      - Разбиение каждой страницы на чанки (с перекрытием) по количеству слов.
      - Построение BM25 индекса для retrieval.
      - Поиск по запросу с возвратом найденного чанка и номера страницы.
    """

    def __init__(self, pdf_path, chunk_size=1000, overlap=50):
        """
        Инициализация объекта.

        :param pdf_path: Путь к PDF-файлу.
        :param chunk_size: Количество слов в одном чанке (по умолчанию 1000).
        :param overlap: Количество слов, перекрывающихся между чанками (по умолчанию 50).
        """
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.pages = None
        self.chunks = None
        self.bm25 = None

    def extract_pages(self):
        """
        Извлекает текст из PDF-файла с сохранением номера страницы.
        :return: Список словарей: {"page": номер, "text": текст страницы}
        """
        pages = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    pages.append({"page": i + 1, "text": text.strip()})
        self.pages = pages
        return pages

    def extract_pages_pdfminer(self):
        """
        Альтернативный метод извлечения текста из PDF-файла с помощью pdfminer.six.
        Разделяет текст по символу перевода страницы ('\f') и возвращает список словарей:
        {"page": номер страницы, "text": текст страницы}.

        :param pdf_path: Если указан, используется вместо self.pdf_path.
        :return: Список словарей с извлечённым текстом.
        """
        full_text = extract_text(self.pdf_path)
        # Разделяем текст по символу перевода страницы
        raw_pages = re.split(r"\f+", full_text)
        pages = []
        for i, text in enumerate(raw_pages):
            text = text.strip()
            if text:
                pages.append({"page": i + 1, "text": text})
        self.pages = pages
        return pages

    def extract_pages_pypdf2(self):
        """
        Извлекает текст из PDF-файла с помощью PyPDF2 и возвращает список словарей:
        {"page": номер страницы, "text": текст страницы}.
        """
        pages = []
        with open(self.pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages.append({"page": i + 1, "text": text.strip()})
        self.pages = pages
        return pages

    def split_pages(self):
        """
        Разбивает каждую страницу на чанки фиксированного размера (по словам) с перекрытием.
        Если на странице текста меньше, чем chunk_size, весь текст будет в одном чанке.

        :return: Список словарей: {"page": номер страницы, "chunk": текст чанка}
        """
        if self.pages is None:
            self.extract_pages()
        all_chunks = []
        for page in self.pages:
            words = page["text"].split()
            start = 0
            while start < len(words):
                chunk_words = words[start : start + self.chunk_size]
                chunk_text = " ".join(chunk_words)
                if len(chunk_text) > 20:  # отсекаем очень короткие чанки
                    all_chunks.append({"page": page["page"], "chunk": chunk_text})
                start += self.chunk_size - self.overlap
        self.chunks = all_chunks
        return all_chunks

    def build_index(self):
        """
        Строит BM25 индекс по токенизированным чанкам.
        :return: Объект BM25Okapi.
        """
        if self.chunks is None:
            self.split_pages()
        tokenized_chunks = [chunk["chunk"].split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        return self.bm25

    def search_query(self, query, top_k=1):
        """
        Выполняет поиск по запросу с использованием BM25 индекса.
        :param query: Текст запроса.
        :param top_k: Количество возвращаемых результатов (по умолчанию 1).
        :return: Список словарей с найденными чанками, содержащими ключ "page" (номер страницы) и "chunk" (текст).
        """
        if self.bm25 is None:
            self.build_index()
        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = [self.chunks[i] for i in top_indices]
        return results

    def save_chunks(self, chunks_path):
        """
        Сохраняет список чанков в указанный файл (pickle).
        :param chunks_path: Путь для сохранения чанков.
        """
        if self.chunks is None:
            self.split_pages()
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"Чанки сохранены в {chunks_path}")

    def load_chunks(self, chunks_path):
        """
        Загружает список чанков из указанного файла (pickle) и сохраняет в self.chunks.
        :param chunks_path: Путь к файлу с сохранёнными чанками.
        """
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        print(f"Чанки загружены из {chunks_path}")
        return self.chunks
