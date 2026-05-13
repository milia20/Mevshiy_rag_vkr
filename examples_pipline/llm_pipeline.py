import difflib

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers import pipeline as hf_pipeline

nltk.download("punkt")
nltk.download("punkt_tab")


class LLMPipeline:
    """
    Класс для работы с LLM в задачах Retrieval-Augmented Generation (RAG).

    Позволяет:
      - Генерировать ответ на основе retrieval контекста и запроса.
      - Генерировать ответ "zero-shot" (без контекста).
      - Оценивать качество сгенерированных ответов с использованием нескольких метрик:
            SequenceMatcher similarity, BLEU, ROUGE-L.
    """

    def __init__(self, model_name, max_length=200, do_sample=False, pipeline_type="text2text-generation"):
        """
        Инициализация объекта.

        :param model_name: Имя модели для генерации ответов (например, "google/flan-t5-base").
        :param max_length: Максимальная длина генерируемого ответа.
        :param do_sample: Использовать ли сэмплинг при генерации.
        """
        self.model_pipeline = hf_pipeline(pipeline_type, model=model_name, tokenizer=model_name)
        self.max_length = max_length
        self.do_sample = do_sample

    def generate_answer(self, query, context):
        """
        Формирует prompt с retrieval контекстом и генерирует ответ.

        :param query: Текст запроса.
        :param context: Текст retrieval контекста (например, найденный чанк).
        :return: Сгенерированный ответ (строка).
        """
        prompt = f"""Use the following context extracted from the document to answer the question.
If the answer is not contained in the context, say "I don't know."

Context:
{context}

Question:
{query}

Answer:"""
        result = self.model_pipeline(prompt, max_length=self.max_length, do_sample=self.do_sample)
        return result[0]["generated_text"].strip()

    def generate_answer_no_context(self, query):
        """
        Формирует prompt без retrieval контекста и генерирует ответ (zero-shot).

        :param query: Текст запроса.
        :return: Сгенерированный ответ (строка).
        """
        prompt = f"""Answer the following question:
{query}

Answer:"""
        result = self.model_pipeline(prompt, max_length=self.max_length, do_sample=self.do_sample)
        return result[0]["generated_text"].strip()

    def compute_bleu(self, reference, candidate):
        """
        Вычисляет BLEU score между reference и candidate.
        """
        ref_tokens = nltk.word_tokenize(reference)
        cand_tokens = nltk.word_tokenize(candidate)
        smoothing = SmoothingFunction().method1
        bleu = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)
        return bleu

    def compute_rouge(self, reference, candidate):
        """
        Вычисляет ROUGE-L F-measure между reference и candidate.
        """
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        return scores["rougeL"].fmeasure

    def evaluate(self, queries, true_answers, retrieval_function, expected_pages, top_k=1):
        """
        Выполняет оценку LLM с retrieval контекстом.

        Для каждого запроса:
          - Извлекается retrieval контекст с помощью retrieval_function.
          - Формируется prompt и генерируется ответ.
          - Вычисляются метрики: SequenceMatcher similarity, BLEU, ROUGE-L.
          - Выводится номер страницы retrieval контекста для проверки.

        :param queries: Список текстовых запросов.
        :param true_answers: Список истинных ответов.
        :param retrieval_function: Функция, которая по запросу возвращает retrieval результат (список словарей с ключами "page" и "chunk").
        :param expected_pages: Список ожидаемых номеров страниц для каждого запроса.
        :param top_k: Количество кандидатов retrieval (по умолчанию 1).
        :return: Словарь со средними значениями метрик.
        """
        total_similarity = 0
        total_bleu = 0
        total_rouge = 0
        total = len(queries)

        for idx, query in enumerate(queries):
            retrieval_result = retrieval_function(query, top_k=top_k)[0]
            retrieved_page = retrieval_result.get("page", "Unknown")
            context = retrieval_result.get("chunk", "")

            generated_answer = self.generate_answer(query, context)
            true_answer = true_answers[idx].strip()

            sim = difflib.SequenceMatcher(None, generated_answer, true_answer).ratio()
            bleu = self.compute_bleu(true_answer, generated_answer)
            rouge = self.compute_rouge(true_answer, generated_answer)

            total_similarity += sim
            total_bleu += bleu
            total_rouge += rouge

            print(f"\nEvaluation with Context - Вопрос {idx+1}: {query}")
            print("Ожидаемая страница:", expected_pages[idx])
            print("Retrieval контекст со страницы:", retrieved_page)
            print("Сгенерированный ответ:")
            print(generated_answer)
            print("Истинный ответ:")
            print(true_answer)
            print(f"SequenceMatcher Similarity: {sim * 100:.2f}%")
            print(f"BLEU score: {bleu * 100:.2f}%")
            print(f"ROUGE-L F1: {rouge * 100:.2f}%")

        avg_sim = total_similarity / total if total > 0 else 0
        avg_bleu = total_bleu / total if total > 0 else 0
        avg_rouge = total_rouge / total if total > 0 else 0

        print("\nСредние метрики с Context:")
        print(f"Средняя SequenceMatcher Similarity: {avg_sim * 100:.2f}%")
        print(f"Средний BLEU score: {avg_bleu * 100:.2f}%")
        print(f"Средний ROUGE-L F1: {avg_rouge * 100:.2f}%")

        return {"avg_similarity": avg_sim, "avg_bleu": avg_bleu, "avg_rouge": avg_rouge}

    def evaluate_no_context(self, queries, true_answers):
        """
        Выполняет оценку LLM в режиме zero-shot (без retrieval контекста).

        Для каждого запроса:
          - Генерируется ответ без предоставления retrieval контекста.
          - Вычисляются метрики: SequenceMatcher similarity, BLEU, ROUGE-L.

        :param queries: Список текстовых запросов.
        :param true_answers: Список истинных ответов.
        :return: Словарь со средними значениями метрик.
        """
        total_similarity = 0
        total_bleu = 0
        total_rouge = 0
        total = len(queries)

        for idx, query in enumerate(queries):
            generated_answer = self.generate_answer_no_context(query)
            true_answer = true_answers[idx].strip()

            sim = difflib.SequenceMatcher(None, generated_answer, true_answer).ratio()
            bleu = self.compute_bleu(true_answer, generated_answer)
            rouge = self.compute_rouge(true_answer, generated_answer)

            total_similarity += sim
            total_bleu += bleu
            total_rouge += rouge

            print(f"\nEvaluation without Context - Вопрос {idx+1}: {query}")
            print("Сгенерированный zero-shot ответ:")
            print(generated_answer)
            print("Истинный ответ:")
            print(true_answer)
            print(f"SequenceMatcher Similarity: {sim * 100:.2f}%")
            print(f"BLEU score: {bleu * 100:.2f}%")
            print(f"ROUGE-L F1: {rouge * 100:.2f}%")

        avg_sim = total_similarity / total if total > 0 else 0
        avg_bleu = total_bleu / total if total > 0 else 0
        avg_rouge = total_rouge / total if total > 0 else 0

        print("\nСредние метрики без Context:")
        print(f"Средняя SequenceMatcher Similarity: {avg_sim * 100:.2f}%")
        print(f"Средний BLEU score: {avg_bleu * 100:.2f}%")
        print(f"Средний ROUGE-L F1: {avg_rouge * 100:.2f}%")

        return {"avg_similarity": avg_sim, "avg_bleu": avg_bleu, "avg_rouge": avg_rouge}
