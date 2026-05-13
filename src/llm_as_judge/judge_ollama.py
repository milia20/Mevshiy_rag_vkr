"""
judge_ollama.py
LLM-as-a-Judge пайплайн оценки качества ответов через Ollama API.

Поддерживаемые модели-судьи:
- openai/gpt-oss-120b
- google/gemma-4-26b-a4b
- qwen/qwen3.6-35b-a3b

Метрики оценки (шкала 1-4):
- Faithfulness (Верность контексту)
- Accuracy (Точность ответа)
- Relevance (Релевантность вопросу)
"""

import csv
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ==================== КОНФИГУРАЦИЯ ====================

# Ollama API endpoint
OLLAMA_BASE_URL = "http://localhost:11434"

# Модели-судьи
JUDGE_MODELS = [
    {"name": "gpt-oss-120b", "model_id": "openai/gpt-oss-120b"},
    {"name": "gemma-4-26b", "model_id": "google/gemma-4-26b-a4b"},
    {"name": "qwen3.6-35b", "model_id": "qwen/qwen3.6-35b-a3b"},
]

# Файлы данных
EVALUATION_RESULTS_FILE = "/workspace/test_datasets/evaluation_results_all.csv"
CONTEXT_FILE = "/workspace/src/all_q_True.csv"
OUTPUT_DIR = Path("/workspace/test_datasets/judge_results_ollama")
OUTPUT_FILE = OUTPUT_DIR / "judge_evaluation_results.csv"
DETAILED_OUTPUT_FILE = OUTPUT_DIR / "judge_detailed_results.json"

# Параметры оценки
MAX_SAMPLES = None  # None = все записи, иначе ограничить количество
TEMPERATURE = 0.1  # Низкая температура для консистентности оценок
MAX_TOKENS = 256  # Ограничение на генерацию (достаточно для оценок 1-4 + обоснование)
TIMEOUT_SECONDS = 120


# ==================== МОДЕЛИ ДАННЫХ ====================


class OllamaOutputSchema:
    """
    Схема для структурированного ответа судьи в Ollama.
    Ollama поддерживает формат JSON Schema через parameter 'format'.
    """

    schema = {
        "type": "object",
        "properties": {
            "evaluation": {"type": "string", "description": "Обоснование оценки (2-4 предложения)"},
            "faithfulness": {"type": "integer", "minimum": 1, "maximum": 4, "description": "Оценка верности контексту"},
            "accuracy": {"type": "integer", "minimum": 1, "maximum": 4, "description": "Оценка точности ответа"},
            "relevance": {"type": "integer", "minimum": 1, "maximum": 4, "description": "Оценка релевантности"},
            "total_rating": {"type": "number", "description": "Среднее по трём метрикам"},
        },
        "required": ["evaluation", "faithfulness", "accuracy", "relevance", "total_rating"],
    }


@dataclass
class EvaluationSample:
    """Один образец для оценки."""

    dataset: str
    question_id: int
    question: str
    answer: str
    model: str
    backend: str
    context: Optional[str] = None
    correct_answer: Optional[str] = None
    reasoning: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class JudgeResult:
    """Результат оценки одним судьёй."""

    judge_name: str
    judge_model: str
    faithfulness: int  # 1-4
    accuracy: int  # 1-4
    relevance: int  # 1-4
    total_rating: float  # среднее по трём метрикам
    evaluation_text: str  # обоснование
    raw_response: str
    parsing_error: Optional[str] = None


@dataclass
class SampleEvaluation:
    """Полная оценка образца всеми судьями."""

    sample: EvaluationSample
    judge_results: List[JudgeResult] = field(default_factory=list)
    consensus_faithfulness: Optional[float] = None
    consensus_accuracy: Optional[float] = None
    consensus_relevance: Optional[float] = None
    consensus_total: Optional[float] = None


# ==================== ШАБЛОНЫ ПРОМПТОВ ====================

FAITHFULNESS_PROMPT = """Ты — строгий эксперт-оценщик качества RAG-систем. Твоя задача — оценить верность ответа предоставленному контексту (Faithfulness).

КРИТЕРИИ ОЦЕНКИ (шкала 1-4):
4 — ОТЛИЧНО: Все утверждения в ответе полностью подтверждаются контекстом. Нет выдуманных фактов, нет противоречий контексту.
3 — ХОРОШО: Большинство утверждений подтверждаются контекстом. Есть незначительные детали, которые нельзя проверить по контексту, но они не противоречат ему.
2 — ПЛОХО: Некоторые ключевые утверждения не подтверждаются контекстом или частично противоречат ему. Есть признаки галлюцинаций.
1 — ОЧЕНЬ ПЛОХО: Ответ содержит серьёзные противоречия контексту или большинство фактов выдуманы.

Важные правила:
- Accept semantic equivalents and morphological variations (разные формы слова считаются эквивалентными)
- Оценивай только соответствие контексту, а не правильность фактов в абсолютном смысле
- Если контекста нет, оценивай внутреннюю непротиворечивость ответа

Формат ответа СТРОГО:
Evaluation: [твоё подробное обоснование на 2-4 предложения]
Total rating: [число от 1 до 4]
"""

ACCURACY_PROMPT = """Ты — строгий эксперт-оценщик качества RAG-систем. Твоя задача — оценить точность ответа по сравнению с эталонным ответом (Accuracy).

КРИТЕРИИ ОЦЕНКИ (шкала 1-4):
4 — ОТЛИЧНО: Ответ полностью соответствует эталону по всем ключевым фактам. Допускаются незначительные формулировочные различия.
3 — ХОРОШО: Основная информация верна, но упущены некоторые детали из эталона или есть мелкие неточности.
2 — ПЛОХО: Есть существенные расхождения с эталоном. Часть ключевой информации отсутствует или неверна.
1 — ОЧЕНЬ ПЛОХО: Ответ существенно отличается от эталона, содержит фактические ошибки или не отвечает на вопрос.

Важные правила:
- Accept semantic equivalents and morphological variations (синонимы и разные формы слов допустимы)
- Сравнивай смысловое содержание, а не дословное совпадение
- Если эталонного ответа нет, оценивай полноту и фактическую корректность ответа

Формат ответа СТРОГО:
Evaluation: [твоё подробное обоснование на 2-4 предложения]
Total rating: [число от 1 до 4]
"""

RELEVANCE_PROMPT = """Ты — строгий эксперт-оценщик качества RAG-систем. Твоя задача — оценить релевантность ответа заданному вопросу (Relevance).

КРИТЕРИИ ОЦЕНКИ (шкала 1-4):
4 — ОТЛИЧНО: Ответ прямо и полно отвечает на вопрос. Вся информация относится к делу.
3 — ХОРОШО: Ответ в основном релевантен, но содержит небольшие отступления или неполностью раскрывает тему.
2 — ПЛОХО: Ответ лишь частично относится к вопросу. Есть значительные отступления или ответ поверхностный.
1 — ОЧЕНЬ ПЛОХО: Ответ не релевантен вопросу или полностью уходит от темы.

Важные правила:
- Оценивай насколько ответ адресует именно этот вопрос
- Учитывай полноту ответа на поставленный вопрос
- Игнорируй грамматические ошибки, фокусируйся на содержании

Формат ответа СТРОГО:
Evaluation: [твоё подробное обоснование на 2-4 предложения]
Total rating: [число от 1 до 4]
"""


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================


def calculate_optimal_context_length(eval_file: str, context_file: str) -> int:
    """
    Вычисляет оптимальную длину контекста на основе максимального размера входных данных.

    Анализирует файлы evaluation_results_all.csv и all_q_True.csv,
    чтобы определить максимальную длину комбинации question+answer+context.

    Возвращает рекомендуемый num_ctx для Ollama.
    """
    try:
        import pandas as pd

        eval_df = pd.read_csv(eval_file)
        ctx_df = pd.read_csv(context_file)

        # Считаем длины текстов
        eval_df["combined_len"] = eval_df["question"].fillna("").astype(str) + eval_df["answer"].fillna("").astype(str)
        ctx_df["combined_len"] = (
            ctx_df["question"].fillna("").astype(str)
            + ctx_df["context"].fillna("").astype(str)
            + ctx_df["correct_answer"].fillna("").astype(str)
        )

        max_chars = max(eval_df["combined_len"].str.len().max(), ctx_df["combined_len"].str.len().max())

        # Добавляем запас для промпта судьи (~1000 символов)
        max_chars += 1000

        # Конвертируем символы в токены (~4 символа на токен для смешанного текста)
        estimated_tokens = int(max_chars / 3.5)

        # Округляем вверх до ближайшей степени 2, минимум 8192
        context_length = 8192
        while context_length < estimated_tokens:
            context_length *= 2

        print(f"✓ Анализ данных:")
        print(f"  - Максимальная длина входа: ~{max_chars} символов")
        print(f"  - Оценка токенов: ~{estimated_tokens}")
        print(f"  - Рекомендуемый num_ctx: {context_length}")

        return context_length

    except Exception as e:
        print(f"⚠ Не удалось рассчитать оптимальный контекст: {e}")
        return 16384  # Значение по умолчанию с запасом


def call_ollama(
    model_id: str,
    prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 256,
    timeout: int = 120,
    context_length: Optional[int] = None,
) -> str:
    """
    Вызов LLM через Ollama REST API со строгой схемой ответа.

    Args:
        model_id: Идентификатор модели
        prompt: Текст запроса
        temperature: Температура генерации (низкая для консистентности)
        max_tokens: Максимальное количество токенов для генерации
        timeout: Таймаут в секундах
        context_length: Длина контекста окна (автоматически рассчитывается если None)

    Returns:
        JSON строка с результатом или текст ошибки
    """
    import requests

    # Автоматический расчёт context_length если не указан
    if context_length is None:
        context_length = calculate_optimal_context_length(EVALUATION_RESULTS_FILE, CONTEXT_FILE)

    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model_id,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": context_length,  # Устанавливаем размер контекстного окна
        },
        "format": OllamaOutputSchema.schema,  # Требует строго структурированный JSON
        "keep_alive": "10m",
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        result = resp.json()

        # Освобождаем модель после использования
        try:
            requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": model_id, "prompt": "", "stream": False, "keep_alive": 0},
                timeout=10,
            )
        except Exception:
            pass

        response_text = result.get("response", "").strip()

        # Проверяем, что ответ валидный JSON
        try:
            parsed = json.loads(response_text)
            # Если это уже JSON, возвращаем как есть
            return response_text
        except json.JSONDecodeError:
            # Если не JSON, оборачиваем в сообщение об ошибке
            return json.dumps({"error": "Invalid JSON response", "raw": response_text}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def parse_judge_response(response: str) -> Tuple[Optional[int], str]:
    """
    Парсит ответ судьи, извлекая оценку.

    Возвращает: (оценка, текст обоснования)
    Если парсинг не удался: (None, полный ответ)
    """
    # Ищем паттерн "Total rating: X"
    match = re.search(r"Total rating:\s*(\d+)", response, re.IGNORECASE)

    if match:
        rating = int(match.group(1))
        # Извлекаем обоснование (всё до Total rating)
        eval_match = re.search(r"Evaluation:\s*(.+?)(?=Total rating:|$)", response, re.IGNORECASE | re.DOTALL)
        evaluation_text = eval_match.group(1).strip() if eval_match else ""

        # Валидация шкалы
        if 1 <= rating <= 4:
            return rating, evaluation_text
        else:
            return None, response

    # Попытка найти просто число в конце
    numbers = re.findall(r"\b([1-4])\b", response)
    if numbers:
        # Берём последнее число (обычно это итоговая оценка)
        return int(numbers[-1]), response[: response.rfind(numbers[-1])].strip()

    return None, response


def load_evaluation_results(filepath: str, max_samples: Optional[int] = None) -> List[EvaluationSample]:
    """Загружает результаты оценки из CSV."""
    samples = []
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_samples and i >= max_samples:
                break

            sample = EvaluationSample(
                dataset=row.get("dataset", ""),
                question_id=int(row.get("question_id", 0)),
                question=row.get("question", ""),
                answer=row.get("answer", ""),
                model=row.get("model", ""),
                backend=row.get("backend", ""),
                reasoning=row.get("reasoning", ""),
                confidence=float(row["confidence"]) if row.get("confidence") else None,
            )
            samples.append(sample)

    print(f"✓ Загружено {len(samples)} образцов для оценки")
    return samples


def load_context_data(filepath: str) -> Dict[Tuple[str, int], Dict[str, str]]:
    """
    Загружает контекстные данные из CSV.

    Возвращает словарь: (dataset, question_id) -> {context, correct_answer}
    """
    context_map = {}
    path = Path(filepath)

    if not path.exists():
        print(f"⚠ Файл контекста не найден: {filepath}")
        return context_map

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("dataset", ""), int(row.get("question_id", 0)))
            context_map[key] = {
                "context": row.get("context", ""),
                "correct_answer": row.get("correct_answer", ""),
            }

    print(f"✓ Загружен контекст для {len(context_map)} вопросов")
    return context_map


def enrich_sample_with_context(sample: EvaluationSample, context_map: Dict) -> EvaluationSample:
    """Добавляет контекст и правильный ответ к образцу, если доступно."""
    key = (sample.dataset, sample.question_id)
    if key in context_map:
        sample.context = context_map[key].get("context")
        sample.correct_answer = context_map[key].get("correct_answer")
    return sample


# ==================== ФУНКЦИИ ОЦЕНКИ ====================


def evaluate_faithfulness(sample: EvaluationSample, judge_model: str, judge_name: str) -> JudgeResult:
    """Оценивает верность ответа контексту (Faithfulness)."""

    context_text = sample.context if sample.context else "[Контекст не предоставлен]"

    user_prompt = f"""ВОПРОС: {sample.question}

КОНТЕКСТ:
{context_text}

ОТВЕТ ДЛЯ ОЦЕНКИ:
{sample.answer}

Оцени верность ответа контексту по шкале 1-4:"""

    messages = f"{FAITHFULNESS_PROMPT}\n\n{user_prompt}"
    raw_response = call_ollama(
        judge_model, messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, timeout=TIMEOUT_SECONDS
    )

    rating, eval_text = parse_judge_response(raw_response)

    return JudgeResult(
        judge_name=judge_name,
        judge_model=judge_model,
        faithfulness=rating if rating else 0,
        accuracy=0,
        relevance=0,
        total_rating=float(rating) if rating else 0.0,
        evaluation_text=eval_text,
        raw_response=raw_response,
        parsing_error=None if rating else "Failed to parse rating",
    )


def evaluate_accuracy(sample: EvaluationSample, judge_model: str, judge_name: str) -> JudgeResult:
    """Оценивает точность ответа по сравнению с эталоном (Accuracy)."""

    reference_text = sample.correct_answer if sample.correct_answer else "[Эталонный ответ не предоставлен]"

    user_prompt = f"""ВОПРОС: {sample.question}

ЭТАЛОННЫЙ ОТВЕТ:
{reference_text}

ОТВЕТ ДЛЯ ОЦЕНКИ:
{sample.answer}

Оцени точность ответа по шкале 1-4:"""

    messages = f"{ACCURACY_PROMPT}\n\n{user_prompt}"
    raw_response = call_ollama(
        judge_model, messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, timeout=TIMEOUT_SECONDS
    )

    rating, eval_text = parse_judge_response(raw_response)

    return JudgeResult(
        judge_name=judge_name,
        judge_model=judge_model,
        faithfulness=0,
        accuracy=rating if rating else 0,
        relevance=0,
        total_rating=float(rating) if rating else 0.0,
        evaluation_text=eval_text,
        raw_response=raw_response,
        parsing_error=None if rating else "Failed to parse rating",
    )


def evaluate_relevance(sample: EvaluationSample, judge_model: str, judge_name: str) -> JudgeResult:
    """Оценивает релевантность ответа вопросу (Relevance)."""

    user_prompt = f"""ВОПРОС: {sample.question}

ОТВЕТ ДЛЯ ОЦЕНКИ:
{sample.answer}

Оцени релевантность ответа вопросу по шкале 1-4:"""

    messages = f"{RELEVANCE_PROMPT}\n\n{user_prompt}"
    raw_response = call_ollama(
        judge_model, messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, timeout=TIMEOUT_SECONDS
    )

    rating, eval_text = parse_judge_response(raw_response)

    return JudgeResult(
        judge_name=judge_name,
        judge_model=judge_model,
        faithfulness=0,
        accuracy=0,
        relevance=rating if rating else 0,
        total_rating=float(rating) if rating else 0.0,
        evaluation_text=eval_text,
        raw_response=raw_response,
        parsing_error=None if rating else "Failed to parse rating",
    )


def evaluate_sample_full(sample: EvaluationSample, judge_config: Dict) -> JudgeResult:
    """
    Полная оценка образца одной моделью-судьёй по всем трём метрикам.

    Возвращает агрегированный результат.
    """
    judge_name = judge_config["name"]
    judge_model = judge_config["model_id"]

    results = []

    # 1. Faithfulness
    print(f"      ⚖️  Faithfulness...")
    fr = evaluate_faithfulness(sample, judge_model, judge_name)
    results.append(("faithfulness", fr))

    # 2. Accuracy
    print(f"      ⚖️  Accuracy...")
    ar = evaluate_accuracy(sample, judge_model, judge_name)
    results.append(("accuracy", ar))

    # 3. Relevance
    print(f"      ⚖️  Relevance...")
    rr = evaluate_relevance(sample, judge_model, judge_name)
    results.append(("relevance", rr))

    # Агрегируем результаты
    all_ratings = []
    all_eval_texts = []
    all_errors = []

    for metric_name, r in results:
        if r.total_rating > 0:
            all_ratings.append(r.total_rating)
        all_eval_texts.append(f"{metric_name.capitalize()}: {r.evaluation_text}")
        if r.parsing_error:
            all_errors.append(f"{metric_name}: {r.parsing_error}")

    avg_rating = sum(all_ratings) / len(all_ratings) if all_ratings else 0.0

    combined_result = JudgeResult(
        judge_name=judge_name,
        judge_model=judge_model,
        faithfulness=results[0][1].faithfulness,
        accuracy=results[1][1].accuracy,
        relevance=results[2][1].relevance,
        total_rating=avg_rating,
        evaluation_text="\n\n".join(all_eval_texts),
        raw_response="\n---\n".join([r[1].raw_response for r in results]),
        parsing_error="; ".join(all_errors) if all_errors else None,
    )

    return combined_result


# ==================== СТАТИСТИКА И ЭКСПОРТ ====================


def calculate_consensus(evaluations: List[JudgeResult]) -> Dict[str, float]:
    """Вычисляет согласованность между судьями."""
    metrics = ["faithfulness", "accuracy", "relevance"]
    consensus = {}

    for metric in metrics:
        values = [getattr(e, metric) for e in evaluations if getattr(e, metric) > 0]
        if len(values) >= 2:
            # Среднее значение
            consensus[f"avg_{metric}"] = sum(values) / len(values)
            # Стандартное отклонение (мера несогласованности)
            mean = consensus[f"avg_{metric}"]
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            consensus[f"std_{metric}"] = variance**0.5
        elif len(values) == 1:
            consensus[f"avg_{metric}"] = values[0]
            consensus[f"std_{metric}"] = 0.0
        else:
            consensus[f"avg_{metric}"] = 0.0
            consensus[f"std_{metric}"] = 0.0

    # Общая согласованность (обратное среднее стандартное отклонение)
    std_values = [consensus[f"std_{m}"] for m in metrics]
    avg_std = sum(std_values) / len(std_values) if std_values else 0.0
    consensus["overall_agreement"] = max(0, 1.0 - avg_std / 3.0)  # Нормализация

    return consensus


def save_results_csv(evaluations: List[SampleEvaluation], filepath: Path) -> None:
    """Сохраняет результаты оценки в CSV формате."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for eval_item in evaluations:
        sample = eval_item.sample
        for judge_result in eval_item.judge_results:
            row = {
                "dataset": sample.dataset,
                "question_id": sample.question_id,
                "question": sample.question,
                "answer": sample.answer,
                "original_model": sample.model,
                "original_backend": sample.backend,
                "judge_name": judge_result.judge_name,
                "judge_model": judge_result.judge_model,
                "faithfulness": judge_result.faithfulness,
                "accuracy": judge_result.accuracy,
                "relevance": judge_result.relevance,
                "total_rating": judge_result.total_rating,
                "evaluation_text": judge_result.evaluation_text,
                "parsing_error": judge_result.parsing_error or "",
                "consensus_faithfulness": eval_item.consensus_faithfulness or "",
                "consensus_accuracy": eval_item.consensus_accuracy or "",
                "consensus_relevance": eval_item.consensus_relevance or "",
                "consensus_total": eval_item.consensus_total or "",
            }
            rows.append(row)

    # Атомарная запись через временный файл
    tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    if filepath.exists():
        filepath.unlink()
    tmp_path.rename(filepath)

    print(f"✓ Результаты сохранены в {filepath}")


def save_results_json(evaluations: List[SampleEvaluation], filepath: Path) -> None:
    """Сохраняет детальные результаты в JSON формате."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for eval_item in evaluations:
        sample = eval_item.sample
        item_dict = {
            "dataset": sample.dataset,
            "question_id": sample.question_id,
            "question": sample.question,
            "answer": sample.answer,
            "original_model": sample.model,
            "original_backend": sample.backend,
            "context_used": sample.context is not None,
            "correct_answer_available": sample.correct_answer is not None,
            "judges": [],
            "consensus": {
                "faithfulness": eval_item.consensus_faithfulness,
                "accuracy": eval_item.consensus_accuracy,
                "relevance": eval_item.consensus_relevance,
                "total": eval_item.consensus_total,
            },
        }

        for judge_result in eval_item.judge_results:
            judge_dict = {
                "name": judge_result.judge_name,
                "model": judge_result.judge_model,
                "faithfulness": judge_result.faithfulness,
                "accuracy": judge_result.accuracy,
                "relevance": judge_result.relevance,
                "total_rating": judge_result.total_rating,
                "evaluation": judge_result.evaluation_text,
                "parsing_error": judge_result.parsing_error,
            }
            item_dict["judges"].append(judge_dict)

        data.append(item_dict)

    tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    if filepath.exists():
        filepath.unlink()
    tmp_path.rename(filepath)

    print(f"✓ Детальные результаты сохранены в {filepath}")


def print_statistics(evaluations: List[SampleEvaluation]) -> None:
    """Выводит статистику оценки."""
    print("\n" + "=" * 60)
    print("📊 СТАТИСТИКА ОЦЕНКИ")
    print("=" * 60)

    total_samples = len(evaluations)
    total_judgments = sum(len(e.judge_results) for e in evaluations)

    # Сбор всех оценок по метрикам
    all_faithfulness = []
    all_accuracy = []
    all_relevance = []
    all_totals = []

    parsing_errors = 0

    for eval_item in evaluations:
        for jr in eval_item.judge_results:
            if jr.faithfulness > 0:
                all_faithfulness.append(jr.faithfulness)
            if jr.accuracy > 0:
                all_accuracy.append(jr.accuracy)
            if jr.relevance > 0:
                all_relevance.append(jr.relevance)
            if jr.total_rating > 0:
                all_totals.append(jr.total_rating)
            if jr.parsing_error:
                parsing_errors += 1

    def calc_stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values) if len(values) > 1 else 0
        return {
            "mean": mean,
            "std": variance**0.5,
            "min": min(values),
            "max": max(values),
        }

    print(f"\nОбщее количество образцов: {total_samples}")
    print(f"Общее количество оценок: {total_judgments}")
    print(f"Ошибки парсинга: {parsing_errors} ({parsing_errors / max(total_judgments, 1) * 100:.1f}%)")

    for metric_name, values in [
        ("Faithfulness", all_faithfulness),
        ("Accuracy", all_accuracy),
        ("Relevance", all_relevance),
        ("Total Rating", all_totals),
    ]:
        stats = calc_stats(values)
        print(f"\n{metric_name}:")
        print(f"  Среднее: {stats['mean']:.2f} ± {stats['std']:.2f}")
        print(f"  Диапазон: [{stats['min']}, {stats['max']}]")

    # Согласованность между судьями
    agreements = [e.consensus_total for e in evaluations if e.consensus_total is not None]
    if agreements:
        avg_agreement = sum(agreements) / len(agreements)
        print(f"\nСредняя согласованность судей: {avg_agreement:.2f}")

    print("=" * 60)


# ==================== ГЛАВНАЯ ФУНКЦИЯ ====================


def run_judge_evaluation(
    evaluation_file: str = EVALUATION_RESULTS_FILE,
    context_file: str = CONTEXT_FILE,
    output_dir: Path = OUTPUT_DIR,
    max_samples: Optional[int] = MAX_SAMPLES,
    models_to_use: Optional[List[str]] = None,
) -> List[SampleEvaluation]:
    """
    Запускает полную оценку образцов всеми моделями-судьями.

    Args:
        evaluation_file: Путь к файлу с результатами оценки
        context_file: Путь к файлу с контекстом
        output_dir: Директория для сохранения результатов
        max_samples: Максимальное количество образцов для оценки
        models_to_use: Список имён моделей для использования (None = все)

    Returns:
        Список результатов оценки
    """
    print("\n" + "=" * 60)
    print("🔍 LLM-AS-A-JUDGE ОЦЕНКА (Ollama)")
    print("=" * 60)
    print(f"Дата начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Файл результатов: {evaluation_file}")
    print(f"Файл контекста: {context_file}")
    print(f"Максимум образцов: {max_samples or 'все'}")

    # Фильтрация моделей
    judges = JUDGE_MODELS
    if models_to_use:
        judges = [j for j in judges if j["name"] in models_to_use]
        print(f"Модели судей: {[j['name'] for j in judges]}")

    # Загрузка данных
    print("\n📥 Загрузка данных...")
    samples = load_evaluation_results(evaluation_file, max_samples)
    context_map = load_context_data(context_file)

    # Обогащение контекстом
    for sample in samples:
        enrich_sample_with_context(sample, context_map)

    # Создание директории вывода
    output_dir.mkdir(parents=True, exist_ok=True)

    # Оценка
    print("\n⚖️  Запуск оценки...")
    evaluations: List[SampleEvaluation] = []

    for i, sample in enumerate(samples):
        print(f"\n[{i + 1}/{len(samples)}] Вопрос #{sample.question_id} ({sample.dataset})")

        sample_eval = SampleEvaluation(sample=sample)

        for judge_config in judges:
            print(f"    Судья: {judge_config['name']} ({judge_config['model_id']})")

            start_time = time.time()
            try:
                result = evaluate_sample_full(sample, judge_config)
                elapsed = time.time() - start_time
                print(f"      ✓ Оценка завершена за {elapsed:.1f}с")
                print(
                    f"      Scores: F={result.faithfulness}, A={result.accuracy}, R={result.relevance}, Total={result.total_rating:.2f}"
                )

                sample_eval.judge_results.append(result)
            except Exception as e:
                print(f"      ✗ Ошибка: {e}")

        # Расчёт консенсуса
        if sample_eval.judge_results:
            consensus = calculate_consensus(sample_eval.judge_results)
            sample_eval.consensus_faithfulness = consensus.get("avg_faithfulness")
            sample_eval.consensus_accuracy = consensus.get("avg_accuracy")
            sample_eval.consensus_relevance = consensus.get("avg_relevance")
            sample_eval.consensus_total = consensus.get("avg_total_rating", 0)
            if not sample_eval.consensus_total:
                ratings = [jr.total_rating for jr in sample_eval.judge_results if jr.total_rating > 0]
                sample_eval.consensus_total = sum(ratings) / len(ratings) if ratings else 0

        evaluations.append(sample_eval)

        # Сохранение промежуточных результатов каждые 10 образцов
        if (i + 1) % 10 == 0:
            save_results_csv(evaluations, output_dir / "judge_evaluation_results_temp.csv")

    # Финальное сохранение
    print("\n💾 Сохранение результатов...")
    save_results_csv(evaluations, output_dir / "judge_evaluation_results.csv")
    save_results_json(evaluations, output_dir / "judge_detailed_results.json")

    # Статистика
    print_statistics(evaluations)

    print(f"\n✅ Оценка завершена: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return evaluations


if __name__ == "__main__":
    # Пример запуска
    run_judge_evaluation(
        max_samples=5,  # Для теста ограничим 5 образцами
        models_to_use=["gpt-oss-120b"],  # Только одна модель для быстрого теста
    )
