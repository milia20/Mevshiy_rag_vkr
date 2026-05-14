"""
judge_lmstudio.py
LLM-as-a-Judge пайплайн оценки качества ответов через LM Studio SDK.

Поддерживаемые модели-судьи:
- openai/gpt-oss-120b
- google/gemma-4-26b-a4b
- qwen/qwen3.6-35b-a3b

Метрики оценки (шкала 1-4):
- Faithfulness (Верность контексту)
- Accuracy (Точность ответа)
- Relevance (Релевантность вопросу)
"""

# ==================== ИМПОРТЫ ====================
import csv
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import lmstudio as lms
    import pandas as pd
    from lmstudio import BaseModel, LlmLoadModelConfigDict, LlmPredictionConfigDict
except ImportError as e:
    lms = None
    BaseModel = object
    pd = None
    print(f"⚠ Предупреждение: {e}")


# ==================== КОНФИГУРАЦИЯ ====================


class Config:
    """Централизованная конфигурация пайплайна."""

    # Модели-судьи
    JUDGE_MODELS: list[dict[str, str]] = [
        {"name": "gpt-oss-120b", "model_id": "openai/gpt-oss-120b"},
        {"name": "gemma-4-26b", "model_id": "google/gemma-4-26b-a4b"},
        {"name": "qwen3.6-35b", "model_id": "qwen/qwen3.6-35b-a3b"},
        {"name": "qwen3.5-9b", "model_id": "qwen/qwen3.5-9b"},
    ]

    # Пути к файлам
    BASE_DIR = Path(__file__).parents[2]
    EVALUATION_RESULTS_FILE = BASE_DIR / "test_datasets" / "evaluation_results_all.csv"
    CONTEXT_FILE = BASE_DIR / "datasets" / "all_q_True.csv"
    OUTPUT_DIR = Path(__file__).parents[1] / "judge_results_lmstudio"

    # Параметры генерации
    TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 256
    TIMEOUT_SECONDS: int = 120

    # Параметры контекста
    CONTEXT_MIN_LENGTH: int = 4095
    CONTEXT_MAX_LENGTH: int = 8192
    CONTEXT_OVERHEAD: int = 512
    CONTEXT_HARD_LIMIT: int = 32768

    # Обработка данных
    MAX_SAMPLES: int | None = None
    SAMPLE_STEP: int = 100  # Для токенизации (какой шаг выборки)
    SAVE_INTERVAL: int = 10  # Сохранять промежуточные результаты каждые N образцов


# ==================== МОДЕЛИ ДАННЫХ ====================


class JudgeOutputSchema(BaseModel):
    """Строгая схема для структурированного ответа судьи."""

    evaluation: str
    faithfulness: int
    accuracy: int
    relevance: int
    total_rating: float


@dataclass
class EvaluationSample:
    """Один образец для оценки."""

    dataset: str
    question_id: int
    question: str
    answer: str
    model: str
    backend: str
    context: str | None = None
    correct_answer: str | None = None
    reasoning: str | None = None
    confidence: float | None = None


@dataclass
class JudgeResult:
    """Результат оценки одним судьёй."""

    judge_name: str
    judge_model: str
    faithfulness: int
    accuracy: int
    relevance: int
    total_rating: float
    evaluation_text: str
    raw_response: str
    parsing_error: str | None = None
    stats: dict[str, Any] | None = None


@dataclass
class SampleEvaluation:
    """Полная оценка образца всеми судьями."""

    sample: EvaluationSample
    judge_results: list[JudgeResult] = field(default_factory=list)
    consensus_faithfulness: float | None = None
    consensus_accuracy: float | None = None
    consensus_relevance: float | None = None
    consensus_total: float | None = None


# ==================== ШАБЛОНЫ ПРОМПТОВ ====================

PROMPT_TEMPLATES = {
    "faithfulness": """Ты — строгий эксперт-оценщик качества RAG-систем. Твоя задача — оценить верность ответа предоставленному контексту (Faithfulness).

КРИТЕРИИ ОЦЕНКИ (шкала 1-4):
4 — ОТЛИЧНО: Все утверждения в ответе полностью подтверждаются контекстом. Нет выдуманных фактов, нет противоречий контексту.
3 — ХОРОШО: Большинство утверждений подтверждаются контекстом. Есть незначительные детали, которые нельзя проверить по контексту, но они не противоречат ему.
2 — ПЛОХО: Некоторые ключевые утверждения не подтверждаются контекстом или частично противоречат ему. Есть признаки галлюцинаций.
1 — ОЧЕНЬ ПЛОХО: Ответ содержит серьёзные противоречия контексту или большинство фактов выдуманы.

Важные правила:
- Accept semantic equivalents and morphological variations
- Оценивай только соответствие контексту, а не правильность фактов в абсолютном смысле
- Если контекста нет, оценивай внутреннюю непротиворечивость ответа

Формат ответа СТРОГО:
Evaluation: [твоё подробное обоснование на 2-4 предложения]
Total rating: [число от 1 до 4]
""",
    "accuracy": """Ты — строгий эксперт-оценщик качества RAG-систем. Твоя задача — оценить точность ответа по сравнению с эталонным ответом (Accuracy).

КРИТЕРИИ ОЦЕНКИ (шкала 1-4):
4 — ОТЛИЧНО: Ответ полностью соответствует эталону по всем ключевым фактам. Допускаются незначительные формулировочные различия.
3 — ХОРОШО: Основная информация верна, но упущены некоторые детали из эталона или есть мелкие неточности.
2 — ПЛОХО: Есть существенные расхождения с эталоном. Часть ключевой информации отсутствует или неверна.
1 — ОЧЕНЬ ПЛОХО: Ответ существенно отличается от эталона, содержит фактические ошибки или не отвечает на вопрос.

Важные правила:
- Accept semantic equivalents and morphological variations
- Сравнивай смысловое содержание, а не дословное совпадение
- Если эталонного ответа нет, оценивай полноту и фактическую корректность ответа

Формат ответа СТРОГО:
Evaluation: [твоё подробное обоснование на 2-4 предложения]
Total rating: [число от 1 до 4]
""",
    "relevance": """Ты — строгий эксперт-оценщик качества RAG-систем. Твоя задача — оценить релевантность ответа заданному вопросу (Relevance).

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
""",
}


# ==================== МЕНЕДЖЕР КОНТЕКСТА ====================


class ContextManager:
    """Управление длиной и обрезкой контекста."""

    @staticmethod
    def _estimate_tokens_by_chars(text: str) -> int:
        """Грубая оценка токенов по символам (4 символа ≈ 1 токен)."""
        return len(text) // 4 if text else 0

    @staticmethod
    def _trim_by_chars(text: str, max_chars: int, strategy: str = "head_tail") -> str:
        """Обрезка текста по символам с сохранением структуры."""
        if len(text) <= max_chars:
            return text

        if strategy == "head_only":
            return text[:max_chars]
        elif strategy == "tail_only":
            return text[-max_chars:]
        else:  # head_tail
            head = int(max_chars * 0.4)
            tail = max_chars - head - 5
            return text[:head] + " [...] " + text[-tail:] if tail > 20 else text[:max_chars]

    @staticmethod
    def _trim_by_tokens(tokens: list[Any], max_tokens: int, strategy: str = "head_tail", llm: Any | None = None) -> str:
        """Обрезка по токенам с попыткой детокенизации."""
        if len(tokens) <= max_tokens:
            return tokens if isinstance(tokens, str) else "".join(tokens)

        if strategy == "head_only":
            trimmed = tokens[:max_tokens]
        elif strategy == "tail_only":
            trimmed = tokens[-max_tokens:]
        else:  # head_tail
            head = int(max_tokens * 0.4)
            tail = max_tokens - head
            trimmed = tokens[:head] + tokens[-tail:]

        # Попытка детокенизировать
        if llm and hasattr(llm, "detokenize"):
            try:
                return llm.detokenize(trimmed)
            except Exception:
                pass

        # Fallback к символьной оценке
        return str(trimmed)

    @classmethod
    def calculate_optimal_length(
        cls,
        eval_file: str,
        context_file: str,
        judge_model_id: str,
        llm_instance: Any | None = None,
    ) -> tuple[int, Callable[[str, str], str]]:
        """
        Вычисляет оптимальную длину контекста.

        Возвращает:
            - context_length: рекомендуемая длина окна (степень двойки, мин. 8192, макс. 8000 при первом запуске)
            - trim_func: функция для обрезки контекста
        """
        if pd is None:
            print("⚠ pandas не установлен, используем дефолтные значения")
            return Config.CONTEXT_MIN_LENGTH, lambda ctx, q="": ctx

        try:
            # Загрузка данных
            eval_df = pd.read_csv(eval_file).fillna("")
            ctx_df = pd.read_csv(context_file).fillna("")

            # Подготовка текстов для анализа
            eval_texts = (eval_df["question"].astype(str) + " " + eval_df["answer"].astype(str)).dropna()
            ctx_texts = (
                ctx_df["question"].astype(str)
                + " "
                + ctx_df["context"].astype(str)
                + " "
                + ctx_df["correct_answer"].astype(str)
            ).dropna()

            all_texts = pd.concat([eval_texts, ctx_texts], ignore_index=True)
            if all_texts.empty:
                return Config.CONTEXT_MIN_LENGTH, lambda ctx, q="": ctx

            # Токенизация через LLM или fallback
            sample_step = max(1, len(all_texts) // Config.SAMPLE_STEP)
            token_lengths = []

            print(f"🔍 Анализ токенов (выборка шаг={sample_step}, всего={len(all_texts)}):")

            for i, text in enumerate(all_texts[::sample_step]):
                try:
                    if llm_instance and hasattr(llm_instance, "tokenize"):
                        tokens = llm_instance.tokenize(str(text))
                        token_lengths.append(len(tokens))
                    else:
                        token_lengths.append(cls._estimate_tokens_by_chars(str(text)))
                except Exception:
                    token_lengths.append(cls._estimate_tokens_by_chars(str(text)))

                if (i + 1) % 20 == 0:
                    print(f"  Обработано {i + 1} образцов...")

            if not token_lengths:
                return Config.CONTEXT_MIN_LENGTH, lambda ctx, q="": ctx

            # Расчёт требуемой длины
            max_tokens = max(token_lengths)
            required = max_tokens + Config.CONTEXT_OVERHEAD

            # ← КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: первый расчёт ограничен Config.CONTEXT_MAX_LENGTH (8000)
            context_length = Config.CONTEXT_MIN_LENGTH
            while context_length < required and context_length < Config.CONTEXT_MAX_LENGTH:
                context_length *= 2

            # Если даже 8000 не хватает — предупреждаем, но не превышаем
            if required > Config.CONTEXT_MAX_LENGTH:
                print(f"⚠ Требуется ~{required} токенов, но первый расчёт ограничен {Config.CONTEXT_MAX_LENGTH}")
                print("  Будет применена автоматическая обрезка длинных контекстов")
                context_length = Config.CONTEXT_MAX_LENGTH

            print(f"✓ Контекст: макс. в образце ~{max_tokens}, с запасом ~{required}, окно: {context_length}")

            # Функция обрезки
            available = context_length - Config.CONTEXT_OVERHEAD
            return context_length, lambda ctx, q="": cls._trim_context_smart(ctx, q, available, llm_instance)

        except Exception as e:
            print(f"⚠ Ошибка расчёта контекста: {e}, используем fallback")
            return Config.CONTEXT_MIN_LENGTH, lambda ctx, q="": cls._trim_by_chars(ctx, 6000)

    @staticmethod
    def _trim_context_smart(
        context: str, question: str, max_tokens: int, llm: Any | None, strategy: str = "head_tail"
    ) -> str:
        """Умная обрезка контекста с учётом токенов."""
        if not context or len(context) < 100:
            return context

        # Оценка токенов вопроса
        try:
            q_tokens = len(llm.tokenize(question)) if llm and question else 0
        except Exception:
            q_tokens = ContextManager._estimate_tokens_by_chars(question)

        available = max_tokens - q_tokens - 50
        if available <= 0:
            print(f"⚠ Вопрос занимает ~{q_tokens} токенов, контекст обрезан до минимума")
            return ""

        # Попытка токенизации контекста
        try:
            ctx_tokens = llm.tokenize(context) if llm else None
        except Exception:
            ctx_tokens = None

        if ctx_tokens and len(ctx_tokens) <= available:
            return context

        if ctx_tokens:
            print(f"✂️ Обрезка контекста: {len(ctx_tokens)} → {available} токенов")
            return ContextManager._trim_by_tokens(ctx_tokens, available, strategy, llm)

        # Fallback: символьная обрезка
        ratio = available * 4 / max(len(context), 1)
        return ContextManager._trim_by_chars(context, int(len(context) * min(ratio, 1)), strategy)


# ==================== МЕНЕДЖЕР МОДЕЛЕЙ ====================


class ModelManager:
    """Управление загрузкой и конфигурацией моделей LM Studio."""

    MODEL_CONFIGS = {
        "google/gemma-4-26b-a4b": {"repeat_penalty": 1.25, "seed": 42},
        "default": {"repeat_penalty": 1.1, "seed": 42},
    }

    @classmethod
    def get_or_load_model(
        cls,
        model_id: str,
        context_length: int,
        ttl: int = 3600,
    ) -> Any:
        """Получает загруженную модель или загружает новую."""
        if lms is None:
            raise RuntimeError("lmstudio package not installed")

        # Проверка уже загруженных моделей
        loaded = lms.list_loaded_models("llm")
        for model in loaded:
            if model.identifier == model_id:
                return model

        # Выгрузка других моделей для освобождения памяти
        for model in loaded:
            try:
                model.unload()
            except Exception:
                pass

        # Загрузка новой модели
        config = cls.MODEL_CONFIGS.get(model_id, cls.MODEL_CONFIGS["default"])
        return lms.llm(
            model_id, ttl=ttl, config=LlmLoadModelConfigDict(contextLength=context_length, seed=config["seed"])
        )

    @classmethod
    def get_prediction_config(cls, model_id: str, **overrides) -> LlmPredictionConfigDict:
        """Возвращает конфигурацию для генерации с учётом специфики модели."""
        base = cls.MODEL_CONFIGS.get(model_id, cls.MODEL_CONFIGS["default"]).copy()
        base.update(overrides)
        return LlmPredictionConfigDict(**base)


# ==================== ПАРСИНГ И ВАЛИДАЦИЯ ====================


def parse_judge_response(response: str) -> tuple[int | None, str]:
    """
    Парсит ответ судьи, извлекая оценку и обоснование.

    Returns:
        (rating: Optional[int], evaluation_text: str)
    """
    # Поиск явного паттерна "Total rating: X"
    match = re.search(r"Total rating:\s*(\d+)", response, re.IGNORECASE)
    if match:
        rating = int(match.group(1))
        eval_match = re.search(r"Evaluation:\s*(.+?)(?=Total rating:|$)", response, re.IGNORECASE | re.DOTALL)
        evaluation = eval_match.group(1).strip() if eval_match else ""

        if 1 <= rating <= 4:
            return rating, evaluation

    # Fallback: поиск числа 1-4 в конце ответа
    numbers = re.findall(r"\b([1-4])\b", response)
    if numbers:
        return int(numbers[-1]), response[: response.rfind(numbers[-1])].strip()

    return None, response


def validate_rating(rating: int | None) -> bool:
    """Проверяет валидность оценки."""
    return rating is not None and 1 <= rating <= 4


# ==================== LM STUDIO API ====================


def call_lmstudio(
    model_id: str,
    prompt: str,
    context_length: int,
    trim_func: Callable[[str, str], str],
    temperature: float = Config.TEMPERATURE,
    max_tokens: int = Config.MAX_TOKENS,
    timeout: int = Config.TIMEOUT_SECONDS,
) -> tuple[str, dict[str, Any] | None]:
    """
    Вызов LLM через LM Studio SDK со строгой схемой ответа.
    """
    if lms is None:
        return "ERROR: lmstudio package not installed", None

    try:
        # Подготовка модели
        llm = ModelManager.get_or_load_model(model_id, context_length)

        # Обрезка контекста в промпте если нужно
        if "# КОНТЕКСТ:" in prompt and trim_func:
            context_match = re.search(r"# КОНТЕКСТ:\s*\n(.*?)\n\n#?ОТВЕТ", prompt, re.DOTALL)
            if context_match:
                original = context_match.group(1).strip()
                trimmed = trim_func(original, "")
                prompt = prompt.replace(original, trimmed)

        # Конфигурация запроса
        pred_config = ModelManager.get_prediction_config(
            model_id,
            maxTokens=max_tokens,
            temperature=temperature,
        )

        # Выполнение запроса
        raw_result = llm.respond(prompt, config=pred_config, response_format=JudgeOutputSchema)

        # Извлечение ответа
        if raw_result.structured and raw_result.parsed:
            response_text = json.dumps(raw_result.parsed, ensure_ascii=False)
        else:
            response_text = raw_result.content

        stats = raw_result.stats.to_dict() if raw_result.stats else None
        return response_text, stats

    except Exception as e:
        return f"ERROR: {e!s}", None


# ==================== ЗАГРУЗКА ДАННЫХ ====================


def load_evaluation_samples(filepath: str, max_samples: int | None = None) -> list[EvaluationSample]:
    """Загружает образцы для оценки из CSV."""
    samples = []
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {filepath}")

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_samples and i >= max_samples:
                break

            samples.append(
                EvaluationSample(
                    dataset=row.get("dataset", ""),
                    question_id=int(row.get("question_id", 0)),
                    question=row.get("question", ""),
                    answer=row.get("answer", ""),
                    model=row.get("model", ""),
                    backend=row.get("backend", ""),
                    context=row.get("context"),
                    correct_answer=row.get("correct_answer"),
                    reasoning=row.get("reasoning", ""),
                    confidence=float(row["confidence"]) if row.get("confidence") else None,
                )
            )

    print(f"✓ Загружено {len(samples)} образцов")
    return samples


def load_context_map(filepath: str) -> dict[tuple[str, int], dict[str, str]]:
    """Загружает контекст и эталонные ответы в словарь для быстрого доступа."""
    path = Path(filepath)

    if not path.exists() or pd is None:
        print(f"⚠ Контекст не загружен: {filepath}")
        return {}

    try:
        df = pd.read_csv(path, encoding="utf-8").fillna("")

        # Гарантия наличия колонок
        for col in ["dataset", "question_id", "context", "correct_answer"]:
            if col not in df.columns:
                df[col] = ""

        df["question_id"] = pd.to_numeric(df["question_id"], errors="coerce").fillna(0).astype(int)

        context_map = df.set_index(["dataset", "question_id"])[["context", "correct_answer"]].to_dict(orient="index")

        print(f"✓ Контекст загружен для {len(context_map)} вопросов")
        return context_map

    except Exception as e:
        print(f"⚠ Ошибка загрузки контекста: {e}")
        return {}


def enrich_sample(sample: EvaluationSample, context_map: dict) -> EvaluationSample:
    """Добавляет контекст и эталонный ответ к образцу если доступно."""
    key = (sample.dataset, sample.question_id)
    if key in context_map:
        sample.context = context_map[key].get("context")
        sample.correct_answer = context_map[key].get("correct_answer")
    return sample


# ==================== ОЦЕНКА: БАЗОВЫЙ КЛАСС ====================


class MetricEvaluator:
    """Базовый класс для оценки по одной метрике."""

    METRIC_NAME: str = ""
    PROMPT_KEY: str = ""

    def __init__(self, sample: EvaluationSample, trim_func: Callable[[str, str], str] | None = None):
        self.sample = sample
        self.trim_func = trim_func

    def _prepare_context(self) -> str:
        """Подготавливает контекст/эталон для промпта."""
        raise NotImplementedError

    def _build_prompt(self, context_text: str) -> str:
        """Собирает финальный промпт для судьи."""
        template = PROMPT_TEMPLATES[self.PROMPT_KEY]
        return f"""{template}

ВОПРОС: {self.sample.question}

{self._prepare_context()}

ОТВЕТ ДЛЯ ОЦЕНКИ:
{self.sample.answer}

Оцени по шкале 1-4:"""

    def evaluate(
        self,
        judge_model: str,
        judge_name: str,
        context_length: int,
        trim_func: Callable[[str, str], str],
    ) -> JudgeResult:
        """Выполняет оценку и возвращает результат."""
        prompt = self._build_prompt(self._prepare_context())

        raw_response, stats = call_lmstudio(
            model_id=judge_model,
            prompt=prompt,
            context_length=context_length,
            trim_func=trim_func,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
            timeout=Config.TIMEOUT_SECONDS,
        )

        rating, eval_text = parse_judge_response(raw_response)

        # Заполнение полей в зависимости от метрики
        metrics = {"faithfulness": 0, "accuracy": 0, "relevance": 0}
        if self.METRIC_NAME in metrics and validate_rating(rating):
            metrics[self.METRIC_NAME] = rating

        return JudgeResult(
            judge_name=judge_name,
            judge_model=judge_model,
            faithfulness=metrics["faithfulness"],
            accuracy=metrics["accuracy"],
            relevance=metrics["relevance"],
            total_rating=float(rating) if validate_rating(rating) else 0.0,
            evaluation_text=eval_text,
            raw_response=raw_response,
            parsing_error=None if validate_rating(rating) else "Failed to parse rating",
            stats=stats,
        )


class FaithfulnessEvaluator(MetricEvaluator):
    METRIC_NAME = "faithfulness"
    PROMPT_KEY = "faithfulness"

    def _prepare_context(self) -> str:
        context = self.sample.context or "[Контекст не предоставлен]"
        if self.trim_func and context != "[Контекст не предоставлен]":
            context = self.trim_func(context, self.sample.question)
        return f"КОНТЕКСТ:\n{context}"


class AccuracyEvaluator(MetricEvaluator):
    METRIC_NAME = "accuracy"
    PROMPT_KEY = "accuracy"

    def _prepare_context(self) -> str:
        reference = self.sample.correct_answer or "[Эталонный ответ не предоставлен]"
        return f"ЭТАЛОННЫЙ ОТВЕТ:\n{reference}"


class RelevanceEvaluator(MetricEvaluator):
    METRIC_NAME = "relevance"
    PROMPT_KEY = "relevance"

    def _prepare_context(self) -> str:
        return ""  # Для relevance контекст не нужен


# ==================== АГРЕГАЦИЯ И СТАТИСТИКА ====================


def aggregate_judge_results(results: list[tuple[str, JudgeResult]]) -> JudgeResult:
    """Агрегирует результаты трёх метрик в один итоговый объект."""
    all_ratings = [r.total_rating for _, r in results if r.total_rating > 0]
    all_evals = [f"{name.capitalize()}: {r.evaluation_text}" for name, r in results]
    all_errors = [f"{name}: {r.parsing_error}" for name, r in results if r.parsing_error]
    all_stats = [r.stats for _, r in results if r.stats]

    # Объединение статистики (среднее по числовым полям)
    combined_stats = None
    if all_stats:
        combined_stats = {}
        for key in all_stats[0]:
            values = [s[key] for s in all_stats if isinstance(s.get(key), (int, float))]
            combined_stats[key] = sum(values) / len(values) if values else all_stats[0].get(key)

    return JudgeResult(
        judge_name=results[0][1].judge_name,
        judge_model=results[0][1].judge_model,
        faithfulness=results[0][1].faithfulness,
        accuracy=results[1][1].accuracy,
        relevance=results[2][1].relevance,
        total_rating=sum(all_ratings) / len(all_ratings) if all_ratings else 0.0,
        evaluation_text="\n\n".join(all_evals),
        raw_response="\n---\n".join([r[1].raw_response for r in results]),
        parsing_error="; ".join(all_errors) if all_errors else None,
        stats=combined_stats,
    )


def calculate_consensus(judge_results: list[JudgeResult]) -> dict[str, float]:
    """Вычисляет согласованность между судьями по всем метрикам."""
    metrics = ["faithfulness", "accuracy", "relevance"]
    consensus = {}

    for metric in metrics:
        values = [getattr(r, metric) for r in judge_results if getattr(r, metric) > 0]
        if values:
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values) if len(values) > 1 else 0
            consensus[f"avg_{metric}"] = mean
            consensus[f"std_{metric}"] = variance**0.5
        else:
            consensus[f"avg_{metric}"] = 0.0
            consensus[f"std_{metric}"] = 0.0

    # Общий рейтинг
    totals = [r.total_rating for r in judge_results if r.total_rating > 0]
    consensus["avg_total_rating"] = sum(totals) / len(totals) if totals else 0.0

    # Общая согласованность (1 - нормализованное среднее отклонение)
    stds = [consensus[f"std_{m}"] for m in metrics]
    avg_std = sum(stds) / len(stds) if stds else 0
    consensus["overall_agreement"] = max(0, 1.0 - avg_std / 3.0)

    return consensus


# ==================== 🔥 ОБНОВЛЁННЫЕ УТИЛИТЫ ДЛЯ ПРОВЕРКИ ПАРОЙ ВОПРОС-СУДЬЯ ====================


def _load_processed_pairs(output_dir: Path, judge_names: list[str]) -> set[tuple[str, int, str]]:
    """
    Загружает множество уже обработанных троек (dataset, question_id, judge_name).

    Приоритет источников:
    1. judge_detailed_results_temp.json (текущий сеанс)
    2. judge_detailed_results.json (предыдущие сеансы)
    3. judge_evaluation_results_temp.csv / .csv (резервный вариант)

    Returns:
        set кортежей (dataset, question_id, judge_name)
    """
    processed = set()

    # Сначала пробуем JSON (более надёжный источник)
    json_files = [
        output_dir / "judge_detailed_results_temp.json",
        output_dir / "judge_detailed_results.json",
    ]

    for filepath in json_files:
        if not filepath.exists():
            continue
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                dataset = item.get("dataset", "")
                qid = item.get("question_id")
                if dataset and qid is not None:
                    # Проверяем, какие судьи уже оценили этот вопрос
                    for judge_entry in item.get("judges", []):
                        judge_name = judge_entry.get("name")
                        if judge_name:
                            processed.add((dataset, int(qid), judge_name))
            print(f"✓ Загружено {len(processed)} пар вопрос-судья из {filepath.name}")
            break  # Берём только первый доступный (temp имеет приоритет)
        except Exception as e:
            print(f"⚠ Не удалось прочитать {filepath}: {e}")
            continue

    # Если JSON не дал результатов, пробуем CSV как fallback
    if not processed:
        csv_files = [
            output_dir / "judge_evaluation_results_temp.csv",
            output_dir / "judge_evaluation_results.csv",
        ]
        for filepath in csv_files:
            if not filepath.exists():
                continue
            try:
                with open(filepath, encoding="utf-8") as f:
                    next(f, None)  # пропускаем заголовок
                    for line in f:
                        parts = line.strip().split(";")
                        if len(parts) >= 7:  # минимальное количество колонок
                            dataset = parts[0].strip()
                            try:
                                qid = int(parts[1].strip())
                                judge_name = parts[6].strip()  # judge_name — 7-я колонка
                                if dataset and judge_name:
                                    processed.add((dataset, qid, judge_name))
                            except ValueError:
                                continue
                print(f"✓ Загружено {len(processed)} пар вопрос-судья из {filepath.name} (CSV fallback)")
                break
            except Exception as e:
                print(f"⚠ Не удалось прочитать CSV {filepath}: {e}")
                continue

    return processed


def _filter_unprocessed_pairs(
    samples: list[EvaluationSample], judges: list[dict[str, str]], processed_pairs: set[tuple[str, int, str]]
) -> tuple[list[tuple[EvaluationSample, str]], int]:
    """
    Фильтрует пары (образец, судья), оставляя только необработанные.

    Returns:
        (список кортежей (sample, judge_name), количество пропущенных пар)
    """
    if not processed_pairs:
        # Если нет обработанных — возвращаем все возможные комбинации
        all_pairs = [(s, j["name"]) for s in samples for j in judges]
        return all_pairs, 0

    pending = []
    skipped = 0

    for sample in samples:
        for judge_cfg in judges:
            key = (sample.dataset, sample.question_id, judge_cfg["name"])
            if key in processed_pairs:
                skipped += 1
            else:
                pending.append((sample, judge_cfg["name"]))

    return pending, skipped


# ==================== ЭКСПОРТ РЕЗУЛЬТАТОВ ====================
def _atomic_write(filepath: Path, content: str | bytes, mode: str = "w") -> None:
    tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")
    with open(tmp_path, mode, encoding="utf-8" if "w" in mode else None) as f:
        f.write(content)
    if filepath.exists():
        filepath.unlink()
    tmp_path.rename(filepath)


def save_results_csv(
    evaluations: list[SampleEvaluation],
    filepath: Path,
    total_samples: int | None = None,
    is_incremental: bool = False,
    append_mode: bool = False,
    existing_pairs: set[tuple[str, int, str]] | None = None,
) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    skip_pairs = existing_pairs if append_mode and existing_pairs else set()

    rows = []
    for ev in evaluations:
        for jr in ev.judge_results:
            pair_key = (ev.sample.dataset, ev.sample.question_id, jr.judge_name)
            if append_mode and pair_key in skip_pairs:
                continue
            rows.append(
                {
                    "dataset": ev.sample.dataset,
                    "question_id": ev.sample.question_id,
                    "question": ev.sample.question,
                    "answer": ev.sample.answer,
                    "original_model": ev.sample.model,
                    "original_backend": ev.sample.backend,
                    "judge_name": jr.judge_name,
                    "judge_model": jr.judge_model,
                    "faithfulness": jr.faithfulness,
                    "accuracy": jr.accuracy,
                    "relevance": jr.relevance,
                    "total_rating": jr.total_rating,
                    "evaluation_text": jr.evaluation_text,
                    "parsing_error": jr.parsing_error or "",
                    "consensus_faithfulness": ev.consensus_faithfulness or "",
                    "consensus_accuracy": ev.consensus_accuracy or "",
                    "consensus_relevance": ev.consensus_relevance or "",
                    "consensus_total": ev.consensus_total or "",
                }
            )

    if not rows:
        processed = len(evaluations) if not append_mode else 0
        remaining = (total_samples - processed) if total_samples and not append_mode else None
        mode_note = " [дополнение]" if is_incremental else ""
        print(
            f"✓ CSV сохранён: {filepath}{mode_note} | Новые записи: {len(rows)}"
            + (f" | Осталось: {remaining}" if remaining is not None else "")
        )
        return

    fieldnames = rows[0].keys()
    output = [";".join(fieldnames)]
    for row in rows:
        output.append(";".join(str(row[f]) for f in fieldnames))

    # При дозаписи — читаем существующий файл и добавляем новые строки
    if append_mode and filepath.exists():
        try:
            with open(filepath, encoding="utf-8") as f:
                existing_content = f.read()
            # Добавляем новые строки после заголовка (первая строка)
            lines = existing_content.strip().split("\n")
            if len(lines) > 1:  # есть заголовок + данные
                new_content = lines[0] + "\n" + "\n".join(lines[1:] + output[1:])
            else:
                new_content = "\n".join(output)
        except Exception:
            new_content = "\n".join(output)  # fallback: перезаписать
    else:
        new_content = "\n".join(output)

    _atomic_write(filepath, new_content)

    # ← Прогресс-лог с информацией об остатке и инкрементальности
    processed = len(evaluations) if not append_mode else 0
    remaining = (total_samples - processed) if total_samples and not append_mode else None
    mode_note = " [дополнение]" if is_incremental else ""
    skip_note = (
        f" | Пропущено дубликатов: {len(evaluations) - len({(e.sample.dataset, e.sample.question_id) for e in evaluations if (e.sample.dataset, e.sample.question_id) not in skip_keys})}"
        if append_mode and skip_keys
        else ""
    )
    print(
        f"✓ CSV сохранён: {filepath}{mode_note} | Новые записи: {len(rows)}{skip_note}"
        + (f" | Осталось: {remaining}" if remaining is not None else "")
    )


def save_results_json(
    evaluations: list[SampleEvaluation],
    filepath: Path,
    total_samples: int | None = None,
    is_incremental: bool = False,
    append_mode: bool = False,
    existing_ids: set[tuple[str, int]] | None = None,
) -> None:
    """
    Сохраняет детальные результаты в JSON.

    Args:
        evaluations: список оценённых образцов
        filepath: путь к файлу
        total_samples: общее количество образцов (для отображения прогресса)
        is_incremental: флаг инкрементального сохранения (дополнение)
        append_mode: режим дозаписи (пропуск уже сохранённых)
        existing_ids: множество уже сохранённых (dataset, question_id) для пропуска
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Загрузка существующих данных при дозаписи
    existing_data = []

    if append_mode and filepath.exists():
        try:
            with open(filepath, encoding="utf-8") as f:
                existing_data = json.load(f)
            # Создаем set ключей из существующих данных для быстрого поиска
            existing_keys = {(item["dataset"], item["question_id"]) for item in existing_data}
        except Exception:
            existing_data = []
            existing_keys = set()
    else:
        existing_keys = set()

    # Фильтрация и добавление новых записей
    for ev in evaluations:
        key_base = (ev.sample.dataset, ev.sample.question_id)
        # Проверяем, есть ли уже этот вопрос с этими судьями
        item_exists = any((item["dataset"], item["question_id"]) == key_base for item in existing_data)

        if item_exists:
            # Обновляем существующий элемент, добавляя новых судей
            for item in existing_data:
                if (item["dataset"], item["question_id"]) == key_base:
                    existing_judges = {j["name"] for j in item["judges"]}
                    for jr in ev.judge_results:
                        if jr.judge_name not in existing_judges:
                            item["judges"].append(
                                {
                                    "name": jr.judge_name,
                                    "model": jr.judge_model,
                                    "faithfulness": jr.faithfulness,
                                    "accuracy": jr.accuracy,
                                    "relevance": jr.relevance,
                                    "total_rating": jr.total_rating,
                                    "evaluation": jr.evaluation_text,
                                    "parsing_error": jr.parsing_error,
                                    "stats": jr.stats,
                                }
                            )
                            existing_judges.add(jr.judge_name)
                    break
        else:
            # Создаём новый элемент
            item = {
                "dataset": ev.sample.dataset,
                "question_id": ev.sample.question_id,
                "question": ev.sample.question,
                "answer": ev.sample.answer,
                "original_model": ev.sample.model,
                "original_backend": ev.sample.backend,
                "context_used": ev.sample.context is not None,
                "correct_answer_available": ev.sample.correct_answer is not None,
                "judges": [
                    {
                        "name": jr.judge_name,
                        "model": jr.judge_model,
                        "faithfulness": jr.faithfulness,
                        "accuracy": jr.accuracy,
                        "relevance": jr.relevance,
                        "total_rating": jr.total_rating,
                        "evaluation": jr.evaluation_text,
                        "parsing_error": jr.parsing_error,
                        "stats": jr.stats,
                    }
                    for jr in ev.judge_results
                ],
                "consensus": {
                    "faithfulness": ev.consensus_faithfulness,
                    "accuracy": ev.consensus_accuracy,
                    "relevance": ev.consensus_relevance,
                    "total": ev.consensus_total,
                },
            }
            existing_data.append(item)

    _atomic_write(filepath, json.dumps(existing_data, ensure_ascii=False, indent=2))
    new_count = sum(
        1
        for ev in evaluations
        for jr in ev.judge_results
        if (ev.sample.dataset, ev.sample.question_id, jr.judge_name) not in existing_keys
    )
    processed = len(evaluations) if not append_mode else 0
    remaining = (total_samples - processed) if total_samples and not append_mode else None
    mode_note = " [дополнение]" if is_incremental else ""
    print(
        f"✓ JSON сохранён: {filepath}{mode_note} | Новые записи: {new_count}"
        + (f" | Осталось: {remaining}" if remaining is not None else "")
    )


def print_statistics(evaluations: list[SampleEvaluation]) -> None:
    print("\n" + "=" * 60)
    print("📊 СТАТИСТИКА ОЦЕНКИ")
    print("=" * 60)
    total_samples = len(evaluations)
    total_judgments = sum(len(e.judge_results) for e in evaluations)
    metrics_data = {"faithfulness": [], "accuracy": [], "relevance": [], "total": []}
    errors_count = 0
    for ev in evaluations:
        for jr in ev.judge_results:
            if jr.faithfulness > 0:
                metrics_data["faithfulness"].append(jr.faithfulness)
            if jr.accuracy > 0:
                metrics_data["accuracy"].append(jr.accuracy)
            if jr.relevance > 0:
                metrics_data["relevance"].append(jr.relevance)
            if jr.total_rating > 0:
                metrics_data["total"].append(jr.total_rating)
            if jr.parsing_error:
                errors_count += 1

    def calc(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values) if len(values) > 1 else 0
        return {"mean": mean, "std": var**0.5, "min": min(values), "max": max(values)}

    print(f"\nОбразцов: {total_samples} | Оценок: {total_judgments} | Ошибки парсинга: {errors_count}")
    labels = {"faithfulness": "Faithfulness", "accuracy": "Accuracy", "relevance": "Relevance", "total": "Total Rating"}
    for key, label in labels.items():
        stats = calc(metrics_data[key])
        print(f"\n{label}: {stats['mean']:.2f} ± {stats['std']:.2f} [{stats['min']}, {stats['max']}]")
    agreements = [e.consensus_total for e in evaluations if e.consensus_total]
    if agreements:
        print(f"\nСредняя согласованность судей: {sum(agreements) / len(agreements):.2f}")
    print("=" * 60)


# ==================== 🔥 ГЛАВНЫЙ ПЛАЙПЛАЙН С ПРЯМЫМ ВЫВОДОМ ====================


def run_judge_evaluation(
    evaluation_file: str = Config.EVALUATION_RESULTS_FILE,
    context_file: str = Config.CONTEXT_FILE,
    output_dir: Path = Config.OUTPUT_DIR,
    max_samples: int | None = Config.MAX_SAMPLES,
    models_to_use: list[str] | None = None,
    resume: bool = True,
) -> list[SampleEvaluation]:
    """
    Запускает полную оценку образцов через LLM-as-a-Judge.

    Ключевые изменения:
    - Проверка пары (вопрос, судья) вместо только вопроса
    - Чтение из judge_detailed_results_temp.json как приоритетного источника
    - Прямой вывод в конце: question_id, judge_name, evaluation result
    """
    if lms is None:
        print("❌ Ошибка: установите пакет lmstudio (pip install lmstudio)")
        return []

    print("\n" + "=" * 60)
    print("🔍 LLM-AS-A-JUDGE (LM Studio SDK)")
    print("=" * 60)
    print(f"Старт: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Макс. образцов: {max_samples or 'все'}")
    print(f"Режим продолжения (resume): {'вкл' if resume else 'выкл'}")

    # Фильтрация моделей
    judges = [j for j in Config.JUDGE_MODELS if not models_to_use or j["name"] in models_to_use]
    judge_names = [j["name"] for j in judges]
    print(f"Модели судей: {judge_names}")

    # Загрузка данных
    print("\n📥 Загрузка данных...")
    all_samples = load_evaluation_samples(evaluation_file, max_samples)
    context_map = load_context_map(context_file)
    for s in all_samples:
        enrich_sample(s, context_map)

    # 🔥 ПРОВЕРКА УЖЕ ОБРАБОТАННЫХ ПАР ВОПРОС-СУДЬЯ
    processed_pairs = set()
    skipped_count = 0

    if resume:
        print("🔎 Проверка уже оценённых пар вопрос-судья...")
        processed_pairs = _load_processed_pairs(output_dir, judge_names)
        if processed_pairs:
            pending_pairs, skipped_count = _filter_unprocessed_pairs(all_samples, judges, processed_pairs)
            print(f"✓ Пропущено уже оценённых пар: {skipped_count}")
            print(f"✓ Осталось оценить пар: {len(pending_pairs)}")
        else:
            print("✓ Нет ранее сохранённых результатов — начинаем с нуля")
            pending_pairs = [(s, j["name"]) for s in all_samples for j in judges]
    else:
        pending_pairs = [(s, j["name"]) for s in all_samples for j in judges]

    # Если все пары уже обработаны
    if not pending_pairs:
        print("\n⚠ Все пары вопрос-судья уже оценены!")
        # Загружаем существующие результаты для вывода
        temp_json = output_dir / "judge_detailed_results_temp.json"
        main_json = output_dir / "judge_detailed_results.json"
        results_file = temp_json if temp_json.exists() else (main_json if main_json.exists() else None)

        if results_file and results_file.exists():
            print(f"\n📋 ПРЯМОЙ ВЫВОД РЕЗУЛЬТАТОВ из {results_file.name}:")
            print("-" * 80)
            with open(results_file, encoding="utf-8") as f:
                data = json.load(f)
            for item in data[:10]:  # первые 10 для примера
                qid = item["question_id"]
                for judge in item["judges"]:
                    print(
                        f"[Q:{qid}] Judge:{judge['name']} | Rating:{judge['total_rating']}/4 | {judge['evaluation'][:100]}..."
                    )
            print("-" * 80)
            print(f"✅ Всего записей: {len(data)} | Работа завершена")
        return []

    # Подготовка вывода
    output_dir.mkdir(parents=True, exist_ok=True)

    # Предварительный расчёт контекста
    print("\n🔧 Расчёт параметров контекста...")
    first_model = judges[0]["model_id"] if judges else Config.JUDGE_MODELS[0]["model_id"]
    context_length, trim_func = ContextManager.calculate_optimal_length(evaluation_file, context_file, first_model)

    # Группировка пар по образцам для эффективной обработки
    # ✅ СТАЛО (исправлено):
    from collections import defaultdict

    # Группируем по хешируемому ключу (dataset, question_id)
    samples_with_judges: dict[tuple[str, int], tuple[EvaluationSample, list[str]]] = defaultdict(lambda: (None, []))

    for sample, judge_name in pending_pairs:
        key = (sample.dataset, sample.question_id)
        stored_sample, judges_list = samples_with_judges[key]
        # Сохраняем сам образец (любой из одинаковых по ключу)
        if stored_sample is None:
            samples_with_judges[key] = (sample, [judge_name])
        else:
            samples_with_judges[key] = (stored_sample, judges_list + [judge_name])

    # Преобразуем в список для обработки
    pending_samples = [
        (sample_data[0], sample_data[1])  # (EvaluationSample, List[judge_names])
        for sample_data in samples_with_judges.values()
    ]

    # pending_samples = list(samples_with_judges.keys())
    total_pairs_to_process = len(pending_pairs)
    total_with_skipped = total_pairs_to_process + skipped_count

    # Основной цикл оценки
    print("\n⚖️  Запуск оценки...")
    evaluations: list[SampleEvaluation] = []
    processed_pairs_count = 0
    for sample, judges_for_sample in pending_samples:  # judges_for_sample — это список имён судей
        print(
            f"\n[{processed_pairs_count + 1}/{total_with_skipped}] #{sample.question_id} ({sample.dataset}) | Судьи: {', '.join(judges_for_sample)}"
        )

        sample_eval = SampleEvaluation(sample=sample)

        for judge_name in judges_for_sample:
            judge_cfg = next(j for j in judges if j["name"] == judge_name)

            print(f"    → Судья: {judge_name}")
            start = time.time()

            try:
                # Инициализация эвалюаторов
                evaluators = [
                    FaithfulnessEvaluator(sample, trim_func),
                    AccuracyEvaluator(sample, trim_func),
                    RelevanceEvaluator(sample, trim_func),
                ]

                # Последовательная оценка по метрикам
                results = []
                for evaluator in evaluators:
                    metric = evaluator.METRIC_NAME
                    result = evaluator.evaluate(judge_cfg["model_id"], judge_name, context_length, trim_func)
                    results.append((metric, result))

                # Агрегация
                final_result = aggregate_judge_results(results)
                elapsed = time.time() - start
                print(
                    f"      ✓ За {elapsed:.1f}с | F={final_result.faithfulness}, A={final_result.accuracy}, R={final_result.relevance}, Total={final_result.total_rating:.2f}"
                )
                sample_eval.judge_results.append(final_result)
                processed_pairs_count += 1

            except Exception as e:
                print(f"      ✗ Ошибка: {e}")

        # Консенсус
        if sample_eval.judge_results:
            consensus = calculate_consensus(sample_eval.judge_results)
            sample_eval.consensus_faithfulness = consensus["avg_faithfulness"]
            sample_eval.consensus_accuracy = consensus["avg_accuracy"]
            sample_eval.consensus_relevance = consensus["avg_relevance"]
            sample_eval.consensus_total = consensus["avg_total_rating"]

        evaluations.append(sample_eval)

        # Промежуточное сохранение
        if (i + 1) % Config.SAVE_INTERVAL == 0:
            remaining = total_pairs_to_process - processed_pairs_count
            print(f"\n💾 Промежуточное сохранение... (осталось пар: {remaining})")
            save_results_csv(
                evaluations,
                output_dir / "judge_evaluation_results_temp.csv",
                total_samples=total_with_skipped,
                is_incremental=True,
                append_mode=True,
                existing_pairs=processed_pairs,
            )
            save_results_json(
                evaluations,
                output_dir / "judge_detailed_results_temp.json",
                total_samples=total_with_skipped,
                is_incremental=True,
                append_mode=True,
                existing_pairs=processed_pairs,
            )
            for ev in evaluations:
                for jr in ev.judge_results:
                    processed_pairs.add((ev.sample.dataset, ev.sample.question_id, jr.judge_name))

    # 🔥 ФИНАЛЬНЫЙ ПРЯМОЙ ВЫВОД
    print("\n" + "=" * 80)
    print("📋 ПРЯМОЙ ВЫВОД РЕЗУЛЬТАТОВ (question_id | judge | result)")
    print("=" * 80)
    for ev in evaluations:
        qid = ev.sample.question_id
        for jr in ev.judge_results:
            # Прямой вывод: ID вопроса | Судья | Оценка | Краткое обоснование
            eval_short = (
                jr.evaluation_text[:120].replace("\n", " ").strip() + "..."
                if len(jr.evaluation_text) > 120
                else jr.evaluation_text
            )
            print(f"[Q:{qid}] Judge:{jr.judge_name:15s} | Rating:{jr.total_rating}/4 | {eval_short}")
    print("=" * 80)

    # Финальное сохранение
    print("\n💾 Сохранение результатов...")
    save_results_csv(
        evaluations,
        output_dir / "judge_evaluation_results.csv",
        total_samples=total_with_skipped,
        is_incremental=False,
        append_mode=resume,
        existing_pairs=processed_pairs if resume else None,
    )
    save_results_json(
        evaluations,
        output_dir / "judge_detailed_results.json",
        total_samples=total_with_skipped,
        is_incremental=False,
        append_mode=resume,
        existing_pairs=processed_pairs if resume else None,
    )

    # Статистика
    print_statistics(evaluations)
    print(f"\n✅ Завершено: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return evaluations


if __name__ == "__main__":
    # Тестовый запуск с режимом продолжения
    run_judge_evaluation(
        max_samples=500,
        models_to_use=["qwen3.5-9b", "gpt-oss-120b"],
        resume=True,
    )
