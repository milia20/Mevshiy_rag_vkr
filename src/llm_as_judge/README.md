# LLM-as-a-Judge Пайплайн Оценки

Этот модуль предоставляет инструменты для автоматической оценки качества ответов RAG-систем с использованием больших
языковых моделей (LLM) в роли судей.

## 📁 Структура Модуля

```
llm_as_judge/
├── judge_ollama.py       # Оценка через Ollama API
├── judge_lmstudio.py     # Оценка через LM Studio SDK
├── judge_analyzer.py     # Анализ результатов и визуализация
└── README.md             # Эта документация
```

## 🎯 Поддерживаемые Модели-Судьи

- **openai/gpt-oss-120b** — мощная модель для точной оценки
- **google/gemma-4-26b-a4b** — эффективная модель от Google
- **qwen/qwen3.6-35b-a3b** — качественная китайская модель

## 📊 Метрики Оценки

Каждая модель оценивает ответы по трём метрикам с использованием дискретной шкалы 1-4:

| Метрика          | Описание              | Критерии                                                                |
|------------------|-----------------------|-------------------------------------------------------------------------|
| **Faithfulness** | Верность контексту    | Соответствие ответа предоставленному контексту, отсутствие галлюцинаций |
| **Accuracy**     | Точность ответа       | Соответствие эталонному ответу, полнота информации                      |
| **Relevance**    | Релевантность вопросу | Насколько ответ адресует заданный вопрос                                |

### Шкала Оценок

- **4 — ОТЛИЧНО**: Полное соответствие критериям
- **3 — ХОРОШО**: Небольшие недочёты
- **2 — ПЛОХО**: Существенные проблемы
- **1 — ОЧЕНЬ ПЛОХО**: Серьёзные нарушения критериев

## 🚀 Быстрый Старт

### Предварительные Требования

```bash
# Для Ollama
pip install requests pandas

# Для LM Studio
pip install lmstudio pandas

# Для анализа и графиков
pip install matplotlib seaborn pandas
```

### Запуск Оценки через Ollama

```python
from src.llm_as_judge.judge_ollama import run_judge_evaluation

# Полная оценка всех образцов всеми судьями
results = run_judge_evaluation(
    evaluation_file="/workspace/test_datasets/evaluation_results_all.csv",
    context_file="/workspace/src/all_q_True.csv",
    max_samples=None,  # None = все записи
    models_to_use=None,  # None = все модели
)

# Оценка только 10 образцов одной моделью
results = run_judge_evaluation(
    max_samples=10,
    models_to_use=["gpt-oss-120b"],
)
```

### Запуск Оценки через LM Studio

```python
from src.llm_as_judge.judge_lmstudio import run_judge_evaluation

# Убедитесь, что LM Studio запущен и модели загружены
results = run_judge_evaluation(
    max_samples=None,
    models_to_use=["gpt-oss-120b", "gemma-4-26b", "qwen3.6-35b"],
)
```

### Анализ Результатов

```python
from src.llm_as_judge.judge_analyzer import analyze_judge_results

# Полный анализ с генерацией графиков
report = analyze_judge_results(
    output_dirs=[
        "/workspace/test_datasets/judge_results_ollama",
        "/workspace/test_datasets/judge_results_lmstudio",
    ],
    analysis_output_dir="/workspace/test_datasets/judge_analysis",
    generate_plots=True,
)
```

## 📝 Формат Входных Данных

### evaluation_results_all.csv

Файл должен содержать следующие колонки:

| Колонка       | Описание                                |
|---------------|-----------------------------------------|
| `dataset`     | Название датасета                       |
| `question_id` | ID вопроса                              |
| `question`    | Текст вопроса                           |
| `answer`      | Ответ модели для оценки                 |
| `model`       | Название модели, сгенерировавшей ответ  |
| `backend`     | Бэкенд (lmstudio_sdk, ollama_sdk, etc.) |
| `reasoning`   | Рассуждение модели (опционально)        |
| `confidence`  | Уверенность модели (опционально)        |

### all_q_True.csv (Контекст)

Файл для обогащения данных контекстом:

| Колонка          | Описание                            |
|------------------|-------------------------------------|
| `dataset`        | Название датасета                   |
| `question_id`    | ID вопроса                          |
| `context`        | Контекст для оценки Faithfulness    |
| `correct_answer` | Эталонный ответ для оценки Accuracy |

## 📤 Формат Выходных Данных

### CSV Результаты

Сохраняются в `<output_dir>/judge_evaluation_results.csv`:

| Колонка                                        | Описание                 |
|------------------------------------------------|--------------------------|
| `dataset`, `question_id`, `question`, `answer` | Исходные данные          |
| `original_model`, `original_backend`           | Модель-генератор         |
| `judge_name`, `judge_model`                    | Модель-судья             |
| `faithfulness`, `accuracy`, `relevance`        | Оценки по метрикам (1-4) |
| `total_rating`                                 | Средняя оценка           |
| `evaluation_text`                              | Обоснование оценки       |
| `consensus_*`                                  | Консенсус между судьями  |

### JSON Результаты

Детальные результаты в `<output_dir>/judge_detailed_results.json`:

```json
[
  {
    "dataset": "CoSQA",
    "question_id": 0,
    "question": "...",
    "answer": "...",
    "judges": [
      {
        "name": "gpt-oss-120b",
        "faithfulness": 4,
        "accuracy": 3,
        "relevance": 4,
        "total_rating": 3.67,
        "evaluation": "..."
      }
    ],
    "consensus": {
      "faithfulness": 3.67,
      "accuracy": 3.0,
      "relevance": 4.0,
      "total": 3.56
    }
  }
]
```

## 📈 Анализ и Визуализация

### Генерируемые Графики

1. **metric_distribution.png** — Распределение оценок по каждой метрике
2. **judge_comparison.png** — Сравнение судей по средним оценкам
3. **correlation_heatmap.png** — Тепловая карта корреляций между метриками
4. **model_scores.png** — Сравнение оригинальных моделей по оценкам
5. **judge_boxplot.png** — Boxplot распределения оценок по судьям

### Таблицы

- **summary_table.csv** — Сводная статистика по всем метрикам
- **judge_comparison.csv** — Детальная статистика по каждому судье
- **model_ranking.csv** — Ранжирование моделей по качеству ответов

## 🔧 Конфигурация

### Параметры Оценки

```python
TEMPERATURE = 0.1      # Низкая температура для консистентности
MAX_TOKENS = 256       # Максимум токенов в ответе судьи (достаточно для оценок 1-4 + обоснование)
TIMEOUT_SECONDS = 120  # Таймаут запроса к API
```

### Автоматический Расчёт Длины Контекста

Модуль автоматически анализирует входные данные (`evaluation_results_all.csv` и `all_q_True.csv`) для определения
оптимального размера контекстного окна:

- **LM Studio**: параметр `contextLength` в `LlmLoadModelConfigDict`
- **Ollama**: параметр `num_ctx` в опциях генерации

Расчёт учитывает:

1. Максимальную комбинацию question+answer+context
2. Запас ~1000 символов для промпта судьи
3. Конвертацию в токены (~3.5 символа на токен)
4. Округление до ближайшей степени 2 (минимум 8192)

**Рекомендация**: Для ваших данных с максимальной длиной ~15K символов будет установлено окно 16384 токена.

### Промпты

Модуль использует три специализированных промпта для каждой метрики.
Все промпты требуют структурированный вывод в формате:

```
Evaluation: [обоснование на 2-4 предложения]
Total rating: [число от 1 до 4]
```

### Обработка Ошибок Парсинга

- Извлечение оценки через regex `Total rating:\s*(\d+)`
- fallback на поиск чисел 1-4 в тексте
- Логирование ошибок парсинга для ручной проверки

## 🎯 Принципы Проектирования

1. **Чёткое определение задачи**: Каждая метрика оценивается отдельно
2. **Дискретная шкала**: 1-4 вместо непрерывной шкалы для стабильности
3. **Структурированный вывод**: Обязательное разделение на Evaluation и Total rating
4. **Калибровка**: Рекомендуется проверка корреляции с экспертными оценками (целевой порог ≥0.8)
5. **Учёт морфологии**: Промпты принимают семантические эквиваленты

## ⚠️ Частые Ошибки и Решения

| Ошибка                   | Последствие           | Решение                           |
|--------------------------|-----------------------|-----------------------------------|
| Нечёткая шкала оценок    | Высокий шум           | Использовать дискретную шкалу 1-4 |
| Отсутствие обоснования   | Невозможность отладки | Требовать поле Evaluation         |
| Оценка «всего сразу»     | Смешение аспектов     | Разделять метрики                 |
| Игнорирование калибровки | Смещение оценок       | Проверять корреляцию с экспертами |
| Парсинг без обработки    | Падение пайплайна     | Добавлять fallback-логику         |

## 📚 Рекомендации

### Для Русскоязычной Документации

- **Язык судьи**: Используйте ту же языковую модель, что и для генерации
- **Морфология**: Промпты учитывают разные формы слов
- **Калибровка**: Привлекайте русскоязычных экспертов для разметки

### Оптимизация Затрат

- Кэширование оценок для повторяющихся троек (question, context, answer)
- Выборочная оценка: применять судью только к топ-1 ответу
- Использование более лёгких моделей для предварительной фильтрации

## 📋 Пример Полного Пайплайна

```python
# 1. Запуск оценки через Ollama
from src.llm_as_judge.judge_ollama import run_judge_evaluation

ollama_results = run_judge_evaluation(
    max_samples=100,
    models_to_use=["gpt-oss-120b", "gemma-4-26b"],
)

# 2. Запуск оценки через LM Studio
from src.llm_as_judge.judge_lmstudio import run_judge_evaluation

lmstudio_results = run_judge_evaluation(
    max_samples=100,
    models_to_use=["qwen3.6-35b"],
)

# 3. Анализ всех результатов
from src.llm_as_judge.judge_analyzer import analyze_judge_results

report = analyze_judge_results(
    output_dirs=[
        "/workspace/test_datasets/judge_results_ollama",
        "/workspace/test_datasets/judge_results_lmstudio",
    ],
    generate_plots=True,
)

# 4. Интерпретация результатов
print(f"Средняя согласованность судей: {report['consensus_stats']}")
```

## 🔗 Полезные Ссылки

- [RAGAS Documentation](https://docs.ragas.io)
- [DeepEval](https://deepeval.com)
- [TruLens](https://github.com/truera/trulens)
- [LangSmith](https://docs.langchain.com)

## 📄 Лицензия

Модуль распространяется под лицензией проекта.
