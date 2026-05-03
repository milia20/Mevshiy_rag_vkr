"""
llm_as_judge.py
Оценка качества ответов RAG-системы с использованием:
• Вопросов из JSONL-файла
• Контекста из Qdrant
• Генерации ответов через LLM
• Оценки по шкале: точно / неточно / неправильно
"""

import json
import traceback
from pathlib import Path
from typing import Any

import jsonlines
import openai
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

# ==================== КОНФИГУРАЦИЯ ====================

# LM Studio API
BASE_URL = "http://localhost:1234/v1"
API_KEY = "lm-studio"

JUDGE_MODELS = [
    {"name": "Судья 1 (qwen3.5-35b)", "model_id": "qwen/qwen3.5-35b-a3b"},
    {"name": "Судья 2 (gemma-3-12b)", "model_id": "google/gemma-3-12b"},
]
GENERATOR_MODEL = "qwen/qwen3.5-35b-a3b"

# Qdrant
QDRANT_HOST = "http://localhost:6333"
COLLECTION_NAME = "benchmark_hybrid"  # Коллекция с данными
SEARCH_METHOD = "hybrid_rrf"  # dense | sparse | hybrid_rrf | hybrid_dbsf
TOP_K = 5  # Количество документов для контекста

# Тестовые данные
QUERIES_FILE = "./custom_dataset/datasets/qna_agent-framework_en.jsonl"
MAX_QUERIES = 20  # Ограничение для быстрого теста

# Эмбеддинг-модели (должны совпадать с теми, что использовались при индексации)
DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL_NAME = "prithivida/Splade_PP_en_v1"


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================


def call_llm(model_id: str, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 1024) -> str:
    """Вызов LLM через LM Studio API"""
    client = openai.OpenAI(base_url=BASE_URL, api_key=API_KEY)

    try:
        response = client.chat.completions.create(
            model=model_id, messages=messages, temperature=temperature, max_tokens=max_tokens, timeout=120
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"❌ Ошибка: {e!s}"


def load_test_queries(filepath: str, max_queries: int | None = None) -> list[dict]:
    """Загружает тестовые запросы из JSONL"""

    queries = []
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {filepath}")

    with jsonlines.open(path, mode="r") as reader:
        for i, line in enumerate(reader):
            if max_queries and i >= max_queries:
                break
            if line.get("question") and line.get("file"):
                queries.append(
                    {
                        "question": line.get("question", ""),
                        "expected_answer": line.get("answer", ""),
                        "expected_file": line.get("file", ""),
                        "metadata": {k: v for k, v in line.items() if k not in ["question", "answer", "file"]},
                    }
                )

    print(f"✓ Загружено {len(queries)} тестовых запросов")
    return queries


def _get_embedding_models():
    """Ленивая загрузка моделей эмбеддингов"""

    return TextEmbedding(model_name=DENSE_MODEL_NAME), SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


def search_qdrant(client: QdrantClient, query: str, collection: str, method: str, top_k: int = 5) -> list[dict]:
    """
    Поиск в Qdrant с поддержкой разных стратегий.
    Возвращает список документов с текстом и метаданными.
    """

    try:
        if method == "dense":
            dense_model, _ = _get_embedding_models()
            query_vec = next(iter(dense_model.embed([query])))
            res = client.query_points(
                collection_name=collection,
                query=query_vec.tolist(),
                limit=top_k,
            )
            points = res.points

        elif method == "sparse":
            _, sparse_model = _get_embedding_models()
            query_vec = next(iter(sparse_model.embed([query])))
            res = client.query_points(
                collection_name=collection,
                query=models.SparseVector(indices=query_vec.indices.tolist(), values=query_vec.values.tolist()),
                using="bm25",
                limit=top_k,
            )
            points = res.points

        elif method in ["hybrid_rrf", "hybrid_dbsf"]:
            dense_model, sparse_model = _get_embedding_models()
            query_dense = next(iter(dense_model.embed([query])))
            query_sparse = next(iter(sparse_model.embed([query])))

            if method == "hybrid_rrf":
                query_param = models.RrfQuery(rrf=models.Rrf(k=30))
            else:
                query_param = models.FusionQuery(fusion=models.Fusion.DBSF)

            res = client.query_points(
                collection_name=collection,
                query=query_param,
                prefetch=[
                    models.Prefetch(query=query_dense.tolist(), using="dense", limit=top_k * 2),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=query_sparse.indices.tolist(), values=query_sparse.values.tolist()
                        ),
                        using="bm25",
                        limit=top_k * 2,
                    ),
                ],
                limit=top_k,
            )
            points = res.points
        else:
            print(f"⚠ Неизвестный метод поиска: {method}, используем dense")
            dense_model, _ = _get_embedding_models()
            query_vec = next(iter(dense_model.embed([query])))
            res = client.query_points(
                collection_name=collection,
                query=query_vec.tolist(),
                limit=top_k,
            )
            points = res.points

    except Exception as e:
        print(f"⚠ Ошибка поиска в Qdrant: {e}")
        return []

    # Извлекаем полезную информацию из точек
    documents = []
    for point in points:
        payload = point.payload or {}
        # Пробуем разные поля для текста
        doc_text = payload.get("text") or payload.get("answer") or payload.get("content") or ""
        if doc_text:
            documents.append(
                {
                    "text": doc_text[:2000],  # Ограничиваем длину контекста
                    "file": payload.get("file", "unknown"),
                    "title": payload.get("title", ""),
                    "score": getattr(point, "score", None),
                }
            )

    return documents


def generate_answer_with_context(
    question: str, context_docs: list[dict], model_id: str, temperature: float = 0.3
) -> str:
    """Генерирует ответ на вопрос, используя документы как контекст"""

    # Формируем контекст из найденных документов
    context_parts = []
    for i, doc in enumerate(context_docs[:TOP_K], 1):
        source = doc.get("title") or doc.get("file") or f"Источник {i}"
        context_parts.append(f"【{source}】\n{doc['text']}")

    context_text = "\n\n".join(context_parts)

    system_prompt = """Ты — эксперт-ассистент, отвечающий на вопросы ТОЛЬКО на основе предоставленного контекста.

ПРАВИЛА:
1. Используй исключительно информацию из документов ниже
2. Если в контексте нет ответа — честно скажи "В предоставленных документах нет информации"
3. Не выдумывай факты, даты, имена или события
4. Будь кратким, но давай полный ответ на вопрос
5. Если документы противоречат друг другу — укажи на это

Формат: прямой ответ на вопрос, без вступлений типа "на основе контекста"."""

    user_prompt = f"""ВОПРОС: {question}

КОНТЕКСТ:
{context_text}

ОТВЕТ:"""

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    return call_llm(model_id, messages, temperature=temperature, max_tokens=512)


def evaluate_answer_3class(question: str, generated: str, reference: str, judge_config: dict) -> dict[str, Any]:
    """
    Оценивает ответ по трёхклассовой шкале:

    🟢 ТОЧНО — полное соответствие эталону по фактам
    🟡 НЕТочно — верно по сути, но есть неточности/неполнота
    🔴 НЕПРАВИЛЬНО — фактические ошибки или нерелевантность
    """

    system_prompt = """Ты — строгий эксперт-оценщик качества ответов.

КЛАССИФИКАЦИЯ:

🟢 ТОЧНО если:
• Все ключевые факты из эталона присутствуют и верны
• Нет выдуманных фактов или искажений
• Ответ полный и прямо отвечает на вопрос

🟡 НЕТОЧНО если:
• Основная мысль верна, но есть мелкие неточности
• Ответ правильный, но неполный (упущены детали)
• Есть незначительные формулировочные расхождения

🔴 НЕПРАВИЛЬНО если:
• Содержит фактические ошибки или противоречия эталону
• Есть галлюцинации (выдуманные факты, даты, имена)
• Ответ не релевантен вопросу или уклоняется от темы

Верни ТОЛЬКО JSON:
{
    "verdict": "точно" | "неточно" | "неправильно",
    "confidence": 0.0-1.0,
    "reasoning": "1-2 предложения обоснования",
    "key_issues": ["список проблем"] или []
}"""

    user_prompt = f"""=== ВОПРОС ===
{question}

=== ЭТАЛОННЫЙ ОТВЕТ ===
{reference}

=== ОТВЕТ ДЛЯ ОЦЕНКИ ===
{generated}

Оцени ответ по трёхклассовой шкале."""

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    raw = call_llm(judge_config["model_id"], messages, temperature=0.1, max_tokens=256)

    # Парсинг JSON с фоллбэком
    try:
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start != -1 and end != -1:
            result = json.loads(raw[start:end])
        else:
            raise ValueError("No JSON found")
    except Exception as e:
        print(e)
        result = {
            "verdict": "неправильно",
            "confidence": 0.0,
            "reasoning": f"Ошибка парсинга: {raw[:100]}",
            "key_issues": ["Не удалось распарсить ответ судьи"],
        }

    # Валидация verdict
    if result.get("verdict") not in ["точно", "неточно", "неправильно"]:
        result["verdict"] = "неправильно"
        result["key_issues"] = (result.get("key_issues") or []) + ["Неверный формат вердикта"]

    return result


def print_statistics(results: list[dict]) -> None:
    """Выводит детальную статистику оценки"""

    print("\n" + "═" * 70)
    print("📊 СТАТИСТИКА ОЦЕНКИ ОТВЕТОВ")
    print("═" * 70)

    if not results:
        print("⚠ Нет данных для анализа")
        return

    total = len(results)

    # Агрегация по вердиктам
    verdicts = {"точно": 0, "неточно": 0, "неправильно": 0}
    for r in results:
        v = r.get("evaluation", {}).get("verdict", "неправильно")
        verdicts[v] = verdicts.get(v, 0) + 1

    # По судьям
    by_judge = {}
    for r in results:
        jname = r.get("judge", {}).get("name", "Unknown")
        verdict = r.get("evaluation", {}).get("verdict", "неправильно")
        if jname not in by_judge:
            by_judge[jname] = {"точно": 0, "неточно": 0, "неправильно": 0}
        by_judge[jname][verdict] += 1

    # По файлам-источникам
    by_file = {}
    for r in results:
        f = r.get("expected_file", "unknown")
        verdict = r.get("evaluation", {}).get("verdict", "неправильно")
        if f not in by_file:
            by_file[f] = {"точно": 0, "неточно": 0, "неправильно": 0}
        by_file[f][verdict] += 1

    # === ОБЩАЯ ТАБЛИЦА ===
    print(f"\n📈 Всего оценено: {total} ответов")
    print("\n" + "─" * 50)
    print(f"{'Вердикт':<15} {'Кол-во':>8} {'Доля':>10}")
    print("─" * 50)

    for v, emoji in [("точно", "🟢"), ("неточно", "🟡"), ("неправильно", "🔴")]:
        count = verdicts[v]
        pct = count / total * 100
        bar = "█" * int(pct / 5)
        print(f"{emoji} {v:<12} {count:>8} {pct:>6.1f}% {bar}")

    # === ПО СУДЬЯМ ===
    if len(by_judge) > 1:
        print("\n👥 Оценка по судьям:")
        print("─" * 50)
        for jname, stats in by_judge.items():
            acc = stats["точно"] / total * 100 if total > 0 else 0
            print(f"\n{jname}:")
            print(f"   🟢 Точно: {stats['точно']} ({stats['точно'] / sum(stats.values()) * 100:.1f}%)")
            print(f"   🟡 Неточно: {stats['неточно']} | 🔴 Неправильно: {stats['неправильно']}")

    # === СОГЛАСОВАННОСТЬ СУДЕЙ ===
    judge_count = 2
    if len(by_judge) >= judge_count:
        print("\n🤝 Согласованность судей:")
        print("─" * 50)

        # Группируем результаты по вопросам
        by_question = {}
        for r in results:
            q = r.get("question", "")
            if q not in by_question:
                by_question[q] = {}
            jname = r.get("judge", {}).get("name", "")
            by_question[q][jname] = r.get("evaluation", {}).get("verdict")

        # Считаем совпадения
        judges = list(by_judge.keys())
        if len(judges) >= judge_count and by_question:
            matches, total_pairs = 0, 0
            for verdicts_q in by_question.values():
                if len(verdicts_q) >= judge_count:
                    vals = list(verdicts_q.values())
                    if vals[0] == vals[1]:
                        matches += 1
                    total_pairs += 1

            if total_pairs > 0:
                agreement = matches / total_pairs * 100
                print(f"   Совпадение вердиктов: {agreement:.1f}% ({matches}/{total_pairs})")
                top = 80
                med = 60
                if agreement >= top:
                    print("   ✅ Высокая согласованность")
                elif agreement >= med:
                    print("   ⚠ Средняя согласованность")
                else:
                    print("   ❌ Низкая согласованность — пересмотрите критерии")

    # === ПО ФАЙЛАМ-ИСТОЧНИКАМ ===
    print("\n📁 Качество по файлам-источникам:")
    print("─" * 50)
    for fname, stats in sorted(by_file.items(), key=lambda x: sum(x[1].values()), reverse=True)[:10]:
        total_f = sum(stats.values())
        acc = stats["точно"] / total_f * 100 if total_f > 0 else 0
        print(f"• {fname}: 🟢{stats['точно']} 🟡{stats['неточно']} 🔴{stats['неправильно']} (точность: {acc:.1f}%)")

    print("\n" + "═" * 70)


def export_results(results: list[dict], filepath: str = "judge_results.json"):
    """Экспортирует результаты в JSON для анализа"""
    # export_data = []
    # Делаем сериализуемую копию
    export_data = [
        {
            "question": r.get("question", "")[:300],
            "expected_file": r.get("expected_file"),
            "context_files": r.get("context_files", []),
            "generated_answer": r.get("generated_answer", "")[:500],
            "judge": r.get("judge", {}).get("name"),
            "evaluation": r.get("evaluation", {}),
        }
        for r in results
    ]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    print(f"💾 Результаты сохранены в {filepath}")


def main():
    """
    Полный пайплайн оценки:
    1. Загружает вопросы из JSONL
    2. Для каждого: поиск в Qdrant → генерация ответа → оценка
    3. Выводит статистику и экспортирует результаты
    """

    print("🚀 LLM-as-a-Judge: оценка ответов RAG-системы")
    print(f"🔗 Qdrant: {QDRANT_HOST} | Коллекция: {COLLECTION_NAME}")
    print(f"🔍 Поиск: {SEARCH_METHOD} (top_k={TOP_K})")
    print(f"⚖️  Судьи: {[j['name'] for j in JUDGE_MODELS]}")
    print("═" * 70)

    qdrant_client = QdrantClient(QDRANT_HOST)

    if not qdrant_client.collection_exists(COLLECTION_NAME):
        print(f"❌ Коллекция '{COLLECTION_NAME}' не найдена!")
        print("💡 Запустите сначала create_benchmark_collections.py")
        return False

    # 2. Загрузка вопросов
    try:
        queries = load_test_queries(QUERIES_FILE, max_queries=MAX_QUERIES)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 Укажите правильный путь к файлу с вопросами")
        return False

    if not queries:
        print("❌ Нет тестовых запросов для оценки")
        return False

    all_results = []

    for i, qdata in enumerate(queries, 1):
        print(f"\n🔄 [{i:2d}/{len(queries)}] {qdata['question'][:70]}...")

        print("   🔍 Поиск документов...")
        context = search_qdrant(
            client=qdrant_client, query=qdata["question"], collection=COLLECTION_NAME, method=SEARCH_METHOD, top_k=TOP_K
        )

        if not context:
            print("   ⚠ Нет найденных документов — пропускаем")
            continue
        print(f"   ✓ Найдено: {len(context)} док. | Файлы: {[d['file'] for d in context[:3]]}")

        # ── Генерация ответа ──
        print("   ✍️  Генерация ответа...")
        generated = generate_answer_with_context(
            question=qdata["question"], context_docs=context, model_id=GENERATOR_MODEL, temperature=0.2
        )
        print(f"   ✓ {generated[:120]}...")

        # ── Оценка каждым судьёй ──
        for judge in JUDGE_MODELS:
            print(f"   ⚖️  {judge['name']}...")
            evaluation = evaluate_answer_3class(
                question=qdata["question"],
                generated=generated,
                reference=qdata.get("expected_answer", ""),
                judge_config=judge,
            )

            verdict = evaluation.get("verdict", "N/A")
            emoji = {"точно": "🟢", "неточно": "🟡", "неправильно": "🔴"}.get(verdict, "⚪")
            print(f"      {emoji} {verdict} | {evaluation.get('reasoning', '')[:50]}")

            all_results.append(
                {
                    "question": qdata["question"],
                    "expected_file": qdata["expected_file"],
                    "expected_answer": qdata.get("expected_answer", ""),
                    "generated_answer": generated,
                    "context_files": [d["file"] for d in context],
                    "context_scores": [d.get("score") for d in context],
                    "judge": judge,
                    "evaluation": evaluation,
                }
            )

        # time.sleep(0.5) # Пауза для стабильности API

    # 4. Статистика и экспорт
    print_statistics(all_results)
    export_results(all_results)

    # 5. Итоговая рекомендация
    total = len(all_results)
    if total > 0:
        exact = sum(1 for r in all_results if r["evaluation"].get("verdict") == "точно")
        accuracy = exact / total * 100

        print("\n🎯 ИТОГ:")
        if accuracy >= 70:
            print("   ✅ Система показывает высокое качество ответов")
        elif accuracy >= 40:
            print("   ⚠ Качество удовлетворительное, есть пространство для улучшения")
        else:
            print("   ❌ Низкое качество — проверьте контекст, промпты или модели")
        print(f"   Доля точных ответов: {accuracy:.1f}% ({exact}/{total})")

    return True


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⛔ Прервано пользователем")
        exit(130)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")

        traceback.print_exc()
        exit(1)
