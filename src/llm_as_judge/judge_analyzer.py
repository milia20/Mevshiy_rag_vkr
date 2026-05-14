"""
judge_analyzer.py
Анализ результатов LLM-as-a-Judge оценки.

Читает сохранённые данные из CSV/JSON и строит графики и таблицы.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib/seaborn не установлены. Графики не будут построены.")
    print("   Установите: pip install matplotlib seaborn")


# ==================== КОНФИГУРАЦИЯ ====================

DEFAULT_OUTPUT_DIRS = [
    Path("/workspace/test_datasets/judge_results_ollama"),
    Path("/workspace/test_datasets/judge_results_lmstudio"),
]

OUTPUT_ANALYSIS_DIR = Path("/workspace/test_datasets/judge_analysis")


# ==================== ЗАГРУЗКА ДАННЫХ ====================


def load_csv_results(filepath: Path) -> pd.DataFrame:
    """Загружает результаты из CSV файла."""
    if not filepath.exists():
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    print(f"✓ Загружено {len(df)} записей из {filepath}")
    return df


def load_json_results(filepath: Path) -> list[dict[str, Any]]:
    """Загружает детальные результаты из JSON файла."""
    if not filepath.exists():
        return []

    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    print(f"✓ Загружено {len(data)} образцов из {filepath}")
    return data


def find_results_files(output_dirs: list[Path]) -> dict[str, list[Path]]:
    """Находит все файлы с результатами в указанных директориях."""
    found_files = {"csv": [], "json": []}

    for dir_path in output_dirs:
        if not dir_path.exists():
            continue

        # Ищем CSV файлы
        csv_files = list(dir_path.glob("*.csv"))
        found_files["csv"].extend(csv_files)

        # Ищем JSON файлы
        json_files = list(dir_path.glob("*.json"))
        found_files["json"].extend(json_files)

    return found_files


# ==================== АНАЛИТИКА ====================


def calculate_basic_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Вычисляет базовую статистику по оценкам."""
    metrics = ["faithfulness", "accuracy", "relevance", "total_rating"]
    stats = {}

    for metric in metrics:
        if metric not in df.columns:
            continue

        values = df[metric].dropna()
        if len(values) == 0:
            continue

        stats[metric] = {
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "max": values.max(),
            "median": values.median(),
            "count": len(values),
        }

    return stats


def calculate_judge_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Статистика по каждому судье."""
    if "judge_name" not in df.columns:
        return pd.DataFrame()

    judge_stats = (
        df.groupby("judge_name")
        .agg(
            {
                "faithfulness": ["mean", "std", "count"],
                "accuracy": ["mean", "std"],
                "relevance": ["mean", "std"],
                "total_rating": ["mean", "std", "min", "max"],
            }
        )
        .round(3)
    )

    return judge_stats


def calculate_consensus_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Статистика согласованности между судьями."""
    if "question_id" not in df.columns or "judge_name" not in df.columns:
        return {}

    # Группируем по вопросам
    question_groups = df.groupby(["dataset", "question_id"])

    agreements = {"faithfulness": [], "accuracy": [], "relevance": [], "total_rating": []}

    for (_, _), group in question_groups:
        if len(group) < 2:
            continue

        for metric in agreements.keys():
            if metric not in group.columns:
                continue

            values = group[metric].dropna()
            if len(values) >= 2:
                # Стандартное отклонение как мера несогласованности
                std = values.std()
                agreements[metric].append(std)

    consensus_stats = {}
    for metric, stds in agreements.items():
        if stds:
            consensus_stats[metric] = {
                "avg_disagreement": sum(stds) / len(stds),
                "max_disagreement": max(stds),
                "min_disagreement": min(stds),
            }

    return consensus_stats


def calculate_model_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Сравнение оригинальных моделей по оценкам судей."""
    if "original_model" not in df.columns:
        return pd.DataFrame()

    model_stats = (
        df.groupby("original_model")
        .agg(
            {
                "faithfulness": ["mean", "std"],
                "accuracy": ["mean", "std"],
                "relevance": ["mean", "std"],
                "total_rating": ["mean", "std", "count"],
            }
        )
        .round(3)
    )

    return model_stats


# ==================== ГРАФИКИ ====================


def plot_metric_distribution(df: pd.DataFrame, save_path: Path) -> None:
    """График распределения оценок по метрикам."""
    if not MATPLOTLIB_AVAILABLE:
        return

    metrics = ["faithfulness", "accuracy", "relevance", "total_rating"]
    available_metrics = [m for m in metrics if m in df.columns]

    if not available_metrics:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(available_metrics[:4]):
        ax = axes[idx]
        values = df[metric].dropna()

        # Гистограмма
        ax.hist(
            values, bins=range(1, 6) if metric != "total_rating" else 10, alpha=0.7, edgecolor="black", color="skyblue"
        )
        ax.axvline(values.mean(), color="red", linestyle="--", linewidth=2, label=f"Среднее: {values.mean():.2f}")
        ax.set_xlabel("Оценка")
        ax.set_ylabel("Количество")
        ax.set_title(f"Распределение: {metric.capitalize()}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ График сохранён: {save_path}")


def plot_judge_comparison(df: pd.DataFrame, save_path: Path) -> None:
    """Сравнение судей по метрикам."""
    if not MATPLOTLIB_AVAILABLE:
        return

    if "judge_name" not in df.columns:
        return

    metrics = ["faithfulness", "accuracy", "relevance"]
    available_metrics = [m for m in metrics if m in df.columns]

    if not available_metrics:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(available_metrics))
    width = 0.25

    judges = df["judge_name"].unique()
    colors = plt.cm.Set3(range(len(judges)))

    for i, (judge, color) in enumerate(zip(judges, colors, strict=False)):
        judge_data = df[df["judge_name"] == judge]
        means = [judge_data[m].mean() for m in available_metrics]

        offset = (i - len(judges) / 2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], means, width, label=judge, color=color, edgecolor="black")

    ax.set_xlabel("Метрика")
    ax.set_ylabel("Средняя оценка")
    ax.set_title("Сравнение судей по метрикам")
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in available_metrics])
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ График сохранён: {save_path}")


def plot_heatmap_correlation(df: pd.DataFrame, save_path: Path) -> None:
    """Тепловая карта корреляций между метриками."""
    if not MATPLOTLIB_AVAILABLE:
        return

    metrics = ["faithfulness", "accuracy", "relevance", "total_rating"]
    available_metrics = [m for m in metrics if m in df.columns]

    if len(available_metrics) < 2:
        return

    # Вычисляем корреляцию
    corr_matrix = df[available_metrics].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".3f", square=True, linewidths=0.5, ax=ax)
    ax.set_title("Корреляция между метриками")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ График сохранён: {save_path}")


def plot_score_by_model(df: pd.DataFrame, save_path: Path) -> None:
    """График средних оценок по оригинальным моделям."""
    if not MATPLOTLIB_AVAILABLE:
        return

    if "original_model" not in df.columns:
        return

    # Агрегируем по моделям
    model_stats = (
        df.groupby("original_model")
        .agg(
            {
                "faithfulness": "mean",
                "accuracy": "mean",
                "relevance": "mean",
                "total_rating": "mean",
            }
        )
        .reset_index()
    )

    if len(model_stats) == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    x = range(len(model_stats))
    width = 0.2

    metrics = ["faithfulness", "accuracy", "relevance", "total_rating"]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]

    for i, (metric, color) in enumerate(zip(metrics, colors, strict=False)):
        if metric not in model_stats.columns:
            continue
        offset = (i - len(metrics) / 2 + 0.5) * width
        ax.bar(
            [xi + offset for xi in x],
            model_stats[metric],
            width,
            label=metric.capitalize(),
            color=color,
            edgecolor="black",
            alpha=0.8,
        )

    ax.set_xlabel("Модель")
    ax.set_ylabel("Средняя оценка")
    ax.set_title("Сравнение моделей по средним оценкам судей")
    ax.set_xticks(x)

    # Поворачиваем подписи моделей для читаемости
    model_labels = model_stats["original_model"].str.slice(-30)  # Обрезаем длинные названия
    ax.set_xticklabels(model_labels, rotation=45, ha="right")

    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ График сохранён: {save_path}")


def plot_boxplot_by_judge(df: pd.DataFrame, save_path: Path) -> None:
    """Boxplot распределения оценок по судьям."""
    if not MATPLOTLIB_AVAILABLE:
        return

    if "judge_name" not in df.columns or "total_rating" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    data_to_plot = []
    labels = []

    for judge in df["judge_name"].unique():
        judge_data = df[df["judge_name"] == judge]["total_rating"].dropna()
        if len(judge_data) > 0:
            data_to_plot.append(judge_data.values)
            labels.append(judge)

    if not data_to_plot:
        return

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

    # Раскрашиваем боксы
    colors = plt.cm.Pastel1(range(len(data_to_plot)))
    for patch, color in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(color)

    ax.set_xlabel("Судья")
    ax.set_ylabel("Оценка (Total Rating)")
    ax.set_title("Распределение оценок по судьям")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ График сохранён: {save_path}")


# ==================== ТАБЛИЦЫ ====================


def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Создаёт сводную таблицу статистики."""
    basic_stats = calculate_basic_stats(df)

    summary_data = []
    for metric, stats in basic_stats.items():
        summary_data.append(
            {
                "Metric": metric,
                "Mean": f"{stats['mean']:.3f}",
                "Std": f"{stats['std']:.3f}",
                "Min": stats["min"],
                "Max": stats["max"],
                "Median": f"{stats['median']:.3f}",
                "Count": stats["count"],
            }
        )

    return pd.DataFrame(summary_data)


def create_judge_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Таблица сравнения судей."""
    judge_stats = calculate_judge_stats(df)
    return judge_stats


def create_model_ranking_table(df: pd.DataFrame) -> pd.DataFrame:
    """Таблица ранжирования моделей."""
    if "original_model" not in df.columns:
        return pd.DataFrame()

    model_stats = (
        df.groupby("original_model")
        .agg(
            {
                "total_rating": ["mean", "std", "count"],
                "faithfulness": "mean",
                "accuracy": "mean",
                "relevance": "mean",
            }
        )
        .round(3)
    )

    # Сортируем по среднему total_rating
    model_stats = model_stats.sort_values(("total_rating", "mean"), ascending=False)

    return model_stats


# ==================== ГЛАВНАЯ ФУНКЦИЯ ====================


def analyze_judge_results(
    output_dirs: list[Path] = DEFAULT_OUTPUT_DIRS,
    analysis_output_dir: Path = OUTPUT_ANALYSIS_DIR,
    generate_plots: bool = True,
) -> dict[str, Any]:
    """
    Полный анализ результатов LLM-as-a-Judge оценки.

    Args:
        output_dirs: Директории с результатами оценки
        analysis_output_dir: Директория для сохранения анализа
        generate_plots: Генерировать ли графики

    Returns:
        Словарь с результатами анализа
    """
    print("\n" + "=" * 60)
    print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ LLM-AS-A-JUDGE")
    print("=" * 60)

    # Создание директории вывода
    analysis_output_dir.mkdir(parents=True, exist_ok=True)

    # Поиск файлов с результатами
    print("\n🔍 Поиск файлов с результатами...")
    found_files = find_results_files(output_dirs)

    if not found_files["csv"] and not found_files["json"]:
        print("❌ Файлы с результатами не найдены!")
        return {}

    print(f"Найдено CSV файлов: {len(found_files['csv'])}")
    print(f"Найдено JSON файлов: {len(found_files['json'])}")

    # Загрузка и объединение данных
    print("\n📥 Загрузка данных...")
    all_dfs = []

    for csv_file in found_files["csv"]:
        if "temp" not in csv_file.name:  # Пропускаем временные файлы
            df = load_csv_results(csv_file)
            if not df.empty:
                df["source_file"] = csv_file.name
                all_dfs.append(df)

    if not all_dfs:
        print("❌ Нет данных для анализа!")
        return {}

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n✓ Объединено {len(combined_df)} записей")

    # Базовая статистика
    print("\n📈 Вычисление статистики...")
    basic_stats = calculate_basic_stats(combined_df)
    judge_stats = calculate_judge_stats(combined_df)
    consensus_stats = calculate_consensus_stats(combined_df)
    model_stats = calculate_model_comparison(combined_df)

    # Вывод статистики
    print("\n" + "=" * 60)
    print("📋 БАЗОВАЯ СТАТИСТИКА")
    print("=" * 60)
    for metric, stats in basic_stats.items():
        print(f"\n{metric.upper()}:")
        print(f"  Среднее: {stats['mean']:.3f} ± {stats['std']:.3f}")
        print(f"  Диапазон: [{stats['min']}, {stats['max']}]")
        print(f"  Медиана: {stats['median']:.3f}")

    if not judge_stats.empty:
        print("\n" + "=" * 60)
        print("📋 СТАТИСТИКА ПО СУДЬЯМ")
        print("=" * 60)
        print(judge_stats.to_string())

    if consensus_stats:
        print("\n" + "=" * 60)
        print("📋 СОГЛАСОВАННОСТЬ СУДЕЙ")
        print("=" * 60)
        for metric, stats in consensus_stats.items():
            print(f"\n{metric.upper()}:")
            print(f"  Среднее расхождение: {stats['avg_disagreement']:.3f}")
            print(f"  Макс. расхождение: {stats['max_disagreement']:.3f}")
            print(f"  Мин. расхождение: {stats['min_disagreement']:.3f}")

    # Сохранение таблиц
    print("\n💾 Сохранение таблиц...")

    summary_table = create_summary_table(combined_df)
    summary_table.to_csv(analysis_output_dir / "summary_table.csv", index=False)
    print(f"✓ Сводная таблица: {analysis_output_dir / 'summary_table.csv'}")

    if not judge_stats.empty:
        judge_stats.to_csv(analysis_output_dir / "judge_comparison.csv")
        print(f"✓ Сравнение судей: {analysis_output_dir / 'judge_comparison.csv'}")

    if not model_stats.empty:
        model_stats.to_csv(analysis_output_dir / "model_ranking.csv")
        print(f"✓ Ранжирование моделей: {analysis_output_dir / 'model_ranking.csv'}")

    # Генерация графиков
    if generate_plots and MATPLOTLIB_AVAILABLE:
        print("\n📊 Генерация графиков...")

        plot_metric_distribution(combined_df, analysis_output_dir / "metric_distribution.png")
        plot_judge_comparison(combined_df, analysis_output_dir / "judge_comparison.png")
        plot_heatmap_correlation(combined_df, analysis_output_dir / "correlation_heatmap.png")
        plot_score_by_model(combined_df, analysis_output_dir / "model_scores.png")
        plot_boxplot_by_judge(combined_df, analysis_output_dir / "judge_boxplot.png")

    # Сохранение полного отчёта в JSON
    report = {
        "basic_stats": basic_stats,
        "consensus_stats": consensus_stats,
        "total_samples": len(combined_df),
        "judges_count": len(combined_df["judge_name"].unique()) if "judge_name" in combined_df.columns else 0,
        "models_count": len(combined_df["original_model"].unique()) if "original_model" in combined_df.columns else 0,
    }

    with open(analysis_output_dir / "analysis_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Отчёт сохранён: {analysis_output_dir / 'analysis_report.json'}")

    print("\n" + "=" * 60)
    print("✅ АНАЛИЗ ЗАВЕРШЁН")
    print("=" * 60)
    print(f"Результаты сохранены в: {analysis_output_dir}")

    return report


if __name__ == "__main__":
    # Пример запуска
    analyze_judge_results(
        generate_plots=True,
    )
