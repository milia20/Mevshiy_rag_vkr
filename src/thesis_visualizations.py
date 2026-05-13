import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.thesis_analysis_report import load_results, pareto_frontier, pick_best_by_method


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def plot_precision_bars(best: pd.DataFrame, out: Path) -> None:
    metrics = [c for c in best.columns if c.startswith("Precision@")]
    if not metrics:
        return

    df = best[["method"] + metrics].melt(id_vars="method", var_name="k", value_name="precision")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="k", y="precision", hue="method")
    plt.title("Precision@k by method (best config per method)")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_qps_vs_recall(best: pd.DataFrame, out: Path) -> None:
    if "QPS" not in best.columns:
        return

    recall_col = "Recall@10" if "Recall@10" in best.columns else None
    if recall_col is None:
        # fallback: use Precision@10
        recall_col = "Precision@10" if "Precision@10" in best.columns else None
    if recall_col is None:
        return

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=best, x="QPS", y=recall_col, hue="method", s=90)
    plt.title(f"QPS vs {recall_col} (best config per method)")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_latency_hist(df: pd.DataFrame, out: Path) -> None:
    if "avg_latency_ms" not in df.columns:
        return
    tmp = df.dropna(subset=["avg_latency_ms"]).copy()
    if tmp.empty:
        return

    plt.figure(figsize=(8, 5))
    sns.histplot(data=tmp, x="avg_latency_ms", hue="method", element="step", stat="density", common_norm=False)
    plt.title("Latency distribution (avg per-query latency from aggregated runs)")
    plt.xlabel("avg_latency_ms")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_param_heatmap_hnsw(df: pd.DataFrame, out: Path, metric: str = "Precision@10") -> None:
    if not all(c in df.columns for c in ("method", "m", "ef_search", metric)):
        return

    tmp = df[df["method"] == "hnsw"].dropna(subset=["m", "ef_search", metric]).copy()
    if tmp.empty:
        return

    pivot = tmp.pivot_table(index="m", columns="ef_search", values=metric, aggfunc="mean")
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
    plt.title(f"HNSW parameter sensitivity: m vs ef_search ({metric})")
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def plot_pareto(df: pd.DataFrame, out: Path, metric: str = "Precision@10") -> None:
    if not all(c in df.columns for c in ("QPS", metric, "method")):
        return

    tmp = df.dropna(subset=["QPS", metric]).copy()
    if tmp.empty:
        return

    frontier = pareto_frontier(tmp, x="QPS", y=metric, maximize_x=True, maximize_y=True)

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=tmp, x="QPS", y=metric, hue="method", alpha=0.55, s=60)
    if not frontier.empty:
        plt.plot(frontier["QPS"], frontier[metric], color="black", linewidth=2, label="Pareto frontier")
    plt.title(f"Pareto frontier: QPS vs {metric}")
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate thesis figures from experiment results")
    parser.add_argument(
        "--input",
        nargs="*",
        default=None,
        help="Input directories to search (defaults: experiments/results and src/indexing/data/benchmarks)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: experiments/results/figures)",
    )
    parser.add_argument(
        "--primary-metric",
        default="Precision@10",
        help="Metric to optimize and use for pareto/heatmaps",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_dirs = (
        [Path(p) for p in args.input]
        if args.input
        else [repo_root / "experiments" / "results", repo_root / "src" / "indexing" / "data" / "benchmarks"]
    )

    fig_dir = Path(args.output) if args.output else (repo_root / "experiments" / "results" / "figures")
    _ensure_dir(fig_dir)

    df_all = load_results(input_dirs)
    if df_all.empty:
        raise SystemExit("No results found")

    df_ok = df_all.copy()
    if "n_failed" in df_ok.columns:
        df_ok = df_ok[df_ok["n_failed"].fillna(0) == 0]

    best = pick_best_by_method(df_ok, primary_metric=args.primary_metric)

    plot_precision_bars(best, fig_dir / "precision_at_k_bar.png")
    plot_qps_vs_recall(best, fig_dir / "qps_vs_recall_scatter.png")
    plot_latency_hist(df_ok, fig_dir / "latency_hist.png")
    plot_param_heatmap_hnsw(df_ok, fig_dir / "hnsw_m_vs_ef_search_heatmap.png", metric=args.primary_metric)
    plot_pareto(df_ok, fig_dir / "pareto_frontier.png", metric=args.primary_metric)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
