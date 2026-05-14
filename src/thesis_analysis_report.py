import argparse
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

METRIC_CANDIDATES = (
    "Precision@1",
    "Precision@3",
    "Precision@5",
    "Precision@10",
    "Recall@10",
    "MRR@10",
    "NDCG@10",
    "QPS",
    "total_search_time",
    "n_queries",
    "n_success",
    "n_failed",
)


@dataclass(frozen=True)
class Paths:
    repo_root: Path

    @property
    def default_input_dirs(self) -> list[Path]:
        return [
            self.repo_root / "experiments" / "results",
            self.repo_root / "src" / "indexing" / "data" / "benchmarks",
        ]

    @property
    def default_output_dir(self) -> Path:
        return self.repo_root / "experiments" / "results"


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows)
    df["_source_file"] = str(path)
    return df


def _read_json(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    df = pd.json_normalize(obj)
    df["_source_file"] = str(path)
    return df


def load_results(input_dirs: Iterable[Path]) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []

    for d in input_dirs:
        if not d.exists():
            continue

        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue

            if p.suffix.lower() == ".jsonl":
                dfs.append(_read_jsonl(p))
            elif p.suffix.lower() == ".json" and "results" in p.name.lower():
                dfs.append(_read_json(p))
            elif p.suffix.lower() == ".csv" and "results" in p.name.lower():
                try:
                    df = pd.read_csv(p)
                    df["_source_file"] = str(p)
                    dfs.append(df)
                except Exception:
                    continue

    if not dfs:
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True, sort=False)

    if "method" not in df_all.columns:
        if "experiment" in df_all.columns:
            df_all["method"] = df_all["experiment"].astype(str)
        else:
            df_all["method"] = "unknown"

    df_all["method"] = df_all["method"].astype(str)

    if "n_queries" in df_all.columns:
        df_all["n_queries"] = pd.to_numeric(df_all["n_queries"], errors="coerce")

    if "QPS" in df_all.columns:
        df_all["QPS"] = pd.to_numeric(df_all["QPS"], errors="coerce")

    for c in METRIC_CANDIDATES:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

    if "total_search_time" in df_all.columns and "n_queries" in df_all.columns:
        df_all["avg_latency_s"] = df_all["total_search_time"] / df_all["n_queries"].replace(0, pd.NA)
        df_all["avg_latency_ms"] = 1000.0 * df_all["avg_latency_s"]

    # Mark failed runs
    if "n_success" in df_all.columns and "n_failed" in df_all.columns:
        df_all["success_rate"] = df_all["n_success"] / (df_all["n_success"] + df_all["n_failed"]).replace(0, pd.NA)

    return df_all


def pick_best_by_method(df: pd.DataFrame, primary_metric: str = "Precision@10") -> pd.DataFrame:
    if df.empty:
        return df

    if primary_metric not in df.columns:
        raise ValueError(f"Primary metric {primary_metric!r} not present in results")

    df_ok = df.copy()
    if "n_failed" in df_ok.columns:
        df_ok = df_ok[df_ok["n_failed"].fillna(0) == 0]

    # Tie-break: highest QPS
    sort_cols = [primary_metric]
    ascending = [False]
    if "QPS" in df_ok.columns:
        sort_cols.append("QPS")
        ascending.append(False)

    best = (
        df_ok.sort_values(sort_cols, ascending=ascending)
        .groupby("method", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    return best


def pareto_frontier(df: pd.DataFrame, x: str, y: str, maximize_x: bool = True, maximize_y: bool = True) -> pd.DataFrame:
    if df.empty:
        return df

    tmp = df[["method", x, y]].dropna().copy()
    if tmp.empty:
        return tmp

    # Sort by x
    tmp = tmp.sort_values(x, ascending=not maximize_x).reset_index(drop=True)

    frontier_idx: list[int] = []
    best_y: float | None = None

    for i, row in tmp.iterrows():
        yi = float(row[y])
        if best_y is None:
            frontier_idx.append(i)
            best_y = yi
            continue

        if maximize_y:
            if yi > best_y:
                frontier_idx.append(i)
                best_y = yi
        else:
            if yi < best_y:
                frontier_idx.append(i)
                best_y = yi

    return tmp.loc[frontier_idx].reset_index(drop=True)


def proportion_ci_wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    p = min(max(p, 0.0), 1.0)
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1 - p) / n) + (z**2) / (4.0 * n**2))
    return (max(0.0, center - half), min(1.0, center + half))


def two_proportion_ztest(p1: float, n1: int, p2: float, n2: int) -> tuple[float, float]:
    """Return (z, p_value) for H0: p1 == p2. Normal approx."""
    if n1 <= 0 or n2 <= 0:
        return (math.nan, math.nan)

    from scipy import stats  # local import to keep base import light

    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return (math.nan, math.nan)
    z = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return (z, p_value)


def build_comparison_table(best: pd.DataFrame) -> pd.DataFrame:
    if best.empty:
        return best

    metric_cols = [
        c for c in best.columns if any(c.startswith(prefix) for prefix in ("Precision@", "Recall@", "MRR@", "NDCG@"))
    ]
    base_cols = ["method"]
    for c in ("QPS", "avg_latency_ms", "n_queries"):
        if c in best.columns:
            base_cols.append(c)

    # Include params automatically: numeric columns that are not metrics or base
    excluded = set(
        base_cols
        + metric_cols
        + ["timestamp", "_source_file", "error", "total_search_time", "n_success", "n_failed", "success_rate"]
    )
    param_cols = [c for c in best.columns if c not in excluded and best[c].notna().any()]

    # Prefer known param order
    preferred = [
        c for c in ("m", "ef_construct", "ef_search", "rrf_k", "k1", "b", "chunk_size", "filtering") if c in param_cols
    ]
    remaining = [c for c in param_cols if c not in preferred]

    cols = base_cols + preferred + remaining + sorted(metric_cols)
    return best.loc[:, [c for c in cols if c in best.columns]].copy()


def export_outputs(
    df_all: pd.DataFrame,
    out_dir: Path,
    primary_metric: str = "Precision@10",
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    best = pick_best_by_method(df_all, primary_metric=primary_metric)
    table = build_comparison_table(best)

    comparison_csv = out_dir / "comparison_table.csv"
    table.to_csv(comparison_csv, index=False, encoding="utf-8")

    latex_path = out_dir / "comparison_table.tex"
    try:
        latex_path.write_text(table.to_latex(index=False, float_format=lambda x: f"{x:.4f}"), encoding="utf-8")
    except Exception:
        # Fallback without float formatting
        latex_path.write_text(table.to_latex(index=False), encoding="utf-8")

    # Stats + report
    report_md = out_dir / "report.md"
    report_md.write_text(generate_markdown_report(df_all, best, primary_metric=primary_metric), encoding="utf-8")

    return {
        "comparison_csv": comparison_csv,
        "comparison_latex": latex_path,
        "report_md": report_md,
        "fig_dir": fig_dir,
    }


def _method_best(best: pd.DataFrame, method: str) -> pd.Series | None:
    if best.empty:
        return None
    m = best[best["method"] == method]
    if m.empty:
        return None
    return m.iloc[0]


def generate_markdown_report(df_all: pd.DataFrame, best: pd.DataFrame, primary_metric: str = "Precision@10") -> str:
    lines: list[str] = []
    lines.append("# Thesis experiment analysis\n")

    lines.append("## Data coverage\n")
    lines.append(f"- **Total runs loaded**: {len(df_all)}")
    lines.append(f"- **Methods**: {', '.join(sorted(df_all['method'].dropna().unique()))}\n")

    if best.empty:
        lines.append("No successful runs found to summarize.")
        return "\n".join(lines) + "\n"

    lines.append("## Best run per method (by " + primary_metric + ")\n")
    lines.append(
        best[["method"] + [c for c in (primary_metric, "QPS", "avg_latency_ms") if c in best.columns]].to_markdown(
            index=False
        )
    )
    lines.append("\n")

    # Confidence intervals (only for proportion-like metrics)
    if "n_queries" in best.columns and primary_metric in best.columns:
        lines.append("## Confidence intervals (Wilson, 95%)\n")
        rows = []
        for _, r in best.iterrows():
            n = int(r.get("n_queries", 0) or 0)
            p = float(r.get(primary_metric, math.nan))
            lo, hi = proportion_ci_wilson(p, n)
            rows.append({"method": r["method"], primary_metric: p, "n_queries": n, "ci_low": lo, "ci_high": hi})
        ci_df = pd.DataFrame(rows)
        lines.append(ci_df.to_markdown(index=False))
        lines.append("\n")

    # Hypothesis tests (approx)
    lines.append("## Hypothesis tests (approximate; aggregated metrics)\n")
    lines.append(
        "Assumptions: metrics like Precision@10 are treated as proportions with sample size = n_queries (normal approximation).\n"
    )

    # H1: hybrid outperforms dense/sparse
    hybrid = None
    for candidate in ("hybrid", "hybrid_rrf", "hybrid_search", "hybrid-fusion"):
        hybrid = _method_best(best, candidate)
        if hybrid is not None:
            break

    dense = _method_best(best, "hnsw")
    sparse = _method_best(best, "bm25")

    if hybrid is None:
        lines.append("- **H1**: hybrid method not present in results -> cannot test.")
    else:
        lines.append("- **H1**: Hybrid search outperforms pure dense/sparse")
        if "n_queries" in best.columns and primary_metric in best.columns:
            n_h = int(hybrid.get("n_queries", 0) or 0)
            p_h = float(hybrid.get(primary_metric, math.nan))

            def _cmp(label: str, other: pd.Series | None) -> str:
                if other is None:
                    return f"  - {label}: missing"
                n_o = int(other.get("n_queries", 0) or 0)
                p_o = float(other.get(primary_metric, math.nan))
                z, p = two_proportion_ztest(p_h, n_h, p_o, n_o)
                return f"  - {label}: z={z:.3f}, p={p:.4g} (hybrid={p_h:.4f}, other={p_o:.4f})"

            lines.append(_cmp("Hybrid vs HNSW", dense))
            lines.append(_cmp("Hybrid vs BM25", sparse))
        else:
            lines.append("  - Insufficient columns (need n_queries + metric) -> cannot compute test.")

    # H2: larger m/ef improve quality but reduce QPS
    lines.append("- **H2**: Larger `m` and `ef_search` improve quality but reduce QPS")
    if all(c in df_all.columns for c in ("m", "ef_search", primary_metric, "QPS")):
        from scipy import stats

        df_h = df_all[df_all["method"] == "hnsw"].dropna(subset=["m", "ef_search", primary_metric, "QPS"]).copy()
        if df_h.empty:
            lines.append("  - No HNSW rows with (m, ef_search, metric, QPS) -> cannot test.")
        else:
            rho_m_q, p_m_q = stats.spearmanr(df_h["m"], df_h[primary_metric])
            rho_ef_q, p_ef_q = stats.spearmanr(df_h["ef_search"], df_h[primary_metric])
            rho_m_s, p_m_s = stats.spearmanr(df_h["m"], df_h["QPS"])
            rho_ef_s, p_ef_s = stats.spearmanr(df_h["ef_search"], df_h["QPS"])

            lines.append(f"  - Spearman corr(m, {primary_metric}) = {rho_m_q:.3f} (p={p_m_q:.4g})")
            lines.append(f"  - Spearman corr(ef_search, {primary_metric}) = {rho_ef_q:.3f} (p={p_ef_q:.4g})")
            lines.append(f"  - Spearman corr(m, QPS) = {rho_m_s:.3f} (p={p_m_s:.4g})")
            lines.append(f"  - Spearman corr(ef_search, QPS) = {rho_ef_s:.3f} (p={p_ef_s:.4g})")
    else:
        lines.append("  - Missing columns for correlation test (need m, ef_search, QPS, metric).")

    # H3/H4: only if columns exist
    lines.append("- **H3**: Filtering dramatically improves precision for targeted queries")
    if "filtering" in df_all.columns and primary_metric in df_all.columns:
        grp = (
            df_all.dropna(subset=["filtering", primary_metric])
            .groupby("filtering")[primary_metric]
            .mean()
            .reset_index()
        )
        lines.append("  - Mean precision by filtering flag:")
        lines.append(grp.to_markdown(index=False).replace("\n", "\n    "))
    else:
        lines.append("  - No `filtering` column found -> cannot test with current data.")

    lines.append("- **H4**: Optimal chunk size depends on document type (256 vs 512 vs 1024)")
    if "chunk_size" in df_all.columns and primary_metric in df_all.columns:
        grp = (
            df_all.dropna(subset=["chunk_size", primary_metric])
            .groupby("chunk_size")[primary_metric]
            .mean()
            .reset_index()
        )
        lines.append("  - Mean precision by chunk_size:")
        lines.append(grp.to_markdown(index=False).replace("\n", "\n    "))
    else:
        lines.append("  - No `chunk_size` column found -> cannot test with current data.")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate thesis analysis tables/figures/report from experiment results"
    )
    parser.add_argument(
        "--input",
        nargs="*",
        default=None,
        help="Input directories to search (defaults: experiments/results and src/indexing/data/benchmarks)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: experiments/results)",
    )
    parser.add_argument(
        "--primary-metric",
        default="Precision@10",
        help="Primary metric for best-per-method selection",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    paths = Paths(repo_root=repo_root)

    input_dirs = [Path(p) for p in args.input] if args.input else paths.default_input_dirs
    out_dir = Path(args.output) if args.output else paths.default_output_dir

    df_all = load_results(input_dirs)
    if df_all.empty:
        raise SystemExit(f"No results found in: {', '.join(str(p) for p in input_dirs)}")

    export_outputs(df_all, out_dir=out_dir, primary_metric=args.primary_metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
