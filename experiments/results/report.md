# Thesis experiment analysis

## Data coverage

- **Total runs loaded**: 52
- **Methods**: bm25, hnsw, hybrid_rrf

## Best run per method (by Precision@10)

| method     | Precision@10 |   QPS | avg_latency_ms |
|:-----------|-------------:|------:|---------------:|
| hnsw       |       0.8872 | 83.64 |        11.9554 |
| hybrid_rrf |        0.541 | 31.94 |        31.3114 |

## Confidence intervals (Wilson, 95%)

| method     | Precision@10 | n_queries |   ci_low |  ci_high |
|:-----------|-------------:|----------:|---------:|---------:|
| hnsw       |       0.8872 |       500 | 0.856467 | 0.912028 |
| hybrid_rrf |        0.541 |       500 | 0.497174 | 0.584201 |

## Hypothesis tests (approximate; aggregated metrics)

Assumptions: metrics like Precision@10 are treated as proportions with sample size = n_queries (normal approximation).

- **H1**: Hybrid search outperforms pure dense/sparse
    - Hybrid vs HNSW: z=-12.115, p=0 (hybrid=0.5410, other=0.8872)
    - Hybrid vs BM25: missing
- **H2**: Larger `m` and `ef_search` improve quality but reduce QPS
    - Spearman corr(m, Precision@10) = -0.138 (p=0.4227)
    - Spearman corr(ef_search, Precision@10) = 0.128 (p=0.4568)
    - Spearman corr(m, QPS) = -0.428 (p=0.009195)
    - Spearman corr(ef_search, QPS) = -0.118 (p=0.4935)
- **H3**: Filtering dramatically improves precision for targeted queries
    - No `filtering` column found -> cannot test with current data.
- **H4**: Optimal chunk size depends on document type (256 vs 512 vs 1024)
    - No `chunk_size` column found -> cannot test with current data.
