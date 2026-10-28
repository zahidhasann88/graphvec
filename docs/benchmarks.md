# Benchmarks

Methodology: fixed seeds, clustered synthetic data
(`benchmarks/datasets.py`), construction timed separately from queries,
recall@k measured against GraphVec's own chunked exact search on the same
index, latency from `perf_counter` around single queries. Environment,
versions, n, dims, metric and ef are recorded in every report JSON.

Run it yourself:

```bash
PYTHONPATH=src python benchmarks/run.py --n 10000 --dims 128 --out results-10k.json
PYTHONPATH=src python benchmarks/run.py --n 100000 --dims 384
graphvec benchmark ./index.gvec --queries queries.npy --top-k 10
python benchmarks/compare.py   # optional, requires hnswlib (pip install ".[bench]")
```

Results depend on your hardware. The figures below come from one measured
run and are not product claims. When reporting numbers in issues or PRs,
attach the runner's JSON output. Scales beyond what fits in RAM for the
graph (notably 10M vectors) are *not* claimed until measured.

## Measured results (this repo, sandbox run 2026-10-27)

Clustered synthetic data, cosine, k=10, ef_search=100, seed 42.
Raw JSON: `benchmarks/results-10k-128.json`, `benchmarks/results-100k-128.json`.
Environment is recorded inside those files; treat numbers as indicative
of this environment only, not as product claims.

| n | dims | build | insert/s | p50 | p95 | p99 | recall@10 | disk |
|---|------|-------|----------|-----|-----|-----|-----------|------|
| 10,000 | 128 | 36.7s | 272 | 1.07ms | 1.16ms | 1.21ms | 1.000 | 7.1MB |
| 100,000 | 128 | 471.5s | 212 | 1.33ms | 1.52ms | 1.76ms | 1.000 | 71.4MB |

1M/10M-vector runs were **not** performed in this environment; at the
measured insertion rate they would take hours in pure Python, which is a
documented limitation, not a hidden one.
