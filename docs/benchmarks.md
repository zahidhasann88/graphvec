# Benchmarks

All numbers here were measured with the scripts in `benchmarks/`; the raw
JSON for every table is in `benchmarks/results/`. They describe one laptop
and are not product claims. Run the scripts on your own hardware before
relying on them.

## Methodology

- **Data** (`benchmarks/datasets.py`): seeded synthetic float32 vectors.
  *Clustered*: points around `n/200` random centres with noise 0.3 per
  dimension. *Gaussian*: standard normal, the hardest case for graph
  indexes. Queries are 100 **held-out points from the same distribution**,
  not copies of indexed vectors.
- **Ground truth**: GraphVec's exact (brute-force) search on the same index.
- **Build**: vectors are added in batches of 10,000 from a memory-mapped
  file; build time excludes dataset generation and saving.
- **Queries**: one query at a time on a read-only (memory-mapped) index;
  latency is wall time around each call. Recall@10 is the overlap with the
  exact top 10.
- **Parameters**: cosine metric, k = 10, M = 16, construction ef = 200,
  seed 42, single thread for every engine.
- **Environment**: Intel Core i7-8665U (4 cores, 1.9 GHz), 15.8 GB RAM,
  Windows 10, Python 3.13.14, NumPy 2.5.4, FAISS 1.15.1. The laptop was
  also running other applications, so expect some timing noise.

## 0.2.0 compared with 0.1.0

5,000 × 128 Gaussian vectors, same machine and data
(`run-5k-128-gaussian-v0.1.0.json`, `run-5k-128-gaussian-v0.2.0.json`).

| | 0.1.0 | 0.2.0 |
|---|---|---|
| Build | 31.4 s (159 vectors/s) | 11.8 s (424 vectors/s) |
| Read-only open | 0.030 s, 9.8 MB heap | 0.012 s, 0.9 MB heap |
| Index on disk | 3.6 MB | 3.9 MB |

| ef_search | 0.1.0 recall@10 | 0.1.0 p50 | 0.2.0 recall@10 | 0.2.0 p50 |
|---|---|---|---|---|
| 20 | 0.611 | 1.14 ms | 0.786 | 1.38 ms |
| 50 | 0.866 | 2.43 ms | 0.917 | 1.83 ms |
| 100 | 0.975 | 4.27 ms | 0.984 | 2.44 ms |
| 200 | 0.998 | 6.20 ms | 0.999 | 3.46 ms |

0.2.0 explores more per search, so compare rows at similar recall: 0.1.0
needs 4.27 ms for 0.975, while 0.2.0 reaches 0.984 in 2.44 ms. The heap
saving grows with the index, since 0.1.0 built Python lists for every
graph edge. The index is slightly larger on disk because layer-0 rows are
fixed-width so they can be memory-mapped.

## GraphVec compared with FAISS HNSW

100,000 × 128 vectors, FAISS `IndexHNSWFlat` with the same M and
construction ef, both single-threaded
(`compare-100k-128-clustered.json`, `compare-100k-128-gaussian.json`).

| Dataset | Engine | Build | Vectors/s |
|---|---|---|---|
| Clustered | GraphVec 0.2.0 | 241.8 s | 414 |
| Clustered | FAISS | 23.5 s | 4,249 |
| Gaussian | GraphVec 0.2.0 | 515.2 s | 194 |
| Gaussian | FAISS | 52.4 s | 1,910 |

Clustered data (recall@10 / p50 latency):

| ef_search | GraphVec | FAISS |
|---|---|---|
| 10 | 0.997 / 0.62 ms | 0.887 / 0.04 ms |
| 20 | 1.000 / 0.66 ms | 0.977 / 0.05 ms |
| 50 | 1.000 / 0.78 ms | 1.000 / 0.08 ms |
| 100 | 1.000 / 1.03 ms | 1.000 / 0.13 ms |

Gaussian data (recall@10 / p50 latency):

| ef_search | GraphVec | FAISS |
|---|---|---|
| 10 | 0.174 / 1.31 ms | 0.066 / 0.05 ms |
| 50 | 0.439 / 3.21 ms | 0.209 / 0.17 ms |
| 100 | 0.572 / 4.44 ms | 0.356 / 0.32 ms |
| 200 | 0.726 / 6.75 ms | 0.518 / 0.59 ms |

What this shows:

- FAISS builds about **10× faster** and, at equal recall, answers queries
  roughly **8× faster** (clustered: 1.000 recall at 0.08 ms vs 0.66 ms;
  Gaussian: about 0.52–0.57 recall at 0.59 ms vs 4.44 ms). That is the cost
  of a NumPy implementation compared with optimised C++.
- At the same `ef_search`, GraphVec's recall is higher because it expands
  16 nodes per round, so the parameter values are not directly comparable.
- High-dimensional Gaussian data is very hard for every graph index: neither
  engine passes 0.73 recall at ef 200 with these parameters. Real
  embeddings usually behave more like the clustered set.
- GraphVec built the Gaussian index with a peak resident memory of 223 MB,
  and a read-only open took 0.15 s.

## Not yet measured

- **100k at 384 and 768 dimensions, and 1M vectors.** These runs were
  stopped on this laptop because the system ran low on memory. A writable
  index keeps all vectors and the graph in RAM (see
  [limitations](limitations.md)), so 1M × 768 needs over 3.3 GB for the
  index alone. Nothing beyond 100,000 vectors is claimed until it is
  measured.
- **hnswlib**: no prebuilt wheel exists for Windows with Python 3.13;
  `compare.py` includes it automatically where it installs.

## Running the benchmarks

```bash
pip install -e ".[bench]"
python benchmarks/run.py --n 100000 --dims 384 --out results.json
python benchmarks/compare.py --n 100000 --dims 128 --out compare.json
python benchmarks/compare.py --n 100000 --dims 128 --uniform   # Gaussian data
```

To compare with an older release, point `run.py` at its source tree:

```bash
git worktree add ../graphvec-0.1.0 v0.1.0
python benchmarks/run.py --n 5000 --uniform --graphvec-src ../graphvec-0.1.0/src
```

Both scripts print JSON with the environment, parameters and every measured
value. Attach that JSON when reporting numbers in issues or pull requests.
The CLI also has a quick check for an existing index:
`graphvec benchmark ./index.gvec --queries queries.npy --top-k 10`.

Results from 0.1.0 (a Linux sandbox, with the older near-duplicate queries)
are kept in `benchmarks/results/v0.1.0-sandbox-*.json` for reference only;
their methodology differs, so they are not comparable with the tables above.
