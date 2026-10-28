# Changelog

## 0.2.0 — 2026-10-28

**Index format 2.** 0.2 reads 0.1 indexes and migrates them to format 2
on the next `save()`/`compact()`. 0.1 cannot read format 2, so keep a copy
if you might downgrade. See docs/file-format.md.

- Faster insertion and search: graph search expands 16 frontier nodes per
  round with batched NumPy operations, and neighbour selection no longer
  builds a full pairwise matrix. See docs/benchmarks.md for measurements.
- Higher recall at the same `ef_search` (searches explore slightly more).
- Lower memory: layer 0 of the graph is stored as flat int32 arrays instead
  of nested Python lists. Read-only opens memory-map it from `graph.bin`,
  so opening is fast and uses little RAM.
- Stricter validation of `graph.bin`: every neighbour id, degree, unused
  slot and the entry point are checked on open.
- `create()` rejects `max_neighbors` or `construction_ef` below 1.
- Closing a read-only index releases its memory-mapped files.
- Benchmarks: memory-bounded runner with held-out queries, an
  `ef_search` sweep, peak memory and open time; `compare.py` compares with
  FAISS HNSW (and hnswlib when installed) on identical data.
- Package metadata: Python version, maturity and topic classifiers.
- Version is now set only in `src/graphvec/__init__.py`.

## 0.1.0 — 2026-10-27
Initial release: HNSW-style graph ANN, exact search, cosine/dot/
squared-Euclidean metrics, versioned file format with checksums and atomic
commits, incremental updates with tombstones + compaction, CLI, benchmarks,
test suite (unit/property/integration/reliability).

Also in 0.1.0:
- `graphvec import` reads Parquet files (`id` and `vector` columns) in
  batches; needs the optional `parquet` extra.
- `save()` checks free disk space first and raises
  `InsufficientDiskSpaceError` without touching the last commit.
- `add()` checks duplicate and deleted IDs once per batch instead of once
  per ID, so large batches are faster.
- `GraphIndex.iter_ids()` lists live IDs; `graphvec export` uses it.
- `GraphIndex.force_unlock()` and `graphvec unlock` remove a stale writer
  lock after a crash. Lock errors now name the PID holding the lock.
- `graphvec info` reports the format version stored in the index.
