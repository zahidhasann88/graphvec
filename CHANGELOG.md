# Changelog

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
