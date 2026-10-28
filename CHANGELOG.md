# Changelog

## Unreleased

The index format is unchanged (version 2); 0.2 and this release read each
other's indexes.

- **Building indexes larger than RAM.** A writable index keeps its vectors
  and layer-0 graph in a working copy: in RAM up to `memory_limit`
  (a new `create()`/`open()` argument, default a quarter of physical
  memory), then in memory-mapped files under `<index>/.work/`. `save()`
  still writes new files and the manifest last, so the last commit
  survives a crash; the next writer deletes a stale `.work/`. The ID map
  stays in RAM. See docs/limitations.md for what remains in memory and the
  disk space a build needs. Measured at 50k × 768: 70 MB of private memory
  with the working copy on disk against 325 MB in RAM, at the same build
  time.
- **Batch insertion.** `add()`, `build()` and `compact()` take
  `insert_batch` (1 to 64, default 1): a group of vectors searches layer
  0 in lockstep, selects neighbours together, and also links to earlier
  members of its group. `insert_batch=1` builds exactly the graph 0.2.0
  built.
  Recall matches 0.2.0. Measured speed is inconclusive so far: user CPU
  time fell 15% on 20k × 128 vectors, but extra page faults on Windows
  offset it; see docs/benchmarks.md.
- Opening an index for writing no longer copies it into RAM up front when
  it exceeds `memory_limit`; it checks free disk space for the working
  copy and raises `InsufficientDiskSpaceError` if it does not fit.
- `compact()` moves live rows in place instead of copying all vectors.
- `save()` writes data files in 64 MiB chunks.
- Benchmarks: `run.py` takes `--insert-batch` and `--memory-limit-mb`,
  and records CPU time, private (non-file-backed) memory and whether the
  working copy was on disk.

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
