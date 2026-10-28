# Limitations & guarantees

- **Memory**: a read-only index memory-maps the vectors and the layer-0
  graph, so resident memory grows only with the pages searches touch; the
  ID map and upper-layer lists (about 1/M of nodes) are in RAM. A writable
  index keeps its vectors and layer-0 graph, roughly
  `n × (4·dims + 16·M + 5)` bytes (1M × 768 vectors is about 3.3 GB), in a
  working copy: in RAM up to `memory_limit` (default a quarter of physical
  memory), then in memory-mapped files under `<index>/.work/`, which lets
  an index grow beyond RAM. What stays in RAM either way is about
  `n × 17` bytes (norms, tombstones, and visited flags while inserting),
  plus the ID map (on the order of 100 bytes per ID in CPython) and the
  upper-layer lists. The largest index we have built is listed in
  [benchmarks](benchmarks.md); larger ones are untested.
- **Disk space while writing**: a working copy on disk is a second copy of
  the vectors and graph, and `save()` writes a third before replacing the
  committed files, so allow about three times the index size during a
  build that exceeds `memory_limit`. Opening such an index for writing
  checks free space for the working copy first.
- **Concurrency**: exactly one writer (lock file); a crash leaves a stale
  lock, which `graphvec unlock` (or `GraphIndex.force_unlock()`) removes.
  GraphIndex cannot tell a crashed writer from a running one, so only
  unlock when no writer is running. Read-only readers may run concurrently,
  but must reopen to see new commits. No transactional guarantee: crash
  safety = fsync + atomic rename, durability depends on the filesystem/OS.
- **Deletes**: tombstones; deleted nodes remain traversal waypoints and
  their IDs cannot be reused until `compact()`. Heavy delete workloads
  degrade recall until `build()`/`compact()`.
- **Speed**: traversal is NumPy-batched Python; it trails compiled ANN
  libraries such as FAISS by a wide margin (see [benchmarks](benchmarks.md)).
  Insertion is single-threaded; `insert_batch` shares Python overhead
  across a group of insertions but does not use more cores.
- **Disk space**: `save()` checks free space first and raises
  `InsufficientDiskSpaceError` (leaving the last commit intact) if the
  estimated write size does not fit.
- **Format**: little-endian; reads format v1 and v2, writes v2 (see
  [file format](file-format.md)); no encryption, no compression.
  File permissions are the user's responsibility.
