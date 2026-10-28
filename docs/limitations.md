# Limitations & guarantees

- **Memory**: vectors can be memory-mapped read-only (resident memory then
  grows only with touched pages), but the graph, ID map and (in writable
  mode) a RAM copy of vectors are in memory. Datasets whose *graph* does
  not fit in RAM are out of scope for this release.
- **Concurrency**: exactly one writer (lock file); stale locks after a
  crash must be removed manually. Read-only readers may run concurrently,
  but must reopen to see new commits. No transactional guarantee: crash
  safety = fsync + atomic rename, durability depends on the filesystem/OS.
- **Deletes**: tombstones; deleted nodes remain traversal waypoints and
  their IDs cannot be reused until `compact()`. Heavy delete workloads
  degrade recall until `build()`/`compact()`.
- **Speed**: traversal is Python-level; expect it to trail compiled ANN
  libraries. Exact search and benchmarks quantify this on your hardware.
- **Disk space**: `save()` checks free space first and raises
  `InsufficientDiskSpaceError` (leaving the last commit intact) if the
  estimated write size does not fit.
- **Format**: little-endian, format v1 only; no encryption, no compression.
  File permissions are the user's responsibility.
