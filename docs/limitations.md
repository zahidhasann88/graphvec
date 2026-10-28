# Limitations & guarantees

- **Memory**: a read-only index memory-maps the vectors and the layer-0
  graph, so resident memory grows only with the pages searches touch; the
  ID map and upper-layer lists (about 1/M of nodes) are in RAM. A writable
  index loads the vectors and the whole graph into RAM, so building or
  updating needs roughly `n × (4·dims + 16·M)` bytes plus the ID map;
  1M × 768 vectors is about 3.3 GB. Building indexes larger than RAM is not
  supported yet.
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
  Insertion is single-threaded.
- **Disk space**: `save()` checks free space first and raises
  `InsufficientDiskSpaceError` (leaving the last commit intact) if the
  estimated write size does not fit.
- **Format**: little-endian; reads format v1 and v2, writes v2 (see
  [file format](file-format.md)); no encryption, no compression.
  File permissions are the user's responsibility.
