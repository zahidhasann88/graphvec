# Architecture and algorithms

## Modules

All code is in `src/graphvec/`:

- `api.py`: `GraphIndex` lifecycle, IDs, add/delete/compact, search.
- `graph.py`: HNSW-style graph: insertion, search, neighbour pruning.
- `exact.py`: chunked brute-force search. It never builds a full queries × vectors matrix.
- `metrics.py`, `validation.py`: metric semantics and input checks.
- `storage.py`, `format.py`: binary file format, atomic saves, checksums, locking, mmap loading.
- `exceptions.py`, `cli.py`: error types and the `graphvec` command.

## Data flow

`add()` validates the input, normalises it for cosine, appends it to a
contiguous float32 array (capacity doubles as needed), and inserts each row
into the graph. `save()` writes each data file to `*.tmp`, fsyncs it,
atomically replaces the old file, then writes `manifest.json` last. The
manifest is the commit point. `open(readonly=True)` memory-maps
`vectors.bin`; the graph and ID tables are loaded into RAM. Only one writer
is allowed (`.writer.lock`, created with `O_EXCL`); many read-only readers can
open the same index.

## Graph construction

GraphVec implements a multi-layer navigable small-world graph from scratch.

- **Levels**: `level = floor(-ln(u) / ln(M))`, with `u` from a seeded NumPy RNG.
  Construction is deterministic for a given seed and insertion order.
- **Insertion**: greedy descent from the entry point through the layers above
  the new node's level, then an ef-bounded best-first search on each lower
  layer (`construction_ef`).
- **Neighbour selection**: a candidate is kept only if it is closer to the new
  node than to any neighbour already chosen; remaining slots are filled to keep
  the graph connected. Links are bidirectional.
- **Pruning**: neighbour lists are pruned back to `2M` (layer 0) or `M` (upper
  layers) with the same heuristic. Pruning is lazy: it runs only when a list
  reaches twice its target size. In our 10k benchmark this kept full recall
  and cut build time by about 40% compared with pruning on every insert.

## Search

- Greedy descent through the upper layers, then best-first traversal on layer 0
  with a visited set, a candidate min-heap, and a bounded result max-heap.
- `ef_search` sets the size of the result heap and controls the
  quality/latency trade-off.
- Traversal stops when the best remaining candidate is worse than the worst
  result and the result heap is full.
- Deleted (tombstoned) nodes are still traversed but never returned.

## Exact search

Chunked dot-product scoring over live rows only. Squared Euclidean distance is
computed as `|a|² + |b|² − 2a·b`, with no square roots.
