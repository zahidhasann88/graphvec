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
manifest is the commit point. Only one writer is allowed (`.writer.lock`,
created with `O_EXCL`); many read-only readers can open the same index.

## Graph storage

Layer 0 holds every node, so it is stored as two flat arrays rather than
Python lists:

- `adj0`: one fixed-width `int32` row per node with `4·M` slots (unused slots
  are `-1`). The width matches the lazy-pruning bound below.
- `deg0`: the number of used slots in each row.

Nodes above layer 0 (about `1/M` of them) keep their upper-layer neighbour
lists in a dict. `open(readonly=True)` memory-maps `vectors.bin` and the
layer-0 arrays in `graph.bin`, so opening is fast and resident memory grows
only with the pages a search touches. A writable index copies both into RAM.

## Graph construction

GraphVec implements a multi-layer navigable small-world graph from scratch.

- **Levels**: `level = floor(-ln(u) / ln(M))`, with `u` from a seeded NumPy RNG.
  Construction is deterministic for a given seed and insertion order.
- **Insertion**: greedy descent from the entry point through the layers above
  the new node's level, then an ef-bounded best-first search on each lower
  layer (`construction_ef`).
- **Neighbour selection**: a candidate is kept only if it is closer to the new
  node than to any neighbour already chosen; remaining slots are filled to keep
  the graph connected. Links are bidirectional. Each pick blocks the
  candidates it rules out with one matrix-vector product.
- **Pruning**: neighbour lists are pruned back to `2M` (layer 0) or `M` (upper
  layers) with the same heuristic. Pruning is lazy: it runs only when a list
  would exceed twice its target size, which amortises its cost.

## Search

- Greedy descent through the upper layers, then best-first traversal of the
  layer below with a visited mask, a frontier of unexpanded nodes, and the
  `ef` closest nodes found so far.
- **Batched expansion**: each round expands the 16 closest frontier nodes at
  once. Their neighbours are gathered, filtered against the visited mask and
  scored with one matrix-vector product, then merged into the results with
  `argpartition`. Python-level work per round is a handful of array
  operations instead of a heap update per neighbour, which is what made
  0.1.0 slow. Expanding several nodes per round also explores slightly more
  of the graph, so recall at a given `ef_search` is higher than with strict
  one-at-a-time expansion.
- `ef_search` sets how many results are kept and controls the
  quality/latency trade-off.
- Frontier nodes farther than the current `ef`-th result are dropped; the
  search ends when the frontier is empty.
- Deleted (tombstoned) nodes are still traversed but never returned.

## Exact search

Chunked dot-product scoring over live rows only. Squared Euclidean distance is
computed as `|a|² + |b|² − 2a·b`, with no square roots.
