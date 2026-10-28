# Architecture and algorithms

## Modules

All code is in `src/graphvec/`:

- `api.py`: `GraphIndex` lifecycle, IDs, add/delete/compact, search.
- `graph.py`: HNSW-style graph: insertion, search, neighbour pruning.
- `exact.py`: chunked brute-force search. It never builds a full queries × vectors matrix.
- `metrics.py`, `validation.py`: metric semantics and input checks.
- `storage.py`, `format.py`: binary file format, atomic saves, checksums, locking, mmap loading.
- `workfile.py`: a writer's working copy, in RAM or in memory-mapped files.
- `exceptions.py`, `cli.py`: error types and the `graphvec` command.

## Data flow

`add()` validates the input, normalises it for cosine, appends it to a
contiguous float32 array (capacity doubles as needed), and inserts the rows
into the graph in groups (see below). `save()` writes each data file to
`*.tmp`, fsyncs it, atomically replaces the old file, then writes
`manifest.json` last. The manifest is the commit point. Only one writer is
allowed (`.writer.lock`, created with `O_EXCL`); many read-only readers can
open the same index.

## Working copy

A writable index never modifies the committed files. It keeps the vectors
and the layer-0 graph arrays in a working copy, and `save()` copies the
working copy into new files. The working copy lives in RAM until those
arrays would exceed `memory_limit` bytes (a `create()`/`open()` argument;
by default a quarter of physical memory). Past that, every array that grows
is copied into a raw file under `<index>/.work/` and memory-mapped, so the
OS can page it out and the index can grow beyond RAM. Growing a mapped
array remaps the same file at a larger size, without copying. Writes to
mapped pages cost extra page faults, so builds are slower on disk (see
[benchmarks](benchmarks.md)), which is why RAM is used while it suffices.

Crash safety is unchanged: a crash leaves the last commit intact plus a
stale `.work/` directory, which the next writer deletes. The ID map,
tombstones, squared norms and upper-layer lists stay in RAM in both modes.
`compact()` moves the live rows down in place a chunk at a time, so it needs
no second copy of the vectors.

## Graph storage

Layer 0 holds every node, so it is stored as two flat arrays rather than
Python lists:

- `adj0`: one fixed-width `int32` row per node with `4·M` slots (unused slots
  are `-1`). The width matches the lazy-pruning bound below.
- `deg0`: the number of used slots in each row.

Nodes above layer 0 (about `1/M` of them) keep their upper-layer neighbour
lists in a dict. `open(readonly=True)` memory-maps `vectors.bin` and the
layer-0 arrays in `graph.bin`, so opening is fast and resident memory grows
only with the pages a search touches. A writable index copies both into its
working copy.

## Graph construction

GraphVec implements a multi-layer navigable small-world graph from scratch.

- **Levels**: `level = floor(-ln(u) / ln(M))`, with `u` from a seeded NumPy RNG.
  Construction is deterministic for a given seed, insertion order and
  `insert_batch`.
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

## Batch insertion

`add(..., insert_batch=32)` inserts vectors in groups of up to 64. The default,
`insert_batch=1`, reproduces one-at-a-time insertion exactly, as in 0.2.0.
Levels are still drawn in id order. In each group:

1. Nodes above layer 0 (about `1/M` of them) are inserted one at a time.
2. The rest descend the upper layers one at a time (a few hops each), then
   search layer 0 **in lockstep**: every round expands up to 16 frontier
   nodes of every query and scores all their neighbours together, one
   matrix-vector product per query. Results and frontiers are padded 2-D
   arrays with one row per query, merged with row-wise `argpartition`;
   visited flags are one bit per query in a shared `uint64` array, cleared
   after the group. Per query, this is exactly the single-query search.
3. The searches ran on the graph as it was before the group, so they cannot
   find each other. To make up for that, each member also gets the earlier
   members of its group as candidates, with distances from one
   matrix-matrix product. Later members are excluded because a node's row
   is rewritten when it links, which would drop a back-link added earlier.
4. Neighbour selection runs in lockstep too: each round picks the next
   unblocked candidate of every member, with one stacked matrix-vector
   product. The members then link in id order.

Python overhead per round is shared by the group, which lowers CPU user
time; the larger temporary arrays cost page faults instead, so the net
effect depends on the platform (see [benchmarks](benchmarks.md) for
measured speed and recall). Grouping changes which graph is built, so recall was measured
against one-at-a-time insertion on the same data.

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
