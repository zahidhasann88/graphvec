# GraphVec

GraphVec is a local, file-based vector search library for Python. It stores an
index in a directory on disk and finds nearest neighbours with its own
HNSW-style graph. It needs no database server, cloud service, or background process.

It is written in pure Python and NumPy. That makes it easy to read and embed,
but it is slower than compiled libraries such as FAISS or hnswlib.

## Installation

Requires Python 3.11+.
```bash
pip install .              # from a clone of this repository
pip install -e ".[dev]"    # with test and lint tools
```

## Quick start

```python
import numpy as np
from graphvec import GraphIndex

rng = np.random.default_rng(42)
vectors = rng.random((1_000, 128), dtype=np.float32)
ids = [f"doc-{i}" for i in range(len(vectors))]

with GraphIndex.create("./vectors.gvec", dimensions=128, metric="cosine") as index:
    index.add(vectors, ids=ids)
    index.save()  # nothing is written to disk until save()

with GraphIndex.open("./vectors.gvec", readonly=True) as index:
    query = rng.random(128, dtype=np.float32)
    for result in index.search(query, k=5, ef_search=100):
        print(result.id, result.score)
```

## Python API

| Method | Purpose |
|---|---|
| `GraphIndex.create(path, dimensions, metric="cosine", max_neighbors=16, construction_ef=200, seed=42, overwrite=False)` | Create a new index directory |
| `GraphIndex.open(path, readonly=False, verify_checksums=False)` | Open an index (read-only mode memory-maps vectors) |
| `add(vectors, ids=None, replace=False)` | Insert vectors; duplicate IDs raise `DuplicateIDError` unless `replace=True` |
| `search(query, k=10, ef_search=100)` / `search_batch(queries, ...)` | Approximate search; returns `SearchResult(id, score, internal_id)` |
| `search_exact(query, k=10)` | Brute-force search, useful to measure recall |
| `get(id)` / `contains(id)` / `delete(ids)` | Read, check, or delete by ID (deletes are tombstones) |
| `compact()` | Remove deleted rows, rebuild the graph, and save |
| `build()` | Rebuild the graph from stored vectors |
| `save()` / `close()` / `validate()` | Commit to disk, release the writer lock, check consistency |

Higher `ef_search` gives better recall but slower queries. All errors inherit from `graphvec.GraphVecError`.

**Metrics** (`metric=`):
- `cosine`: score is cosine similarity, higher is better. Vectors are normalised
  when added, so `get()` returns the normalised vector. Zero vectors score 0.
- `dot`: score is the dot product, higher is better.
- `squared_euclidean`: score is the squared L2 distance, lower is better.

## Command line

```bash
graphvec create ./index.gvec --dimensions 128 --metric cosine
graphvec import ./index.gvec vectors.npy --ids ids.json   # or .jsonl / .parquet
graphvec search ./index.gvec --query-file query.npy --top-k 10
graphvec info ./index.gvec
graphvec validate ./index.gvec
graphvec benchmark ./index.gvec --queries queries.npy --top-k 10
graphvec compact ./index.gvec
graphvec export ./index.gvec --output vectors.npy         # also writes vectors.npy.ids.json
```

JSONL records look like `{"id": "doc-1", "vector": [0.1, ...]}`; Parquet files need `id` and
`vector` columns and `pip install ".[parquet]"`. Most commands accept `--json`; see `graphvec --help`.

## Limitations
- The graph and ID map are held in memory; only vectors can be memory-mapped.
- One writer at a time. A crash can leave a stale `.writer.lock` file to delete by hand.
- Deleted IDs cannot be reused until `compact()`.
- Insertion is slow (hundreds of vectors per second in our measurements).
- More detail and crash-safety guarantees: [docs/limitations.md](docs/limitations.md).

## Documentation

[Architecture and algorithms](docs/architecture.md) · [File format](docs/file-format.md) ·
[Benchmarks](docs/benchmarks.md) · [Changelog](CHANGELOG.md) · [Security](SECURITY.md)

## Development

Add tests for behaviour changes, and never commit benchmark numbers you did not measure.
```bash
pytest                                   # add -m slow for large tests
ruff check . && ruff format --check .
mypy src/graphvec
```

## License

MIT. See [LICENSE](LICENSE).
