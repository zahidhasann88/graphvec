"""Public Python API: :class:`GraphIndex` and :class:`SearchResult`."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from graphvec import storage
from graphvec.exact import exact_search
from graphvec.exceptions import (
    ClosedIndexError,
    DuplicateIDError,
    IncompatibleIndexError,
    InvalidVectorError,
    MissingIDError,
)
from graphvec.format import FORMAT_VERSION
from graphvec.graph import Graph
from graphvec.metrics import normalize_rows, normalize_vector, scores_from_distances
from graphvec.validation import validate_metric, validate_queries, validate_query, validate_vectors


@dataclass(frozen=True)
class SearchResult:
    """One search hit. ``score`` follows the metric semantics in metrics.py."""

    id: str
    score: float
    internal_id: int | None = None


class GraphIndex:
    """A file-backed HNSW-style approximate nearest-neighbour index.

    Lifecycle: ``create`` (or ``open``) -> ``add``/``delete`` -> ``save`` ->
    ``close``. ``build()`` rebuilds the graph from all live vectors (useful
    after many deletions); ``save()`` is the commit point -- unsaved changes
    are lost on ``close()``.
    """

    def __init__(self, path, dimensions, metric, max_neighbors, construction_ef, seed, readonly):
        self.path = Path(path)
        self.dimensions = int(dimensions)
        self.metric = metric
        self.max_neighbors = int(max_neighbors)
        self.construction_ef = int(construction_ef)
        self.seed = int(seed)
        self.readonly = bool(readonly)
        self.format_version = FORMAT_VERSION
        self._closed = False
        self._lock = None
        self._graph = Graph(dimensions, metric, max_neighbors, construction_ef, seed)
        # in-memory state (vectors may be a memmap in readonly mode)
        self._vectors: np.ndarray = np.empty((0, dimensions), dtype=np.float32)
        self._norms_sq = np.empty(0, dtype=np.float64)
        self._ids: list[str] = []
        self._id_to_internal: dict[str, int] = {}
        self._deleted = np.empty(0, dtype=bool)
        self._capacity = 0

    @classmethod
    def create(
        cls,
        path,
        dimensions: int,
        metric: str = "cosine",
        max_neighbors: int = 16,
        construction_ef: int = 200,
        seed: int = 42,
        overwrite: bool = False,
    ) -> GraphIndex:
        validate_metric(metric)
        if dimensions <= 0:
            raise InvalidVectorError("dimensions must be positive")
        p = Path(path)
        if p.exists() and any(p.iterdir()) if p.is_dir() else p.exists():
            if not overwrite:
                raise IncompatibleIndexError(
                    f"{p} already exists; pass overwrite=True to replace it, or open() it"
                )
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        p.mkdir(parents=True, exist_ok=True)
        idx = cls(p, dimensions, metric, max_neighbors, construction_ef, seed, readonly=False)
        idx._lock = storage.acquire_lock(p)
        idx.save()  # initial empty commit so open() works immediately
        return idx

    @classmethod
    def open(cls, path, readonly: bool = False, verify_checksums: bool = False) -> GraphIndex:
        p = Path(path)
        if not p.is_dir():
            raise FileNotFoundError(f"Index directory not found: {p}")
        loaded = storage.load_index(p, readonly=readonly, verify_checksums=verify_checksums)
        m = loaded["manifest"]
        idx = cls(
            p,
            int(m["dimensions"]),
            m["metric"],
            int(m.get("max_neighbors", 16)),
            int(m.get("construction_ef", 200)),
            int(m.get("seed", 42)),
            readonly=readonly,
        )
        idx.format_version = int(m.get("format_version", FORMAT_VERSION))
        idx._vectors = loaded["vectors"]
        idx._capacity = int(idx._vectors.shape[0])
        idx._ids = loaded["ids"]
        idx._id_to_internal = {e: i for i, e in enumerate(idx._ids) if not loaded["deleted"][i]}
        # tombstoned ids stay reserved (cannot be re-added until compact)
        idx._deleted = np.asarray(loaded["deleted"], dtype=bool)
        idx._norms_sq = (
            np.einsum("ij,ij->i", idx._vectors, idx._vectors).astype(np.float64)
            if len(idx._vectors)
            else np.empty(0, dtype=np.float64)
        )
        idx._graph.layers = loaded["layers"]
        idx._graph.levels = loaded["levels"]
        idx._graph.entry_point = loaded["entry_point"]
        idx._graph.max_level = loaded["max_level"]
        if not readonly:
            # writers need a mutable in-memory copy instead of the read-only memmap
            idx._vectors = np.array(idx._vectors, dtype=np.float32)
            idx._lock = storage.acquire_lock(p)
        return idx

    def _ensure_open(self) -> None:
        if self._closed:
            raise ClosedIndexError("Index is closed")

    def _ensure_writable(self) -> None:
        self._ensure_open()
        if self.readonly:
            raise IncompatibleIndexError("Index was opened read-only")

    def __len__(self) -> int:
        return int((~self._deleted).sum()) if len(self._deleted) else 0

    @property
    def total_count(self) -> int:
        return len(self._ids)

    def _prepare(self, vectors: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            return normalize_rows(vectors)
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def _grow(self, extra: int) -> None:
        need = len(self._ids) + extra
        if need <= self._capacity:
            return
        new_cap = max(need, max(16, self._capacity * 2))
        new = np.zeros((new_cap, self.dimensions), dtype=np.float32)
        if len(self._ids):
            new[: len(self._ids)] = self._vectors[: len(self._ids)]
        self._vectors = new
        self._capacity = new_cap
        nn = np.zeros(new_cap, dtype=np.float64)
        nn[: len(self._norms_sq)] = self._norms_sq
        self._norms_sq = nn
        nd = np.ones(new_cap, dtype=bool)
        nd[: len(self._deleted)] = self._deleted
        self._deleted = nd

    def add(self, vectors, ids=None, replace: bool = False) -> list[str]:
        """Insert vectors incrementally into the graph.

        Duplicate IDs raise :class:`DuplicateIDError` unless ``replace=True``
        (old row is tombstoned, new row inserted).
        """
        self._ensure_writable()
        arr = validate_vectors(np.asarray(vectors), self.dimensions)
        n = arr.shape[0]
        if ids is None:
            start = len(self._ids)
            ext_ids = [f"vec-{start + i}" for i in range(n)]
        else:
            ext_ids = [str(x) for x in ids]
            if len(ext_ids) != n:
                raise InvalidVectorError(f"ids length {len(ext_ids)} != vectors {n}")
        if len(set(ext_ids)) != len(ext_ids):
            raise DuplicateIDError("Duplicate IDs within this batch")
        if not replace:
            tombstoned = {self._ids[i] for i in np.flatnonzero(self._deleted[: len(self._ids)])}
            for e in ext_ids:
                if e in self._id_to_internal:
                    raise DuplicateIDError(f"ID {e!r} already exists; pass replace=True to replace")
                if e in tombstoned:
                    raise DuplicateIDError(f"ID {e!r} is tombstoned; compact() before re-adding")
        if replace:
            for e in ext_ids:
                if e in self._id_to_internal:
                    self.delete(e)
        prepared = self._prepare(arr)
        self._grow(n)
        for row, ext in zip(prepared, ext_ids, strict=False):
            internal = len(self._ids)
            self._vectors[internal] = row
            self._norms_sq[internal] = float(row @ row)
            self._ids.append(ext)
            self._id_to_internal[ext] = internal
            self._deleted[internal] = False
            # the graph must only see rows inserted so far, not unused capacity
            count = len(self._ids)
            self._graph.insert(
                internal,
                self._vectors[:count],
                self._norms_sq[:count],
                self._deleted[:count],
            )
        return ext_ids

    def build(self) -> None:
        """Rebuild the graph from scratch over all live+tombstoned rows.

        Tombstoned rows remain in the graph as traversal waypoints until
        ``compact()``; rebuilding re-runs insertion in internal-id order
        with a freshly seeded RNG (deterministic for a given dataset).
        """
        self._ensure_writable()
        self._graph = Graph(
            self.dimensions, self.metric, self.max_neighbors, self.construction_ef, self.seed
        )
        count = len(self._ids)
        for internal in range(count):
            self._graph.insert(
                internal, self._vectors[:count], self._norms_sq[:count], self._deleted[:count]
            )

    def delete(self, ids) -> int:
        """Tombstone one ID (``str``) or many. Returns number deleted."""
        self._ensure_writable()
        if isinstance(ids, (str, bytes)):
            ids = [ids]
        count = 0
        for e in ids:
            e = str(e)
            internal = self._id_to_internal.pop(e, None)
            if internal is None:
                raise MissingIDError(f"ID {e!r} not found")
            self._deleted[internal] = True
            count += 1
        return count

    def compact(self) -> None:
        """Physically drop tombstoned rows and rebuild the graph."""
        self._ensure_writable()
        live = [i for i in range(len(self._ids)) if not self._deleted[i]]
        vecs = (
            np.array(self._vectors[live], dtype=np.float32)
            if live
            else np.empty((0, self.dimensions), np.float32)
        )
        ext = [self._ids[i] for i in live]
        self._vectors = vecs
        self._capacity = len(ext)
        self._ids = []
        self._id_to_internal = {}
        self._norms_sq = (
            np.einsum("ij,ij->i", vecs, vecs).astype(np.float64) if live else np.empty(0)
        )
        self._deleted = np.zeros(len(ext), dtype=bool)
        # rows are already normalised, so insert directly instead of via add()
        self._graph = Graph(
            self.dimensions, self.metric, self.max_neighbors, self.construction_ef, self.seed
        )
        for internal, ext_id in enumerate(ext):
            self._ids.append(ext_id)
            self._id_to_internal[ext_id] = internal
            self._graph.insert(internal, self._vectors, self._norms_sq, self._deleted)
        self.save()

    def get(self, id: str) -> np.ndarray:
        self._ensure_open()
        internal = self._id_to_internal.get(str(id))
        if internal is None:
            raise MissingIDError(f"ID {id!r} not found")
        return np.array(self._vectors[internal], dtype=np.float32)

    def iter_ids(self) -> Iterator[str]:
        """Yield the IDs of live (non-deleted) vectors in insertion order."""
        self._ensure_open()
        for internal, ext in enumerate(self._ids):
            if not self._deleted[internal]:
                yield ext

    def contains(self, id: str) -> bool:
        self._ensure_open()
        return str(id) in self._id_to_internal

    __contains__ = contains

    def _prepare_query(self, query) -> np.ndarray:
        q = validate_query(query, self.dimensions)
        if self.metric == "cosine":
            q = normalize_vector(q)
        return q

    def search(self, query, k: int = 10, ef_search: int = 100) -> list[SearchResult]:
        self._ensure_open()
        if k <= 0:
            raise ValueError("k must be positive")
        q = self._prepare_query(query)
        if len(self) == 0:
            return []
        q_norm = float(q @ q)
        raw = self._graph.search(
            q,
            q_norm,
            k,
            ef_search,
            self._vectors[: len(self._ids)],
            self._norms_sq[: len(self._ids)],
            self._deleted[: len(self._ids)],
        )
        scores = scores_from_distances(self.metric, np.array([d for d, _ in raw]))
        return [
            SearchResult(id=self._ids[nid], score=float(s), internal_id=int(nid))
            for (_d, nid), s in zip(raw, scores, strict=False)
        ]

    def search_batch(self, queries, k: int = 10, ef_search: int = 100) -> list[list[SearchResult]]:
        self._ensure_open()
        arr = validate_queries(queries, self.dimensions)
        return [self.search(q, k=k, ef_search=ef_search) for q in arr]

    def search_exact(self, query, k: int = 10) -> list[SearchResult]:
        """Exact baseline (chunked brute force) with the same result type."""
        self._ensure_open()
        if k <= 0:
            raise ValueError("k must be positive")
        q = self._prepare_query(query)
        live_mask = ~self._deleted[: len(self._ids)] if len(self._ids) else np.empty(0, bool)
        res = exact_search(
            self._vectors[: len(self._ids)], q.reshape(1, -1), k, self.metric, live_mask=live_mask
        )[0]
        idxs, scores = res
        return [
            SearchResult(id=self._ids[int(i)], score=float(s), internal_id=int(i))
            for i, s in zip(idxs, scores, strict=False)
        ]

    def _manifest(self) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "dimensions": self.dimensions,
            "metric": self.metric,
            "max_neighbors": self.max_neighbors,
            "construction_ef": self.construction_ef,
            "seed": self.seed,
            "count": len(self._ids),
            "live_count": len(self),
        }

    def save(self) -> None:
        self._ensure_writable()
        count = len(self._ids)
        manifest = storage.save_index(
            self.path,
            self._manifest(),
            np.ascontiguousarray(self._vectors[:count], dtype=np.float32),
            list(self._ids),
            self._deleted[:count],
            self._graph.layers,
            self._graph.levels,
            self._graph.entry_point,
            self._graph.max_level,
        )
        self._manifest_cache = manifest

    def validate(self, verify_checksums: bool = True) -> dict:
        """Validate in-memory + on-disk consistency. Returns a report dict."""
        self._ensure_open()
        errors: list[str] = []
        diag = self._graph.diagnostics()
        for node, node_layers in enumerate(self._graph.layers):
            for layer_list in node_layers:
                for nb in layer_list:
                    if nb < 0 or nb >= len(self._ids):
                        errors.append(f"node {node} has out-of-range neighbour {nb}")
        if len(set(self._ids)) != len(self._ids):
            errors.append("duplicate external ids in memory")
        if self.path.exists() and (self.path / "manifest.json").exists():
            try:
                storage.load_index(self.path, readonly=True, verify_checksums=verify_checksums)
            except Exception as exc:  # noqa: BLE001 - report, don't raise
                errors.append(f"on-disk validation failed: {exc}")
        return {
            "ok": not errors,
            "errors": errors,
            "live_count": len(self),
            "total_count": len(self._ids),
            "dimensions": self.dimensions,
            "metric": self.metric,
            "graph": diag,
        }

    @staticmethod
    def force_unlock(path) -> int | None:
        """Remove a stale writer lock left by a crashed process.

        Only call this when no other process is writing to the index; GraphIndex
        cannot tell a crashed writer from a running one. Returns the PID recorded
        in the lock, or None if there was no readable lock.
        """
        p = Path(path)
        if not p.is_dir():
            raise FileNotFoundError(f"Index directory not found: {p}")
        return storage.force_unlock(p)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        vec = self._vectors
        if isinstance(vec, np.memmap):
            try:
                vec._mmap.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        storage.release_lock(self._lock)
        self._lock = None

    def __enter__(self) -> GraphIndex:
        self._ensure_open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
