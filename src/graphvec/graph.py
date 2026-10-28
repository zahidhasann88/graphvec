"""GraphVec's own HNSW-style navigable small-world graph.

Design notes (see docs/architecture.md):
- Multi-layer graph; level of a node is drawn deterministically from the
  index RNG: ``level = floor(-ln(u) / ln(M))``.
- Layer 0 targets ``2*M`` neighbours, upper layers ``M``. Lists are pruned
  lazily, only once they would exceed twice their target size.
- Layer 0 is stored as flat arrays: ``adj0`` holds one fixed-width int32 row
  per node (``4*M`` slots, unused slots are -1) and ``deg0`` the number of
  used slots. These arrays can be memory-mapped straight from ``graph.bin``.
  The few nodes above layer 0 keep their upper-layer lists in ``upper``.
- Neighbour selection uses the standard diversification heuristic: a
  candidate is kept only if it is closer to the target than to every
  already-selected neighbour (with a fallback fill to keep connectivity).
- Deleted nodes are traversed through but never returned.
- All distances reduce to dot products: vectors are stored normalised for
  cosine; squared-Euclidean uses ``|a|^2+|b|^2-2a.b``.
"""

from __future__ import annotations

import math

import numpy as np

_CHUNK_ROWS = 65_536
MAX_LEVEL = 64
EXPAND_BATCH = 16


def _keep_closest(
    ids: np.ndarray, dists: np.ndarray, new_ids: np.ndarray, new_dists: np.ndarray, limit: int
) -> tuple[np.ndarray, np.ndarray]:
    """Merge new entries in and keep the ``limit`` smallest distances."""
    ids = np.concatenate((ids, new_ids))
    dists = np.concatenate((dists, new_dists))
    if ids.size > limit:
        keep = np.argpartition(dists, limit - 1)[:limit]
        ids, dists = ids[keep], dists[keep]
    return ids, dists


class Graph:
    def __init__(
        self,
        dimensions: int,
        metric: str,
        max_neighbors: int = 16,
        construction_ef: int = 200,
        seed: int = 42,
    ) -> None:
        self.dimensions = dimensions
        self.metric = metric
        self.max_neighbors = max_neighbors
        self.construction_ef = construction_ef
        self.seed = seed
        self.m0 = 2 * max_neighbors
        self.width = 2 * self.m0
        self.expand_batch = EXPAND_BATCH
        self._rng = np.random.default_rng(seed)
        self._level_mult = 1.0 / math.log(max_neighbors) if max_neighbors > 1 else 1.0
        self.count = 0
        self.levels = np.zeros(0, dtype=np.uint8)
        self.deg0 = np.zeros(0, dtype=np.int32)
        self.adj0 = np.full((0, self.width), -1, dtype=np.int32)
        # node -> neighbour lists for layers 1..level (only nodes with level > 0)
        self.upper: dict[int, list[list[int]]] = {}
        self.entry_point: int = -1
        self.max_level: int = -1

    def load(
        self,
        levels: np.ndarray,
        deg0: np.ndarray,
        adj0: np.ndarray,
        upper: dict[int, list[list[int]]],
        entry_point: int,
        max_level: int,
    ) -> None:
        """Adopt stored graph arrays (plain arrays or read-only memmaps)."""
        self.levels, self.deg0, self.adj0 = levels, deg0, adj0
        self.upper = upper
        self.entry_point = int(entry_point)
        self.max_level = int(max_level)
        self.count = len(levels)

    def release(self) -> None:
        """Drop array references so memory-mapped files can be closed."""
        self.levels = np.zeros(0, dtype=np.uint8)
        self.deg0 = np.zeros(0, dtype=np.int32)
        self.adj0 = np.full((0, self.width), -1, dtype=np.int32)
        self.upper = {}

    def _ensure_capacity(self, n: int) -> None:
        cap = len(self.deg0)
        if n <= cap:
            return
        new_cap = max(n, 16, cap * 2)
        levels = np.zeros(new_cap, dtype=np.uint8)
        deg0 = np.zeros(new_cap, dtype=np.int32)
        adj0 = np.full((new_cap, self.width), -1, dtype=np.int32)
        levels[: self.count] = self.levels[: self.count]
        deg0[: self.count] = self.deg0[: self.count]
        adj0[: self.count] = self.adj0[: self.count]
        self.levels, self.deg0, self.adj0 = levels, deg0, adj0

    def neighbors(self, node: int, layer: int) -> np.ndarray:
        if layer == 0:
            return self.adj0[node, : self.deg0[node]]
        return np.asarray(self.upper[node][layer - 1], dtype=np.int32)

    def _neighbors_of(self, nodes: np.ndarray, layer: int) -> np.ndarray:
        """Concatenated neighbour ids of several nodes (may contain repeats)."""
        if nodes.size == 1:
            return self.neighbors(int(nodes[0]), layer)
        if layer == 0:
            rows = np.asarray(self.adj0[nodes])
            return np.asarray(rows[rows >= 0])
        lists = [self.upper[n][layer - 1] for n in nodes.tolist()]
        return np.fromiter(
            (v for lst in lists for v in lst), dtype=np.int32, count=sum(map(len, lists))
        )

    def _dist(
        self, query: np.ndarray, q_norm_sq: float, vec: np.ndarray, vec_norm_sq: float
    ) -> float:
        dot = float(vec @ query)
        if self.metric == "cosine":
            return 1.0 - dot
        if self.metric == "dot":
            return -dot
        d = vec_norm_sq + q_norm_sq - 2.0 * dot
        return d if d > 0.0 else 0.0

    def _dists_to_many(
        self,
        query: np.ndarray,
        q_norm_sq: float,
        node_ids: np.ndarray,
        vectors: np.ndarray,
        norms_sq: np.ndarray,
    ) -> np.ndarray:
        """Batched internal distances from ``query`` to many nodes (one matvec)."""
        dots = vectors[node_ids] @ query
        if self.metric == "cosine":
            return np.asarray(1.0 - dots)
        if self.metric == "dot":
            return np.asarray(-dots)
        dists = norms_sq[node_ids] + q_norm_sq - 2.0 * dots
        np.maximum(dists, 0.0, out=dists)
        return np.asarray(dists)

    def _search_layer(
        self,
        query: np.ndarray,
        q_norm_sq: float,
        entry: int,
        ef: int,
        layer: int,
        vectors: np.ndarray,
        norms_sq: np.ndarray,
        deleted: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Best-first search of one layer: up to ``ef`` closest (ids, distances).

        With ``deleted`` given, deleted nodes are traversed but not returned.
        The frontier and result sets are NumPy arrays, and several frontier
        nodes are expanded per round, so each round is a few array operations
        instead of per-neighbour heap updates in Python.
        """
        visited = np.zeros(self.count, dtype=bool)
        visited[entry] = True
        d0 = self._dist(query, q_norm_sq, vectors[entry], float(norms_sq[entry]))
        # frontier: found but not yet expanded. top: the ef closest found,
        # deleted included; its worst distance bounds the traversal.
        f_ids = np.array([entry], dtype=np.int64)
        f_d = np.array([d0])
        t_ids, t_d = f_ids, f_d
        if deleted is None:
            r_ids, r_d = t_ids, t_d
        elif deleted[entry]:
            r_ids, r_d = np.empty(0, dtype=np.int64), np.empty(0)
        else:
            r_ids, r_d = t_ids, t_d
        bound = math.inf
        batch_size = self.expand_batch

        while f_ids.size:
            if f_ids.size > batch_size:
                take = np.argpartition(f_d, batch_size - 1)[:batch_size]
                rest = np.ones(f_ids.size, dtype=bool)
                rest[take] = False
                batch, f_ids, f_d = f_ids[take], f_ids[rest], f_d[rest]
            else:
                batch, f_ids, f_d = f_ids, f_ids[:0], f_d[:0]
            nbrs = self._neighbors_of(batch, layer)
            nbrs = nbrs[~visited[nbrs]]
            if nbrs.size:
                if batch.size > 1:
                    nbrs = np.unique(nbrs)
                visited[nbrs] = True
                dists = self._dists_to_many(query, q_norm_sq, nbrs, vectors, norms_sq)
                keep = dists < bound
                if not keep.all():
                    nbrs, dists = nbrs[keep], dists[keep]
                if nbrs.size:
                    nbrs = nbrs.astype(np.int64, copy=False)
                    t_ids, t_d = _keep_closest(t_ids, t_d, nbrs, dists, ef)
                    if deleted is not None:
                        live = ~deleted[nbrs]
                        r_ids, r_d = _keep_closest(r_ids, r_d, nbrs[live], dists[live], ef)
                    f_ids = np.concatenate((f_ids, nbrs))
                    f_d = np.concatenate((f_d, dists))
            if t_ids.size >= ef:
                bound = float(t_d.max())
                close = f_d <= bound
                if not close.all():
                    f_ids, f_d = f_ids[close], f_d[close]
        if deleted is None:
            return t_ids, t_d
        return r_ids, r_d

    def _greedy_layer(
        self,
        query: np.ndarray,
        q_norm_sq: float,
        entry: int,
        layer: int,
        vectors: np.ndarray,
        norms_sq: np.ndarray,
    ) -> int:
        best = entry
        best_d = self._dist(query, q_norm_sq, vectors[best], float(norms_sq[best]))
        while True:
            nbrs = self.neighbors(best, layer)
            if not nbrs.size:
                return best
            dists = self._dists_to_many(query, q_norm_sq, nbrs, vectors, norms_sq)
            j = int(np.argmin(dists))
            if float(dists[j]) < best_d:
                best, best_d = int(nbrs[j]), float(dists[j])
            else:
                return best

    def _select(
        self,
        cand_ids: np.ndarray,
        cand_dists: np.ndarray,
        m: int,
        vectors: np.ndarray,
        norms_sq: np.ndarray,
    ) -> np.ndarray:
        """Pick up to ``m`` diverse neighbours from candidates (distances to target)."""
        dists = np.asarray(cand_dists, dtype=np.float64)
        order = np.lexsort((cand_ids, dists))
        ids, dists = cand_ids[order], dists[order]
        n = len(ids)
        if n <= m:
            return ids
        sub = vectors[ids]
        ns = norms_sq[ids] if self.metric == "squared_euclidean" else None
        # A candidate is blocked once some selected neighbour is closer to it
        # than the target is; each pick blocks others with one matvec.
        blocked = np.zeros(n, dtype=bool)
        selected: list[int] = []
        i = 0
        while len(selected) < m and i < n:
            if blocked[i]:
                free = np.flatnonzero(~blocked[i:])
                if not free.size:
                    break
                i += int(free[0])
            selected.append(i)
            # only later candidates can still be blocked by this pick
            dots = sub[i + 1 :] @ sub[i]
            if self.metric == "cosine":
                pair = 1.0 - dots
            elif self.metric == "dot":
                pair = -dots
            else:
                assert ns is not None
                pair = ns[i + 1 :] + ns[i] - 2.0 * dots
                np.maximum(pair, 0.0, out=pair)
            blocked[i + 1 :] |= pair < dists[i + 1 :]
            i += 1
        if len(selected) < m:  # fill for connectivity
            chosen = np.zeros(n, dtype=bool)
            chosen[selected] = True
            selected += np.flatnonzero(~chosen)[: m - len(selected)].tolist()
        return ids[selected]

    def _prune(
        self, node: int, cands: np.ndarray, m: int, vectors: np.ndarray, norms_sq: np.ndarray
    ) -> np.ndarray:
        dists = self._dists_to_many(vectors[node], float(norms_sq[node]), cands, vectors, norms_sq)
        return self._select(cands, dists, m, vectors, norms_sq)

    def _connect(
        self,
        node: int,
        chosen: np.ndarray,
        layer: int,
        vectors: np.ndarray,
        norms_sq: np.ndarray,
    ) -> None:
        """Link ``node`` to ``chosen`` in both directions, pruning full lists."""
        if layer == 0:
            k = len(chosen)
            self.adj0[node, :k] = chosen
            self.deg0[node] = k
            deg = self.deg0[chosen]
            room = deg < self.width
            fits = chosen[room]
            self.adj0[fits, deg[room]] = node
            self.deg0[fits] += 1
            for nb in chosen[~room].tolist():
                cands = np.append(self.adj0[nb], np.int32(node))
                kept = self._prune(nb, cands, self.m0, vectors, norms_sq)
                self.adj0[nb] = -1
                self.adj0[nb, : len(kept)] = kept
                self.deg0[nb] = len(kept)
            return
        self.upper[node][layer - 1] = chosen.tolist()
        for nb in chosen.tolist():
            lst = self.upper[nb][layer - 1]
            lst.append(node)
            if len(lst) > 2 * self.max_neighbors:
                cands = np.asarray(lst, dtype=np.int32)
                kept = self._prune(nb, cands, self.max_neighbors, vectors, norms_sq)
                self.upper[nb][layer - 1] = kept.tolist()

    def _random_level(self) -> int:
        u = float(self._rng.random())
        u = max(u, 1e-12)
        lvl: int = int(-math.log(u) * self._level_mult)
        return lvl

    def insert(
        self,
        internal_id: int,
        vectors: np.ndarray,
        norms_sq: np.ndarray,
        deleted: np.ndarray | None = None,
    ) -> None:
        """Insert node ``internal_id``; ``vectors`` must already include it."""
        level = self._random_level()
        self._ensure_capacity(internal_id + 1)
        self.count = max(self.count, internal_id + 1)
        self.levels[internal_id] = level
        self.deg0[internal_id] = 0
        self.adj0[internal_id] = -1
        if level > 0:
            self.upper[internal_id] = [[] for _ in range(level)]
        else:
            self.upper.pop(internal_id, None)

        if self.entry_point == -1:
            self.entry_point = internal_id
            self.max_level = level
            return

        vec = vectors[internal_id]
        q_norm = float(norms_sq[internal_id])
        cur = self.entry_point
        # greedy descent above the new node's level
        for layer in range(self.max_level, level, -1):
            cur = self._greedy_layer(vec, q_norm, cur, layer, vectors, norms_sq)

        ef = max(self.construction_ef, self.max_neighbors)
        for layer in range(min(level, self.max_level), -1, -1):
            ids, dists = self._search_layer(vec, q_norm, cur, ef, layer, vectors, norms_sq)
            if ids.size:
                cur = int(ids[np.argmin(dists)])  # entry for the next layer down
            other = ids != internal_id
            if other.any():
                m = self.m0 if layer == 0 else self.max_neighbors
                chosen = self._select(
                    ids[other].astype(np.int32), dists[other], m, vectors, norms_sq
                )
                self._connect(internal_id, chosen, layer, vectors, norms_sq)

        if level > self.max_level:
            self.max_level = level
            self.entry_point = internal_id

    def search(
        self, query, q_norm_sq, k, ef_search, vectors, norms_sq, deleted
    ) -> list[tuple[float, int]]:
        """Return list of (internal_distance, internal_id), best first."""
        if self.entry_point == -1:
            return []
        cur = self.entry_point
        for layer in range(self.max_level, 0, -1):
            cur = self._greedy_layer(query, q_norm_sq, cur, layer, vectors, norms_sq)
        ef = max(ef_search, k)
        ids, dists = self._search_layer(
            query, q_norm_sq, cur, ef, 0, vectors, norms_sq, deleted=deleted
        )
        order = np.lexsort((ids, dists))[:k]
        return list(zip(dists[order].tolist(), ids[order].tolist(), strict=True))

    def check(self, n: int) -> list[str]:
        """Structural problems that would make traversal unsafe or wrong."""
        errors: list[str] = []
        if self.count != n:
            return [f"graph has {self.count} nodes but the index has {n}"]
        if n == 0:
            if self.entry_point != -1:
                errors.append("empty graph has an entry point")
            return errors
        if not 0 <= self.entry_point < n:
            return [f"entry point {self.entry_point} out of range"]
        levels = self.levels[:n]
        if int(np.max(levels)) > MAX_LEVEL:
            errors.append("node level out of range")
        elif self.max_level != int(levels[self.entry_point]):
            errors.append("max_level disagrees with the entry point's level")
        deg = self.deg0[:n]
        if int(np.min(deg)) < 0 or int(np.max(deg)) > self.width:
            return errors + ["layer-0 degree out of range"]
        slots = np.arange(self.width)
        for start in range(0, n, _CHUNK_ROWS):
            end = min(start + _CHUNK_ROWS, n)
            rows = np.asarray(self.adj0[start:end])
            in_use = slots < deg[start:end, None]
            used = rows[in_use]
            if used.size and (int(np.min(used)) < 0 or int(np.max(used)) >= n):
                errors.append("layer-0 neighbour id out of range")
                break
            if not (rows[~in_use] == -1).all():
                errors.append("layer-0 unused neighbour slots are not empty")
                break
        if len(self.upper) != int(np.count_nonzero(levels)):
            errors.append("upper-layer lists disagree with node levels")
        for node, lists in self.upper.items():
            if not 0 <= node < n or len(lists) != int(levels[node]):
                errors.append(f"node {node} has inconsistent upper layers")
                break
            if any(len(lst) > self.width or any(not 0 <= v < n for v in lst) for lst in lists):
                errors.append(f"node {node} has an invalid upper-layer neighbour list")
                break
        return errors

    def diagnostics(self) -> dict:
        n = self.count
        upper_edges = sum(len(lst) for lists in self.upper.values() for lst in lists)
        total_edges = int(self.deg0[:n].sum()) + upper_edges
        reachable = 0
        if self.entry_point != -1 and n:
            # breadth-first over layer 0, one frontier at a time
            seen = np.zeros(n, dtype=bool)
            seen[self.entry_point] = True
            frontier = np.array([self.entry_point], dtype=np.int64)
            slots = np.arange(self.width)
            while frontier.size:
                found = []
                for start in range(0, frontier.size, _CHUNK_ROWS):
                    part = frontier[start : start + _CHUNK_ROWS]
                    rows = np.asarray(self.adj0[part])
                    found.append(rows[slots < self.deg0[part][:, None]])
                nbrs = np.unique(np.concatenate(found))
                frontier = nbrs[~seen[nbrs]].astype(np.int64)
                seen[frontier] = True
            reachable = int(seen.sum())
        return {
            "nodes": n,
            "edges": total_edges,
            "max_level": self.max_level,
            "entry_point": self.entry_point,
            "layer0_reachable": reachable,
            "avg_degree": (total_edges / n if n else 0.0),
        }
