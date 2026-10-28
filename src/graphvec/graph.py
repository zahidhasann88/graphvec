"""GraphVec's own HNSW-style navigable small-world graph.

Design notes (see docs/architecture.md):
- Multi-layer graph; level of a node is drawn deterministically from the
  index RNG: ``level = floor(-ln(u) / ln(M))``.
- Layer 0 allows ``2*M`` neighbours, upper layers ``M``.
- Neighbour selection uses the standard diversification heuristic: a
  candidate is kept only if it is closer to the target than to every
  already-selected neighbour (with a fallback fill to keep connectivity).
- Deleted nodes are traversed through but never returned.
- All distances reduce to dot products: vectors are stored normalised for
  cosine; squared-Euclidean uses ``|a|^2+|b|^2-2a.b``.
"""

from __future__ import annotations

import heapq
import math

import numpy as np


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
        self._rng = np.random.default_rng(seed)
        self._level_mult = 1.0 / math.log(max_neighbors) if max_neighbors > 1 else 1.0
        # adjacency[node][layer] -> list of neighbour internal ids
        self.layers: list[list[list[int]]] = []
        self.levels: list[int] = []
        self.entry_point: int = -1
        self.max_level: int = -1

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

    def _dist_nodes(self, a: int, b: int, vectors: np.ndarray, norms_sq: np.ndarray) -> float:
        return self._dist(vectors[a], float(norms_sq[a]), vectors[b], float(norms_sq[b]))

    def _dists_to_many(
        self,
        query: np.ndarray,
        q_norm_sq: float,
        node_ids: list[int],
        vectors: np.ndarray,
        norms_sq: np.ndarray,
    ) -> np.ndarray:
        """Batched internal distances from ``query`` to many nodes (one matvec)."""
        sub = vectors[node_ids]
        dots = sub @ query
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
    ) -> list[tuple[float, int]]:
        """Return result heap as list of (-dist, id) (max-heap by distance)."""
        visited: set[int] = {entry}
        d0 = self._dist(query, q_norm_sq, vectors[entry], float(norms_sq[entry]))
        candidates: list[tuple[float, int]] = [(d0, entry)]  # min-heap
        # Results are a max-heap via negative distance. Deleted nodes stay in
        # the traversal frontier but are never added to results.
        results: list[tuple[float, int]] = []
        if deleted is None or not deleted[entry]:
            results.append((-d0, entry))
        bound_all = deleted is None
        worst_all: list[tuple[float, int]] = [(-d0, entry)]  # traversal bound incl. deleted

        while candidates:
            dist, node = heapq.heappop(candidates)
            worst = -worst_all[0][0] if worst_all else math.inf
            if dist > worst and len(worst_all) >= ef:
                break
            fresh = [nb for nb in self.layers[node][layer] if nb not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            dists = self._dists_to_many(query, q_norm_sq, fresh, vectors, norms_sq)
            for nb, d_raw in zip(fresh, dists, strict=True):
                d = float(d_raw)
                worst = -worst_all[0][0] if worst_all else math.inf
                if len(worst_all) < ef or d < worst:
                    heapq.heappush(candidates, (d, nb))
                    heapq.heappush(worst_all, (-d, nb))
                    if len(worst_all) > ef:
                        heapq.heappop(worst_all)
                    if deleted is None or not deleted[nb]:
                        heapq.heappush(results, (-d, nb))
                        if len(results) > ef:
                            heapq.heappop(results)
            _ = bound_all
        return results

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
            neigh = self.layers[best][layer]
            if not neigh:
                return best
            dists = self._dists_to_many(query, q_norm_sq, neigh, vectors, norms_sq)
            j = int(np.argmin(dists))
            if float(dists[j]) < best_d:
                best, best_d = neigh[j], float(dists[j])
            else:
                return best

    def _select_neighbors(
        self,
        target_vec,
        target_norm,
        candidates: list[tuple[float, int]],
        m: int,
        vectors,
        norms_sq,
    ) -> list[int]:
        return self._select(candidates, m, vectors, norms_sq, heuristic=True)

    def _select(
        self,
        candidates: list[tuple[float, int]],
        m: int,
        vectors,
        norms_sq,
        heuristic: bool,
    ) -> list[int]:
        if len(candidates) <= m:
            return [cid for _, cid in sorted(candidates)]
        candidates = sorted(candidates)  # by distance to target
        _ = heuristic  # both paths use the heuristic; kept for call-site clarity
        cand_ids = [cid for _, cid in candidates]
        cand_dists = [float(d) for d, _ in candidates]
        # Pairwise distances among candidates in one matmul, then the
        # diversification heuristic runs on plain floats.
        sub = vectors[cand_ids]
        dots = sub @ sub.T
        if self.metric == "cosine":
            pair = 1.0 - dots
        elif self.metric == "dot":
            pair = -dots
        else:
            ns = norms_sq[cand_ids]
            pair = ns[:, None] + ns[None, :] - 2.0 * dots
            np.maximum(pair, 0.0, out=pair)
        pair_list = pair.tolist()  # python floats: far faster than np scalar indexing
        selected_idx: list[int] = []
        for i, dist in enumerate(cand_dists):
            if len(selected_idx) >= m:
                break
            row = pair_list[i]
            if all(row[j] >= dist for j in selected_idx):
                selected_idx.append(i)
        if len(selected_idx) < m:  # fill for connectivity
            chosen = set(selected_idx)
            for i in range(len(cand_ids)):
                if len(selected_idx) >= m:
                    break
                if i not in chosen:
                    selected_idx.append(i)
        return [cand_ids[i] for i in selected_idx]

    def _prune(self, node: int, layer: int, vectors, norms_sq) -> None:
        m = self.max_neighbors * 2 if layer == 0 else self.max_neighbors
        neigh = self.layers[node][layer]
        # Lazy pruning: only pay for heuristic re-selection once the list
        # has grown to twice its target size; it is then cut back to m.
        # This amortises pruning cost without losing diversified links.
        if len(neigh) <= 2 * m:
            return
        dists = self._dists_to_many(vectors[node], float(norms_sq[node]), neigh, vectors, norms_sq)
        cands = [(float(d), nb) for d, nb in zip(dists, neigh, strict=True)]
        self.layers[node][layer] = self._select(cands, m, vectors, norms_sq, heuristic=False)

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
        while len(self.levels) <= internal_id:
            self.levels.append(0)
            self.layers.append([])
        self.levels[internal_id] = level
        self.layers[internal_id] = [[] for _ in range(level + 1)]

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
            res = self._search_layer(vec, q_norm, cur, ef, layer, vectors, norms_sq, deleted=None)
            cands = [(-negd, nid) for negd, nid in res if nid != internal_id]
            m = self.max_neighbors * 2 if layer == 0 else self.max_neighbors
            chosen = self._select_neighbors(vec, q_norm, cands, m, vectors, norms_sq)
            self.layers[internal_id][layer] = chosen
            for nb in chosen:
                if internal_id not in self.layers[nb][layer]:
                    self.layers[nb][layer].append(internal_id)
                self._prune(nb, layer, vectors, norms_sq)
            if res:
                # entry for next layer down: closest found
                # (heap items are (-dist, id); max == smallest dist)
                cur = max(res)[1]

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
        res = self._search_layer(query, q_norm_sq, cur, ef, 0, vectors, norms_sq, deleted=deleted)
        out = sorted((-negd, nid) for negd, nid in res)
        return out[:k]

    def diagnostics(self) -> dict:
        total_edges = sum(len(layer_list) for node in self.layers for layer_list in node)
        nodes = len(self.layers)
        # connectivity on layer 0 via BFS from entry point
        reachable = 0
        if self.entry_point != -1 and nodes:
            seen = {self.entry_point}
            stack = [self.entry_point]
            while stack:
                cur = stack.pop()
                for nb in self.layers[cur][0]:
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            reachable = len(seen)
        return {
            "nodes": nodes,
            "edges": total_edges,
            "max_level": self.max_level,
            "entry_point": self.entry_point,
            "layer0_reachable": reachable,
            "avg_degree": (total_edges / nodes if nodes else 0.0),
        }
