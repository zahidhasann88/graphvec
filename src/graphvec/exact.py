"""Exact (brute-force) nearest-neighbour search, chunked over the dataset.

Never materialises a (num_queries x dataset) matrix larger than one chunk.
"""

from __future__ import annotations

import numpy as np

from graphvec.metrics import is_higher_better, pairwise_scores


def exact_search(
    matrix: np.ndarray,
    queries: np.ndarray,
    k: int,
    metric: str,
    live_mask: np.ndarray | None = None,
    chunk_size: int = 1024,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return per query ``(internal_indices, public_scores)`` sorted best-first.

    Stable tie-breaking: equal scores are ordered by ascending internal id
    (achieved via a stable sort on the score key).
    """
    n = matrix.shape[0]
    results: list[tuple[np.ndarray, np.ndarray]] = []
    higher = is_higher_better(metric)
    live = np.ones(n, dtype=bool) if live_mask is None else live_mask
    live_idx = np.flatnonzero(live)
    if n == 0 or live_idx.size == 0 or k <= 0:
        return [(np.empty(0, dtype=np.int64), np.empty(0)) for _ in range(len(queries))]
    norms_sq = np.einsum("ij,ij->i", matrix, matrix) if metric == "squared_euclidean" else None
    take = min(k, live_idx.size)
    for q in queries:
        best_idx = np.empty(0, dtype=np.int64)
        best_scores = np.empty(0, dtype=np.float64)
        for start in range(0, live_idx.size, chunk_size):
            chunk_ids = live_idx[start : start + chunk_size]
            chunk_norms = None if norms_sq is None else norms_sq[chunk_ids]
            scores = pairwise_scores(metric, q, matrix[chunk_ids], chunk_norms)
            cand_idx = np.concatenate([best_idx, chunk_ids])
            cand_scores = np.concatenate([best_scores, scores])
            if cand_scores.size > take:
                # partition then stable-sort only the shortlist
                part = np.argpartition(-cand_scores if higher else cand_scores, take - 1)[:take]
                cand_idx, cand_scores = cand_idx[part], cand_scores[part]
            best_idx, best_scores = cand_idx, cand_scores

        # stable tie-break by internal id for equal scores
        def sort_key(i: int):
            return (-best_scores[i] if higher else best_scores[i], int(best_idx[i]))  # noqa: B023

        order: list[int] = sorted(range(len(best_idx)), key=sort_key)
        results.append((best_idx[order], best_scores[order]))
    return results
