"""Distance / similarity metrics.

Semantics (documented, stable):
- ``squared_euclidean``: score is squared L2 distance, *lower is better*.
  No square root is taken (monotonic in L2, sufficient for ranking).
- ``cosine``: score is cosine similarity in [-1, 1], *higher is better*.
  Vectors are L2-normalised at insert time (zero vectors stay zero and
  have similarity 0 with everything). Queries are normalised per search.
- ``dot``: score is dot product, *higher is better*.

Internally every metric is reduced to a minimisation distance:
euclidean -> squared L2, cosine -> 1 - cos, dot -> -dot.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Metric = Literal["cosine", "dot", "squared_euclidean"]
SUPPORTED_METRICS: tuple[str, ...] = ("cosine", "dot", "squared_euclidean")


def is_higher_better(metric: str) -> bool:
    return metric in ("cosine", "dot")


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """Return a float32 copy with rows L2-normalised; zero rows stay zero."""
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    out = np.zeros_like(arr)
    np.divide(arr, norms, out=out, where=norms > 0)
    return out


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return np.zeros_like(arr)
    return (arr / norm).astype(np.float32)


def scores_from_distances(metric: str, distances: np.ndarray) -> np.ndarray:
    """Convert internal minimisation distances back to public scores."""
    if metric == "cosine":
        return 1.0 - distances
    if metric == "dot":
        return -distances
    return distances


def pairwise_scores(
    metric: str, query: np.ndarray, matrix: np.ndarray, matrix_norms_sq: np.ndarray | None = None
) -> np.ndarray:
    """Public scores of one query against every row of ``matrix``.

    ``matrix`` is assumed already prepared (normalised for cosine).
    ``query`` is assumed already prepared likewise.
    """
    dots = matrix @ query
    if metric == "cosine":
        # Rows/query normalised -> dot == cosine; zero rows give 0.
        return np.asarray(dots, dtype=np.float64)
    if metric == "dot":
        return np.asarray(dots, dtype=np.float64)
    # squared euclidean: ||a||^2 + ||b||^2 - 2 a.b  (clip tiny negatives)
    if matrix_norms_sq is None:
        matrix_norms_sq = np.einsum("ij,ij->i", matrix, matrix)
    q_norm = float(query @ query)
    dists = matrix_norms_sq + q_norm - 2.0 * dots
    np.maximum(dists, 0.0, out=dists)
    return np.asarray(dists, dtype=np.float64)
