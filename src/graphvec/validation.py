"""Input validation at API boundaries."""

from __future__ import annotations

import numpy as np

from graphvec.exceptions import InvalidDimensionError, InvalidVectorError, UnsupportedMetricError
from graphvec.metrics import SUPPORTED_METRICS


def validate_metric(metric: str) -> str:
    if metric not in SUPPORTED_METRICS:
        raise UnsupportedMetricError(
            f"Unsupported metric {metric!r}; choose one of {list(SUPPORTED_METRICS)}"
        )
    return metric


def validate_vectors(vectors: np.ndarray, dimensions: int) -> np.ndarray:
    """Return vectors as a contiguous float32 (n, d) array, or raise."""
    arr = np.asarray(vectors)
    if arr.ndim != 2:
        raise InvalidDimensionError(f"Expected a 2-D array of vectors, got shape {arr.shape}")
    if arr.shape[1] != dimensions:
        raise InvalidDimensionError(f"Expected dimension {dimensions}, got {arr.shape[1]}")
    if arr.shape[0] == 0:
        return np.empty((0, dimensions), dtype=np.float32)
    if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.complexfloating):
        raise InvalidVectorError(f"Vectors must be real numeric, got dtype {arr.dtype}")
    out = np.ascontiguousarray(arr, dtype=np.float32)
    if not np.isfinite(out).all():
        raise InvalidVectorError("Vectors must not contain NaN or infinite values")
    return out


def validate_query(query: np.ndarray, dimensions: int) -> np.ndarray:
    arr = np.asarray(query)
    if arr.ndim != 1:
        raise InvalidDimensionError(f"Expected a 1-D query vector, got shape {arr.shape}")
    if arr.shape[0] != dimensions:
        raise InvalidDimensionError(f"Expected dimension {dimensions}, got {arr.shape[0]}")
    out = np.ascontiguousarray(arr, dtype=np.float32)
    if not np.isfinite(out).all():
        raise InvalidVectorError("Query must not contain NaN or infinite values")
    return out


def validate_queries(queries: np.ndarray, dimensions: int) -> np.ndarray:
    arr = np.asarray(queries)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return validate_vectors(arr, dimensions)
