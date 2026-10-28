"""Deterministic synthetic datasets for benchmarks."""

from __future__ import annotations

import numpy as np


def make_dataset(n: int, d: int, seed: int = 42, clustered: bool = True):
    rng = np.random.default_rng(seed)
    if clustered:
        centers = rng.standard_normal((max(1, n // 200), d)).astype(np.float32)
        assign = rng.integers(0, len(centers), size=n)
        vecs = (centers[assign] + 0.3 * rng.standard_normal((n, d))).astype(np.float32)
    else:
        vecs = rng.standard_normal((n, d)).astype(np.float32)
    queries = (
        vecs[rng.integers(0, n, size=min(100, n))] + 0.05 * rng.standard_normal((min(100, n), d))
    ).astype(np.float32)
    return vecs, queries
