"""Deterministic synthetic datasets for benchmarks.

Queries are held-out points drawn from the same distribution as the indexed
vectors, not perturbed copies of them, so recall is not trivially perfect.
Generation is chunked float32 and can stream straight into a ``.npy`` file,
so large datasets never need a float64 copy or the whole set in RAM.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CHUNK = 65_536


def make_dataset(
    n: int,
    d: int,
    seed: int = 42,
    n_queries: int = 100,
    clustered: bool = True,
    out: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(vectors, queries)``; memory-mapped from ``out`` when given.

    ``clustered`` draws points around ``(n + n_queries) // 200`` random centres
    (noise 0.3 per dimension); otherwise points are standard Gaussian, which
    is the hardest case for graph indexes.
    """
    rng = np.random.default_rng(seed)
    total = n + n_queries
    if out is None:
        data = np.empty((total, d), dtype=np.float32)
    else:
        data = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32, shape=(total, d))
    centers = (
        rng.standard_normal((max(1, total // 200), d), dtype=np.float32) if clustered else None
    )
    for start in range(0, total, CHUNK):
        m = min(CHUNK, total - start)
        chunk = rng.standard_normal((m, d), dtype=np.float32)
        if centers is not None:
            chunk *= 0.3
            chunk += centers[rng.integers(0, len(centers), size=m)]
        data[start : start + m] = chunk
    if isinstance(data, np.memmap):
        data.flush()
    return data[:n], data[n:]
