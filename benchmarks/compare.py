"""Optional comparison against hnswlib (never a core dependency).

Run in an environment with ``pip install hnswlib``. Uses identical data,
seed, k and ef for both engines and reports measured recall/latency.
Exits with a clear message when hnswlib is unavailable.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))


def main(n: int = 10_000, d: int = 128) -> None:
    try:
        import hnswlib
    except ImportError:
        print("hnswlib not installed; install the 'bench' extra to compare. Skipping.")
        return
    import tempfile

    from datasets import make_dataset

    from graphvec import GraphIndex

    vecs, queries = make_dataset(n, d)
    t0 = time.perf_counter()
    h = hnswlib.Index(space="cosine", dim=d)
    h.init_index(max_elements=n, ef_construction=200, M=16)
    h.add_items(vecs, np.arange(n))
    h.set_ef(100)
    hb = time.perf_counter() - t0

    tmp = Path(tempfile.mkdtemp()) / "g.gvec"
    t0 = time.perf_counter()
    idx = GraphIndex.create(tmp, dimensions=d, metric="cosine")
    idx.add(vecs, ids=[str(i) for i in range(n)])
    gb = time.perf_counter() - t0

    def bench(fn):
        lat = []
        for q in queries:
            t0 = time.perf_counter()
            fn(q)
            lat.append((time.perf_counter() - t0) * 1000)
        return float(np.percentile(lat, 50)), float(np.percentile(lat, 95))

    exact = [{r.internal_id for r in idx.search_exact(q, k=10)} for q in queries]

    def recall_h():
        return np.mean(
            [
                len(set(h.knn_query(q, k=10)[0][0]) & e) / 10
                for q, e in zip(queries, exact, strict=False)
            ]
        )

    def recall_g():
        return np.mean(
            [
                len({r.internal_id for r in idx.search(q, k=10, ef_search=100)} & e) / 10
                for q, e in zip(queries, exact, strict=False)
            ]
        )

    print(
        "hnswlib: "
        f"build={hb:.2f}s p50/p95={bench(lambda q: h.knn_query(q, k=10))} "
        f"recall@10={recall_h():.3f}"
    )
    print(
        "graphvec: "
        f"build={gb:.2f}s p50/p95="
        f"{bench(lambda q: idx.search(q, k=10, ef_search=100))} "
        f"recall@10={recall_g():.3f}"
    )
    idx.close()


if __name__ == "__main__":
    main()
