"""Reproducible GraphVec benchmark runner.

Usage:
    PYTHONPATH=src python benchmarks/run.py --n 10000 --dims 128 --metric cosine
Reports *measured* numbers only (build time, latency percentiles, recall@k
against exact search, disk size). Records seed, versions and parameters.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from datasets import make_dataset  # noqa: E402

import graphvec  # noqa: E402
from graphvec import GraphIndex  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--dims", type=int, default=128)
    ap.add_argument("--metric", default="cosine")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ef-search", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    vecs, queries = make_dataset(args.n, args.dims, seed=args.seed)
    tmp = tempfile.mkdtemp()
    path = Path(tmp) / "bench.gvec"
    t0 = time.perf_counter()
    idx = GraphIndex.create(path, dimensions=args.dims, metric=args.metric, seed=args.seed)
    idx.add(vecs, ids=[f"v{i}" for i in range(args.n)])
    build_s = time.perf_counter() - t0
    idx.save()

    lat, hits, total = [], 0, 0
    for q in queries:
        t0 = time.perf_counter()
        res = idx.search(q, k=args.k, ef_search=args.ef_search)
        lat.append((time.perf_counter() - t0) * 1000)
        exact = {r.id for r in idx.search_exact(q, k=args.k)}
        hits += len({r.id for r in res} & exact)
        total += len(exact)
    disk = sum(p.stat().st_size for p in path.iterdir() if p.is_file())
    arr = np.array(lat)
    report = {
        "graphvec_version": graphvec.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "n": args.n,
        "dims": args.dims,
        "metric": args.metric,
        "k": args.k,
        "ef_search": args.ef_search,
        "seed": args.seed,
        "build_seconds": build_s,
        "insert_vectors_per_sec": args.n / build_s,
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "qps_single": float(1000 / arr.mean()),
        f"recall@{args.k}": hits / total if total else 0.0,
        "index_bytes_on_disk": int(disk),
    }
    idx.close()
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
