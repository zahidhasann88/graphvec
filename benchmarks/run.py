"""Reproducible GraphVec benchmark runner.

Usage:
    python benchmarks/run.py --n 100000 --dims 384 --out results.json

Builds an index in batches from a memory-mapped dataset, reopens it
read-only (memory-mapped) for querying, computes exact ground truth, then
reports *measured* numbers only: build time, insertion rate, peak memory,
open time, disk size, and recall@1 / recall@k with latency percentiles for
each ``ef_search``. Records seed, parameters, versions and hardware.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

_SRC = Path(__file__).parent.parent / "src"
if "--graphvec-src" in sys.argv:  # must be known before graphvec is imported
    _SRC = Path(sys.argv[sys.argv.index("--graphvec-src") + 1])
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).parent))

from datasets import make_dataset  # noqa: E402

import graphvec  # noqa: E402
from graphvec import GraphIndex  # noqa: E402


def peak_rss_mb() -> float | None:
    """Peak resident memory of this process, if the platform reports it."""
    try:
        import psutil

        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)  # Windows
        if peak is not None:
            return round(peak / 2**20, 1)
    except ImportError:
        pass
    try:
        import resource

        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(kb / 2**10 if sys.platform != "darwin" else kb / 2**20, 1)
    except ImportError:
        return None


def environment() -> dict:
    env = {
        "graphvec_version": graphvec.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil

        env["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    return env


def ground_truth(index: GraphIndex, queries: np.ndarray, k: int) -> list[list[str]]:
    return [[r.id for r in index.search_exact(q, k=k)] for q in queries]


def measure(search, queries: np.ndarray, truth: list[list[str]], k: int) -> dict:
    """Latency percentiles and recall@1 / recall@k for one search function."""
    lat, hits1, hitsk = [], 0, 0
    for q, t in zip(queries, truth, strict=True):
        t0 = time.perf_counter()
        ids = search(q)
        lat.append((time.perf_counter() - t0) * 1000)
        hits1 += int(bool(ids) and ids[0] == t[0])
        hitsk += len(set(ids[:k]) & set(t[:k]))
    arr = np.array(lat)
    return {
        "recall@1": hits1 / len(queries),
        f"recall@{k}": hitsk / (k * len(queries)),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "qps_single_thread": round(float(1000 / arr.mean()), 1),
    }


def build_graphvec(path: Path, vecs: np.ndarray, args) -> dict:
    t0 = time.perf_counter()
    idx = GraphIndex.create(
        path,
        dimensions=vecs.shape[1],
        metric=args.metric,
        max_neighbors=args.m,
        construction_ef=args.construction_ef,
        seed=args.seed,
    )
    last_rate = None
    for start in range(0, len(vecs), args.batch_size):
        chunk = np.asarray(vecs[start : start + args.batch_size])
        t1 = time.perf_counter()
        idx.add(chunk, ids=[f"v{i}" for i in range(start, start + len(chunk))])
        last_rate = len(chunk) / (time.perf_counter() - t1)
        done = start + len(chunk)
        print(f"  inserted {done}/{len(vecs)} ({last_rate:.0f}/s)", file=sys.stderr, flush=True)
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    idx.save()
    save_s = time.perf_counter() - t0
    idx.close()
    return {
        "build_seconds": round(build_s, 1),
        "insert_vectors_per_sec": round(len(vecs) / build_s, 1),
        "insert_rate_last_batch": round(last_rate or 0.0, 1),
        "save_seconds": round(save_s, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--dims", type=int, default=128)
    ap.add_argument("--metric", default="cosine")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ef-search", default="10,20,50,100,200")
    ap.add_argument("--m", type=int, default=16, help="max_neighbors")
    ap.add_argument("--construction-ef", type=int, default=200)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=10_000)
    ap.add_argument("--uniform", action="store_true", help="Gaussian data instead of clusters")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workdir", default=None, help="where data and index go (default: temp)")
    ap.add_argument(
        "--graphvec-src",
        default=None,
        help="benchmark the GraphVec package in this src/ directory (e.g. an older release)",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    work = Path(args.workdir or tempfile.mkdtemp(prefix="graphvec-bench-"))
    work.mkdir(parents=True, exist_ok=True)
    vecs, queries = make_dataset(
        args.n,
        args.dims,
        seed=args.seed,
        n_queries=args.queries,
        clustered=not args.uniform,
        out=work / "data.npy",
    )
    path = work / "bench.gvec"
    if path.exists():
        shutil.rmtree(path)

    report = {
        "n": args.n,
        "dims": args.dims,
        "metric": args.metric,
        "k": args.k,
        "max_neighbors": args.m,
        "construction_ef": args.construction_ef,
        "dataset": "gaussian" if args.uniform else "clustered",
        "queries": args.queries,
        "seed": args.seed,
        "environment": environment(),
    }
    report.update(build_graphvec(path, vecs, args))
    report["peak_rss_mb_build"] = peak_rss_mb()

    t0 = time.perf_counter()
    GraphIndex.open(path, readonly=True).close()
    report["open_readonly_seconds"] = round(time.perf_counter() - t0, 3)
    # Heap the open allocates (Python + NumPy); memory-mapped pages are not heap.
    # Measured separately because tracing slows Python code down.
    tracemalloc.start()
    idx = GraphIndex.open(path, readonly=True)
    report["open_readonly_heap_mb"] = round(tracemalloc.get_traced_memory()[0] / 2**20, 1)
    tracemalloc.stop()
    report["index_bytes_on_disk"] = sum(p.stat().st_size for p in path.iterdir() if p.is_file())
    report["graph_bytes_on_disk"] = (path / "graph.bin").stat().st_size

    t0 = time.perf_counter()
    truth = ground_truth(idx, queries, args.k)
    report["exact_search_ms_per_query"] = round((time.perf_counter() - t0) * 1000 / len(queries), 2)
    report["by_ef_search"] = {}
    for ef in (int(x) for x in args.ef_search.split(",")):
        report["by_ef_search"][ef] = measure(
            lambda q, ef=ef: [r.id for r in idx.search(q, k=args.k, ef_search=ef)],
            queries,
            truth,
            args.k,
        )
    idx.close()

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
    del vecs, queries  # release the memory-mapped dataset before cleanup
    if args.workdir is None:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
