"""Compare GraphVec with established HNSW implementations on identical data.

Usage:
    pip install ".[bench]"
    python benchmarks/compare.py --n 100000 --dims 128 --out compare.json

Engines: GraphVec, FAISS ``IndexHNSWFlat`` and hnswlib, each when installed.
All use the same dataset, M, construction ef, k and ef_search values, the
same exact ground truth, and a single thread, so differences come from the
implementations rather than the setup. Reports measured numbers only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from datasets import make_dataset  # noqa: E402
from run import build_graphvec, environment, ground_truth, measure, peak_rss_mb  # noqa: E402

from graphvec import GraphIndex  # noqa: E402


def prepare(vecs: np.ndarray, metric: str) -> np.ndarray:
    arr = np.ascontiguousarray(vecs, dtype=np.float32)
    if metric == "cosine":
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.where(norms > 0, norms, 1)
    return arr


def run_faiss(vecs, queries, truth, args, efs) -> dict | None:
    try:
        import faiss
    except ImportError:
        return None
    faiss.omp_set_num_threads(1)
    data = prepare(vecs, args.metric)
    qs = prepare(queries, args.metric)
    kind = faiss.METRIC_L2 if args.metric == "squared_euclidean" else faiss.METRIC_INNER_PRODUCT
    t0 = time.perf_counter()
    index = faiss.IndexHNSWFlat(data.shape[1], args.m, kind)
    index.hnsw.efConstruction = args.construction_ef
    index.add(data)
    build = time.perf_counter() - t0
    out: dict = {
        "version": faiss.__version__,
        "build_seconds": round(build, 1),
        "insert_vectors_per_sec": round(len(data) / build, 1),
        "by_ef_search": {},
    }
    for ef in efs:
        index.hnsw.efSearch = ef

        def search(q, k=args.k):
            _, labels = index.search(q.reshape(1, -1), k)
            return [f"v{i}" for i in labels[0] if i >= 0]

        out["by_ef_search"][ef] = measure(search, qs, truth, args.k)
    return out


def run_hnswlib(vecs, queries, truth, args, efs) -> dict | None:
    try:
        import hnswlib
    except ImportError:
        return None
    space = {"cosine": "cosine", "dot": "ip", "squared_euclidean": "l2"}[args.metric]
    data = np.ascontiguousarray(vecs, dtype=np.float32)
    t0 = time.perf_counter()
    index = hnswlib.Index(space=space, dim=data.shape[1])
    index.init_index(
        max_elements=len(data),
        ef_construction=args.construction_ef,
        M=args.m,
        random_seed=args.seed,
    )
    index.set_num_threads(1)
    index.add_items(data, np.arange(len(data)))
    build = time.perf_counter() - t0
    out: dict = {
        "build_seconds": round(build, 1),
        "insert_vectors_per_sec": round(len(data) / build, 1),
        "by_ef_search": {},
    }
    for ef in efs:
        index.set_ef(max(ef, args.k))

        def search(q, k=args.k):
            labels, _ = index.knn_query(q, k=k)
            return [f"v{i}" for i in labels[0]]

        out["by_ef_search"][ef] = measure(search, queries, truth, args.k)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--dims", type=int, default=128)
    ap.add_argument("--metric", default="cosine")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ef-search", default="10,20,50,100,200")
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--construction-ef", type=int, default=200)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=10_000)
    ap.add_argument("--uniform", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    efs = [int(x) for x in args.ef_search.split(",")]

    work = Path(tempfile.mkdtemp(prefix="graphvec-compare-"))
    vecs, queries = make_dataset(
        args.n, args.dims, args.seed, args.queries, not args.uniform, out=work / "data.npy"
    )
    report: dict = {
        "n": args.n,
        "dims": args.dims,
        "metric": args.metric,
        "k": args.k,
        "max_neighbors": args.m,
        "construction_ef": args.construction_ef,
        "dataset": "gaussian" if args.uniform else "clustered",
        "threads": 1,
        "environment": environment(),
        "engines": {},
    }

    path = work / "bench.gvec"
    gv = build_graphvec(path, vecs, args)
    gv["peak_rss_mb_build"] = peak_rss_mb()
    t0 = time.perf_counter()
    idx = GraphIndex.open(path, readonly=True)
    gv["open_readonly_seconds"] = round(time.perf_counter() - t0, 3)
    gv["index_bytes_on_disk"] = sum(p.stat().st_size for p in path.iterdir() if p.is_file())
    truth = ground_truth(idx, queries, args.k)
    gv["by_ef_search"] = {
        ef: measure(
            lambda q, ef=ef: [r.id for r in idx.search(q, k=args.k, ef_search=ef)],
            queries,
            truth,
            args.k,
        )
        for ef in efs
    }
    idx.close()
    report["engines"]["graphvec"] = gv

    for name, runner in (("faiss_hnsw", run_faiss), ("hnswlib", run_hnswlib)):
        result = runner(vecs, queries, truth, args, efs)
        if result is None:
            print(f"{name} not installed; skipping", file=sys.stderr)
        else:
            report["engines"][name] = result

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
    del vecs, queries
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
