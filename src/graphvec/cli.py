"""``graphvec`` command-line interface (typer + rich)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from graphvec import __version__, storage
from graphvec.api import GraphIndex
from graphvec.exceptions import GraphVecError

app = typer.Typer(help="GraphVec — local-first file-based vector search.", no_args_is_help=True)
console = Console()
err_console = Console(stderr=True)


def _fail(exc: Exception) -> None:
    err_console.print(f"[red]Error:[/red] {exc}")
    raise typer.Exit(code=1)


def _out(data: dict, as_json: bool) -> None:
    if as_json:
        console.print_json(json.dumps(data))
    else:
        t = Table(show_header=False)
        for k, v in data.items():
            t.add_row(str(k), str(v))
        console.print(t)


@app.command()
def create(
    path: str,
    dimensions: int = typer.Option(..., "--dimensions", "-d"),
    metric: str = typer.Option("cosine", "--metric"),
    max_neighbors: int = typer.Option(16, "--max-neighbors"),
    construction_ef: int = typer.Option(200, "--construction-ef"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Create a new index."""
    try:
        with GraphIndex.create(
            path,
            dimensions=dimensions,
            metric=metric,
            max_neighbors=max_neighbors,
            construction_ef=construction_ef,
            overwrite=overwrite,
        ) as idx:
            _out({"path": path, "dimensions": idx.dimensions, "metric": idx.metric}, as_json)
    except GraphVecError as e:
        _fail(e)


@app.command(name="import")
def import_cmd(
    path: str,
    source: str,
    ids_file: str = typer.Option(None, "--ids"),
    batch_size: int = typer.Option(1000, "--batch-size"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Import vectors from .npy, .jsonl or .parquet (bounded-memory batches)."""
    try:
        ext_ids: list[str] | None = None
        if ids_file:
            ext_ids = json.loads(Path(ids_file).read_text())
        with GraphIndex.open(path) as idx:
            total = 0
            if source.endswith(".npy"):
                arr = np.load(source, mmap_mode="r")
                if ext_ids is None:
                    ext_ids = [f"vec-{i}" for i in range(arr.shape[0])]
                for s in range(0, arr.shape[0], batch_size):
                    chunk = np.array(arr[s : s + batch_size])
                    idx.add(chunk, ids=ext_ids[s : s + batch_size])
                    total += len(chunk)
                    err_console.print(f"imported {total}/{arr.shape[0]}")
            elif source.endswith(".jsonl"):
                batch_vecs, batch_ids = [], []
                with open(source) as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        batch_vecs.append(rec["vector"])
                        batch_ids.append(str(rec["id"]))
                        if len(batch_vecs) >= batch_size:
                            idx.add(np.array(batch_vecs, dtype=np.float32), ids=batch_ids)
                            total += len(batch_vecs)
                            batch_vecs, batch_ids = [], []
                if batch_vecs:
                    idx.add(np.array(batch_vecs, dtype=np.float32), ids=batch_ids)
                    total += len(batch_vecs)
            elif source.endswith(".parquet"):
                try:
                    import pyarrow.parquet as pq
                except ImportError as exc:
                    raise typer.BadParameter(
                        'Parquet import needs pyarrow: pip install "graphvec[parquet]"'
                    ) from exc
                pf = pq.ParquetFile(source)
                for batch in pf.iter_batches(batch_size=batch_size, columns=["id", "vector"]):
                    vecs = np.array(batch.column("vector").to_pylist(), dtype=np.float32)
                    idx.add(vecs, ids=[str(x) for x in batch.column("id").to_pylist()])
                    total += len(vecs)
                    err_console.print(f"imported {total}/{pf.metadata.num_rows}")
            else:
                raise typer.BadParameter("Source must be .npy, .jsonl or .parquet")
            idx.save()
            _out({"imported": total, "live_count": len(idx)}, as_json)
    except GraphVecError as e:
        _fail(e)


@app.command()
def search(
    path: str,
    query: str = typer.Option(None, "--query", help='JSON list, e.g. "[0.1,0.2]"'),
    query_file: str = typer.Option(None, "--query-file", help=".npy (1-D) or .json file"),
    top_k: int = typer.Option(10, "--top-k"),
    ef_search: int = typer.Option(100, "--ef-search"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Search an index."""
    try:
        if query_file:
            q = (
                np.load(query_file)
                if query_file.endswith(".npy")
                else np.array(json.loads(Path(query_file).read_text()), dtype=np.float32)
            )
            q = np.asarray(q)
            if q.ndim == 2:  # a query matrix file: search with its first row
                q = q[0]
        elif query:
            q = np.array(json.loads(query), dtype=np.float32)
        else:
            raise typer.BadParameter("Provide --query or --query-file")
        with GraphIndex.open(path, readonly=True) as idx:
            results = idx.search(np.asarray(q).ravel(), k=top_k, ef_search=ef_search)
        data: list[dict[str, object]] = [{"id": r.id, "score": r.score} for r in results]
        if as_json:
            console.print_json(json.dumps(data))
        else:
            t = Table(title="Results")
            t.add_column("id")
            t.add_column("score")
            for r in data:
                t.add_row(
                    str(r["id"]),
                    f"{r['score']:.6f}" if isinstance(r["score"], float) else str(r["score"]),
                )
            console.print(t)
    except GraphVecError as e:
        _fail(e)


@app.command()
def info(path: str, as_json: bool = typer.Option(False, "--json")) -> None:
    """Show index information."""
    try:
        with GraphIndex.open(path, readonly=True) as idx:
            report = idx.validate(verify_checksums=False)
            sizes = {p.name: p.stat().st_size for p in Path(path).iterdir() if p.is_file()}
            data = {
                "path": path,
                "dimensions": idx.dimensions,
                "metric": idx.metric,
                "live_count": len(idx),
                "total_count": idx.total_count,
                "max_neighbors": idx.max_neighbors,
                "construction_ef": idx.construction_ef,
                "format_version": idx.format_version,
                "graph": report["graph"],
                "files_bytes": sizes,
            }
        _out(data, as_json)
    except GraphVecError as e:
        _fail(e)


@app.command()
def validate(path: str, as_json: bool = typer.Option(False, "--json")) -> None:
    """Validate index integrity."""
    try:
        with GraphIndex.open(path, readonly=True) as idx:
            report = idx.validate()
        _out(report, as_json)
        if not report["ok"]:
            raise typer.Exit(code=1)
    except GraphVecError as e:
        _fail(e)


@app.command()
def benchmark(
    path: str,
    queries_file: str = typer.Option(..., "--queries"),
    top_k: int = typer.Option(10, "--top-k"),
    ef_search: int = typer.Option(100, "--ef-search"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Benchmark search latency/throughput and recall vs exact search."""
    try:
        queries = np.load(queries_file)
        with GraphIndex.open(path, readonly=True) as idx:
            lat = []
            hits = total = 0
            for q in queries:
                t0 = time.perf_counter()
                res = idx.search(q, k=top_k, ef_search=ef_search)
                lat.append((time.perf_counter() - t0) * 1000)
                exact = idx.search_exact(q, k=top_k)
                hits += len({r.id for r in res} & {r.id for r in exact})
                total += max(len(exact), 1)
        arr = np.array(lat)
        data = {
            "queries": int(len(queries)),
            "top_k": top_k,
            "ef_search": ef_search,
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "mean_ms": float(arr.mean()),
            "qps": float(1000.0 / arr.mean()) if arr.mean() else 0.0,
            f"recall@{top_k}": hits / total if total else 0.0,
        }
        _out(data, as_json)
    except GraphVecError as e:
        _fail(e)


@app.command()
def compact(path: str) -> None:
    """Compact an index (drop tombstones, rebuild graph)."""
    try:
        with GraphIndex.open(path) as idx:
            idx.compact()
        console.print("compacted")
    except GraphVecError as e:
        _fail(e)


@app.command()
def export(
    path: str,
    output: str = typer.Option(..., "--output"),
) -> None:
    """Export live vectors to a .npy file (plus <output>.ids.json)."""
    try:
        with GraphIndex.open(path, readonly=True) as idx:
            ids = list(idx.iter_ids())
            vecs = (
                np.stack([idx.get(e) for e in ids])
                if ids
                else np.empty((0, idx.dimensions), np.float32)
            )
        np.save(output, vecs)
        Path(output + ".ids.json").write_text(json.dumps(ids))
        console.print(f"exported {len(ids)} vectors to {output}")
    except GraphVecError as e:
        _fail(e)


@app.command()
def unlock(
    path: str,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt"),
) -> None:
    """Remove a stale writer lock left by a crashed process."""
    try:
        if not Path(path).is_dir():
            raise FileNotFoundError(f"Index directory not found: {path}")
        pid = storage.read_lock_pid(Path(path))
        if not (Path(path) / storage.LOCK_FILE).exists():
            console.print("not locked")
            return
        owner = f"process {pid}" if pid is not None else "an unknown process"
        if not yes:
            typer.confirm(
                f"Lock is held by {owner}. Remove it? Only do this if that process is not running",
                abort=True,
            )
        GraphIndex.force_unlock(path)
        console.print("unlocked")
    except (GraphVecError, FileNotFoundError) as e:
        _fail(e)


@app.command()
def version() -> None:
    """Print GraphVec version."""
    console.print(__version__)


if __name__ == "__main__":
    app()
    sys.exit(0)
