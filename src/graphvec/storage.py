"""Persistence: atomic writes, checksums, locking, mmap loading."""

from __future__ import annotations

import json
import os
import shutil
import struct
from pathlib import Path

import numpy as np

from graphvec import format as fmt
from graphvec.exceptions import (
    ConcurrentWriterError,
    CorruptIndexError,
    InsufficientDiskSpaceError,
    PersistenceError,
    UnsupportedFormatVersionError,
)
from graphvec.graph import MAX_LEVEL, Graph

MANIFEST = "manifest.json"


def _fsync_file(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass  # best-effort durability (documented)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


LOCK_FILE = ".writer.lock"


def read_lock_pid(index_dir: Path) -> int | None:
    """PID recorded in the writer lock, or None if unlocked or unreadable."""
    try:
        return int((index_dir / LOCK_FILE).read_text().strip())
    except (OSError, ValueError):
        return None


def acquire_lock(index_dir: Path) -> Path:
    lock = index_dir / LOCK_FILE
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        pid = read_lock_pid(index_dir)
        owner = f"process {pid}" if pid is not None else "another writer"
        raise ConcurrentWriterError(
            f"Index at {index_dir} is locked by {owner}. If that process is no longer "
            f"running, remove the stale lock with: graphvec unlock {index_dir}"
        ) from exc
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return lock


def force_unlock(index_dir: Path) -> int | None:
    """Remove the writer lock unconditionally. Returns the PID it recorded."""
    pid = read_lock_pid(index_dir)
    release_lock(index_dir / LOCK_FILE)
    return pid


def release_lock(lock: Path | None) -> None:
    if lock is not None:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _estimate_save_bytes(vectors: np.ndarray, ids: list[str], graph: Graph) -> int:
    """Upper-bound estimate of the bytes ``save_index`` writes."""
    n = len(ids)
    upper = sum(8 + 4 * len(lst) for lists in graph.upper.values() for lst in lists)
    graph_bytes = n * (5 + 4 * graph.width) + upper
    id_bytes = sum(4 + len(e.encode("utf-8")) for e in ids)
    headers = 4 * 64 + (1 << 16)  # file headers + manifest
    return int(vectors.nbytes) + graph_bytes + id_bytes + n + headers


def check_disk_space(index_dir: Path, needed: int) -> None:
    try:
        free = shutil.disk_usage(index_dir).free
    except OSError:
        return  # cannot determine; let the write itself fail if space runs out
    if free < needed:
        raise InsufficientDiskSpaceError(
            f"Writing to {index_dir} needs about {needed:,} bytes but only {free:,} are free"
        )


_WRITE_BYTES = 64 << 20


def _write_array(fh, arr: np.ndarray, dtype) -> None:
    """Write ``arr`` a chunk of rows at a time, so a memmap is never read whole."""
    row_bytes = max(1, arr[:1].nbytes)
    step = max(1, _WRITE_BYTES // row_bytes)
    for start in range(0, len(arr), step):
        fh.write(np.ascontiguousarray(arr[start : start + step], dtype=dtype).data)


def _write_graph(fh, graph: Graph, n: int) -> None:
    fh.write(
        fmt.GRAPH_HEADER.pack(
            fmt.GRAPH_MAGIC,
            fmt.FORMAT_VERSION,
            n,
            graph.entry_point,
            graph.max_level,
            graph.max_neighbors,
            graph.width,
            0,
        )
    )
    _write_array(fh, graph.deg0[:n], "<i4")
    _write_array(fh, graph.adj0[:n], "<i4")
    _write_array(fh, graph.levels[:n], np.uint8)
    fh.write(struct.pack("<Q", len(graph.upper)))
    for node in sorted(graph.upper):
        lists = graph.upper[node]
        fh.write(struct.pack("<iI", node, len(lists)))
        for lst in lists:
            fh.write(struct.pack(f"<I{len(lst)}i", len(lst), *lst))


def save_index(
    index_dir: Path,
    manifest: dict,
    vectors: np.ndarray,
    ids: list[str],
    deleted: np.ndarray,
    graph: Graph,
) -> dict:
    """Write data files then the manifest (commit point). Returns manifest."""
    try:
        index_dir.mkdir(parents=True, exist_ok=True)
        check_disk_space(index_dir, _estimate_save_bytes(vectors, ids, graph))
        n, d = vectors.shape
        tmp = index_dir / "vectors.bin.tmp"
        with open(tmp, "wb") as fh:
            fh.write(fmt.HEADER.pack(fmt.VEC_MAGIC, fmt.FORMAT_VERSION, n, d))
            _write_array(fh, vectors, "<f4")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, index_dir / "vectors.bin")

        tmp = index_dir / "graph.bin.tmp"
        with open(tmp, "wb") as fh:
            _write_graph(fh, graph, n)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, index_dir / "graph.bin")

        tmp = index_dir / "ids.bin.tmp"
        with open(tmp, "wb") as fh:
            fh.write(struct.pack("<4sIQ", fmt.IDS_MAGIC, fmt.FORMAT_VERSION, n))
            for ext in ids:
                raw = ext.encode("utf-8")
                fh.write(struct.pack("<I", len(raw)))
                fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, index_dir / "ids.bin")

        tmp = index_dir / "tombstones.bin.tmp"
        with open(tmp, "wb") as fh:
            fh.write(struct.pack("<4sIQ", fmt.TOMB_MAGIC, fmt.FORMAT_VERSION, n))
            fh.write(np.asarray(deleted, dtype=np.uint8).tobytes())
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, index_dir / "tombstones.bin")

        manifest = dict(manifest)
        manifest["format_version"] = fmt.FORMAT_VERSION
        manifest["count"] = int(n)
        manifest["live_count"] = int((~np.asarray(deleted, dtype=bool)).sum()) if n else 0
        manifest["checksums"] = {
            name: fmt.sha256_file(index_dir / name)
            for name in ("vectors.bin", "graph.bin", "ids.bin", "tombstones.bin")
        }
        _atomic_write_bytes(
            index_dir / MANIFEST, json.dumps(manifest, indent=2, sort_keys=True).encode()
        )
        for name in ("vectors.bin", "graph.bin", "ids.bin", "tombstones.bin"):
            _fsync_file(index_dir / name)
        return manifest
    except OSError as exc:
        raise PersistenceError(f"Failed to save index at {index_dir}: {exc}") from exc


def _check_magic_version(
    magic: bytes, expected: bytes, version: int, want_version: int, what: str
) -> None:
    if magic != expected:
        raise CorruptIndexError(f"{what}: bad magic {magic!r}, expected {expected!r}")
    if version != want_version:
        raise UnsupportedFormatVersionError(
            f"{what}: format version {version} does not match the manifest ({want_version})"
        )


def _read_graph_v1(data: bytes, n: int, graph: Graph) -> None:
    """Parse a format-1 graph.bin (variable-length lists) into ``graph``."""
    head = struct.calcsize("<4sIQqi")
    if len(data) < head:
        raise CorruptIndexError("graph.bin truncated (header)")
    _magic, _version, _count, entry_point, max_level = struct.unpack_from("<4sIQqi", data, 0)
    off = head
    levels = np.zeros(n, dtype=np.uint8)
    deg0 = np.zeros(n, dtype=np.int32)
    adj0 = np.full((n, graph.width), -1, dtype=np.int32)
    upper: dict[int, list[list[int]]] = {}
    try:
        for i in range(n):
            (lvl,) = struct.unpack_from("<i", data, off)
            off += 4
            if lvl < 0 or lvl > MAX_LEVEL:
                raise CorruptIndexError(f"graph.bin implausible node level {lvl}")
            node_layers: list[list[int]] = []
            for _layer in range(lvl + 1):
                (deg,) = struct.unpack_from("<I", data, off)
                off += 4
                if deg > graph.width:
                    raise CorruptIndexError(f"graph.bin implausible degree {deg}")
                node_layers.append(list(struct.unpack_from(f"<{deg}i", data, off)))
                off += 4 * deg
            levels[i] = lvl
            deg0[i] = len(node_layers[0])
            adj0[i, : deg0[i]] = node_layers[0]
            if lvl:
                upper[i] = node_layers[1:]
    except struct.error as exc:
        raise CorruptIndexError(f"graph.bin truncated: {exc}") from exc
    graph.load(levels, deg0, adj0, upper, entry_point, max_level)


def _read_graph_v2(gpath: Path, n: int, graph: Graph, readonly: bool) -> None:
    """Read a format-2 graph.bin; layer-0 arrays are memory-mapped when readonly."""
    size = gpath.stat().st_size
    with open(gpath, "rb") as fh:
        header = fh.read(fmt.GRAPH_HEADER.size)
    if len(header) < fmt.GRAPH_HEADER.size:
        raise CorruptIndexError("graph.bin truncated (header)")
    _magic, _version, _count, entry_point, max_level, max_neighbors, width, _reserved = (
        fmt.GRAPH_HEADER.unpack(header)
    )
    if max_neighbors != graph.max_neighbors or width != graph.width:
        raise CorruptIndexError("graph.bin parameters disagree with the manifest")
    off_deg = fmt.GRAPH_HEADER.size
    off_adj = off_deg + 4 * n
    off_lvl = off_adj + 4 * n * width
    off_upper = off_lvl + n
    if size < off_upper + 8:
        raise CorruptIndexError(
            f"graph.bin truncated: size {size}, expected at least {off_upper + 8}"
        )

    if n == 0:
        levels = np.zeros(0, dtype=np.uint8)
        deg0 = np.zeros(0, dtype=np.int32)
        adj0 = np.full((0, width), -1, dtype=np.int32)
    else:
        deg0 = np.memmap(gpath, dtype="<i4", mode="r", offset=off_deg, shape=(n,))
        adj0 = np.memmap(gpath, dtype="<i4", mode="r", offset=off_adj, shape=(n, width))
        levels = np.memmap(gpath, dtype=np.uint8, mode="r", offset=off_lvl, shape=(n,))
        if not readonly:
            deg0 = np.array(deg0, dtype=np.int32)
            adj0 = np.array(adj0, dtype=np.int32)
            levels = np.array(levels, dtype=np.uint8)

    with open(gpath, "rb") as fh:
        fh.seek(off_upper)
        tail = fh.read()
    upper: dict[int, list[list[int]]] = {}
    try:
        (n_upper,) = struct.unpack_from("<Q", tail, 0)
        if n_upper > n:
            raise CorruptIndexError("graph.bin implausible upper-layer node count")
        off = 8
        for _ in range(n_upper):
            node, lvl = struct.unpack_from("<iI", tail, off)
            off += 8
            if not 0 <= node < n or not 0 < lvl <= MAX_LEVEL or node in upper:
                raise CorruptIndexError(f"graph.bin invalid upper-layer node {node}")
            lists: list[list[int]] = []
            for _layer in range(lvl):
                (deg,) = struct.unpack_from("<I", tail, off)
                off += 4
                if deg > width:
                    raise CorruptIndexError(f"graph.bin implausible degree {deg}")
                lists.append(list(struct.unpack_from(f"<{deg}i", tail, off)))
                off += 4 * deg
            upper[node] = lists
    except struct.error as exc:
        raise CorruptIndexError(f"graph.bin truncated: {exc}") from exc
    if off != len(tail):
        raise CorruptIndexError("graph.bin has trailing data")
    graph.load(levels, deg0, adj0, upper, entry_point, max_level)


def load_index(index_dir: Path, readonly: bool, verify_checksums: bool = False) -> dict:
    manifest_path = index_dir / MANIFEST
    if not manifest_path.exists():
        raise CorruptIndexError(
            f"No manifest.json in {index_dir}; not a GraphVec index (or an interrupted create)"
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CorruptIndexError(f"manifest.json is corrupt: {exc}") from exc
    version = manifest.get("format_version")
    if version not in fmt.READABLE_FORMAT_VERSIONS:
        raise UnsupportedFormatVersionError(
            f"Index format_version={version!r}, supported={list(fmt.READABLE_FORMAT_VERSIONS)}"
        )
    d = int(manifest["dimensions"])
    n = int(manifest["count"])
    max_neighbors = int(manifest.get("max_neighbors", 16))
    if d <= 0 or n < 0 or d > 1_000_000 or n > 2**31 - 1:
        raise CorruptIndexError(f"Implausible manifest dimensions/count: d={d} n={n}")
    if not 1 <= max_neighbors <= 4096:
        raise CorruptIndexError(f"Implausible manifest max_neighbors={max_neighbors}")

    if verify_checksums and "checksums" in manifest:
        for name, want in manifest["checksums"].items():
            p = index_dir / name
            if not p.exists() or fmt.sha256_file(p) != want:
                raise CorruptIndexError(f"Checksum mismatch for {name}")

    vpath = index_dir / "vectors.bin"
    expected = fmt.HEADER.size + n * d * 4
    if not vpath.exists() or vpath.stat().st_size != expected:
        got = vpath.stat().st_size if vpath.exists() else -1
        raise CorruptIndexError(f"vectors.bin truncated/corrupt: size {got}, expected {expected}")
    with open(vpath, "rb") as fh:
        magic, fversion, count, dims = fmt.HEADER.unpack(fh.read(fmt.HEADER.size))
    _check_magic_version(magic, fmt.VEC_MAGIC, fversion, version, "vectors.bin")
    if int(count) != n or int(dims) != d:
        raise CorruptIndexError("vectors.bin header disagrees with manifest")
    if n == 0:
        vectors = np.empty((0, d), dtype=np.float32)
    elif readonly:
        vectors = np.memmap(vpath, dtype="<f4", mode="r", offset=fmt.HEADER.size, shape=(n, d))
    else:
        vectors = np.array(
            np.memmap(vpath, dtype="<f4", mode="r", offset=fmt.HEADER.size, shape=(n, d)),
            dtype=np.float32,
        )

    ipath = index_dir / "ids.bin"
    ids: list[str] = []
    try:
        with open(ipath, "rb") as fh:
            magic, fversion, count = struct.unpack("<4sIQ", fh.read(struct.calcsize("<4sIQ")))
            _check_magic_version(magic, fmt.IDS_MAGIC, fversion, version, "ids.bin")
            if int(count) != n:
                raise CorruptIndexError("ids.bin count disagrees with manifest")
            for _ in range(n):
                raw_len = fh.read(4)
                if len(raw_len) != 4:
                    raise CorruptIndexError("ids.bin truncated")
                (ln,) = struct.unpack("<I", raw_len)
                if ln > 1_000_000:
                    raise CorruptIndexError(f"ids.bin implausible id length {ln}")
                raw = fh.read(ln)
                if len(raw) != ln:
                    raise CorruptIndexError("ids.bin truncated")
                ids.append(raw.decode("utf-8"))
    except FileNotFoundError as exc:
        raise CorruptIndexError(f"Missing ids.bin in {index_dir}") from exc

    tpath = index_dir / "tombstones.bin"
    if not tpath.exists() or tpath.stat().st_size != struct.calcsize("<4sIQ") + n:
        raise CorruptIndexError("tombstones.bin truncated/corrupt")
    with open(tpath, "rb") as fh:
        magic, fversion, count = struct.unpack("<4sIQ", fh.read(struct.calcsize("<4sIQ")))
        _check_magic_version(magic, fmt.TOMB_MAGIC, fversion, version, "tombstones.bin")
        deleted = np.frombuffer(fh.read(n), dtype=np.uint8).astype(bool)

    gpath = index_dir / "graph.bin"
    if not gpath.exists():
        raise CorruptIndexError(f"Missing graph.bin in {index_dir}")
    with open(gpath, "rb") as fh:
        head = fh.read(struct.calcsize("<4sIQ"))
    if len(head) < struct.calcsize("<4sIQ"):
        raise CorruptIndexError("graph.bin truncated (header)")
    magic, fversion, count = struct.unpack("<4sIQ", head)
    _check_magic_version(magic, fmt.GRAPH_MAGIC, fversion, version, "graph.bin")
    if int(count) != n:
        raise CorruptIndexError("graph.bin count disagrees with manifest")
    graph = Graph(
        d,
        manifest["metric"],
        max_neighbors,
        int(manifest.get("construction_ef", 200)),
        int(manifest.get("seed", 42)),
    )
    if version == 1:
        _read_graph_v1(gpath.read_bytes(), n, graph)
    else:
        _read_graph_v2(gpath, n, graph, readonly)
    errors = graph.check(n)
    if errors:
        raise CorruptIndexError(f"graph.bin: {errors[0]}")

    return {
        "manifest": manifest,
        "vectors": vectors,
        "ids": ids,
        "deleted": deleted,
        "graph": graph,
    }
