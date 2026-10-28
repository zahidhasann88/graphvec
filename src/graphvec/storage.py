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


def acquire_lock(index_dir: Path) -> Path:
    lock = index_dir / ".writer.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ConcurrentWriterError(
            f"Index at {index_dir} is locked by another writer ({lock} exists). "
            "If no writer is running, remove the stale lock file manually."
        ) from exc
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return lock


def release_lock(lock: Path | None) -> None:
    if lock is not None:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _estimate_save_bytes(vectors: np.ndarray, ids: list[str], layers: list[list[list[int]]]) -> int:
    """Upper-bound estimate of the bytes ``save_index`` writes."""
    n = len(ids)
    graph = 4 * n + sum(4 + 4 * len(nbrs) for node in layers for nbrs in node)
    id_bytes = sum(4 + len(e.encode("utf-8")) for e in ids)
    headers = 4 * 64 + (1 << 16)  # file headers + manifest
    return int(vectors.nbytes) + graph + id_bytes + n + headers


def _check_disk_space(index_dir: Path, needed: int) -> None:
    try:
        free = shutil.disk_usage(index_dir).free
    except OSError:
        return  # cannot determine; let the write itself fail if space runs out
    if free < needed:
        raise InsufficientDiskSpaceError(
            f"Saving to {index_dir} needs about {needed:,} bytes but only {free:,} are free"
        )


def save_index(
    index_dir: Path,
    manifest: dict,
    vectors: np.ndarray,
    ids: list[str],
    deleted: np.ndarray,
    layers: list[list[list[int]]],
    levels: list[int],
    entry_point: int,
    max_level: int,
) -> dict:
    """Write data files then the manifest (commit point). Returns manifest."""
    try:
        index_dir.mkdir(parents=True, exist_ok=True)
        _check_disk_space(index_dir, _estimate_save_bytes(vectors, ids, layers))
        n, d = vectors.shape
        tmp = index_dir / "vectors.bin.tmp"
        with open(tmp, "wb") as fh:
            fh.write(fmt.HEADER.pack(fmt.VEC_MAGIC, fmt.FORMAT_VERSION, n, d))
            fh.write(np.ascontiguousarray(vectors, dtype="<f4").tobytes())
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, index_dir / "vectors.bin")

        tmp = index_dir / "graph.bin.tmp"
        with open(tmp, "wb") as fh:
            fh.write(
                struct.pack(
                    "<4sIQqi", fmt.GRAPH_MAGIC, fmt.FORMAT_VERSION, n, entry_point, max_level
                )
            )
            for i in range(n):
                fh.write(struct.pack("<i", int(levels[i])))
                for layer_list in layers[i]:
                    fh.write(struct.pack("<I", len(layer_list)))
                    if layer_list:
                        fh.write(struct.pack(f"<{len(layer_list)}i", *layer_list))
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


def _check_magic_version(magic: bytes, expected: bytes, version: int, what: str) -> None:
    if magic != expected:
        raise CorruptIndexError(f"{what}: bad magic {magic!r}, expected {expected!r}")
    if version != fmt.FORMAT_VERSION:
        raise UnsupportedFormatVersionError(
            f"{what}: format version {version} unsupported "
            f"(this build supports {fmt.FORMAT_VERSION})"
        )


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
    if manifest.get("format_version") != fmt.FORMAT_VERSION:
        raise UnsupportedFormatVersionError(
            f"Index format_version={manifest.get('format_version')!r}, "
            f"supported={fmt.FORMAT_VERSION}"
        )
    d = int(manifest["dimensions"])
    n = int(manifest["count"])
    if d <= 0 or n < 0 or d > 1_000_000 or n > 2**31 - 1:
        raise CorruptIndexError(f"Implausible manifest dimensions/count: d={d} n={n}")

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
        magic, version, count, dims = fmt.HEADER.unpack(fh.read(fmt.HEADER.size))
    _check_magic_version(magic, fmt.VEC_MAGIC, version, "vectors.bin")
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
            magic, version, count = struct.unpack("<4sIQ", fh.read(struct.calcsize("<4sIQ")))
            _check_magic_version(magic, fmt.IDS_MAGIC, version, "ids.bin")
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
        magic, version, count = struct.unpack("<4sIQ", fh.read(struct.calcsize("<4sIQ")))
        _check_magic_version(magic, fmt.TOMB_MAGIC, version, "tombstones.bin")
        deleted = np.frombuffer(fh.read(n), dtype=np.uint8).astype(bool)

    gpath = index_dir / "graph.bin"
    layers: list[list[list[int]]] = []
    levels: list[int] = []
    try:
        data = gpath.read_bytes()
    except FileNotFoundError as exc:
        raise CorruptIndexError(f"Missing graph.bin in {index_dir}") from exc
    off = 0
    head = struct.calcsize("<4sIQqi")
    if len(data) < head:
        raise CorruptIndexError("graph.bin truncated (header)")
    magic, version, count, entry_point, max_level = struct.unpack_from("<4sIQqi", data, off)
    off += head
    _check_magic_version(magic, fmt.GRAPH_MAGIC, version, "graph.bin")
    if int(count) != n:
        raise CorruptIndexError("graph.bin count disagrees with manifest")
    try:
        for _i in range(n):
            (lvl,) = struct.unpack_from("<i", data, off)
            off += 4
            if lvl < 0 or lvl > 64:
                raise CorruptIndexError(f"graph.bin implausible node level {lvl}")
            node_layers: list[list[int]] = []
            for _layer in range(lvl + 1):
                (deg,) = struct.unpack_from("<I", data, off)
                off += 4
                if deg > 10000:
                    raise CorruptIndexError(f"graph.bin implausible degree {deg}")
                neigh = list(struct.unpack_from(f"<{deg}i", data, off)) if deg else []
                off += 4 * deg
                for nb in neigh:
                    if nb < 0 or nb >= n:
                        raise CorruptIndexError(f"graph.bin neighbour id {nb} out of range [0,{n})")
                node_layers.append(neigh)
            levels.append(int(lvl))
            layers.append(node_layers)
    except struct.error as exc:
        raise CorruptIndexError(f"graph.bin truncated: {exc}") from exc

    return {
        "manifest": manifest,
        "vectors": vectors,
        "ids": ids,
        "deleted": deleted,
        "layers": layers,
        "levels": levels,
        "entry_point": int(entry_point),
        "max_level": int(max_level),
    }
