import shutil

import numpy as np
import pytest

from graphvec import GraphIndex, storage
from graphvec.exceptions import (
    ConcurrentWriterError,
    CorruptIndexError,
    InsufficientDiskSpaceError,
)


def built(tmp_path):
    rng = np.random.default_rng(0)
    p = tmp_path / "i.gvec"
    with GraphIndex.create(p, dimensions=8) as idx:
        idx.add(rng.random((50, 8)).astype(np.float32), ids=[f"x{i}" for i in range(50)])
        idx.save()
    return p


def test_truncated_vectors(tmp_path):
    p = built(tmp_path)
    data = (p / "vectors.bin").read_bytes()
    (p / "vectors.bin").write_bytes(data[:20])
    with pytest.raises(CorruptIndexError):
        GraphIndex.open(p, readonly=True)


def test_corrupt_manifest(tmp_path):
    p = built(tmp_path)
    (p / "manifest.json").write_text("{not json")
    with pytest.raises(CorruptIndexError):
        GraphIndex.open(p, readonly=True)


def test_leftover_tmp_ignored(tmp_path):
    p = built(tmp_path)
    (p / "vectors.bin.tmp").write_bytes(b"garbage")
    with GraphIndex.open(p, readonly=True) as idx:
        assert len(idx) == 50


def test_checksum_detects_tamper(tmp_path):
    p = built(tmp_path)
    raw = bytearray((p / "ids.bin").read_bytes())
    raw[-1] ^= 0xFF
    (p / "ids.bin").write_bytes(bytes(raw))
    with pytest.raises((CorruptIndexError, UnicodeDecodeError)):
        GraphIndex.open(p, readonly=True, verify_checksums=True)


def test_concurrent_writer_rejected(tmp_path):
    p = built(tmp_path)
    w1 = GraphIndex.open(p)
    with pytest.raises(ConcurrentWriterError):
        GraphIndex.open(p)
    # readonly still allowed while writer holds lock
    ro = GraphIndex.open(p, readonly=True)
    ro.close()
    w1.close()
    # after close, writer can open again
    GraphIndex.open(p).close()


def test_create_refuses_overwrite(tmp_path):
    p = built(tmp_path)
    with pytest.raises(Exception):
        GraphIndex.create(p, dimensions=8)


def test_insufficient_disk_space_keeps_committed_data(tmp_path, monkeypatch):
    p = built(tmp_path)
    full = shutil.disk_usage(p)._replace(free=0)
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda _path: full)
    with GraphIndex.open(p) as idx:
        idx.delete("x0")
        with pytest.raises(InsufficientDiskSpaceError):
            idx.save()
    monkeypatch.undo()
    with GraphIndex.open(p, readonly=True, verify_checksums=True) as idx:
        assert len(idx) == 50


def test_force_unlock_clears_stale_lock(tmp_path):
    p = built(tmp_path)
    (p / ".writer.lock").write_text("999999")  # left behind by a crashed writer
    with pytest.raises(ConcurrentWriterError, match="999999"):
        GraphIndex.open(p)
    assert GraphIndex.force_unlock(p) == 999999
    assert GraphIndex.force_unlock(p) is None
    GraphIndex.open(p).close()
