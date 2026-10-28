import numpy as np
import pytest

from graphvec import GraphIndex
from graphvec.workfile import WORK_DIR, FileBacking, WorkingCopy

ON_DISK, IN_RAM = 0, None  # memory_limit values


def vectors(n, d=8, seed=0):
    return np.random.default_rng(seed).random((n, d)).astype(np.float32)


def test_disk_working_copy_uses_files_and_removes_them_on_close(tmp_path):
    p = tmp_path / "i.gvec"
    with GraphIndex.create(p, dimensions=8, memory_limit=ON_DISK) as idx:
        idx.add(vectors(100))
        work = sorted(f.name.split(".")[0] for f in (p / WORK_DIR).iterdir())
        assert work == ["adj0", "deg0", "levels", "vectors"]
        assert not isinstance(idx._vectors, np.memmap)  # plain view of a mapped file
        assert isinstance(idx._vectors.base, np.memmap)
        idx.save()
    assert not (p / WORK_DIR).exists()
    with GraphIndex.open(p, readonly=True) as ro:
        assert ro._backing is None and len(ro) == 100


def test_ram_working_copy_writes_no_files(tmp_path):
    p = tmp_path / "i.gvec"
    with GraphIndex.create(p, dimensions=8) as idx:
        idx.add(vectors(100))
        assert not idx._backing.on_disk and not (p / WORK_DIR).exists()
        assert idx._vectors.base is None or not isinstance(idx._vectors.base, np.memmap)


def test_spills_to_disk_past_the_memory_limit(tmp_path):
    p = tmp_path / "i.gvec"
    v = vectors(1000)
    # 8 dims: 32 bytes per vector plus 4*2*16*4 + 5 = 517 bytes of graph per node
    with GraphIndex.create(p, dimensions=8, metric="dot", memory_limit=100_000) as idx:
        idx.add(v[:100], ids=[f"x{i}" for i in range(100)])
        assert not idx._backing.on_disk
        for start in range(100, 1000, 37):  # many capacity doublings
            stop = min(start + 37, 1000)
            idx.add(v[start:stop], ids=[f"x{i}" for i in range(start, stop)])
        assert idx._backing.on_disk
        assert isinstance(idx._vectors.base, np.memmap)
        assert isinstance(idx._graph.adj0.base, np.memmap)
        assert np.array_equal(idx.get("x0"), v[0]) and np.array_equal(idx.get("x999"), v[999])
        assert idx.validate()["ok"]
        idx.save()
    with GraphIndex.open(p, readonly=True, verify_checksums=True) as ro:
        assert len(ro) == 1000
        assert np.array_equal(ro.get("x500"), v[500])


def test_negative_memory_limit_rejected(tmp_path):
    with pytest.raises(ValueError):
        GraphIndex.create(tmp_path / "i.gvec", dimensions=8, memory_limit=-1)
    GraphIndex.create(tmp_path / "i.gvec", dimensions=8, overwrite=True).close()  # lock released


@pytest.mark.parametrize("memory_limit", [ON_DISK, IN_RAM])
def test_reopened_writer_copies_commit_and_extends_it(tmp_path, memory_limit):
    p = tmp_path / "i.gvec"
    v = vectors(300)
    with GraphIndex.create(p, dimensions=8) as idx:
        idx.add(v[:200], ids=[f"x{i}" for i in range(200)])
        idx.save()
    with GraphIndex.open(p, memory_limit=memory_limit) as idx:
        assert idx._backing.on_disk == (memory_limit == ON_DISK)
        idx.add(v[200:], ids=[f"x{i}" for i in range(200, 300)])
        assert idx.validate()["ok"]
        idx.save()
    with GraphIndex.open(p, readonly=True) as ro:
        assert len(ro) == 300
        top = ro.search(v[250], k=1, ef_search=50)[0]
        assert top.id == "x250"


def test_unsaved_changes_never_reach_the_commit(tmp_path):
    p = tmp_path / "i.gvec"
    with GraphIndex.create(p, dimensions=8) as idx:
        idx.add(vectors(50), ids=[f"x{i}" for i in range(50)])
        idx.save()
    before = {f.name: f.read_bytes() for f in p.iterdir() if f.is_file()}
    idx = GraphIndex.open(p, memory_limit=ON_DISK)
    idx.add(vectors(500, seed=2))  # never saved
    idx.delete("x3")
    # simulate a crash: the working files and the lock are left behind
    idx._backing = None
    lock, idx._lock = idx._lock, None
    idx.close()
    assert (p / WORK_DIR).exists()
    lock.unlink()  # what `graphvec unlock` does after a crash
    assert {f.name: f.read_bytes() for f in p.iterdir() if f.is_file()} == before
    with GraphIndex.open(p, memory_limit=ON_DISK) as idx:  # stale copy is replaced
        assert len(idx) == 50 and "x3" in idx
        assert len(list((p / WORK_DIR).iterdir())) == 4


@pytest.mark.parametrize("memory_limit", [ON_DISK, IN_RAM])
def test_compact_moves_rows_in_place(tmp_path, memory_limit):
    p = tmp_path / "i.gvec"
    v = vectors(400)
    with GraphIndex.create(p, dimensions=8, metric="dot", memory_limit=memory_limit) as idx:
        idx.add(v, ids=[f"x{i}" for i in range(400)])
        idx.delete([f"x{i}" for i in range(0, 400, 3)])
        idx.compact()
        assert len(idx) == idx.total_count == 266
        for i in (1, 2, 4, 397, 398):
            assert np.array_equal(idx.get(f"x{i}"), v[i])
        assert idx.validate()["ok"]
        idx.add(v[:5], ids=[f"x{i}" for i in range(0, 15, 3)])  # reuse compacted ids
        assert len(idx) == 271


def test_disk_and_ram_working_copies_build_identical_graphs(tmp_path):
    v = vectors(500, d=16)
    graphs = []
    for name, limit in (("disk", ON_DISK), ("ram", IN_RAM)):
        with GraphIndex.create(tmp_path / name, dimensions=16, memory_limit=limit) as idx:
            idx.add(v)
            idx.save()
        graphs.append((tmp_path / name / "graph.bin").read_bytes())
    assert graphs[0] == graphs[1]


def test_file_backing_grow_remaps_and_copies(tmp_path):
    b = FileBacking(tmp_path / "w")
    a = b.grow("x", np.arange(4, dtype=np.int32), 4, 8)
    a[4:] = [4, 5, 6, 7]
    bigger = b.grow("x", a, 8, 64)  # same file, remapped larger
    assert bigger[:8].tolist() == list(range(8))
    fresh = b.grow("x", np.full(3, 9, dtype=np.int32), 3, 16)  # foreign array: new file
    assert fresh[:3].tolist() == [9, 9, 9]
    del a, bigger, fresh
    b.close()
    assert not (tmp_path / "w").exists()


def test_working_copy_removes_stale_directory(tmp_path):
    stale = tmp_path / WORK_DIR
    stale.mkdir()
    (stale / "vectors.0.bin").write_bytes(b"left by a crash")
    WorkingCopy(stale, memory_limit=1 << 20)
    assert not stale.exists()
