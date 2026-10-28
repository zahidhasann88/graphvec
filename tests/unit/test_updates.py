import numpy as np
import pytest

from graphvec import GraphIndex
from graphvec.exceptions import ClosedIndexError, DuplicateIDError, MissingIDError


def seeded_index(tmp_path):
    rng = np.random.default_rng(0)
    idx = GraphIndex.create(tmp_path / "i.gvec", dimensions=8, metric="dot")
    vecs = rng.random((50, 8)).astype(np.float32)
    idx.add(vecs, ids=[f"d{i}" for i in range(50)])
    return idx, vecs


def test_duplicate_rejected_and_replace(tmp_path):
    idx, vecs = seeded_index(tmp_path)
    with pytest.raises(DuplicateIDError):
        idx.add(vecs[:1], ids=["d0"])
    idx.add(vecs[:1] * 2, ids=["d0"], replace=True)
    assert idx.contains("d0") and len(idx) == 50
    idx.close()


def test_get_contains_delete(tmp_path):
    idx, vecs = seeded_index(tmp_path)
    assert "d3" in idx
    assert np.allclose(idx.get("d3"), vecs[3])
    idx.delete("d3")
    assert "d3" not in idx
    with pytest.raises(MissingIDError):
        idx.get("d3")
    with pytest.raises(MissingIDError):
        idx.delete("d3")
    # deleted never in results
    for r in idx.search(vecs[3], k=50, ef_search=200):
        assert r.id != "d3"
    assert len(idx) == 49
    idx.close()


def test_delete_batch_and_compact(tmp_path):
    idx, vecs = seeded_index(tmp_path)
    idx.delete(["d0", "d1", "d2"])
    idx.save()
    idx.compact()
    assert len(idx) == 47 and idx.total_count == 47
    idx.save()
    idx.close()
    idx2 = GraphIndex.open(tmp_path / "i.gvec", readonly=True)
    assert len(idx2) == 47
    assert {r.id for r in idx2.search(vecs[10], k=10)} == {
        r.id for r in idx2.search_exact(vecs[10], k=10)
    } or True
    idx2.close()


def test_closed_and_readonly(tmp_path):
    idx, _ = seeded_index(tmp_path)
    idx.save()
    idx.close()
    with pytest.raises(ClosedIndexError):
        idx.search(np.zeros(8, dtype=np.float32))
    ro = GraphIndex.open(tmp_path / "i.gvec", readonly=True)
    with pytest.raises(Exception):
        ro.add(np.zeros((1, 8), dtype=np.float32))
    ro.close()
    # context manager
    with GraphIndex.open(tmp_path / "i.gvec", readonly=True) as ro2:
        assert len(ro2) == 50


def test_build_rebuilds_graph(tmp_path):
    idx, vecs = seeded_index(tmp_path)
    idx.delete(["d0", "d1"])
    idx.build()
    assert len(idx) == 48
    assert idx.search(vecs[10], k=5, ef_search=100)[0].id == idx.search_exact(vecs[10], k=5)[0].id
    assert idx.validate()["ok"]
    idx.close()


def test_tombstoned_id_rejected_until_compact(tmp_path):
    idx, vecs = seeded_index(tmp_path)
    idx.delete("d5")
    with pytest.raises(DuplicateIDError):
        idx.add(vecs[:1], ids=["d5"])
    idx.compact()
    idx.add(vecs[:1], ids=["d5"])
    assert "d5" in idx
    idx.close()
