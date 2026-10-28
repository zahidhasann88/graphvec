import numpy as np
import pytest

from graphvec import GraphIndex
from graphvec.exceptions import InvalidDimensionError, InvalidVectorError, UnsupportedMetricError
from graphvec.metrics import normalize_rows


def make(tmp_path, metric="cosine", d=16, n=200, seed=0):
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, d)).astype(np.float32)
    idx = GraphIndex.create(tmp_path / "i.gvec", dimensions=d, metric=metric)
    idx.add(vecs, ids=[f"id-{i}" for i in range(n)])
    return idx, vecs


def test_normalize_zero_row():
    out = normalize_rows(np.array([[0, 0], [3, 4]], dtype=np.float32))
    assert np.allclose(out[0], [0, 0])
    assert np.allclose(out[1], [0.6, 0.8])


@pytest.mark.parametrize("metric", ["cosine", "dot", "squared_euclidean"])
def test_exact_matches_numpy_reference(tmp_path, metric):
    idx, vecs = make(tmp_path, metric=metric)
    q = vecs[3]
    res = idx.search_exact(q, k=10)
    # independent reference
    if metric == "cosine":
        a = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        scores = a @ (q / np.linalg.norm(q))
        order = np.argsort(-scores, kind="stable")[:10]
    elif metric == "dot":
        scores = vecs @ q
        order = np.argsort(-scores, kind="stable")[:10]
    else:
        scores = ((vecs - q) ** 2).sum(axis=1)
        order = np.argsort(scores, kind="stable")[:10]
    assert [r.id for r in res] == [f"id-{i}" for i in order]
    assert np.allclose([r.score for r in res], scores[order], atol=1e-4)
    idx.close()


@pytest.mark.parametrize("metric", ["cosine", "dot", "squared_euclidean"])
def test_ann_recall(tmp_path, metric):
    idx, vecs = make(tmp_path, metric=metric, n=500, d=32, seed=1)
    rng = np.random.default_rng(5)
    recalls = []
    for _ in range(20):
        q = rng.standard_normal(32).astype(np.float32)
        ann = {r.id for r in idx.search(q, k=10, ef_search=100)}
        exact = {r.id for r in idx.search_exact(q, k=10)}
        recalls.append(len(ann & exact) / 10)
    assert float(np.mean(recalls)) >= 0.8
    idx.close()


def test_reject_bad_metric(tmp_path):
    with pytest.raises(UnsupportedMetricError):
        GraphIndex.create(tmp_path / "x.gvec", dimensions=4, metric="manhattan")


def test_reject_nan_and_dim(tmp_path):
    idx = GraphIndex.create(tmp_path / "i.gvec", dimensions=4)
    with pytest.raises(InvalidVectorError):
        idx.add(np.array([[np.nan, 0, 0, 0]], dtype=np.float32))
    with pytest.raises(InvalidDimensionError):
        idx.add(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(InvalidDimensionError):
        idx.search(np.zeros(3, dtype=np.float32))
    idx.close()


def test_empty_and_small(tmp_path):
    idx = GraphIndex.create(tmp_path / "i.gvec", dimensions=4)
    assert idx.search(np.zeros(4, dtype=np.float32)) == []
    idx.add(np.eye(4, dtype=np.float32), ids=list("abcd"))
    res = idx.search(np.array([1, 0, 0, 0], dtype=np.float32), k=10)
    assert res[0].id == "a" and len(res) == 4
    with pytest.raises(ValueError):
        idx.search(np.zeros(4, dtype=np.float32), k=0)
    idx.close()


def test_zero_vector_cosine(tmp_path):
    idx = GraphIndex.create(tmp_path / "i.gvec", dimensions=3, metric="cosine")
    idx.add(np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32), ids=["z", "x"])
    res = idx.search(np.array([1, 0, 0], dtype=np.float32), k=2)
    assert res[0].id == "x"
    assert any(r.id == "z" and abs(r.score) < 1e-6 for r in res)
    idx.close()


def test_batch_search(tmp_path):
    idx, vecs = make(tmp_path, n=50)
    out = idx.search_batch(vecs[:5], k=5)
    assert len(out) == 5 and all(len(r) == 5 for r in out)
    assert out[0][0].id == "id-0"
    idx.close()
