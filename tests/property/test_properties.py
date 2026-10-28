import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from graphvec import GraphIndex


@settings(
    max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    arr=arrays(
        np.float32, (12, 6), elements=st.floats(-5, 5, allow_nan=False, allow_infinity=False)
    ),
)
def test_result_invariants(tmp_path_factory, arr):
    path = tmp_path_factory.mktemp("idx") / "i.gvec"
    idx = GraphIndex.create(path, dimensions=6, metric="squared_euclidean")
    ids = [f"x{i}" for i in range(len(arr))]
    idx.add(arr, ids=ids)
    q = arr[0]
    res = idx.search(q, k=5, ef_search=50)
    assert len({r.id for r in res}) == len(res), "duplicate ids"
    assert all(r.id in ids for r in res)
    scores = [r.score for r in res]
    assert scores == sorted(scores), "squared euclidean must be ascending"
    # scores agree with metric recomputed independently
    by_id = {i: v for i, v in zip(ids, arr, strict=False)}
    for r in res:
        assert abs(r.score - float(((by_id[r.id] - q) ** 2).sum())) < 1e-3
    idx.close()


@settings(
    max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(n=st.integers(min_value=1, max_value=30))
def test_save_reopen_preserves_results(tmp_path_factory, n):
    tmp_path = tmp_path_factory.mktemp("idx2")
    rng = np.random.default_rng(n)
    arr = rng.random((n, 8)).astype(np.float32)
    idx = GraphIndex.create(tmp_path / "i.gvec", dimensions=8, metric="cosine")
    idx.add(arr, ids=[f"v{i}" for i in range(n)])
    idx.save()
    before = [r.id for r in idx.search(arr[0], k=min(5, n), ef_search=50)]
    idx.close()
    idx2 = GraphIndex.open(tmp_path / "i.gvec", readonly=True)
    after = [r.id for r in idx2.search(arr[0], k=min(5, n), ef_search=50)]
    assert before == after
    idx2.close()
