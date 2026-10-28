import numpy as np
import pytest

from graphvec import GraphIndex
from graphvec.graph import Graph


def data(n=600, d=16, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, d)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v, np.einsum("ij,ij->i", v, v).astype(np.float64)


def sequential(v, norms, m=8, ef=40):
    g = Graph(v.shape[1], "cosine", m, ef, seed=3)
    for i in range(len(v)):
        g.insert(i, v[: i + 1], norms[: i + 1])
    return g


def same_graph(a: Graph, b: Graph) -> bool:
    n = a.count
    return (
        n == b.count
        and a.entry_point == b.entry_point
        and np.array_equal(a.levels[:n], b.levels[:n])
        and np.array_equal(a.deg0[:n], b.deg0[:n])
        and np.array_equal(a.adj0[:n], b.adj0[:n])
        and a.upper == b.upper
    )


def test_batch_of_one_matches_sequential_insert():
    v, norms = data()
    g = Graph(v.shape[1], "cosine", 8, 40, seed=3)
    g.insert_batch(0, 300, v[:300], norms[:300], batch=1)
    g.insert_batch(300, len(v), v, norms, batch=1)
    assert same_graph(g, sequential(v, norms))


def test_lockstep_search_matches_single_query_search():
    v, norms = data()
    g = sequential(v, norms)
    queries, entries = np.arange(500, 540), np.full(40, g.entry_point)
    for q, e in zip(queries.tolist(), entries.tolist(), strict=True):
        for layer in range(g.max_level, 0, -1):
            e = g._greedy_layer(v[q], float(norms[q]), e, layer, v, norms)
        entries[queries == q] = e
    ids, dists = g._search_many(queries, entries, 40, v, norms)
    for j, q in enumerate(queries.tolist()):
        want, _ = g._search_layer(v[q], float(norms[q]), int(entries[j]), 40, 0, v, norms)
        assert set(ids[j][ids[j] >= 0].tolist()) == set(want.tolist())
    assert not g._visited.any()  # visited bits are cleared for the next group


@pytest.mark.parametrize("metric", ["cosine", "dot", "squared_euclidean"])
def test_lockstep_select_matches_single_select(metric):
    rng = np.random.default_rng(2)
    v = rng.standard_normal((300, 16)).astype(np.float32)
    norms = np.einsum("ij,ij->i", v, v).astype(np.float64)
    g = Graph(16, metric, max_neighbors=4)
    rows, width = 12, 60
    ids = np.full((rows, width), -1, dtype=np.int64)
    dists = np.full((rows, width), np.inf)
    for r in range(rows):
        k = [3, 8, 9, 60][r % 4]  # include rows with no more than m candidates
        ids[r, :k] = rng.choice(300, k, replace=False)
        dists[r, :k] = rng.random(k)
    got = g._select_many(ids, dists, 8, v, norms)
    for r in range(rows):
        keep = ids[r] >= 0
        want = g._select(ids[r][keep].astype(np.int32), dists[r][keep], 8, v, norms)
        assert got[r].tolist() == want.tolist()


@pytest.mark.parametrize("metric", ["cosine", "dot", "squared_euclidean"])
def test_batched_graph_is_valid_and_accurate(tmp_path, metric):
    rng = np.random.default_rng(1)
    v = rng.standard_normal((1500, 24)).astype(np.float32)
    with GraphIndex.create(tmp_path / "i.gvec", dimensions=24, metric=metric) as idx:
        idx.add(v[:700], insert_batch=64)
        idx.add(v[700:], insert_batch=64)
        report = idx.validate()
        assert report["ok"], report["errors"]
        assert report["graph"]["layer0_reachable"] == 1500
        queries = rng.standard_normal((40, 24)).astype(np.float32)
        hits = 0
        for q in queries:
            got = {r.id for r in idx.search(q, k=10, ef_search=100)}
            hits += len(got & {r.id for r in idx.search_exact(q, k=10)})
        assert hits / 400 >= 0.9


def test_batched_build_is_deterministic(tmp_path):
    v, _ = data(400)
    files = []
    for name in ("a", "b"):
        with GraphIndex.create(tmp_path / name, dimensions=16) as idx:
            idx.add(v, insert_batch=32)
            idx.save()
        files.append((tmp_path / name / "graph.bin").read_bytes())
    assert files[0] == files[1]


def test_insert_batch_out_of_range(tmp_path):
    with GraphIndex.create(tmp_path / "i.gvec", dimensions=4) as idx:
        for bad in (0, 65):
            with pytest.raises(ValueError):
                idx.add(np.ones((2, 4), dtype=np.float32), insert_batch=bad)
        assert len(idx) == 0
