import json

import numpy as np
import pytest
from typer.testing import CliRunner

from graphvec import GraphIndex
from graphvec.cli import app

runner = CliRunner()


def test_full_lifecycle(tmp_path):
    rng = np.random.default_rng(7)
    vecs = rng.random((300, 32)).astype(np.float32)
    path = tmp_path / "idx.gvec"
    with GraphIndex.create(path, dimensions=32, metric="cosine") as idx:
        idx.add(vecs[:200], ids=[f"a{i}" for i in range(200)])
        idx.save()
    with GraphIndex.open(path) as idx:
        idx.add(vecs[200:], ids=[f"a{i}" for i in range(200, 300)])
        idx.delete(["a0", "a1"])
        idx.save()
        assert len(idx) == 298
        idx.compact()
        assert idx.validate()["ok"]
    with GraphIndex.open(path, readonly=True) as idx:
        assert len(idx) == 298
        res = idx.search(vecs[10], k=10)
        assert res and res[0].id == "a10"


def test_cli_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    vecs = rng.random((100, 16)).astype(np.float32)
    np.save(tmp_path / "v.npy", vecs)
    (tmp_path / "ids.json").write_text(json.dumps([f"c{i}" for i in range(100)]))
    np.save(tmp_path / "q.npy", vecs[:5])
    p = str(tmp_path / "cli.gvec")
    assert (
        runner.invoke(app, ["create", p, "--dimensions", "16", "--metric", "cosine"]).exit_code == 0
    )
    r = runner.invoke(
        app, ["import", p, str(tmp_path / "v.npy"), "--ids", str(tmp_path / "ids.json")]
    )
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["info", p, "--json"])
    assert r.exit_code == 0 and "16" in r.output
    assert json.loads(r.output)["format_version"] == 2
    r = runner.invoke(
        app, ["search", p, "--query-file", str(tmp_path / "q.npy"), "--top-k", "5", "--json"]
    )
    assert r.exit_code == 0, r.output
    r = runner.invoke(
        app, ["search", p, "--query", json.dumps(vecs[0].tolist()), "--top-k", "5", "--json"]
    )
    assert r.exit_code == 0, r.output
    assert "c0" in r.output
    assert runner.invoke(app, ["validate", p]).exit_code == 0
    r = runner.invoke(app, ["benchmark", p, "--queries", str(tmp_path / "q.npy"), "--json"])
    assert r.exit_code == 0, r.output
    assert runner.invoke(app, ["version"]).exit_code == 0
    assert runner.invoke(app, ["compact", p]).exit_code == 0
    r = runner.invoke(app, ["export", p, "--output", str(tmp_path / "out.npy")])
    assert r.exit_code == 0 and (tmp_path / "out.npy").exists()


def test_cli_import_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    vecs = np.random.default_rng(5).random((30, 8)).astype(np.float32)
    table = pa.table({"id": [f"p{i}" for i in range(30)], "vector": vecs.tolist()})
    pq.write_table(table, tmp_path / "v.parquet")
    p = str(tmp_path / "pq.gvec")
    assert runner.invoke(app, ["create", p, "--dimensions", "8"]).exit_code == 0
    r = runner.invoke(
        app, ["import", p, str(tmp_path / "v.parquet"), "--batch-size", "7", "--json"]
    )
    assert r.exit_code == 0, r.output
    with GraphIndex.open(p, readonly=True) as idx:
        assert len(idx) == 30 and "p29" in idx
        assert np.allclose(idx.get("p3"), vecs[3] / np.linalg.norm(vecs[3]), atol=1e-6)


def test_cli_rejects_unknown_source(tmp_path):
    p = str(tmp_path / "u.gvec")
    assert runner.invoke(app, ["create", p, "--dimensions", "8"]).exit_code == 0
    (tmp_path / "v.csv").write_text("1,2")
    r = runner.invoke(app, ["import", p, str(tmp_path / "v.csv")])
    assert r.exit_code != 0


def test_cli_unlock(tmp_path):
    p = tmp_path / "l.gvec"
    assert runner.invoke(app, ["create", str(p), "--dimensions", "4"]).exit_code == 0
    assert runner.invoke(app, ["unlock", str(p)]).output.strip() == "not locked"
    (p / ".writer.lock").write_text("123")
    r = runner.invoke(app, ["unlock", str(p)], input="n\n")
    assert r.exit_code != 0 and (p / ".writer.lock").exists()
    r = runner.invoke(app, ["unlock", str(p), "--yes"])
    assert r.exit_code == 0 and not (p / ".writer.lock").exists()
    assert runner.invoke(app, ["unlock", str(tmp_path / "missing")]).exit_code == 1
