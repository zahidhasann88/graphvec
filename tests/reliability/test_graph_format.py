import json
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

from graphvec import GraphIndex
from graphvec import format as fmt
from graphvec.exceptions import CorruptIndexError

FIXTURES = Path(__file__).parent.parent / "fixtures"


def expected_v1():
    exp = json.loads((FIXTURES / "v1_expected.json").read_text())
    return exp, np.array(exp["queries"], dtype=np.float32)


def results(idx, queries):
    return [[r.id for r in idx.search(q, k=5, ef_search=40)] for q in queries]


def test_v1_index_opens_readonly():
    exp, queries = expected_v1()
    with GraphIndex.open(FIXTURES / "v1_index", readonly=True) as idx:
        assert idx.format_version == 1
        assert len(idx) == exp["live"] and "f3" not in idx
        assert np.allclose(idx.get("f10"), exp["get_f10"])
        assert results(idx, queries) == exp["results"]
        assert idx.validate()["ok"]


def test_v1_index_migrates_on_save(tmp_path):
    exp, queries = expected_v1()
    p = tmp_path / "v1"
    shutil.copytree(FIXTURES / "v1_index", p)
    with GraphIndex.open(p) as idx:
        idx.save()
        assert idx.format_version == 2
    with GraphIndex.open(p, readonly=True, verify_checksums=True) as idx:
        assert idx.format_version == 2
        assert json.loads((p / "manifest.json").read_text())["format_version"] == 2
        assert len(idx) == exp["live"]
        assert results(idx, queries) == exp["results"]


def built(tmp_path, n=80):
    rng = np.random.default_rng(4)
    p = tmp_path / "g.gvec"
    with GraphIndex.create(p, dimensions=8, max_neighbors=4) as idx:
        idx.add(rng.standard_normal((n, 8)).astype(np.float32))
        idx.save()
    return p


def test_readonly_open_memory_maps_graph(tmp_path):
    p = built(tmp_path)
    with GraphIndex.open(p, readonly=True) as idx:
        assert isinstance(idx._graph.adj0, np.memmap)
        assert idx.search(np.ones(8, dtype=np.float32), k=3)
    with GraphIndex.open(p) as idx:
        assert not isinstance(idx._graph.adj0, np.memmap)


def test_closed_reader_releases_files(tmp_path):
    p = built(tmp_path)
    reader = GraphIndex.open(p, readonly=True)
    reader.search(np.ones(8, dtype=np.float32), k=3)
    reader.close()
    with GraphIndex.open(p) as writer:
        writer.add(np.ones((1, 8), dtype=np.float32), ids=["extra"])
        writer.save()  # replaces graph.bin and vectors.bin
    with GraphIndex.open(p, readonly=True) as idx:
        assert "extra" in idx


def graph_layout(p):
    n = json.loads((p / "manifest.json").read_text())["count"]
    width = 4 * 4
    off_deg = fmt.GRAPH_HEADER.size
    off_adj = off_deg + 4 * n
    return n, width, off_deg, off_adj


def patch(path, offset, fmt_str, *values):
    data = bytearray(path.read_bytes())
    struct.pack_into(fmt_str, data, offset, *values)
    path.write_bytes(bytes(data))


def test_out_of_range_neighbour_rejected(tmp_path):
    p = built(tmp_path)
    n, _width, _off_deg, off_adj = graph_layout(p)
    patch(p / "graph.bin", off_adj, "<i", n + 5)
    with pytest.raises(CorruptIndexError, match="out of range"):
        GraphIndex.open(p, readonly=True)


def test_negative_neighbour_rejected(tmp_path):
    p = built(tmp_path)
    *_rest, off_adj = graph_layout(p)
    patch(p / "graph.bin", off_adj, "<i", -3)
    with pytest.raises(CorruptIndexError):
        GraphIndex.open(p, readonly=True)


def test_dirty_unused_slot_rejected(tmp_path):
    p = built(tmp_path)
    n, width, off_deg, off_adj = graph_layout(p)
    deg0 = struct.unpack_from("<i", (p / "graph.bin").read_bytes(), off_deg)[0]
    assert deg0 < width
    patch(p / "graph.bin", off_adj + 4 * deg0, "<i", 0)
    with pytest.raises(CorruptIndexError, match="unused"):
        GraphIndex.open(p, readonly=True)


def test_bad_degree_rejected(tmp_path):
    p = built(tmp_path)
    *_rest, off_deg, _off_adj = graph_layout(p)
    patch(p / "graph.bin", off_deg, "<i", 10_000)
    with pytest.raises(CorruptIndexError, match="degree"):
        GraphIndex.open(p, readonly=True)


def test_truncated_graph_rejected(tmp_path):
    p = built(tmp_path)
    data = (p / "graph.bin").read_bytes()
    (p / "graph.bin").write_bytes(data[: len(data) // 2])
    with pytest.raises(CorruptIndexError, match="truncated"):
        GraphIndex.open(p, readonly=True)


def test_trailing_bytes_rejected(tmp_path):
    p = built(tmp_path)
    with open(p / "graph.bin", "ab") as fh:
        fh.write(b"\x00" * 4)
    with pytest.raises(CorruptIndexError, match="trailing"):
        GraphIndex.open(p, readonly=True)


def test_graph_parameters_must_match_manifest(tmp_path):
    p = built(tmp_path)
    manifest = json.loads((p / "manifest.json").read_text())
    manifest["max_neighbors"] = 8
    (p / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(CorruptIndexError):
        GraphIndex.open(p, readonly=True)
