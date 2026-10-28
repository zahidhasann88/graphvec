"""Incremental insert / delete / compact example."""

import numpy as np

from graphvec import GraphIndex

rng = np.random.default_rng(1)
with GraphIndex.create("./inc.gvec", dimensions=32, metric="cosine", overwrite=True) as idx:
    idx.add(rng.random((100, 32)).astype(np.float32), ids=[f"d{i}" for i in range(100)])
    idx.add(rng.random((10, 32)).astype(np.float32), ids=[f"d{i}" for i in range(100, 110)])
    idx.delete(["d0", "d1"])
    idx.save()
    print("live:", len(idx), "valid:", idx.validate()["ok"])
    idx.compact()
    print("after compact, total:", idx.total_count)
