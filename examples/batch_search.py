"""Batch search example."""

import numpy as np

from graphvec import GraphIndex

rng = np.random.default_rng(0)
with GraphIndex.create("./batch.gvec", dimensions=64, metric="dot", overwrite=True) as idx:
    idx.add(rng.random((1000, 64)).astype(np.float32), ids=[f"v{i}" for i in range(1000)])
    idx.save()
    for i, results in enumerate(idx.search_batch(rng.random((5, 64)).astype(np.float32), k=3)):
        print(i, [(r.id, round(r.score, 3)) for r in results])
