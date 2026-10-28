"""Basic GraphVec example (matches README quick-start)."""

import numpy as np

from graphvec import GraphIndex

rng = np.random.default_rng(42)
vectors = rng.random((10_000, 384), dtype=np.float32)
ids = [f"doc-{i}" for i in range(len(vectors))]

with GraphIndex.create("./vectors.gvec", dimensions=384, metric="cosine", overwrite=True) as index:
    index.add(vectors, ids=ids)
    index.save()

with GraphIndex.open("./vectors.gvec", readonly=True) as index:
    results = index.search(rng.random(384, dtype=np.float32), k=10, ef_search=100)
    for result in results:
        print(result.id, result.score)
