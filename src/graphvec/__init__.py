"""GraphVec — local-first, file-based ANN vector search."""

from graphvec.api import GraphIndex, SearchResult
from graphvec.exceptions import (
    ClosedIndexError,
    ConcurrentWriterError,
    CorruptIndexError,
    DuplicateIDError,
    GraphVecError,
    IncompatibleIndexError,
    InsufficientDiskSpaceError,
    InvalidDimensionError,
    InvalidVectorError,
    MissingIDError,
    PersistenceError,
    UnsupportedFormatVersionError,
    UnsupportedMetricError,
)

__version__ = "0.2.0"
__all__ = [
    "GraphIndex",
    "SearchResult",
    "__version__",
    "GraphVecError",
    "InvalidDimensionError",
    "InvalidVectorError",
    "UnsupportedMetricError",
    "DuplicateIDError",
    "MissingIDError",
    "CorruptIndexError",
    "UnsupportedFormatVersionError",
    "IncompatibleIndexError",
    "ConcurrentWriterError",
    "ClosedIndexError",
    "InsufficientDiskSpaceError",
    "PersistenceError",
]
