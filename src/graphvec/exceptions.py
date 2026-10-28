"""GraphVec exception hierarchy."""


class GraphVecError(Exception):
    """Base class for all GraphVec errors."""


class InvalidDimensionError(GraphVecError, ValueError):
    """Vector or query dimension does not match the index."""


class InvalidVectorError(GraphVecError, ValueError):
    """Vector contains NaN/inf or is otherwise invalid."""


class UnsupportedMetricError(GraphVecError, ValueError):
    """Requested distance metric is not supported."""


class DuplicateIDError(GraphVecError, ValueError):
    """An external ID already exists in the index."""


class MissingIDError(GraphVecError, KeyError):
    """An external ID does not exist (or is deleted) in the index."""


class CorruptIndexError(GraphVecError):
    """On-disk index files are truncated, corrupt, or inconsistent."""


class UnsupportedFormatVersionError(CorruptIndexError):
    """Index format version is not supported by this release."""


class IncompatibleIndexError(GraphVecError):
    """Index parameters are incompatible with the request."""


class ConcurrentWriterError(GraphVecError):
    """Another writer holds the index lock."""


class ClosedIndexError(GraphVecError):
    """Operation attempted on a closed index."""


class InsufficientDiskSpaceError(GraphVecError):
    """Not enough free disk space for the requested operation."""


class PersistenceError(GraphVecError):
    """A save/commit operation failed part-way through."""
