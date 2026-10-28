"""Persistent binary format (format version 1). All integers little-endian.

Directory layout (``<path>.gvec/``):
- ``manifest.json``  commit point; written via tmp+atomic replace, last.
- ``vectors.bin``    magic GVEC + u32 version + u64 count + u32 dims, then
                     count*dims float32 little-endian rows.
- ``graph.bin``      magic GVGR + u32 version + u64 count + i64 entry_point
                     + i32 max_level, then per node: i32 level, and per layer
                     u32 degree followed by degree x i32 neighbour ids.
- ``ids.bin``        magic GVID + u32 version + u64 count, then per id
                     u32 byte-length + UTF-8 bytes.
- ``tombstones.bin`` magic GVTB + u32 version + u64 count, then count bytes
                     (0 = live, 1 = deleted).

Compatibility: readers reject unknown major format versions. Minor
additions must keep these headers parseable; migrations, when introduced,
will read vN and write vN+1 only on explicit ``save()``/``compact()``.
"""

from __future__ import annotations

import hashlib
import struct

FORMAT_VERSION = 1
VEC_MAGIC = b"GVEC"
GRAPH_MAGIC = b"GVGR"
IDS_MAGIC = b"GVID"
TOMB_MAGIC = b"GVTB"

HEADER = struct.Struct("<4sIQI")  # magic, version, count, dims (vectors only)


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            data = fh.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()
