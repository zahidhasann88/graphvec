"""Persistent binary format (format version 2). All integers little-endian.

Directory layout (``<path>.gvec/``):
- ``manifest.json``  commit point; written via tmp+atomic replace, last.
- ``vectors.bin``    magic GVEC + u32 version + u64 count + u32 dims, then
                     count*dims float32 little-endian rows.
- ``graph.bin``      magic GVGR + u32 version + u64 count + i64 entry_point
                     + i32 max_level + u32 max_neighbors (M) + u32 width
                     (``4*M``) + u32 reserved (40-byte header), then:
                     count x i32 layer-0 degree; count x width i32 layer-0
                     neighbour ids (unused slots -1); count x u8 node level;
                     u64 number of nodes with level > 0, and for each of them
                     (ascending id) i32 node + u32 level, then per layer
                     1..level a u32 degree followed by degree x i32 ids.
- ``ids.bin``        magic GVID + u32 version + u64 count, then per id
                     u32 byte-length + UTF-8 bytes.
- ``tombstones.bin`` magic GVTB + u32 version + u64 count, then count bytes
                     (0 = live, 1 = deleted).

Compatibility: every file header carries the manifest's format version.
Version 1 differs only in ``graph.bin`` (i32 level, then per layer a u32
degree and the ids, for every node). Readers accept both; writers always
write version 2, so opening a version-1 index for writing and saving it
migrates it. Unknown versions are rejected.
"""

from __future__ import annotations

import hashlib
import struct

FORMAT_VERSION = 2
READABLE_FORMAT_VERSIONS = (1, 2)
VEC_MAGIC = b"GVEC"
GRAPH_MAGIC = b"GVGR"
IDS_MAGIC = b"GVID"
TOMB_MAGIC = b"GVTB"

HEADER = struct.Struct("<4sIQI")  # magic, version, count, dims (vectors only)
GRAPH_HEADER = struct.Struct("<4sIQqiIII")


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            data = fh.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()
