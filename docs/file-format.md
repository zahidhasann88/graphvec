# File format (version 1)

An index is a directory containing `manifest.json`, `vectors.bin`,
`graph.bin`, `ids.bin`, `tombstones.bin`. All binary integers/floats are
little-endian. Each `.bin` starts with a 4-byte magic (`GVEC`/`GVGR`/
`GVID`/`GVTB`) and a u32 format version; layouts are specified in
`src/graphvec/format.py`. `manifest.json` records dimensions, metric,
counts, parameters and SHA-256 checksums of the data files, and is written
last via temp-file + atomic rename + fsync — it is the commit point.
Leftover `*.tmp` files from interrupted writes are ignored on open.

Guarantees: readers validate magic, version, file sizes, counts, ID
lengths and graph neighbour ranges, and fail with `CorruptIndexError` /
`UnsupportedFormatVersionError` rather than loading malformed data. No
pickle is ever used. Compatibility: unknown format versions are rejected;
a future migration will read vN and write vN+1 only on explicit save.
Portability assumes little-endian hosts (x86/ARM mainstream) — documented
limitation for exotic big-endian hosts.
