# File format (version 2)

An index is a directory containing `manifest.json`, `vectors.bin`,
`graph.bin`, `ids.bin`, `tombstones.bin`. All binary integers/floats are
little-endian. Each `.bin` starts with a 4-byte magic (`GVEC`/`GVGR`/
`GVID`/`GVTB`) and a u32 format version that must match the manifest;
byte-level layouts are specified in `src/graphvec/format.py`.
`manifest.json` records dimensions, metric, counts, parameters and SHA-256
checksums of the data files, and is written last via temp-file + atomic
rename + fsync — it is the commit point. Leftover `*.tmp` files from
interrupted writes are ignored on open. A writer may also keep a `.work/`
directory of raw, uncommitted working files (see
[architecture](architecture.md#working-copy)); it is not part of the
format, readers ignore it, and the next writer deletes a stale one.

## graph.bin (version 2)

A 40-byte header (magic, version, count, entry point, max level, `M`, row
width `4·M`), then three fixed-size arrays that can be memory-mapped
directly: layer-0 degrees (`count` × i32), layer-0 neighbour rows
(`count` × width × i32, unused slots `-1`) and node levels (`count` × u8).
A variable-length section follows with the neighbour lists of the nodes
that have upper layers.

## Guarantees

Readers validate magic, version, file sizes, counts, ID lengths, degrees,
every layer-0 and upper-layer neighbour id, unused slots and the entry
point, and fail with `CorruptIndexError` / `UnsupportedFormatVersionError`
rather than loading malformed data. No pickle is ever used. Portability
assumes little-endian hosts (x86/ARM mainstream) — a documented limitation
for exotic big-endian hosts.

## Compatibility and migration

- GraphVec 0.2 reads version 1 (0.1.x) and version 2 indexes.
- It always writes version 2. Opening a version-1 index for writing and
  calling `save()` (or `compact()`) migrates it in place; read-only opens
  never modify files. Only `graph.bin` changes between the versions.
- GraphVec 0.1.x cannot read version 2. Keep a copy of an index if you may
  need to downgrade.
- Unknown versions are rejected. Future versions follow the same rule:
  read the previous version, write the new one only on an explicit save.
