"""Growable arrays for a writable index's working copy.

A writable index keeps its vectors and layer-0 graph arrays in a working
copy that ``save()`` commits. :class:`WorkingCopy` holds them in RAM
(:class:`MemoryBacking`) until they would exceed a memory limit, then moves
them to raw files under ``<index>/.work/`` that are memory-mapped
(:class:`FileBacking`), so the OS can page them out and an index can grow
beyond RAM. Memory-mapped writes cost extra page faults, which is why RAM
is used while it suffices.

The committed files are never written in place, so crash safety still
rests on ``save()`` writing new files and the manifest last; a crash only
leaves a stale ``.work/`` directory, which the next writer deletes.

Rows at or beyond the ``count`` passed to ``grow`` are uninitialised in a
file-backed array (callers must write a row before reading it).
"""

from __future__ import annotations

import ctypes
import itertools
import os
import shutil
import sys
from pathlib import Path

import numpy as np

WORK_DIR = ".work"
_COPY_BYTES = 64 << 20


def copy_rows(dst: np.ndarray, src: np.ndarray, count: int) -> None:
    """Copy ``src[:count]`` into ``dst`` in chunks, so memmaps are never read whole."""
    row_bytes = max(1, src[:1].nbytes)
    step = max(1, _COPY_BYTES // row_bytes)
    for start in range(0, count, step):
        end = min(start + step, count)
        dst[start:end] = src[start:end]


def physical_memory() -> int | None:
    """Total physical memory in bytes, or None if it cannot be determined."""
    if sys.platform == "win32":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong),
                ("avail_phys", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("avail_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return int(status.total_phys)
        return None
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        return None


def default_memory_limit() -> int:
    """A quarter of physical memory (1 GiB if unknown).

    The rest is left for the ID map, search temporaries, readers and the
    OS page cache that serves the memory-mapped files.
    """
    total = physical_memory()
    return total // 4 if total else 1 << 30


class MemoryBacking:
    """Arrays held in process memory."""

    def grow(self, name: str, old: np.ndarray, count: int, capacity: int, fill=0) -> np.ndarray:
        new = np.full((capacity, *old.shape[1:]), fill, dtype=old.dtype)
        copy_rows(new, old, count)
        return new

    def close(self) -> None:
        pass


class FileBacking:
    """Arrays memory-mapped from raw files in a working directory.

    Growing remaps the same file at a larger size; NumPy extends the file
    and the existing rows stay where they are, so nothing is copied. An
    array that did not come from this backing (a fresh graph, an array in
    RAM, or arrays loaded from the committed files) is copied into a new
    file instead.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, tuple[Path, np.ndarray]] = {}
        self._stale: list[Path] = []
        self._gen = itertools.count()

    def grow(self, name: str, old: np.ndarray, count: int, capacity: int, fill=0) -> np.ndarray:
        self._sweep()
        current = self._files.get(name)
        if current is not None and current[1] is old:
            path = current[0]
        else:
            if current is not None:
                self._stale.append(current[0])
            path = self.directory / f"{name}.{next(self._gen)}.bin"
            path.touch()
        mm = np.memmap(path, dtype=old.dtype, mode="r+", shape=(capacity, *old.shape[1:]))
        new = mm.view(np.ndarray)  # plain ndarray: cheaper indexing, still file-backed
        if current is None or current[1] is not old:
            copy_rows(new, old, count)
        self._files[name] = (path, new)
        return new

    def _sweep(self) -> None:
        """Delete replaced files once nothing maps them (Windows refuses before)."""
        keep = []
        for path in self._stale:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                keep.append(path)
        self._stale = keep

    def close(self) -> None:
        """Forget every array and remove the working directory (best effort).

        Callers must drop their own references first, or Windows keeps the
        files mapped and they are only removed by the next writer.
        """
        self._files.clear()
        self._stale.clear()
        try:
            shutil.rmtree(self.directory)
        except OSError:
            pass


class WorkingCopy:
    """RAM until the arrays would exceed ``memory_limit`` bytes, then files.

    The switch happens inside ``grow``: from then on every array that grows
    is copied into a file, and all of them follow within one doubling,
    since the vectors and the graph grow together. It never switches back.
    """

    def __init__(self, directory: Path, memory_limit: int | None = None) -> None:
        self.directory = Path(directory)
        self.memory_limit = default_memory_limit() if memory_limit is None else memory_limit
        if self.memory_limit < 0:
            raise ValueError("memory_limit must be at least 0")
        shutil.rmtree(self.directory, ignore_errors=True)  # left by a crashed writer
        self._memory = MemoryBacking()
        self._files: FileBacking | None = None
        self._sizes: dict[str, int] = {}

    @property
    def on_disk(self) -> bool:
        return self._files is not None

    def grow(self, name: str, old: np.ndarray, count: int, capacity: int, fill=0) -> np.ndarray:
        row_bytes = old.itemsize * int(np.prod(old.shape[1:], dtype=np.int64))
        self._sizes[name] = capacity * row_bytes
        if self._files is None and sum(self._sizes.values()) > self.memory_limit:
            self._files = FileBacking(self.directory)
        if self._files is not None:
            return self._files.grow(name, old, count, capacity, fill)
        return self._memory.grow(name, old, count, capacity, fill)

    def close(self) -> None:
        if self._files is not None:
            self._files.close()
            self._files = None
