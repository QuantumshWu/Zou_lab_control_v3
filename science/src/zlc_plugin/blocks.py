"""Shared-memory blocks (docs/PLUGIN_PROTOCOL.md §6).

A process writes bulk values into blocks of its own ``Arena`` and lends them
by name; the core counts every holder and sends ``release`` when the last one
is gone. A block returns to its arena only when the core has released every
lend AND no local view of the allocation is alive, so neither the core nor
this process can see its contents change.

Finalizers only enqueue: they may run inside any lock (garbage collection),
so the arena and the release sender drain their queues themselves.
"""

from __future__ import annotations

import itertools
import mmap
import os
import queue
import threading
import weakref
from typing import Any, Callable

import numpy as np

from zlc_data._arrays import ImmutableBuffer

GRANULARITY = 64 << 10


def _address(buffer: object) -> int:
    return np.frombuffer(buffer, dtype=np.uint8, count=1).__array_interface__["data"][0]


class _Allocation:
    """The one writable owner of a block; numpy views keep it alive."""

    __slots__ = ("_mapping", "__weakref__")

    def __init__(self, mapping: mmap.mmap) -> None:
        self._mapping = mapping

    def __buffer__(self, flags: int) -> memoryview:
        return memoryview(self._mapping)


class _Block:
    __slots__ = ("name", "size", "mapping", "address", "lends", "local")

    def __init__(self, name: str, size: int) -> None:
        self.name = name
        self.size = size
        self.mapping = mmap.mmap(-1, size, tagname=name)
        self.address = _address(self.mapping)
        self.lends = 0
        self.local = False


class Arena:
    """Blocks this process writes; sizes are 64 KiB multiples and recycled exactly."""

    def __init__(self, session: str) -> None:
        self._prefix = f"Local\\zlc.{session}.{os.getpid()}."
        self._counter = itertools.count()
        self._lock = threading.Lock()
        self._free: dict[int, list[_Block]] = {}
        self._by_name: dict[str, _Block] = {}
        self._by_owner: dict[int, _Block] = {}
        self._dropped: queue.SimpleQueue[str] = queue.SimpleQueue()

    def allocate(self, shape: tuple[int, ...] | int, dtype: Any) -> np.ndarray:
        """A writable C-contiguous array in a block, sent by reference when lent."""

        kind = np.dtype(dtype)
        shape = (shape,) if isinstance(shape, int) else tuple(int(item) for item in shape)
        count = int(np.prod(shape, dtype=np.int64))
        size = max(GRANULARITY, -(-count * kind.itemsize // GRANULARITY) * GRANULARITY)
        with self._lock:
            self._drain()
            pool = self._free.get(size)
            block = pool.pop() if pool else None
            if block is None:
                block = _Block(f"{self._prefix}{next(self._counter)}", size)
                self._by_name[block.name] = block
            block.local = True
            owner = _Allocation(block.mapping)
            self._by_owner[id(owner)] = block
        weakref.finalize(owner, self._dropped.put, block.name)
        return np.frombuffer(owner, dtype=kind, count=count).reshape(shape)

    def lend(self, array: np.ndarray, holder: "Attachments") -> dict[str, Any]:
        """The ``$array`` reference for ``array``, copying it into a block if it is in none."""

        found = self._find(array) or holder.find(array)
        if found is None:
            fresh = self.allocate(array.shape, array.dtype)
            fresh[...] = array
            found = self._find(fresh)
            assert found is not None
        name, size, offset = found
        with self._lock:
            block = self._by_name.get(name)
            if block is not None:
                block.lends += 1
        return {"block": name, "size": size, "offset": offset, "dtype": array.dtype.str, "shape": list(array.shape)}

    def released(self, name: str) -> None:
        """The core released one lend of ``name``."""

        with self._lock:
            self._drain()
            block = self._by_name.get(name)
            if block is None or block.lends == 0:
                return
            block.lends -= 1
            self._recycle(block)

    def _find(self, array: np.ndarray) -> tuple[str, int, int] | None:
        current: object = array
        while isinstance(current, np.ndarray):
            current = current.base
        block = self._by_owner.get(id(current)) if isinstance(current, _Allocation) else None
        return None if block is None else _placement(array, block.name, block.size, block.address)

    def _drain(self) -> None:
        while True:
            try:
                name = self._dropped.get_nowait()
            except queue.Empty:
                return
            block = self._by_name[name]
            block.local = False
            self._by_owner = {key: value for key, value in self._by_owner.items() if value is not block}
            self._recycle(block)

    def _recycle(self, block: _Block) -> None:
        if block.lends == 0 and not block.local:
            self._free.setdefault(block.size, []).append(block)


def _placement(array: np.ndarray, name: str, size: int, base: int) -> tuple[str, int, int] | None:
    if not array.flags.c_contiguous:
        return None
    offset = array.__array_interface__["data"][0] - base
    if offset < 0 or offset + array.nbytes > size or offset % max(1, array.dtype.alignment):
        return None
    return name, size, offset


class Attachments:
    """Blocks other processes lent to this one, mapped read-only once each."""

    def __init__(self, release: Callable[[str], None]) -> None:
        self._release = release
        self._lock = threading.Lock()
        self._mappings: dict[str, tuple[mmap.mmap, int, int]] = {}
        self._views: dict[int, str] = {}

    def attach(self, reference: dict[str, Any]) -> np.ndarray:
        """A read-only array over one lend; dropping its last view releases the lend."""

        name = str(reference["block"])
        size = int(reference["size"])
        with self._lock:
            entry = self._mappings.get(name)
            if entry is None or entry[1] < size:
                mapping = mmap.mmap(-1, size, tagname=name, access=mmap.ACCESS_READ)
                entry = (mapping, size, _address(mapping))
                self._mappings[name] = entry
        view = ImmutableBuffer(entry[0])
        self._views[id(view)] = name
        weakref.finalize(view, self._dropped, id(view), name)
        dtype = np.dtype(reference["dtype"])
        shape = tuple(int(item) for item in reference["shape"])
        count = int(np.prod(shape, dtype=np.int64))
        return np.frombuffer(view, dtype=dtype, count=count, offset=int(reference["offset"])).reshape(shape)

    def find(self, array: np.ndarray) -> tuple[str, int, int] | None:
        current: object = array
        while isinstance(current, np.ndarray):
            current = current.base
        name = self._views.get(id(current)) if type(current) is ImmutableBuffer else None
        if name is None:
            return None
        mapping, size, base = self._mappings[name]
        return _placement(array, name, size, base)

    def _dropped(self, identity: int, name: str) -> None:
        self._views.pop(identity, None)
        self._release(name)


__all__ = ["Arena", "Attachments", "GRANULARITY"]
