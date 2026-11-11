"""Shared-memory blocks (docs/PLUGIN_PROTOCOL.md §6).

A process writes bulk values into blocks of its own ``Arena`` and lends them
by name; the core counts every holder and sends ``release`` when the last one
is gone. A block returns to its arena only when the core has released every
lend AND no local view of the allocation is alive, so neither the core nor
this process can see its contents change.

An arena keeps the free blocks its work comes back for and gives the rest back
to the system. Which are which is read from its own use, never from a number:
a size's free blocks that sat untouched through one whole turn of that size --
as many allocations as it has blocks -- were not needed by that turn, and a
size that has gone unallocated for longer than it ever did before, counted in
the arena's own allocations, is not in use at all. A block given back is unmapped here and
said to the core (``retired``), which unmaps it too; a process that only
borrowed it lets its mapping go with its last view.

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


class _Use:
    """How one size of block has been used since its free blocks were last looked at."""

    __slots__ = ("blocks", "turn", "least_free", "used_at", "widest")

    def __init__(self, clock: int) -> None:
        #: The blocks of this size that exist, free or not.
        self.blocks = 0
        #: Its own allocations since the last look, and the fewest free blocks it had after any of them.
        self.turn = 0
        self.least_free = 0
        #: The arena's allocation count when this size was last allocated, and the most allocations of
        #: other sizes that ever passed between two of its own.
        self.used_at = clock
        self.widest = 0


class Arena:
    """Blocks this process writes; sizes are 64 KiB multiples and recycled exactly.

    ``retired`` is told the names of blocks given back to the system (module docstring).
    """

    def __init__(self, session: str, retired: Callable[[list[str]], None] | None = None) -> None:
        self._prefix = f"Local\\zlc.{session}.{os.getpid()}."
        self._counter = itertools.count()
        self._lock = threading.Lock()
        self._free: dict[int, list[_Block]] = {}
        self._by_name: dict[str, _Block] = {}
        self._by_owner: dict[int, _Block] = {}
        self._dropped: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._retired = retired
        self._use: dict[int, _Use] = {}
        #: Every allocation this arena has made: the clock a size's idleness is counted on.
        self._clock = 0

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
            self._clock += 1
            use = self._use.get(size)
            if use is None:
                use = self._use[size] = _Use(self._clock)
            if block is None:
                block = _Block(f"{self._prefix}{next(self._counter)}", size)
                self._by_name[block.name] = block
                use.blocks += 1
            block.local = True
            owner = _Allocation(block.mapping)
            self._by_owner[id(owner)] = block
            given_back = self._give_back(size, use)
        weakref.finalize(owner, self._dropped.put, block.name)
        if given_back and self._retired is not None:
            self._retired(given_back)
        return np.frombuffer(owner, dtype=kind, count=count).reshape(shape)

    def _give_back(self, size: int, use: _Use) -> list[str]:
        """After an allocation of ``size``: the free blocks no turn of their size came back for (lock held)."""

        surplus: list[_Block] = []
        pool = self._free.get(size, [])
        use.widest = max(use.widest, self._clock - use.used_at)
        use.used_at = self._clock
        use.turn += 1
        use.least_free = min(use.least_free, len(pool)) if use.turn > 1 else len(pool)
        if use.turn >= use.blocks:
            # One whole turn of this size: what stayed free through all of it was not needed by it.
            surplus.extend(pool[: use.least_free])
            del pool[: use.least_free]
            use.turn = 0
        for other, idle in self._use.items():
            if other == size or not self._free.get(other):
                continue
            if self._clock - idle.used_at > idle.widest:
                # Unallocated for longer than it ever went before: it is not in use.
                surplus.extend(self._free[other])
                self._free[other] = []
                idle.used_at, idle.turn = self._clock, 0
        names = []
        for block in surplus:
            try:
                block.mapping.close()
            except BufferError:
                # A view outlived its allocation's owner: the block stays until it has gone.
                self._free.setdefault(block.size, []).append(block)
                continue
            del self._by_name[block.name]
            self._use[block.size].blocks -= 1
            names.append(block.name)
        return names

    @property
    def mapped_bytes(self) -> int:
        """The shared memory this arena holds mapped now, in use or free."""

        with self._lock:
            return sum(block.size for block in self._by_name.values())

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
        #: Live views per block, and the blocks whose last view has gone: unmapped at the next attach,
        #: so memory its owner gives back is not kept mapped here for ever.
        self._live: dict[str, int] = {}
        self._idle: queue.SimpleQueue[str] = queue.SimpleQueue()

    def attach(self, reference: dict[str, Any], *, acquired: Callable[[str], None] | None = None) -> np.ndarray:
        """A read-only array over one lend; dropping its last view releases the lend."""

        name = str(reference["block"])
        size = int(reference["size"])
        with self._lock:
            self._unmap_idle()
            self._live[name] = self._live.get(name, 0) + 1
            entry = self._mappings.get(name)
            if entry is None or entry[1] < size:
                mapping = mmap.mmap(-1, size, tagname=name, access=mmap.ACCESS_READ)
                entry = (mapping, size, _address(mapping))
                self._mappings[name] = entry
        view = ImmutableBuffer(entry[0])
        self._views[id(view)] = name
        weakref.finalize(view, self._dropped, id(view), name)
        if acquired is not None:
            acquired(name)
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

    def _unmap_idle(self) -> None:
        """Let go of the blocks no view of which is left (lock held)."""

        waiting = []
        while True:
            try:
                name = self._idle.get_nowait()
            except queue.Empty:
                break
            self._live[name] = self._live.get(name, 0) - 1
            if self._live[name] > 0:
                continue
            entry = self._mappings.get(name)
            try:
                if entry is not None:
                    entry[0].close()
            except BufferError:
                # The view's buffer is released a moment after it says it has gone: next time.
                self._live[name] = 1
                waiting.append(name)
                continue
            self._mappings.pop(name, None)
            self._live.pop(name, None)
        for name in waiting:
            self._idle.put(name)

    @property
    def mapped_blocks(self) -> int:
        """The blocks of other processes this one holds mapped now."""

        with self._lock:
            return len(self._mappings)

    def _dropped(self, identity: int, name: str) -> None:
        self._views.pop(identity, None)
        # A finalizer only enqueues (module docstring): the mapping goes at the next attach.
        self._idle.put(name)
        self._release(name)


__all__ = ["Arena", "Attachments", "GRANULARITY"]
