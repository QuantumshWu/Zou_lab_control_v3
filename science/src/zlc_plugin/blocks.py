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
size that has gone unallocated for longer than it ever went between two of its
own allocations is not in use at all. That second rule is read on the clock,
not on the arena's allocations: when the work ends nothing allocates, and the
blocks it leaves free are given back once their size has been idle for longer
than its own longest gap -- the moment the arena's watcher is armed for, so
idle memory goes back while the process is idle, not at its next allocation.
A block given back is unmapped here and said to the core (``retired``), which
unmaps it too; a process that only borrowed it lets its mapping go with its
last view.

Finalizers only enqueue (and wake the watcher): they may run inside any lock
(garbage collection), so the arena and the release sender drain their queues
themselves.
"""

from __future__ import annotations

import itertools
import mmap
import os
import queue
import threading
import time
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

    def __init__(self, clock: float) -> None:
        #: The blocks of this size that exist, free or not.
        self.blocks = 0
        #: Its own allocations since the last look, and the fewest free blocks it had after any of them.
        self.turn = 0
        self.least_free = 0
        #: When this size was last allocated (or its idle blocks given back), and the longest time
        #: that ever passed between two of its own allocations.
        self.used_at = clock
        self.widest = 0.0


class Arena:
    """Blocks this process writes; sizes are 64 KiB multiples and recycled exactly.

    ``retired`` is told the names of blocks given back to the system (module docstring);
    ``clock`` is the one idleness is read on.
    """

    def __init__(self, session: str, retired: Callable[[list[str]], None] | None = None,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self._prefix = f"Local\\zlc.{session}.{os.getpid()}."
        self._counter = itertools.count()
        self._lock = threading.Lock()
        self._free: dict[int, list[_Block]] = {}
        self._by_name: dict[str, _Block] = {}
        self._by_owner: dict[int, _Block] = {}
        self._dropped: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._retired = retired
        self._use: dict[int, _Use] = {}
        self._clock = clock
        #: Wakes the watcher: a block came free, or a local view of one went.
        self._wake = threading.Event()
        self._watcher: threading.Thread | None = None
        weakref.finalize(self, self._wake.set)

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
            now = self._clock()
            use = self._use.get(size)
            if use is None:
                use = self._use[size] = _Use(now)
            if block is None:
                block = _Block(f"{self._prefix}{next(self._counter)}", size)
                self._by_name[block.name] = block
                use.blocks += 1
            block.local = True
            owner = _Allocation(block.mapping)
            self._by_owner[id(owner)] = block
            given_back = self._give_back(size, use, now)
        weakref.finalize(owner, _dropped, self._dropped, self._wake, block.name)
        self._tell(given_back)
        return np.frombuffer(owner, dtype=kind, count=count).reshape(shape)

    def _tell(self, names: list[str]) -> None:
        if names and self._retired is not None:
            self._retired(names)

    def _give_back(self, size: int, use: _Use, now: float) -> list[str]:
        """After an allocation of ``size``: the free blocks no turn of their size came back for,
        and every idle size's (lock held)."""

        surplus: list[_Block] = []
        pool = self._free.get(size, [])
        use.widest = max(use.widest, now - use.used_at)
        use.used_at = now
        use.turn += 1
        use.least_free = min(use.least_free, len(pool)) if use.turn > 1 else len(pool)
        if use.turn >= use.blocks:
            # One whole turn of this size: what stayed free through all of it was not needed by it.
            surplus.extend(pool[: use.least_free])
            del pool[: use.least_free]
            use.turn = 0
        return self._unmap(surplus) + self._give_back_idle(now)

    def _give_back_idle(self, now: float) -> list[str]:
        """The free blocks of every size not in use: unallocated for longer than it ever went
        between two of its own allocations (lock held)."""

        surplus: list[_Block] = []
        for size, idle in self._use.items():
            if self._free.get(size) and now - idle.used_at > idle.widest:
                surplus.extend(self._free[size])
                self._free[size] = []
                idle.used_at, idle.turn = now, 0
        return self._unmap(surplus)

    def _idle_deadline(self) -> float | None:
        """When the next size holding free blocks becomes idle (lock held); None while none holds any."""

        due = [idle.used_at + idle.widest for size, idle in self._use.items() if self._free.get(size)]
        return min(due) if due else None

    def _watch(self) -> float | None:
        """One look of the watcher: what has gone idle is given back; how long until the next look."""

        with self._lock:
            self._drain()
            now = self._clock()
            names = self._give_back_idle(now)
            due = self._idle_deadline()
        self._tell(names)
        if due is None or (due <= now and not names):
            # Nothing free, or only a block a view still pins (``_unmap``): until a block comes free.
            return None
        return max(0.0, due - now)

    def _unmap(self, surplus: list[_Block]) -> list[str]:
        """Unmap the given-back blocks; their names (lock held)."""

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
            # Free now: the watcher looks when its size would be idle (lock held; only wakes it).
            if self._watcher is None:
                self._watcher = threading.Thread(target=_watcher, args=(weakref.ref(self), self._wake),
                                                 name="zlc-arena-idle", daemon=True)
                self._watcher.start()
            self._wake.set()


def _dropped(dropped: queue.SimpleQueue, wake: threading.Event, name: str) -> None:
    """An allocation's last local view went (a finalizer: only enqueue and wake)."""

    dropped.put(name)
    wake.set()


def _watcher(arena: "weakref.ref[Arena]", wake: threading.Event) -> None:
    """Gives an arena's idle blocks back while nothing allocates; ends with the arena.

    It sleeps until the next size holding free blocks would be idle, or until a block
    comes free; it holds the arena only while it looks."""

    while True:
        owner = arena()
        if owner is None:
            return
        timeout = owner._watch()
        del owner
        wake.wait(timeout)
        wake.clear()


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
