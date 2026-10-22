"""Internal ndarray ownership helpers for immutable data values."""

from __future__ import annotations

import numpy as np


def canonical_dtype(dtype) -> np.dtype:
    """Return the platform-independent little-endian spelling of ``dtype``."""

    result = np.dtype(dtype)
    if result.hasobject or result.fields is not None:
        raise TypeError("object and structured dtypes are not supported")
    if result.kind not in "biufc":
        raise TypeError(
            f"data values require bool or numeric dtype, got {result}"
        )
    if result.kind in "iu" and result.itemsize not in (1, 2, 4, 8):
        raise TypeError(f"unsupported integer dtype width: {result}")
    if result.kind == "f" and result.itemsize not in (2, 4, 8):
        raise TypeError(f"unsupported real dtype width: {result}")
    if result.kind == "c" and result.itemsize not in (8, 16):
        raise TypeError(f"unsupported complex dtype width: {result}")
    return result.newbyteorder("<")


class ImmutableBuffer:
    """Read-only memory whose writer has handed it over for good.

    zlc_plugin wraps a shared-memory block another process lent this way: the
    producer may not write it again until every view of this object is gone,
    so an ndarray over it is as immutable as one over ``bytes``.
    """

    __slots__ = ("_source", "__weakref__")

    def __init__(self, source: object) -> None:
        if not memoryview(source).readonly:
            raise ValueError("an ImmutableBuffer needs a read-only source")
        self._source = source

    def __buffer__(self, flags: int) -> memoryview:
        return memoryview(self._source)


def _immutable_owner(value: object) -> object | None:
    """Find a known immutable owner through readonly ndarray/memoryview links."""

    if type(value) is not np.ndarray or value.dtype.hasobject or value.dtype.fields:
        return None
    current: object = value
    seen: set[int] = set()
    while isinstance(current, (np.ndarray, memoryview)):
        identity = id(current)
        if identity in seen:
            return None
        seen.add(identity)
        if isinstance(current, np.ndarray):
            if current.flags.writeable or current.flags.owndata:
                return None
            current = current.base
        else:
            if not current.readonly:
                return None
            current = current.obj
    if isinstance(current, bytes) or type(current) is ImmutableBuffer:
        return current
    # Optional until the embedding adapter installs this exact module. This is
    # a concrete owner type, not a registry or a flags/attribute duck-type test.
    try:
        from zlc_python import OwnedBuffer
    except ModuleNotFoundError as error:
        if error.name == "zlc_python":
            return None
        raise
    return current if type(current) is OwnedBuffer else None


def is_intrinsically_immutable_array(value: object) -> bool:
    """Return whether a view rests on immutable bytes or a Rust OwnedBuffer.

    ``flags.writeable = False`` is not sufficient: an owning ndarray, or a view
    over one, can be made writable again.  By contrast, slices, transposes and
    broadcasts whose complete readonly base chain ends in either owner cannot acquire a
    writable buffer.  Recognising that ownership fact lets every downstream
    value object retain the exact view instead of copying the same camera frame
    at every semantic boundary.
    """

    return _immutable_owner(value) is not None


def immutable_array(values, *, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    """Return an intrinsically immutable ndarray, copying only mutable input.

    A mutable producer/builder buffer crosses this boundary through one compact
    ``bytes`` snapshot.  An already bytes/Rust-backed immutable slice, transpose or
    broadcast is returned unchanged: it cannot be made writable and therefore
    needs no second ownership copy.  Strides are part of the retained view;
    canonical persistence owners may materialise contiguous bytes when they
    actually serialize it.
    """

    source = np.asarray(values)
    source_dtype = canonical_dtype(source.dtype)
    if source_dtype != dtype:
        raise TypeError(f"values dtype {source.dtype} does not match schema dtype {dtype}")
    if source.shape != shape:
        raise ValueError(f"values shape {source.shape} does not match expected {shape}")
    if source.dtype == dtype and is_intrinsically_immutable_array(source):
        return source
    normalized = source.astype(dtype, copy=False)
    return np.ndarray(shape, dtype=dtype, buffer=normalized.tobytes(order="C"))


def immutable_bool_array(values, *, shape: tuple[int, ...]) -> np.ndarray:
    source = np.asarray(values)
    if source.dtype != np.dtype(bool):
        raise TypeError(f"validity mask dtype must be bool, got {source.dtype}")
    return immutable_array(source, dtype=np.dtype(bool), shape=shape)


def owning_bytes(source: np.ndarray) -> bytes | None:
    """The ``bytes`` object whose whole content ``source`` is, else None.

    A C-contiguous view whose base chain ends in ``bytes`` of exactly its
    size IS that object: a consumer that needs the plane as bytes takes the
    object instead of copying it out.  ``b"".join`` copies exact ``bytes``
    items with the GIL released; given any other buffer it holds the GIL for
    the whole copy, which for a cycle of camera frames stalls every other
    thread of the process for tens of milliseconds.
    """

    owner: object = source
    while isinstance(owner, (np.ndarray, memoryview)):
        owner = owner.base if isinstance(owner, np.ndarray) else owner.obj
    if source.flags.c_contiguous and isinstance(owner, bytes) and len(owner) == source.nbytes:
        return owner
    return None


def compact_immutable_array(source: np.ndarray) -> np.ndarray:
    """Retain this plane, not unrelated bytes backing a transient slice."""
    if owning_bytes(source) is not None:
        return source
    owner = _immutable_owner(source)
    if owner is not None and not isinstance(owner, bytes) and source.flags.c_contiguous:
        # _immutable_owner already established the exact Rust class. A dtype or
        # shape view may cover the same whole allocation; retain that view's
        # semantics, not the owner's original scalar spelling. Partial slices
        # still compact so they do not pin unrelated bytes in a larger frame.
        whole = np.asarray(owner)
        if (
            source.nbytes == whole.nbytes
            and source.__array_interface__["data"][0] == whole.__array_interface__["data"][0]
        ):
            return source
    return np.ndarray(source.shape, dtype=source.dtype, buffer=source.tobytes(order="C"))
