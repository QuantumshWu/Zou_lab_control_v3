"""Canonical phase mathematics, independent of device installation."""

from __future__ import annotations

import numpy as np

_TWO_PI = 2.0 * np.pi
_MAX_WRAPPED_PHASE = np.nextafter(np.float32(_TWO_PI), np.float32(0.0))


def _shape(shape_yx: object) -> tuple[int, int]:
    if (
        not isinstance(shape_yx, tuple)
        or len(shape_yx) != 2
        or any(type(value) is not int or value <= 0 for value in shape_yx)
    ):
        raise TypeError("SLM shape_yx must be a two-positive-integer tuple")
    return shape_yx


def canonical_phase(radians: object, shape_yx: tuple[int, int]) -> np.ndarray:
    """Return one immutable owned phase snapshot in wrapped float32 radians."""

    shape = _shape(shape_yx)
    source = np.asarray(radians)
    if source.shape != shape:
        raise ValueError(
            f"SLM phase shape {source.shape!r} differs from device shape {shape!r}"
        )
    if source.dtype.kind not in "iuf":
        raise TypeError("SLM phase must contain real numeric values")
    if (source.dtype == np.float32 and np.min(source) >= 0.0
            and np.max(source) <= _MAX_WRAPPED_PHASE):
        # Canonical float32 commands need no double-precision modulo. Taking
        # abs only changes -0 to +0, matching remainder's exact bytes.
        wrapped = np.abs(source)
    else:
        values = np.asarray(source, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("SLM phase must contain only finite values")
        wrapped = np.asarray(np.remainder(values, _TWO_PI), dtype=np.float32)
        # float32(2*pi) rounds above the mathematical upper bound. Clamp that
        # single rounding case so the public interval stays strictly [0, 2*pi).
        wrapped = np.minimum(wrapped, _MAX_WRAPPED_PHASE)
    # An ndarray backed by immutable bytes cannot be made writable again by a
    # caller, unlike an owning array with only its WRITEABLE flag cleared.
    return np.frombuffer(
        np.ascontiguousarray(wrapped).tobytes(),
        dtype=np.float32,
    ).reshape(shape)
