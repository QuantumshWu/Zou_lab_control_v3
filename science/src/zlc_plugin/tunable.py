"""Tunable devices read and tuned in a unit the operator picked (PROTOCOL §8).

A device's tunable quartet (``tunable_fields``/``tune``/``tunable_values``/
``settings_provenance``) speaks each knob in its own unit.  Device Control may
show a knob in another spelling (MHz for a hertz field, mW for a dBm one): a
native adapter owns conversions that depend on its state (an RF amplitude
into the channel's load), every other one goes through the shared registry,
an exact decimal shift within a unit family.  The SDK's ``fields`` and
``tune`` ops serve these; the plugin package re-exports them for its own use.
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

from zlc_data.units import DEFAULT_UNITS


def _converted(value: Any, source: str, target: str) -> float:
    """One device value between units, by the registry's own conversion.

    The same conversion the forms show a value through: exact decimal
    shifts within a unit family, and never a reading of typed text, where
    a bare prefix letter or a float32's short spelling would decide it.
    """

    return float(DEFAULT_UNITS.convert_decimal(value, source, target))


def refresh_tunable_fields(device: Any) -> tuple[Any, ...]:
    """Read current device fields once through the adapter's refresh entry."""
    refresh = getattr(device, "refresh_tunable_fields", None)
    return tuple(refresh() if callable(refresh) else device.tunable_fields())


def read_tunable_in_unit(device: Any, name: str, unit: str = "") -> Any:
    """Read a knob without changing it; blank requests its standing unit.

    Native adapters own load/waveform-dependent conversion. Ordinary adapters
    retain their declared unit and use the shared unit conversion at this edge.
    """
    native = getattr(device, "read_tunable_in_unit", None)
    if callable(native):
        return native(name, unit)
    field = next((item for item in device.tunable_fields() if item.metadata.name == name), None)
    if field is None:
        raise ValueError(f"device has no tunable field {name!r}")
    source = field.metadata.unit or "1"
    target = unit or source
    if target == source:
        return field

    def converted(value):
        return None if value is None else _converted(value, source, target)

    # A step is a difference: it is read in the other spelling only where the two
    # differ by a factor (Hz and MHz), not where a level is a power (dBm and mW).
    step = field.metadata.step
    if step is not None:
        scaled = converted(step)
        step = scaled if converted(0.0) == 0.0 and math.isclose(converted(2 * step), 2 * scaled, rel_tol=1e-9) else None
    return replace(
        field,
        metadata=replace(field.metadata, unit=target,
                         value_type="float" if field.metadata.value_type == "int" else field.metadata.value_type,
                         default=converted(field.metadata.default),
                         minimum=converted(field.metadata.minimum),
                         maximum=converted(field.metadata.maximum),
                         step=step),
        current=converted(field.current),
        device_limits=(None if field.device_limits is None
                       else tuple(converted(value) for value in field.device_limits)),
    )


def tune_in_unit(device: Any, name: str, value: Any, unit: str) -> Any:
    """Apply an authored quantity and return readback in the requested unit."""
    if not unit or value is None:
        return device.tune(name, value)
    native = getattr(device, "tune_in_unit", None)
    if callable(native):
        return native(name, value, unit)
    field = read_tunable_in_unit(device, name)
    target = field.metadata.unit or "1"
    source = unit or target
    if source == target:
        return device.tune(name, value)
    requested = _converted(value, source, target)
    if field.metadata.value_type == "int" and requested.is_integer():
        requested = int(requested)
    actual = device.tune(name, requested)
    return _converted(actual, target, source)


def convert_tunable_value(device: Any, name: str, value: Any, source_unit: str, target_unit: str) -> Any:
    """Read-only conversion of a desired quantity, never a device command."""
    if value is None or source_unit == target_unit:
        return value
    native = getattr(device, "convert_tunable_value", None)
    if callable(native):
        return native(name, value, source_unit, target_unit)
    if isinstance(value, (tuple, list)):
        return tuple(_converted(item, source_unit or "1", target_unit or "1") for item in value)
    return _converted(value, source_unit or "1", target_unit or "1")


__all__ = ["convert_tunable_value", "read_tunable_in_unit", "refresh_tunable_fields", "tune_in_unit"]
