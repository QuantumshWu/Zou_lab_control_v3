"""The device-independent surface of one motion axis.

A motion axis -- a linear or rotation stage behind its controller -- is, to
this system, its TUNABLE SURFACE plus three actions.  As for an RF source the
capability Protocol IS the quartet every tunable device speaks
(``tunable_fields`` / ``tune`` / ``tunable_values`` /
``settings_provenance``), so a scan axis, Device Control and the device-axis
executor need no second, motion-only vocabulary:

* ``position`` is the knob: an ABSOLUTE position in the axis's own unit
  (``mm`` for a linear stage, ``deg`` for a rotation stage), bounded by the
  travel the controller itself enforces.  ``tune`` returns only once the
  axis has stopped, and returns the position it MEASURES there, never the
  target: setpoint and readback are different facts a scan records apart;
* ``state`` is a reading (``derived``: shown, never edited), the
  controller's own word for what it is doing, with any fault it reports;
* the actions are commands: ``home`` (find the reference), ``stop``.

The rules that must not fork per driver:

* HOMING IS AN EXPLICIT ACTION.  A move asked of an axis that has no
  reference is refused by name and moves nothing.  Homing drives the stage
  to a switch somewhere in its travel -- through whatever stands in the
  beam path -- so it is never a side effect of asking for a position, as
  opening a device never changes its outputs (``rf/contract.py``: opening
  is read-only);
* STOP IS NEVER QUEUED BEHIND A MOVE.  ``stop`` and Stop all's
  ``make_safe`` are declared ``concurrent`` (PLUGIN_PROTOCOL §4): they run
  beside the instance's lane while ``tune`` holds it for a move, reach the
  controller between two status polls, and the move then reports where it
  stopped -- as a failure, because it did not arrive;
* a value is checked against the travel within binary noise (``within``), as
  a scan's coordinates carry it, and anything else is refused before the
  controller is asked.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Protocol, runtime_checkable

from zlc_atom.authoring import TunableField, TuneRefused

POSITION_FIELD = "position"
VELOCITY_FIELD = "velocity"
STATE_FIELD = "state"

#: The units an axis position is stated in, as its stage is linear or rotary.
AXIS_UNITS = ("mm", "deg")

#: Stop all's output for an axis: its motion (``make_safe`` stops it).
MOTION_OUTPUT = "motion"


@runtime_checkable
class MotionAxis(Protocol):
    """What an installed ``motion.axis`` capability answers."""

    def tunable_fields(self) -> tuple[TunableField, ...]: ...

    def tune(self, name: str, value: Any) -> Any: ...

    def tunable_values(self) -> Mapping[str, Any]: ...

    def settings_provenance(self) -> Mapping[str, Any]: ...

    def home(self) -> str: ...

    def stop(self) -> str: ...

    def make_safe(self, output: str) -> None: ...

    def close(self) -> None: ...


#: How far past an edge binary noise may carry a value authored on it,
#: relative to the magnitudes compared: doubles hold 15-16 significant digits.
_EDGE_NOISE = 1e-12


def within(value: object, low: float, high: float, *, name: str, unit: str) -> float:
    """``value`` as a float inside [low, high], a rounding past an edge being that edge.

    Anything that is not a finite number inside the travel is TuneRefused:
    nothing was sent, so the axis is where it was.
    """

    if isinstance(value, bool):
        raise TuneRefused(f"{name} takes a number in {unit}, not {value!r}")
    try:
        requested = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise TuneRefused(f"{name} takes a number in {unit}, not {value!r}") from None
    if not math.isfinite(requested):
        raise TuneRefused(f"{name} must be finite")
    slack = _EDGE_NOISE * max(abs(requested), abs(low), abs(high), 1.0)
    if requested < low - slack or requested > high + slack:
        raise TuneRefused(f"{name} {requested:g} {unit} lies outside the travel [{low:g}, {high:g}] {unit}")
    return min(max(requested, low), high)


__all__ = [
    "AXIS_UNITS",
    "MOTION_OUTPUT",
    "MotionAxis",
    "POSITION_FIELD",
    "STATE_FIELD",
    "TuneRefused",
    "VELOCITY_FIELD",
    "within",
]
