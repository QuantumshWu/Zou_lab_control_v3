"""The core's pulse board as a node run drives it, through catalogue commands.

The pulse stack is native (docs/PULSE_V3.md): a plugin never decodes or
compiles a pulse. A pulse workspace resource resolves, in the core, to its
description (``pulse_describe``), which is what a node's build and queries
receive; a run plays a document through ``pulse_run``, waits through
``pulse_wait_done`` and makes the board safe through ``pulse_stop``. These
are the same catalogue entries the UI and MCP use, so a run's claims apply
to a node exactly as to any other client.

A run's ``pulse_stop`` answers once the board is SAFE and raises when it would
not go safe (v2 ``sequencer.safe()``). The documents a run plays are pinned at
their first fire: a later fire of a file changed meanwhile is refused with
``PULSE_CHANGED`` in its words, so the run's record names one pulse.  A run
that ends with a FIRE of its unreported has the board made SAFE by the core.

The description a pulse resource resolves to::

    {"path": "<absolute document path>", "name": "<file stem>",
     "api_fields": [{"field_id": "duration:short", "label": "short", "kind": "duration",
                     "unit": "s", "value": 0.005}, ...],     # in the order the pulse plays them
     "periods": [{"period_id": "p3", "name": "short", "label": "short"}, ...]}

``kind`` is ``duration``, ``dac``, ``bracket_count`` or ``delay``; ``unit`` is the
field's own (a DAC field's is ``value``).

What a scan reads besides (v2 ``scan_columns_for`` / ``api_parameter_columns_for``
/ the empty-bracket rule): every ``api_fields`` entry and every entry of
``scan_fields`` -- the scan slots the template carries, in slot order, each with
the ``value`` it holds in the pulse -- also states ``is_dac`` and
``limit_lo``/``limit_hi`` (the widest the field may be given), in the column's
own unit (a DAC column's is the signed code, 0 = 0 V); ``unscannable`` maps an
API field the pulse offers no plan to the sentence that says why (an empty
bracket's repeat count).

``pulse_run`` may carry ``scan: {"fields": [field_id, ...], "rows": [[value, ...], ...]}``:
the board plays those scan slots as its table (rows in the fields' own units,
in that field order; slots not named play their authored value), each row
``run_repeats`` times and the table ``scan_repeats`` times. It quantizes the
table to what it plays (durations on the clock grid; two requested rows that
play the same are refused) and answers ``scan: {"fields", "rows"}`` -- the rows
as played, in the same order and units. More fields than the board has slots
are refused. Without ``scan`` no slot streams: each plays its authored value.

The ``pulse_run`` receipt::

    {"command_id": 7, "api_values": {"duration:short": 0.005, ...},   # in the fields' own units
     "applied": {...},          # the board's archive snapshot of this FIRE (v2 sequencer_archive_snapshot)
     "timing": {"clock_hz": 50e6, "digest": "<program digest>",
                "duration_ticks": 123456,               # one run of the program
                "earliest_lane_delay_ticks": -3,        # min(0, every lane's delay)
                "periods": [{"period_id": "p3", "name": "short", "label": "short",
                             "starts": [40000], "duration_ticks": 250000}, ...],
                "rising_edges": {"<digital lane>": [40000, 900000], ...}}}

``timing`` is the program as compiled (Config filled) for this FIRE, over one
run with every loop body replayed at most twice: a period inside a loop shows
at least two ``starts`` without its loop being expanded, and so do the rising
edges of each digital lane that is not a clock output (v2 ``frame_visits`` /
``trigger_edge_ticks`` with ``bracket_bodies=2``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

#: The contract family of a pulse document chosen as a workspace resource.
PULSE_RESOURCE_PREFIX = "zlc.pulse/"
#: The DONE report statuses a run can end with.
STATUS_DONE, STATUS_ERROR, STATUS_UNDERFLOW = "done", "error", "underflow"


def pulse_description(resource: object) -> Mapping[str, Any]:
    """The core's description of a chosen pulse resource, checked for its shape."""

    if not isinstance(resource, Mapping) or not isinstance(resource.get("path"), str) \
            or not isinstance(resource.get("api_fields"), (list, tuple)):
        raise TypeError("a pulse resource must be the core's pulse description")
    return resource


def duration_fields(description: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """The pulse's API duration fields, in the order the pulse plays them."""

    return tuple(field for field in description["api_fields"] if field.get("kind") == "duration")


class PulseBoard:
    """One run's use of the board: play a document, wait for it, make it safe."""

    def __init__(self, context: object) -> None:
        core = getattr(context, "core", None)
        if not callable(core):
            raise TypeError("driving the pulse board needs a run context with core()")
        self._core = core

    def run(
        self,
        document: str | Path,
        *,
        api_values: Mapping[str, tuple[float, str]] | None = None,
        run_repeats: int = 1,
        scan_repeats: int = 1,
        scan: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Resolve, compile, load and fire ``document`` with these API values,
        each ``(number, unit)``; answers the receipt (module docstring):
        ``command_id``, the resolved ``api_values`` in their fields' own
        units, ``applied`` (the board's execution record for the archive)
        and ``timing`` (what the compiled program plays when).  ``scan`` is a
        table for the board to play, ``{fields, rows}`` (module docstring)."""

        params: dict[str, Any] = dict(
            document=str(document),
            api_values={key: [float(value), str(unit)] for key, (value, unit) in (api_values or {}).items()},
            run_repeats=int(run_repeats), scan_repeats=int(scan_repeats))
        if scan is not None:
            params["scan"] = {"fields": [str(field) for field in scan["fields"]],
                              "rows": [[float(value) for value in row] for row in scan["rows"]]}
        return self._core("pulse_run", **params)

    def load(
        self,
        document: str | Path,
        *,
        api_values: Mapping[str, tuple[float, str]] | None = None,
        scan: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """LOAD ``document`` without firing (v2 ``arm_sequencer``), so the FIRE
        later is the FIRE alone; the receipt is :meth:`run`'s without
        ``command_id``, its ``applied`` and ``timing`` those of the LOAD."""

        params: dict[str, Any] = dict(
            document=str(document), stage="load",
            api_values={key: [float(value), str(unit)] for key, (value, unit) in (api_values or {}).items()})
        if scan is not None:
            params["scan"] = {"fields": [str(field) for field in scan["fields"]],
                              "rows": [[float(value) for value in row] for row in scan["rows"]]}
        return self._core("pulse_run", **params)

    def fire_loaded(self, document: str | Path, *, run_repeats: int = 1, scan_repeats: int = 1) -> Mapping[str, Any]:
        """FIRE what :meth:`load` left on the board (refused if it no longer
        holds that program and table); the receipt keeps LOAD's named ``scan``
        and ``api_values`` and adds the FIRE's ``command_id``, ``applied`` and ``timing``."""

        return self._core("pulse_run", document=str(document), stage="fire",
                          run_repeats=int(run_repeats), scan_repeats=int(scan_repeats))

    def wait_done(self, command_id: int, timeout_seconds: float) -> Mapping[str, Any] | None:
        """That FIRE's completion report, or None when it has not completed in time."""

        report = self._core("pulse_wait_done", command_id=int(command_id), timeout_seconds=float(timeout_seconds))
        return None if not report or not report.get("done") else report

    def wait_report(self, command_id: int, should_stop: Callable[[], bool]) -> Mapping[str, Any]:
        """Wait for that FIRE's report and stay stoppable meanwhile.

        The board is asked in slices and Stop is read between them, so a
        Stop leaves within one slice and the caller's cleanup makes the board
        safe. A FIRE that can no longer report (the board was made safe, its
        report was already taken) is refused by ``pulse_wait_done`` itself:
        nothing here waits unconditionally.
        """

        while True:
            if should_stop():
                raise RuntimeError(CANCELLED)
            report = self.wait_done(command_id, _REPORT_SLICE_SECONDS)
            if report is not None:
                return report

    def stop(self) -> Mapping[str, Any]:
        """SAFE now; answers once the board is safe, raises when it is not."""

        return self._core("pulse_stop")


#: What a run that ends because it was stopped says, wherever it noticed.
CANCELLED = "the run was cancelled"
#: In the words of a fire refused because the run's pulse file changed (the core's PULSE_CHANGED).
PULSE_CHANGED = "changed during this run"
#: How long one wait for a report blocks before Stop is read again.
_REPORT_SLICE_SECONDS = 0.1


def check_cancelled(context: object) -> None:
    """Leave now if the operator has pressed Stop."""

    if context.cancel_requested():
        raise RuntimeError(CANCELLED)


__all__ = [
    "CANCELLED",
    "PULSE_CHANGED",
    "PULSE_RESOURCE_PREFIX",
    "PulseBoard",
    "STATUS_DONE",
    "STATUS_ERROR",
    "STATUS_UNDERFLOW",
    "check_cancelled",
    "duration_fields",
    "pulse_description",
]
