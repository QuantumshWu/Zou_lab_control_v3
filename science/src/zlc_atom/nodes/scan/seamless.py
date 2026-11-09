"""Host-advanced axes around an optional seamless hardware scan table.

Manual, device and API axes advance between fires. Planned board axes advance
inside a fire; unplanned slots keep their Pulse field values. Without board
axes, each host point plays the fixed Pulse with no table. Run repeats supplies
shots_per_point without rewriting PulseBracket or adding artificial scan coordinates.

The board is the core's (_framework.pulse): every fire carries this point's
API values and a native scan handle. The core builds and quantizes the table,
retains its effective coordinates and full execution snapshot, and answers a
compact receipt. The source signal arrives as this run's core-consumed exact
input, in played order.
Inner coordinates are the quantized values played; host coordinates keep
their nominal settings, with hardware facts retained in the native snapshot.

Acquisition preparation happens once at Scan Start. Device writes use their
actual readback; manual changes are controlled by the operator. No implicit
settling delay is added to either operation.
The native scan session places source publications in scan/repeat order. Device knobs return to their original values and units on
completion, Stop or failure through the existing cleanup path.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import monotonic

from zlc_data.units import DEFAULT_UNITS, figure_padded
from zlc_plugin.tunable import snapshot as device_snapshot

from zlc_atom.nodes._framework.pulse import STATUS_DONE, PulseBoard, check_cancelled, pulse_description

from .dataset import SCAN_OUTPUT
from .devices import ScanDeviceKnobs
from .plan import (
    API_PARAM_FAMILY,
    DEVICE_PARAM_FAMILY,
    MANUAL_PARAM_FAMILY,
    PULSE_PARAM_FAMILY,
    ScanPlan,
    ScanPort,
    api_overrides_from_authored,
    api_scan_ports_for,
    bind_plan,
    device_port_parts,
    hardware_scan_ports_for,
    port_label,
    resolve_api_overrides,
    split_outer_axes,
    scan_row_count,
    host_advanced_port,
    unscannable_api_ports,
)

#: The one operator-input kind this engine raises, and it asks the one
#: thing only a person can do: move a knob no machine here reaches.
MANUAL_AXIS_REQUEST = "manual-axis"

#: How long the source may stay silent once the board has reported the fire
#: done, beyond three shot periods: a frame's readout and transfer and the
#: processors between the camera and the scanned signal.  After that the
#: frames will not come -- a trigger the camera missed -- and waiting longer
#: is a scan that hangs with nothing on screen saying why.
_SOURCE_GRACE_SECONDS = 5.0
#: How often an idle wait for the source asks the board whether it is done;
#: a remote board answers each ask over its own connection.
_BOARD_POLL_SECONDS = 0.5
#: How long one wait for the source blocks before Stop and the board are looked at.
_INPUT_SLICE_SECONDS = 0.1


def estimate_scan_time(
    description: Mapping[str, object],
    plan: ScanPlan,
    *,
    api_values: object = "",
    repeats: int = 1,
    shots_per_point: int = 1,
    board_estimate: object = None,
) -> dict[str, object]:
    """Project Scan's workload and scheduled Pulse time, without device or file I/O.

    The counts are the plan's: shots per point and sweep, fires. The Pulse
    seconds need the program compiled for every API timing combination, which
    is the board's: ``board_estimate(**request)`` is the core's
    ``pulse_estimate`` (``{document, api_points, scan}``), compiled for the
    attached board with the Config it holds. Without one, or when the board
    cannot answer, the seconds are not known. v2's sum: each API timing
    point's table ticks, times the other host points and the shots per point
    over the clock; the output delay bound per fire beside it.
    """

    if not isinstance(plan, ScanPlan):
        raise TypeError("estimate_scan_time requires a ScanPlan")
    for name, value in (("repeats", repeats), ("shots_per_point", shots_per_point)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    description = pulse_description(description)
    resolve_api_overrides(description, api_overrides_from_authored(api_values))
    pulse_axes = tuple(axis for axis in plan.axes if axis.port.startswith((API_PARAM_FAMILY, PULSE_PARAM_FAMILY)))
    ports = bind_plan(ScanPlan(pulse_axes), (*hardware_scan_ports_for(description), *api_scan_ports_for(description)),
                      withheld=unscannable_api_ports(description)) if pulse_axes else ()
    outer_axes, board_axes = split_outer_axes(plan)
    outer_count = scan_row_count(outer_axes)
    estimate = _pulse_seconds(description, ports, outer_axes, board_axes, outer_count, repeats, shots_per_point, board_estimate)
    return {
        "points_per_sweep": plan.point_count,
        "full_sweeps": repeats,
        "shots_per_point": shots_per_point,
        "shots_per_sweep": plan.point_count * shots_per_point,
        "total_shots": plan.point_count * shots_per_point * repeats,
        "fire_count": outer_count * repeats if outer_axes else 1,
        "excluded_overheads": [
            "acquisition preparation and readout", "device writes and communication",
            "compilation/upload", "processing/fit/publication", "manual confirmation",
        ],
        "assumptions": [
            "Resolved Pulse and supplied Config values remain unchanged until/during execution.",
            "Pulse seconds include Bracket, shots per point and full sweeps; not wall-clock ETA.",
            "Output delay is an additional per-Fire upper bound, not a measured settling time.",
        ],
        **estimate,
    }


#: v2's preview budget: at most this many API timing combinations are compiled.
_ESTIMATE_PROGRAMS = 256


def _pulse_seconds(description, ports, outer_axes, board_axes, outer_count, repeats, shots_per_point, board_estimate):
    """The Pulse seconds of the plan (v2 estimate_scan_time), the board compiling."""

    unknown = {"status": "unknown", "first_sweep_pulse_seconds": None, "total_pulse_seconds": None,
               "first_sweep_output_delay_bound_seconds": None, "total_output_delay_bound_seconds": None}
    if board_estimate is None:
        return {**unknown, "reason": "Pulse seconds need the program compiled for this plan, which is the board's."}
    if any(axis.coordinate_of for axis in (*outer_axes, *board_axes)):
        return {**unknown, "reason": "Linked scan timing requires the paired table; point and shot counts are exact."}
    by_port = {port.port: port for port in ports}
    fields = {str(field["field_id"]): field for field in (*description["api_fields"], *description.get("scan_fields", ()))}
    api_axes = tuple(axis for axis in outer_axes if axis.port.startswith(API_PARAM_FAMILY))
    timing_axes = tuple(axis for axis in api_axes if fields[axis.port[len(API_PARAM_FAMILY):]].get("kind") != "dac")
    timing_count = math.prod(len(axis.values) for axis in timing_axes)
    if timing_count > _ESTIMATE_PROGRAMS:
        return {**unknown, "reason": f"{timing_count} API timing combinations exceed the {_ESTIMATE_PROGRAMS}-program preview budget."}

    def native(axis, value):
        field = fields[axis.port.split(":", 1)[1]]
        return [axis.native_value(by_port[axis.port], value), str(field.get("unit") or "")]

    fixed = {axis.port[len(API_PARAM_FAMILY):]: native(axis, axis.values[0]) for axis in api_axes}
    api_points = []
    for point in itertools.product(*(axis.values for axis in timing_axes)):
        values = dict(fixed)
        values.update({axis.port[len(API_PARAM_FAMILY):]: native(axis, value) for axis, value in zip(timing_axes, point, strict=True)})
        api_points.append(values)
    request = {"document": str(description["path"]), "api_points": api_points}
    if board_axes:
        request["scan"] = {"fields": [axis.port[len(PULSE_PARAM_FAMILY):] for axis in board_axes],
                           "columns": [[axis.native_value(by_port[axis.port], value) for value in axis.values] for axis in board_axes]}
    try:
        answer = board_estimate(**request)
    except Exception as error:  # noqa: BLE001 -- the board's refusal is the estimate's reason
        return {**unknown, "reason": f"The board did not estimate this plan: {error}"}
    clock = float(answer["clock_hz"])
    multiplier = outer_count // max(timing_count, 1)
    sweep_seconds = int(answer["table_ticks"]) * multiplier * shots_per_point / clock
    delay_seconds = int(answer["delay_ticks"]) * multiplier / clock
    return {"status": "estimated", "first_sweep_pulse_seconds": sweep_seconds, "total_pulse_seconds": sweep_seconds * repeats,
            "first_sweep_output_delay_bound_seconds": delay_seconds,
            "total_output_delay_bound_seconds": delay_seconds * (repeats if outer_axes else 1)}


class SeamlessScanMeasurement:
    """Fire the plan's table per host point through the board; take what plays."""

    def __init__(
        self,
        *,
        pulse: Mapping[str, object],
        plan: ScanPlan,
        ports: tuple[ScanPort, ...],
        source_signal: str,
        tunables: Mapping[str, object] | None = None,
        repeats: int,
        shots_per_point: int,
        acquisition_logic: str = "",
        api_values: Mapping[str, tuple[float, str]] | None = None,
        unapplied_api_values: Mapping[str, float] | None = None,
        sequencer_key: str = "",
    ) -> None:
        self.pulse = pulse_description(pulse)
        #: The file the operator chose; the pulse is named by it wherever a
        #: record names the pulse.
        self.pulse_path = Path(str(self.pulse["path"]))
        #: This run's own API values over what the pulse carries; a plan
        #: axis on one of them overrides it point by point.
        self.api_values = dict(api_values or {})
        #: API values the draft keeps for names this pulse does not declare:
        #: applied to nothing, and named in the run record as such.
        self.unapplied_api_values = dict(unapplied_api_values or {})
        #: The sequencer instance the run claims, as its record names it.
        self.sequencer_key = str(sequencer_key)
        self.source_signal = str(source_signal)
        if not self.source_signal.strip():
            raise ValueError("a seamless scan needs the signal it collects")
        self.acquisition_logic = str(acquisition_logic).strip()
        self.plan = plan
        #: The host's axes and the board's, split once: who moves an axis
        #: decides where its loop lives, and that never changes for the
        #: life of one measurement.  Manual, device and API axes are all
        #: host-advanced -- the run pauses between fires either way; what
        #: differs is only whether a hand, a ``tune()`` call or new API
        #: values in the program move the knob.
        self.outer_axes, self.board_axes = split_outer_axes(plan)
        self.tunables = dict(tunables or {})
        bound = tuple(ports)
        self.ports = bound
        if len(bound) != sum(1 for axis in plan.axes if not axis.port.startswith(MANUAL_PARAM_FAMILY)):
            raise ValueError("one bound port per device and board axis; manual axes bind to nobody")
        by_port = {port.port: port for port in bound}
        self.outer_ports = tuple(
            None if axis.port.startswith(MANUAL_PARAM_FAMILY) else by_port[axis.port] for axis in self.outer_axes
        )
        self.board_ports = tuple(by_port[axis.port] for axis in self.board_axes)
        for axis in self.outer_axes:
            if not axis.port.startswith(DEVICE_PARAM_FAMILY):
                continue
            key, _field = device_port_parts(axis.port)
            if key not in self.tunables:
                raise ValueError(f"device axis {axis.port!r} has no installed device {key!r} behind it")
        self.repeats = int(repeats)
        if self.repeats < 1:
            raise ValueError("repeats must be at least 1")
        self.shots_per_point = int(shots_per_point)
        if self.shots_per_point < 1:
            raise ValueError("shots_per_point must be at least 1")

    @property
    def dataset_output_declarations(self):
        return (SCAN_OUTPUT,)

    def resolved_device_claims(self) -> dict[str, dict[str, object]]:
        """Fields this plan will tune: the core claims them for the run, so a
        Device Control tune of the same field is refused while the scan owns it."""

        return device_claims(self.plan)

    def _api_values_for(self, outer_row: Sequence[float]) -> dict[str, tuple[float, str]]:
        """The point's API values over the run's own: each walked API
        parameter in its declared unit -- the port's, like every axis."""

        values = dict(self.api_values)
        for position, (axis, port) in enumerate(zip(self.outer_axes, self.outer_ports)):
            if axis.port.startswith(API_PARAM_FAMILY):
                values[axis.port[len(API_PARAM_FAMILY):]] = (axis.native_value(port, outer_row[position]), port.unit)
        return values

    def _apply_device_setting(self, context: object, knobs: ScanDeviceKnobs, *,
                              changed: Sequence[tuple[str, float, int, int]],
                              readbacks: dict[str, dict[str, object]]) -> None:
        """Move the installed knobs this row names, through the one owner
        of the device-axis law.  The board has completed its previous run
        before any device is moved."""

        for port, value, index, points in changed:
            check_cancelled(context)
            axis = next(axis for axis in self.outer_axes if axis.port == port)
            bound = next(bound for bound in self.ports if bound.port == port)
            context.report_progress(f"Setting {bound.label} ({figure_padded(index + 1, points)}/{points})")
            unit = axis.unit or bound.unit
            actual = knobs.move(port, value, unit)
            key, field = device_port_parts(port)
            previous = readbacks.get(key, {})
            # Keyed by the device's id; its Device Manager name beside it, for whoever shows it.
            readbacks[key] = {
                "role": getattr(self.tunables[key], "role", "") or key,
                "values": {**previous.get("values", {}), field: actual},
                "units": {**previous.get("units", {}), field: unit},
                "provenance": dict(self.tunables[key].settings_provenance()),
            }

    def _ask_for_setting(self, context: object, *, changed: Sequence[tuple[str, float, int, int]]) -> None:
        """Stop for the hand, and only for what the hand has to move."""

        for port, value, index, points in changed:
            name = port_label(port)
            context.report_progress(f"Waiting for {name}")
            context.request_operator_input(
                MANUAL_AXIS_REQUEST,
                title=f"Set {name}",
                message=f"Set {name} to {value:g}, then continue.",
                payload={"axis": name, "value": float(value), "point": index + 1, "points": points},
            )

    def _play_table(
        self,
        context: object,
        board: PulseBoard,
        *,
        outer_row: Sequence[float],
        state: dict[str, object],
        inner_count: int,
        sweeps: int,
        row_offset: int,
        scan_repeat_base: int,
        progress_base: int,
        progress_total: int,
    ) -> None:
        """Play one segment: fire the template with this point's values and
        place every shot the source delivers.

        The first fire's receipt completes the run: the rows the board plays
        are the scan's coordinates, and its archive snapshot the run record's.
        Every later fire must play the Config the first one did: each reads
        the saved Config again, so a Save between two points would change the
        pulse under a run record that names the first values.
        """

        shots = self.shots_per_point
        readouts = sweeps * inner_count * shots
        try:
            # Native Pulse resolves this handle directly; no inner rows cross Python.
            native = state["native"]
            scan_plan = native.pulse_segment(row_offset // inner_count)
            native.arm(row_offset=row_offset, scan_repeat_base=scan_repeat_base, sweeps=sweeps,
                       quiet_seconds=0.0 if state.get("completed_segment") else 0.05,
                       actual_devices=state["readbacks"],
                       new_source=not state.get("configured") and bool(self.acquisition_logic))
            check_cancelled(context)
            if not state.get("configured"):
                # The first point is configured; restoration values stay in ScanDeviceKnobs. Each
                # device is recorded under its id, named, in the units its operator keeps.
                state["tunables"] = {
                    key: {**device_snapshot(device, getattr(device, "units", None)), "role": getattr(device, "role", "") or key}
                    for key, device in sorted(self.tunables.items())}
                board.load(self.pulse_path, api_values=self._api_values_for(outer_row), scan_plan=scan_plan)
                if self.acquisition_logic:
                    context.report_progress("Preparing the acquisition")
                    context.acquire(self.acquisition_logic, completion="ready")
                context.record_devices()
                receipt = board.fire_loaded(self.pulse_path, run_repeats=shots, scan_repeats=sweeps, scan_plan=scan_plan)
            else:
                receipt = board.run(self.pulse_path, api_values=self._api_values_for(outer_row),
                                    run_repeats=shots, scan_repeats=sweeps, scan_plan=scan_plan)
            command_id = int(receipt["command_id"])
            applied = dict(receipt["applied"])
            config = applied.get("config")
            if not state.get("configured"):
                self._begin(context, state, receipt)
            elif config != state["config"]:
                raise RuntimeError(
                    "the Config file changed during the scan; its later points "
                    "would play values its run record does not name"
                )
            context.report_progress(
                f"Scanning point {figure_padded(progress_base + 1, progress_total)}/{progress_total}; shots",
                current=progress_base * shots,
                total=progress_total * shots,
            )
            # Readouts are assigned by arrival, so the board's own report is
            # the one witness that no more triggers are coming.  Once it is
            # in, a source that falls silent gets a bounded grace instead of
            # being waited for forever; one still delivering is behind, not
            # stuck, so the grace runs from its latest value.
            report = None
            report_seen = next_ask = 0.0
            last_delivery = monotonic()
            delivered = 0
            while delivered < readouts:
                check_cancelled(context)
                answer = native.collect(timeout_seconds=_INPUT_SLICE_SECONDS, max_events=1024)
                delivered = int(answer["segment_written"])
                if int(answer["collected"]):
                    last_delivery = monotonic()
                    context.report_progress(
                        f"Scanning point {figure_padded(progress_base + (delivered - 1) // shots + 1, progress_total)}"
                        f"/{progress_total}; shots",
                        current=progress_base * shots + delivered,
                        total=progress_total * shots,
                    )
                    continue
                now = monotonic()
                if report is None:
                    if now < next_ask:
                        continue
                    next_ask = now + _BOARD_POLL_SECONDS
                    report = board.wait_done(command_id, 0.0)
                    if report is None:
                        continue
                    if report["status"] != STATUS_DONE:
                        raise RuntimeError(f"the pulse failed: {report.get('fault') or report['status']}")
                    report_seen = now
                grace = _SOURCE_GRACE_SECONDS + 3.0 * float(report["elapsed_seconds"]) / readouts
                if now - max(report_seen, last_delivery) >= grace:
                    raise RuntimeError(
                        f"the board played {readouts} shots and the source "
                        f"delivered {delivered}: a missed camera trigger "
                        "leaves every later frame without its shot"
                    )
            if report is None:
                report = board.wait_report(command_id, context.cancel_requested)
                if report["status"] != STATUS_DONE:
                    raise RuntimeError(f"the pulse failed: {report.get('fault') or report['status']}")
            state["completed_segment"] = True
        except BaseException:
            board.stop()
            raise

    def _begin(self, context: object, state: dict[str, object], receipt: Mapping[str, object]) -> None:
        """The first fire's facts: the played table, the dataset it fills and the run record."""

        native = state["native"]
        table = receipt.get("scan")
        if (not isinstance(table, Mapping) or table.get("handle") != native.handle
                or table.get("row_count") != native.inner_count):
            raise RuntimeError("Scan cannot start: the sequencer FIRE receipt is missing its native scan identity or point count")
        applied = dict(receipt["applied"])
        state["native"].configure(
            record=self.run_record(sequencer=applied, tunables=state["tunables"]),
            output=SCAN_OUTPUT.name,
        )
        state.update(configured=True, config=applied.get("config"))

    def _native_spec(self) -> dict[str, object]:
        ports = {port.port: port for port in self.ports}
        return {
            "axes": [
                {"port": axis.port, "identity": port_label(axis.port),
                 "label": ports[axis.port].label if axis.port in ports else port_label(axis.port),
                 "unit": axis.unit or (ports[axis.port].unit if axis.port in ports else ""),
                 "values": list(axis.values), "coordinate_of": axis.coordinate_of,
                 "host_advanced": host_advanced_port(axis.port)}
                for axis in self.plan.axes
            ],
            "scan_repeats": self.repeats, "run_repeats": self.shots_per_point,
        }

    def execute(self, context: object):
        """Play the whole plan and return the dataset it filled.

        A plan the board owns entirely plays from ONE fire.  A plan carrying a
        host axis -- manual, device or API -- plays one fire per host point
        instead, and ``repeats`` walks the whole plan again rather than
        lengthening a fire -- the same sentence either way, spent where the
        plan leaves room for it.
        """

        board = PulseBoard(context)
        native = context.scan(self._native_spec())
        if native.ports != tuple(axis.port for axis in self.plan.axes):
            raise RuntimeError("the native scan axis order differs from the bound plan")
        # Both board table expansion and receipt retention stay native. The
        # plugin pages only host points for device/API/operator decisions.
        state: dict[str, object] = {"native": native, "readbacks": {}}
        inner_count = native.inner_count
        segment = dict(state=state, inner_count=inner_count,
                       progress_total=self.repeats * native.point_count)
        knobs = ScanDeviceKnobs(self.tunables)
        # The board is released per fire, inside ``_play_table``; what the
        # whole plan owes the bench is the knobs back where they were,
        # however it ended.
        try:
            board.stop()
            check_cancelled(context)
            if not self.outer_axes:
                self._play_table(context, board, outer_row=(), sweeps=self.repeats, row_offset=0,
                                 scan_repeat_base=0, progress_base=0, **segment)
            else:
                standing: tuple[float, ...] | None = None
                done = 0
                for sweep in range(self.repeats):
                    for index, outer_row in enumerate(native.iter_rows("outer")):
                        changed = tuple(
                            (axis.port, outer_row[position], index, native.outer_count)
                            for position, axis in enumerate(self.outer_axes)
                            if standing is None or standing[position] != outer_row[position]
                        )
                        # The hand first, then the machine: an operator
                        # asked to turn a thumbscrew should not find the
                        # bench half reconfigured under them while the
                        # dialog is open.  An API value moves with the fire.
                        self._ask_for_setting(context, changed=tuple(
                            entry for entry in changed if entry[0].startswith(MANUAL_PARAM_FAMILY)))
                        self._apply_device_setting(context, knobs, changed=tuple(
                            entry for entry in changed if entry[0].startswith(DEVICE_PARAM_FAMILY)),
                            readbacks=state["readbacks"])
                        for port, _value, point, points in changed:
                            if port.startswith(API_PARAM_FAMILY):
                                check_cancelled(context)
                                context.report_progress(
                                    f"Setting {port_label(port)} ({figure_padded(point + 1, points)}/{points})")
                        standing = outer_row
                        self._play_table(context, board, outer_row=outer_row, sweeps=1,
                                         row_offset=index * inner_count, scan_repeat_base=sweep,
                                         progress_base=done, **segment)
                        done += inner_count
            check_cancelled(context)
        except BaseException as error:
            # The scan's own failure stays the error; a knob that would not
            # go back is told beside it.
            try:
                knobs.restore()
            except BaseException as failure:
                error.add_note(
                    "restoring the scanned device fields also reported: "
                    f"{type(failure).__name__}: {failure}"
                )
            raise
        knobs.restore()
        return {"points": native.point_count, "shots": native.written}

    def run_record(self, *, sequencer: Mapping[str, object],
                   tunables: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
        """Factored authored plan and device facts; native configure records effective axes."""

        axes = self.plan.to_tree()["axes"]
        named_devices = {"sequencer": self.sequencer_key}
        for axis in self.outer_axes:
            if axis.port.startswith(DEVICE_PARAM_FAMILY):
                key, _field = device_port_parts(axis.port)
                named_devices[f"tunable:{key}"] = key
        return {
            "source_signal": self.source_signal,
            "named_devices": named_devices,
            "device_snapshots": {
                "sequencer": dict(sequencer),
                **{f"tunable:{key}": dict(snapshot) for key, snapshot in tunables.items()
                   if f"tunable:{key}" in named_devices},
            },
            "pulse": {"name": self.pulse_path.stem, "path": str(self.pulse_path)},
            "plan": {"axes": axes},
            "scan_shape": list(self.plan.shape),
            "scan_repeats": self.repeats,
            "run_repeats": self.shots_per_point,
            "acquisition_logic": self.acquisition_logic or None,
            **({"unapplied_api_values": dict(self.unapplied_api_values)} if self.unapplied_api_values else {}),
        }


def device_claims(plan: ScanPlan) -> dict[str, dict[str, object]]:
    """The device fields a plan tunes, by the argument the run receives each
    device as (``tunable:<key>``)."""

    selected: dict[str, list[str]] = {}
    for axis in plan.axes:
        if axis.port.startswith(DEVICE_PARAM_FAMILY):
            key, field = device_port_parts(axis.port)
            selected.setdefault(key, []).append(field)
    return {f"tunable:{key}": {"instance": key, "fields": fields} for key, fields in selected.items()}


__all__ = ["MANUAL_AXIS_REQUEST", "SeamlessScanMeasurement", "device_claims", "estimate_scan_time"]
