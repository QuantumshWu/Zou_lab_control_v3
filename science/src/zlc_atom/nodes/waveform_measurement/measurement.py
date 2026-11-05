"""Ordered native waveform records, independent of read and display cadence.

Adapters own bounded receive queues. Runtime alone retains run data/history.
Each native record is published in order, preserving its samples and timestamp;
what one read takes goes out as one commit. With a software trigger an event is one
trace around an edge, with the shape of a record of a source triggered in hardware
(``trigger.py``), so a continuous and a finite run read alike.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache
from math import ceil
from time import monotonic, sleep, time_ns

import numpy as np

from zlc_data import COMPONENT, READOUT_EVENT, SAMPLE_TIME, AxisSpec, DomainSpec, OwnedSnapshot
from zlc_plugin import DatasetOutputDeclaration, LiveEvents
from zlc_plugin.figures import AxisRef

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
from zlc_atom.data import cell_axis_id, snapshot_from_array
from zlc_atom.devices.waveform.contract import (
    WaveformAcquisitionMode,
    WaveformCaptureTerminalRecord,
    WaveformOutput,
    WaveformRecord,
    WaveformSource,
    WaveformWorkingPoint,
    validate_waveform_outputs,
)
from zlc_atom.nodes._framework.descriptor import NodePreviewSpec

from .trigger import MINIMUM_SAMPLES, SLOPES, TraceAssembler, TriggerSettings, WaveformTrigger, channel_output


#: How often a capture comes back to see whether it has been asked to stop; the
#: read length is the cancel latency, and it belongs to the loop. A free-running
#: source is also read at most this often, so reads and commits follow this cadence
#: and not the sample rate: an event is published at most this late.
_CANCEL_RESPONSE_SECONDS = 0.05
_RECEIVE_BUFFER_SECONDS = 2.0
#: Sample bytes one read of a backlog may carry (a channel frame holds 16 MiB of JSON).
_BATCH_BYTES = 2 << 20
#: How long without a shot before a capture says it is waiting, and how often it
#: says what it saw meanwhile: a trigger that fires now and then does not flicker
#: the status between two words.
_WAITING_AFTER_SECONDS = 1.0

TRIGGER_OFF = "off"
_TRIGGER_ON = ("trigger_slope", SLOPES)

WAVEFORM_MEASUREMENT_SCHEMA = AuthoringSchema((
    AuthoringField("repeat", "int", "Records (0 = continuous)", 0, minimum=0,
                   description="how many records to keep; with a trigger, one record for each of that many triggers"),
    AuthoringField(
        "trigger_slope", "choice", "Slope", TRIGGER_OFF,
        choices=(AuthoringChoice(TRIGGER_OFF, "Off"), AuthoringChoice("rising", "Rising"),
                 AuthoringChoice("falling", "Falling")),
        segmented=True, group="Trigger",
        description="lock on a level crossing, for a slow free-running source: each crossing gives one record "
                    "with the crossing in its middle, on a time axis that is zero there",
    ),
    AuthoringField("trigger_channel", "text", "Channel", "", required=True, enabled_when=_TRIGGER_ON, group="Trigger"),
    AuthoringField("trigger_level", "float", "Level", 0.0, enabled_when=_TRIGGER_ON, group="Trigger"),
    AuthoringField("trigger_samples", "int", "Samples", 400, minimum=MINIMUM_SAMPLES, enabled_when=_TRIGGER_ON, group="Trigger",
                   description="samples of each record, half of them before the crossing"),
))


def waveform_outputs(
    outputs: tuple[WaveformOutput, ...],
) -> tuple[DatasetOutputDeclaration, ...]:
    """One signal per quantity a source carries, each with a shot history."""

    return tuple(
        DatasetOutputDeclaration(
            output.name, f"waveform.{output.name}", index_by_source=True
        )
        for output in validate_waveform_outputs(outputs)
    )


def waveform_preview(source: WaveformSource, *, trigger_channel: str | None = None, node: str | None = None) -> NodePreviewSpec:
    """The first quantity, as a rolling trace of shots or as the trace one shot is.

    Triggered, every shot is a trace, drawn one on the other. ``trigger_channel`` names
    the channel to watch (empty: not chosen yet), and ``node``, the producer its axes are
    named after, lets the view hold it.
    """

    outputs = waveform_outputs(source.outputs)
    if trigger_channel is None:
        return NodePreviewSpec(outputs[0], "rolling" if int(source.record_samples) == 1 else "curve")
    chosen = channel_output(trigger_channel, source.outputs)
    if chosen is None or node is None:
        return NodePreviewSpec(outputs[0] if chosen is None else next(o for o in outputs if o.name == chosen.name), "curve")
    held = AxisRef.cell_data(cell_axis_id(node, chosen.name, 0, COMPONENT).value).fate_field
    records = AxisRef.repeat(f"{node}.{chosen.name}.repeat").fate_field
    position = chosen.channel_labels.index(str(trigger_channel).partition("/")[2])
    return NodePreviewSpec(next(o for o in outputs if o.name == chosen.name), "curve",
                           semantic={held: ["scope-value", position], records: "group"})


@lru_cache(maxsize=64)
def _channel_axis(producer: str, signal: str, labels: tuple[str, ...]) -> AxisSpec:
    return AxisSpec(
        cell_axis_id(producer, signal, 0, COMPONENT),
        "channel",
        COMPONENT,
        len(labels),
        coordinate_labels=labels,
    )


@lru_cache(maxsize=64)
def _sample_axis(producer: str, signal: str, interval: float, samples: int, continuous: bool, first: int) -> AxisSpec:
    role = SAMPLE_TIME if continuous else READOUT_EVENT
    return AxisSpec(cell_axis_id(producer, signal, 1, role),
        "sample time" if continuous else "record time", role, int(samples),
        coordinates=tuple((first + index) * interval for index in range(int(samples))), unit="s")


def shot_snapshot(
    record: WaveformRecord,
    *,
    output: WaveformOutput,
    producer: str,
    generation: object,
    revision: int,
    sample_interval_seconds: float,
    continuous: bool = False,
    first_sample: int = 0,
) -> OwnedSnapshot:
    """Continuous samples are Point rows; triggered samples stay in the cell.

    ``first_sample`` is where the record's time axis starts, in samples: a software
    trigger's trace has its zero at the crossing.

    A non-finite sample is one the source could not obtain (a polled sensor
    that did not answer writes NaN): it is published INVALID through the
    Data validity mask, as other producers mark theirs (``np.isfinite``).
    The mask is always present, so a run's schema does not change on the
    first missing sample.
    """

    columns = output.columns
    first = columns[0]
    selection = (
        slice(first, first + len(columns))
        if columns == tuple(range(first, first + len(columns)))
        else list(columns)
    )
    picked = record.samples[:, selection]
    labels = output.channel_labels
    point_axes = ()
    if picked.shape[0] == 1:
        values = picked[0][None]
        cell_axes: tuple[AxisSpec, ...] = (_channel_axis(producer, output.name, labels),)
    elif continuous:
        values = picked[None]
        cell_axes = (_channel_axis(producer, output.name, labels),)
        point_axes = (_sample_axis(producer, output.name, sample_interval_seconds, picked.shape[0], True, first_sample),)
    else:
        values = picked.T[None]
        cell_axes = (
            _channel_axis(producer, output.name, labels),
            _sample_axis(producer, output.name, sample_interval_seconds, picked.shape[0], continuous, first_sample),
        )
    return snapshot_from_array(
        values,
        producer=producer,
        signal=output.name,
        point_axes=point_axes,
        cell_axes=cell_axes,
        value_unit=output.unit,
        generation=str(getattr(generation, "value", generation)),
        revision=int(revision),
        validity=np.isfinite(values),
    )


def _stopped_terminal(
    terminal: WaveformCaptureTerminalRecord,
) -> WaveformCaptureTerminalRecord:
    """The device's terminal, proving the capture stopped and joined.

    The source stops before the caller drains its already accepted records.
    """

    if not (terminal.source_stopped and terminal.joined):
        raise RuntimeError(
            "waveform terminal evidence is incomplete: the capture did not stop and join"
        )
    return terminal


def _working_point_snapshot(
    source: WaveformSource, point: WaveformWorkingPoint
) -> dict[str, object]:
    return {
        "acquisition_mode": str(point.acquisition_mode),
        "time_basis": point.time_basis,
        "sample_interval_seconds": float(point.sample_interval_seconds),
        "record_samples": int(source.record_samples),
        "outputs": [
            {
                "name": output.name,
                "unit": output.unit,
                "channels": list(output.channel_labels),
            }
            for output in source.outputs
        ],
        "settings": dict(point.settings),
    }


@dataclass(frozen=True)
class WaveformMeasurementRequest:
    """Native record count, independent of internal receive capacity."""

    sampler_key: str
    repeat: int
    trigger: TriggerSettings | None = None

    def __post_init__(self) -> None:
        key = str(self.sampler_key).strip()
        if not key:
            raise ValueError("sampler_key must be non-empty")
        if isinstance(self.repeat, bool) or int(self.repeat) != self.repeat or self.repeat < 0:
            raise ValueError("repeat must be a non-negative integer")
        object.__setattr__(self, "sampler_key", key)
        object.__setattr__(self, "repeat", int(self.repeat))


class FiniteCapture:
    """An armed finite capture: ``repeat`` shots, each one reading."""

    def __init__(
        self,
        node: "WaveformMeasurementNode",
        *,
        should_stop: Callable[[], bool] | None,
    ) -> None:
        self.node = node
        self.should_stop = should_stop
        self.closed = False
        self.terminal: WaveformCaptureTerminalRecord | None = None

    def collect(self, *, commit: Callable[..., object], progress: Callable[[int], object] = lambda shots: None) -> int:
        """Read every shot and publish it; answers how many shots were kept."""

        if self.closed:
            raise RuntimeError("finite capture is closed")
        node = self.node

        def publish(records: list[WaveformRecord]) -> None:
            if node.publish_finite(records, commit):
                progress(node.published)

        try:
            while node.published < node.repeat and (self.should_stop is None or not self.should_stop()):
                records = node.take_records(node.repeat - node.published)
                publish(records)
                if not records:
                    node.check_quiet(current=node.published, total=node.repeat)
            terminal = self.close()
            # Stop freezes the receive count before draining. No subsequent
            # read can pull a sample acquired after this capture stopped:
            # what is left is what the source still holds. A triggered source ran
            # on past the run's shots, and only a capture cut short wants its tail.
            if node.published < node.repeat:
                publish(node.drain_records(terminal.produced_count - node.records_read))
            self.terminal = replace(terminal, no_more_records=node.triggered or node.records_read == terminal.produced_count)
        except BaseException as error:
            # The source is stopped whatever happened, and a source that
            # failed raises again from its terminal.
            try:
                if not self.closed:
                    self.closed = True
                    node.sampler.finish_record_capture()
            except BaseException as cleanup_error:
                error.add_note(f"waveform cleanup also failed: {cleanup_error}")
            raise
        return node.published

    def close(self) -> WaveformCaptureTerminalRecord:
        if self.terminal is not None:
            return self.terminal
        if self.closed:
            raise RuntimeError("finite capture closed without terminal evidence")
        terminal = _stopped_terminal(self.node.sampler.finish_record_capture())
        self.closed = True
        self.terminal = terminal
        return terminal


class MonitorCapture:
    """A repeat-zero monitor: every shot replaces the last as the latest event."""

    def __init__(
        self,
        node: "WaveformMeasurementNode",
        *,
        commit_events: Callable[..., object],
    ) -> None:
        if not callable(commit_events):
            raise TypeError("a monitor requires commit_events")
        self.node = node
        self.closed = False
        self.terminal: WaveformCaptureTerminalRecord | None = None
        self._commit = commit_events

    def poll(self) -> int:
        """Publish every record the source holds, each its own event, in order, in one commit.

        One read takes them all and one commit publishes them: reaching the
        core is paid once per poll, not once per record.
        """
        if self.closed:
            raise RuntimeError("monitor capture is closed")
        records = self.node.take_records()
        published = self.node.publish_monitor(records, self._commit)
        self.node.check_quiet()
        return published

    def close(self, *, drain: bool = True) -> WaveformCaptureTerminalRecord:
        if self.terminal is not None:
            return self.terminal
        self.closed = True
        terminal = _stopped_terminal(self.node.sampler.finish_record_capture())
        if drain:
            self.node.publish_monitor(self.node.drain_records(terminal.produced_count - self.node.records_read), self._commit)
            terminal = replace(terminal, no_more_records=True)
        self.terminal = terminal
        return terminal


class WaveformMeasurementNode:
    """Commit each shot of a waveform source to its declared signals."""

    def __init__(
        self,
        *,
        sampler: WaveformSource,
        request: WaveformMeasurementRequest,
        producer: str,
    ) -> None:
        if not isinstance(sampler, WaveformSource):
            raise TypeError("sampler must implement WaveformSource")
        if not isinstance(request, WaveformMeasurementRequest):
            raise TypeError("request must be WaveformMeasurementRequest")
        self.sampler = sampler
        self._request = request
        self._outputs = waveform_outputs(sampler.outputs)
        self._quantities = {output.name: output for output in sampler.outputs}
        self.instance_id = str(producer).strip()
        if not self.instance_id:
            raise ValueError("producer must be non-empty")
        self._generation: object | None = None
        self._working_point: WaveformWorkingPoint | None = None
        self._run_record: dict[str, object] | None = None
        self._time_origin: float | None = None
        self._last_time: float | None = None
        self._next_ordinal = 0
        #: Events published so far: each event's ordinal and its place in a finite run.
        self.published = 0
        #: Records taken from the source so far; what a stopped source still holds is the rest.
        self.records_read = 0
        #: The source's facts, read once per run: the sampler is a device
        #: reached through the core, and every attribute read is a round trip.
        self._record_samples = 0
        self._source_timeout = 0.0
        self._batch_records = 1
        #: The run's software trigger, when the request has one.
        self._trigger: WaveformTrigger | None = None
        self._traces: TraceAssembler | None = None
        #: Per output: one event's schema, and a finite run's complete schema.
        self._schemas: dict[str, tuple[object, object]] = {}
        self._next_read = 0.0
        self._last_status = 0.0
        #: Silence is a fault only for a source that samples on its own clock:
        #: one with its own trigger waits for it, however long the pulse that
        #: fires it is stopped or being edited.  "No software trigger yet" is
        #: never silence, because the source's records still arrive.
        self._free_running = False
        self._last_delivery = 0.0
        self._last_shot = 0.0
        self._waiting = False
        self._report_status: Callable[..., object] | None = None

    @property
    def request(self) -> WaveformMeasurementRequest:
        return self._request

    @property
    def repeat(self) -> int:
        return self._request.repeat

    @property
    def triggered(self) -> bool:
        return self._trigger is not None

    @property
    def sampler_key(self) -> str:
        return self._request.sampler_key

    @property
    def generation(self) -> object:
        if self._generation is None:
            raise RuntimeError("waveform acquisition has no active generation")
        return self._generation

    @property
    def working_point(self) -> WaveformWorkingPoint:
        point = self._working_point
        if point is None:
            raise RuntimeError("waveform working point is not frozen")
        return point

    @property
    def run_record(self) -> dict[str, object]:
        record = self._run_record
        if record is None:
            raise RuntimeError("waveform run record was not frozen")
        return record

    # ------------------------------------------------------------ freezing
    def _configure(self) -> WaveformWorkingPoint:
        """Freeze how the source samples for this run."""

        point = self.sampler.working_point()
        if not isinstance(point, WaveformWorkingPoint):
            raise TypeError("waveform source working_point must return WaveformWorkingPoint")
        self._working_point = point
        self._record_samples = int(self.sampler.record_samples)
        self._source_timeout = float(self.sampler.timeout)
        self._time_origin = None
        self._last_time = None
        self._next_ordinal = 0
        self.published = self.records_read = 0
        self._schemas = {}
        self._free_running = point.acquisition_mode == WaveformAcquisitionMode.FREE_RUNNING.value
        self._trigger = self._traces = None
        parameters: dict[str, object] = {"repeat": self.repeat}
        settings = self._request.trigger
        if settings is not None:
            if not self._free_running:
                raise ValueError(
                    "this source has its own hardware trigger; the software trigger is for free-running sources")
            self._trigger = settings.resolve(self._quantities.values())
            parameters["trigger"] = {
                "channel": settings.channel, "level": settings.level, "slope": settings.slope,
                "samples": self._trigger.samples, "column": self._trigger.column,
            }
        self._run_record = {
            "parameters": parameters,
            "named_devices": {"sampler": self.sampler_key},
            "device_snapshots": {"sampler": _working_point_snapshot(self.sampler, point)},
            # Host call time, not a device-clock to UTC calibration.
            "started_at_ns": time_ns(),
        }
        return point

    def _check_stream(self, records: list[WaveformRecord]) -> None:
        """The source's records follow one another, with this run's sample count and a clock that advances."""

        for record in records:
            if record.source_ordinal != self._next_ordinal:
                raise RuntimeError(f"waveform record gap: expected {self._next_ordinal}, got {record.source_ordinal}")
            if record.samples.shape[0] != self._record_samples:
                raise ValueError("waveform record sample count changed during capture")
            self._next_ordinal += 1

    def _event_schemas(self, samples: int, columns: int) -> dict[str, tuple[object, object]]:
        """One event's schema per output, the same object for the whole run, and for a
        finite run its complete schema, built once rather than from an O(repeat) code
        vector on every record."""

        if not self._schemas:
            blank = WaveformRecord(np.zeros((samples, columns), dtype=np.float32), 0, 0.0, 1)
            for declaration in self._outputs:
                schema = shot_snapshot(
                    blank, output=self._quantities[declaration.name], producer=self.instance_id, generation=self.generation,
                    revision=0, sample_interval_seconds=self.working_point.sample_interval_seconds,
                    continuous=self._continuous, first_sample=0 if self._traces is None else -self._traces.before,
                ).block.schema
                complete = None
                if self.repeat:
                    (repeat_axis,) = schema.repeat_domain.axes
                    complete = replace(schema, repeat_domain=DomainSpec(
                        (self.repeat,), (replace(repeat_axis, size=self.repeat),), (tuple(range(self.repeat)),)))
                self._schemas[declaration.name] = (schema, complete)
        return self._schemas

    @property
    def _continuous(self) -> bool:
        """A record's samples are rows of the source's own stream, not a trace of its own."""

        return self._free_running and self._trigger is None

    def _as_events(self, records: list[WaveformRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(samples ``(events, samples, columns)``, source times, host receipts) of these records.

        Triggered, an event is a trace, and these records may complete any number of them.
        """

        times = np.fromiter((record.time_seconds for record in records), np.float64, len(records))
        received = np.fromiter((record.host_received_at_ns for record in records), np.int64, len(records))
        samples = np.stack([record.samples for record in records])
        if self._trigger is None:
            return samples, times, received
        count = self._record_samples
        offsets = self.working_point.sample_interval_seconds * np.arange(count)
        traces = self._traces.push(samples.reshape(-1, samples.shape[2]), (times[:, None] + offsets).ravel(), np.repeat(received, count))
        if traces.edges:
            self._triggered()
        return traces.samples, traces.times, traces.received

    def _commit(self, samples: np.ndarray, times: np.ndarray, received: np.ndarray, commit: Callable[..., object],
                *, finite: bool = False) -> int:
        """Publish these events as one commit, the same events however they are cut."""

        count = samples.shape[0]
        if not count:
            return 0
        if self._last_time is not None and times[0] <= self._last_time or np.any(np.diff(times) <= 0.0):
            raise RuntimeError("waveform record clock did not advance")
        if self._time_origin is None:
            self._time_origin = float(times[0])
        self._last_time = float(times[-1])
        shot_times = times - self._time_origin
        ordinals = np.arange(self.published, self.published + count, dtype=np.uint64)
        schemas = self._event_schemas(samples.shape[1], samples.shape[2])
        outputs: dict[str, LiveEvents] = {}
        for declaration in self._outputs:
            picked = samples[:, :, list(self._quantities[declaration.name].columns)]
            # Each row in its event's physical shape (see shot_snapshot).
            values = picked[:, None] if self._continuous or samples.shape[1] == 1 else picked.transpose(0, 2, 1)[:, None, None]
            valid = np.isfinite(values)
            (schema, complete) = schemas[declaration.name]
            outputs[declaration.name] = LiveEvents(declaration, schema, values, True if valid.all() else valid, complete if finite else None)
        basis, instance = self.working_point.time_basis, self.instance_id
        records = [{"record_timing": {instance: {str(ordinal): {
            "record_time_seconds": shot, "source_time_seconds": source, "time_basis": basis, "host_received_at_ns": host}}}}
            for ordinal, shot, source, host in zip(ordinals.tolist(), shot_times.tolist(), times.tolist(), received.tolist())]
        commit(outputs, ordinals=ordinals, shot_times=None if finite else shot_times,
               origins=np.stack((ordinals, np.zeros(count, dtype=np.uint64)), axis=1) if finite else None,
               records=records)
        self.published += count
        return count

    def publish_monitor(self, records: list[WaveformRecord], commit: Callable[..., object]) -> int:
        """Everything these records complete goes out at once."""

        if not records:
            return 0
        self._check_stream(records)
        samples, times, received = self._as_events(records)
        return self._commit(samples, times, received, commit)

    def publish_finite(self, records: list[WaveformRecord], commit: Callable[..., object]) -> int:
        """Up to the run's rest of ``repeat`` events."""

        if not records:
            return 0
        self._check_stream(records)
        samples, times, received = self._as_events(records)
        last = self.repeat - self.published
        return self._commit(samples[:last], times[:last], received[:last], commit, finite=True)

    @property
    def silence_seconds(self) -> float:
        """How long a capture waits for the next record before calling the source silent."""

        return self._record_samples * self.working_point.sample_interval_seconds + self._source_timeout

    def take_records(self, limit: int | None = None) -> list[WaveformRecord]:
        """Every record the source holds (up to ``limit``), waiting a cancel slice for one.

        One read per batch, not per record: a capture that falls behind
        catches up at the cost of publishing alone, because reaching the
        source (a round trip through the core) is paid once for the backlog.
        A free-running source is read once a slice, so that it costs reads by
        the slice and not by the sample.
        """

        if self._free_running:
            wait = self._next_read - monotonic()
            if wait > 0.0:
                sleep(wait)
            self._next_read = monotonic() + _CANCEL_RESPONSE_SECONDS
        count = self._batch_records if limit is None or self._trigger is not None else max(1, min(int(limit), self._batch_records))
        records = list(self.sampler.read_records(count, timeout=_CANCEL_RESPONSE_SECONDS, exact=False))
        if records:
            self.records_read += len(records)
            self._last_delivery = monotonic()
            if self._trigger is None:
                self._triggered()
        return records

    def _triggered(self) -> None:
        """A shot came: whatever was being waited for has happened."""

        self._last_shot = monotonic()
        if self._waiting:
            self._waiting = False
            if self._report_status is not None:
                self._report_status("Triggered")

    def _waiting_status(self) -> str:
        trigger = self._trigger
        seen = None if trigger is None else self._traces.take_range()
        if seen is None:
            return "Waiting for trigger"
        return (f"Waiting for trigger: {trigger.name} ranged {seen[0]:.4g} to {seen[1]:.4g} {trigger.unit}, "
                f"level {trigger.level:.6g}")

    def check_quiet(self, *, current: int | None = None, total: int | None = None) -> None:
        """A dead source is a fault; a quiet trigger is a status, saying what it sees."""

        now = monotonic()
        if self._free_running and now - self._last_delivery >= self.silence_seconds:
            raise TimeoutError("waveform source stopped delivering records")
        waits = not self._free_running or self._trigger is not None
        if waits and now - self._last_shot >= _WAITING_AFTER_SECONDS and (
                not self._waiting or now - self._last_status >= _WAITING_AFTER_SECONDS):
            self._last_status = now
            self._waiting = True
            if self._report_status is not None:
                self._report_status(self._waiting_status(), current=current, total=total)
    def drain_records(self, count: int) -> list[WaveformRecord]:
        """Exactly the ``count`` records a stopped source still holds, a batch per read."""

        records: list[WaveformRecord] = []
        while len(records) < count:
            records.extend(self.sampler.read_records(
                min(count - len(records), self._batch_records), timeout=0.0, exact=True))
        self.records_read += len(records)
        return records

    def _arm(self) -> None:
        # Capacity absorbs stalls; it never sets sample/record/plot cadence.
        capacity = max(2, ceil(_RECEIVE_BUFFER_SECONDS / (
            self._record_samples * self.working_point.sample_interval_seconds)))
        # A batch is the whole buffer, bounded so that its samples stay well
        # inside one channel frame however long the records are.
        columns = 1 + max(column for output in self._quantities.values() for column in output.columns)
        self._batch_records = max(1, min(capacity, _BATCH_BYTES // (4 * columns * self._record_samples)))
        # Triggered, the source streams until the run has its traces: how many of
        # its records that takes depends on the signal.
        self.sampler.arm(None if self._trigger is not None else self.repeat or None, buffer_record_count=capacity)
        self._run_record["acquisition"] = {"buffer_record_count": capacity}
        # Hardware may quantize the requested rate at arm (notably DAQ).
        # Freeze its accepted clock before publishing even the first record.
        self._working_point = self.sampler.working_point()
        self._run_record["device_snapshots"]["sampler"] = _working_point_snapshot(
            self.sampler, self._working_point
        )
        if self._trigger is not None:
            self._traces = TraceAssembler(self._trigger, self._working_point.sample_interval_seconds)
        self._last_delivery = self._last_shot = monotonic()
        self._next_read = 0.0
        self._waiting = False

    # ------------------------------------------------------------- capture
    def _armed(self, *, after_arm: Callable[[], object] | None = None) -> None:
        try:
            self._configure()
            self._arm()
            if after_arm is not None:
                after_arm()
        except BaseException as error:
            # A sampler still holding the last capture's failure raises it
            # from finish: noted, never in place of this one.
            try:
                self.sampler.finish_record_capture()
            except BaseException as cleanup:
                error.add_note(f"sampler cleanup also failed: {cleanup}")
            raise

    def execute(self, context: object) -> dict[str, object]:
        """The run: arm, report ready, publish every record through the run's context."""

        # The run's instance names the signals and the evidence, not this node's type.
        self.instance_id = context.instance_id
        self._generation = context.generation
        self._report_status = context.report_progress
        if self.repeat == 0:
            self._armed(after_arm=context.record_devices)
            capture = MonitorCapture(self, commit_events=context.commit_events)
            try:
                context.set_run_record(self.run_record)
                context.report_ready()
                while not context.cancel_requested():
                    capture.poll()
            except BaseException as error:
                try:
                    capture.close(drain=False)
                except BaseException as cleanup_error:
                    error.add_note(f"waveform cleanup also failed: {cleanup_error}")
                raise
            capture.close()
            return {"shots": self.published}
        self._armed(after_arm=context.record_devices)
        capture = FiniteCapture(self, should_stop=context.cancel_requested)
        try:
            context.set_run_record(self.run_record)
            context.report_ready()
        except BaseException:
            capture.close()
            raise

        def progress(shots: int) -> None:
            context.report_progress("Capturing", current=shots, total=int(self.repeat))

        return {"shots": capture.collect(commit=context.commit_events, progress=progress)}


__all__ = [
    "TRIGGER_OFF",
    "WAVEFORM_MEASUREMENT_SCHEMA",
    "FiniteCapture",
    "MonitorCapture",
    "WaveformMeasurementNode",
    "WaveformMeasurementRequest",
    "shot_snapshot",
    "waveform_outputs",
    "waveform_preview",
]
