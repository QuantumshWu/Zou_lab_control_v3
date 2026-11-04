"""Ordered native waveform records, independent of read and display cadence.

Adapters own bounded receive queues. Runtime alone retains run data/history.
Each native record is published in order, preserving its samples and timestamp.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache
from math import ceil
from time import monotonic, sleep, time_ns

import numpy as np

from zlc_data import COMPONENT, READOUT_EVENT, SAMPLE_TIME, AxisSpec, DatasetSchema, DomainSpec, OwnedSnapshot
from zlc_plugin import DatasetOutputDeclaration, LiveDatasetOutput

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

from .trigger import SLOPES, TriggerDetector, TriggeredRecord, TriggerSettings, WaveformTrigger


#: How often a capture comes back to see whether it has been asked to stop;
#: the read length is the cancel latency, and it belongs to the loop.
_CANCEL_RESPONSE_SECONDS = 0.05
_RECEIVE_BUFFER_SECONDS = 2.0
#: Sample bytes one read of a backlog may carry (a channel frame holds 16 MiB of JSON).
_BATCH_BYTES = 2 << 20
#: How long without a shot before a capture says it is waiting: a trigger that
#: fires now and then does not flicker the status between two words.
_WAITING_AFTER_SECONDS = 1.0

TRIGGER_OFF = "off"
_TRIGGER_ON = ("trigger_slope", SLOPES)
TRIGGER_FIELDS = ("trigger_slope", "trigger_channel", "trigger_level", "trigger_record_samples", "trigger_position")

WAVEFORM_MEASUREMENT_SCHEMA = AuthoringSchema((
    AuthoringField("repeat", "int", "Records (0 = continuous)", 0, minimum=0),
    AuthoringField(
        "trigger_slope", "choice", "Trigger", TRIGGER_OFF,
        choices=(AuthoringChoice(TRIGGER_OFF, "Off"), AuthoringChoice("rising", "Rising"),
                 AuthoringChoice("falling", "Falling")),
        segmented=True, group="Trigger",
        description="lock on a level crossing, for a slow free-running source: each trigger is one record",
    ),
    AuthoringField("trigger_channel", "text", "Channel", "", required=True, enabled_when=_TRIGGER_ON, group="Trigger"),
    AuthoringField("trigger_level", "float", "Level", 0.0, enabled_when=_TRIGGER_ON, group="Trigger",
                   description="in the channel's unit"),
    AuthoringField("trigger_record_samples", "int", "Record samples", 50, minimum=2,
                   enabled_when=_TRIGGER_ON, group="Trigger"),
    AuthoringField("trigger_position", "float", "Pre-trigger (%)", 20.0, minimum=0.0, maximum=100.0, step=5.0,
                   enabled_when=_TRIGGER_ON, group="Trigger",
                   description="the share of a record before the trigger, at negative times"),
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


def waveform_preview(source: WaveformSource, *, trigger_channel: str | None = None) -> NodePreviewSpec:
    """The first quantity, as a rolling trace of shots or as the trace one shot is.

    Triggered, every shot is a trace, and the one to watch is the quantity the
    trigger locks: ``trigger_channel`` names it (empty: not chosen yet).
    """

    outputs = waveform_outputs(source.outputs)
    if trigger_channel is None:
        return NodePreviewSpec(outputs[0], "rolling" if int(source.record_samples) == 1 else "curve")
    name = str(trigger_channel).partition("/")[0]
    return NodePreviewSpec(next((output for output in outputs if output.name == name), outputs[0]), "curve")


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
def _sample_axis(producer: str, signal: str, interval: float, samples: int, continuous: bool, origin: int = 0) -> AxisSpec:
    role = SAMPLE_TIME if continuous else READOUT_EVENT
    return AxisSpec(cell_axis_id(producer, signal, 1, role),
        "sample time" if continuous else "record time", role, int(samples),
        coordinates=tuple((index - origin) * interval for index in range(int(samples))), unit="s")


def shot_snapshot(
    record: WaveformRecord,
    *,
    output: WaveformOutput,
    producer: str,
    generation: object,
    revision: int,
    sample_interval_seconds: float,
    continuous: bool = False,
    trigger_sample: int = 0,
) -> OwnedSnapshot:
    """Continuous samples are Point rows; triggered samples stay in the cell.

    ``trigger_sample`` is the time origin of a record cut by the software
    trigger: the samples before it, its pre-trigger share, read at negative times.

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
        point_axes = (_sample_axis(producer, output.name, sample_interval_seconds, picked.shape[0], True),)
    else:
        values = picked.T[None]
        cell_axes = (
            _channel_axis(producer, output.name, labels),
            _sample_axis(producer, output.name, sample_interval_seconds, picked.shape[0], continuous, trigger_sample),
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
        self.completed_shots = 0
        self.terminal: WaveformCaptureTerminalRecord | None = None
        #: Records already taken from the source, not yet handed out.
        self._taken: deque[WaveformRecord] = deque()

    def collect(
        self,
        *,
        commit_shot: Callable[[WaveformRecord, int], None],
    ) -> int:
        """Read every shot and publish it; answers how many shots were kept."""

        if self.closed:
            raise RuntimeError("finite capture is closed")
        try:
            for index in range(self.node.repeat):
                record = self.next_shot()
                if record is None:
                    break
                commit_shot(record, index)
            terminal = self.close()
            # Stop freezes the receive count before draining. No subsequent
            # read can pull a sample acquired after this capture stopped:
            # what is left is what was taken and what the source still holds.
            if self.node.triggered:
                # The source ran on; only a capture cut short still wants its tail.
                if self.completed_shots < self.node.repeat:
                    self._taken.extend(self.node.drain_triggered(terminal.produced_count))
                left = min(self.node.repeat - self.completed_shots, len(self._taken))
            else:
                left = min(self.node.repeat, terminal.produced_count) - self.completed_shots
                self._taken.extend(self.node.drain_records(left - len(self._taken)))
            for index in range(self.completed_shots, self.completed_shots + left):
                commit_shot(self._taken.popleft(), index)
                self.completed_shots += 1
            self.terminal = replace(terminal, no_more_records=(
                self.node.triggered or self.completed_shots == terminal.produced_count))
        except BaseException as error:
            # The source is stopped whatever happened, and a source that
            # failed raises again from its terminal.
            try:
                if not self.closed:
                    self.closed = True
                    self.node.sampler.finish_record_capture()
            except BaseException as cleanup_error:
                error.add_note(f"waveform cleanup also failed: {cleanup_error}")
            raise
        return self.completed_shots

    def next_shot(self) -> WaveformRecord | None:
        """Wait for the next native record; never substitute a newer one."""
        if self.closed:
            raise RuntimeError("finite capture is closed")
        node = self.node
        while self.should_stop is None or not self.should_stop():
            if not self._taken:
                # Everything the source holds, up to this capture's rest, in one read.
                self._taken.extend(node.take_records(node.repeat - self.completed_shots))
            if self._taken:
                self.completed_shots += 1
                return self._taken.popleft()
            node.check_quiet(current=self.completed_shots, total=node.repeat)
        return None

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
        commit_live: Callable[..., object],
    ) -> None:
        if not callable(commit_live):
            raise TypeError("a monitor requires commit_live")
        self.node = node
        self.closed = False
        self.terminal: WaveformCaptureTerminalRecord | None = None
        self._revision = 0
        self._commit_live = commit_live

    @property
    def revision(self) -> int:
        return self._revision

    def poll(self) -> int:
        """Publish every record the source holds, each its own event, in order.

        One read takes them all: the cost of reaching the source is paid
        once per poll, not once per record, so a backlog shrinks instead of
        growing whenever publishing a record is cheaper than the record period.
        """
        if self.closed:
            raise RuntimeError("monitor capture is closed")
        records = self.node.take_records()
        if not records:
            self.node.check_quiet()
            return 0
        for record in records:
            self._publish(record)
        return len(records)

    def _publish(self, record: WaveformRecord) -> None:
        self._revision += 1
        self._commit_live(self.node._shot_outputs(record, revision=self._revision))

    def close(self, *, drain: bool = True) -> WaveformCaptureTerminalRecord:
        if self.terminal is not None:
            return self.terminal
        self.closed = True
        terminal = _stopped_terminal(self.node.sampler.finish_record_capture())
        if drain:
            remaining = (self.node.drain_triggered(terminal.produced_count) if self.node.triggered
                         else self.node.drain_records(terminal.produced_count - self._revision))
            for record in remaining:
                self._publish(record)
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
        #: Per output: the event schema a finite run's canonical schema was
        #: built from, and that canonical schema.  The event schema is one
        #: cached object for the whole run, so the canonical one is built once
        #: rather than from an O(repeat) code vector on every record.
        self._canonical: dict[str, tuple[DatasetSchema, DatasetSchema]] = {}
        #: The source's facts, read once per run: the sampler is a device
        #: reached through the core, and every attribute read is a round trip.
        self._record_samples = 0
        self._source_timeout = 0.0
        self._batch_records = 1
        #: The run's software trigger, when the request has one.
        self._trigger: WaveformTrigger | None = None
        self._detector: TriggerDetector | None = None
        self._source_records_read = 0
        self._next_read = 0.0
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
    def broken_records(self) -> int:
        """Triggered records a gap in the source's stream cut short."""

        return 0 if self._detector is None else self._detector.broken_records

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
        self._canonical = {}
        self._free_running = point.acquisition_mode == WaveformAcquisitionMode.FREE_RUNNING.value
        self._trigger = self._detector = None
        self._source_records_read = 0
        parameters: dict[str, object] = {"repeat": self.repeat}
        settings = self._request.trigger
        if settings is not None:
            if not self._free_running:
                raise ValueError(
                    "this source has its own hardware trigger; the software trigger is for free-running sources")
            self._trigger = settings.resolve(self._quantities.values())
            parameters["trigger"] = {
                "channel": settings.channel, "level": settings.level, "slope": settings.slope,
                "record_samples": self._trigger.record_samples, "pre_trigger_samples": self._trigger.pre_samples,
                "position_percent": settings.position_percent, "column": self._trigger.column,
            }
        self._run_record = {
            "parameters": parameters,
            "named_devices": {"sampler": self.sampler_key},
            "device_snapshots": {"sampler": _working_point_snapshot(self.sampler, point)},
            # Host call time, not a device-clock to UTC calibration.
            "started_at_ns": time_ns(),
        }
        return point

    def _shot_time(self, record: WaveformRecord) -> float:
        if record.source_ordinal != self._next_ordinal:
            raise RuntimeError(f"waveform record gap: expected {self._next_ordinal}, got {record.source_ordinal}")
        if record.samples.shape[0] != (self._record_samples if self._trigger is None else self._trigger.record_samples):
            raise ValueError("waveform record sample count changed during capture")
        if self._last_time is not None and record.time_seconds <= self._last_time:
            raise RuntimeError("waveform record clock did not advance")
        if self._time_origin is None:
            self._time_origin = record.time_seconds
        self._last_time = record.time_seconds
        self._next_ordinal += 1
        return record.time_seconds - self._time_origin

    def _shot_outputs(
        self,
        record: WaveformRecord,
        *,
        revision: int,
        index: int | None = None,
    ) -> dict[str, LiveDatasetOutput]:
        point = self.working_point
        outputs: dict[str, LiveDatasetOutput] = {}
        shot_time = self._shot_time(record)
        timing = {
            "record_time_seconds": shot_time, "source_time_seconds": record.time_seconds,
            "time_basis": point.time_basis, "host_received_at_ns": record.host_received_at_ns,
        }
        if isinstance(record, TriggeredRecord):
            timing["crossing_offset_seconds"] = record.crossing_offset_seconds
        evidence = {"record_timing": {self.instance_id: {str(record.source_ordinal): timing}}}
        for declaration in self._outputs:
            snapshot = shot_snapshot(
                record,
                output=self._quantities[declaration.name],
                producer=self.instance_id,
                generation=self.generation,
                revision=revision,
                sample_interval_seconds=point.sample_interval_seconds,
                continuous=self._free_running and self._trigger is None,
                trigger_sample=0 if self._trigger is None else self._trigger.pre_samples,
            )
            if index is None:
                outputs[declaration.name] = LiveDatasetOutput(
                    declaration,
                    snapshot,
                    shot_time_seconds=shot_time,
                    event_record=evidence,
                )
                continue
            schema = snapshot.block.schema
            built = self._canonical.get(declaration.name)
            if built is None or built[0] is not schema:
                (repeat_axis,) = schema.repeat_domain.axes
                built = (schema, replace(
                    schema,
                    repeat_domain=DomainSpec(
                        (self.repeat,),
                        (replace(repeat_axis, size=self.repeat),),
                        (tuple(range(self.repeat)),),
                    ),
                ))
                self._canonical[declaration.name] = built
            outputs[declaration.name] = LiveDatasetOutput(
                declaration,
                snapshot,
                canonical_schema=built[1],
                cell_origin=(index, 0),
                event_record=evidence,
            )
        return outputs

    @property
    def silence_seconds(self) -> float:
        """How long a capture waits for the next record before calling the source silent."""

        return self._record_samples * self.working_point.sample_interval_seconds + self._source_timeout

    def take_records(self, limit: int | None = None) -> list[WaveformRecord]:
        """Every record the source holds (up to ``limit``), waiting a cancel slice for one.

        One read per batch, not per record: a capture that falls behind
        catches up at the cost of publishing alone, because reaching the
        source (a round trip through the core) is paid once for the backlog.
        """

        if self._trigger is not None:
            return self._take_triggered()
        count = self._batch_records if limit is None else max(1, min(int(limit), self._batch_records))
        records = list(self.sampler.read_records(count, timeout=_CANCEL_RESPONSE_SECONDS, exact=False))
        if records:
            self._delivered()
        return records

    def _take_triggered(self) -> list[TriggeredRecord]:
        """The shots the source's next records complete; one read a slice.

        A source of one-sample records would answer a read per sample, and most
        reads trigger nothing, so the slice bounds what looking costs: a record
        is published at most one slice late.
        """

        wait = self._next_read - monotonic()
        if wait > 0.0:
            sleep(wait)
        self._next_read = monotonic() + _CANCEL_RESPONSE_SECONDS
        records = self.sampler.read_records(self._batch_records, timeout=_CANCEL_RESPONSE_SECONDS, exact=False)
        if not records:
            return []
        self._last_delivery = monotonic()
        self._source_records_read += len(records)
        shots = self._detector.feed(records)
        if shots:
            self._delivered()
        return shots

    def drain_triggered(self, produced: int) -> list[TriggeredRecord]:
        """The shots in what a stopped source still holds of its ``produced`` records."""

        records = self.drain_records(produced - self._source_records_read)
        self._source_records_read = produced
        return self._detector.feed(records)

    def _delivered(self) -> None:
        self._last_delivery = self._last_shot = monotonic()
        if self._waiting:
            self._waiting = False
            if self._report_status is not None:
                self._report_status("Triggered")

    def check_quiet(self, *, current: int | None = None, total: int | None = None) -> None:
        """Nothing to publish this read: a dead source is a fault, a quiet trigger is a status."""

        now = monotonic()
        if self._free_running and now - self._last_delivery >= self.silence_seconds:
            raise TimeoutError("waveform source stopped delivering records")
        if (self._trigger is not None or not self._free_running) and not self._waiting \
                and now - self._last_shot >= _WAITING_AFTER_SECONDS:
            self._waiting = True
            if self._report_status is not None:
                self._report_status("Waiting for trigger", current=current, total=total)

    def drain_records(self, count: int) -> list[WaveformRecord]:
        """Exactly the ``count`` records a stopped source still holds, a batch per read."""

        records: list[WaveformRecord] = []
        while len(records) < count:
            records.extend(self.sampler.read_records(
                min(count - len(records), self._batch_records), timeout=0.0, exact=True))
        return records

    def _arm(self) -> None:
        # Capacity absorbs stalls; it never sets sample/record/plot cadence.
        capacity = max(2, ceil(_RECEIVE_BUFFER_SECONDS / (
            self._record_samples * self.working_point.sample_interval_seconds)))
        # A batch is the whole buffer, bounded so that its samples stay well
        # inside one channel frame however long the records are.
        columns = 1 + max(column for output in self._quantities.values() for column in output.columns)
        self._batch_records = max(1, min(capacity, _BATCH_BYTES // (4 * columns * self._record_samples)))
        # Triggered, the source streams until the run has its shots: how many of
        # its records that takes is the signal's business.
        self.sampler.arm(None if self._trigger is not None else self.repeat or None, buffer_record_count=capacity)
        self._run_record["acquisition"] = {"buffer_record_count": capacity}
        # Hardware may quantize the requested rate at arm (notably DAQ).
        # Freeze its accepted clock before publishing even the first record.
        self._working_point = self.sampler.working_point()
        self._run_record["device_snapshots"]["sampler"] = _working_point_snapshot(
            self.sampler, self._working_point
        )
        if self._trigger is not None:
            self._detector = TriggerDetector(self._trigger, self._working_point.sample_interval_seconds)
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
            capture = MonitorCapture(self, commit_live=context.commit_live)
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
            return self._result(capture.revision)
        self._armed(after_arm=context.record_devices)
        capture = FiniteCapture(self, should_stop=context.cancel_requested)
        try:
            context.set_run_record(self.run_record)
            context.report_ready()
        except BaseException:
            capture.close()
            raise

        def commit_shot(record: WaveformRecord, index: int) -> None:
            context.commit_live(self._shot_outputs(record, revision=index + 1, index=index))
            context.report_progress("Capturing", current=index + 1, total=int(self.repeat))

        return self._result(capture.collect(commit_shot=commit_shot))

    def _result(self, shots: int) -> dict[str, object]:
        return {"shots": shots} if self._trigger is None else {"shots": shots, "records_broken": self.broken_records}


__all__ = [
    "TRIGGER_FIELDS",
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
