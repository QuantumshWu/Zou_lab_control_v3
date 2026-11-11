"""Finite exact and live-monitor camera capture shared by the node."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from time import monotonic, time_ns

import numpy as np
from zlc_plugin import journal
from zlc_plugin.capture import validate_batch, terminal_evidence

from zlc_data import (
    AxisId,
    AxisRoleId,
    AxisSpec,
    CoordinateFrameId,
    DatasetSchema,
    DomainSpec,
    READOUT_EVENT,
    SPATIAL_X,
    SPATIAL_Y,
    integer,
    owning_bytes,
)
from zlc_plugin import DatasetOutputDeclaration, LiveDatasetOutput

from zlc_atom.devices.camera.contract import (
    CameraAdapter,
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
    roi_request,
)
from zlc_atom.devices.camera.photoelectrons import PHOTOELECTRONS
from zlc_atom.data import cell_axis_id, snapshot_from_array
from zlc_atom.nodes._framework.frames import FRAME_AXIS_NAME
from zlc_atom.nodes.camera.alignment import AlignmentError, Shot, TriggerAlignment, TriggerSchedule, frame_time


_CAMERA_FRAME_CONTRACT = "camera.frames"

#: The camera's ONE output: a cycle of frames on the dataset's POINT axis.
#: A cycle is one acquisition event; publishing its frames as N sibling
#: signals leaked the acquisition configuration into the signal vocabulary
#: (a panel bound to "frame_1" broke the moment frames_per_cycle changed),
#: and no consumer could see the whole cycle at once.  The frames live on
#: the POINT axis -- (repeat cycles) x (frame points) x (y, x) -- because a
#: frame is a point of the acquisition, not structure inside a pixel plane:
#: that is what lets a grid facet the frames side by side and lets a scan
#: compose them with scan axes in the same Point domain. The frame axis role
#: is READOUT_EVENT.
CAMERA_FRAMES_OUTPUT = DatasetOutputDeclaration("frames", _CAMERA_FRAME_CONTRACT)


def describe_camera_frames(values, output_name, source=None):
    """The camera draft's frame axis, named and labelled, before any image
    has been acquired."""
    if output_name != CAMERA_FRAMES_OUTPUT.name:
        return None
    try:
        count = int(values.get("frames_per_cycle") or 0)
    except (TypeError, ValueError):
        return None
    return (FRAME_AXIS_NAME, tuple(str(frame) for frame in range(count))) if count > 0 else None

#: How often a capture comes back to see whether it has been asked to stop.
#:
#: Cancellation is only ever seen BETWEEN reads, so the read length IS the
#: cancel latency, and it belongs to the LOOP rather than to the device.  It
#: used to be the CAMERA's timeout, which is a device fact about how long a
#: frame may take (2 s virtual, 10 s on the qCMOS), so an external-trigger
#: camera whose triggers had stopped sat inside one read until that deadline
#: -- and Stop, a Start queued behind the camera, and closing the console all
#: waited for it.  Fifty milliseconds is also the qCMOS driver's own wait
#: slice, so its SDK call cadence is unchanged.
#:
#: A monitor read is a POLL and takes whatever has arrived; a finite read
#: waits for a COMPLETE cycle and so must keep the camera's own timeout as
#: its deadline -- but it waits in slices of this, which is the difference
#: between a Stop that lands now and one that lands ten seconds from now.
_CANCEL_RESPONSE_SECONDS = 0.05
_RECEIVE_BUFFER_MIB = 128


def _frame_point_axis(producer: str, frames: int) -> AxisSpec:
    """The frame axis one cycle publishes in its Point domain."""

    return AxisSpec(
        AxisId(f"{producer}.frames.frame"),
        FRAME_AXIS_NAME,
        READOUT_EVENT,
        int(frames),
        tuple(range(int(frames))),
    )


@lru_cache(maxsize=32)
def _sensor_pixel_axis(
    producer: str,
    index: int,
    role: AxisRoleId,
    origin: int,
    step: int,
    size: int,
) -> AxisSpec:
    """One spatial axis of a frame, in the sensor's own pixels.

    Which pixels a frame covers is a fact about the working point, not about
    the frame -- a run publishes thousands of frames from one crop -- so the
    coordinates are built once per crop rather than per publication.

    The identity is the dataset's own, from the one function that generates
    it: saved boards, semantic choices and every downstream reference name a
    data axis by its id, so a producer adding coordinates must hand back the
    same name.
    """

    return AxisSpec(
        cell_axis_id(producer, CAMERA_FRAMES_OUTPUT.name, index, role),
        role.value,
        role,
        size,
        coordinates=tuple(origin + step * offset for offset in range(size)),
        unit="pixel",
        coordinate_frame=CoordinateFrameId("sensor_pixel_xy"),
    )


def _pixel_axes(
    point: "CameraWorkingPoint | None",
    shape_yx: tuple[int, int],
    *,
    producer: str,
) -> tuple[AxisSpec | AxisRoleId, AxisSpec | AxisRoleId]:
    """The sensor pixels a frame covers, as its two spatial axes.

    A frame is a crop of a sensor, and the numbers that mean anything about it
    are the sensor's own pixel coordinates: where the traps are, where an ROI
    starts, what a region drawn on the picture refers to.  Published without
    axes the picture was indexed from zero instead, so every coordinate on it
    was an offset into that particular crop -- the same region meant somewhere
    else the moment the ROI moved, which is exactly what a region does when it
    is used to set the ROI.

    The frame axis belongs to Point, so the cell is (y, x): those are the two
    cell axes the dataset will generate, in that order.
    """

    if point is None:
        return (SPATIAL_Y, SPATIAL_X)
    origin_y, origin_x = (int(value) for value in point.roi_origin_yx)
    step_y, step_x = (int(value) for value in getattr(point, "binning_yx", (1, 1)))
    height, width = (int(value) for value in shape_yx)
    return (
        _sensor_pixel_axis(
            producer, 0, SPATIAL_Y, origin_y, max(1, step_y), height
        ),
        _sensor_pixel_axis(
            producer, 1, SPATIAL_X, origin_x, max(1, step_x), width
        ),
    )


def photoelectron_frame(counts: object, point: CameraWorkingPoint) -> np.ndarray:
    """Counts as photoelectrons through the camera's stated conversion.

    One allocation: the subtraction widens straight into float32 and the
    scale lands in place.  It is also the ONE spelling of the arithmetic --
    a saturation level is a count pushed through this same expression, and
    comparing two spellings for equality holds only while they agree to the
    last bit.
    """

    image = np.subtract(counts, np.float32(point.offset_counts), dtype=np.float32)
    image *= np.float32(point.electrons_per_count)
    return image


def frames_snapshot(
    cycles: "Sequence[Sequence[CameraFrameRecord]]",
    *,
    producer: str,
    generation: object,
    revision: int,
    working_point: "CameraWorkingPoint | None" = None,
    value_unit: str | None,
    present: "Sequence[Sequence[bool]] | None" = None,
):
    """Cycles of frames as one dataset: (cycle) x (frame) x (y, x).

    ``present`` says, per cycle and frame, whether that frame was taken: given, the dataset carries
    validity, and a frame that was not is invalid whatever stands in its place.

    Every publisher of camera frames -- the finite capture, the monitor slot,
    and the SLM feedback and rearrangement tasks -- means exactly this
    dataset, so they build it here.  A stacking per publisher is how the
    frame point column drifts between the live view and the saved run.
    """

    frames = tuple(tuple(cycle) for cycle in cycles)
    if not frames:
        raise ValueError("camera publication requires at least one cycle")
    sizes = {len(cycle) for cycle in frames}
    if len(sizes) != 1 or not sizes.pop():
        raise ValueError("every published camera cycle must have the same frames")
    # Every record owns its frame as contiguous immutable bytes, which the
    # Dataset keeps as it is.  One frame is handed on as that view; several
    # are joined with the one copy the Dataset needs, from the frames' own
    # bytes objects so that copy runs without the GIL (owning_bytes).
    images = [record.image for cycle in frames for record in cycle]
    first = images[0]
    values = (
        first
        if len(images) == 1
        else np.frombuffer(
            b"".join([
                image if (owner := owning_bytes(image)) is None else owner
                for image in images
            ]),
            dtype=first.dtype,
        )
    )
    validity = None
    if present is not None:
        taken = np.asarray(present, dtype=bool).reshape(len(frames), len(frames[0]), *([1] * first.ndim))
        validity = np.broadcast_to(taken, (len(frames), len(frames[0]), *first.shape)).copy()
    return snapshot_from_array(
        values.reshape(len(frames), len(frames[0]), *first.shape),
        validity=validity,
        producer=producer,
        signal=CAMERA_FRAMES_OUTPUT.name,
        point_axes=(_frame_point_axis(producer, len(frames[0])),),
        cell_axes=_pixel_axes(
            working_point,
            first.shape,
            producer=producer,
        ),
        value_unit=value_unit,
        generation=str(getattr(generation, "value", generation)),
        revision=int(revision),
    )


def _finite_cycle_output(
    node: "CameraMeasurementNode",
    cycle: Sequence[CameraFrameRecord],
    index: int,
    *,
    present: Sequence[bool] | None = None,
    record: dict[str, object] | None = None,
) -> LiveDatasetOutput:
    """One new cycle placed in the fixed authored finite run geometry."""

    event = frames_snapshot(
        (cycle,),
        producer=node.instance_id,
        generation=node.generation,
        revision=index + 1,
        working_point=node.actual_working_point,
        value_unit=node.frame_value_unit,
        present=None if present is None else (present,),
    )
    event_schema = event.block.schema
    # Built once per run: the event schema is one cached object for the whole
    # run, and the canonical one rebuilt from an O(repeat) code vector on
    # every cycle was also a new object the plane compared code by code.
    built = node._canonical
    if built is None or built[0] is not event_schema:
        (repeat_axis,) = event_schema.repeat_domain.axes
        built = (event_schema, replace(
            event_schema,
            repeat_domain=DomainSpec(
                (node.repeat,),
                (replace(repeat_axis, size=node.repeat),),
                (tuple(range(node.repeat)),),
            ),
        ))
        node._canonical = built
    canonical = built[1]
    return LiveDatasetOutput(
        CAMERA_FRAMES_OUTPUT,
        event,
        canonical_schema=canonical,
        cell_origin=(index, 0),
        event_record={**node._camera_event_record(_taken(cycle, present)), **(record or {})},
    )


def _taken(cycle: Sequence[CameraFrameRecord], present: Sequence[bool] | None) -> tuple[CameraFrameRecord, ...]:
    """The frames of a cycle that were really taken (a stand-in for a missing one is no camera's)."""

    return tuple(cycle) if present is None else tuple(record for record, taken in zip(cycle, present) if taken)


def _monitor_cycle_output(
    node: "CameraMeasurementNode",
    cycle: Sequence[CameraFrameRecord],
    revision: int,
    *,
    present: Sequence[bool] | None = None,
    record: dict[str, object] | None = None,
) -> LiveDatasetOutput:
    event = frames_snapshot(
        (cycle,),
        producer=node.instance_id,
        generation=node.generation,
        revision=revision,
        working_point=node.actual_working_point,
        value_unit=node.frame_value_unit,
        present=None if present is None else (present,),
    )
    return LiveDatasetOutput(
        CAMERA_FRAMES_OUTPUT,
        event,
        event_record={**node._camera_event_record(_taken(cycle, present)), **(record or {})},
    )


def _validated_cycle(records: Sequence[CameraFrameRecord], *, expected_start: int,
                     frames_per_cycle: int, group_ends: frozenset[int]) -> tuple[CameraFrameRecord, ...]:
    """Assemble owned records only at a boundary already proved by the core."""
    cycle = tuple(records)
    if (len(cycle) != frames_per_cycle or not cycle
            or cycle[0].source_ordinal != expected_start or cycle[-1].source_ordinal not in group_ends):
        raise RuntimeError("camera buffer does not match its validated cycle boundary")
    return cycle


def _camera_working_point_snapshot(point: CameraWorkingPoint) -> dict[str, object]:
    """Return the adapter readback as plain, archive-ready run metadata."""

    return {
        "acquisition_mode": point.acquisition_mode,
        "frame_shape_yx": [int(value) for value in point.frame_shape_yx],
        "sensor_shape_yx": [int(value) for value in point.sensor_shape_yx],
        "roi_origin_yx": [int(value) for value in point.roi_origin_yx],
        "roi_shape_yx": [int(value) for value in point.roi_shape_yx],
        "binning_yx": [int(value) for value in point.binning_yx],
        "dtype": point.dtype.str,
        "count_unit": str(point.count_unit),
        "offset_counts": (
            None if point.offset_counts is None else float(point.offset_counts)
        ),
        "electrons_per_count": (
            None
            if point.electrons_per_count is None
            else float(point.electrons_per_count)
        ),
        "exposure_seconds": float(point.exposure_seconds),
        "required_external_trigger_interval_seconds": (
            None
            if point.required_external_trigger_interval_seconds is None
            else float(point.required_external_trigger_interval_seconds)
        ),
        "external_trigger_integration_start_offset_seconds": (
            None
            if point.external_trigger_integration_start_offset_seconds is None
            else float(point.external_trigger_integration_start_offset_seconds)
        ),
        "gain": float(point.gain),
        "readout_mode": str(point.readout_mode),
    }


@dataclass(frozen=True)
class CameraMeasurementRequest:
    """One frozen camera selection and acquisition working point."""

    camera_key: str
    #: None keeps the exposure / ROI the camera has now.  So does a None
    #: ROI field, for that field alone: ``(None, None, None, 64)`` changes
    #: only the height, and the run completes the rectangle from the
    #: camera's current ROI when it arms.
    exposure_seconds: float | None
    roi_xywh: tuple[int | None, int | None, int | None, int | None] | None
    repeat: int
    frames_per_cycle: int
    #: Read the camera in photoelectrons instead of counts, through the
    #: conversion the CAMERA's configuration states.  A camera that states
    #: none falls back to raw counts rather than inventing a conversion; the
    #: effective choice rides in the run record.
    photoelectrons: bool = True
    #: The sequencer lane that triggers this camera (a raw lane, ``ch11``), or None to count frames.
    #: Named, a cycle is that lane's triggers of one shot and ``frames_per_cycle`` is not read.
    trigger_channel: str | None = None

    def __post_init__(self) -> None:
        camera_key = str(self.camera_key).strip()
        if not camera_key:
            raise ValueError("camera_key must be non-empty")
        exposure = None if self.exposure_seconds is None else float(self.exposure_seconds)
        if exposure is not None and (not np.isfinite(exposure) or exposure <= 0):
            raise ValueError("exposure_seconds must be positive and finite")
        repeat = int(self.repeat)
        frames_per_cycle = int(self.frames_per_cycle)
        if repeat < 0:
            raise ValueError("repeat must be non-negative")
        if frames_per_cycle <= 0:
            raise ValueError("frames_per_cycle must be positive")
        object.__setattr__(self, "camera_key", camera_key)
        roi = self.roi_xywh
        if roi == (None, None, None, None):
            roi = None
        if isinstance(roi, tuple) and len(roi) == 4 and None in roi:
            roi = tuple(
                None if value is None
                else integer(value, f"roi_xywh[{index}]", minimum=0 if index < 2 else 1)
                for index, value in enumerate(roi)
            )
        else:
            roi = roi_request(roi)
        object.__setattr__(self, "exposure_seconds", exposure)
        object.__setattr__(self, "roi_xywh", roi)
        object.__setattr__(self, "repeat", repeat)
        object.__setattr__(self, "frames_per_cycle", frames_per_cycle)
        object.__setattr__(self, "photoelectrons", bool(self.photoelectrons))
        object.__setattr__(self, "trigger_channel", str(self.trigger_channel or "").strip() or None)


class FiniteCapture:
    """An armed finite capture whose triggers are supplied by another owner."""

    def __init__(
        self,
        node: "CameraMeasurementNode",
        *,
        repeat: int,
        frames_per_cycle: int,
        timeout: float,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        self.node = node
        self.camera = node.camera
        self.repeat = int(repeat)
        self.frames_per_cycle = int(frames_per_cycle)
        self.timeout = float(timeout)
        #: Asked between reads: the run's cancel.
        self.should_stop = should_stop
        self.closed = False
        self.completed_cycles = 0
        #: Whether the owner asked this capture to stop, which is the one
        #: reason a device may honestly report more frames than the completed
        #: cycles account for: the cycle it was walking away from.
        self.stopped = False
        self.terminal: CameraCaptureTerminalRecord | None = None
        self._pending_records: list[CameraFrameRecord] = []

    def read_cycles(
        self,
        on_cycle: Callable[[tuple[CameraFrameRecord, ...], int], None],
    ) -> CameraCaptureTerminalRecord:
        """Hand each newly completed cycle and its run index to ``on_cycle``.

        Runtime owns every prior cycle and the fixed authored shape; only the
        new cycle is handed on. A normal Stop first stops intake, then hands on
        every accepted complete cycle (including one partly read when Stop
        arrived) before the terminal; a final partial cycle is counted but never
        published as complete. A failure hands on nothing more.
        """
        if self.closed:
            raise RuntimeError("finite capture is closed")
        if not callable(on_cycle):
            raise TypeError("on_cycle must be callable")
        try:
            for index in range(self.repeat):
                cycle = self.next_cycle()
                if cycle is None:
                    break
                on_cycle(cycle, index)
            terminal = self.close()
            complete = min(self.repeat, self._terminal_proof["complete_groups"])
            while self.completed_cycles < complete:
                pending = self._pending_records
                pending.extend(self.node.read_records(
                    self.frames_per_cycle - len(pending), timeout=0.0, exact=True,
                ))
                cycle = _validated_cycle(
                    pending, expected_start=self.completed_cycles * self.frames_per_cycle,
                    frames_per_cycle=self.frames_per_cycle, group_ends=self.node._capture_group_ends,
                )
                pending.clear()
                index = self.completed_cycles
                self.completed_cycles += 1
                on_cycle(cycle, index)
            remaining = terminal.produced_count - self.node._next_record_ordinal
            if remaining:
                self.node.read_records(remaining, timeout=0.0, exact=True)
            self._pending_records.clear()
            terminal = replace(terminal, no_more_frames=True)
            self.terminal = terminal
        except BaseException as error:
            if not self.closed:
                try:
                    self.camera.finish_record_capture()
                except BaseException as cleanup:
                    error.add_note(f"camera cleanup also failed: {cleanup}")
                finally:
                    self.closed = True
            raise
        return terminal

    def next_cycle(self) -> tuple[CameraFrameRecord, ...] | None:
        """The next complete cycle of this capture, or None if asked to stop.

        A scan takes its cycles one at a time -- that is what lets the dataset
        grow on screen while the board is still playing -- and so does a
        finite measurement.  Same read.

        The wait is sliced.  A cycle is complete only when all of its frames
        have arrived, and how long that may take is the CAMERA's timeout (2 s
        virtual, 10 s on the qCMOS) -- but a Stop is only ever seen between
        reads, so waiting that out in one call is a Stop the operator watches
        the console refuse for ten seconds.  The deadline is unchanged; only
        the granularity is.
        """

        if self.closed:
            raise RuntimeError("finite capture is closed")
        records = self._pending_records
        deadline = monotonic() + self.timeout
        while len(records) < self.frames_per_cycle:
            if self.should_stop is not None and self.should_stop():
                self.stopped = True
                return None
            arrived = self.node.read_records(
                self.frames_per_cycle - len(records),
                timeout=min(_CANCEL_RESPONSE_SECONDS, max(0.0, deadline - monotonic())),
                exact=False,
            )
            if arrived:
                records.extend(arrived)
                # A frame arrived, so the camera is delivering: the deadline
                # is how long a FRAME may take, not how long a cycle may.
                deadline = monotonic() + self.timeout
                continue
            if monotonic() >= deadline:
                break
        if len(records) != self.frames_per_cycle:
            point = self.node.actual_working_point
            exposure = "unknown" if point is None else format(point.exposure_seconds, "g")
            interval = None if point is None else point.required_external_trigger_interval_seconds
            raise RuntimeError(
                f"the camera returned {len(records)} frame(s) of a "
                f"{self.frames_per_cycle}-frame cycle before timeout. Camera "
                f"readback: exposure {exposure}s, minimum external-trigger "
                "interval "
                f"{'unknown' if interval is None else format(interval, 'g') + 's'}"
                ". Camera Measurement has no Pulse schedule; inspect the "
                "external rising-edge count, spacing, and source state"
            )
        expected = self.completed_cycles * self.frames_per_cycle
        cycle = _validated_cycle(
            records,
            expected_start=expected,
            frames_per_cycle=self.frames_per_cycle, group_ends=self.node._capture_group_ends,
        )
        self.completed_cycles += 1
        records.clear()
        return cycle

    def close(self) -> CameraCaptureTerminalRecord:
        """Finish the capture with the device's evidence, or not at all.

        Only a finish the device completed and the count check accepted is
        cached; a finish the device refused leaves the capture open so the
        next close retries the same finish.  A capture that a failed collect
        already closed has no terminal to hand out and says so.
        """

        if self.terminal is not None:
            return self.terminal
        if self.closed:
            raise RuntimeError("finite capture closed without terminal evidence")
        terminal = self.camera.finish_record_capture()
        proof = terminal_evidence(terminal, expected=self.completed_cycles * self.frames_per_cycle,
                                  stopped=self.stopped, group_size=self.frames_per_cycle)
        self._terminal_proof = proof
        self.closed = True
        self.terminal = terminal
        return terminal


class MonitorCapture:
    """A repeat-zero capture publishing every accepted complete cycle."""

    def __init__(
        self,
        camera: CameraAdapter,
        *,
        node: "CameraMeasurementNode",
        commit_live: Callable[..., object],
    ) -> None:
        if not callable(commit_live):
            raise TypeError("a monitor requires commit_live")
        self.camera = camera
        self.node = node
        self.closed = False
        self.terminal: CameraCaptureTerminalRecord | None = None
        self.latest_record: CameraFrameRecord | None = None
        self._pending_records: list[CameraFrameRecord] = []
        self._revision = 0
        self._commit_live = commit_live

    def poll(self) -> CameraFrameRecord | None:
        if self.closed:
            raise RuntimeError("monitor capture is closed")
        records = self.node.read_records(
            16, timeout=_CANCEL_RESPONSE_SECONDS, exact=False
        )
        if not records:
            return None
        for record in records:
            self._accept_record(record)
        self.latest_record = records[-1]
        return self.latest_record

    def _accept_record(self, record: CameraFrameRecord) -> None:
        """Publish only a physically aligned, contiguous camera cycle."""

        # ``read_records`` refuses any gap in the ordinals from the first frame
        # on, so consecutive records ARE the aligned cycle.
        pending = self._pending_records
        pending.append(record)
        if record.source_ordinal in self.node._capture_group_ends:
            cycle = tuple(pending)
            pending.clear()
            self._revision += 1
            self._commit_live(
                {
                    CAMERA_FRAMES_OUTPUT.name: _monitor_cycle_output(
                        self.node,
                        cycle,
                        self._revision,
                    )
                }
            )

    def close(self, *, drain: bool = True) -> CameraCaptureTerminalRecord:
        """Disarm the camera, then publish the accepted complete cycles.

        Only the terminal the device produced is cached and handed back: a
        disarm the device refused is raised and left for the next close to
        retry, never answered with a made-up all-clear over a camera that is
        still armed.  The device disarm comes first, so a drain failure cannot
        skip it; the core seals the run.
        """

        if self.terminal is not None:
            return self.terminal
        terminal = self.camera.finish_record_capture()
        terminal_evidence(terminal, expected=self.node._next_record_ordinal, stopped=True,
                          group_size=self.node.frames_per_cycle)
        self.closed = True
        if drain:
            remaining = terminal.produced_count - self.node._next_record_ordinal
            while remaining:
                records = self.node.read_records(min(16, remaining), timeout=0.0, exact=True)
                for record in records:
                    self._accept_record(record)
                remaining -= len(records)
            # A partial cycle is counted but is not a scientific publication.
            self._pending_records.clear()
            terminal = replace(terminal, no_more_frames=True)
        self.terminal = terminal
        return terminal


class AlignedCapture:
    """A capture whose frames are sorted into shots by the schedule of the lane that triggers the camera.

    The board is asked what it plays whenever frames are read (``schedule(known revision)``); each
    FIRE is a play of its own, with its own alignment.  A frame at no trigger's time is dropped and
    said; a shot a trigger of which brought no frame is published with that frame invalid, so every
    consumer still counts it.  Until a pulse plays, frames belong to nothing and none is published.
    """

    def __init__(self, node: "CameraMeasurementNode", *, commit_live: Callable[..., object],
                 schedule: Callable[[int | None], Mapping[str, object]], armed_us: int,
                 say: Callable[..., object] | None = None) -> None:
        self.node = node
        self.camera = node.camera
        self._commit_live = commit_live
        self._schedule = schedule
        self._armed_us = int(armed_us)
        self._say = say
        self._revision: int | None = None
        self._play: tuple[int, int] | None = None
        self._alignment: TriggerAlignment[CameraFrameRecord] | None = None
        #: When the last frame of the play in hand was received (ns), for a shot whose end never comes.
        self._last_frame_ns: int | None = None
        self.published = 0
        self.incomplete = 0
        self.dropped = 0
        self.closed = False
        self.terminal: CameraCaptureTerminalRecord | None = None

    @property
    def finished(self) -> bool:
        """A finite measurement has published the shots it was asked for."""

        return bool(self.node.repeat) and self.published >= self.node.repeat

    # ------------------------------------------------------------------ which play
    def _follow_board(self, records: Sequence[CameraFrameRecord]) -> Sequence[CameraFrameRecord]:
        """Learn what the board plays now; the frames of ``records`` that are the new play's."""

        answer = self._schedule(self._revision)
        if answer.get("unchanged"):
            return records
        schedule = TriggerSchedule.from_answer(answer)
        self._revision = schedule.revision
        fired = schedule.command_id is not None and schedule.state in ("playing", "done", "stopped")
        play = (int(schedule.command_id), schedule.sent_us) if fired else None
        if play == self._play:
            return records
        # Frames this computer received before the new FIRE was sent are the old play's last.
        mine = [record for record in records if play is not None and record.host_received_at_ns // 1000 >= schedule.sent_us]
        for record in records[: len(records) - len(mine)]:
            self._accept(record)
        self._end_play()
        self._play = play
        # A play that had ended before this camera was armed sends it nothing.
        if play is not None and schedule.points and (schedule.playing or schedule.sent_us >= self._armed_us):
            frames = schedule.frames
            if self.node._aligned_frames not in (None, frames):
                raise AlignmentError(
                    f"{schedule.lane} now triggers this camera {frames} times a shot; this run publishes cycles of "
                    f"{self.node._aligned_frames}: restart the measurement for the new pulse"
                )
            self.node._aligned_frames = frames
            exposure = self.node.actual_working_point.exposure_seconds if self.node.actual_working_point is not None else 0.0
            self._alignment = TriggerAlignment(schedule, from_first_trigger=schedule.sent_us >= self._armed_us,
                                               separation=float(exposure or 0.0))
            journal.note("camera", "aligned_to", key=self.node.request.camera_key, lane=schedule.lane, command_id=schedule.command_id,
                         frames=frames, points=len(schedule.points), from_first_trigger=schedule.sent_us >= self._armed_us)
        return mine

    def _end_play(self) -> None:
        """The play in hand is over: its whole shots are out; the one it stopped in the middle of is no shot."""

        alignment, self._alignment = self._alignment, None
        self._last_frame_ns = None
        if alignment is None:
            return
        for found in alignment.flush():
            if isinstance(found, Shot) and found.complete:
                self._publish(found)
            elif isinstance(found, Shot):
                journal.note("camera", "shot_abandoned", key=self.node.request.camera_key, shot=found.index,
                             frames=[None if frame is None else int(frame.source_ordinal) for frame in found.frames])
            else:
                self._drop(found.frame, found.reason)

    # ------------------------------------------------------------------ frames
    def poll(self) -> None:
        if self.closed:
            raise RuntimeError("aligned capture is closed")
        records = self.node.read_records(16, timeout=_CANCEL_RESPONSE_SECONDS, exact=False, group_size=1)
        for record in self._follow_board(records):
            self._accept(record)
        # A shot whose last frames never come is over once its own length has passed since its last frame.
        alignment = self._alignment
        if alignment is not None and self._last_frame_ns is not None and alignment.overdue_after is not None \
                and (time_ns() - self._last_frame_ns) / 1e9 > alignment.overdue_after:
            self._last_frame_ns = None
            self._take(alignment.flush())

    def _accept(self, record: CameraFrameRecord) -> None:
        if self._alignment is None:
            self._drop(record, "no pulse was playing on its trigger lane")
            return
        self._last_frame_ns = int(record.host_received_at_ns) or time_ns()
        self._take(self._alignment.accept(frame_time(record), record))

    def _take(self, found: Sequence[object]) -> None:
        for item in found:
            if isinstance(item, Shot):
                self._publish(item)
            else:
                self._drop(item.frame, item.reason)

    def _drop(self, record: CameraFrameRecord, reason: str) -> None:
        self.dropped += 1
        journal.note("camera", "frame_dropped", key=self.node.request.camera_key, ordinal=int(record.source_ordinal), reason=reason)
        if self._say is not None:
            self._say(f"frame {int(record.source_ordinal)} is no trigger of the pulse ({reason}); {self.dropped} dropped so far")

    def _publish(self, shot: "Shot[CameraFrameRecord]") -> None:
        if self.finished:
            return
        taken = [frame for frame in shot.frames if frame is not None]
        # A frame that never came is stood in for by one that did, and marked invalid.
        cycle = tuple(frame if frame is not None else taken[0] for frame in shot.frames)
        present = tuple(frame is not None for frame in shot.frames)
        record = {"trigger_alignment": {"shot": shot.index, "missing": [slot for slot, here in enumerate(present) if not here]}}
        if not shot.complete:
            self.incomplete += 1
            journal.note("camera", "shot_incomplete", key=self.node.request.camera_key, shot=shot.index, missing=record["trigger_alignment"]["missing"])
            if self._say is not None:
                self._say(f"shot {shot.index + 1} lacks {len(shot.frames) - len(taken)} of its {len(shot.frames)} frames: published invalid")
        output = (_finite_cycle_output(self.node, cycle, self.published, present=present, record=record) if self.node.repeat
                  else _monitor_cycle_output(self.node, cycle, self.published + 1, present=present, record=record))
        self._commit_live({CAMERA_FRAMES_OUTPUT.name: output})
        self.published += 1

    def close(self, *, drain: bool = True) -> CameraCaptureTerminalRecord:
        """Disarm the camera, then give the frames it had already taken their places."""

        if self.terminal is not None:
            return self.terminal
        terminal = self.camera.finish_record_capture()
        self.closed = True
        if drain:
            remaining = terminal.produced_count - self.node._next_record_ordinal
            while remaining > 0:
                records = self.node.read_records(min(16, remaining), timeout=0.0, exact=True, group_size=1)
                for record in records:
                    self._accept(record)
                remaining -= len(records)
            self._end_play()
            terminal = replace(terminal, no_more_frames=True)
        self.terminal = terminal
        return terminal


class CameraMeasurementNode:
    """Commit each atomic camera cycle to one stable ``frames`` signal."""

    def __init__(
        self,
        *,
        camera: CameraAdapter,
        request: CameraMeasurementRequest,
        producer: str = "camera_measurement",
        record_received: Callable[[CameraFrameRecord], None] | None = None,
    ) -> None:
        if not isinstance(camera, CameraAdapter):
            raise TypeError("camera must implement CameraAdapter")
        if not isinstance(request, CameraMeasurementRequest):
            raise TypeError("request must be CameraMeasurementRequest")
        self.camera = camera
        self._record_received = record_received
        self._request = request
        self._actual_working_point: CameraWorkingPoint | None = None
        #: The whole ROI this run asked the camera for, blanks completed.
        self._run_roi: tuple[int, int, int, int] | None = None
        self._run_record: dict[str, object] | None = None
        self._settings_session_id: str | None = None
        self.instance_id = str(producer).strip()
        if not self.instance_id:
            raise ValueError("producer must be non-empty")
        self._generation: object | None = None
        #: The event schema a finite run's canonical schema was built from,
        #: and that canonical schema (``_finite_cycle_output``).
        self._canonical: tuple[DatasetSchema, DatasetSchema] | None = None
        #: The frames of one cycle as the trigger lane's pulse has them, once that pulse is known.
        self._aligned_frames: int | None = None

    @property
    def request(self) -> CameraMeasurementRequest:
        return self._request

    @property
    def actual_working_point(self) -> CameraWorkingPoint | None:
        return self._actual_working_point

    @property
    def reads_photoelectrons(self) -> bool:
        point = self._actual_working_point
        if point is None:
            raise RuntimeError("camera working point is not frozen")
        return bool(
            self.request.photoelectrons
            and point.electrons_per_count is not None
        )

    @property
    def frame_value_unit(self) -> str | None:
        point = self._actual_working_point
        if point is None:
            raise RuntimeError("camera working point is not frozen")
        return None if self.reads_photoelectrons else point.count_unit

    @property
    def generation(self) -> object:
        if self._generation is None:
            raise RuntimeError("camera acquisition has no active generation")
        return self._generation

    @property
    def camera_key(self) -> str:
        return self.request.camera_key

    @property
    def exposure_seconds(self) -> float | None:
        return self.request.exposure_seconds

    @property
    def roi_xywh(self) -> tuple[int, int, int, int] | None:
        return self.request.roi_xywh

    @property
    def repeat(self) -> int:
        return self.request.repeat

    @property
    def aligned(self) -> bool:
        """Whether frames are sorted by the trigger lane's schedule rather than counted."""

        return self.request.trigger_channel is not None

    @property
    def frames_per_cycle(self) -> int:
        return self.request.frames_per_cycle

    def _configure_for_run(self) -> CameraWorkingPoint:
        self._actual_working_point = None
        self._run_record = None
        self._next_record_ordinal = 0
        self._capture_group_ends: frozenset[int] = frozenset()
        self._canonical = None
        # This measurement owns both: it exists to point the camera.  The
        # geometry first, because it is the expensive one to get wrong.  A
        # blank ROI field is the camera's current value for that field, read
        # now, when the run arms; whether the completed rectangle fits the
        # sensor is the camera's to say.
        roi = self.request.roi_xywh
        if roi is not None and None in roi:
            point = self.camera.working_point()
            (origin_y, origin_x), (height, width) = point.roi_origin_yx, point.roi_shape_yx
            roi = tuple(
                int(current) if value is None else value
                for value, current in zip(roi, (origin_x, origin_y, width, height))
            )
        self._run_roi = roi
        self.camera.set_roi(roi)
        point = self.camera.set_exposure_seconds(self.request.exposure_seconds)
        if not isinstance(point, CameraWorkingPoint):
            raise TypeError("camera set_exposure_seconds must return CameraWorkingPoint")
        return point

    def _buffer_frame_count(self, point: CameraWorkingPoint) -> int:
        # SDK ring and accepted FIFO each hold at most N raw frames. Scientific
        # history and one-frame copy scratch are not receive-buffer storage.
        frame_bytes = int(np.prod(point.frame_shape_yx)) * point.dtype.itemsize
        count = _RECEIVE_BUFFER_MIB * 1024 * 1024 // (2 * frame_bytes)
        if count < self.frames_per_cycle:
            raise ValueError(
                f"Internal receive buffer {_RECEIVE_BUFFER_MIB} MiB cannot hold "
                f"one {self.frames_per_cycle}-frame cycle at {frame_bytes} bytes/frame"
            )
        if self.repeat:
            count = min(count, self.repeat * self.frames_per_cycle)
        return count

    def _configure_capture(self) -> CameraWorkingPoint:
        """Apply and freeze the requested working point without arming."""

        # ``set_exposure_seconds`` already returns the authoritative readback
        # after ROI and exposure have been applied.  Reading the complete
        # qCMOS property surface again made every finite Start pay a third
        # full working-point query before ARM.
        self._freeze_working_point(self._configure_for_run())
        assert self._actual_working_point is not None
        return self._actual_working_point

    def _arm_configured(
        self,
        *,
        should_stop: Callable[[], bool] | None,
    ) -> FiniteCapture:
        """Arm a capture whose hardware working point is already frozen."""

        if self._actual_working_point is None or self._run_record is None:
            raise RuntimeError("camera capture must be configured before arm")
        timeout = float(self.camera.timeout)
        total = self.request.repeat * self.request.frames_per_cycle
        groups = (self.request.frames_per_cycle,) * self.request.repeat
        buffer_frames = self._buffer_frame_count(self._actual_working_point)
        self.camera.arm(
            total,
            source_group_sizes=groups,
            buffer_frame_count=buffer_frames,
            timeout=timeout,
        )
        self._freeze_working_point(self.camera.working_point())
        self._run_record["acquisition"] = {"buffer_frame_count": buffer_frames}
        return FiniteCapture(
            self,
            repeat=self.request.repeat,
            frames_per_cycle=self.request.frames_per_cycle,
            timeout=timeout,
            should_stop=should_stop,
        )

    def read_records(
        self,
        count: int,
        *,
        timeout: float,
        exact: bool,
        group_size: int | None = None,
    ) -> tuple[CameraFrameRecord, ...]:
        """Take frames from the camera, in the unit this run publishes.

        THE read.  A capture never touches the adapter itself, and that is
        the whole point of this method existing: when the finite read did the
        conversion and the monitor read did not, a live panel showed counts
        while the same run's saved samples were electrons, and nothing on
        screen said which was which.  One intake cannot disagree with itself.

        Counts stay the sensor's own integers -- the pipeline is built on
        that, and a 2048x2048 frame costs 9.6 ms and twice the memory to
        widen.  Photoelectrons are float32: the conversion is affine, so it
        moves no decision (thresholds move with it), and what it buys is
        numbers a physicist can read.
        """

        # An acquisition with first-frame actions must not wait for the rest of
        # its cycle. Ordinary Camera Measurement retains its existing batching.
        read_count = 1 if self._record_received is not None and not exact else int(count)
        records = tuple(
            self.camera.read_frame_records(read_count, timeout=timeout, exact=exact)
        )
        if records:
            accepted = validate_batch((record.source_ordinal for record in records),
                                      expected_start=self._next_record_ordinal,
                                      # A capture that sorts frames by time takes each as a group of its own.
                                      group_size=self.frames_per_cycle if group_size is None else group_size)
            self._next_record_ordinal = int(accepted["next"])
            self._capture_group_ends = frozenset(accepted["group_ends"])
        for record in records:
            # Every frame as this measurement took it: which, when the camera and the host each
            # say it came, and whether it closes a cycle. The journal's one line per frame.
            journal.note("camera", "frame", key=self.request.camera_key, ordinal=int(record.source_ordinal),
                         cycle_end=record.source_ordinal in self._capture_group_ends if group_size is None else None,
                         frame_stamp=record.frame_stamp, camera_stamp=record.camera_stamp,
                         camera_seconds=record.timestamp_seconds, camera_microseconds=record.timestamp_microseconds,
                         host_received_at_ns=record.host_received_at_ns)
            if self._record_received is not None:
                self._record_received(record)
        if not self.reads_photoelectrons:
            return records
        point = self._actual_working_point
        assert point is not None and point.electrons_per_count is not None
        # The record owns the converted frame as bytes: that copy and the one
        # allocation of the conversion are all a frame costs here.
        return tuple(
            replace(record, image=photoelectron_frame(record.image, point))
            for record in records
        )

    def _freeze_working_point(self, point: CameraWorkingPoint) -> None:
        if not isinstance(point, CameraWorkingPoint):
            raise TypeError("camera working_point must return CameraWorkingPoint")
        self._actual_working_point = point
        photoelectrons = self.reads_photoelectrons
        record = {
            "parameters": {
                "exposure_seconds": self.request.exposure_seconds,
                "roi_xywh": None if self._run_roi is None else list(self._run_roi),
                "repeat": self.request.repeat,
                # Aligned to a trigger lane, a cycle's size is the lane's: known when its pulse is.
                "frames_per_cycle": self._aligned_frames if self.request.trigger_channel else self.request.frames_per_cycle,
                "trigger_channel": self.request.trigger_channel,
                PHOTOELECTRONS: photoelectrons,
            },
            "named_devices": {"camera": self.request.camera_key},
            "device_snapshots": {
                "camera": _camera_working_point_snapshot(point),
            },
        }
        self._run_record = record
        self._settings_session_id = None

    def _camera_event_record(
        self,
        records: Sequence[CameraFrameRecord],
    ) -> dict[str, object]:
        """The settings frozen on THESE frames, never current state.

        An event record describes the chunk it travels with -- this cycle,
        these frames -- and nothing before it: the Runtime merges the epoch
        ranges of every retained chunk into the canonical prefix, so a cycle
        that also carried the epochs of earlier cycles was the same fact stated
        twice, once wrongly.  What does span the acquisition is the device
        session: a camera that changed identity between cycles is refused.
        """

        frames = tuple(records)
        references = tuple(
            (record.settings_session_id, record.settings_epochs)
            for record in frames
            if record.settings_session_id is not None
        )
        if not references:
            return {}
        if len(references) != len(frames):
            raise RuntimeError("one camera cycle mixed settings-aware and unaware frames")
        session_ids = {str(session_id) for session_id, _epochs in references}
        if len(session_ids) != 1:
            raise RuntimeError("camera device session changed inside one cycle")
        session_id = next(iter(session_ids))
        if self._settings_session_id not in (None, session_id):
            raise RuntimeError("camera device session changed during one acquisition")
        self._settings_session_id = session_id
        ordered = sorted({
            epoch
            for _session_id, epochs in references
            for epoch in epochs
        })
        ranges: list[list[int]] = []
        for epoch in ordered:
            if ranges and epoch == ranges[-1][1] + 1:
                ranges[-1][1] = epoch
            else:
                ranges.append([epoch, epoch])
        return {"device_settings": {
            "camera": {
                "device_session_id": session_id,
                "epoch_ranges": ranges,
                "mixed": len(ordered) > 1,
            }
        }}

    @property
    def run_record(self) -> dict[str, object]:
        """What this acquisition IS, frozen when the camera was armed."""

        record = self._run_record
        if record is None:
            raise RuntimeError("camera run record was not frozen after arm")
        return record

    def prepare(
        self,
        *,
        should_stop: Callable[[], bool] | None = None,
        after_arm: Callable[[], object] | None = None,
    ) -> FiniteCapture:
        """Arm the camera for one finite acquisition.

        ``should_stop`` is the run's cancel: a capture asks it between reads,
        which is the only place a blocking read can be interrupted.  The
        publisher declares run metadata before its first commit; a task may
        finish static geometry from that first frame before declaring it.
        """

        if self.request.repeat <= 0:
            raise ValueError("finite prepare requires request.repeat greater than zero")
        try:
            self._configure_capture()
            capture = self._arm_configured(should_stop=should_stop)
            if after_arm is not None:
                after_arm()
            return capture
        except BaseException as error:
            try:
                self.camera.finish_record_capture()
            except BaseException as cleanup:
                error.add_note(f"camera cleanup also failed: {cleanup}")
            raise

    def execute(self, context: object) -> dict[str, object]:
        """The run: arm, report ready, publish every cycle through the run's context."""

        # The run's instance names the signals and the axes, not this node's type.
        self.instance_id = context.instance_id
        self._generation = context.generation
        if self.aligned:
            return self._execute_aligned(context)
        if self.repeat == 0:
            capture = self.monitor(commit_live=context.commit_live, after_arm=context.record_devices)
            try:
                self._run_record = context.set_run_record(self.run_record)
                context.report_ready()
                while not context.cancel_requested():
                    capture.poll()
            except BaseException as error:
                try:
                    capture.close(drain=False)
                except BaseException as cleanup:
                    error.add_note(f"camera cleanup also failed: {cleanup}")
                raise
            capture.close()
            return {"cycles": capture._revision}
        capture = self.prepare(should_stop=context.cancel_requested, after_arm=context.record_devices)
        try:
            self._run_record = context.set_run_record(self.run_record)
            context.report_ready()
        except BaseException:
            # Stop can arrive between the completed arm and its acknowledgement.
            capture.stopped = True
            capture.close()
            raise

        def commit_cycle(cycle: object, index: int) -> None:
            context.commit_live(
                {CAMERA_FRAMES_OUTPUT.name: _finite_cycle_output(self, cycle, index)}
            )
            # A cycle landed is a repeat played; the bar says so as it happens.
            context.report_progress(
                "Capturing", current=index + 1, total=int(self.repeat)
            )

        capture.read_cycles(commit_cycle)
        return {"cycles": capture.completed_cycles}

    def _execute_aligned(self, context: object) -> dict[str, object]:
        """The run of a measurement that names its trigger lane: arm, report ready, and publish each shot
        the lane's pulse plays as it completes -- until Stop, or ``repeat`` shots."""

        lane = str(self.request.trigger_channel)
        try:
            point = self._configure_for_run()
            buffer_frames = self._buffer_frame_count(point)
            # Every frame is taken as it comes: which shot it is of is found from when it came.
            self.camera.arm(None, source_group_sizes=None, buffer_frame_count=buffer_frames, timeout=float(self.camera.timeout))
            armed_us = time_ns() // 1000
            self._freeze_working_point(self.camera.working_point())
            self._run_record["acquisition"] = {"buffer_frame_count": buffer_frames}
            context.record_devices()
            capture = AlignedCapture(
                self, commit_live=context.commit_live, armed_us=armed_us,
                schedule=lambda known: context.core("pulse_schedule", lane=lane, known=known),
                say=context.report_progress,
            )
        except BaseException as error:
            try:
                self.camera.finish_record_capture()
            except BaseException as cleanup:
                error.add_note(f"camera cleanup also failed: {cleanup}")
            raise
        try:
            self._run_record = context.set_run_record(self.run_record)
            context.report_ready()
            while not context.cancel_requested() and not capture.finished:
                capture.poll()
        except BaseException as error:
            try:
                capture.close(drain=False)
            except BaseException as cleanup:
                error.add_note(f"camera cleanup also failed: {cleanup}")
            raise
        capture.close()
        return {"cycles": capture.published, "incomplete_cycles": capture.incomplete, "frames_dropped": capture.dropped}

    def monitor(self, *, commit_live: Callable[..., object], after_arm: Callable[[], object] | None = None) -> MonitorCapture:
        """Arm a repeat-zero capture that publishes every complete cycle."""

        cycle_size = self.frames_per_cycle
        if self.request.repeat != 0:
            raise ValueError("monitor requires request.repeat equal to zero")
        try:
            point = self._configure_for_run()
            buffer_frames = self._buffer_frame_count(point)
            # The camera's own timeout still governs ARMING -- how long the
            # device may take to become ready is the device's fact.
            timeout = float(self.camera.timeout)
            self.camera.arm(
                None,
                source_group_sizes=None if cycle_size == 1 else (cycle_size,),
                buffer_frame_count=buffer_frames,
                timeout=timeout,
            )
            self._freeze_working_point(self.camera.working_point())
            self._run_record["acquisition"] = {"buffer_frame_count": buffer_frames}
            if after_arm is not None:
                after_arm()
            return MonitorCapture(self.camera, node=self, commit_live=commit_live)
        except BaseException as error:
            try:
                self.camera.finish_record_capture()
            except BaseException as cleanup:
                error.add_note(f"camera cleanup also failed: {cleanup}")
            raise


__all__ = [
    "CAMERA_FRAMES_OUTPUT",
    "describe_camera_frames",
    "frames_snapshot",
    "CameraMeasurementNode",
    "CameraMeasurementRequest",
    "FiniteCapture",
    "MonitorCapture",
    "photoelectron_frame",
]
