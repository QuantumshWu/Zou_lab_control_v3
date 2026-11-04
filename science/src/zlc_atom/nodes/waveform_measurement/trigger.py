"""A software trigger over the records a free-running source already delivers.

Oscilloscope-style locking for slow streams (about 1 kHz sampling, records of
tens of points); a fast signal belongs on a scope with a hardware trigger.  The
stream goes in as the source's records, cut however the source cuts them, and
what comes out is one record per trigger: the same raw samples, never
resampled, with the sub-sample time of the level crossing beside them.

Re-arming is the only holdoff: a trigger may fire once the record before it is
complete and the pre-trigger share has refilled, so two records never share a
sample.  A record shorter than the signal period catches every edge; a record
of one period about every second edge.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, floor

import numpy as np

from zlc_atom.devices.waveform.contract import WaveformOutput, WaveformRecord

SLOPES = ("rising", "falling")

#: To fire rising, the signal must first have been this share of its running
#: peak-to-peak below the level since re-arm (mirrored for falling), which is
#: what makes a noisy crossing of the WRONG slope unable to fire.  At 1 kHz with
#: 30-sample records and no such condition, wrong-edge records reach 7.7% at
#: 11 Hz with 5% noise, 32% at 4.7 Hz and 50% at 1.3 Hz; with this share they
#: are 0% in all three.  It scales with the signal, so it needs no knob.
HYSTERESIS_SHARE = 0.25

#: The running peak-to-peak looks back this far (or two records, if longer): it
#: must span a period of the slowest signal this trigger serves (1 Hz), and an
#: amplitude that drops is forgotten after this long.
PEAK_WINDOW_SECONDS = 2.0

#: No trigger before the peak-to-peak has seen this much of the stream: with
#: only the first samples in hand it is the noise, and hysteresis scaled to the
#: noise would let the first record fire on a wrong edge.
SETTLE_SECONDS = 0.25

#: The time axis travels as JSON coordinates in each output's schema, about
#: 25 bytes a point, inside one 16 MiB channel frame.
_MAX_SCHEMA_COORDINATES = 500_000


@dataclass(frozen=True)
class WaveformTrigger:
    """The trigger of one run, frozen: which record column, where and how long."""

    column: int
    level: float
    slope: str
    record_samples: int
    pre_samples: int


@dataclass(frozen=True, eq=False)
class TriggeredRecord(WaveformRecord):
    """One triggered record: ``time_seconds`` is the interpolated level crossing.

    ``crossing_offset_seconds`` (at most 0, within one sample) is where that
    crossing lies from the sample at time zero of the record's time axis.
    """

    crossing_offset_seconds: float = 0.0


def channel_value(output: WaveformOutput, label: str) -> str:
    """How the form names one channel of a source."""

    return f"{output.name}/{label}"


def channel_choices(outputs: Sequence[WaveformOutput], time_basis: str) -> list[list[str]]:
    """Every channel with its unit, the unit its trigger level is typed in."""

    note = " - timed by host receipt, limited precision" if time_basis == "host_receive" else ""
    return [[channel_value(output, label), f"{output.name} {label} ({output.unit}){note}"]
            for output in outputs for label in output.channel_labels]


@dataclass(frozen=True)
class TriggerSettings:
    """The operator's trigger, as the form states it."""

    channel: str
    level: float
    slope: str
    record_samples: int
    position_percent: float

    def __post_init__(self) -> None:
        if self.slope not in SLOPES:
            raise ValueError(f"trigger slope must be one of {SLOPES}, not {self.slope!r}")
        if not str(self.channel).strip():
            raise ValueError("a trigger needs its channel")
        if int(self.record_samples) < 2:
            raise ValueError("a triggered record needs at least 2 samples")
        if not 0.0 <= float(self.position_percent) <= 100.0:
            raise ValueError("the pre-trigger share must lie between 0 and 100 %")

    def output_name(self) -> str:
        return str(self.channel).partition("/")[0]

    def resolve(self, outputs: Sequence[WaveformOutput]) -> WaveformTrigger:
        """This trigger on this source's columns, or why it cannot be."""

        samples = int(self.record_samples)
        if samples * len(outputs) > _MAX_SCHEMA_COORDINATES:
            raise ValueError(
                f"a triggered record of {samples} samples is too long for this source: at most "
                f"{_MAX_SCHEMA_COORDINATES // len(outputs)} samples with {len(outputs)} outputs"
            )
        for output in outputs:
            for label, column in zip(output.channel_labels, output.columns):
                if channel_value(output, label) == self.channel:
                    pre = min(samples - 1, floor(samples * float(self.position_percent) / 100.0))
                    return WaveformTrigger(column, float(self.level), self.slope, samples, pre)
        offered = [channel_value(output, label) for output in outputs for label in output.channel_labels]
        raise ValueError(f"the source has no channel {self.channel!r}; it offers {offered}")


class TriggerDetector:
    """The stream in, triggered records out; holds at most about one record of it.

    One ordinal gap in the stream ends what lies before it: a record being
    filled is lost (``broken_records``), never stitched across the gap, and the
    pre-trigger share refills before the next trigger may fire.
    """

    def __init__(self, trigger: WaveformTrigger, sample_interval_seconds: float) -> None:
        self._trigger = trigger
        self._interval = float(sample_interval_seconds)
        # Falling is rising on the negated signal, the level and extremes with it.
        self._sign = 1.0 if trigger.slope == "rising" else -1.0
        self._level = self._sign * trigger.level
        self._peak_window = ceil(max(PEAK_WINDOW_SECONDS / self._interval, 2 * trigger.record_samples))
        self._peaks: deque[tuple[int, float, float]] = deque()
        self._peak_samples = 0
        self._expected_ordinal: int | None = None
        self._produced = 0
        #: Records a gap in the stream cut short: the one way a trigger is lost.
        self.broken_records = 0
        self._clear_stream()
        # Indices are into the kept tail followed by the block being read.
        self._rearm = max(trigger.pre_samples, ceil(SETTLE_SECONDS / self._interval))

    def _clear_stream(self) -> None:
        self._samples: np.ndarray | None = None
        self._times: np.ndarray | None = None
        self._rearm = self._trigger.pre_samples
        self._seen_beyond = False
        #: (index of the trigger sample, crossing time, offset) of the record being filled.
        self._pending: tuple[int, float, float] | None = None

    def feed(self, records: Sequence[WaveformRecord]) -> list[TriggeredRecord]:
        """The records these complete, in order; the stream continues in the next call."""

        fired: list[TriggeredRecord] = []
        run: list[WaveformRecord] = []
        for record in records:
            if self._expected_ordinal is not None and record.source_ordinal != self._expected_ordinal:
                self._scan(run, fired)
                run = []
                self.broken_records += self._pending is not None
                self._clear_stream()
            self._expected_ordinal = record.source_ordinal + 1
            run.append(record)
        self._scan(run, fired)
        return fired

    def _beyond(self, fresh: np.ndarray) -> float:
        """The value the signal must have passed since re-arm: the hysteresis.

        ``HYSTERESIS_SHARE`` of the running peak-to-peak from the level.  Once the
        look-back is full, its far extreme is known and the value goes no further
        than halfway to it: a level within a quarter of the signal's edge would
        otherwise ask for a value the signal never reaches, and wait forever on a
        perfectly good signal.  Before that, an extreme not yet seen would pull the
        value up against the level, where noise can pass it.
        """

        low, high = float(np.fmin.reduce(fresh)), float(np.fmax.reduce(fresh))
        if low == low:  # not all missing samples
            self._peaks.append((fresh.size, low, high))
            self._peak_samples += fresh.size
            while self._peak_samples - self._peaks[0][0] >= self._peak_window:
                self._peak_samples -= self._peaks.popleft()[0]
        if not self._peaks:
            return float("-inf")
        low = min(item[1] for item in self._peaks)
        high = max(item[2] for item in self._peaks)
        beyond = self._level - HYSTERESIS_SHARE * (high - low)
        return max(beyond, 0.5 * (self._level + low)) if self._peak_samples >= self._peak_window else beyond

    def _scan(self, run: list[WaveformRecord], fired: list[TriggeredRecord]) -> None:
        if not run:
            return
        trigger = self._trigger
        held = () if self._samples is None else (self._samples,)
        held_times = () if self._times is None else (self._times,)
        scanned = 0 if self._samples is None else self._samples.shape[0]
        self._samples = np.concatenate([*held, *(record.samples for record in run)])
        self._times = np.concatenate([*held_times, *(
            record.time_seconds + self._interval * np.arange(record.samples.shape[0]) for record in run)])
        signal = self._sign * self._samples[:, trigger.column].astype(np.float64)
        beyond = self._beyond(signal[scanned:])
        level = self._level
        first = max(scanned, 1)
        edges = np.flatnonzero((signal[first - 1:-1] < level) & (signal[first:] >= level)) + first
        passed = np.flatnonzero(signal[scanned:] < beyond) + scanned
        length = signal.size
        position = max(scanned, self._rearm)
        # Jumps from one condition to the next, so the cost is per trigger
        # however many times noise crosses the level.
        while True:
            if self._pending is not None:
                (index, crossing, offset) = self._pending
                start = index - trigger.pre_samples
                if start + trigger.record_samples > length:
                    break
                fired.append(TriggeredRecord(
                    self._samples[start:start + trigger.record_samples], self._produced, crossing,
                    run[-1].host_received_at_ns, crossing_offset_seconds=offset,
                ))
                self._produced += 1
                self._pending = None
            if not self._seen_beyond:
                found = int(np.searchsorted(passed, position))
                if found == passed.size:
                    break
                position = int(passed[found]) + 1
                self._seen_beyond = True
            found = int(np.searchsorted(edges, max(position, self._rearm)))
            if found == edges.size:
                break
            index = int(edges[found])
            share = (level - signal[index - 1]) / (signal[index] - signal[index - 1])
            before, at = self._times[index - 1], self._times[index]
            crossing = float(before + share * (at - before))
            self._pending = (index, crossing, crossing - float(at))
            self._rearm = position = index + trigger.record_samples
            self._seen_beyond = False
        # Keep what the record being filled, the next pre-trigger share and the
        # next crossing's previous sample can still need.
        keep = max(0, length - max(trigger.pre_samples, 1))
        if self._pending is not None:
            keep = min(keep, self._pending[0] - trigger.pre_samples)
            self._pending = (self._pending[0] - keep, *self._pending[1:])
        self._rearm -= keep
        self._samples = self._samples[keep:].copy()
        self._times = self._times[keep:].copy()


__all__ = ["HYSTERESIS_SHARE", "SLOPES", "TriggerDetector", "TriggerSettings", "TriggeredRecord",
           "WaveformTrigger", "channel_choices", "channel_value"]
