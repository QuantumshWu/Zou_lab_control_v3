"""A software trigger over the records a free-running source already delivers.

Oscilloscope-style locking for slow streams (about 1 kHz sampling, records of
tens of points); a fast signal belongs on a scope with a hardware trigger.  The
stream goes in as the source's records, cut however the source cuts them, and
what comes out is one record per trigger: the same raw samples, never
resampled, with the sub-sample time of the level crossing beside them.

There is no dead time: a scope's is a limit of its hardware.  Every edge fires,
an edge inside the record before it included, so records overlap and the same
sample can sit in several, each record its own copy.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, floor, inf

import numpy as np

from zlc_atom.devices.waveform.contract import WaveformOutput, WaveformRecord

SLOPES = ("rising", "falling")

#: The running extremes look back this far.  To fire rising, the signal must
#: have passed halfway from the level to its running minimum since the previous
#: firing (falling: to its running maximum).  At a centred level that is 25% of
#: the peak-to-peak; it scales with the signal, so it needs no knob, and a level
#: near an extreme asks for nothing the signal does not reach.  The look-back is
#: bounded so a large transient cannot disable arming for good.  What that limits:
#: a signal that stays on one side of the level for this whole time has no extreme
#: left on the other side, so with noise on it, a slope slower than a period of
#: about twice this long can fire on the wrong edge.
EXTREMES_WINDOW_SECONDS = 2.0
#: The look-back is kept as this many minima of fixed runs of samples, so what a
#: sample is compared with does not depend on how the stream was cut into reads.
_EXTREMES_RUNS = 16

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

    Whether the signal is armed runs over every sample, whatever records are
    being filled.  An edge within the first pre-trigger share of the stream has
    no record to go in, and it arms and fires like any other otherwise.

    One ordinal gap in the stream ends what lies before it: the records being
    filled are lost (``broken_records``), never stitched across the gap, and the
    pre-trigger share refills before the next record.
    """

    def __init__(self, trigger: WaveformTrigger, sample_interval_seconds: float) -> None:
        self._trigger = trigger
        self._interval = float(sample_interval_seconds)
        # Falling is rising on the negated signal, the level and the extreme with it.
        self._sign = 1.0 if trigger.slope == "rising" else -1.0
        self._level = self._sign * trigger.level
        self._run_samples = max(1, ceil(EXTREMES_WINDOW_SECONDS / self._interval / _EXTREMES_RUNS))
        self._run_minima: deque[float] = deque(maxlen=_EXTREMES_RUNS)
        self._run_minimum = inf
        self._run_filled = 0
        self._armed = False
        self._expected_ordinal: int | None = None
        self._produced = 0
        #: Records a gap in the stream cut short: the one way a trigger is lost.
        self.broken_records = 0
        self._clear_stream()

    def _clear_stream(self) -> None:
        self._samples: np.ndarray | None = None
        self._times: np.ndarray | None = None
        #: (index of the trigger sample, crossing time, offset) of each record being filled.
        self._pending: deque[tuple[int, float, float]] = deque()

    def feed(self, records: Sequence[WaveformRecord]) -> list[TriggeredRecord]:
        """The records these complete, in order; the stream continues in the next call."""

        fired: list[TriggeredRecord] = []
        run: list[WaveformRecord] = []
        for record in records:
            if self._expected_ordinal is not None and record.source_ordinal != self._expected_ordinal:
                self._scan(run, fired)
                run = []
                self.broken_records += len(self._pending)
                self._clear_stream()
            self._expected_ordinal = record.source_ordinal + 1
            run.append(record)
        self._scan(run, fired)
        return fired

    def _running_minimum(self, fresh: np.ndarray) -> np.ndarray:
        """The signal's minimum over the look-back, up to and including each sample."""

        minima = np.empty(fresh.size)
        at = 0
        while at < fresh.size:
            part = fresh[at:at + self._run_samples - self._run_filled]
            before = min(self._run_minimum, min(self._run_minima, default=inf))
            minima[at:at + part.size] = np.fmin(before, np.fmin.accumulate(part))
            self._run_minimum = float(np.fmin(self._run_minimum, np.fmin.reduce(part)))
            self._run_filled += part.size
            if self._run_filled == self._run_samples:
                self._run_minima.append(self._run_minimum)
                self._run_minimum, self._run_filled = inf, 0
            at += part.size
        return minima

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
        level = self._level
        fresh = signal[scanned:]
        arming = np.flatnonzero((fresh < 0.5 * (level + self._running_minimum(fresh))) & (fresh < level)) + scanned
        first = max(scanned, 1)
        edges = np.flatnonzero((signal[first - 1:-1] < level) & (signal[first:] >= level)) + first
        position = scanned
        # Jumps from one condition to the next, so the cost is per trigger
        # however many times noise crosses the level.
        while True:
            if not self._armed:
                found = int(np.searchsorted(arming, position))
                if found == arming.size:
                    break
                position = int(arming[found]) + 1
                self._armed = True
            found = int(np.searchsorted(edges, position))
            if found == edges.size:
                break
            index = int(edges[found])
            self._armed = False
            position = index + 1
            if index >= trigger.pre_samples:
                share = (level - signal[index - 1]) / (signal[index] - signal[index - 1])
                before, at = self._times[index - 1], self._times[index]
                crossing = float(before + share * (at - before))
                self._pending.append((index, crossing, crossing - float(at)))
        length = signal.size
        while self._pending and self._pending[0][0] - trigger.pre_samples + trigger.record_samples <= length:
            (index, crossing, offset) = self._pending.popleft()
            start = index - trigger.pre_samples
            fired.append(TriggeredRecord(
                self._samples[start:start + trigger.record_samples], self._produced, crossing,
                run[-1].host_received_at_ns, crossing_offset_seconds=offset,
            ))
            self._produced += 1
        # Keep what the records being filled, the next pre-trigger share and the
        # next crossing's previous sample can still need.
        keep = max(0, length - max(trigger.pre_samples, 1))
        if self._pending:
            keep = min(keep, self._pending[0][0] - trigger.pre_samples)
            self._pending = deque((index - keep, crossing, offset) for index, crossing, offset in self._pending)
        self._samples = self._samples[keep:].copy()
        self._times = self._times[keep:].copy()


__all__ = ["EXTREMES_WINDOW_SECONDS", "SLOPES", "TriggerDetector", "TriggerSettings", "TriggeredRecord",
           "WaveformTrigger", "channel_choices", "channel_value"]
