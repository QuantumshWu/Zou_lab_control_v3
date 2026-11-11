"""A software trigger over the samples a free-running source already delivers.

Oscilloscope-style locking for slow streams (about 1 kHz sampling); a fast signal
belongs on a scope with a hardware trigger. Each edge gives one trace of the asked
number of samples with the crossing in its middle, so a trace is an ordinary record:
what a plot needs to know of a trigger is its time axis, zero at the crossing.

The crossing lies between two samples, wherever the sampling clock happened to fall.
As an oscilloscope places its trace by that fraction, every trace is taken at the
crossing's own sample phase: linearly between the two samples around each point, so
it passes the level exactly at time zero. Aligning on the first sample past the level
would instead move each trace by up to one sample from the next.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, inf
from typing import NamedTuple

import numpy as np
from zlc_data.units import resolve_unit

from zlc_atom.devices.waveform.contract import WaveformOutput

SLOPES = ("rising", "falling")
MINIMUM_SAMPLES = 2

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


def unit_text(unit: str) -> str:
    """A unit as a person reads it: ``µT``, ``rad/s``, ``m/s²``."""

    symbol = resolve_unit(unit).symbol
    above, below = [], []
    for part in symbol.split("*"):
        base, _, power = part.partition("^")
        exponent = int(power) if power else 1
        (above if exponent > 0 else below).append(base + {1: "", 2: "²", 3: "³"}.get(abs(exponent), f"^{abs(exponent)}"))
    return "/".join(["·".join(above) or "1", *below])


def channel_value(output: WaveformOutput, label: str) -> str:
    """How the form names one channel of a source."""

    return f"{output.name}/{label}"


def channel_choices(outputs: Sequence[WaveformOutput], time_basis: str) -> list[list[str]]:
    """Every channel with its unit, the unit its trigger level is typed in."""

    note = " - timed by host receipt, limited precision" if time_basis == "host_receive" else ""
    return [[channel_value(output, label), f"{output.name} {label} ({unit_text(output.unit)}){note}"]
            for output in outputs for label in output.channel_labels]


def channel_output(channel: object, outputs: Sequence[WaveformOutput]) -> WaveformOutput | None:
    """The quantity a form's channel belongs to, when the source has that channel."""

    for output in outputs:
        if any(channel_value(output, label) == channel for label in output.channel_labels):
            return output
    return None


@dataclass(frozen=True)
class WaveformTrigger:
    """The trigger of one run, frozen: which record column, at what level, which way."""

    column: int
    level: float
    slope: str
    #: How the status names this channel, and its unit.
    name: str
    unit: str
    #: Samples of each trace.
    samples: int


@dataclass(frozen=True)
class TriggerSettings:
    """The operator's trigger, as the form states it."""

    channel: str
    level: float
    slope: str
    samples: int

    def __post_init__(self) -> None:
        if self.slope not in SLOPES:
            raise ValueError(f"trigger slope must be one of {SLOPES}, not {self.slope!r}")
        if int(self.samples) < MINIMUM_SAMPLES:
            raise ValueError(f"a trigger trace needs at least {MINIMUM_SAMPLES} samples")
        if not str(self.channel).strip():
            raise ValueError("a trigger needs its channel")

    def resolve(self, outputs: Sequence[WaveformOutput]) -> WaveformTrigger:
        """This trigger on this source's columns, or why it cannot be."""

        for output in outputs:
            for label, column in zip(output.channel_labels, output.columns):
                if channel_value(output, label) == self.channel:
                    return WaveformTrigger(column, float(self.level), self.slope, f"{output.name} {label}", unit_text(output.unit), int(self.samples))
        offered = [channel_value(output, label) for output in outputs for label in output.channel_labels]
        raise ValueError(f"the source has no channel {self.channel!r}; it offers {offered}")


class TriggerDetector:
    """One channel's samples in, in order; which of them are trigger edges out.

    Every edge of an armed signal fires, and arming runs over every sample.  What
    was scanned leaves nothing behind but the running extremes and the last sample.
    """

    def __init__(self, trigger: WaveformTrigger, sample_interval_seconds: float) -> None:
        # Falling is rising on the negated signal, the level and the extreme with it.
        self._sign = 1.0 if trigger.slope == "rising" else -1.0
        self._level = self._sign * trigger.level
        self._run_samples = max(1, ceil(EXTREMES_WINDOW_SECONDS / float(sample_interval_seconds) / _EXTREMES_RUNS))
        self._run_minima: deque[float] = deque(maxlen=_EXTREMES_RUNS)
        self._run_minimum = inf
        self._run_filled = 0
        self._armed = False
        self._last: float | None = None
        #: The range seen since ``take_range``: what a waiting status reports.
        self._low, self._high = inf, -inf

    def take_range(self) -> tuple[float, float] | None:
        """The channel's range since the last call (None: no finite sample came)."""

        seen = None if self._low > self._high else (self._low, self._high)
        self._low, self._high = inf, -inf
        return seen

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

    def scan(self, samples: np.ndarray) -> np.ndarray:
        """One bool per sample: it is a trigger edge."""

        marks = np.zeros(samples.size, dtype=bool)
        if not samples.size:
            return marks
        fresh = np.asarray(samples, dtype=np.float64)
        if np.isfinite(fresh).any():
            self._low = min(self._low, float(np.nanmin(fresh)))
            self._high = max(self._high, float(np.nanmax(fresh)))
        fresh = self._sign * fresh
        level = self._level
        arming = np.flatnonzero((fresh < 0.5 * (level + self._running_minimum(fresh))) & (fresh < level))
        previous = np.concatenate(((np.nan if self._last is None else self._last,), fresh[:-1]))
        edges = np.flatnonzero((previous < level) & (fresh >= level))
        self._last = float(fresh[-1])
        position = 0
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
            marks[index] = True
        return marks


class Traces(NamedTuple):
    """What one read completed: ``samples`` ``(traces, samples, columns)``, each trace's
    crossing time and its crossing sample's host receipt, and how many edges it found."""

    samples: np.ndarray
    times: np.ndarray
    received: np.ndarray
    edges: int


class TraceAssembler:
    """Samples in, in order; a trace out for every edge, once its last sample has come.

    It keeps only the newest samples a trace can still need, so it holds a bounded
    number however long the signal stays quiet. Every edge gives a trace, and traces
    overlap when edges come closer than a trace is long. An edge before there are
    enough samples ahead of it gives none: every trace has the same length.
    """

    def __init__(self, trigger: WaveformTrigger, sample_interval_seconds: float) -> None:
        self._trigger = trigger
        self._interval = float(sample_interval_seconds)
        self._detector = TriggerDetector(trigger, sample_interval_seconds)
        self._length = trigger.samples
        self._before = trigger.samples // 2
        self._held: np.ndarray | None = None
        self._held_times = np.empty(0)
        self._held_received = np.empty(0, dtype=np.int64)
        #: The stream position of the first held sample.
        self._first = 0
        #: The stream positions of the edges still waiting for samples.
        self._pending: deque[int] = deque()

    @property
    def before(self) -> int:
        """Samples of a trace ahead of its crossing."""

        return self._before

    def take_range(self) -> tuple[float, float] | None:
        return self._detector.take_range()

    def push(self, samples: np.ndarray, times: np.ndarray, received: np.ndarray) -> Traces:
        """``samples`` ``(samples, columns)``, each one's source time and host receipt."""

        held = samples if self._held is None else np.concatenate((self._held, samples))
        held_times = np.concatenate((self._held_times, times))
        held_received = np.concatenate((self._held_received, received))
        column = self._trigger.column
        edges = np.flatnonzero(self._detector.scan(samples[:, column])) + self._first + held.shape[0] - samples.shape[0]
        self._pending.extend(int(edge) for edge in edges)
        total = self._first + held.shape[0]
        # Each point of a trace lies this fraction of the way between two samples, so a
        # trace reads one sample past its length.
        span = self._length + 1
        traces, crossings, receipts = [], [], []
        while self._pending and self._pending[0] - 1 - self._before + span <= total:
            edge = self._pending.popleft()
            start = edge - 1 - self._before
            if start < 0:
                continue
            at, part = edge - self._first, start - self._first
            below, above = float(held[at - 1, column]), float(held[at, column])
            fraction = (self._trigger.level - below) / (above - below)
            traces.append((1.0 - fraction) * held[part:part + self._length] + fraction * held[part + 1:part + span])
            crossings.append(held_times[at - 1] + fraction * self._interval)
            receipts.append(held_received[at])
        # What is still pending starts inside the newest span.
        self._held, self._held_times, self._held_received = held[-span:], held_times[-span:], held_received[-span:]
        self._first = total - self._held.shape[0]
        return Traces(np.stack(traces) if traces else np.empty((0, self._length, samples.shape[1]), dtype=samples.dtype),
                      np.array(crossings, dtype=np.float64), np.array(receipts, dtype=np.int64), edges.size)


__all__ = ["EXTREMES_WINDOW_SECONDS", "MINIMUM_SAMPLES", "SLOPES", "TraceAssembler", "Traces", "TriggerDetector", "TriggerSettings", "WaveformTrigger",
           "channel_choices", "channel_output", "channel_value", "unit_text"]
