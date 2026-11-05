"""A software trigger over the samples a free-running source already delivers.

Oscilloscope-style locking for slow streams (about 1 kHz sampling); a fast signal
belongs on a scope with a hardware trigger.  It holds nothing and changes nothing:
every sample is published as it arrives, and an edge is a mark on the event it fell
on, which is where a board's view may stand (``RunContext.commit_events``).  Each
view then ends on an edge, so the phase on screen is the same at every refresh.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, inf

import numpy as np
from zlc_data.units import resolve_unit

from zlc_atom.devices.waveform.contract import WaveformOutput

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


@dataclass(frozen=True)
class TriggerSettings:
    """The operator's trigger, as the form states it."""

    channel: str
    level: float
    slope: str

    def __post_init__(self) -> None:
        if self.slope not in SLOPES:
            raise ValueError(f"trigger slope must be one of {SLOPES}, not {self.slope!r}")
        if not str(self.channel).strip():
            raise ValueError("a trigger needs its channel")

    def resolve(self, outputs: Sequence[WaveformOutput]) -> WaveformTrigger:
        """This trigger on this source's columns, or why it cannot be."""

        for output in outputs:
            for label, column in zip(output.channel_labels, output.columns):
                if channel_value(output, label) == self.channel:
                    return WaveformTrigger(column, float(self.level), self.slope, f"{output.name} {label}", unit_text(output.unit))
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


__all__ = ["EXTREMES_WINDOW_SECONDS", "SLOPES", "TriggerDetector", "TriggerSettings", "WaveformTrigger",
           "channel_choices", "channel_output", "channel_value", "unit_text"]
