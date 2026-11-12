"""Which trigger of the pulse each camera frame is.

A camera that groups its frames by counting them is right only while it receives exactly the
triggers the pulse sends: one more -- a glitch on the line -- and every later shot is paired one
frame off, with nothing to say so.  The board knows when its trigger lane rises in every shot it
plays (``pulse_schedule``); the camera stamps every frame with its own clock.  So a frame is the
trigger whose expected time it arrives at, and a frame at no trigger's time is not of the pulse.

Only intervals are compared, each inside one clock: the camera's stamps against each other, the
schedule's edges against each other.  Neither the moment the board began nor the offset between
the two clocks is needed, and the difference in their rates is taken out at every matched frame.
The one thing that must hold is that the first frame is known: the first trigger of the play when
the camera was armed before it, else the trigger its spacing to the next frames picks out.

Nothing here is a tuned number.  A frame belongs to a trigger when it is nearer to it than half
the smallest gap between any two triggers of the schedule: closer than that, no other trigger can
claim it.  Of two frames near one trigger the nearer is the trigger's; a frame that came before
its trigger's time keeps the place only until a nearer one could no longer come -- at once when
the camera could not have exposed again that soon (``separation``, its exposure time).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Mapping, Sequence, TypeVar

Frame = TypeVar("Frame")


class AlignmentError(RuntimeError):
    """The frames cannot be told apart by time; the sentence says what to do."""


@dataclass(frozen=True)
class TriggerSchedule:
    """What the board plays, for one trigger lane (the core's ``pulse_schedule`` answer)."""

    revision: int
    state: str
    lane: str
    #: Per scan point: when the lane rises after the point begins, and the point's length (s).
    points: tuple[tuple[tuple[float, ...], float], ...]
    #: Shots of each point before the next; 0: the first point, without end.
    run_repeats: int
    #: Sweeps of the table; 0: without end.
    scan_repeats: int
    command_id: int | None = None
    sent_us: int = 0

    @classmethod
    def from_answer(cls, answer: Mapping[str, Any]) -> "TriggerSchedule":
        points = tuple(
            (tuple(float(edge) for edge in point["rising"]), float(point["duration"]))
            for point in answer.get("points") or ()
        )
        command = answer.get("command_id")
        return cls(
            revision=int(answer["revision"]), state=str(answer["state"]), lane=str(answer.get("lane") or ""),
            points=points, run_repeats=int(answer.get("run_repeats") or 0), scan_repeats=int(answer.get("scan_repeats") or 0),
            command_id=None if command is None else int(command), sent_us=int(answer.get("sent_us") or 0),
        )

    @property
    def playing(self) -> bool:
        return self.state == "playing"

    @property
    def frames(self) -> int:
        """How often the lane rises in one shot: the frames of one cycle."""

        counts = {len(rising) for rising, _duration in self.points}
        if len(counts) != 1:
            raise AlignmentError(
                f"{self.lane} rises a different number of times at different scan points ({sorted(counts)}): "
                "a camera cycle has one size"
            )
        count = counts.pop()
        if count == 0:
            raise AlignmentError(f"{self.lane} never rises in this pulse: it triggers no frame")
        return count

    @property
    def cyclic(self) -> bool:
        """One shot repeated: every shot has the same triggers, so any shot can be the first."""

        return self.run_repeats == 0 or len(self.points) == 1

    def shot(self, index: int) -> tuple[tuple[float, ...], float] | None:
        """The triggers and length of the ``index``-th shot played; None once the play has ended."""

        if not self.points:
            return None
        if self.run_repeats == 0:
            return self.points[0]
        sweep, within = divmod(index, len(self.points) * self.run_repeats)
        if self.scan_repeats and sweep >= self.scan_repeats:
            return None
        return self.points[within // self.run_repeats]

    @property
    def tolerance(self) -> float:
        """Half the smallest gap between two triggers anywhere in the play (s)."""

        gaps = [later - earlier for rising, _ in self.points for earlier, later in zip(rising, rising[1:])]
        # Between shots: the shortest tail of any point before the earliest first trigger of any.
        tail = min(duration - rising[-1] for rising, duration in self.points if rising)
        head = min(rising[0] for rising, _ in self.points if rising)
        gaps.append(tail + head)
        smallest = min(gaps)
        if smallest <= 0:
            raise AlignmentError(f"two triggers of {self.lane} fall at the same time: frames cannot be told apart")
        return smallest / 2


@dataclass(frozen=True)
class Shot(Generic[Frame]):
    """One shot's frames in trigger order; None where its trigger brought no frame."""

    index: int
    frames: tuple[Frame | None, ...]

    @property
    def complete(self) -> bool:
        return all(frame is not None for frame in self.frames)


@dataclass(frozen=True)
class Extra(Generic[Frame]):
    """A frame that is in no shot, and why: it is no trigger of the pulse, or the camera saw
    only the end of its shot."""

    frame: Frame
    reason: str


class TriggerAlignment(Generic[Frame]):
    """Frames, as they arrive with their camera times, sorted into the shots of one play."""

    def __init__(self, schedule: TriggerSchedule, *, from_first_trigger: bool, separation: float = 0.0) -> None:
        self.schedule = schedule
        self.frames = schedule.frames
        self.tolerance = schedule.tolerance
        #: The least time between two frames of this camera (s): its exposure. Unknown is 0, and then a
        #: frame early for its trigger is only settled by the next frame.
        self.separation = float(separation)
        if not from_first_trigger and not schedule.cyclic:
            raise AlignmentError(
                "this scan was already playing when the camera started, and its shots differ from point to point: "
                "start the camera measurement before the pulse"
            )
        self._first = from_first_trigger
        #: The camera time the current shot began at, once one frame is known.
        self._start: float | None = None
        self._index = 0
        self._filled: list[Frame | None] = [None] * self.frames
        #: Per trigger of the shot in hand, while its frame may still be replaced by a nearer one: how far
        #: before the trigger's time that frame came (s), and that time.
        self._open: list[tuple[float, float] | None] = [None] * self.frames
        #: The shot the alignment was found in the middle of: its earlier frames were never seen.
        self._partial = False
        #: Frames kept until their spacing says which triggers they are (a camera started mid-play).
        self._waiting: list[tuple[float, Frame]] = []
        self.ended = False

    # ------------------------------------------------------------------ finding the first frame
    def _hypotheses(self) -> list[int]:
        """The triggers the first waiting frame can be, given how the waiting frames are spaced."""

        rising, duration = self.schedule.points[0]
        first = self._waiting[0][0]
        fits = []
        for slot in range(self.frames):
            start = first - rising[slot]
            taken: set[tuple[int, int]] = set()
            for time, _frame in self._waiting:
                shots = round((time - start - rising[0]) / duration)
                # The nearest trigger of the shot the frame falls in, or of its neighbours.
                best = min(
                    ((abs(time - (start + shot * duration + edge)), shot, at)
                     for shot in (shots - 1, shots, shots + 1) for at, edge in enumerate(rising)),
                )
                if best[0] > self.tolerance or (best[1], best[2]) in taken:
                    break
                taken.add((best[1], best[2]))
            else:
                fits.append(slot)
        return fits

    def _find_first(self, time: float, frame: Frame) -> list[Shot[Frame] | Extra[Frame]]:
        self._waiting.append((time, frame))
        out: list[Shot[Frame] | Extra[Frame]] = []
        while self._waiting:
            fits = self._hypotheses()
            if not fits:
                # No trigger explains the first frame beside the others: it is not of the pulse.
                out.append(Extra(self._waiting.pop(0)[1], "its spacing to the next frames matches no trigger"))
                continue
            if len(fits) == 1 and (self.frames == 1 or len(self._waiting) > 1):
                slot = fits[0]
                waiting, self._waiting = self._waiting, []
                self._start = waiting[0][0] - self.schedule.points[0][0][slot]
                self._partial = slot > 0
                for held_time, held in waiting:
                    out.extend(self._place(held_time, held))
                return out
            if len(self._waiting) > 2 * self.frames:
                raise AlignmentError(
                    f"the {self.frames} triggers of {self.schedule.lane} are evenly spaced, so a frame cannot be told "
                    "from the next by its time: start the camera measurement before the pulse"
                )
            return out
        return out

    # ------------------------------------------------------------------ sorting
    def accept(self, time: float, frame: Frame) -> list[Shot[Frame] | Extra[Frame]]:
        """One frame at camera time ``time`` (s): the shots it completes and the frames it shows to be extra."""

        if self._start is None:
            if not self._first:
                return self._find_first(time, frame)
            self._start = time - self.schedule.points[0][0][0]
        return self._place(time, frame)

    def _place(self, time: float, frame: Frame) -> list[Shot[Frame] | Extra[Frame]]:
        out: list[Shot[Frame] | Extra[Frame]] = []
        while True:
            shot = self.schedule.shot(self._index)
            if shot is None:
                self.ended = True
                out.append(Extra(frame, "the play had sent its last trigger"))
                return out
            rising, duration = shot
            assert self._start is not None
            # A frame that came early for its trigger is settled once time has passed where a nearer one could be.
            for at, held in enumerate(self._open):
                if held is not None and time >= held[1] + held[0]:
                    self._open[at] = None
            distance, slot = min((abs(time - (self._start + edge)), at) for at, edge in enumerate(rising))
            if distance <= self.tolerance:
                expected = self._start + rising[slot]
                held = self._open[slot]
                if self._filled[slot] is not None and (held is None or distance >= held[0]):
                    out.append(Extra(frame, "its trigger already has a frame"))
                elif self._filled[slot] is not None:
                    out.append(Extra(self._filled[slot], "a later frame is nearer its trigger"))
                if self._filled[slot] is None or (held is not None and distance < held[0]):
                    self._filled[slot] = frame
                    early = expected - time
                    self._open[slot] = (early, expected) if early > 0 and 2 * early >= self.separation else None
                    if self._open[slot] is None:
                        # The shot began where this frame says it did: the two clocks' rates differ, never by a trigger.
                        self._start = time - rising[slot]
                if all(item is not None for item in self._filled) and not any(self._open):
                    out.extend(self._close(duration))
                return out
            if time > self._start + rising[-1] + self.tolerance:
                # Past every trigger of this shot: what it lacks never came.
                out.extend(self._close(duration))
                continue
            out.append(Extra(frame, "it came between two triggers"))
            if all(item is not None for item in self._filled) and not any(self._open):
                out.extend(self._close(duration))
            return out

    def _close(self, duration: float) -> list[Shot[Frame] | Extra[Frame]]:
        shot = Shot(self._index, tuple(self._filled))
        partial, self._partial = self._partial, False
        assert self._start is not None
        self._start += duration
        self._index += 1
        self._filled = [None] * self.frames
        self._open = [None] * self.frames
        # A shot no frame of was seen says nothing; the one the camera started in the middle of is no
        # shot, and each frame of it that came is dropped as every frame outside a shot is (said so).
        if partial:
            return [Extra(frame, "the camera started in the middle of its shot") for frame in shot.frames if frame is not None]
        return [shot] if any(frame is not None for frame in shot.frames) else []

    def flush(self) -> list[Shot[Frame] | Extra[Frame]]:
        """The play is over (or its last frame is overdue): the shot in hand as it stands."""

        out: list[Shot[Frame] | Extra[Frame]] = [Extra(frame, "the play ended before its place was known") for _time, frame in self._waiting]
        self._waiting = []
        shot = self.schedule.shot(self._index)
        if self._start is not None and shot is not None and any(frame is not None for frame in self._filled):
            out.extend(self._close(shot[1]))
        return out

    @property
    def overdue_after(self) -> float | None:
        """How long after its last frame the shot in hand is over (s); None while no shot is in hand."""

        shot = self.schedule.shot(self._index)
        if self._start is None or shot is None or not any(frame is not None for frame in self._filled):
            return None
        return shot[1]


def frame_time(record: Any) -> float:
    """A frame's time on the clock that stamped it (s): the camera's own where it stamps frames."""

    seconds, micro = getattr(record, "timestamp_seconds", None), getattr(record, "timestamp_microseconds", None)
    if seconds is not None and micro is not None:
        return int(seconds) + int(micro) / 1e6
    return int(record.host_received_at_ns) / 1e9


def shot_order(shots: Sequence[Shot[Any]]) -> list[int]:
    return [shot.index for shot in shots]


__all__ = ["AlignmentError", "Extra", "Shot", "TriggerAlignment", "TriggerSchedule", "frame_time"]
