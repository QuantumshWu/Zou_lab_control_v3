"""A simulated SMC100 on a simulated serial line: the lowest layer only.

The virtual stage is the REAL driver (``newport_smc100.Smc100Axis``) over this
line, as the virtual RF source is the real Vaunix driver over an in-memory
brick: every framing, error check, refusal and wait the bench runs on
hardware runs here.  What is simulated is the controller, as Newport's manual
(Firmware V3.0, EDH0206En2021) describes it:

* the line answers queries ``nnAAvalue`` CR LF after the manual's ~10 ms
  (controller 1) or ~16 ms (the others), and set commands NOT AT ALL; a
  refused command memorizes its TE letter, which TE reads and clears;
* the states and their transitions: NOT REFERENCED from reset at power-up,
  HOMING (OR, only from NOT REFERENCED; aborted to NOT REFERENCED with the
  homing time-out bit past OT), READY, MOVING (PA/PR, only from READY, inside
  SL..SR), DISABLE (MM0/MM1), a reboot (RS) that answers nothing while it
  initializes;
* motion: a trapezoidal profile at VA and AC, ST decelerating at AC from
  wherever the stage is; PT answers the profile time of a relative move;
* TS's positioner error bits are reported once and cleared by the TS that
  reads them; ``fail_next_move`` injects a following error (DISABLE from
  MOVING) half-way through the next move.

Where the manual is silent the choice is stated: a query (``SL?``, ``VA?``,
``ID?`` ...) is answered in every state (its usage table is about the set
form); TP answers in NOT REFERENCED (V3.0's usage table); ST during HOMING
stops the search and leaves the controller NOT REFERENCED from HOMING.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Callable

#: The manual's figures for one answer (§6.4).
FIRST_REPLY_SECONDS = 0.010
OTHER_REPLY_SECONDS = 0.016

_NOT_REF_RESET, _NOT_REF_HOMING, _NOT_REF_DISABLE, _NOT_REF_READY, _NOT_REF_MOVING = 0x0A, 0x0B, 0x0D, 0x0E, 0x0F
_HOMING, _MOVING = 0x1E, 0x28
_READY_HOMING, _READY_MOVING, _READY_DISABLE = 0x32, 0x33, 0x34
_DISABLE_READY, _DISABLE_MOVING = 0x3C, 0x3D
_FOLLOWING_ERROR, _HOMING_TIMEOUT = 0x0020, 0x0040


@dataclass
class _Phase:
    """Constant acceleration from (t0, x0, v0) for ``seconds``."""

    t0: float
    x0: float
    v0: float
    a: float
    seconds: float

    def at(self, t: float) -> tuple[float, float]:
        s = min(max(t - self.t0, 0.0), self.seconds)
        return self.x0 + self.v0 * s + 0.5 * self.a * s * s, self.v0 + self.a * s

    @property
    def end(self) -> float:
        return self.t0 + self.seconds


def _profile(t0: float, x0: float, x1: float, velocity: float, acceleration: float) -> list[_Phase]:
    """Rest at x0 to rest at x1: accelerate, cruise, decelerate (a triangle when short)."""

    distance = abs(x1 - x0)
    if distance == 0.0:
        return []
    sign = 1.0 if x1 > x0 else -1.0
    ramp = velocity / acceleration
    if acceleration * ramp * ramp >= distance:  # never reaches velocity
        ramp = math.sqrt(distance / acceleration)
        cruise = 0.0
        peak = acceleration * ramp
    else:
        cruise = (distance - acceleration * ramp * ramp) / velocity
        peak = velocity
    phases = [_Phase(t0, x0, 0.0, sign * acceleration, ramp)]
    x, v = phases[-1].at(phases[-1].end)
    if cruise > 0:
        phases.append(_Phase(phases[-1].end, x, sign * peak, 0.0, cruise))
        x, v = phases[-1].at(phases[-1].end)
    phases.append(_Phase(phases[-1].end, x, v, -sign * acceleration, ramp))
    return phases


def profile_seconds(distance: float, velocity: float, acceleration: float) -> float:
    phases = _profile(0.0, 0.0, abs(distance), velocity, acceleration)
    return phases[-1].end if phases else 0.0


def _number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


class SimulatedSmc100:
    """One controller and its stage, at one RS-485 address."""

    def __init__(
        self,
        address: int = 1,
        *,
        stage: str = "VIRTUAL25CC",
        version: str = "SMC_CC - Controller-driver version 3.0.0 (simulated)",
        travel: tuple[float, float] = (0.0, 25.0),
        velocity: float = 5.0,
        acceleration: float = 20.0,
        home_velocity: float = 2.5,
        home_position: float = 0.0,
        home_timeout: float = 30.0,
        position: float = 12.0,
        init_seconds: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.address = int(address)
        self.stage, self.version = stage, version
        self.clock = clock
        self.configured = {"SL": float(travel[0]), "SR": float(travel[1]), "VA": float(velocity), "AC": float(acceleration),
                           "OH": float(home_velocity), "OT": float(home_timeout)}
        self.home_position = float(home_position)
        self.init_seconds = float(init_seconds)
        self.fail_next_move = False
        self.commands: list[str] = []
        self._position = float(position)
        self._power_up(at=None)

    # ------------------------------------------------------------- state
    def _power_up(self, at: float | None) -> None:
        self.working = dict(self.configured)
        self.state = _NOT_REF_RESET
        self.error = "@"
        self.errors = 0
        self._phases: list[_Phase] = []
        self._fault_at: float | None = None
        self._home_deadline: float | None = None
        self._stopped_homing = False
        self._booting_until = at

    def booting(self) -> bool:
        return self._booting_until is not None and self.clock() < self._booting_until

    def _advance(self) -> None:
        now = self.clock()
        if self._booting_until is not None and now >= self._booting_until:
            self._booting_until = None
        if self.state not in (_MOVING, _HOMING):
            return
        if self._fault_at is not None and now >= self._fault_at:
            self._position = self.position(self._fault_at)
            self._phases, self._fault_at = [], None
            self.state, self.errors = _DISABLE_MOVING, self.errors | _FOLLOWING_ERROR
            return
        if self.state == _HOMING and self._home_deadline is not None and now >= self._home_deadline:
            self._position = self.position(self._home_deadline)
            self._phases, self._home_deadline = [], None
            self.state, self.errors = _NOT_REF_HOMING, self.errors | _HOMING_TIMEOUT
            return
        if not self._phases or now >= self._phases[-1].end:
            if self._phases:
                self._position = self._phases[-1].at(self._phases[-1].end)[0]
            homing = self.state == _HOMING
            self._phases, self._home_deadline = [], None
            if homing and self._stopped_homing:
                self.state = _NOT_REF_HOMING
            else:
                self.state = _READY_HOMING if homing else _READY_MOVING
                if homing:
                    self._position = self.home_position

    def position(self, t: float | None = None) -> float:
        t = self.clock() if t is None else t
        for phase in self._phases:
            if t <= phase.end:
                return phase.at(t)[0]
        return self._phases[-1].at(self._phases[-1].end)[0] if self._phases else self._position

    def _velocity_now(self, t: float) -> float:
        for phase in self._phases:
            if t <= phase.end:
                return phase.at(t)[1]
        return 0.0

    def _start(self, target: float, velocity: float) -> None:
        now = self.clock()
        self._position = self.position(now)
        self._phases = _profile(now, self._position, target, velocity, self.working["AC"])
        self._stopped_homing = False
        if self.fail_next_move and self.state == _MOVING and self._phases:
            self.fail_next_move = False
            self._fault_at = now + (self._phases[-1].end - now) / 2

    # ----------------------------------------------------------- commands
    def handle(self, command: str, argument: str) -> str | None:
        """One command line's effect; its answer (value only), or None for no answer."""

        self._advance()
        if self.booting():
            return None
        self.commands.append(f"{command}{argument}")
        refuse = self._refuse
        query = argument == "?"
        if command == "TE":
            code, self.error = self.error, "@"
            return code
        if command == "TS":
            errors, self.errors = self.errors, 0
            return f"{errors:04X}{self.state:02X}"
        if command == "TP":
            return _number(self.position())
        if command == "VE":
            return f" {self.version}"
        if command == "ID":
            return self.stage if query else refuse(self._state_letter())
        if command in ("SL", "SR", "VA", "AC", "OT", "OH"):
            if query:
                return _number(self.working[command])
            return self._set(command, argument)
        if command == "PT":
            distance = self._value(argument)
            if distance is None or self.state not in (_READY_HOMING, _READY_MOVING, _READY_DISABLE, _MOVING,
                                                      _DISABLE_READY, _DISABLE_MOVING):
                return refuse("C" if distance is None else self._state_letter())
            return _number(profile_seconds(distance, self.working["VA"], self.working["AC"]))
        if command in ("PA", "PR"):
            return self._move(command, argument)
        if command == "OR":
            if self.state == _HOMING:
                return refuse("E")
            if self.state not in (_NOT_REF_RESET, _NOT_REF_HOMING, _NOT_REF_DISABLE, _NOT_REF_READY, _NOT_REF_MOVING):
                return refuse(self._state_letter())
            self.state = _HOMING
            self._start(self.home_position, self.working["OH"])
            self._home_deadline = self.clock() + self.working["OT"]
            return None
        if command == "ST":
            if self.state == _MOVING or self.state == _HOMING:
                now = self.clock()
                x, v = self.position(now), self._velocity_now(now)
                a = self.working["AC"]
                self._position = x
                self._phases = [_Phase(now, x, v, -math.copysign(a, v), abs(v) / a)] if v else []
                self._fault_at = None
                if self.state == _HOMING:
                    self._stopped_homing = True
                    self._home_deadline = None
                return None
            if self.state in (_NOT_REF_RESET, _NOT_REF_HOMING, _NOT_REF_DISABLE, _NOT_REF_READY, _NOT_REF_MOVING):
                return refuse("H")
            return None
        if command == "MM":
            if argument == "0" and self._ready():
                self.state = _DISABLE_READY
                return None
            if argument == "1" and self.state in (_DISABLE_READY, _DISABLE_MOVING):
                self.state = _READY_DISABLE
                return None
            if argument not in ("0", "1"):
                return refuse("C")
            return refuse(self._state_letter() if self.state not in (_DISABLE_READY, _DISABLE_MOVING) else "D")
        if command == "RS":
            self._position = self.position()
            self._power_up(at=self.clock() + self.init_seconds)
            return None
        return refuse("A")

    def _refuse(self, code: str) -> None:
        self.error = code
        return None

    def _ready(self) -> bool:
        return self.state in (_READY_HOMING, _READY_MOVING, _READY_DISABLE)

    def _state_letter(self) -> str:
        if self.state < 0x14:
            return "H"
        return {_HOMING: "L", _MOVING: "M", _DISABLE_READY: "J", _DISABLE_MOVING: "J"}.get(self.state, "K")

    @staticmethod
    def _value(argument: str) -> float | None:
        try:
            value = float(argument)
        except ValueError:
            return None
        return value if math.isfinite(value) else None

    def _set(self, command: str, argument: str) -> None:
        value = self._value(argument)
        if value is None:
            return self._refuse("C")
        if not (self._ready() or self.state in (_DISABLE_READY, _DISABLE_MOVING)):
            return self._refuse(self._state_letter())
        position = self.position()
        bad = {
            "VA": not 1e-6 < value <= self.configured["VA"],
            "AC": not 1e-6 < value <= self.configured["AC"],
            "SL": value > position or value > 0,
            "SR": value < position or value < 0,
            "OH": True, "OT": True,  # configuration parameters: CONFIGURATION state only
        }[command]
        if bad:
            return self._refuse("C" if command not in ("OH", "OT") else self._state_letter())
        self.working[command] = value
        return None

    def _move(self, command: str, argument: str) -> None:
        value = self._value(argument)
        if value is None:
            return self._refuse("C")
        if not self._ready():
            return self._refuse(self._state_letter())
        target = value if command == "PA" else self.position() + value
        if not self.working["SL"] <= target <= self.working["SR"]:
            return self._refuse("G")
        self.state = _MOVING
        self._start(target, self.working["VA"])
        return None


class SimulatedSerialPort:
    """The serial line to a chain of simulated controllers (pyserial's slice the link uses)."""

    def __init__(self, *controllers: SimulatedSmc100, timeout: float = 0.2) -> None:
        self.controllers = {controller.address: controller for controller in controllers}
        self.timeout = timeout
        self.is_open = True
        self._condition = threading.Condition()
        self._pending = b""
        self._replies: list[tuple[float, bytes]] = []
        self.silent = False

    def _clock(self) -> float:
        return next(iter(self.controllers.values())).clock() if self.controllers else time.monotonic()

    def write(self, data: bytes) -> int:
        if not self.is_open:
            raise OSError("the simulated port is closed")
        with self._condition:
            self._pending += bytes(data)
            while b"\r\n" in self._pending:
                line, self._pending = self._pending.split(b"\r\n", 1)
                self._line(line.decode("ascii").replace(" ", ""))
            self._condition.notify_all()
        return len(data)

    def _line(self, text: str) -> None:
        digits = len(text) - len(text.lstrip("0123456789"))
        address = int(text[:digits]) if digits else 0
        command, argument = text[digits:digits + 2].upper(), text[digits + 2:]
        targets = list(self.controllers.values()) if address == 0 and command in ("ST", "MM") else (
            [self.controllers[address]] if address in self.controllers else [])
        for controller in targets:
            answer = controller.handle(command, argument)
            if answer is not None and address and not self.silent:
                delay = FIRST_REPLY_SECONDS if address == 1 else OTHER_REPLY_SECONDS
                self._replies.append((controller.clock() + delay, f"{address}{command}{answer}\r\n".encode("ascii")))

    def read_until(self, expected: bytes = b"\n", size: int | None = None) -> bytes:
        deadline = time.monotonic() + float(self.timeout)
        with self._condition:
            while True:
                if self._replies and self._replies[0][0] <= self._clock():
                    return self._replies.pop(0)[1]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return b""
                due = self._replies[0][0] - self._clock() if self._replies else remaining
                self._condition.wait(max(0.0005, min(remaining, due)))

    def reset_input_buffer(self) -> None:
        with self._condition:
            now = self._clock()
            self._replies = [reply for reply in self._replies if reply[0] > now]

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.is_open = False


__all__ = ["SimulatedSerialPort", "SimulatedSmc100", "profile_seconds"]
