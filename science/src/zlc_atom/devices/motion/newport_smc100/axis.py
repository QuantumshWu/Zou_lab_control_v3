"""A Newport SMC100CC/PP controller and its stage as one motion axis.

Opening reads; it moves nothing.  The controller answers its version (VE),
its stage (ID), its travel (SL/SR), velocity (VA) and home time-out (OT),
its state (TS) and position (TP); what it will not answer in its present
state stays unknown, with its refusal, until the next read.  The unit is
the operator's word (mm or deg): the controller states its numbers in
whatever unit its stage configuration chose (its encoder increment, SU),
and does not name it.

A move (``tune("position", x)``) is PA, then TS polled until the stage
leaves MOVING, then TP: ``tune`` answers the position measured where the
stage stopped.  It is refused before anything is sent unless the controller
is READY and ``x`` lies inside SL..SR.  Its allowance is the controller's
own profile time for that distance (PT) twice over plus a second of
settling; a move still MOVING after that is stopped and reported.  A move
that cannot finish inside one core call (``CALL_BUDGET_SECONDS``) is
refused with the arithmetic rather than started.

NOT REFERENCED: refused, by name ("run Home first").  The SMC100 itself
refuses PA in that state (TE H), and the framework never edits an
instrument as a side effect (rf/contract.py: opening is read-only); homing
travels to a switch through whatever stands in the stage's path, so it is
an act the operator asks for, ``home``, and nothing else starts it.

``stop`` is declared concurrent (device_types.py): it reaches the
controller between two status polls of a move the lane is waiting on.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import threading
import time
from typing import Any, Callable
from uuid import uuid4

from zlc_atom.authoring import AuthoringField, TunableField, TuneRefused
from zlc_atom.devices.motion.contract import (
    AXIS_UNITS,
    POSITION_FIELD,
    STATE_FIELD,
    VELOCITY_FIELD,
    within,
)

from .protocol import (
    ADDRESSES,
    BAUD,
    CONFIGURATION,
    DISABLE,
    HOMING,
    JOGGING,
    MOVING,
    NOT_REFERENCED,
    READY,
    ControllerStatus,
    Smc100CommandError,
    Smc100Error,
    Smc100Link,
    Smc100NoReply,
    number_text,
    open_serial,
    parse_number,
)

_LOG = logging.getLogger(__name__)

#: How often a move or a homing is checked (TS).  One TS takes 10-16 ms, so
#: the line stays mostly idle and a finished move is seen within this.
POLL_SECONDS = 0.05
#: A move gets twice the controller's own profile time (PT) plus this, for
#: the loop to settle into its window, before it is overdue.
MOVE_TIME_FACTOR = 2.0
SETTLE_SECONDS = 1.0
#: One core call may take 120 s before the core gives the device up
#: (zlc-plugin-host CALL_DEADLINE); a wait stays well inside it.
CALL_BUDGET_SECONDS = 100.0
#: The manual: the controller's initialization "lasts less than 5 s".
RESET_SECONDS = 5.0


@dataclass(frozen=True)
class Smc100Config:
    """Which port, which controller address on it, and the stage's unit."""

    port: str
    address: int = 1
    unit: str = "mm"

    def __post_init__(self) -> None:
        port = str(self.port).strip()
        if not port:
            raise ValueError("serial port is required")
        if isinstance(self.address, bool) or int(self.address) not in ADDRESSES:
            raise ValueError("an SMC100 address is 1..31")
        if self.unit not in AXIS_UNITS:
            raise ValueError(f"the unit is one of {AXIS_UNITS}")
        object.__setattr__(self, "port", port)
        object.__setattr__(self, "address", int(self.address))


class Smc100Axis:
    """The ``motion.axis`` capability over one SMC100 controller."""

    def __init__(
        self,
        config: Smc100Config,
        *,
        serial_port=None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        if serial_port is None:
            try:
                serial_port = open_serial(config.port)
            except Exception as error:  # noqa: BLE001 -- pyserial's own words, with the port named
                raise RuntimeError(f"could not open {config.port} for the SMC100: {error}") from error
        self._link = Smc100Link(serial_port, config.address)
        self._clock = clock
        #: Moves, homing, settings and reads: one at a time.  Stop never takes it.
        self._lock = threading.RLock()
        self._stop_requested = threading.Event()
        #: Wakes a poll early when Stop is sent, so the move sees it at once.
        self._wake = threading.Event()
        self._device_session_id = uuid4().hex
        self._settings_epoch = 0
        self._limits: tuple[float, float] | None = None
        self._velocity: float | None = None
        self._home_timeout: float | None = None
        self._position: float | None = None
        self._status: ControllerStatus | None = None
        #: The positioner errors TS last reported.  TS clears them as it reads
        #: them, so they are kept here until the next command that drives.
        self._fault: tuple[str, ...] = ()
        #: Why a reading is missing: what the controller said when asked.
        self._unread: dict[str, str] = {}
        try:
            try:
                self.version = self._link.query("VE", "")
            except Smc100NoReply:
                raise RuntimeError(
                    f"no SMC100 answered at address {config.address} on {config.port} "
                    f"({BAUD} baud, 8N1, Xon/Xoff): {config.address}VE got no reply; check the port, "
                    "the address (the controller's own, set with Newport's utility) and its power"
                ) from None
            self.stage = self._optional(lambda: self._link.query("ID"), "stage") or ""
            self._read_settings()
            self._read_motion()
            _LOG.info(
                "SMC100 OPEN %s address %d: %s, stage %s, %s, travel %s %s device=%s",
                config.port, config.address, self.version, self.stage or "unnamed", self._state_text(),
                "unknown" if self._limits is None else f"{self._limits[0]:g}..{self._limits[1]:g}",
                config.unit, self.identity,
            )
        except BaseException:
            self._link.close()
            raise

    # ----------------------------------------------------------- readings
    @property
    def identity(self) -> str:
        return f"newport-smc100:{self.config.port}/{self.config.address}"

    @property
    def unit(self) -> str:
        return self.config.unit

    def _optional(self, read: Callable[[], Any], key: str) -> Any:
        """A reading the controller may refuse in its present state: None, and why."""

        try:
            value = read()
        except Smc100CommandError as error:
            self._unread[key] = str(error)
            return None
        self._unread.pop(key, None)
        return value

    def _number(self, command: str) -> float:
        return parse_number(command, self._link.query(command))

    def _read_settings(self) -> None:
        low = self._optional(lambda: self._number("SL"), "travel")
        high = self._optional(lambda: self._number("SR"), "travel") if low is not None else None
        if low is not None and high is not None:
            if not (math.isfinite(low) and math.isfinite(high)) or low > high:
                raise Smc100Error(f"the controller reports travel SL {low!r} .. SR {high!r}, which is not a range")
            self._limits = (low, high)
        else:
            self._limits = None
        self._velocity = self._optional(lambda: self._number("VA"), "velocity")
        self._home_timeout = self._optional(lambda: self._number("OT"), "home_timeout")

    def _read_status(self) -> ControllerStatus:
        status = self._link.status()
        if status.faults:
            if status.faults != self._fault:
                _LOG.info("SMC100 FAULT %s (%s) device=%s", ", ".join(status.faults), status.words, self.identity)
            self._fault = status.faults
        if self._status is None or status.code != self._status.code:
            _LOG.info("SMC100 STATE %s device=%s", status.words, self.identity)
        self._status = status
        return status

    def _read_position(self) -> float | None:
        self._position = self._optional(lambda: parse_number("TP", self._link.query("TP", "")), "position")
        return self._position

    def _read_motion(self) -> ControllerStatus:
        status = self._read_status()
        self._read_position()
        return status

    def _state_text(self) -> str:
        if self._status is None:
            return "unread"
        text = self._status.words
        return f"{text}; fault: {', '.join(self._fault)}" if self._fault else text

    def _driving(self) -> None:
        """A command that drives starts with no stale fault and no stale Stop."""

        self._fault = ()
        self._stop_requested.clear()
        self._wake.clear()

    # ----------------------------------------------------------- tunables
    def tunable_fields(self) -> tuple[TunableField, ...]:
        with self._lock:
            unit = self.config.unit
            limits = self._limits
            travel = (
                f"the controller's travel SL..SR, {limits[0]:g} to {limits[1]:g} {unit}"
                if limits is not None
                else f"the travel is unread ({self._unread.get('travel', 'not asked')}); Refresh after homing"
            )
            return (
                TunableField(
                    AuthoringField(
                        POSITION_FIELD, "float", "Position", None,
                        minimum=None if limits is None else limits[0],
                        maximum=None if limits is None else limits[1],
                        unit=unit,
                        description=(
                            f"absolute position of stage {self.stage or '(unnamed)'}; Apply moves there and "
                            f"reads where it stopped (TP). Bounded by {travel}. Refused unless READY: "
                            "an unreferenced stage is homed only by Home"
                        ),
                    ),
                    self._position,
                    True,
                    (POSITION_FIELD,),
                    limits,
                ),
                TunableField(
                    AuthoringField(
                        VELOCITY_FIELD, "float", f"Velocity ({unit}/s)", None,
                        description=(
                            "the velocity of the next moves (VA), up to the stage's configured maximum; "
                            "a working value the controller forgets at Reset or power-up"
                        ),
                    ),
                    self._velocity,
                    True,
                    (VELOCITY_FIELD,),
                ),
                TunableField(
                    AuthoringField(
                        STATE_FIELD, "str", "State", None, derived=True,
                        description="the controller's state (TS), read; Home, Stop, Enable, Disable and Reset change it",
                    ),
                    self._state_text(),
                    False,
                    (STATE_FIELD,),
                ),
            )

    def tunable_values(self) -> dict[str, Any]:
        with self._lock:
            self._read_motion()
            self._velocity = self._optional(lambda: self._number("VA"), "velocity")
            return {POSITION_FIELD: self._position, VELOCITY_FIELD: self._velocity, STATE_FIELD: self._state_text()}

    def refresh_tunable_fields(self) -> tuple[TunableField, ...]:
        """Explicit hardware refresh (Device Control): settings, state and position."""

        with self._lock:
            self._read_settings()
            self._read_motion()
            return self.tunable_fields()

    def settings_provenance(self) -> dict[str, object]:
        with self._lock:
            return {"device_session_id": self._device_session_id, "settings_epoch": self._settings_epoch}

    def tune(self, name: str, value: Any) -> Any:
        selected = str(name)
        try:
            if selected == POSITION_FIELD:
                effective = self._move_to(value)
            elif selected == VELOCITY_FIELD:
                effective = self._set_velocity(value)
            elif selected == STATE_FIELD:
                raise TuneRefused("the state is read from the controller; Home, Stop, Enable, Disable and Reset change it")
            else:
                raise ValueError(
                    f"this axis has no tunable field {selected!r}; it offers "
                    f"{POSITION_FIELD}, {VELOCITY_FIELD} and {STATE_FIELD} (read only)"
                )
        except Exception as error:
            _LOG.info(
                "TUNE REFUSED field=%s value=%r error=%s: %s -- device=%s",
                selected, value, type(error).__name__, str(error).replace(chr(10), " "), self.identity,
            )
            raise
        _LOG.info("TUNE field=%s value=%r effective=%r device=%s", selected, value, effective, self.identity)
        return effective

    def _set_velocity(self, value: Any) -> float:
        if isinstance(value, bool):
            raise TuneRefused(f"velocity takes a number in {self.config.unit}/s")
        try:
            wanted = float(value)
        except (TypeError, ValueError):
            raise TuneRefused(f"velocity takes a number in {self.config.unit}/s, not {value!r}") from None
        if not math.isfinite(wanted) or wanted <= 0:
            raise TuneRefused(f"velocity must be a positive number of {self.config.unit}/s")
        with self._lock:
            try:
                self._link.execute("VA", number_text(wanted))
            except Smc100CommandError as error:
                # C: above the stage's configured maximum; H/L/M: not in this state.
                raise TuneRefused(f"the controller refused velocity {wanted:g} {self.config.unit}/s: {error}") from None
            before = self._velocity
            self._velocity = self._number("VA")
            if self._velocity != before:
                self._settings_epoch += 1
            return self._velocity

    def _refusal(self, status: ControllerStatus, asked: str) -> str:
        """Why ``asked`` cannot be done in ``status``, for the operator."""

        state = f"the controller is {status.words}" + (f" (fault: {', '.join(self._fault)})" if self._fault else "")
        return {
            NOT_REFERENCED: f"{state}: run Home first; {asked} never homes by itself",
            DISABLE: f"{state}, motor off: run Enable first",
            MOVING: f"{state}: wait for it, or Stop",
            HOMING: f"{state}: wait for it, or Stop",
            CONFIGURATION: f"{state}: leave its configuration (Newport's utility) or Reset it",
            JOGGING: f"{state}: the SMC-RC keypad has it; leave jogging there first",
        }.get(status.kind, f"{state}; nothing was sent")

    def _move_to(self, value: Any) -> float:
        unit = self.config.unit
        with self._lock:
            status = self._read_status()
            if status.kind != READY:
                raise TuneRefused(self._refusal(status, "a move"))
            if self._limits is None:
                self._read_settings()
            if self._limits is None:
                raise TuneRefused(f"the travel (SL/SR) could not be read: {self._unread.get('travel')}")
            low, high = self._limits
            target = within(value, low, high, name=POSITION_FIELD, unit=unit)
            start = self._read_position()
            distance = abs(target - start) if start is not None else high - low
            try:
                profile = parse_number("PT", self._link.query("PT", number_text(distance))) if distance > 0 else 0.0
            except Smc100CommandError as error:
                raise TuneRefused(f"the controller would not time a move of {distance:g} {unit}: {error}") from None
            allowance = MOVE_TIME_FACTOR * profile + SETTLE_SECONDS
            if allowance > CALL_BUDGET_SECONDS:
                raise TuneRefused(
                    f"moving {distance:g} {unit} takes {profile:.1f} s at the present velocity (PT), "
                    f"{allowance:.0f} s with its allowance: longer than the {CALL_BUDGET_SECONDS:g} s one "
                    "command may wait; raise the velocity or move in steps"
                )
            self._driving()
            try:
                self._link.execute("PA", number_text(target))
            except Smc100CommandError as error:
                # The controller did not execute it: the stage is where it was.
                raise TuneRefused(f"the controller refused the move to {target:g} {unit}: {error}") from None
            _LOG.info("SMC100 MOVE to %s %s (PT %.3f s) device=%s", number_text(target), unit, profile, self.identity)
            final, overdue = self._wait(MOVING, allowance, stop_when_overdue=True)
            position = self._read_position()
            self._settings_epoch += 1
            where = "an unread position" if position is None else f"{position:g} {unit}"
            if overdue:
                raise RuntimeError(
                    f"the move to {target:g} {unit} was still MOVING after {allowance:.1f} s "
                    f"(twice its PT {profile:.2f} s and {SETTLE_SECONDS:g} s); it was stopped and is "
                    f"{final.words} at {where}"
                )
            if self._stop_requested.is_set():
                raise RuntimeError(f"Stop ended the move to {target:g} {unit} at {where} ({final.words})")
            if final.kind != READY:
                raise RuntimeError(
                    f"the move to {target:g} {unit} ended {final.words}"
                    + (f" with {', '.join(self._fault)}" if self._fault else "") + f", at {where}"
                )
            return position

    def _wait(self, kind: str, allowance: float, *, stop_when_overdue: bool) -> tuple[ControllerStatus, bool]:
        """Poll TS while the controller is in ``kind``: the state it left to (or the one it
        overran its allowance in), and whether it overran.  ``stop_when_overdue``: an overrun
        is stopped, then given the allowance again to come to rest."""

        started = self._clock()
        overdue = False
        while True:
            status = self._read_status()
            if status.kind != kind:
                return status, overdue
            if self._clock() - started > allowance:
                if not stop_when_overdue or overdue:
                    return status, overdue
                overdue = True
                self._send_stop()
                started = self._clock()
            if self._wake.wait(POLL_SECONDS):
                self._wake.clear()

    # ------------------------------------------------------------ actions
    def home(self) -> str:
        """OR, from NOT REFERENCED only, answered when the controller leaves HOMING."""

        unit = self.config.unit
        with self._lock:
            status = self._read_status()
            if status.kind == READY or status.kind == DISABLE:
                raise RuntimeError(
                    f"the controller is {status.words}: already referenced. It homes only from "
                    "NOT REFERENCED; Reset it first to home again"
                )
            if status.kind != NOT_REFERENCED:
                raise RuntimeError(self._refusal(status, "Home"))
            self._driving()
            self._link.execute("OR")
            _LOG.info("SMC100 HOME device=%s", self.identity)
            allowance = min(
                CALL_BUDGET_SECONDS,
                (self._home_timeout + SETTLE_SECONDS) if self._home_timeout is not None else CALL_BUDGET_SECONDS,
            )
            final, _ = self._wait(HOMING, allowance, stop_when_overdue=False)
            if final.kind == HOMING:
                return (
                    f"still HOMING after {allowance:.0f} s (its own home time-out: "
                    f"{'unread' if self._home_timeout is None else f'{self._home_timeout:g} s'}); "
                    "Refresh to see it finish"
                )
            self._read_settings()
            position = self._read_position()
            self._settings_epoch += 1
            if final.kind != READY:
                raise RuntimeError(
                    f"homing ended {final.words}" + (f" with {', '.join(self._fault)}" if self._fault else "")
                    + (" (Stop)" if self._stop_requested.is_set() else "")
                )
            return f"homed: {final.words} at {'an unread position' if position is None else f'{position:g} {unit}'}"

    def _send_stop(self) -> None:
        self._stop_requested.set()
        self._wake.set()
        self._link.execute("ST")

    def stop(self) -> str:
        """ST now -- concurrent with a move that holds the lane; never waits for it."""

        try:
            self._send_stop()
        except Smc100CommandError as error:
            status = self._read_status()
            if status.kind in (MOVING, HOMING):
                raise RuntimeError(f"the controller refused Stop while {status.words}: {error}") from None
            _LOG.info("SMC100 STOP nothing moving (%s) device=%s", status.words, self.identity)
            return f"nothing was moving: {status.words}"
        _LOG.info("SMC100 STOP device=%s", self.identity)
        status = self._read_status()
        return f"stop sent: {status.words}"

    def make_safe(self, output: str) -> None:
        """Stop all: stop the motion; an axis at rest is already safe."""

        del output
        self.stop()

    def enable(self) -> str:
        """MM1: DISABLE -> READY (the set-point becomes the current position)."""

        return self._mode("1", DISABLE, "Enable")

    def disable(self) -> str:
        """MM0: READY -> DISABLE (motor off; the encoder is still read)."""

        return self._mode("0", READY, "Disable")

    def _mode(self, argument: str, origin: str, asked: str) -> str:
        with self._lock:
            status = self._read_status()
            if status.kind != origin:
                raise RuntimeError(f"{asked} acts from {origin}; {self._refusal(status, asked)}")
            self._driving()
            self._link.execute("MM", argument)
            status = self._read_motion()
            self._settings_epoch += 1
            return status.words

    def reset(self) -> str:
        """RS, a reboot: answered once the controller answers again (NOT REFERENCED)."""

        with self._lock:
            self._driving()
            # A rebooting controller answers no TE: what it did is read from TS once it answers.
            self._link.send("RS")
            _LOG.info("SMC100 RESET device=%s", self.identity)
            started = self._clock()
            while True:
                time.sleep(POLL_SECONDS)
                try:
                    status = self._read_status()
                    break
                except Smc100NoReply:
                    if self._clock() - started > 2 * RESET_SECONDS:
                        raise RuntimeError(
                            f"the controller did not answer within {2 * RESET_SECONDS:g} s of its reset"
                        ) from None
            self._read_settings()
            self._read_position()
            self._settings_epoch += 1
            if status.kind != NOT_REFERENCED:
                raise RuntimeError(f"Reset did not reboot the controller: it is {status.words}")
            return status.words

    def status(self) -> str:
        """The controller's state, fault, position, travel and velocity, read now."""

        with self._lock:
            self._read_settings()
            self._read_motion()
            unit = self.config.unit
            position = "unread" if self._position is None else f"{self._position:g} {unit}"
            travel = (
                f"unread ({self._unread.get('travel')})" if self._limits is None
                else f"{self._limits[0]:g}..{self._limits[1]:g} {unit}"
            )
            velocity = "unread" if self._velocity is None else f"{self._velocity:g} {unit}/s"
            return (
                f"{self._state_text()}; position {position}; travel {travel}; velocity {velocity}; "
                f"stage {self.stage or 'unnamed'}; {self.version}"
            )

    def close(self) -> None:
        self._link.close()


# ------------------------------------------------------------------ finding
def probe_port(device: str, addresses: tuple[int, ...], *, opener=open_serial) -> list[tuple[int, str, str]]:
    """``(address, version, stage)`` of each controller that answers VE on ``device``.

    VE and ID? only read; nothing is set.  A port another program holds
    cannot be opened and is passed over.
    """

    try:
        port = opener(device)
    except Exception:  # noqa: BLE001 -- held by another program, or gone
        return []
    found = []
    try:
        for address in addresses:
            link = Smc100Link(port, address)
            try:
                version = link.query("VE", "")
            except Smc100Error:
                continue
            try:
                stage = link.query("ID")
            except Smc100Error:
                stage = ""
            found.append((address, version, stage))
    except Exception:  # noqa: BLE001 -- a port that fails mid-probe has no controller we can use
        return found
    finally:
        try:
            port.close()
        except Exception:  # noqa: BLE001
            pass
    return found


def discover_smc100(
    *,
    addresses: tuple[int, ...] = (1,),
    excluded: frozenset[str] = frozenset(),
    ports: tuple[str, ...] | None = None,
    first: bool = False,
    opener=open_serial,
) -> tuple[tuple[str, int, str, str], ...]:
    """``(port, address, version, stage)`` for each controller answering on a free serial port."""

    from zlc_atom.devices import free_serial_ports

    found: list[tuple[str, int, str, str]] = []
    for device in free_serial_ports(excluded)[0] if ports is None else ports:
        for hit in probe_port(device, addresses, opener=opener):
            found.append((device, *hit))
            if first:
                return tuple(found)
    return tuple(found)


__all__ = [
    "CALL_BUDGET_SECONDS",
    "POLL_SECONDS",
    "Smc100Axis",
    "Smc100Config",
    "discover_smc100",
    "probe_port",
]
