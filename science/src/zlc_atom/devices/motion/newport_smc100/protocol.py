"""The SMC100's ASCII command link: framing, the one error register, the state table.

Every fact here is from Newport's manual, "SMC100CC & SMC100PP Single-Axis
Motion Controller/Driver, User's Manual, Firmware V3.0" (EDH0206En2021,
10/11; the V2.0 SMC100CC manual EDH0206En1022 says the same):

* the line (§2.8, §3.2): 57600 baud, 8 data bits, no parity, 1 stop bit,
  Xon/Xoff flow control, commands terminated by CR LF -- the same through
  the SMC-USB adapter, a virtual COM port;
* a command (§6.3) is ``nn AA xx``: the controller address 1..31 (several
  controllers share one COM port over their internal RS-485 link, each
  decoding its own address), the two-letter name, and a value or ``?``.
  One command per line: anything after the first command is ignored;
* a query answers ``nnAAvalue`` CR LF; a SET COMMAND ANSWERS NOTHING
  (§6.5).  A command that cannot be executed "memorizes an error" that only
  ``TE`` reads, and clears; a later error overwrites an unread one, so the
  manual asks for ``TE`` after every command;
* ``TS`` answers ``abcdef``: four hex digits of positioner error bits, then
  the two-digit controller state.  Reading ``TS`` clears the positioner
  error buffer;
* ``nTP?``-type commands take "about 10 ms" on controller 1 and "about
  16 ms" on the others, send to answer (§6.4); no other command spacing is
  stated, and a move does not hold up the commands after it.

What this link enforces: ONE owner of the port, and each exchange --
including a set command with the ``TE`` that judges it -- under one lock, so
no other thread's command can land between a command and its error check
(``TE`` is one register: a Stop sent from another thread must not read, and
clear, the error of the move it interrupts).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time

BAUD = 57600
TERMINATOR = b"\r\n"
ADDRESSES = range(1, 32)

#: How long one reply may take.  The manual's figure is 10-16 ms; twelve
#: times that leaves the USB adapter's own latency room without letting a
#: missing controller stall the lane for long.
REPLY_TIMEOUT_SECONDS = 0.2

# Reopening a USB/serial link need not behave like an established session.
# Only the read-only identity handshake uses this allowance and retries.
STARTUP_REPLY_TIMEOUT_SECONDS = 1.0
STARTUP_ATTEMPTS = 3
STARTUP_RETRY_SECONDS = 0.1

_LOG = logging.getLogger(__name__)

#: Controller states (TS, the last two digits): code -> (kind, the manual's words).
#: The kind is what decides what may be asked of the controller.
NOT_REFERENCED = "NOT REFERENCED"
CONFIGURATION = "CONFIGURATION"
HOMING = "HOMING"
MOVING = "MOVING"
READY = "READY"
DISABLE = "DISABLE"
JOGGING = "JOGGING"

STATES: dict[int, tuple[str, str]] = {
    0x0A: (NOT_REFERENCED, "NOT REFERENCED from reset"),
    0x0B: (NOT_REFERENCED, "NOT REFERENCED from HOMING"),
    0x0C: (NOT_REFERENCED, "NOT REFERENCED from CONFIGURATION"),
    0x0D: (NOT_REFERENCED, "NOT REFERENCED from DISABLE"),
    0x0E: (NOT_REFERENCED, "NOT REFERENCED from READY"),
    0x0F: (NOT_REFERENCED, "NOT REFERENCED from MOVING"),
    0x10: (NOT_REFERENCED, "NOT REFERENCED, ESP stage error"),
    0x11: (NOT_REFERENCED, "NOT REFERENCED from JOGGING"),
    0x14: (CONFIGURATION, "CONFIGURATION"),
    0x1E: (HOMING, "HOMING commanded from RS-232-C"),
    0x1F: (HOMING, "HOMING commanded by SMC-RC"),
    0x28: (MOVING, "MOVING"),
    0x32: (READY, "READY from HOMING"),
    0x33: (READY, "READY from MOVING"),
    0x34: (READY, "READY from DISABLE"),
    0x35: (READY, "READY from JOGGING"),
    0x3C: (DISABLE, "DISABLE from READY"),
    0x3D: (DISABLE, "DISABLE from MOVING"),
    0x3E: (DISABLE, "DISABLE from JOGGING"),
    0x46: (JOGGING, "JOGGING from READY"),
    0x47: (JOGGING, "JOGGING from DISABLE"),
}

#: Positioner error bits (TS, the first four hex digits).
POSITIONER_ERRORS: tuple[tuple[int, str], ...] = (
    (0x0200, "80 W output power exceeded"),
    (0x0100, "DC voltage too low"),
    (0x0080, "wrong ESP stage"),
    (0x0040, "homing time-out"),
    (0x0020, "following error"),
    (0x0010, "short circuit detected"),
    (0x0008, "RMS current limit"),
    (0x0004, "peak current limit"),
    (0x0002, "positive end of run"),
    (0x0001, "negative end of run"),
)

#: Command errors (TE), in the manual's words.
COMMAND_ERRORS: dict[str, str] = {
    "@": "no error",
    "A": "unknown message code or floating point controller address",
    "B": "controller address not correct",
    "C": "parameter missing or out of range",
    "D": "command not allowed",
    "E": "home sequence already started",
    "F": "ESP stage name unknown",
    "G": "displacement out of limits",
    "H": "command not allowed in NOT REFERENCED state",
    "I": "command not allowed in CONFIGURATION state",
    "J": "command not allowed in DISABLE state",
    "K": "command not allowed in READY state",
    "L": "command not allowed in HOMING state",
    "M": "command not allowed in MOVING state",
    "N": "current position out of software limit",
    "S": "communication time out",
    "U": "error during EEPROM access",
    "V": "error during command execution",
    "W": "command not allowed for PP version",
    "X": "command not allowed for CC version",
}


class Smc100Error(RuntimeError):
    """The controller did not do what was asked."""


class Smc100NoReply(Smc100Error):
    """No (complete) answer within the reply time-out."""


class Smc100ConnectionError(Smc100Error):
    """The read-only startup handshake failed; includes its exchange history."""


class Smc100CommandError(Smc100Error):
    """The controller refused a command; ``code`` is its TE letter."""

    def __init__(self, command: str, code: str) -> None:
        self.command = command
        self.code = code
        super().__init__(f"{command} refused: TE {code} ({COMMAND_ERRORS.get(code, 'an error code the manual does not list')})")


@dataclass(frozen=True)
class ControllerStatus:
    """One TS answer: the state and the positioner errors it reported (and cleared)."""

    code: int
    errors: int

    @property
    def kind(self) -> str:
        return STATES.get(self.code, ("UNKNOWN", ""))[0]

    @property
    def words(self) -> str:
        return STATES.get(self.code, ("", f"state 0x{self.code:02X}, which the manual does not list"))[1]

    @property
    def faults(self) -> tuple[str, ...]:
        return tuple(text for bit, text in POSITIONER_ERRORS if self.errors & bit)


def parse_status(value: str) -> ControllerStatus:
    text = value.strip()
    if len(text) != 6:
        raise Smc100Error(f"TS answered {value!r}, not six hex digits")
    try:
        return ControllerStatus(int(text[4:], 16), int(text[:4], 16))
    except ValueError:
        raise Smc100Error(f"TS answered {value!r}, not six hex digits") from None


def number_text(value: float) -> str:
    """A value as the controller reads it: a dot decimal, never an exponent."""

    text = f"{float(value):.9f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def parse_number(command: str, value: str) -> float:
    try:
        return float(value)
    except ValueError:
        raise Smc100Error(f"{command} answered {value!r}, not a number") from None


def open_serial(port: str):
    """The controller's line, read with a time-out (no busy wait)."""

    import serial

    return serial.Serial(
        port, BAUD, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
        xonxoff=True, timeout=REPLY_TIMEOUT_SECONDS, write_timeout=1.0,
    )


class Smc100Link:
    """One controller address over a serial port this link owns."""

    def __init__(self, port, address: int) -> None:
        if isinstance(address, bool) or int(address) not in ADDRESSES:
            raise ValueError("an SMC100 address is 1..31")
        self.port = port
        self.address = int(address)
        self._lock = threading.Lock()

    def _line(self, command: str, argument: str) -> bytes:
        return f"{self.address}{command}{argument}".encode("ascii") + TERMINATOR

    def identify(self, *, reply_timeout: float | None = None) -> str:
        """Establish communication using TE / VE / TE on the same open port.

        The first TE consumes a previous session's command error, separately
        from the version query. Only these read-only exchanges are retried.
        In particular, a failed handshake never resets or homes the stage.
        Discovery can use a shorter per-reply budget for absent addresses.
        """

        timeout = STARTUP_REPLY_TIMEOUT_SECONDS if reply_timeout is None else reply_timeout
        history: list[str] = []
        with self._lock:
            previous_timeout = self.port.timeout
            try:
                for attempt in range(1, STARTUP_ATTEMPTS + 1):
                    exchanges: list[str] = []
                    try:
                        try:
                            before = self._startup_query("TE", timeout, exchanges)
                        except Smc100NoReply:
                            # VE can still identify a controller whose TE is unavailable
                            # (e.g. JOGGING). Keep that evidence in the failure report.
                            before = None
                        if before is not None and (len(before) != 1 or before not in COMMAND_ERRORS):
                            raise Smc100Error(f"invalid initial TE value {before!r}")
                        if before not in (None, "@"):
                            _LOG.info(
                                "SMC100 CONNECT address %d: prior TE %s (%s)",
                                self.address, before, COMMAND_ERRORS[before],
                            )
                        version_error = None
                        try:
                            version = self._startup_query("VE", timeout, exchanges)
                        except Smc100Error as error:
                            version_error = error
                        try:
                            after = self._startup_query("TE", timeout, exchanges)
                        except Smc100NoReply as error:
                            if version_error is None and version:
                                raise Smc100Error(
                                    f"VE answered {version!r}, but error-register synchronization failed: {error}; "
                                    "TE is unavailable in JOGGING (leave jogging on the SMC-RC and Init again)"
                                ) from None
                            raise
                        if version_error is not None:
                            raise Smc100Error(
                                f"{version_error}; TE after VE returned {after!r} "
                                f"({COMMAND_ERRORS.get(after, 'invalid command error reply')})"
                            )
                        if not version:
                            raise Smc100Error("VE returned an empty version")
                        if after != "@":
                            raise Smc100Error(
                                f"TE after VE returned {after!r} "
                                f"({COMMAND_ERRORS.get(after, 'invalid command error reply')})"
                            )
                    except Smc100Error as error:
                        detail = f"attempt {attempt}: {error}; " + "; ".join(exchanges)
                        history.append(detail)
                        _LOG.info("SMC100 CONNECT address %d: %s", self.address, detail)
                        if attempt < STARTUP_ATTEMPTS:
                            time.sleep(STARTUP_RETRY_SECONDS)
                    else:
                        _LOG.info(
                            "SMC100 CONNECT address %d succeeded on attempt %d: %s",
                            self.address, attempt, "; ".join(exchanges),
                        )
                        return version
            finally:
                self.port.timeout = previous_timeout
        raise Smc100ConnectionError(
            f"SMC100 startup handshake failed at address {self.address} "
            f"after {STARTUP_ATTEMPTS} attempts ({timeout:g} s per reply). "
            + " | ".join(history)
        )

    def _startup_query(self, command: str, timeout: float, exchanges: list[str]) -> str:
        """Read one startup response, retaining partial and unrelated RX in diagnostics.

        This tolerance is limited to connection setup. Normal exchanges still
        fail on an unexpected reply, and set commands are never replayed.
        """

        started = time.monotonic()
        received = bytearray()
        pending = bytearray()
        try:
            self.port.reset_input_buffer()
            self.port.write(self._line(command, ""))
            # write_timeout bounds writes; the reply waits for transmission.
            # pySerial's Windows flush() can wait indefinitely on flow control.
            deadline = time.monotonic() + timeout
            prefix = f"{self.address}{command}".encode("ascii")
            while (remaining := deadline - time.monotonic()) > 0:
                self.port.timeout = min(REPLY_TIMEOUT_SECONDS, remaining)
                chunk = self.port.read_until(TERMINATOR, size=256)
                # Bound diagnostics and garbage accumulation on a wrong port.
                received.extend(chunk[:max(0, 256 - len(received))])
                pending.extend(chunk)
                while TERMINATOR in pending:
                    raw, _, tail = pending.partition(TERMINATOR)
                    pending = bytearray(tail)
                    if raw[:len(prefix)].upper() == prefix:
                        try:
                            return raw[len(prefix):].decode("ascii").strip()
                        except UnicodeDecodeError:
                            pass
                if len(pending) >= 256:
                    raise Smc100Error(f"{prefix.decode()} received an overlong unterminated reply")
            raise Smc100NoReply(f"{prefix.decode()} had no matching complete reply within {timeout:g} s")
        finally:
            exchanges.append(
                f"TX {self._line(command, '')!r}, RX {bytes(received)!r}, "
                f"{time.monotonic() - started:.3f} s"
            )

    def _send(self, command: str, argument: str) -> None:
        # A reply that arrived after its exchange gave up belongs to nobody.
        self.port.reset_input_buffer()
        self.port.write(self._line(command, argument))
        flush = getattr(self.port, "flush", None)
        if callable(flush):
            flush()

    def _reply(self, command: str) -> str | None:
        """The value of the answer to ``command``, or None when none came in time."""

        raw = self.port.read_until(TERMINATOR)
        if not raw.endswith(TERMINATOR):
            return None
        text = raw[: -len(TERMINATOR)].decode("ascii", errors="replace")
        prefix = f"{self.address}{command}"
        if not text.upper().startswith(prefix.upper()):
            raise Smc100Error(f"{prefix} was answered {text!r}, which is not its answer")
        return text[len(prefix):].strip()

    def _error_locked(self) -> str:
        self._send("TE", "")
        code = self._reply("TE")
        if code is None:
            raise Smc100NoReply(f"controller {self.address} did not answer TE within {REPLY_TIMEOUT_SECONDS:g} s")
        return code[:1] or "@"

    def query(self, command: str, argument: str = "?") -> str:
        """Ask and return the value; a refusal is read from TE and raised by its letter."""

        with self._lock:
            self._send(command, argument)
            value = self._reply(command)
            if value is not None:
                return value
            code = self._error_locked()
        if code != "@":
            raise Smc100CommandError(f"{self.address}{command}{argument}", code)
        raise Smc100NoReply(
            f"controller {self.address} did not answer {self.address}{command}{argument} "
            f"within {REPLY_TIMEOUT_SECONDS:g} s"
        )

    def execute(self, command: str, argument: str = "") -> None:
        """A set command (no answer), judged by the TE read straight after it."""

        with self._lock:
            self._send(command, argument)
            code = self._error_locked()
        if code != "@":
            raise Smc100CommandError(f"{self.address}{command}{argument}", code)

    def send(self, command: str, argument: str = "") -> None:
        """A command with no error check: only RS, after which no TE answers."""

        with self._lock:
            self._send(command, argument)

    def status(self) -> ControllerStatus:
        return parse_status(self.query("TS", ""))

    def close(self) -> None:
        with self._lock:
            self.port.close()


__all__ = [
    "ADDRESSES",
    "BAUD",
    "COMMAND_ERRORS",
    "CONFIGURATION",
    "ControllerStatus",
    "DISABLE",
    "HOMING",
    "JOGGING",
    "MOVING",
    "NOT_REFERENCED",
    "POSITIONER_ERRORS",
    "READY",
    "REPLY_TIMEOUT_SECONDS",
    "STATES",
    "Smc100CommandError",
    "Smc100ConnectionError",
    "Smc100Error",
    "Smc100Link",
    "Smc100NoReply",
    "number_text",
    "open_serial",
    "parse_number",
    "parse_status",
]
