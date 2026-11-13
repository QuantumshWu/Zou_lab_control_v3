"""Headless hardware and virtual-device contracts."""

from collections import deque
from typing import Callable
import re
import threading
import time

from zlc_plugin.faults import serial_fault


class RecordQueue:
    """Bounded FIFO of already-owned records, shared by device adapters.

    Adapters construct their records and own SDK start/read/stop. This queue
    checks capture ordinals, preserves every accepted record, and reports
    overflow/failure before handing out data. An optional producer worker is
    stopped and joined here; a device-owned SDK lane is never replaced.
    """

    def __init__(self, what: str, *, join_timeout_seconds: float) -> None:
        self._what = str(what)
        self._join_timeout = float(join_timeout_seconds)
        self._condition = threading.Condition()
        self._queue: deque[object] = deque()
        self._armed = False
        self._accepting = False
        self._expected: int | None = None
        self._buffer_record_count = 1
        self._produced_count = 0
        self._failure: BaseException | None = None
        self._worker: threading.Thread | None = None
        self._stop: threading.Event | None = None
        self._ready = threading.Event()

    @property
    def armed(self) -> bool:
        with self._condition:
            return self._armed

    @property
    def accepting(self) -> bool:
        with self._condition:
            return self._accepting

    @property
    def produced_count(self) -> int:
        with self._condition:
            return self._produced_count

    @property
    def pending_count(self) -> int:
        with self._condition:
            return len(self._queue)

    @property
    def capacity(self) -> int:
        with self._condition:
            return self._buffer_record_count

    @property
    def failure(self) -> BaseException | None:
        with self._condition:
            return self._failure

    def arm(
        self,
        records: int | None,
        *,
        buffer_record_count: int,
        worker: Callable[[threading.Event], threading.Thread] | None = None,
    ) -> None:
        """Start a capture; ``worker`` builds the producer thread from its stop event."""

        buffer_count = int(buffer_record_count)
        if buffer_count <= 0:
            raise ValueError("buffer_record_count must be positive")
        expected = None if records is None else int(records)
        if expected is not None and expected <= 0:
            raise ValueError("finite records must be positive")
        with self._condition:
            if self._armed:
                raise RuntimeError(f"{self._what} is already armed")
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError(f"{self._what} is still finishing its previous capture")
            self._queue.clear()
            self._failure = None
            self._armed = True
            self._accepting = True
            self._expected = expected
            self._buffer_record_count = buffer_count
            self._produced_count = 0
            self._ready.clear()
            if worker is None:
                return
            stop = threading.Event()
            try:
                thread = worker(stop)
                self._worker = thread
                self._stop = stop
                thread.start()
            except BaseException:
                self._worker = None
                self._stop = None
                self._armed = False
                self._accepting = False
                raise

    def mark_ready(self) -> None:
        """The producer has started the hardware needed for this capture."""

        self._ready.set()

    def wait_ready(self, timeout: float) -> None:
        if not self._ready.wait(timeout):
            raise TimeoutError(f"{self._what} did not become ready")
        with self._condition:
            if self._failure is not None:
                raise RuntimeError(f"{self._what} failed while arming: {self._failure}") from self._failure
            if not self._armed:
                raise RuntimeError(f"{self._what} stopped before becoming ready")

    def push(self, record: object) -> bool:
        """Keep one complete record; failures stop acceptance without dropping."""

        with self._condition:
            if not self._accepting:
                return False
            ordinal = getattr(record, "source_ordinal", None)
            if type(ordinal) is not int or ordinal != self._produced_count:
                self.fail(RuntimeError(
                    f"{self._what} record ordinal {ordinal!r} is not "
                    f"the expected {self._produced_count} "
                    f"(observed_at_ns={time.time_ns()})"
                ))
                return False
            if len(self._queue) >= self._buffer_record_count:
                self.fail(RuntimeError(
                    f"{self._what} capture buffer overflow at record {self._produced_count} "
                    f"(capacity {self._buffer_record_count}, observed_at_ns={time.time_ns()}); "
                    "no queued record was discarded"
                ))
                return False
            self._produced_count += 1
            self._queue.append(record)
            if self._expected is not None and self._produced_count >= self._expected:
                self._accepting = False
            self._condition.notify_all()
            return True

    def fail(self, error: BaseException) -> None:
        """Fail this capture and notify readers; device lifetime is independent."""

        with self._condition:
            if self._failure is None:
                self._failure = error
            self._accepting = False
            self._ready.set()
            self._condition.notify_all()

    def read(self, n: int, *, timeout: float, exact: bool) -> list[object]:
        """``exact``: wait up to ``timeout`` for all ``n`` records, else raise.

        Not exact: up to ``n`` records, as soon as there is one -- what a
        reader that drains the queue in batches needs, so a backlog is taken
        in one call instead of one call per record.
        """

        requested = int(n)
        if requested <= 0:
            raise ValueError("n must be positive")
        wanted = requested if exact else 1
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while True:
                if self._failure is not None:
                    raise RuntimeError(
                        f"{self._what} failed while producing records: {self._failure}"
                    ) from self._failure
                if len(self._queue) >= wanted:
                    break
                if not self._armed or not self._accepting:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(remaining)
            if exact and len(self._queue) < requested:
                raise TimeoutError(f"{self._what} did not deliver the requested exact records")
            count = requested if exact else min(requested, len(self._queue))
            return [self._queue.popleft() for _ in range(count)]

    def finish(self) -> int:
        """End queue acceptance and join its worker, retaining queued records."""

        with self._condition:
            if not self._armed:
                if self._failure is not None:
                    raise RuntimeError(
                        f"{self._what} failed while producing records: {self._failure}"
                    ) from self._failure
                return self._produced_count
            self._accepting = False
            self._ready.set()
            worker = self._worker
            stop = self._stop
            if stop is not None:
                stop.set()
            self._condition.notify_all()
        if worker is not None:
            worker.join(timeout=self._join_timeout)
        with self._condition:
            if worker is not None and worker.is_alive():
                raise RuntimeError(
                    f"{self._what} did not stop producing within {self._join_timeout:g} s"
                )
            self._worker = None
            self._stop = None
            self._armed = False
            self._condition.notify_all()
            if self._failure is not None:
                raise RuntimeError(
                    f"{self._what} failed while producing records: {self._failure}"
                ) from self._failure
            return self._produced_count

    def close(self) -> None:
        """Release a stopped device's tail and failure, unlike capture Stop."""
        with self._condition:
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError(f"{self._what} is still producing records")
            self._accepting = False
            self._armed = False
            self._queue.clear()
            self._failure = None
            self._worker = None
            self._stop = None
            self._ready.set()
            self._condition.notify_all()


def discovered(instance: str, parameters: dict) -> dict:
    """One hardware-scan find: the id (and name) it suggests, as v2's scan named it, and its parameters."""

    return {"instance_id": instance, "role": instance, "parameters": dict(parameters)}


def underscored(text: object) -> str:
    """``text`` with every character that is not a letter or digit as ``_`` (v2's scan names: ``COM3``, ``ASRL3::INSTR``)."""

    return "".join(c if c.isalnum() else "_" for c in str(text))


def serial_name(port: object) -> str:
    """A COM port as the core names one in ``serial:COM3``: upper-case, without ``\\\\.\\``."""

    text = str(port).strip()
    return (text[4:] if text.startswith("\\\\.\\") else text).upper()


def free_serial_ports(excluded: frozenset[str] = frozenset()) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """This machine's serial ports in order: (those free to open, those bound and passed over).

    ``excluded`` is the core's bound resources (``serial:COM3``); a port an
    open device or another row of the apparatus binds is never opened -- the
    pulse board's UART while its server starts, an IMU that is opening.
    """

    from serial.tools import list_ports

    bound = {serial_name(item.split(":", 1)[1]) for item in excluded if item.lower().startswith("serial:")}
    ports = [info.device for info in sorted(list_ports.comports(), key=lambda item: item.device)]
    return tuple(port for port in ports if serial_name(port) not in bound), tuple(port for port in ports if serial_name(port) in bound)


#: Why a serial port could not be opened, by the OS's error code: one table for every serial open
#: of the application (Rust's twin: ``zlc_pulse::serial::PortFault``). Any other code is ``refused``.
SERIAL_FAULTS = {
    # ERROR_FILE_NOT_FOUND, ERROR_PATH_NOT_FOUND, ERROR_NO_SUCH_DEVICE, ERROR_DEVICE_NOT_CONNECTED: no such port here.
    2: "not present", 3: "not present", 433: "not present", 1167: "not present",
    # ERROR_ACCESS_DENIED, ERROR_SHARING_VIOLATION: a COM port opens once, and another program holds it.
    5: "in use by another program", 32: "in use by another program",
    # ERROR_INVALID_FUNCTION, ERROR_GEN_FAILURE: the name exists and its driver refused the open.
    1: "driver refused", 31: "driver refused",
}


class SerialPortError(OSError):
    """A serial port that could not be opened, in the one wording every serial open uses
    (:func:`open_serial_port`): ``COM10 could not be opened: driver refused (os error 1: Incorrect
    function) -- serial ports on this PC: COM3 (USB-SERIAL CH340)``."""

    def __init__(self, message: str, *, port: str = "", fault: str = "", code: int | None = None) -> None:
        super().__init__(message)
        self.port, self.fault, self.code = port, fault, code

    @property
    def short(self) -> str:
        """The class alone, as a search lists it beside the other ports it tried."""

        return f"could not open ({self.fault})"


class SerialPortUnusable(SerialPortError):
    """The port cannot be used at all (not present, its driver refused, another OS error): a port a
    row names is a record of an earlier find, so the core's Init looks for the device again with its
    type's ``resolve``."""


class SerialPortBusy(SerialPortError):
    """Another program holds the port: the device is still behind it, and looking elsewhere finds nothing."""


def serial_ports_here() -> str:
    """The serial ports this PC has, with their friendly names: ``serial ports on this PC: COM3
    (USB-SERIAL CH340), COM4``, or that it has none."""

    try:
        from serial.tools import list_ports

        listed = sorted(list_ports.comports(), key=lambda item: item.device)
    except Exception as error:  # noqa: BLE001 -- this is already part of an error message
        return f"the serial ports on this PC could not be listed ({error})"
    if not listed:
        return "this PC has no serial port"

    def named(item) -> str:
        friendly = re.sub(r"\s*\(COM\d+\)\s*$", "", str(item.description or "")).strip()
        return f"{item.device} ({friendly})" if friendly and friendly.lower() != "n/a" and friendly != item.device else str(item.device)

    return "serial ports on this PC: " + ", ".join(named(item) for item in listed)


#: pyserial's words for a failed Windows open, ``... FileNotFoundError(2, 'The system cannot find
#: the file specified.', None, 2)``: the system's text and its Windows error code (the last field).
_WINDOWS_ERROR = re.compile(r"\(\d+, (['\"])(?P<text>.*?)\1, None, (?P<code>\d+)\)")


def serial_open_failure(port: str, error: BaseException) -> SerialPortError:
    """``error`` from opening ``port``, in the shared wording and typed by its class; the open or
    search under way is told it (``zlc_plugin.faults.serial_fault``)."""

    code, text = getattr(error, "winerror", None), str(getattr(error, "strerror", "") or "")
    if not isinstance(code, int):
        found = _WINDOWS_ERROR.search(str(error))
        code, text = (int(found["code"]), found["text"]) if found else (None, str(error))
    text = text.strip().rstrip(".\u3002 ")
    fault = SERIAL_FAULTS.get(code, "refused") if code is not None else "refused"
    said = f"os error {code}: {text}" if code is not None else text
    kind = SerialPortBusy if fault == SERIAL_FAULTS[5] else SerialPortUnusable
    # Told to the open or search under way, which carries it to the core (`serial_faults`).
    serial_fault(serial_name(port), fault, code)
    return kind(f"{port} could not be opened: {fault} ({said}) -- {serial_ports_here()}", port=port, fault=fault, code=code)


def open_serial_port(port: str, baud: int, **settings):
    """``serial.Serial(port, baud, **settings)``: every serial open of the devices goes through here,
    so a port that cannot be opened is said one way (:class:`SerialPortError`)."""

    import serial

    try:
        return serial.Serial(port, baud, **settings)
    except OSError as error:  # serial.SerialException is one
        raise serial_open_failure(port, error) from error


def search_serial_ports(ports, probe: Callable[[str], object], *, first: bool = False,
                        unopened: list[str] | None = None) -> tuple[tuple[str, object], ...]:
    """One search over ``ports``: ``probe(port)`` on every one at once, each on a thread of its own,
    so the search takes about one port's probe however many there are (Rust's twin:
    ``zlc_pulse::serial::search_ports``).

    ``probe`` opens its port, asks or listens, releases the port, and answers what it found there
    (falsy: nothing). Each port is probed once, and the search returns only once every probe has
    returned, so no port is still held. A probe that could not open its port raises
    :class:`SerialPortError`: told to ``unopened`` as ``COM4: could not open (in use by another
    program)``; any other failure is a port with nothing on it. Answers ``(port, found)`` in
    ``ports``' order, whichever answered sooner; ``first``: only the first of them.
    """

    ports = tuple(ports)
    outcomes: list[tuple[object, SerialPortError | None]] = [(None, None)] * len(ports)

    def probed(index: int, port: str) -> None:
        try:
            outcomes[index] = (probe(port), None)
        except SerialPortError as error:
            outcomes[index] = (None, error)
        except Exception:  # noqa: BLE001 -- a port that fails its probe has nothing we can use on it
            pass

    threads = [threading.Thread(target=probed, args=(index, port), name=f"zlc-serial-search-{port}", daemon=True)
               for index, port in enumerate(ports)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if unopened is not None:
        unopened.extend(f"{port}: {error.short}" for port, (_, error) in zip(ports, outcomes) if error is not None)
    found = tuple((port, hit) for port, (hit, _) in zip(ports, outcomes) if hit)
    return found[:1] if first else found


__all__ = ["RecordQueue", "SERIAL_FAULTS", "SerialPortBusy", "SerialPortError", "SerialPortUnusable", "discovered", "free_serial_ports",
           "open_serial_port", "search_serial_ports", "serial_name", "serial_open_failure", "serial_ports_here", "underscored"]
