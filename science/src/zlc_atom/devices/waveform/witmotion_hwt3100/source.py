"""WitMotion HWT3100-485 magnetometer, polled over Modbus RTU.

The sensor speaks only when asked, so the HOST's poll rate is the sample
rate: a reader thread sends one read of 0xDB..0xDE on a fixed schedule and
every valid reply is one record -- Bx, By, Bz in microtesla (as the N100
publishes its field) and the heading in degrees, a separate quantity.

The vendor's 100 Hz figure is its ASCII auto-output mode, whose lines carry
no sample counter; Modbus polling is used instead, so every sample is a
request this bench made and a reply whose CRC and address it checked.

Opening the source runs the Init-time baud procedure (``modbus.establish_baud``):
the sensor is put on the authored baud, found at another supported rate and
switched if needed, or left where it answers with the reason recorded.

The rate is a Device Control setting (default 100 Hz).  A rate the line
cannot carry at the baud the sensor is on is refused with the arithmetic;
it is never quietly slowed.

A poll that gets no valid answer -- no reply, a CRC or address error, a
Modbus exception -- is still a record: its values are NaN, which the waveform
measurement publishes as INVALID (the Data validity mask), stamped when the
poll gave up, so the hole is visible in time and in validity.  The schedule
is kept: a poll that runs late is published late, with its real receive
time, and slots the schedule had to skip to catch up are counted, not
invented.  Only a sensor that gives no valid reply for longer than the
record timeout fails a capture.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import logging
import math
import threading
import time
from uuid import uuid4

import numpy as np

from zlc_atom.authoring import AuthoringField, TunableField, TuneRefused
from zlc_atom.devices import RecordQueue, SerialPortError, free_serial_ports, open_serial_port
from zlc_atom.devices.vendor import installed_distribution
from zlc_atom.devices.waveform.contract import (
    WaveformAcquisitionMode,
    WaveformCaptureTerminalRecord,
    WaveformOutput,
    WaveformRecord,
    WaveformWorkingPoint,
)

from . import modbus
from .modbus import (
    LOGGED_ADDRESS,
    SUPPORTED_BAUDS,
    WITMOTION_DEFAULT_ADDRESS,
    ModbusError,
    max_poll_rate_hz,
    poll_seconds,
)

_LOG = logging.getLogger(__name__)

DEFAULT_BAUD = 115200
DEFAULT_RATE_HZ = 100.0
RATE_FIELD = "sample_rate_hz"

#: How long one reply may take before it counts as missing.  At 9600 baud a
#: whole poll occupies the line for ~29 ms; this leaves room for the
#: sensor's own (unstated) turnaround.
REPLY_TIMEOUT_SECONDS = 0.1

#: Two separate quantities, as the N100 keeps its quantities apart: the
#: field in uT over columns 0..2, the heading in degrees in column 3.
HWT3100_OUTPUTS = (
    WaveformOutput("magnetic_field", "uT", ("x", "y", "z"), (0, 1, 2)),
    WaveformOutput("heading", "deg", ("heading",), (3,)),
)
_COLUMNS = 4
#: A poll with no valid answer: every column NaN.  The waveform measurement
#: publishes non-finite samples as invalid (``shot_snapshot``'s validity).
_MISSING = (math.nan,) * _COLUMNS


@dataclass(frozen=True)
class Hwt3100Config:
    """Which port, which Modbus address, which baud to put the sensor on."""

    port: str
    address: int = LOGGED_ADDRESS
    baud: int = DEFAULT_BAUD
    timeout_seconds: float = 2.0

    def __post_init__(self) -> None:
        port = str(self.port).strip()
        if not port:
            raise ValueError("serial port is required")
        if isinstance(self.address, bool) or not 0 <= int(self.address) <= 247:
            raise ValueError("Modbus address must be 0..247")
        if int(self.baud) not in SUPPORTED_BAUDS:
            raise ValueError(f"baud must be one of {SUPPORTED_BAUDS}")
        if float(self.timeout_seconds) <= 0:
            raise ValueError("timeout_seconds must be positive")
        object.__setattr__(self, "port", port)
        object.__setattr__(self, "address", int(self.address))
        object.__setattr__(self, "baud", int(self.baud))
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))


def open_serial(port: str, baud: int):
    return open_serial_port(port, baud, timeout=0.005, write_timeout=1.0)


def rate_refusal(rate_hz: float, baud: int) -> str:
    """Why ``rate_hz`` cannot be polled at ``baud``, or "" when it can."""

    ceiling = max_poll_rate_hz(baud)
    if rate_hz <= ceiling:
        return ""
    return (
        f"{rate_hz:g} Hz cannot be polled at {baud} baud: one poll is an 8-byte request and "
        f"a 13-byte reply with a Modbus inter-frame silence after each, "
        f"{poll_seconds(baud) * 1e3:.2f} ms of line time, so at most {ceiling:.1f} Hz"
    )


class Hwt3100WaveformSource:
    """Every polled reply is one record of four samples."""

    def __init__(self, config: Hwt3100Config, *, serial_port=None, establish=modbus.establish_baud) -> None:
        self.config = config
        if serial_port is None:
            # A port that cannot be opened says so in the shared wording (SerialPortError).
            serial_port = open_serial(config.port, config.baud)
        self._serial = serial_port
        self._stop = threading.Event()
        self._settings_lock = threading.RLock()
        self._capture_lock = threading.Lock()
        self._device_session_id = uuid4().hex
        self._settings_epoch = 0
        self._records = RecordQueue("the HWT3100", join_timeout_seconds=config.timeout_seconds)
        self._reader: threading.Thread | None = None
        self._first_reply = threading.Event()
        self._last_error = ""
        self._round_trip_max = 0.0
        self._reschedule = threading.Event()
        #: Polls that gave no valid sample, by ``ModbusError.kind``, and
        #: schedule slots skipped to catch up: since open, and in the
        #: current (or last) capture.
        self._invalid: Counter[str] = Counter()
        self._skipped = 0
        self._capture_invalid: Counter[str] = Counter()
        self._capture_skipped = 0
        self._last_valid_at = 0.0
        try:
            try:
                outcome = establish(self._serial, config.address, config.baud)
            except (RuntimeError, ValueError):
                raise
            except Exception as error:  # noqa: BLE001 -- a port that will not take a baud
                raise RuntimeError(
                    f"{config.port} refused a serial setting while looking for the HWT3100 "
                    f"(baud {SUPPORTED_BAUDS}): {error}"
                ) from error
            self._baud = outcome.baud
            self._baud_note = outcome.note
            ceiling = max_poll_rate_hz(self._baud)
            if DEFAULT_RATE_HZ <= ceiling:
                self._rate = DEFAULT_RATE_HZ
                self._rate_note = ""
            else:
                self._rate = float(math.floor(ceiling))
                self._rate_note = (
                    f"started at {self._rate:g} Hz, not {DEFAULT_RATE_HZ:g} Hz: "
                    + rate_refusal(DEFAULT_RATE_HZ, self._baud)
                )
            _LOG.info(
                "HWT3100 OPEN port=%s address=0x%02X %s; polling %g Hz%s device=%s",
                config.port, config.address, self._baud_note, self._rate,
                f" ({self._rate_note})" if self._rate_note else "", self.identity,
            )
            self._reader = threading.Thread(
                target=self._poll_loop, name=f"zlc-hwt3100-{config.port}", daemon=True
            )
            self._reader.start()
            if not self._first_reply.wait(config.timeout_seconds):
                raise RuntimeError(
                    f"the HWT3100 on {config.port} answered the baud check but no poll "
                    f"succeeded within {config.timeout_seconds:g} s: {self._last_error}"
                )
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------- polling
    def _poll_loop(self) -> None:
        request = modbus.read_request(self.config.address)
        period = 1.0 / self._rate
        due = time.perf_counter()
        try:
            while not self._stop.is_set():
                if self._reschedule.is_set():
                    self._reschedule.clear()
                    period = 1.0 / self._rate
                    due = time.perf_counter()
                wait = due - time.perf_counter()
                if wait > 0:
                    # time.sleep, not Event.wait: on Windows only sleep uses
                    # the high-resolution timer a 10 ms period needs.
                    time.sleep(min(wait, 0.05))
                    continue
                sent = time.perf_counter()
                try:
                    data = modbus.exchange(
                        self._serial, request, address=self.config.address,
                        timeout=REPLY_TIMEOUT_SECONDS,
                    )
                    values = modbus.decode_field(data)
                    failure = None
                except ModbusError as error:
                    values = _MISSING
                    failure = error
                stamp = time.perf_counter()
                received = time.time_ns()
                with self._capture_lock:
                    capturing = self._records.accepting
                    if failure is None:
                        self._round_trip_max = max(self._round_trip_max, stamp - sent)
                        self._last_valid_at = stamp
                        self._first_reply.set()
                    else:
                        self._last_error = str(failure)
                        self._invalid[failure.kind] += 1
                        if capturing:
                            self._capture_invalid[failure.kind] += 1
                    if capturing:
                        self._records.push(WaveformRecord(
                            np.asarray(values, dtype=np.float32).reshape(1, _COLUMNS),
                            self._records.produced_count, stamp, received,
                        ))
                        silent = stamp - self._last_valid_at
                        if failure is not None and silent > self.config.timeout_seconds:
                            self._records.fail(RuntimeError(
                                f"the HWT3100 on {self.config.port} gave no valid reply for "
                                f"{silent:.2f} s, longer than the {self.config.timeout_seconds:g} s "
                                f"record timeout; last: {failure}"
                            ))
                due += period
                behind = time.perf_counter() - due
                if behind > period:
                    # A whole slot or more behind: catch up to now rather than
                    # firing the missed polls back to back; count what was skipped.
                    skipped = int(behind // period)
                    with self._capture_lock:
                        self._skipped += skipped
                        if self._records.accepting:
                            self._capture_skipped += skipped
                    due += skipped * period
        except BaseException as error:  # noqa: BLE001 -- surfaced to the reader of records
            self._last_error = str(error)
            self._records.fail(error)

    # ------------------------------------------------------------ contract
    @property
    def timeout(self) -> float:
        return self.config.timeout_seconds

    @property
    def identity(self) -> str:
        return f"witmotion-hwt3100:{self.config.port}"

    def device_info(self) -> dict[str, object]:
        """Where the sensor answered when it opened -- port, Modbus address, the baud it is on and
        how that baud was found -- and the pyserial that reached it.  The sensor has no register
        that states its model or firmware (``modbus``), so nothing is asked."""

        return {
            "port": self.config.port,
            "address": f"0x{self.config.address:02X}",
            "baud": self._baud,
            "baud_note": self._baud_note,
            "driver": installed_distribution("pyserial"),
        }

    @property
    def outputs(self) -> tuple[WaveformOutput, ...]:
        return HWT3100_OUTPUTS

    @property
    def record_samples(self) -> int:
        return 1

    @property
    def baud(self) -> int:
        return self._baud

    def working_point(self) -> WaveformWorkingPoint:
        return WaveformWorkingPoint(
            WaveformAcquisitionMode.FREE_RUNNING,
            1.0 / self._rate,
            {
                "port": self.config.port,
                "address": self.config.address,
                "baud": self._baud,
                "baud_note": self._baud_note,
                "sample_rate_hz": self._rate,
                "rate_note": self._rate_note,
                "max_poll_rate_hz": round(max_poll_rate_hz(self._baud), 3),
                "slowest_round_trip_ms": round(self._round_trip_max * 1e3, 3),
                "invalid_polls": dict(self._invalid),
                "skipped_slots": self._skipped,
            },
            time_basis="host_receive",
        )

    # -------------------------------------------------------------- knobs
    def _rate_field(self) -> TunableField:
        ceiling = max_poll_rate_hz(self._baud)
        floor = 1.0 / self.config.timeout_seconds
        return TunableField(
            metadata=AuthoringField(
                RATE_FIELD,
                "float",
                "Sample rate",
                None,
                minimum=floor,
                maximum=ceiling,
                unit="Hz",
                description=(
                    f"how often this bench polls the sensor, and so its sample rate. At "
                    f"{self._baud} baud the line carries at most {ceiling:.1f} Hz; below "
                    f"{floor:g} Hz a sample would not arrive within the record timeout"
                ),
            ),
            current=self._rate,
            live_write=True,
            dependency_group=(RATE_FIELD,),
        )

    def tunable_fields(self) -> tuple[TunableField, ...]:
        with self._settings_lock:
            return (self._rate_field(),)

    def tunable_values(self) -> dict[str, object]:
        with self._settings_lock:
            return {RATE_FIELD: self._rate}

    def settings_provenance(self) -> dict[str, object]:
        with self._settings_lock:
            return {"device_session_id": self._device_session_id, "settings_epoch": self._settings_epoch}

    def tune(self, name: str, value: object) -> object:
        if str(name) != RATE_FIELD:
            raise ValueError(f"this sensor has no setting {name!r}; it offers {RATE_FIELD!r}")
        try:
            if isinstance(value, bool):
                raise TypeError
            wanted = float(value)
        except (TypeError, ValueError):
            raise TuneRefused(f"{value!r} is not a rate in Hz") from None
        if not math.isfinite(wanted) or wanted <= 0:
            raise TuneRefused(f"{value!r} is not a positive rate in Hz")
        with self._settings_lock:
            refusal = rate_refusal(wanted, self._baud)
            if refusal:
                raise TuneRefused(refusal)
            floor = 1.0 / self.config.timeout_seconds
            if wanted < floor:
                raise TuneRefused(
                    f"{wanted:g} Hz is below {floor:g} Hz: a sample would not arrive within "
                    f"the {self.config.timeout_seconds:g} s record timeout"
                )
            if self._records.armed:
                raise RuntimeError("the sample rate stamps every record, so a capture must be finished first")
            self._rate = wanted
            self._rate_note = ""
            self._settings_epoch += 1
            self._reschedule.set()
            return self._rate

    # ------------------------------------------------------------- capture
    def arm(self, records: int | None, *, buffer_record_count: int) -> None:
        with self._settings_lock:
            with self._capture_lock:
                self._records.arm(records, buffer_record_count=buffer_record_count)
                self._capture_invalid = Counter()
                self._capture_skipped = 0
                if self._reader is None or not self._reader.is_alive():
                    self._records.fail(RuntimeError(
                        f"the HWT3100 poll thread is not running: {self._last_error}"
                    ))
                else:
                    self._records.mark_ready()
            try:
                self._records.wait_ready(self.timeout)
            except BaseException:
                self._records.finish()
                raise

    def read_records(self, n: int, *, timeout: float, exact: bool) -> list[WaveformRecord]:
        return self._records.read(n, timeout=timeout, exact=exact)

    def finish_record_capture(self) -> WaveformCaptureTerminalRecord:
        with self._capture_lock:
            try:
                produced = self._records.finish()
            finally:
                _LOG.info(
                    "HWT3100 CAPTURE %d records, %d invalid %s, %d skipped slots device=%s",
                    self._records.produced_count, sum(self._capture_invalid.values()),
                    dict(self._capture_invalid), self._capture_skipped, self.identity,
                )
            return WaveformCaptureTerminalRecord(produced, True, not self._records.pending_count, True)

    @property
    def capture_gaps(self) -> dict[str, object]:
        """Invalid polls by kind and skipped slots in the current (or last) capture."""

        with self._capture_lock:
            return {"invalid": dict(self._capture_invalid), "skipped_slots": self._capture_skipped}

    def capture_state(self) -> bool:
        return self._records.armed

    def close(self) -> None:
        with self._settings_lock:
            self._stop.set()
            try:
                if self._records.armed:
                    self._records.finish()
            finally:
                reader = self._reader
                if reader is not None and reader.is_alive() and reader is not threading.current_thread():
                    reader.join(timeout=self.config.timeout_seconds)
                if reader is not None and reader.is_alive():
                    raise RuntimeError("HWT3100 poll thread did not stop; serial port retained")
                self._serial.close()
                self._records.close()


# ------------------------------------------------------------------ finding
def probe_port(device: str, addresses: tuple[int, ...], *, opener=open_serial, timeout: float = REPLY_TIMEOUT_SECONDS,
               unopened: list[str] | None = None):
    """``(address, baud)`` an HWT3100 answers a field read at on ``device``, or None.

    Discovery cannot listen for this sensor -- it speaks only when asked --
    so each candidate port gets one READ request (function 0x03) per
    address and supported baud.  Nothing is ever written to a register.
    A port that cannot be opened is told to ``unopened`` with its class.
    """

    try:
        port = opener(device, SUPPORTED_BAUDS[0])
    except SerialPortError as error:
        if unopened is not None:
            unopened.append(f"{device}: {error.short}")
        return None
    except Exception:  # noqa: BLE001 -- a stand-in opener's own failure
        return None
    try:
        for baud in (DEFAULT_BAUD, *(b for b in SUPPORTED_BAUDS if b != DEFAULT_BAUD)):
            port.baudrate = baud
            for address in addresses:
                if modbus.answers(port, address, timeout=timeout, attempts=1):
                    return address, baud
    except Exception:  # noqa: BLE001 -- a port that fails mid-probe has no sensor we can use
        return None
    finally:
        try:
            port.close()
        except Exception:  # noqa: BLE001
            pass
    return None


def discover_hwt3100(
    *,
    addresses: tuple[int, ...] = (LOGGED_ADDRESS, WITMOTION_DEFAULT_ADDRESS),
    excluded: frozenset[str] = frozenset(),
    ports: tuple[str, ...] | None = None,
    first: bool = False,
    opener=open_serial,
    unopened: list[str] | None = None,
) -> tuple[tuple[str, int, int], ...]:
    """``(port, address, baud)`` for every free serial port an HWT3100 answers on."""

    found = []
    for device in free_serial_ports(excluded)[0] if ports is None else ports:
        hit = probe_port(device, addresses, opener=opener, unopened=unopened)
        if hit is not None:
            found.append((device, *hit))
            if first:
                break
    return tuple(found)


__all__ = [
    "DEFAULT_BAUD",
    "DEFAULT_RATE_HZ",
    "HWT3100_OUTPUTS",
    "Hwt3100Config",
    "Hwt3100WaveformSource",
    "RATE_FIELD",
    "discover_hwt3100",
    "probe_port",
    "rate_refusal",
]
