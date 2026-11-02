"""The HWT3100-485's Modbus RTU link: frames, checks, and the line's capacity.

Everything here is taken from the vendor's own configuration tool
(``HWT3100-485_Modbus.XML``) and a log it recorded, not from a datasheet:

* function 0x03 reads holding registers, 0x06 writes one; the CRC is
  Modbus CRC-16 (polynomial 0xA001 reflected, initial 0xFFFF), low byte
  first on the wire;
* 0xDB, 0xDC, 0xDD are the magnetic field X, Y, Z as signed 16-bit counts,
  ``count * 13 / 100`` milligauss; 0xDE is the heading, ``count / 10``
  degrees.  One read of four registers from 0xDB returns all four;
* 0xD2 is the line rate, written as a code: 0 -> 9600, 1 -> 115200,
  2 -> 460800 baud.

The logged sensor answered at address 0x00::

    00 03 08 02 AB 10 48 DE DD 03 26 C3 9E  ->  88.8, 541.8, -1102.8 mG, 80.6 deg

What the tool also does and this driver does NOT: it reads 0xD5 (three
registers, meaning stated nowhere), and its "初始化" writes 0 to register
0x00, whose effect is not stated either.  Neither is touched here.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
import time

#: Registers this driver reads, and the one it writes.
FIELD_REGISTER = 0xDB
FIELD_REGISTER_COUNT = 4
BAUD_REGISTER = 0xD2

READ_HOLDING = 0x03
WRITE_SINGLE = 0x06

#: The line rates the sensor offers and the code 0xD2 takes for each.
BAUD_CODES = {9600: 0, 115200: 1, 460800: 2}
SUPPORTED_BAUDS = tuple(sorted(BAUD_CODES))

#: The address the vendor's own log shows the sensor answering at, and the
#: one WitMotion's other Modbus sensors ship with.
LOGGED_ADDRESS = 0x00
WITMOTION_DEFAULT_ADDRESS = 0x50

#: One count of field in microtesla: 13/100 mG, and 1 mG = 0.1 uT.
MICROTESLA_PER_COUNT = 13.0 / 100.0 / 10.0
DEGREES_PER_COUNT = 0.1

#: Bytes of one poll: the 8-byte read request and its reply,
#: address + function + byte count + 4 registers + CRC = 13.
REQUEST_BYTES = 8
REPLY_BYTES = 5 + 2 * FIELD_REGISTER_COUNT
#: Bits per character on an 8N1 line: start, eight data, stop.
BITS_PER_CHAR = 10
#: Modbus over Serial Line (V1.02, 2.5.1.1): frames are separated by at
#: least 3.5 character times, and above 19200 baud a fixed 1.75 ms is the
#: recommended interval.  One gap follows the request, one the reply.
SILENT_CHARS = 3.5
FIXED_SILENCE_SECONDS = 1.75e-3
FIXED_SILENCE_ABOVE_BAUD = 19200


class ModbusError(RuntimeError):
    """A reply that is not a valid answer to the request just sent.

    ``kind`` names which check failed, so a poller can count them apart:
    ``no_reply``, ``crc``, ``address``, ``exception`` or ``malformed``.
    """

    def __init__(self, message: str, kind: str = "malformed") -> None:
        super().__init__(message)
        self.kind = kind


class ModbusTimeout(ModbusError):
    """No (complete) reply within the allowed time."""

    def __init__(self, message: str) -> None:
        super().__init__(message, "no_reply")


def crc16(data: bytes) -> int:
    """Modbus CRC-16 (reflected 0xA001, initial 0xFFFF)."""

    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def with_crc(body: bytes) -> bytes:
    """``body`` followed by its CRC, low byte first."""

    return body + struct.pack("<H", crc16(body))


def read_request(address: int, register: int = FIELD_REGISTER, count: int = FIELD_REGISTER_COUNT) -> bytes:
    return with_crc(struct.pack(">BBHH", address, READ_HOLDING, register, count))


def write_request(address: int, register: int, value: int) -> bytes:
    return with_crc(struct.pack(">BBHH", address, WRITE_SINGLE, register, value))


def expected_length(head: bytes, function: int) -> int | None:
    """How long the reply that starts with ``head`` is, once three bytes are in."""

    if len(head) < 3:
        return None
    if head[1] & 0x80:
        return 5
    if function == READ_HOLDING:
        return 5 + head[2]
    return 8


def check_reply(reply: bytes, *, address: int, function: int) -> bytes:
    """The data bytes of a valid reply, or a ModbusError saying what was wrong."""

    shown = reply.hex(" ")
    if len(reply) < 5:
        raise ModbusError(f"reply too short ({len(reply)} bytes): {shown}")
    if crc16(reply[:-2]) != struct.unpack("<H", reply[-2:])[0]:
        raise ModbusError(f"reply failed its CRC: {shown}", "crc")
    if reply[0] != address:
        raise ModbusError(
            f"reply came from address 0x{reply[0]:02X}, not the configured 0x{address:02X}: {shown}",
            "address",
        )
    if reply[1] == function | 0x80:
        raise ModbusError(f"the sensor refused the request with Modbus exception {reply[2]}: {shown}", "exception")
    if reply[1] != function:
        raise ModbusError(f"reply is for function 0x{reply[1]:02X}, not 0x{function:02X}: {shown}")
    if function == READ_HOLDING:
        if reply[2] != len(reply) - 5:
            raise ModbusError(f"reply byte count {reply[2]} does not match its length: {shown}")
        return reply[3:-2]
    return reply[2:-2]


def decode_field(data: bytes) -> tuple[float, float, float, float]:
    """Bx, By, Bz in microtesla and the heading in degrees, from 0xDB..0xDE."""

    if len(data) != 2 * FIELD_REGISTER_COUNT:
        raise ModbusError(f"expected {2 * FIELD_REGISTER_COUNT} data bytes, got {len(data)}")
    x, y, z, heading = struct.unpack(">4h", data)
    return (
        x * MICROTESLA_PER_COUNT,
        y * MICROTESLA_PER_COUNT,
        z * MICROTESLA_PER_COUNT,
        heading * DEGREES_PER_COUNT,
    )


def poll_seconds(baud: int) -> float:
    """The shortest time one poll occupies the line at ``baud``.

    Request and reply characters plus the inter-frame silence after each.
    The sensor's own turnaround is not in this (nothing states it); a poll
    that takes longer than its period shows up as a gap in a capture.
    """

    char = BITS_PER_CHAR / float(baud)
    silence = SILENT_CHARS * char
    if baud > FIXED_SILENCE_ABOVE_BAUD:
        silence = max(silence, FIXED_SILENCE_SECONDS)
    return (REQUEST_BYTES + REPLY_BYTES) * char + 2 * silence


def max_poll_rate_hz(baud: int) -> float:
    return 1.0 / poll_seconds(baud)


def exchange(port, request: bytes, *, address: int, timeout: float) -> bytes:
    """Send one request and return the data bytes of its checked reply.

    Stale bytes (a late answer to an earlier request) are dropped first, so a
    reply belongs to the request just sent.
    """

    function = request[1]
    port.reset_input_buffer()
    port.write(request)
    flush = getattr(port, "flush", None)
    if callable(flush):
        flush()
    got = bytearray()
    deadline = time.perf_counter() + float(timeout)
    want: int | None = None
    while True:
        need = (want if want is not None else 3) - len(got)
        if need <= 0:
            if want is None:
                want = expected_length(bytes(got), function)
                continue
            break
        chunk = port.read(need)
        if chunk:
            got += chunk
        elif time.perf_counter() >= deadline:
            if got:
                raise ModbusTimeout(f"incomplete reply within {timeout:g} s: {bytes(got).hex(' ')}")
            raise ModbusTimeout(f"no reply within {timeout:g} s")
    return check_reply(bytes(got), address=address, function=function)


def read_field(port, address: int, *, timeout: float) -> tuple[float, float, float, float]:
    return decode_field(exchange(port, read_request(address), address=address, timeout=timeout))


@dataclass(frozen=True)
class BaudOutcome:
    """Where the sensor answers after the Init-time baud procedure, and how it got there."""

    baud: int
    note: str
    switched_from: int | None = None


def answers(port, address: int, *, timeout: float, attempts: int = 2) -> bool:
    for _ in range(attempts):
        try:
            read_field(port, address, timeout=timeout)
            return True
        except ModbusError:
            continue
    return False


def establish_baud(
    port,
    address: int,
    target: int,
    *,
    timeout: float = 0.1,
    settle_seconds: float = 0.05,
    verify_seconds: float = 0.6,
) -> BaudOutcome:
    """Put the sensor on ``target`` baud, or find where it is and say why it stayed.

    ``port`` is an open serial handle; its ``baudrate`` is moved in place.
    1. Ask at ``target``.  Answered: done.
    2. Ask at the other supported rates.  Nothing answers: an error naming
       what was tried.
    3. Found elsewhere: write the target's code to 0xD2 (at the rate it
       answered at), move to ``target`` and check it answers there.
    4. It does not: go back to the rate it last answered at, check it still
       does, and say so.  Nothing is saved (register 0x00 is not written).
    """

    if target not in BAUD_CODES:
        raise ValueError(f"{target} baud is not one of {SUPPORTED_BAUDS}")
    port.baudrate = target
    if answers(port, address, timeout=timeout):
        return BaudOutcome(target, f"answered at {target} baud")
    found = None
    for baud in SUPPORTED_BAUDS:
        if baud == target:
            continue
        port.baudrate = baud
        if answers(port, address, timeout=timeout):
            found = baud
            break
    if found is None:
        raise RuntimeError(
            f"no HWT3100 answered at address 0x{address:02X} at "
            + ", ".join(f"{baud}" for baud in (target, *(b for b in SUPPORTED_BAUDS if b != target)))
            + " baud"
        )
    refused = ""
    try:
        exchange(
            port, write_request(address, BAUD_REGISTER, BAUD_CODES[target]),
            address=address, timeout=timeout,
        )
    except ModbusTimeout:
        # The sensor may switch before (or instead of) echoing the write.
        pass
    except ModbusError as error:
        refused = str(error)
    if not refused:
        time.sleep(settle_seconds)
        port.baudrate = target
        deadline = time.perf_counter() + verify_seconds
        while time.perf_counter() < deadline:
            if answers(port, address, timeout=timeout, attempts=1):
                return BaudOutcome(
                    target, f"found at {found} baud and switched to {target} baud", switched_from=found,
                )
    port.baudrate = found
    if not answers(port, address, timeout=timeout):
        raise RuntimeError(
            f"the HWT3100 answered at {found} baud, was asked to move to {target} baud "
            f"({refused or 'no answer there'}), and now answers at neither"
        )
    why = f"it refused the write: {refused}" if refused else f"it did not answer at {target} baud afterwards"
    return BaudOutcome(found, f"stays at {found} baud: asked to switch to {target} baud but {why}")


__all__ = [
    "BAUD_CODES",
    "BAUD_REGISTER",
    "BaudOutcome",
    "FIELD_REGISTER",
    "LOGGED_ADDRESS",
    "ModbusError",
    "ModbusTimeout",
    "SUPPORTED_BAUDS",
    "WITMOTION_DEFAULT_ADDRESS",
    "answers",
    "check_reply",
    "crc16",
    "decode_field",
    "establish_baud",
    "exchange",
    "max_poll_rate_hz",
    "poll_seconds",
    "read_field",
    "read_request",
    "with_crc",
    "write_request",
]
