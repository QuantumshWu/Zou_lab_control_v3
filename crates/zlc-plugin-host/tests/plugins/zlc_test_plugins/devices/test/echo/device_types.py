"""A device for the host's own tests: lends frames, receives arrays, can crash or hang."""

import os
import math
import time
from pathlib import Path

from zlc_atom.authoring import AuthoringField, AuthoringSchema, TunableField
from zlc_plugin import Command, DeviceType, FileParam, Opened, PhysicalOutput, allocate, notifier

SCHEMA = AuthoringSchema((AuthoringField("gain", "float", "Gain", 1.0, minimum=0.0),))
SIZE = AuthoringSchema((AuthoringField("size", "int", "Size", 4, required=True, minimum=1),))
NOTE = AuthoringSchema((AuthoringField("path", "str", "File", required=True), AuthoringField("text", "str", "Text", "")))
FILE = AuthoringSchema((AuthoringField("path", "str", "File", required=True),))


class EchoDevice:
    def __init__(self, values):
        self.opened_with = dict(values)
        self.safe = []
        self.amplitude = 0.0

    def fail_close_once(self):
        self._fail_close = True

    def close(self):
        if getattr(self, "_fail_close", False):
            self._fail_close = False
            raise RuntimeError("test device handle is still held")

    def tunable_fields(self):
        return (TunableField(AuthoringField("amplitude", "float", "Amplitude", 0.0, unit="dBm", minimum=-30.0, maximum=20.0),
                             self.amplitude, True, ("amplitude",)),)

    def tune(self, name, value):
        assert name == "amplitude"
        self.amplitude = float(value)
        return self.amplitude

    def tune_in_unit(self, name, value, unit):
        if unit == "dBm":
            return self.tune(name, value)
        assert unit == "Vpp"
        self.tune(name, 10 * math.log10((float(value) / (2 * math.sqrt(2))) ** 2 / 100 * 1000))
        return float(value)

    def tunable_values(self):
        return {"amplitude": self.amplitude}

    def settings_provenance(self):
        return {"load": 100}

    def frame(self, size, value):
        array = allocate((size, size), "<u2")
        array[...] = value
        return array

    def total(self, array):
        return int(array.astype("i8").sum())

    def fire(self):
        return "fired"

    def tell(self, size):
        """A notice outside the answer, its frame lent (as an SLM Editor tells its solved draft)."""
        notifier()("told", {"frame": self.frame(size, 7)})
        return size

    def convert_tunable_value(self, name, value, source_unit, target_unit):
        """A read: its own conversion law (here, the gain it was opened with)."""
        return tuple(float(item) * self.opened_with.get("gain", 1.0) for item in value)

    def make_safe(self, output):
        self.safe.append(output)

    def made_safe(self):
        return list(self.safe)

    def crash(self):
        os._exit(3)

    def sleep(self, seconds):
        time.sleep(seconds)


class PlainDevice:
    """No tunable quartet and no snapshot of its own: it does not report its settings."""

    def close(self):
        pass


class FabricDevice(EchoDevice):
    """A concurrent hardware write gated by the fabric integration test."""

    def __init__(self, values):
        super().__init__(values)
        self.output_on = False
        self.fail_safe = False

    def slow_write(self, started, released):
        Path(started).write_text("started", encoding="utf-8")
        until = time.monotonic() + 15
        while not Path(released).exists():
            if time.monotonic() >= until:
                raise TimeoutError("test did not release the hardware write")
            time.sleep(0.01)
        self.output_on = True

    def fail_safe_once(self):
        self.fail_safe = True

    def make_safe(self, output):
        if self.fail_safe:
            self.fail_safe = False
            raise RuntimeError("hardware did not confirm SAFE")
        self.output_on = False
        super().make_safe(output)


class PatternDevice(EchoDevice):
    """A device whose state is more than its knobs (an SLM's pattern) answers its own snapshot."""

    def snapshot(self):
        if getattr(self, "_snapshot_failure", False):
            self._snapshot_failure = False
            raise RuntimeError("snapshot readback unavailable")
        if getattr(self, "_extended_snapshot", False):
            return {"pattern": "flat", "amplitude": self.amplitude, "transport": {"device": "USB", "channel": 2},
                    "snapshot": {"phase": [[0.0, 1.5], [None, 2.0]], "mode": "校准"}, "role": "device-owned role"}
        return {"pattern": "flat", "amplitude": self.amplitude}

    def fail_snapshot_once(self):
        self._snapshot_failure = True

    def extended_snapshot(self):
        self._extended_snapshot = True


SERVED = AuthoringSchema((AuthoringField("port", "int", "Serve on port", 18990, minimum=1, maximum=65535),))
WIRED = AuthoringSchema((AuthoringField("port", "str", "Serial port", "", blank_label="found on Init"),))
#: The serial ports the wired test device is found on, in the order they are tried.
WIRES = ("COM3", "COM4", "COM5")


def _scan_wires(excluded):
    """A scan that is handed the bound resources: it reports them, and finds a device on every other wire."""
    free = [port for port in WIRES if f"serial:{port}" not in excluded]
    return (tuple({"instance_id": f"wired_{port}", "role": f"wired_{port}", "parameters": {"port": port}} for port in free),
            (f"excluded {', '.join(sorted(excluded)) or 'nothing'}",))


def _pin_wire(values, excluded):
    """A row left without its port: the first wire nothing binds."""
    free = [port for port in WIRES if f"serial:{port}" not in excluded]
    if not free:
        raise RuntimeError(f"no free wire (tried none; passed over {', '.join(WIRES)}: bound)")
    return {"parameters": {"port": free[0]}, "note": f"found on {free[0]} (given {values['port']!r})"}


def _served(key, values):
    """A device with a server of its own: its admissions are what Remote did to it."""
    device = EchoDevice(values)
    device.admitted = []
    device.fail_withdraw_once = lambda: setattr(device, "refuse_withdraw", True)

    def admit_peers(admitted):
        if not admitted and getattr(device, "refuse_withdraw", False):
            device.refuse_withdraw = False
            raise RuntimeError("the device server could not refuse peers")
        device.admitted.append(admitted)

    return Opened(device, lambda: None, admit_peers=admit_peers)


class FailedInitDevice:
    def __init__(self, path):
        self.path = path
        self.close_calls = 0

    @property
    def identity(self):
        raise RuntimeError("initialization verification failed")

    def close(self):
        self.close_calls += 1
        if self.close_calls == 1:
            raise RuntimeError("initial cleanup failed; handle retained")
        Path(self.path).write_text("released", encoding="utf-8")


DEVICE_TYPES = (
    DeviceType("test.fabric", "test", SCHEMA, ("test.echo",), factory=lambda key, values: FabricDevice(values),
               concurrent=("slow_write",), physical_outputs=(PhysicalOutput("beam"),)),
    DeviceType("test.plain", "test", SCHEMA, ("test.echo",), factory=lambda key, values: PlainDevice()),
    DeviceType("test.pattern", "test", SCHEMA, ("test.echo",), factory=lambda key, values: PatternDevice(values)),
    DeviceType("test.failed_init", "test", FILE, ("test.echo",), factory=lambda key, values: FailedInitDevice(values["path"])),
    # A device on a serial wire: it binds its port, its scan is handed the bound ones, and a blank port is found on Init.
    DeviceType("test.wired", "test", WIRED, ("test.echo",), factory=lambda key, values: PlainDevice(), resources=("serial:{port}",),
               discover=_scan_wires, resolve=_pin_wire),
    # A scan written before the bound resources existed: it takes nothing and still answers.
    DeviceType("test.scanned", "test", SCHEMA, ("test.echo",), factory=lambda key, values: PlainDevice(),
               discover=lambda: (({"instance_id": "scanned", "role": "scanned", "parameters": {}},), ("scanned as before",))),
    DeviceType("test.served", "test", SERVED, ("test.echo",), factory=_served,
               announce=("test.echo", lambda values: {"port": int(values["port"])})),
    DeviceType("test.echo", "test", SCHEMA, ("test.echo",), factory=lambda key, values: EchoDevice(values),
               physical_outputs=(PhysicalOutput("beam"),),
               commands=(Command("frame_total", "Total of a test frame.", read_only=True,
                                 handler=lambda device, size: device.total(device.frame(size, 2)), params=SIZE),
                         Command("tell", "Tell the core a test frame.", handler=lambda device, size: device.tell(size), params=SIZE),
                         # The operator's files: the core writes what keep wrote, and reads what take reads.
                         Command("keep", "Write a test file.", handler=lambda device, path, text: Path(path).write_text(text, encoding="utf-8"),
                                 params=NOTE, file=FileParam("path", "write")),
                         Command("take", "Read a test file.", handler=lambda device, path: Path(path).read_text(encoding="utf-8"),
                                 params=FILE, file=FileParam("path", "read")))),
)
