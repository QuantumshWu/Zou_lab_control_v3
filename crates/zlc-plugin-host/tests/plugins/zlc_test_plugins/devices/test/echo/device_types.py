"""A device for the host's own tests: lends frames, receives arrays, can crash or hang."""

import os
import time
from pathlib import Path

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_plugin import Command, DeviceType, FileParam, Opened, PhysicalOutput, allocate, notifier

SCHEMA = AuthoringSchema((AuthoringField("gain", "float", "Gain", 1.0, minimum=0.0),))
SIZE = AuthoringSchema((AuthoringField("size", "int", "Size", 4, required=True, minimum=1),))
NOTE = AuthoringSchema((AuthoringField("path", "str", "File", required=True), AuthoringField("text", "str", "Text", "")))
FILE = AuthoringSchema((AuthoringField("path", "str", "File", required=True),))


class EchoDevice:
    def __init__(self, values):
        self.opened_with = dict(values)
        self.safe = []

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


SERVED = AuthoringSchema((AuthoringField("port", "int", "Serve on port", 18990, minimum=1, maximum=65535),))


def _served(key, values):
    """A device with a server of its own: its admissions are what Remote did to it."""
    device = EchoDevice(values)
    device.admitted = []
    return Opened(device, lambda: None, admit_peers=device.admitted.append)


DEVICE_TYPES = (
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
