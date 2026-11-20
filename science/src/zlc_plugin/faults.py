"""The serial ports that would not open while a call ran (``serial_faults``, PLUGIN_PROTOCOL §7).

Every serial open of the devices goes through ``zlc_atom.devices.open_serial_port``, which tells
each failure here (:func:`serial_fault`); an ``open`` or ``resolve`` watches while it runs
(:func:`watching_serial_faults`) and carries what it met beside its answer or on its error. Rust's
twin: ``zlc_pulse::serial::FaultWatch``. Imported where the ports are opened, on every probe
thread of a search: it holds nothing but a lock and the watches.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Any, Iterator

#: The calls under way that watch for serial opens failing, each the list it is told them in.
_WATCHES: list[list[dict[str, Any]]] = []
_LOCK = threading.Lock()


def serial_fault(port: str, fault: str, code: int | None) -> None:
    """A serial open failed in this process: told as ``{port, fault, code}`` to every call
    watching now, whatever thread met it."""

    fact = {"port": str(port), "fault": str(fault), "code": code}
    with _LOCK:
        for met in _WATCHES:
            met.append(dict(fact))


@contextlib.contextmanager
def watching_serial_faults() -> Iterator[list[dict[str, Any]]]:
    """The serial opens that fail while the block runs, on any thread (a search's probes too)."""

    met: list[dict[str, Any]] = []
    with _LOCK:
        _WATCHES.append(met)
    try:
        yield met
    finally:
        with _LOCK:
            _WATCHES[:] = [watch for watch in _WATCHES if watch is not met]


__all__ = ["serial_fault", "watching_serial_faults"]
