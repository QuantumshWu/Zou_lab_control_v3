"""A plugin's lines in the application's flight journal.

The core keeps one journal of what the application did (``zlc_core::journal``): commands, runs,
board commands, publications, drawn reads.  What only a plugin sees -- a camera frame arriving, a
capture armed or ended, a scan segment fired -- it says here, and the core files the line among its
own, stamped with the time this process observed it.

A note names things and times; it never carries data.  Saying one costs a message to the core and
nothing else: outside a plugin process (a test, an offline helper) it is silently not kept, and a
journal that cannot be reached never fails the work that tried to speak to it.
"""

from __future__ import annotations

import time
from typing import Any


def note(source: str, event: str, **facts: Any) -> None:
    """Say that ``event`` happened to ``source`` now, with these plain facts."""

    from . import host  # the process is made after this module can first be imported

    process = host._PROCESS
    if process is None:
        return
    line = {"us": time.time_ns() // 1000, "src": str(source), "ev": str(event), "facts": facts}
    try:
        process.event("journal", {"lines": [line]}, plain=True)
    except Exception:  # noqa: BLE001 -- the journal is evidence, never a reason to fail
        pass


__all__ = ["note"]
