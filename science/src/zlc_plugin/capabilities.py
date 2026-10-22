"""Device SPI contracts by capability token.

Paths, not classes: importing this table must not import every device family
(v2's CAPABILITY_TYPES did). A family adds its token here when its contract
module exists in this package.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

CAPABILITIES: dict[str, str] = {
    "camera.adapter": "zlc_atom.devices.camera.contract:CameraAdapter",
    "rf.source": "zlc_atom.devices.rf.contract:RfSource",
    "slm.phase": "zlc_atom.devices.slm:SlmAdapter",
    "waveform.source": "zlc_atom.devices.waveform.contract:WaveformSource",
}

#: v2's tunable quartet: served for every device that implements all four.
TUNABLE = ("tunable_fields", "tune", "tunable_values", "settings_provenance")
#: The device methods that only read (PLUGIN_PROTOCOL §8): the knobs and their
#: provenance, and a device's own unit conversions (an RF amplitude into its
#: channel's load).  A draft query may call these on the devices its draft
#: names; it may change nothing.
READS = ("tunable_fields", "tunable_values", "settings_provenance", "convert_tunable_value", "read_tunable_in_unit")


def reads(adapter: object) -> tuple[str, ...]:
    """The reads ``adapter`` implements."""

    return tuple(name for name in READS if callable(getattr(adapter, name, None)))


def contract(token: str) -> type:
    module, _, name = CAPABILITIES[token].partition(":")
    return getattr(importlib.import_module(module), name)


def members(token: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(properties, methods) a proxy for ``token`` must offer."""

    kind = contract(token)
    properties: list[str] = []
    methods: list[str] = []
    for name in dir(kind):
        if name.startswith("_"):
            continue
        member: Any = inspect.getattr_static(kind, name)
        if isinstance(member, property):
            properties.append(name)
        elif callable(member):
            methods.append(name)
    return tuple(properties), tuple(methods)


__all__ = ["CAPABILITIES", "READS", "TUNABLE", "contract", "members", "reads"]
