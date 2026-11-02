"""The WitMotion HWT3100-485 this bench can install, and how to find one."""

from __future__ import annotations

from typing import Any, Mapping

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
from zlc_atom.devices import discovered, underscored
from zlc_atom.devices.waveform.wheeltec_n100.source import free_serial_ports
from zlc_plugin import DeviceType

from .modbus import LOGGED_ADDRESS, SUPPORTED_BAUDS
from .source import (
    DEFAULT_BAUD,
    Hwt3100Config,
    Hwt3100WaveformSource,
    discover_hwt3100,
)


#: Which port, which Modbus address, which baud to put the sensor on.  The
#: sample rate is NOT authored: as for the N100 it is moved from Device
#: Control (it is this bench's poll rate, 100 Hz on every Init).  The baud
#: is authored and only authored -- Init puts the sensor on it.
WITMOTION_HWT3100_SCHEMA = AuthoringSchema(
    (
        AuthoringField("port", "str", "Serial port", "", blank_label="found on Init"),
        AuthoringField(
            "address", "int", "Modbus address", LOGGED_ADDRESS, minimum=0, maximum=247,
            description="0 as the vendor's own log shows; WitMotion's usual default is 80 (0x50)",
        ),
        AuthoringField(
            "baud",
            "choice",
            "Baud",
            str(DEFAULT_BAUD),
            choices=tuple(AuthoringChoice(str(baud), f"{baud}") for baud in SUPPORTED_BAUDS),
            segmented=True,
            description=(
                "Init puts the sensor on this rate: found at another supported rate, it is "
                "switched (register 0xD2) and checked; it limits the highest sample rate"
            ),
        ),
        AuthoringField("timeout_seconds", "float", "Record timeout (s)", 2.0, minimum=0.05),
    )
)


def _factory(key: str, values: Mapping[str, Any]) -> Hwt3100WaveformSource:
    return Hwt3100WaveformSource(Hwt3100Config(
        port=str(values["port"]),
        address=int(values["address"]),
        baud=int(values["baud"]),
        timeout_seconds=float(values["timeout_seconds"]),
    ))


def _discover(excluded: frozenset[str] = frozenset()) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    # No sensor answering is an empty answer, as every other family gives.
    # A port the apparatus binds is passed over, never opened.
    return tuple(
        discovered(
            f"hwt3100_{underscored(port)}",
            WITMOTION_HWT3100_SCHEMA.project_values({"port": port, "address": address}),
        )
        for port, address, _baud in discover_hwt3100(excluded=excluded)
    ), ()


def _resolve(values: Mapping[str, Any], excluded: frozenset[str]) -> dict[str, Any]:
    """A row left without its port, on Init: the first free port the sensor answers
    a read on at the row's address (any supported baud), pinned into the apparatus."""

    free, passed = free_serial_ports(excluded)
    address = int(values["address"])
    found = discover_hwt3100(addresses=(address,), ports=free, first=True)
    bound = [f"passed over {', '.join(passed)}: bound"] if passed else []
    if not found:
        said = "; ".join([f"asked {', '.join(free) or 'no free port'}", *bound])
        raise RuntimeError(
            f"no WitMotion HWT3100 answered at Modbus address {address} (0x{address:02X}) at "
            f"{', '.join(str(b) for b in SUPPORTED_BAUDS)} baud on the free serial ports ({said}); "
            "set its port and address, or power it and Init again"
        )
    port, _address, baud = found[0]
    return {
        "parameters": {"port": port},
        "note": f"HWT3100 answering on {port} at {baud} baud" + "".join(f" ({said})" for said in bound),
    }


DEVICE_TYPES = (
    DeviceType(
        "waveform.witmotion_hwt3100",
        "waveform",
        WITMOTION_HWT3100_SCHEMA,
        ("waveform.source",),
        factory=_factory,
        identity="witmotion-hwt3100:{port}",
        resources=("serial:{port}",),
        label="WitMotion HWT3100 magnetometer",
        description=(
            "WitMotion HWT3100-485 three-axis magnetometer polled over RS485 Modbus RTU: "
            "magnetic field (uT) and heading (deg), one sample per poll; the poll rate is "
            "tunable (default 100 Hz) up to what the baud carries, and Init sets the baud."
        ),
        discover=_discover,
        resolve=_resolve,
    ),
)

__all__ = ["DEVICE_TYPES", "WITMOTION_HWT3100_SCHEMA"]
