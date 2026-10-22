"""The WHEELTEC N100 this bench can install, and how to find one."""

from __future__ import annotations

from typing import Any, Mapping

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices import discovered, underscored
from zlc_plugin import DeviceType

from .source import (
    DEFAULT_BAUD,
    WheeltecN100Config,
    WheeltecN100WaveformSource,
    discover_n100,
)


#: Which port, at the line rate the module was configured to.  The packet
#: rate is NOT authored: it is a setting that lives in the module's own
#: flash, so authoring it here would be a second place it is written down
#: and a bench that disagreed with its own hardware after a power cycle.
#: It is measured off the stream when the port opens, and moved from Device
#: Control, where the module answers with the rate it actually took.
WHEELTEC_N100_SCHEMA = AuthoringSchema(
    (
        AuthoringField("port", "str", "Serial port", "", required=True),
        AuthoringField("baud", "int", "Baud", DEFAULT_BAUD, minimum=1200),
        AuthoringField(
            "timeout_seconds", "float", "Record timeout (s)", 2.0, minimum=0.05
        ),
    )
)


def _factory(key: str, values: Mapping[str, Any]) -> WheeltecN100WaveformSource:
    return WheeltecN100WaveformSource(WheeltecN100Config(
        port=str(values["port"]),
        baud=int(values["baud"]),
        timeout_seconds=float(values["timeout_seconds"]),
    ))


def _discover() -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    # No module on the air is an empty answer, as every other family gives:
    # a bench without an N100 has nothing to offer, not a fault to report.
    return tuple(
        discovered(f"n100_{underscored(port)}", WHEELTEC_N100_SCHEMA.project_values({"port": port}))
        for port in discover_n100()
    ), ()


DEVICE_TYPES = (
    DeviceType(
        "waveform.wheeltec_n100",
        "waveform",
        WHEELTEC_N100_SCHEMA,
        ("waveform.source",),
        factory=_factory,
        identity="wheeltec-n100:{port}",
        label="WHEELTEC N100 IMU",
        description=(
            "WHEELTEC N100 nine-axis IMU over its FDILink serial stream: magnetic field (uT), "
            "angular rate, acceleration and temperature, one sample per record; the packet "
            "rate is tunable and lives in the module's flash."
        ),
        discover=_discover,
    ),
)

__all__ = ["DEVICE_TYPES", "WHEELTEC_N100_SCHEMA"]
