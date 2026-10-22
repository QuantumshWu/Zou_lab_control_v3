"""The virtual RF source this bench can install."""

from __future__ import annotations

from typing import Any, Mapping

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.rf.contract import (
    WINDOW_AUTHORING_FIELDS,
    validate_window_values,
)
from zlc_plugin import DeviceType, PhysicalOutput


#: The virtual brick reuses the REAL RF policy fields: an absent edge and an
#: authored edge mean exactly the same thing on simulation and hardware.
VIRTUAL_RF_SCHEMA = AuthoringSchema(
    (
        AuthoringField("serial", "int", "Serial number", 1001, minimum=1),
        *WINDOW_AUTHORING_FIELDS,
    ),
    validator=validate_window_values,
)


def _rf_factory(key: str, values: Mapping[str, Any]) -> object:
    from zlc_atom.devices.rf.vaunix_lms import VaunixLmsConfig
    from .source import virtual_rf_source

    return virtual_rf_source(VaunixLmsConfig(
        serial=int(values["serial"]),
        frequency_low_hz=values["frequency_low_hz"],
        frequency_high_hz=values["frequency_high_hz"],
        power_low_dbm=values["power_low_dbm"],
        power_high_dbm=values["power_high_dbm"],
    ))


DEVICE_TYPES = (
    DeviceType(
        "rf.virtual",
        "rf",
        VIRTUAL_RF_SCHEMA,
        ("rf.source",),
        factory=_rf_factory,
        label="Virtual RF source",
        description=(
            "The Vaunix LMS driver over an in-memory brick (500 MHz-8 GHz, -40 to +10 dBm): "
            "the same grids, windows and read-backs as the real synthesizer."
        ),
        # Stop all turns each channel's output off (RfSourceBase.make_safe).
        physical_outputs=(PhysicalOutput("rf"),),
    ),
)

__all__ = ["DEVICE_TYPES", "VIRTUAL_RF_SCHEMA"]
