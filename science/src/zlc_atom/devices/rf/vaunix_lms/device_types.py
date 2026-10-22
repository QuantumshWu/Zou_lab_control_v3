"""The Vaunix Lab Brick this bench can install, and how to find one."""

from __future__ import annotations

from typing import Any, Mapping

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices import discovered
from zlc_atom.devices.rf.contract import (
    WINDOW_AUTHORING_FIELDS,
    validate_window_values,
)
from zlc_plugin import DeviceType, PhysicalOutput

from .source import CtypesLmsLibrary, VaunixLmsConfig, VaunixLmsRfSource


VAUNIX_LMS_SCHEMA = AuthoringSchema(
    (
        # No serial yet is a VACANCY, not the number zero: a default that
        # violates its own minimum poisons every draft projection.
        AuthoringField(
            "serial", "int", "Serial number", None, required=True, minimum=1
        ),
        *WINDOW_AUTHORING_FIELDS,
    ),
    validator=validate_window_values,
)


def _vaunix_factory(key: str, values: Mapping[str, Any]) -> VaunixLmsRfSource:
    return VaunixLmsRfSource(VaunixLmsConfig(
        serial=int(values["serial"]),
        frequency_low_hz=values["frequency_low_hz"],
        frequency_high_hz=values["frequency_high_hz"],
        power_low_dbm=values["power_low_dbm"],
        power_high_dbm=values["power_high_dbm"],
    ))


def _discover_vaunix() -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    """Every attached Lab Brick, by serial -- a count read, no opens.

    A missing vendor DLL raises the INSTRUCTION (which file, into which
    folder) rather than an empty result: the scan strip is exactly where
    an operator wondering "why no bricks?" is looking.
    """

    from zlc_atom.devices.vendor import resolve_vendor_file

    library = CtypesLmsLibrary(
        resolve_vendor_file(
            __file__, "vnx_fmsynth.dll", what="the Vaunix LMS SDK (64-bit)"
        )
    )
    return tuple(
        discovered(f"lms_{serial}", VAUNIX_LMS_SCHEMA.project_values({"serial": int(serial)}))
        for serial in library.device_serials()
    ), ()


DEVICE_TYPES = (
    DeviceType(
        "rf.vaunix_lms",
        "rf",
        VAUNIX_LMS_SCHEMA,
        ("rf.source",),
        factory=_vaunix_factory,
        label="Vaunix Lab Brick LMS",
        description=(
            "Vaunix Lab Brick LMS synthesizer over its vendor DLL: frequency on a 10 Hz grid, "
            "power on a 0.25 dB grid (off-grid values are refused, not rounded) and output, "
            "inside the bench's policy window."
        ),
        # Stop all turns each channel's output off (RfSourceBase.make_safe).
        physical_outputs=(PhysicalOutput("rf"),),
        discover=_discover_vaunix,
    ),
)

__all__ = ["DEVICE_TYPES", "VAUNIX_LMS_SCHEMA"]
