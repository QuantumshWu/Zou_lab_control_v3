"""The Rigol DG4000 this bench can install, and how to find one."""

from __future__ import annotations

from typing import Any, Mapping

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices import discovered, underscored
from zlc_atom.devices.rf.contract import (
    WINDOW_AUTHORING_FIELDS,
    validate_window_values,
)
from zlc_plugin import DeviceType, PhysicalOutput

from .source import RigolDg4000Config, RigolDg4000RfSource


#: Bounds are optional bench policy, not instrument facts.  Omitting an edge
#: means no policy limit on that side; it can still be set or cleared later in
#: Device Control through the same shared RF contract.
RIGOL_DG4000_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "resource",
            "str",
            "VISA resource",
            "",
            required=True,
        ),
        AuthoringField(
            "timeout_seconds",
            "float",
            "VISA timeout (s)",
            5.0,
            minimum=0.1,
            unit="s",
        ),
        *WINDOW_AUTHORING_FIELDS,
    ),
    validator=validate_window_values,
)


def _rigol_factory(key: str, values: Mapping[str, Any]) -> RigolDg4000RfSource:
    return RigolDg4000RfSource(RigolDg4000Config(
        resource=str(values["resource"]),
        timeout_seconds=float(values["timeout_seconds"]),
        frequency_low_hz=values["frequency_low_hz"],
        frequency_high_hz=values["frequency_high_hz"],
        power_low_dbm=values["power_low_dbm"],
        power_high_dbm=values["power_high_dbm"],
    ))


def _discover_rigol(excluded: frozenset[str] = frozenset()) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    """Every DG4000 that answers on this machine, named by what it answered.

    The serial is the instrument's own, off ``*IDN?``, so unplugging one and
    scanning again offers the same card rather than a differently numbered
    stranger.  When an instrument gives no serial the resource it was found
    at is the name -- still stable, still that instrument, just longer.
    A VISA that lists nothing to ask is raised by the bus probe itself,
    and an instrument the apparatus binds is not asked.
    """

    from .source import discover_dg4000

    return tuple(
        discovered(f"dg4000_{sighting.serial or underscored(sighting.resource)}",
                   RIGOL_DG4000_SCHEMA.project_values({"resource": sighting.resource}))
        for sighting in discover_dg4000(excluded=excluded)
    ), ()


DEVICE_TYPES = (
    DeviceType(
        "rf.rigol_dg4000",
        "rf",
        RIGOL_DG4000_SCHEMA,
        ("rf.source",),
        factory=_rigol_factory,
        identity="visa:{resource}",
        resources=("visa:{resource}",),
        label="Rigol DG4000",
        description=(
            "Rigol DG4000 two-channel generator as an RF source over VISA SCPI: per channel "
            "frequency, power (dBm, Vpp, Vrms, W through its load), output and FSK hop "
            "frequency, each written then read back, inside the bench's policy window."
        ),
        # Stop all turns each channel's output off (RfSourceBase.make_safe).
        physical_outputs=(PhysicalOutput("rf"),),
        discover=_discover_rigol,
    ),
)

__all__ = ["DEVICE_TYPES", "RIGOL_DG4000_SCHEMA"]
