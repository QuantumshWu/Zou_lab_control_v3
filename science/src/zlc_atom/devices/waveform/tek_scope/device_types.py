"""The Tektronix scope this bench can install, and how to find one."""

from __future__ import annotations

from typing import Any, Mapping

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices import discovered, underscored
from zlc_atom.devices.visa import identity_fields
from zlc_plugin import DeviceType

from .source import TekScopeConfig, TekScopeWaveformSource, discover_tek_scopes


#: Where the scope is and which channels a record carries, by number.
TEK_SCOPE_SCHEMA = AuthoringSchema(
    (
        AuthoringField("resource", "str", "VISA resource", "", required=True),
        AuthoringField("channels", "str", "Channels (e.g. 1,2)", "1", required=True),
        AuthoringField("timeout_seconds", "float", "Timeout (s)", 5.0, minimum=0.1),
    )
)


def scope_channels(text: object) -> tuple[int, ...]:
    """The channel numbers an operator typed, e.g. ``"1, 3"``."""

    parts = [part.strip() for part in str(text).split(",")]
    try:
        return tuple(int(part) for part in parts if part)
    except ValueError as error:
        raise ValueError(
            f"channels must be channel numbers separated by commas, not {text!r}"
        ) from error


def _factory(key: str, values: Mapping[str, Any]) -> TekScopeWaveformSource:
    return TekScopeWaveformSource(TekScopeConfig(
        resource=str(values["resource"]),
        channels=scope_channels(values["channels"]),
        timeout_seconds=float(values["timeout_seconds"]),
    ))


def _discover() -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    def found(resource: str, identity: str) -> dict[str, Any]:
        fields = identity_fields(identity)
        tail = fields[2] if len(fields) > 2 and fields[2] else underscored(resource)
        return discovered(f"scope_{tail}", TEK_SCOPE_SCHEMA.project_values({"resource": resource}))

    return tuple(found(resource, identity) for resource, identity in discover_tek_scopes()), ()


DEVICE_TYPES = (
    DeviceType(
        "waveform.tek_scope",
        "waveform",
        TEK_SCOPE_SCHEMA,
        ("waveform.source",),
        factory=_factory,
        identity="visa:{resource}",
        label="Tektronix oscilloscope",
        description=(
            "Tektronix DPO/MSO/MDO/TDS scope over VISA: one record per single-sequence "
            "acquisition, every configured channel scaled to volts; time/div and volts/div "
            "are tunable while not armed."
        ),
        discover=_discover,
    ),
)

__all__ = ["DEVICE_TYPES", "TEK_SCOPE_SCHEMA", "scope_channels"]
