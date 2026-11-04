"""Discoverable waveform measurement: one node for every waveform source."""

from __future__ import annotations

from collections.abc import Mapping

from zlc_plugin import DeviceRequirement, NodeKind, NodeType

from zlc_atom.devices.waveform.contract import WaveformAcquisitionMode

from .measurement import (
    TRIGGER_FIELDS,
    TRIGGER_OFF,
    WAVEFORM_MEASUREMENT_SCHEMA,
    WaveformMeasurementNode,
    WaveformMeasurementRequest,
    waveform_outputs,
    waveform_preview,
)
from .trigger import TriggerSettings, channel_choices


def _outputs(values: Mapping[str, object], devices: Mapping[str, object] | None = None) -> tuple:
    """One signal per quantity the bound source carries; nothing until one is bound."""

    del values
    source = (devices or {}).get("sampler")
    return () if source is None else waveform_outputs(source.outputs)


def _previews(values: Mapping[str, object], devices: Mapping[str, object] | None = None) -> tuple:
    source = (devices or {}).get("sampler")
    if source is None:
        return ()
    triggered = values.get("trigger_slope", TRIGGER_OFF) != TRIGGER_OFF
    return (waveform_preview(source, trigger_channel=values.get("trigger_channel", "") if triggered else None),)


def _trigger_availability(devices: Mapping[str, object] | None) -> dict[str, str]:
    """A source with its own hardware trigger is triggered there, not here."""

    source = (devices or {}).get("sampler")
    if source is None or source.working_point().acquisition_mode == WaveformAcquisitionMode.FREE_RUNNING.value:
        return {}
    return {name: "this source has its own hardware trigger" for name in TRIGGER_FIELDS}


def _trigger_channels(values: object, resources: object, devices: Mapping[str, object] | None = None) -> dict[str, list]:
    """The bound source's channels, each with the unit its level is typed in."""

    del values, resources
    source = (devices or {}).get("sampler")
    return {} if source is None else {"trigger_channel": channel_choices(source.outputs, source.working_point().time_basis)}


def _build(
    *,
    sampler: object,
    sampler_key: str,
    **values: object,
) -> WaveformMeasurementNode:
    authored = WAVEFORM_MEASUREMENT_SCHEMA.project_values(values)
    return WaveformMeasurementNode(
        sampler=sampler,  # type: ignore[arg-type]
        request=WaveformMeasurementRequest(
            sampler_key=sampler_key,
            repeat=int(authored["repeat"]),
            trigger=None if authored["trigger_slope"] == TRIGGER_OFF else TriggerSettings(
                str(authored["trigger_channel"]), float(authored["trigger_level"]), str(authored["trigger_slope"]),
                int(authored["trigger_record_samples"]), float(authored["trigger_position"]),
            ),
        ),
        producer="waveform_measurement",
    )


LOGIC_NODE = NodeType(
    "waveform_measurement",
    NodeKind.MEASUREMENT,
    WAVEFORM_MEASUREMENT_SCHEMA,
    description=(
        "Acquire every buffered record from a waveform source, retaining its samples "
        "and declared time coordinates for Rolling/Curve plots and fitting. Outputs "
        "are dynamic: one signal per quantity declared by the selected source "
        "(for example scope voltage or IMU acceleration), not a fixed output list. A slow "
        "free-running source can be locked by a software trigger on a channel, level and "
        "slope: each trigger is one record, timed from its level crossing."
    ),
    reports_ready=True,
    # What it publishes is what its instrument carries: four quantities off
    # an IMU packet, one voltage off a scope.  The run freezes every knob
    # the instrument has, because a capture takes the instrument as it
    # stands.
    declare_outputs=_outputs,
    declare_previews=_previews,
    resolve_field_availability=_trigger_availability,
    resolve_field_choices=_trigger_channels,
    device_requirements=(DeviceRequirement("waveform.source", "sampler", None),),
    build=_build,
)

__all__ = ["LOGIC_NODE"]
