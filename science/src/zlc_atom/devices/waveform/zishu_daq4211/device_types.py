"""The ZishuTech DAQ-4211 this bench can install, and how to find one."""

from __future__ import annotations

from typing import Any, Mapping

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
from zlc_atom.devices import discovered, underscored
from zlc_plugin import DeviceType

from .source import (
    INPUT_RANGES,
    ChannelReading,
    ZishuDaq4211Config,
    ZishuDaq4211WaveformSource,
    discover_daq4211,
)


#: One row per analog input in use, saying what is wired to it.  The card
#: measures volts; what those volts MEAN is the bench's statement, and this
#: is where the bench states it -- an FLC 100 is 50 µT per volt about its own
#: 2.5 V reference, a bare tap is one volt per volt about zero.  Rows sharing
#: a quantity become one published signal with those channels' labels, which
#: is how three FLC 100 heads become one three-component magnetic field.
CHANNEL_COLUMNS = (
    AuthoringField("channel", "int", "AI channel", 0, minimum=0, required=True),
    AuthoringField("quantity", "str", "Quantity", "voltage", required=True),
    AuthoringField("unit", "str", "Unit", "V", required=True),
    AuthoringField("label", "str", "Channel label", "AI0", required=True),
    AuthoringField(
        "unit_per_volt",
        "float",
        "Unit per volt",
        1.0,
        required=True,
        description="the sensor's scale: 50 for an FLC 100 read in uT",
    ),
    AuthoringField(
        "offset_volts",
        "float",
        "Zero at (V)",
        0.0,
        description="the volts that mean zero: 2.5 for an FLC 100 read single-ended",
    ),
)

ZISHU_DAQ4211_SCHEMA = AuthoringSchema(
    (
        AuthoringField("serial", "str", "Device serial", "", required=True),
        AuthoringField(
            "sample_rate_hz",
            "int",
            "Sample rate (Hz)",
            10000,
            minimum=1,
            unit="Hz",
            description="every enabled channel is sampled at this rate, together",
        ),
        AuthoringField(
            "record_samples",
            "int",
            "Samples per record",
            100,
            minimum=1,
            description="one record is one shot; the card streams in ~50 ms packets",
        ),
        AuthoringField(
            "input_range",
            "choice",
            "Input range",
            "10",
            choices=tuple(
                AuthoringChoice(f"{value:g}", f"±{value:g} V")
                for value in sorted(INPUT_RANGES, reverse=True)
            ),
            segmented=True,
            description="all channels share one range",
        ),
        AuthoringField(
            "readings",
            "rows",
            "Channels",
            (
                {
                    "channel": 0,
                    "quantity": "voltage",
                    "unit": "V",
                    "label": "AI0",
                    "unit_per_volt": 1.0,
                    "offset_volts": 0.0,
                },
            ),
            required=True,
            columns=CHANNEL_COLUMNS,
            description="one row per analog input in use",
        ),
        AuthoringField(
            "timeout_seconds", "float", "Record timeout (s)", 2.0, minimum=0.05
        ),
    )
)


def authored_config(values: dict) -> ZishuDaq4211Config:
    """One card's configuration, as the operator wrote it down."""

    authored = ZISHU_DAQ4211_SCHEMA.project_values(values)
    return ZishuDaq4211Config(
        serial=str(authored["serial"]),
        readings=tuple(
            ChannelReading(
                channel=int(row["channel"]),
                quantity=str(row["quantity"]),
                unit=str(row["unit"]),
                label=str(row["label"]),
                unit_per_volt=float(row["unit_per_volt"]),
                offset_volts=float(row["offset_volts"]),
            )
            for row in authored["readings"]
        ),
        sample_rate_hz=int(authored["sample_rate_hz"]),
        record_samples=int(authored["record_samples"]),
        input_range_volts=float(authored["input_range"]),
        timeout_seconds=float(authored["timeout_seconds"]),
    )


def _factory(key: str, values: Mapping[str, Any]) -> ZishuDaq4211WaveformSource:
    return ZishuDaq4211WaveformSource(authored_config(dict(values)))


def _discover() -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    # No card is an empty answer, as every other family gives; a missing
    # libdaq2 still raises, from library(), with where to put it.
    return tuple(
        discovered(f"daq_{underscored(serial)}", ZISHU_DAQ4211_SCHEMA.project_values({"serial": serial}))
        for serial in discover_daq4211()
    ), ()


DEVICE_TYPES = (
    DeviceType(
        "waveform.zishu_daq4211",
        "waveform",
        ZISHU_DAQ4211_SCHEMA,
        ("waveform.source",),
        factory=_factory,
        identity="zishu-daq:{serial}",
        label="ZishuTech DAQ-4211",
        description=(
            "ZishuTech DAQ-4211: eight synchronous 16-bit analog inputs over USB, each "
            "authored channel mapped from volts to its physical quantity; one record per "
            "record_samples samples."
        ),
        # libdaq2 is a process singleton that counts claims per serial.
        process_group="zishu_daq4211",
        discover=_discover,
    ),
)

__all__ = ["CHANNEL_COLUMNS", "DEVICE_TYPES", "ZISHU_DAQ4211_SCHEMA", "authored_config"]
