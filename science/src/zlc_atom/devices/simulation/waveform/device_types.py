"""The virtual waveform sources this bench can install: an IMU and a scope."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.waveform.contract import WaveformOutput
from zlc_atom.devices.waveform.wheeltec_n100 import N100_OUTPUTS
from zlc_plugin import DeviceType, Opened

from ..authoring import SIMULATION_GROUP, acquire_world
from .source import VirtualWaveformConfig, VirtualWaveformSource


VIRTUAL_IMU_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "packet_rate_hz", "float", "Packet rate (Hz)", 400.0, minimum=1.0
        ),
    )
)

VIRTUAL_SCOPE_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "sample_rate_hz", "float", "Sample rate (Hz)", 10000.0, minimum=1.0
        ),
        AuthoringField("record_samples", "int", "Record samples", 1000, minimum=1),
        AuthoringField("channels", "int", "Channels", 2, minimum=1, maximum=4),
    )
)


def _imu_factory(key: str, values: Mapping[str, Any]) -> Opened:
    """An N100 on this world: the packet stream reads the world's bias field."""

    world, release = acquire_world()
    rng = np.random.default_rng((world.config.seed, 0x1D0))
    gravity = np.asarray((0.0, 0.0, 9.80665), dtype=np.float64)

    def samples(times: np.ndarray) -> np.ndarray:
        count = int(times.size)
        out = np.empty((count, 10), dtype=np.float32)
        out[:, 0:3] = np.asarray(world.magnetic_field_microtesla()) + rng.normal(
            0.0, 0.15, (count, 3)
        )
        out[:, 3:6] = rng.normal(0.0, 0.002, (count, 3))
        out[:, 6:9] = gravity + rng.normal(0.0, 0.02, (count, 3))
        out[:, 9] = 300.0 + rng.normal(0.0, 0.01, count)
        return out

    try:
        source = VirtualWaveformSource(
            VirtualWaveformConfig(float(values["packet_rate_hz"]), 1, N100_OUTPUTS),
            sample_source=samples,
        )
    except BaseException:
        release()
        raise

    def close() -> None:
        try:
            source.close()
        finally:
            release()

    return Opened(source, close)


def _scope_factory(key: str, values: Mapping[str, Any]) -> VirtualWaveformSource:
    """A scope whose channels see a 50 Hz line at a different amplitude each."""

    channels = int(values["channels"])
    rng = np.random.default_rng(0x5C0)
    amplitudes = 0.1 * (np.arange(channels, dtype=np.float64) + 1.0)
    phases = np.arange(channels, dtype=np.float64) * (np.pi / 4.0)

    def samples(times: np.ndarray) -> np.ndarray:
        phase = 2.0 * np.pi * 50.0 * times[:, None] + phases[None, :]
        return amplitudes[None, :] * np.sin(phase) + rng.normal(
            0.0, 0.005, (int(times.size), channels)
        )

    return VirtualWaveformSource(
        VirtualWaveformConfig(
            float(values["sample_rate_hz"]),
            int(values["record_samples"]),
            (
                WaveformOutput(
                    "voltage",
                    "V",
                    tuple(f"CH{index + 1}" for index in range(channels)),
                    tuple(range(channels)),
                ),
            ),
        ),
        sample_source=samples,
    )


DEVICE_TYPES = (
    DeviceType(
        "waveform.virtual_imu",
        "waveform",
        VIRTUAL_IMU_SCHEMA,
        ("waveform.source",),
        factory=_imu_factory,
        label="Virtual IMU",
        description=(
            "Simulated WHEELTEC N100 packet stream whose magnetometer reads the simulated "
            "world's bias field (the coil DACs the simulated board last played)."
        ),
        process_group=SIMULATION_GROUP,
    ),
    DeviceType(
        "waveform.virtual_scope",
        "waveform",
        VIRTUAL_SCOPE_SCHEMA,
        ("waveform.source",),
        factory=_scope_factory,
        label="Virtual scope",
        description="Simulated oscilloscope: each channel sees a 50 Hz line at its own amplitude, plus noise.",
    ),
)

__all__ = ["DEVICE_TYPES", "VIRTUAL_IMU_SCHEMA", "VIRTUAL_SCOPE_SCHEMA"]
