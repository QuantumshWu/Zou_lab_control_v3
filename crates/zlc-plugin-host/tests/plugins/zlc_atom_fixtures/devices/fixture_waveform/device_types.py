"""A virtual waveform source of any layout, and a meter of what its run costs.

The virtual IMU and scope have the layouts of their instruments; a host test of
Waveform Measurement also needs eight channels at 1 kHz, or 100 samples a record.
The meter reports from inside the one plugin process, because a device a run has
claimed answers only that run.
"""

import time

import numpy as np
from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.simulation.waveform.source import VirtualWaveformConfig, VirtualWaveformSource
from zlc_atom.devices.waveform.contract import WaveformOutput
from zlc_plugin import DeviceType

WAVEFORM = AuthoringSchema((
    AuthoringField("rate_hz", "float", "Sample rate", 400.0, minimum=1.0),
    AuthoringField("record_samples", "int", "Samples a record", 1, minimum=1),
    AuthoringField("outputs", "int", "Outputs", 1, minimum=1),
    AuthoringField("channels", "int", "Channels", 1, minimum=1),
    AuthoringField("frequency_hz", "float", "Sine on the first channel", 10.0),
    AuthoringField("noise", "float", "Noise, of the amplitude", 0.0, minimum=0.0),
))
METER = AuthoringSchema(())


def _waveform(key, values):
    channels, outputs = int(values["channels"]), int(values["outputs"])
    frequency, noise = float(values["frequency_hz"]), float(values["noise"])
    rng = np.random.default_rng(1)
    parts = np.array_split(np.arange(channels), outputs)
    declared = tuple(
        WaveformOutput(f"q{index}", "V", tuple(f"c{column}" for column in part), tuple(int(c) for c in part))
        for index, part in enumerate(parts))

    def samples(times):
        wave = np.zeros((times.size, channels), dtype=np.float32)
        wave[:, 0] = np.sin(2.0 * np.pi * frequency * times) + noise * rng.standard_normal(times.size)
        return wave

    return VirtualWaveformSource(VirtualWaveformConfig(float(values["rate_hz"]), int(values["record_samples"]), declared),
                                 sample_source=samples)


class Meter:
    """Round trips by operation, and seconds spent in the named calls of this process."""

    def __init__(self):
        self.counts, self.seconds = {}, {}
        self._undo = []

    def _timed(self, owner, name, label=None):
        original = getattr(owner, name, None)
        if original is None:
            return
        tag = label or name

        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.counts[tag] = self.counts.get(tag, 0) + 1
                self.seconds[tag] = self.seconds.get(tag, 0.0) + time.perf_counter() - started
        setattr(owner, name, wrapper)
        self._undo.append((owner, name, original))

    def start(self):
        from zlc_atom.nodes.waveform_measurement import measurement
        from zlc_plugin import context, host, wire
        meter = self
        original = host.Process.request

        def request(process, op, params, **options):
            started = time.perf_counter()
            try:
                return original(process, op, params, **options)
            finally:
                tag = f"rt:{op}"
                meter.counts[tag] = meter.counts.get(tag, 0) + 1
                meter.seconds[tag] = meter.seconds.get(tag, 0.0) + time.perf_counter() - started
        host.Process.request = request
        self._undo.append((host.Process, "request", original))
        self._timed(wire.Channel, "send", "channel.send")
        self._timed(host.Process, "encode", "encode")
        self._timed(host.Process, "decode", "decode")
        self._timed(measurement, "shot_snapshot")
        self._timed(context, "_planes")
        self._timed(context.RunContext, "commit_live")
        return True

    def reset(self):
        self.counts, self.seconds = {}, {}
        return True

    def read(self):
        return {name: [self.counts[name], self.seconds[name]] for name in self.counts}

    def stop(self):
        while self._undo:
            owner, name, original = self._undo.pop()
            setattr(owner, name, original)
        return True

    def close(self):
        self.stop()


DEVICE_TYPES = (
    DeviceType("fixture.waveform", "waveform", WAVEFORM, ("waveform.source",), factory=_waveform),
    DeviceType("fixture.meter", "meter", METER, (), factory=lambda key, values: Meter()),
)
