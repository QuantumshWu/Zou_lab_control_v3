"""Scalar device-unit boundaries: no VISA session or real device is opened."""
import pathlib
import sys
import threading
import types
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'science' / 'src'))
from zlc_atom.authoring import AuthoringField, TunableField
from zlc_atom.devices.rf.contract import FREQUENCY_FIELD, POWER_FIELD
from zlc_atom.devices.rf.rigol_dg4000.source import RigolDg4000RfSource
from zlc_plugin.tunable import convert_tunable_field, read_tunable_in_unit, tune_in_unit


def field(unit='Hz', current=1001000.0, step=1000.0):
    return TunableField(AuthoringField('frequency', 'float', 'Frequency', 1001000.0,
        minimum=1000.0, maximum=200000000.0, step=step, unit=unit), current, True,
        ('frequency',), device_limits=(1000.0, 200000000.0))


class Device:
    def __init__(self, entry): self.entry = entry; self.writes = []; self.readback = None
    def tunable_fields(self): return (self.entry,)
    def tune(self, name, value):
        self.writes.append(value)
        return value if self.readback is None else self.readback


class NumericUnits(unittest.TestCase):
    def test_shared_scalar_conversion_moves_reading_step_and_bounds_together(self):
        device = Device(field())
        shown = read_tunable_in_unit(device, 'frequency', 'MHz')
        self.assertEqual((shown.current, shown.metadata.step), (1.001, 0.001))
        self.assertEqual((shown.metadata.minimum, shown.metadata.maximum), (0.001, 200.0))
        self.assertEqual(shown.device_limits, (0.001, 200.0))
        self.assertEqual(tune_in_unit(device, 'frequency', 1.001, 'MHz'), 1.001)
        self.assertEqual(device.writes, [1001000.0])

    def test_real_quantized_readback_and_high_precision_step_are_not_rounded(self):
        device = Device(field(step=0.12345678901234566))
        device.readback = 179999999.9997
        self.assertEqual(tune_in_unit(device, 'frequency', 180, 'MHz'), 179.9999999997)
        self.assertEqual(read_tunable_in_unit(device, 'frequency', 'MHz').metadata.step, 1.2345678901234566e-7)

    def test_nonlinear_display_has_no_constant_step_and_does_not_convert_zero(self):
        entry = TunableField(AuthoringField('power', 'float', 'Power', 1.0,
            minimum=0.1, maximum=100.0, step=0.1, unit='mW'), 1.0, True, ('power',))
        shown = convert_tunable_field(entry, 'dBm')
        self.assertIsNone(shown.metadata.step)
        self.assertEqual(shown.current, 0.0)

    def rigol(self):
        # Bypass only transport setup. Invoke the shipped native unit overrides
        # with the same field/readback seam that their SCPI implementation uses.
        device = object.__new__(RigolDg4000RfSource)
        device._condition = threading.Condition(threading.RLock())
        device._routing = {'frequency': ('ch1', FREQUENCY_FIELD), 'power': ('ch1', POWER_FIELD)}
        entry = field()
        device.tunable_fields = lambda: (entry,)
        writes = []
        device.tune = lambda name, value: writes.append(value) or value
        return device, writes

    def test_rigol_native_frequency_override_uses_the_shared_field_conversion(self):
        device, writes = self.rigol()
        shown = device.read_tunable_in_unit('frequency', 'MHz')
        self.assertEqual(shown.metadata.step, 0.001)
        self.assertEqual(shown.current, 1.001)
        self.assertEqual(device.convert_tunable_value('frequency', 1.001, 'MHz', 'Hz'), 1001000.0)
        self.assertEqual(device.tune_in_unit('frequency', 1.001, 'MHz'), 1.001)
        self.assertEqual(writes, [1001000.0])

    def test_rigol_prefixes_never_take_a_power_or_binary_round_trip(self):
        device, _ = self.rigol()
        self.assertEqual(device._convert_amplitude('ch1', 0.043, 'Vpp', 'mVpp'), 43.0)
        self.assertEqual(device._convert_amplitude('ch1', 0.017, 'mW', 'W'), 0.000017)
        self.assertEqual(device._convert_amplitude('ch1', 0.1799999999997, 'Vpp', 'mVpp'), 179.9999999997)

    def test_rigol_scpi_amplitude_writes_and_readbacks_keep_prefix_precision(self):
        device, _ = self.rigol()
        commands = []
        device._amplitudes = {'ch1': {'unit': 'VPP', 'current': 0.1, 'minimum': 0.001, 'maximum': 1.0}}
        device._link = types.SimpleNamespace(write=commands.append, query=lambda _: '0.043')
        self.assertEqual(device._write_power_in_unit('ch1', 43.0, 'mVpp'), 43.0)
        self.assertEqual(float(commands[0].split()[-1].removesuffix('VPP')), 0.043)
        device._link.query = lambda _: '0.1799999999997'
        self.assertEqual(device._write_power_in_unit('ch1', 180, 'mVpp'), 179.9999999997)


if __name__ == '__main__': unittest.main()
