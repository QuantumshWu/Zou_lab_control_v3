"""Every serial open of the devices words its failure one way: the OS's class (not present / in use /
driver refused / refused with its code), and the serial ports this PC does have; typed so the core
knows a port that cannot be used at all (`SerialPortUnusable`, looked for again on Init) from one
another program holds (`SerialPortBusy`). A stand-in `serial` module: no port is opened."""
import pathlib
import sys
import types
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'science' / 'src'))

#: What each port does when opened: pyserial's own words for a failed Windows open, or None (opens).
BEHAVIOUR = {}
LISTED = []


class SerialException(OSError):
    pass


class Serial:
    def __init__(self, port, baud, **settings):
        said = BEHAVIOUR.get(port, f"could not open port {port!r}: FileNotFoundError(2, 'The system cannot find the file specified.', None, 2)")
        if said is not None:
            raise SerialException(said)
        self.port, self.baudrate, self.settings = port, baud, settings
        self.in_waiting = 0

    def read(self, size=1):
        return b''

    def write(self, data):
        return len(data)

    def close(self):
        pass


fake = types.ModuleType('serial')
fake.Serial, fake.SerialException = Serial, SerialException
fake.EIGHTBITS, fake.PARITY_NONE, fake.STOPBITS_ONE = 8, 'N', 1
tools = types.ModuleType('serial.tools')
list_ports = types.ModuleType('serial.tools.list_ports')
list_ports.comports = lambda: list(LISTED)
fake.tools, tools.list_ports = tools, list_ports
sys.modules.update({'serial': fake, 'serial.tools': tools, 'serial.tools.list_ports': list_ports})

from zlc_atom.devices import SerialPortBusy, SerialPortUnusable, open_serial_port, serial_ports_here  # noqa: E402
from zlc_atom.devices.motion.newport_smc100 import Smc100Axis, Smc100Config  # noqa: E402
from zlc_atom.devices.motion.newport_smc100.device_types import DEVICE_TYPES as SMC100_TYPES  # noqa: E402
from zlc_atom.devices.waveform.wheeltec_n100.device_types import DEVICE_TYPES as N100_TYPES  # noqa: E402
from zlc_atom.devices.waveform.wheeltec_n100.source import WheeltecN100Config, WheeltecN100WaveformSource  # noqa: E402
from zlc_atom.devices.waveform.witmotion_hwt3100.device_types import DEVICE_TYPES as HWT_TYPES  # noqa: E402
from zlc_atom.devices.waveform.witmotion_hwt3100.source import Hwt3100Config, Hwt3100WaveformSource  # noqa: E402
from zlc_plugin.wire import error_payload, raise_remote  # noqa: E402

PORTS = 'serial ports on this PC: COM5 (USB-SERIAL CH340), COM7'


class SerialOpenWording(unittest.TestCase):
    def setUp(self):
        BEHAVIOUR.clear()
        LISTED[:] = [types.SimpleNamespace(device='COM7', description='n/a'),
                     types.SimpleNamespace(device='COM5', description='USB-SERIAL CH340 (COM5)')]

    def test_each_class_is_named_with_the_ports_here(self):
        BEHAVIOUR.update({
            'COM10': "could not open port 'COM10': OSError(22, '函数不正确。', None, 1)",
            'COM7': "could not open port 'COM7': PermissionError(13, 'Access is denied.', None, 5)",
            'COM9': "Cannot configure port, something went wrong. Original message: OSError(22, 'The parameter is incorrect.', None, 87)",
        })
        cases = {
            'COM3': (SerialPortUnusable, 'not present', 'COM3 could not be opened: not present (os error 2: The system cannot find the file specified)'),
            'COM10': (SerialPortUnusable, 'driver refused', 'COM10 could not be opened: driver refused (os error 1: 函数不正确)'),
            'COM7': (SerialPortBusy, 'in use by another program', 'COM7 could not be opened: in use by another program (os error 5: Access is denied)'),
            'COM9': (SerialPortUnusable, 'refused', 'COM9 could not be opened: refused (os error 87: The parameter is incorrect)'),
        }
        for port, (kind, fault, failed) in cases.items():
            with self.assertRaises(kind) as raised:
                open_serial_port(port, 115200, timeout=0.05)
            self.assertEqual((str(raised.exception), raised.exception.fault, raised.exception.short),
                             (f'{failed} -- {PORTS}', fault, f'could not open ({fault})'))
        LISTED.clear()
        self.assertEqual(serial_ports_here(), 'this PC has no serial port')

    def test_the_type_crosses_to_the_core_and_back(self):
        with self.assertRaises(SerialPortUnusable) as raised:
            open_serial_port('COM3', 9600)
        payload = error_payload(raised.exception)
        # The core reads `<type>: <message>`, the last segment of the type (CallError's Display).
        self.assertEqual(payload['type'].rsplit(':', 1)[-1], 'SerialPortUnusable')
        with self.assertRaises(SerialPortUnusable) as again:
            raise_remote(payload)
        self.assertEqual(str(again.exception), str(raised.exception))

    def test_every_device_open_says_it_so(self):
        opens = {
            'WHEELTEC N100': lambda: WheeltecN100WaveformSource(WheeltecN100Config(port='COM3')),
            'HWT3100': lambda: Hwt3100WaveformSource(Hwt3100Config(port='COM3')),
            'SMC100': lambda: Smc100Axis(Smc100Config(port='COM3')),
        }
        for name, opened in opens.items():
            with self.subTest(name), self.assertRaises(SerialPortUnusable) as raised:
                opened()
            self.assertTrue(str(raised.exception).startswith('COM3 could not be opened: not present (os error 2: '), str(raised.exception))
            self.assertTrue(str(raised.exception).endswith(PORTS))

    def test_a_resolve_says_which_ports_could_not_be_opened(self):
        BEHAVIOUR['COM7'] = "could not open port 'COM7': PermissionError(13, 'Access is denied.', None, 5)"
        BEHAVIOUR['COM5'] = "could not open port 'COM5': OSError(22, 'Incorrect function.', None, 1)"
        unopened = 'COM5: could not open (driver refused); COM7: could not open (in use by another program)'
        for types_, values, asked in ((N100_TYPES, {'port': '', 'baud': 921600, 'timeout_seconds': 2.0}, 'listened to'),
                                      (HWT_TYPES, {'port': '', 'address': 0}, 'asked'),
                                      (SMC100_TYPES, {'port': '', 'address': 1, 'unit': 'mm'}, 'asked')):
            kind = types_[0]
            with self.subTest(kind.type_id), self.assertRaises(RuntimeError) as raised:
                kind.resolve(kind.parameters.project_values(values), frozenset())
            self.assertIn(f'({asked} COM5, COM7; {unopened})', str(raised.exception))


if __name__ == '__main__':
    unittest.main()
