"""A device's description is read from the instrument once per change and answered after."""
import pathlib
import queue
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'science' / 'src'))
from zlc_plugin import host
from zlc_plugin.declare import DeviceType
from zlc_atom.authoring import AuthoringField, TunableField


class Source:
    """An instrument whose every hardware read is counted; a tune of `limit` moves the frequency's upper limit."""

    def __init__(self):
        self.reads = 0
        self.conversions = 0
        self.limit = 10.0
        self.frequency = 1.0

    def _fields(self):
        return (TunableField(AuthoringField('frequency', 'float', 'Frequency', None, minimum=0.0, maximum=self.limit, unit='Hz'),
                             self.frequency, True, ('frequency',)),
                TunableField(AuthoringField('limit', 'float', 'Limit', None, unit='Hz'), self.limit, False, ('limit',)))

    def refresh_tunable_fields(self):
        self.reads += 1
        return self._fields()

    def tunable_fields(self):
        return self._fields()

    def tune(self, name, value):
        setattr(self, name, float(value))
        return float(value)

    def tunable_values(self):
        self.reads += 1
        return {'frequency': self.frequency, 'limit': self.limit}

    def settings_provenance(self):
        return {}

    def convert_tunable_value(self, name, value, source, target):
        self.conversions += 1
        return tuple(float(item) * 1e-6 for item in value)

    def read_tunable_in_unit(self, name, unit=''):
        from zlc_plugin.tunable import convert_tunable_field
        return convert_tunable_field(next(field for field in self._fields() if field.metadata.name == name), unit)

    def close(self):
        pass


class Owner:
    def __init__(self):
        self.instances = {}
        self.answers = {}
        self._releases = queue.SimpleQueue()

    def decode(self, value):
        return value

    def reply(self, identity, result=None, **kwargs):
        self.answers[identity] = (result, kwargs)


class Description(unittest.TestCase):
    def setUp(self):
        self.owner = Owner()
        self.instance = host.Instance(self.owner, 'rf')
        self.source = Source()
        self.instance.declared = DeviceType('rf.test', 'rf', None, (), lambda key, values: self.source)
        self.instance.adapter = self.source
        self.identity = 0

    def tearDown(self):
        self.instance.lane.put(None)
        self.instance._thread.join(2)
        self.instance.shutdown()

    def ask(self, op, params):
        self.identity += 1
        self.instance._handle({'id': self.identity, 'op': op, 'p': params})
        result, kwargs = self.owner.answers[self.identity]
        self.assertNotIn('error', kwargs, kwargs)
        return result

    def maximum(self, rows):
        return next(row['field']['maximum'] for row in rows if row['field']['name'] == 'frequency')

    def test_fields_are_read_once_and_answered_until_the_device_changes(self):
        spi = lambda method, *args: self.ask('spi', {'method': method, 'args': list(args), 'kwargs': {}})
        for _ in range(23):
            rows = spi('tunable_fields')
            spi('read_tunable_in_unit', 'frequency', '')
        self.assertEqual(self.source.reads, 1, 'every editor and run answered from one reading')
        self.assertEqual(self.maximum(rows), 10.0)
        # Readings are not the description: they stay live and leave it standing.
        self.ask('values', {})
        spi('tunable_values')
        reads = self.source.reads
        spi('tunable_fields')
        self.assertEqual(self.source.reads, reads)
        # A tune changes the device: what it says is read again, once.
        self.ask('tune', {'field': 'limit', 'value': 20.0})
        for _ in range(5):
            rows = spi('tunable_fields')
        self.assertEqual(self.maximum(rows), 20.0)
        self.assertEqual(self.source.reads, reads + 1)
        # So does a tune through a run's proxy (a non-read SPI call).
        spi('tune', 'limit', 30.0)
        self.assertEqual(self.maximum(spi('tunable_fields')), 30.0)
        self.assertEqual(self.source.reads, reads + 2)

    def test_conversions_hold_until_the_device_changes(self):
        convert = lambda: self.ask('spi', {'method': 'convert_tunable_value', 'args': ['frequency', [1.0, 2.0], 'Hz', 'MHz'], 'kwargs': {}})
        for _ in range(12):
            self.assertEqual(list(convert()), [1e-6, 2e-6])
        self.assertEqual(self.source.conversions, 1)
        self.ask('spi', {'method': 'tune', 'args': ['frequency', 3.0], 'kwargs': {}})
        convert()
        self.assertEqual(self.source.conversions, 2)

    def test_device_control_refresh_reads_the_instrument_again(self):
        self.ask('fields', {})
        self.ask('spi', {'method': 'tunable_fields', 'args': [], 'kwargs': {}})
        self.assertEqual(self.source.reads, 1, "a node's ask is answered from Device Control's reading")
        self.source.limit = 40.0  # turned on the front panel: only the operator's Refresh can know
        self.assertEqual(self.maximum(self.ask('fields', {})), 40.0)
        self.assertEqual(self.source.reads, 2)
        self.assertEqual(self.maximum(self.ask('spi', {'method': 'tunable_fields', 'args': [], 'kwargs': {}})), 40.0)
        self.assertEqual(self.source.reads, 2)


class DeviceInfo(unittest.TestCase):
    """What a device states of itself at open (`device_info`), kept with its open in the session journal."""

    def test_an_adapter_without_it_says_nothing(self):
        self.assertIsNone(host.device_info(object()))

    def test_its_facts_are_text_and_the_empty_ones_are_left_out(self):
        class Camera:
            def device_info(self):
                return {'model': 'a2A1920', 'serial': 40123456, 'firmware': None, 'driver': ' '}
        self.assertEqual(host.device_info(Camera()), {'model': 'a2A1920', 'serial': '40123456'})

    def test_a_failure_to_say_is_said_and_fails_nothing(self):
        class Broken:
            def device_info(self):
                raise OSError('link down')
        self.assertEqual(host.device_info(Broken()), {'error': 'OSError: link down'})

    def test_a_scpi_instrument_says_its_idn_without_being_asked_again(self):
        from zlc_atom.devices.waveform.tek_scope.source import TekScopeConfig, TekScopeWaveformSource

        class Link:
            def __init__(self):
                self.asked = []
            def query(self, command):
                self.asked.append(command)
                return {'*IDN?': 'TEKTRONIX,MSO44,C012345,CF:91.1CT FV:1.44.3.433\n',
                        ':HORizontal:RECOrdlength?': '10000'}[command]
            def write(self, command):
                self.asked.append(command)
            def close(self):
                pass

        link = Link()
        scope = TekScopeWaveformSource(TekScopeConfig('USB0::0x0699::0x0527::C012345::INSTR'), link=link)
        asked = len(link.asked)
        info = host.device_info(scope)
        self.assertEqual((info['vendor'], info['model'], info['serial'], info['firmware']),
                         ('TEKTRONIX', 'MSO44', 'C012345', 'CF:91.1CT FV:1.44.3.433'))
        self.assertEqual(info['resource'], 'USB0::0x0699::0x0527::C012345::INSTR')
        self.assertEqual(len(link.asked), asked, 'nothing is asked again')

    def test_an_sdk_version_is_its_installed_distribution_not_a_module_attribute(self):
        import importlib.metadata
        import types
        from unittest.mock import patch
        from zlc_atom.devices.vendor import installed_distribution
        from zlc_atom.devices.camera.pylon import PylonCameraAdapter, PylonCameraConfig
        from python_pylon_lifecycle import Camera

        self.assertEqual(installed_distribution('numpy'), f"numpy {importlib.metadata.version('numpy')}")
        self.assertIsNone(installed_distribution('zlc-no-such-distribution'))
        # The lab's pypylon has no __version__; its metadata says the version all the same.
        installed = importlib.metadata.version
        answer = lambda name: '26.1.0' if name == 'pypylon' else installed(name)  # noqa: E731
        adapter = PylonCameraAdapter(PylonCameraConfig('123'), camera=Camera())
        pylon = types.SimpleNamespace(GetPylonVersionString=lambda: '8.1.0.123')
        with patch.dict(sys.modules, {'pypylon': types.SimpleNamespace(pylon=pylon), 'pypylon.pylon': pylon}), \
                patch.object(importlib.metadata, 'version', answer):
            info = host.device_info(adapter)
        self.assertEqual((info['sdk'], info['driver']), ('pypylon 26.1.0', 'pylon 8.1.0.123'))
        adapter.close()


class DcamDeviceStrings(unittest.TestCase):
    """dcamdev_getstring: DCAMDEV_STRING laid out as dcamapi4.h, answered into the caller's buffer."""

    def test_the_struct_is_the_vendor_layout(self):
        import ctypes
        from zlc_atom.devices.camera.dcam._dcam_driver import _DcamDeviceString

        offsets = {name: getattr(_DcamDeviceString, name).offset for name, _ in _DcamDeviceString._fields_}
        # int32 size, int32 iString, char* text, int32 textbytes, padded to the pointer's alignment.
        self.assertEqual(offsets, {'size': 0, 'string_id': 4, 'text': 8, 'text_bytes': 16})
        self.assertEqual(ctypes.sizeof(_DcamDeviceString), 24)

    def test_a_string_is_asked_of_the_open_handle_and_read_from_the_buffer(self):
        import ctypes
        import types
        from zlc_atom.devices.camera.dcam._dcam_driver import (
            DcamDriverError, DcamIdString, DcamSdkDevice, DcamSdkDriver, _DcamDeviceString)

        asked = []
        answers = {int(DcamIdString.MODEL): b'C15550-20UP', int(DcamIdString.CAMERAID): b'S/N: 000123'}

        def getstring(handle, request):
            request = request.contents
            asked.append((handle, request.size, request.string_id, request.text_bytes))
            text = answers.get(request.string_id)
            if text is None:
                return -2147481566  # DCAMERR_NOTSUPPORT
            ctypes.memmove(request.text, text + b'\0', len(text) + 1)
            return 1

        prototype = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(_DcamDeviceString))
        callback = prototype(getstring)

        class Dll:
            """dcamdev_getstring a real C function pointer; every other export a recorder."""

            def __init__(self):
                self.dcamdev_getstring = ctypes.cast(callback, prototype)

            def __getattr__(self, name):
                export = types.SimpleNamespace()
                setattr(self, name, export)
                return export

        dll = Dll()
        driver = object.__new__(DcamSdkDriver)
        driver._dll = dll
        driver._bind()
        self.assertEqual(dll.dcamdev_getstring.argtypes, [ctypes.c_void_p, ctypes.POINTER(_DcamDeviceString)])
        self.assertIs(dll.dcamdev_getstring.restype, ctypes.c_int32)
        device = DcamSdkDevice(dll, 0x1234)
        self.assertEqual(device.get_string(DcamIdString.MODEL), 'C15550-20UP')
        self.assertEqual(device.get_string(DcamIdString.CAMERAID), 'S/N: 000123')
        with self.assertRaisesRegex(DcamDriverError, r'dcamdev_getstring\(0x04000108\) failed with DCAM error -2147481566'):
            device.get_string(DcamIdString.DCAMAPIVERSION)
        self.assertEqual(asked[0], (0x1234, 24, int(DcamIdString.MODEL), 256))
        device._closed = True
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            device.get_string(DcamIdString.MODEL)

    def test_the_camera_says_what_it_answered_at_open_and_names_what_it_refused(self):
        from zlc_atom.devices.camera.dcam._dcam_driver import DcamDriverError, DcamIdString
        from zlc_atom.devices.camera.dcam.adapter import DcamCameraAdapter

        class Device:
            def __init__(self):
                self.asked = 0

            def get_string(self, string_id):
                self.asked += 1
                if string_id == DcamIdString.BUS:
                    raise DcamDriverError('dcamdev_getstring(0x04000101)', -2147481566)
                return {DcamIdString.VENDOR: 'HAMAMATSU', DcamIdString.MODEL: 'C15550-20UP',
                        DcamIdString.CAMERAID: 'S/N: 000123', DcamIdString.CAMERAVERSION: '1.10.A',
                        DcamIdString.DRIVERVERSION: '5.2.6812', DcamIdString.MODULEVERSION: '24.4.6764',
                        DcamIdString.DCAMAPIVERSION: '4.00'}[string_id]

        adapter = object.__new__(DcamCameraAdapter)
        adapter._device, adapter._open = Device(), True
        adapter._device_facts = adapter._read_device_facts_on_owner()
        asked = adapter._device.asked
        info = host.device_info(adapter)
        self.assertEqual(adapter._device.asked, asked, 'nothing is asked again')
        self.assertEqual({key: info[key] for key in ('vendor', 'model', 'serial', 'firmware', 'driver', 'module', 'sdk')},
                         {'vendor': 'HAMAMATSU', 'model': 'C15550-20UP', 'serial': 'S/N: 000123', 'firmware': '1.10.A',
                          'driver': '5.2.6812', 'module': '24.4.6764', 'sdk': 'DCAM-API 4.00'})
        self.assertEqual(info['unanswered'], 'bus (dcamdev_getstring(0x04000101) failed with DCAM error -2147481566)')


class SerialAndDllDeviceInfo(unittest.TestCase):
    def test_the_hwt3100_says_where_it_answered_without_asking_the_sensor_again(self):
        from python_hwt3100 import Sensor, source

        hwt = source(Sensor(baud=115200))
        try:
            hwt._stop.set()
            hwt._reader.join(1.0)
            requests = len(hwt._serial.requests)
            info = host.device_info(hwt)
            self.assertEqual(len(hwt._serial.requests), requests, 'nothing is asked of the sensor')
            self.assertEqual({key: info[key] for key in ('port', 'address', 'baud')},
                             {'port': 'COM9', 'address': '0x00', 'baud': '115200'})
            self.assertIn('baud_note', info)
        finally:
            hwt.close()

    def test_the_lms_says_what_its_dll_exports_and_nothing_it_does_not(self):
        import types
        from zlc_atom.devices.rf.vaunix_lms import CtypesLmsLibrary, VaunixLmsConfig, VaunixLmsRfSource
        from zlc_atom.devices.simulation.rf import InMemoryLmsLibrary

        path = r'C:\vendor\vnx_fmsynth.dll'

        def library(**exports):
            made = object.__new__(CtypesLmsLibrary)
            made._path, made._dll = path, types.SimpleNamespace(**exports)
            return made

        def named(text, status=None):
            def model(handle, buffer):
                self.assertEqual(handle, 7)
                buffer.value = text
                return len(text) if status is None else status
            return model

        self.assertEqual(library(fnLMS_GetDLLVersion=lambda: 0x0104, fnLMS_GetModelNameA=named(b'LMS-103')).device_info(7),
                         {'dll': path, 'sdk': 'vnx_fmsynth 0x0104', 'model': 'LMS-103'})
        self.assertEqual(library(fnLMS_GetModelName=named(b'LMS-802')).device_info(7), {'dll': path, 'model': 'LMS-802'})
        self.assertEqual(library().device_info(7), {'dll': path})
        self.assertEqual(library(fnLMS_GetModelNameA=named(b'', -0x80000000)).device_info(7),
                         {'dll': path, 'unanswered': 'model (status 0x80000000)'})
        source = VaunixLmsRfSource(VaunixLmsConfig(1001), library=InMemoryLmsLibrary((1001,)))
        self.assertEqual(host.device_info(source), {'serial': '1001', 'model': 'virtual LMS'})
        source.close()


if __name__ == '__main__':
    unittest.main()
