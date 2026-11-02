"""WitMotion HWT3100-485 over a simulated Modbus RTU sensor: no serial port is opened."""
import pathlib
import struct
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'science' / 'src'))
from zlc_atom.authoring import TuneRefused
from zlc_atom.devices.waveform.witmotion_hwt3100 import modbus
from zlc_atom.devices.waveform.witmotion_hwt3100.device_types import DEVICE_TYPES, WITMOTION_HWT3100_SCHEMA
from zlc_atom.devices.waveform.witmotion_hwt3100.source import (
    Hwt3100Config, Hwt3100WaveformSource, discover_hwt3100, rate_refusal)

#: The reply the vendor's own log recorded at address 0x00.
LOGGED_REPLY = bytes.fromhex('00 03 08 02 AB 10 48 DE DD 03 26 C3 9E')


class Sensor:
    """One simulated HWT3100 on the bus: registers, address, line rate."""

    def __init__(self, address=0x00, baud=9600, registers=(0x02AB, 0x1048, 0xDEDD, 0x0326)):
        self.address, self.baud = address, baud
        self.registers = dict(zip(range(0xDB, 0xDF), registers))
        self.lock = threading.Lock()
        self.corrupt_crc = False
        self.reply_from = None          # answer from another address
        self.silent = False             # never answer
        self.refuse_baud_write = False  # Modbus exception on 0xD2
        self.ignore_baud_write = False  # echo, but stay on the old rate
        self.writes = []                # (register, value)
        self.drop = 0                   # leave the next n requests unanswered
        self.corrupt = 0                # corrupt the CRC of the next n replies
        self.misaddress = 0             # answer the next n from address 0x50

    def _once(self, name):
        if getattr(self, name):
            setattr(self, name, getattr(self, name) - 1)
            return True
        return False

    def answer(self, frame):
        if self.silent or len(frame) != 8 or modbus.crc16(frame[:-2]) != struct.unpack('<H', frame[-2:])[0]:
            return b''
        address, function, register, value = struct.unpack('>BBHH', frame[:6])
        if address != self.address or self._once('drop'):
            return b''
        source = self.address if self.reply_from is None else self.reply_from
        misaddressed = self._once('misaddress')
        if misaddressed:
            source = 0x50
        if function == 0x03:
            words = [self.registers.get(register + i, 0) & 0xFFFF for i in range(value)]
            reply = modbus.with_crc(struct.pack('>BBB', source, 3, 2 * value) + b''.join(struct.pack('>H', w) for w in words))
        elif function == 0x06:
            self.writes.append((register, value))
            if register == 0xD2 and self.refuse_baud_write:
                return modbus.with_crc(bytes([source, 0x86, 0x02]))
            reply = bytes(frame)
            if register == 0xD2 and not self.ignore_baud_write:
                self.baud = {v: k for k, v in modbus.BAUD_CODES.items()}[value]
        else:
            return b''
        if self.corrupt_crc or (not misaddressed and self._once('corrupt')):
            reply = reply[:-1] + bytes([reply[-1] ^ 0xFF])
        return reply


class Port:
    """A serial handle onto ``Sensor``: at the wrong baud the sensor hears noise."""

    def __init__(self, sensor, baud):
        self.sensor, self.baudrate, self.closed = sensor, baud, False
        self.rx = bytearray()
        self.requests = []

    @property
    def in_waiting(self):
        return len(self.rx)

    def reset_input_buffer(self):
        self.rx.clear()

    def write(self, data):
        self.requests.append((self.baudrate, bytes(data)))
        with self.sensor.lock:
            if self.baudrate == self.sensor.baud:
                old = self.sensor.baud
                reply = self.sensor.answer(bytes(data))
                # The echo of a baud write still goes out at the old rate.
                self.rx += reply if self.baudrate == old else b''
            else:
                self.rx += b''  # a wrong-rate request is noise to the sensor; nothing comes back
        return len(data)

    def flush(self):
        pass

    def read(self, n):
        if not self.rx:
            time.sleep(0.001)
            return b''
        out = bytes(self.rx[:n])
        del self.rx[:n]
        return out

    def close(self):
        self.closed = True


def source(sensor, *, baud=115200, address=0x00, timeout=1.0):
    return Hwt3100WaveformSource(
        Hwt3100Config('COM9', address=address, baud=baud, timeout_seconds=timeout),
        serial_port=Port(sensor, 9600))


class Frames(unittest.TestCase):
    def test_logged_reply_decodes_as_the_vendor_tool_showed(self):
        data = modbus.check_reply(LOGGED_REPLY, address=0, function=3)
        bx, by, bz, heading = modbus.decode_field(data)
        # The vendor tool shows mG to 0.1: 88.8, 541.8, -1102.8 mG; 80.6 deg.
        self.assertEqual([round(v * 10, 1) for v in (bx, by, bz)], [88.8, 541.8, -1102.8])
        self.assertEqual(struct.unpack('>h', bytes.fromhex('DEDD'))[0], -8483)  # signed int16
        self.assertAlmostEqual(bz, -8483 * 0.013)
        self.assertAlmostEqual(heading, 80.6)

    def test_request_crc_is_low_byte_first(self):
        # The vendor log's reply CRC (C3 9E) is low byte first; a request must be built the same way.
        self.assertEqual(modbus.with_crc(LOGGED_REPLY[:-2]), LOGGED_REPLY)
        request = modbus.read_request(0x00)
        self.assertEqual(request[:6], bytes.fromhex('00 03 00 DB 00 04'))
        self.assertEqual(modbus.crc16(request[:-2]), struct.unpack('<H', request[-2:])[0])
        self.assertEqual(modbus.write_request(0x50, 0xD2, 1)[:6], bytes.fromhex('50 06 00 D2 00 01'))

    def test_bad_frames_are_refused_with_the_reason(self):
        with self.assertRaisesRegex(modbus.ModbusError, 'CRC'):
            modbus.check_reply(LOGGED_REPLY[:-1] + b'\x00', address=0, function=3)
        with self.assertRaisesRegex(modbus.ModbusError, 'address 0x00, not the configured 0x50'):
            modbus.check_reply(LOGGED_REPLY, address=0x50, function=3)
        with self.assertRaisesRegex(modbus.ModbusError, 'exception 2'):
            modbus.check_reply(modbus.with_crc(bytes([0, 0x83, 2])), address=0, function=3)

    def test_line_capacity(self):
        # 21 bytes + two 3.5-char gaps = 28 chars of 10 bits.
        self.assertAlmostEqual(modbus.poll_seconds(9600), 280 / 9600)
        self.assertLess(modbus.max_poll_rate_hz(9600), 100)
        self.assertGreater(modbus.max_poll_rate_hz(115200), 100)
        self.assertGreater(modbus.max_poll_rate_hz(460800), modbus.max_poll_rate_hz(115200))
        self.assertIn('at most 34.3 Hz', rate_refusal(100, 9600))
        self.assertEqual(rate_refusal(100, 115200), '')


class Link(unittest.TestCase):
    def test_signed_values_and_scaling(self):
        sensor = Sensor(baud=115200, registers=(0xFFFF, 0x8000, 0x7FFF, 0xFFF6))
        port = Port(sensor, 115200)
        bx, by, bz, heading = modbus.read_field(port, 0, timeout=0.1)
        self.assertAlmostEqual(bx, -0.013)
        self.assertAlmostEqual(by, -32768 * 0.013)
        self.assertAlmostEqual(bz, 32767 * 0.013)
        self.assertAlmostEqual(heading, -1.0)

    def test_address_mismatch_timeout_and_crc_error(self):
        sensor = Sensor(baud=115200)
        port = Port(sensor, 115200)
        sensor.reply_from = 0x50
        with self.assertRaisesRegex(modbus.ModbusError, 'address 0x50'):
            modbus.read_field(port, 0, timeout=0.05)
        sensor.reply_from = None
        with self.assertRaises(modbus.ModbusTimeout):
            modbus.read_field(port, 0x50, timeout=0.05)  # nobody at 0x50
        sensor.corrupt_crc = True
        with self.assertRaisesRegex(modbus.ModbusError, 'CRC'):
            modbus.read_field(port, 0, timeout=0.05)


class Baud(unittest.TestCase):
    def test_already_at_target_writes_nothing(self):
        sensor = Sensor(baud=115200)
        outcome = modbus.establish_baud(Port(sensor, 9600), 0, 115200)
        self.assertEqual((outcome.baud, outcome.switched_from, sensor.writes), (115200, None, []))

    def test_found_elsewhere_is_switched_and_verified(self):
        sensor = Sensor(baud=9600)
        port = Port(sensor, 9600)
        outcome = modbus.establish_baud(port, 0, 460800)
        self.assertEqual((outcome.baud, outcome.switched_from, sensor.baud), (460800, 9600, 460800))
        self.assertEqual(sensor.writes, [(0xD2, 2)], 'only the baud register is written, never 0x00')
        self.assertEqual(port.baudrate, 460800)

    def test_switch_that_does_not_take_falls_back_and_says_so(self):
        sensor = Sensor(baud=9600)
        sensor.ignore_baud_write = True
        port = Port(sensor, 9600)
        outcome = modbus.establish_baud(port, 0, 115200, verify_seconds=0.2)
        self.assertEqual((outcome.baud, port.baudrate), (9600, 9600))
        self.assertIn('did not answer at 115200', outcome.note)

    def test_refused_switch_falls_back(self):
        sensor = Sensor(baud=9600)
        sensor.refuse_baud_write = True
        outcome = modbus.establish_baud(Port(sensor, 9600), 0, 115200)
        self.assertEqual(outcome.baud, 9600)
        self.assertIn('refused', outcome.note)

    def test_no_sensor_names_what_was_tried(self):
        with self.assertRaisesRegex(RuntimeError, r'no HWT3100 answered at address 0x00 at 115200, 9600, 460800 baud'):
            modbus.establish_baud(Port(Sensor(address=0x50, baud=9600), 9600), 0, 115200)


class Source(unittest.TestCase):
    def test_polls_at_100_hz_and_publishes_two_quantities(self):
        sensor = Sensor(baud=9600)
        device = source(sensor)
        try:
            self.assertEqual(device.baud, 115200)
            self.assertEqual([(o.name, o.unit, o.columns) for o in device.outputs],
                             [('magnetic_field', 'uT', (0, 1, 2)), ('heading', 'deg', (3,))])
            point = device.working_point()
            self.assertEqual((point.sample_interval_seconds, point.time_basis), (0.01, 'host_receive'))
            self.assertIn('switched to 115200', point.settings['baud_note'])
            device.arm(50, buffer_record_count=100)
            records = device.read_records(50, timeout=3.0, exact=True)
            device.finish_record_capture()
            self.assertEqual([r.source_ordinal for r in records], list(range(50)))
            row = records[0].samples[0]
            self.assertAlmostEqual(float(row[0]), 8.879, places=3)
            self.assertAlmostEqual(float(row[3]), 80.6, places=4)
            span = records[-1].time_seconds - records[0].time_seconds
            self.assertAlmostEqual(span / 49, 0.01, delta=0.002)
        finally:
            device.close()

    def test_rate_refused_at_9600_with_reason_and_accepted_within_capacity(self):
        sensor = Sensor(baud=9600)
        sensor.ignore_baud_write = True  # stuck at 9600
        device = Hwt3100WaveformSource(Hwt3100Config('COM9', baud=115200, timeout_seconds=1.0),
                                       serial_port=Port(sensor, 9600),
                                       establish=lambda p, a, t: modbus.establish_baud(p, a, t, verify_seconds=0.2))
        try:
            self.assertEqual(device.baud, 9600)
            self.assertEqual(device.tunable_values()['sample_rate_hz'], 34.0)
            self.assertIn('started at 34 Hz', device.working_point().settings['rate_note'])
            with self.assertRaisesRegex(TuneRefused, r'100 Hz cannot be polled at 9600 baud.*at most 34.3 Hz'):
                device.tune('sample_rate_hz', 100)
            self.assertEqual(device.tune('sample_rate_hz', 20), 20.0)
            self.assertEqual(device.working_point().sample_interval_seconds, 0.05)
            self.assertEqual(device.settings_provenance()['settings_epoch'], 1)
            with self.assertRaisesRegex(TuneRefused, 'record timeout'):
                device.tune('sample_rate_hz', 0.5)
        finally:
            device.close()

    def capture_around(self, sensor, upset, *, settle=20, timeout=1.0):
        """Records of a capture in which ``upset(sensor)`` strikes after ``settle`` valid ones."""
        device = source(sensor, timeout=timeout)
        try:
            device.arm(None, buffer_record_count=1000)
            records = device.read_records(settle, timeout=1.0, exact=True)
            upset(sensor)
            records += device.read_records(30, timeout=2.0, exact=True)
            device.finish_record_capture()
            return records, device.capture_gaps, device.working_point().settings
        finally:
            device.close()

    def invalid(self, records):
        return [i for i, r in enumerate(records) if np.isnan(r.samples).all()]

    def test_lost_replies_are_invalid_samples_and_the_capture_goes_on(self):
        records, gaps, settings = self.capture_around(Sensor(baud=115200), lambda s: setattr(s, 'drop', 2))
        missing = self.invalid(records)
        self.assertEqual(missing, [20, 21])
        self.assertEqual([r.source_ordinal for r in records], list(range(50)))
        times = [r.time_seconds for r in records]
        self.assertTrue(all(b > a for a, b in zip(times, times[1:])), 'real receive times, still advancing')
        # each missing reply cost its 0.1 s timeout: the hole shows in time
        self.assertGreater(times[21] - times[19], 0.19)
        self.assertTrue(np.isfinite(records[22].samples).all(), 'valid again after the hole')
        self.assertEqual(gaps['invalid'], {'no_reply': 2})
        self.assertGreater(gaps['skipped_slots'], 0, 'slots spent waiting are counted, not invented')
        self.assertEqual(settings['invalid_polls'].get('no_reply'), 2)

    def test_crc_and_address_errors_are_invalid_samples(self):
        def upset(sensor):
            sensor.corrupt, sensor.misaddress = 1, 1
        records, gaps, _ = self.capture_around(Sensor(baud=115200), upset)
        self.assertEqual(len(self.invalid(records)), 2)
        self.assertEqual(gaps['invalid'], {'crc': 1, 'address': 1})

    def test_silent_beyond_the_record_timeout_fails_with_the_reason(self):
        sensor = Sensor(baud=115200)
        device = source(sensor, timeout=0.5)
        try:
            device.arm(None, buffer_record_count=1000)
            device.read_records(5, timeout=1.0, exact=True)
            sensor.silent = True
            with self.assertRaisesRegex(RuntimeError, r'gave no valid reply for 0\.\d+ s, longer than the 0\.5 s record timeout; last: no reply'):
                device.read_records(1000, timeout=3.0, exact=True)
            with self.assertRaises(RuntimeError):
                device.finish_record_capture()
        finally:
            sensor.silent = False
            device.close()

    def test_measurement_publishes_nan_samples_as_invalid(self):
        from zlc_atom.devices.waveform.contract import WaveformRecord
        from zlc_atom.devices.waveform.witmotion_hwt3100 import HWT3100_OUTPUTS
        from zlc_atom.nodes.waveform_measurement.measurement import shot_snapshot
        def published(values):
            record = WaveformRecord(np.asarray([values], dtype=np.float32), 0, 1.0)
            return [shot_snapshot(record, output=o, producer='hwt', generation='g', revision=1,
                                  sample_interval_seconds=0.01, continuous=True) for o in HWT3100_OUTPUTS]
        valid_field, valid_heading = published([1.0, 2.0, 3.0, 4.0])
        lost_field, lost_heading = published([np.nan] * 4)
        from zlc_data import expand_dataset_validity
        mask = lambda snapshot: expand_dataset_validity(snapshot.block.validity, snapshot.block.schema)
        self.assertTrue(mask(valid_field).all() and mask(valid_heading).all())
        self.assertFalse(mask(lost_field).any() or mask(lost_heading).any())
        self.assertEqual(valid_field.block.schema, lost_field.block.schema, 'a missing sample does not change the schema')

    def test_no_sensor_fails_open_cleanly(self):
        sensor = Sensor(baud=9600, address=0x50)
        port = Port(sensor, 9600)
        with self.assertRaisesRegex(RuntimeError, 'no HWT3100 answered at address 0x00'):
            Hwt3100WaveformSource(Hwt3100Config('COM9'), serial_port=port)
        self.assertTrue(port.closed)


class Discovery(unittest.TestCase):
    def test_finds_the_sensor_at_either_address_and_any_baud(self):
        sensors = {'COM3': Sensor(address=0x50, baud=460800), 'COM4': None, 'COM5': Sensor(address=0, baud=9600)}

        def opener(device, baud):
            if sensors[device] is None:
                raise OSError('in use')
            return Port(sensors[device], baud)

        found = discover_hwt3100(ports=('COM3', 'COM4', 'COM5'), opener=opener)
        self.assertEqual(found, (('COM3', 0x50, 460800), ('COM5', 0, 9600)))
        for sensor in (sensors['COM3'], sensors['COM5']):
            self.assertEqual(sensor.writes, [], 'a scan only reads')

    def test_type_descriptor(self):
        (declared,) = DEVICE_TYPES
        self.assertEqual((declared.type_id, declared.family, declared.capabilities, declared.resources),
                         ('waveform.witmotion_hwt3100', 'waveform', ('waveform.source',), ('serial:{port}',)))
        self.assertEqual(WITMOTION_HWT3100_SCHEMA.project_values({}),
                         {'port': '', 'address': 0, 'baud': '115200', 'timeout_seconds': 2.0})


if __name__ == '__main__':
    unittest.main()
