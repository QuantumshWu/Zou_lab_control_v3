"""Newport SMC100 over a simulated controller on a simulated serial line: no port is opened."""
import pathlib
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'science' / 'src'))
from zlc_atom.authoring import TuneRefused
from zlc_atom.devices.motion.newport_smc100 import Smc100Axis, Smc100Config, Smc100CommandError, Smc100Link
from zlc_atom.devices.motion.newport_smc100 import axis as axis_module
from zlc_atom.devices.motion.newport_smc100 import protocol as protocol_module
from zlc_atom.devices.motion.newport_smc100.axis import probe_port
from zlc_atom.devices.motion.newport_smc100.device_types import DEVICE_TYPES, NEWPORT_SMC100_SCHEMA
from zlc_atom.devices.motion.newport_smc100.protocol import parse_status
from zlc_atom.devices.simulation.motion import SimulatedSerialPort, SimulatedSmc100, profile_seconds
from zlc_atom.devices.simulation.motion.device_types import DEVICE_TYPES as VIRTUAL_TYPES
from zlc_atom.nodes.scan.plan import scan_ports_for_devices
from zlc_plugin import host


def controller(address=1, **overrides):
    """A brisk stage, so moves take tenths of a second."""
    settings = dict(travel=(-5.0, 20.0), velocity=40.0, acceleration=400.0, home_velocity=40.0,
                    position=7.0, init_seconds=0.1)
    return SimulatedSmc100(address, **{**settings, **overrides})


def stage(sim=None, address=1, unit='mm'):
    sim = sim or controller()
    return Smc100Axis(Smc100Config('COM9', address, unit), serial_port=SimulatedSerialPort(sim)), sim


def fields(axis):
    return {field.metadata.name: field for field in axis.tunable_fields()}


class StartupPort(SimulatedSerialPort):
    """Real protocol exchanges with faults confined to selected version replies."""

    def __init__(self, sim, faults=(), **kwargs):
        super().__init__(sim, **kwargs)
        self.faults = list(faults)
        self.sent = []
        self.read_timeouts = []

    def _line(self, text):
        self.sent.append(text)
        before = len(self._replies)
        super()._line(text)
        if not text.endswith('VE') or not self.faults:
            return
        fault = self.faults.pop(0)
        if fault in ('refuse', 'drop', 'empty', 'reply-and-error'):
            if fault in ('refuse', 'drop'):
                del self._replies[before:]
            elif fault == 'empty':
                self._replies[before:] = [(self._clock() + 0.01, b'1VE\r\n')]
            if fault in ('refuse', 'reply-and-error'):
                self.controllers[1].error = 'A'
        elif fault == 'delayed':
            due, raw = self._replies[before]
            self._replies[before] = (due + 0.25, raw)
        elif fault == 'fragmented':
            due, raw = self._replies[before]
            self._replies[before:] = [(due, raw[:8]), (due + 0.01, raw[8:])]
        elif fault == 'unexpected':
            due, _raw = self._replies[before]
            self._replies.insert(before, (due, b'2VE another controller\r\n'))
        else:
            raise AssertionError(f'unknown startup fault {fault}')

    def read_until(self, *args, **kwargs):
        self.read_timeouts.append(self.timeout)
        return super().read_until(*args, **kwargs)


def quick_startup():
    return mock.patch.multiple(protocol_module, STARTUP_REPLY_TIMEOUT_SECONDS=0.08, STARTUP_RETRY_SECONDS=0.0)


class Protocol(unittest.TestCase):
    def test_set_commands_answer_nothing_and_refusals_come_from_te(self):
        sim = controller(); port = SimulatedSerialPort(sim); link = Smc100Link(port, 1)
        self.assertEqual(link.query('VA'), '40')
        with self.assertRaises(Smc100CommandError) as refused:
            link.execute('PA', '3')                   # NOT REFERENCED at power-up
        self.assertEqual(refused.exception.code, 'H')
        self.assertIn('NOT REFERENCED', str(refused.exception))
        self.assertEqual(sim.commands[-2:], ['PA3', 'TE'])
        self.assertEqual(link.query('TE', ''), '@')   # TE cleared by the read
        link.execute('OR')                            # accepted: no reply, TE '@'
        self.assertEqual(link.status().kind, 'HOMING')

    def test_status_words_and_faults(self):
        status = parse_status('00203D')
        self.assertEqual((status.kind, status.words, status.faults), ('DISABLE', 'DISABLE from MOVING', ('following error',)))
        self.assertEqual(parse_status('00000A').kind, 'NOT REFERENCED')
        self.assertEqual(parse_status('000099').kind, 'UNKNOWN')

    def test_another_address_on_the_chain_is_not_answered(self):
        port = SimulatedSerialPort(controller())
        with quick_startup(), self.assertRaises(RuntimeError) as missing:
            Smc100Axis(Smc100Config('COM9', 2), serial_port=port)
        self.assertIn('COM9 at address 2', str(missing.exception))
        self.assertFalse(port.is_open, 'a failed open releases the port')
        self.assertEqual(port.timeout, 0.2)


class Startup(unittest.TestCase):
    def test_old_te_is_reported_separately_and_cannot_poison_the_first_move(self):
        sim = controller(); sim.error = 'A'
        port = StartupPort(sim)
        with self.assertLogs(protocol_module.__name__, level='INFO') as logged:
            axis = Smc100Axis(Smc100Config('COM9'), serial_port=port)
        self.addCleanup(axis.close)
        self.assertTrue(any('TE A' in line for line in logged.output), logged.output)
        self.assertEqual(sim.commands[:3], ['TE', 'VE', 'TE'])
        self.assertEqual(axis.version, sim.version)
        self.assertEqual(port.timeout, 0.2)
        self.assertIn('homed: READY', axis.home())
        self.assertAlmostEqual(axis.tune('position', 3.0), 3.0)

    def test_transient_version_failures_recover_on_the_same_open_port(self):
        for fault in ('refuse', 'drop', 'empty', 'reply-and-error'):
            with self.subTest(fault=fault):
                sim = controller(); port = StartupPort(sim, (fault,))
                with quick_startup():
                    axis = Smc100Axis(Smc100Config('COM9'), serial_port=port)
                try:
                    self.assertEqual(axis.version, sim.version)
                    self.assertEqual(sim.commands.count('VE'), 2)
                    self.assertTrue(port.is_open)
                    self.assertEqual(port.timeout, 0.2)
                    self.assertFalse(any(c[:2] in ('PA', 'PR', 'OR', 'MM', 'RS', 'ST') for c in sim.commands))
                finally:
                    axis.close()

    def test_startup_waits_for_a_version_reply_slower_than_normal_queries(self):
        sim = controller(); port = StartupPort(sim, ('delayed',))
        started = time.monotonic()
        axis = Smc100Axis(Smc100Config('COM9'), serial_port=port)
        self.addCleanup(axis.close)
        self.assertEqual(axis.version, sim.version)
        self.assertEqual(sim.commands.count('VE'), 1)
        self.assertGreaterEqual(time.monotonic() - started, 0.25)
        self.assertEqual(port.timeout, 0.2)

    def test_startup_recovers_from_fragmented_or_unrelated_frames(self):
        for fault in ('fragmented', 'unexpected'):
            with self.subTest(fault=fault):
                sim = controller(); port = StartupPort(sim, (fault,))
                with quick_startup():
                    version = Smc100Link(port, 1).identify()
                self.assertEqual(version, sim.version)
                self.assertLessEqual(sim.commands.count('VE'), protocol_module.STARTUP_ATTEMPTS)
                self.assertEqual(port.timeout, 0.2)
                self.assertTrue(set(sim.commands) <= {'TE', 'VE'})
                port.close()

    def test_unfinished_remote_command_recovers_without_driving_the_stage(self):
        sim = controller(); port = StartupPort(sim)
        # Bytes already at the controller, beyond a host-side input-buffer reset.
        port._pending = b'1XX'
        with quick_startup():
            axis = Smc100Axis(Smc100Config('COM9'), serial_port=port)
        self.addCleanup(axis.close)
        self.assertEqual(sim.commands[0], 'XX1TE')
        self.assertEqual(axis.version, sim.version)
        self.assertEqual(sim.commands.count('VE'), 2)
        self.assertEqual(axis.tunable_values()['state'], 'NOT REFERENCED from reset')
        self.assertFalse(any(c[:2] in ('PA', 'PR', 'OR', 'MM', 'RS', 'ST') for c in sim.commands))
        self.assertEqual(port.timeout, 0.2)

    def test_version_without_te_reports_synchronization_failure_and_closes(self):
        class NoTePort(StartupPort):
            def _line(self, text):
                before = len(self._replies)
                super()._line(text)
                if text.endswith('TE'):
                    del self._replies[before:]

        sim = controller(); port = NoTePort(sim)
        started = time.monotonic()
        with quick_startup(), self.assertRaises(RuntimeError) as failed:
            Smc100Axis(Smc100Config('COM9'), serial_port=port)
        diagnostic = str(failed.exception)
        self.assertIn('VE answered', diagnostic)
        self.assertIn(sim.version, diagnostic)
        self.assertIn('error-register synchronization failed', diagnostic)
        self.assertNotIn('no SMC100', diagnostic)
        self.assertEqual(sim.commands.count('VE'), protocol_module.STARTUP_ATTEMPTS)
        self.assertTrue(set(sim.commands) <= {'TE', 'VE'})
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertEqual(port.timeout, 0.2)
        self.assertFalse(port.is_open)

    def test_persistent_version_refusal_is_bounded_and_closes_the_port(self):
        sim = controller()
        port = StartupPort(sim, ('refuse',) * protocol_module.STARTUP_ATTEMPTS)
        started = time.monotonic()
        with quick_startup(), self.assertRaises(RuntimeError) as failed:
            Smc100Axis(Smc100Config('COM9'), serial_port=port)
        self.assertIn('COM9 at address 1', str(failed.exception))
        self.assertIn("TE after VE returned 'A'", str(failed.exception))
        self.assertEqual(sim.commands.count('VE'), protocol_module.STARTUP_ATTEMPTS)
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertEqual(port.timeout, 0.2)
        self.assertFalse(port.is_open)
        self.assertTrue(set(sim.commands) <= {'TE', 'VE'})

    def test_silent_startup_is_bounded_and_closes_the_port(self):
        sim = controller(); port = StartupPort(sim); port.silent = True
        started = time.monotonic()
        with quick_startup(), self.assertRaises(RuntimeError) as failed:
            Smc100Axis(Smc100Config('COM9'), serial_port=port)
        self.assertIn('COM9 at address 1', str(failed.exception))
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertLessEqual(len(sim.commands), 3 * protocol_module.STARTUP_ATTEMPTS)
        self.assertEqual(port.timeout, 0.2)
        self.assertFalse(port.is_open)
        self.assertTrue(set(sim.commands) <= {'TE', 'VE'})

    def test_normal_operations_keep_refusals_and_frame_errors_strict(self):
        sim = controller(); port = StartupPort(sim, ('refuse',))
        link = Smc100Link(port, 1)
        with self.assertRaises(Smc100CommandError):
            link.query('VE', '')
        self.assertEqual(sim.commands.count('VE'), 1)
        port.faults = ['unexpected']
        with self.assertRaises(protocol_module.Smc100Error):
            link.query('VE', '')
        self.assertEqual(sim.commands.count('VE'), 2)
        self.assertEqual(port.timeout, 0.2)
        port.close()


class Axis(unittest.TestCase):
    def test_open_reads_and_moves_nothing(self):
        axis, sim = stage()
        rows = fields(axis)
        self.assertEqual(rows['position'].current, 7.0)
        self.assertEqual((rows['position'].metadata.minimum, rows['position'].metadata.maximum), (-5.0, 20.0))
        self.assertEqual(rows['position'].device_limits, (-5.0, 20.0))
        self.assertEqual(rows['position'].metadata.unit, 'mm')
        self.assertEqual(rows['velocity'].current, 40.0)
        self.assertTrue(rows['state'].metadata.derived)
        self.assertFalse(rows['state'].live_write)
        self.assertEqual(rows['state'].current, 'NOT REFERENCED from reset')
        self.assertFalse([c for c in sim.commands if c[:2] in ('PA', 'PR', 'OR', 'MM', 'RS', 'ST') or c[:2] in ('VA', 'SL', 'SR') and c[2:] != '?'])
        self.assertEqual(axis.identity, 'newport-smc100:COM9/1')

    def test_unreferenced_move_is_refused_and_never_homes(self):
        axis, sim = stage()
        with self.assertRaises(TuneRefused) as refused:
            axis.tune('position', 3.0)
        self.assertIn('run Home first', str(refused.exception))
        self.assertNotIn('OR', [c[:2] for c in sim.commands])
        self.assertNotIn('PA', [c[:2] for c in sim.commands])

    def test_home_then_absolute_move_reads_back_where_it_stopped(self):
        axis, sim = stage()
        self.assertIn('homed: READY from HOMING at 0 mm', axis.home())
        started = time.monotonic()
        landed = axis.tune('position', 12.5)
        took = time.monotonic() - started
        self.assertAlmostEqual(landed, 12.5, places=6)
        self.assertGreaterEqual(took, profile_seconds(12.5, 40.0, 400.0))
        self.assertEqual(axis.tunable_values()['state'], 'READY from MOVING')
        self.assertEqual(axis.settings_provenance()['settings_epoch'], 2)
        with self.assertRaises(RuntimeError) as again:
            axis.home()
        self.assertIn('Reset it first', str(again.exception))

    def test_travel_is_checked_before_anything_is_sent(self):
        axis, sim = stage(); axis.home()
        sent = len(sim.commands)
        for value in (20.5, -6, float('nan'), True, 'x'):
            with self.assertRaises(TuneRefused):
                axis.tune('position', value)
        self.assertNotIn('PA', [c[:2] for c in sim.commands[sent:]])
        # A value a rounding past an edge it was authored on is that edge.
        self.assertAlmostEqual(axis.tune('position', 20.0 + 1e-14), 20.0)

    def test_stop_from_another_thread_ends_the_move_where_it_is(self):
        axis, sim = stage(controller(velocity=10.0, acceleration=100.0)); axis.home()
        outcome = {}

        def move():
            try:
                outcome['landed'] = axis.tune('position', 18.0)
            except Exception as error:  # noqa: BLE001
                outcome['error'] = error
        mover = threading.Thread(target=move); mover.start()
        time.sleep(0.4)
        self.assertIn('stop sent', axis.stop())
        mover.join(5)
        self.assertFalse(mover.is_alive())
        self.assertIn('Stop ended the move to 18 mm', str(outcome['error']))
        position = axis.tunable_values()['position']
        self.assertTrue(1.0 < position < 18.0, position)
        self.assertEqual(axis.stop(), 'stop sent: READY from MOVING')  # READY accepts ST

    def test_stop_with_nothing_to_stop_is_not_a_failure(self):
        axis, _ = stage()
        self.assertIn('nothing was moving: NOT REFERENCED from reset', axis.stop())
        axis.make_safe('motion')

    def test_following_error_is_reported_and_enable_recovers(self):
        axis, sim = stage(); axis.home()
        sim.fail_next_move = True
        with self.assertRaises(RuntimeError) as failed:
            axis.tune('position', 15.0)
        self.assertIn('ended DISABLE from MOVING with following error', str(failed.exception))
        self.assertEqual(fields(axis)['state'].current, 'DISABLE from MOVING; fault: following error')
        with self.assertRaises(TuneRefused) as refused:
            axis.tune('position', 1.0)
        self.assertIn('run Enable first', str(refused.exception))
        self.assertEqual(axis.enable(), 'READY from DISABLE')
        self.assertAlmostEqual(axis.tune('position', 1.0), 1.0)

    def test_velocity_is_the_controllers_and_its_maximum_is_its_refusal(self):
        axis, _ = stage()
        with self.assertRaises(TuneRefused) as unreferenced:
            axis.tune('velocity', 5.0)
        self.assertIn('TE H', str(unreferenced.exception))
        axis.home()
        self.assertEqual(axis.tune('velocity', 5.0), 5.0)
        with self.assertRaises(TuneRefused) as over:
            axis.tune('velocity', 41.0)
        self.assertIn('TE C', str(over.exception))
        with self.assertRaises(TuneRefused):
            axis.tune('state', 'READY')

    def test_a_move_longer_than_one_call_is_refused_with_its_arithmetic(self):
        axis, sim = stage(controller(velocity=0.1, acceleration=1.0)); axis.home()
        with self.assertRaises(TuneRefused) as refused:
            axis.tune('position', 19.0)
        self.assertIn('raise the velocity or move in steps', str(refused.exception))
        self.assertNotIn('PA', [c[:2] for c in sim.commands])

    def test_an_overdue_move_is_stopped_and_reported(self):
        axis, sim = stage(); axis.home()
        factor = axis_module.MOVE_TIME_FACTOR, axis_module.SETTLE_SECONDS
        axis_module.MOVE_TIME_FACTOR, axis_module.SETTLE_SECONDS = 0.2, 0.0
        try:
            with self.assertRaises(RuntimeError) as overdue:
                axis.tune('position', 19.0)
        finally:
            axis_module.MOVE_TIME_FACTOR, axis_module.SETTLE_SECONDS = factor
        self.assertIn('was still MOVING after', str(overdue.exception))
        self.assertIn('ST', sim.commands)

    def test_reset_reboots_to_not_referenced(self):
        axis, sim = stage(); axis.home()
        self.assertEqual(axis.reset(), 'NOT REFERENCED from reset')
        self.assertIn('NOT REFERENCED', axis.status())

    def test_rotation_stage_counts_in_degrees_and_converts(self):
        axis, _ = stage(unit='deg'); axis.home()
        self.assertEqual(fields(axis)['position'].metadata.unit, 'deg')
        from zlc_plugin.tunable import tune_in_unit
        self.assertAlmostEqual(tune_in_unit(axis, 'position', 0.1, 'rad'), 0.1, places=6)

    def test_position_is_the_one_scan_axis(self):
        axis, _ = stage()
        ports = scan_ports_for_devices({'stage': axis})
        self.assertEqual([(p.port, p.unit, p.lo, p.hi, p.current) for p in ports],
                         [('device:stage:position', 'mm', -5.0, 20.0, 7.0)])


class Finding(unittest.TestCase):
    def test_probe_reads_version_and_stage_only(self):
        sims = {'COM4': controller(), 'COM5': None}

        def opener(device):
            sim = sims[device]
            return SimulatedSerialPort(*(() if sim is None else (sim,)))
        with quick_startup(), mock.patch.object(axis_module, 'REPLY_TIMEOUT_SECONDS', 0.08):
            self.assertEqual(probe_port('COM5', (1,), opener=opener), [])
            found = probe_port('COM4', (1, 2), opener=opener)
        self.assertEqual([(a, s) for a, _v, s in found], [(1, 'VIRTUAL25CC')])
        self.assertEqual(sims['COM4'].commands, ['TE', 'VE', 'TE', 'ID?'])

    def test_probe_shares_startup_recovery_and_closes_its_port(self):
        sim = controller(); sim.error = 'A'
        port = StartupPort(sim, ('refuse',))
        with quick_startup(), mock.patch.object(axis_module, 'REPLY_TIMEOUT_SECONDS', 0.08):
            found = probe_port('COM9', (1,), opener=lambda _device: port)
        self.assertEqual(found, [(1, sim.version, sim.stage)])
        self.assertEqual(sim.commands.count('VE'), 2)
        self.assertTrue(set(sim.commands) <= {'TE', 'VE', 'ID?'})
        self.assertEqual(port.timeout, 0.2)
        self.assertFalse(port.is_open)

    def test_declaration(self):
        declared = DEVICE_TYPES[0]
        self.assertEqual((declared.type_id, declared.family, declared.capabilities), ('motion.newport_smc100', 'motion', ('motion.axis',)))
        self.assertEqual(declared.resources, ('serial:{port}',))
        self.assertEqual(set(declared.concurrent), {'stop', 'make_safe'})
        self.assertEqual({c.name: c.drives for c in declared.commands},
                         {'home': True, 'stop': False, 'enable': True, 'disable': True, 'reset': True, 'status': False})
        self.assertEqual(NEWPORT_SMC100_SCHEMA.project_values({'port': 'COM3'}), {'port': 'COM3', 'address': 1, 'unit': 'mm'})
        with self.assertRaises(ValueError):
            NEWPORT_SMC100_SCHEMA.project_values({'port': 'COM3', 'address': 32})
        from zlc_plugin.capabilities import contract
        from zlc_atom.devices.motion.contract import MotionAxis
        self.assertIs(contract('motion.axis'), MotionAxis)
        axis, _ = stage()
        self.assertIsInstance(axis, MotionAxis)

    def test_virtual_stage_opens_unreferenced(self):
        axis = VIRTUAL_TYPES[0].factory('stage', {'address': 1})
        try:
            self.assertEqual(axis.tunable_values()['state'], 'NOT REFERENCED from reset')
            self.assertEqual(fields(axis)['position'].device_limits, (0.0, 25.0))
        finally:
            axis.close()


class Owner:
    """The plugin process as the lane sees it (python_lifecycle.py's stand-in)."""
    _request = host.Process._request

    def __init__(self):
        self.instances, self.answers, self.events = {}, {}, {}

    def decode(self, value):
        return value

    def reply(self, identity, result=None, **kwargs):
        self.answers[identity] = (result, kwargs)
        self.events.setdefault(identity, threading.Event()).set()

    def send(self, identity, op, params):
        self.events.setdefault(identity, threading.Event())
        self._request({'id': identity, 'op': op, 'p': {'instance': 'stage', **params}})

    def wait(self, identity, seconds=5):
        return self.events.setdefault(identity, threading.Event()).wait(seconds)


class Host(unittest.TestCase):
    def test_stop_and_make_safe_answer_while_a_move_holds_the_lane(self):
        axis, _ = stage(controller(velocity=5.0, acceleration=50.0)); axis.home()
        owner = Owner(); instance = host.Instance(owner, 'stage'); owner.instances['stage'] = instance
        instance.declared, instance.adapter, instance.closer = DEVICE_TYPES[0], axis, axis.close
        try:
            owner.send(1, 'tune', {'field': 'position', 'value': 19.0})
            time.sleep(0.3)
            owner.send(2, 'fields', {})                       # waits on the lane behind the move
            owner.send(3, 'command', {'name': 'stop', 'params': {}})
            self.assertTrue(owner.wait(3, 1.0), 'Stop waited for the move')
            self.assertIn('stop sent', owner.answers[3][0])
            self.assertTrue(owner.wait(1))
            self.assertIn('Stop ended the move', str(owner.answers[1][1]['error']))
            self.assertTrue(owner.wait(2))
            owner.send(4, 'tune', {'field': 'position', 'value': 19.0})
            time.sleep(0.3)
            owner.send(5, 'make_safe', {'output': 'motion'})
            self.assertTrue(owner.wait(5, 1.0), 'Stop all waited for the move')
            self.assertTrue(owner.wait(4))
            self.assertIn('Stop ended the move', str(owner.answers[4][1].get('error')), owner.answers)
            self.assertFalse(instance.concurrent('command', {'name': 'home'}))
            self.assertFalse(instance.concurrent('spi', {'method': 'tune'}))
            self.assertTrue(instance.concurrent('spi', {'method': 'stop'}))
        finally:
            instance.lane.put(None); instance._thread.join(2); instance.shutdown()


if __name__ == '__main__':
    unittest.main()
