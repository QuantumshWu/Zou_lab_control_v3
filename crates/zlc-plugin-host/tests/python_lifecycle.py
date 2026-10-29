"""Lifecycle races in the real Python SDK, with exclusive stand-in device handles."""
import pathlib
import sys
import threading
import types
import unittest
from unittest.mock import patch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'science' / 'src'))
from zlc_plugin import host

class Owner:
    _request = host.Process._request
    _event = host.Process._event
    def __init__(self):
        self.instances = {}
        self.answers = {}
        self.events = {}
        self.hook = lambda identity: None
    def decode(self, value): return value
    def reply(self, identity, result=None, **kwargs):
        self.answers[identity] = (result, kwargs)
        self.hook(identity)
        self.events.setdefault(identity, threading.Event()).set()
    def send(self, identity, op, name='n100_com3'):
        self.events.setdefault(identity, threading.Event())
        self._request({'id':identity,'op':op,'p':{'instance':name}})
    def wait(self, identity):
        return self.events.setdefault(identity, threading.Event()).wait(2)
    def cleanup(self):
        for instance in list(self.instances.values()):
            instance.lane.put(None)
            instance._thread.join(2)
            instance.shutdown()

class Lifecycle(unittest.TestCase):
    def test_device_record_precedes_ready_and_publication_and_is_cached(self):
        import numpy as np
        from zlc_data import AxisId, AxisSpec, DomainSpec, SPATIAL_X, ValueSchema, ValidityContract, ValidityMode, owned_snapshot_from_arrays
        from zlc_plugin import DatasetOutputDeclaration, LiveDatasetOutput
        from zlc_plugin.context import RunContext

        calls = []
        def request(op, params):
            calls.append((op, params))
            if op == 'run.snapshot': return {'devices': {'rf': {}}}
            return {}
        process = types.SimpleNamespace(request=request, event=lambda op, params: calls.append((op, params)))
        run = types.SimpleNamespace(instance=types.SimpleNamespace(name='camera', process=process), run_id='run-1', params={})
        context = RunContext(run)
        self.assertIsNone(context.record_devices())
        self.assertEqual(calls, [], 'direct SDK runs do not request desktop device recording')
        run.params['record_devices'] = True
        self.assertEqual(context.record_devices(), {'devices': {'rf': {}}})
        context.set_run_record({'configured': True})
        context.report_ready()
        schema = ValueSchema(ValidityContract(ValidityMode.VALUE), np.dtype('<f8'), 'count')
        snapshot = owned_snapshot_from_arrays(schema, np.ones((1, 1, 1)), 0,
            cell_domain=DomainSpec((1,), (AxisSpec(AxisId('x'), 'x', SPATIAL_X, 1),)))
        output = LiveDatasetOutput(DatasetOutputDeclaration('reading', 'test.reading'), snapshot)
        context.commit_live({'reading': output})
        self.assertEqual([op for op, _ in calls], ['run.snapshot', 'run.ready', 'run.begin', 'run.commit'])
        self.assertFalse(calls[0][1]['wait'])
        self.assertEqual(context.record_devices(), {'devices': {'rf': {}}})
        context.commit_live({'reading': output})
        self.assertEqual([op for op, _ in calls].count('run.snapshot'), 1)
        context = RunContext(run)
        process.request = lambda op, params: calls.append((op, params))
        self.assertIsNone(context.record_devices())
        self.assertIsNone(context.record_devices(wait=True))
        self.assertTrue(calls[-1][1]['wait'], 'null answers must not be cached as device records')

    def test_device_recording_follows_accepted_armed_settings_and_keeps_cleanup(self):
        from zlc_atom.nodes.camera.measurement import CameraMeasurementNode
        from zlc_atom.nodes.waveform_measurement.measurement import WaveformMeasurementNode

        for kind in ('camera', 'camera monitor', 'waveform'):
            with self.subTest(kind=kind):
                events = []
                if kind.startswith('camera'):
                    node = object.__new__(CameraMeasurementNode)
                    node._request = types.SimpleNamespace(repeat=1 if kind == 'camera' else 0, frames_per_cycle=1)
                    node.camera = types.SimpleNamespace(finish_record_capture=lambda: events.append('close'))
                    if kind == 'camera':
                        node._configure_capture = lambda: events.append('configure')
                        node._arm_configured = lambda **kwargs: events.extend(['arm', 'accepted'])
                        prepare = node.prepare
                    else:
                        node._configure_for_run = lambda: events.append('configure')
                        node._buffer_frame_count = lambda point: 2
                        node.camera.timeout = 1
                        node.camera.arm = lambda *args, **kwargs: events.append('arm')
                        node.camera.working_point = lambda: None
                        node._freeze_working_point = lambda point: events.append('accepted')
                        node._run_record = {}
                        prepare = lambda **kwargs: node.monitor(commit_live=lambda outputs: None, **kwargs)
                else:
                    node = object.__new__(WaveformMeasurementNode)
                    node.sampler = types.SimpleNamespace(finish_record_capture=lambda: events.append('close'))
                    node._configure = lambda: events.append('configure')
                    node._arm = lambda: events.extend(['arm', 'accepted'])
                    prepare = node._armed
                prepare(after_arm=lambda: events.append('snapshot'))
                self.assertEqual(events, ['configure', 'arm', 'accepted', 'snapshot'])
                events.clear()
                def refused():
                    events.append('snapshot'); raise RuntimeError('run cancelled')
                with self.assertRaisesRegex(RuntimeError, 'run cancelled'): prepare(after_arm=refused)
                self.assertEqual(events, ['configure', 'arm', 'accepted', 'snapshot', 'close'])

    def test_close_waits_for_concurrent_sdk_calls_and_rejects_new_ones(self):
        owner = Owner(); instance = host.Instance(owner, 'camera'); owner.instances['camera'] = instance
        entered = threading.Event(); release = threading.Event(); closed = threading.Event()
        instance.adapter = object(); instance.closer = closed.set
        def spi(instance, params):
            entered.set(); release.wait(2)
        with patch.dict(host._HANDLERS, {'spi': spi}):
            try:
                instance.dispatch_concurrent({'id': 1, 'op': 'spi', 'p': {}})
                self.assertTrue(entered.wait(2))
                instance.shutdown(wait=False)
                instance.dispatch_concurrent({'id': 2, 'op': 'spi', 'p': {}})
                self.assertIn('closing', str(owner.answers[2][1]['error']))
                self.assertFalse(closed.wait(.05), 'closed SDK while an admitted concurrent call still used it')
                release.set(); instance.shutdown()
                self.assertTrue(closed.is_set())
                self.assertEqual(instance._active_calls, 0)
            finally: release.set(); owner.cleanup()

    def test_shutdown_waits_for_open_and_closes_on_its_lane(self):
        owner = Owner(); entered = threading.Event(); release = threading.Event()
        held = set(); threads = []
        def opened(instance, params):
            held.add('camera'); threads.append(threading.get_ident())
            entered.set(); release.wait(2)
            instance.adapter = object()
            def close():
                threads.append(threading.get_ident()); held.clear()
            instance.closer = close
            return {}
        with patch.dict(host._HANDLERS, {'open': opened}):
            try:
                owner.send(1, 'open'); self.assertTrue(entered.wait(2))
                instance = owner.instances['n100_com3']
                closer = threading.Thread(target=instance.shutdown)
                closer.start(); closer.join(.05)
                self.assertTrue(closer.is_alive(), 'shutdown acknowledged before Open returned its handle')
                self.assertEqual(held, {'camera'})
                release.set(); closer.join(2)
                self.assertFalse(closer.is_alive())
                self.assertFalse(held)
                self.assertEqual(len(threads), 2)
                self.assertEqual(threads[0], threads[1], 'SDK Close left the owning lane')
            finally:
                release.set(); owner.cleanup()

    def test_channel_failure_releases_devices_and_wakes_reverse_calls(self):
        received = threading.Event(); disconnect = threading.Event()
        class Channel:
            def send(self, message): received.set()
            def receive(self):
                disconnect.wait(2)
                raise OSError('channel lost')
        process = host.Process(Channel()); errors = []; held = {'camera'}
        instance = host.Instance(process, 'camera'); process.instances['camera'] = instance
        instance.closer = held.clear
        def reverse_call():
            try: process.request('core', {})
            except Exception as error: errors.append(error)
        waiting = threading.Thread(target=reverse_call); waiting.start()
        self.assertTrue(received.wait(2)); disconnect.set()
        with self.assertRaisesRegex(OSError, 'channel lost'): process.serve()
        waiting.join(2)
        self.assertFalse(waiting.is_alive(), 'shutdown left a run waiting for a reply on the dead channel')
        self.assertFalse(held)
        self.assertEqual(len(errors), 1)
        with self.assertRaisesRegex(host.PluginError, 'closing'): process.request('core', {})

    def test_close_reply_is_a_barrier_before_immediate_reinit(self):
        for name in ['n100_com3', 'pylon_camera']:
            with self.subTest(name=name):
                owner=Owner(); held=set()
                def opened(instance, params):
                    if name in held: raise RuntimeError('device is already in use')
                    held.add(name)
                    instance.adapter=object()
                    instance.closer=lambda: held.remove(name)
                    return {}
                with patch.dict(host._HANDLERS, {'open':opened}):
                    try:
                        owner.send(1,'open',name);self.assertTrue(owner.wait(1))
                        owner.hook=lambda identity: owner.send(3,'open',name) if identity==2 else None
                        owner.send(2,'close',name);self.assertTrue(owner.wait(2))
                        self.assertTrue(owner.wait(3),'re-Init was queued on a dead instance lane')
                        self.assertNotIn('error',owner.answers[3][1])
                        self.assertIn(name, owner.instances)
                        owner.hook=lambda identity: None
                        owner.send(4,'close',name);self.assertTrue(owner.wait(4))
                        self.assertFalse(held)
                    finally: owner.cleanup()

    def test_failed_open_releases_handle_before_error_and_allows_retry(self):
        owner=Owner(); held=set(); count=0
        def opened(instance, params):
            nonlocal count
            count+=1
            if held: raise RuntimeError('device is already in use')
            held.add('port');instance.adapter=object();instance.closer=lambda:held.clear()
            if count==1: raise RuntimeError('initialization verification failed')
            return {}
        with patch.dict(host._HANDLERS, {'open':opened}):
            try:
                owner.hook=lambda identity: owner.send(2,'open') if identity==1 else None
                owner.send(1,'open');self.assertTrue(owner.wait(1))
                self.assertTrue(owner.wait(2),'failed Init poisoned the next Init')
                self.assertNotIn('error',owner.answers[2][1])
            finally: owner.cleanup()

    def test_failed_close_keeps_the_handle_for_cleanup_retry(self):
        owner=Owner(); instance=host.Instance(owner,'n100_com3');owner.instances[instance.name]=instance
        device=object();instance.adapter=device; calls=[]
        def close():
            calls.append(1)
            if len(calls)==1: raise RuntimeError('reader has not stopped yet')
        instance.closer=close
        try:
            owner.send(1,'close');self.assertTrue(owner.wait(1))
            self.assertIs(instance.adapter,device)
            self.assertIs(owner.instances.get(instance.name),instance)
            owner.send(2,'close');self.assertTrue(owner.wait(2))
            self.assertEqual(len(calls),2)
            self.assertNotIn(instance.name,owner.instances)
        finally: owner.cleanup()


    def test_abandoned_operation_keeps_exclusive_ownership_until_cleanup(self):
        owner=Owner(); entered=threading.Event(); release=threading.Event(); held=set()
        def opened(instance, params):
            held.add('port');instance.adapter=object();instance.closer=lambda: held.clear()
            entered.set();release.wait(2)
            return {}
        with patch.dict(host._HANDLERS, {'open':opened}):
            try:
                owner.send(1,'open');self.assertTrue(entered.wait(2))
                old=owner.instances['n100_com3']
                owner._event({'op':'abandon','p':{'instance':'n100_com3'}})
                owner.send(2,'open');self.assertTrue(owner.wait(2))
                self.assertIn('still releasing',str(owner.answers[2][1]['error']))
                self.assertEqual(held,{'port'})
                release.set();old._thread.join(2)
                self.assertFalse(held)
                self.assertNotIn('n100_com3',owner.instances)
                owner.send(3,'open');self.assertTrue(owner.wait(3))
                self.assertNotIn('error',owner.answers[3][1])
            finally: release.set();owner.cleanup()

    def test_invalid_open_is_retired_before_its_error_reply(self):
        owner=Owner(); attempts=[]
        def opened(instance, params):
            attempts.append(instance)
            if len(attempts)==1: raise host._Invalid(ValueError('bad settings'))
            return {}
        with patch.dict(host._HANDLERS, {'open':opened}):
            try:
                owner.hook=lambda identity: owner.send(2,'open') if identity==1 else None
                owner.send(1,'open');self.assertTrue(owner.wait(2))
                self.assertEqual(owner.answers[1][1]['kind'],'invalid')
                self.assertIsNot(attempts[0],attempts[1])
            finally: owner.cleanup()

if __name__=='__main__': unittest.main()
