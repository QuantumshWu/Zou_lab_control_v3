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
    def test_predecode_refusal_releases_each_unread_block_lend_once(self):
        import queue
        owner = Owner(); owner._releases = queue.SimpleQueue()
        message = {"id": 1, "op": "spi", "p": {"instance": "missing"}, "blocks": ["same", "same"]}
        owner._request(message)
        self.assertEqual([owner._releases.get_nowait(), owner._releases.get_nowait()], [("blocks", "same"), ("blocks", "same")])
        self.assertTrue(owner._releases.empty())
        host._discard_frame(owner, message)
        self.assertTrue(owner._releases.empty(), "a refusal cannot release the same frame twice")

    def test_partial_array_decode_failure_transfers_finalizer_ownership_without_double_release(self):
        import gc, queue, uuid
        import numpy as np
        from zlc_plugin.blocks import Arena, Attachments
        releases = queue.SimpleQueue()
        attachments = Attachments(lambda name: releases.put(("blocks", name)))
        arena = Arena(uuid.uuid4().hex)
        array = arena.allocate(4, np.float64)
        first = arena.lend(array, attachments)
        second = arena.lend(array, attachments)
        message = {"p": {"first": {"$array": first}, "second": {"$array": {**second, "dtype": "not-a-dtype"}}},
                   "blocks": [first["block"], second["block"]]}
        process = types.SimpleNamespace(attachments=attachments, _releases=releases)
        with self.assertRaises(TypeError): host._decode_frame(process, message)
        gc.collect()
        released = []
        while not releases.empty(): released.append(releases.get_nowait())
        self.assertEqual(released, [("blocks", first["block"])] * 2)
        self.assertNotIn("blocks", message)

    def test_transport_delayed_frame_cannot_start_a_fresh_relative_sdk_deadline(self):
        import time
        owner = Owner(); writes = []
        instance = host.Instance(owner, "stage"); owner.instances["stage"] = instance
        with patch.dict(host._HANDLERS, {"spi": lambda instance, params: writes.append("late write")}):
            try:
                owner._request({"id": 1, "op": "spi", "p": {"instance": "stage"}, "deadline_ms": 10_000,
                    "deadline_monotonic_ns": time.monotonic_ns() - 1,
                    "deadline_unix_ns": time.time_ns() - 1})
                self.assertTrue(owner.wait(1))
                self.assertIn("expired", str(owner.answers[1][1]["error"]))
                self.assertEqual(writes, [])
            finally: owner.cleanup()

    def test_expired_queued_device_call_never_runs(self):
        owner = Owner(); entered = threading.Event(); release = threading.Event(); writes = []
        def call(instance, params):
            if params.get("wait"):
                entered.set(); release.wait(2)
            else:
                writes.append(params.get("value"))
            return {}
        with patch.dict(host._HANDLERS, {"spi": call}):
            instance = host.Instance(owner, "stage"); owner.instances["stage"] = instance
            try:
                owner._request({"id": 1, "op": "spi", "p": {"instance": "stage", "wait": True}})
                self.assertTrue(entered.wait(2))
                owner._request({"id": 2, "op": "spi", "deadline_ms": 0, "p": {"instance": "stage", "value": 5}})
                release.set(); self.assertTrue(owner.wait(2))
                self.assertIn("expired", str(owner.answers[2][1]["error"]))
                self.assertEqual(writes, [])
            finally: release.set(); owner.cleanup()

    def test_full_lane_rejects_new_calls_but_abandon_keeps_control_wakeup(self):
        owner = Owner(); entered = threading.Event(); release = threading.Event()
        def call(instance, params):
            entered.set(); release.wait(2); return {}
        with patch.object(host, "_LANE_REQUESTS", 2), patch.dict(host._HANDLERS, {"spi": call}):
            instance = host.Instance(owner, "stage"); owner.instances["stage"] = instance
            try:
                owner.send(1, "spi", "stage"); self.assertTrue(entered.wait(2))
                owner.send(2, "spi", "stage"); owner.send(3, "spi", "stage"); owner.send(4, "spi", "stage")
                self.assertIn("queued", str(owner.answers[4][1]["error"]))
                instance.abandon()
                release.set(); instance._thread.join(2)
                self.assertFalse(instance._thread.is_alive())
                self.assertNotIn("stage", owner.instances)
            finally: release.set(); owner.cleanup()

    def test_concurrent_call_admission_is_bounded(self):
        owner = Owner(); entered = threading.Event(); release = threading.Event()
        instance = host.Instance(owner, "stage"); owner.instances["stage"] = instance
        def call(instance, params): entered.set(); release.wait(2); return {}
        with patch.object(host, "_CONCURRENT_CALLS", 1), patch.dict(host._HANDLERS, {"spi": call}):
            try:
                instance.dispatch_concurrent({"id": 1, "op": "spi", "p": {}})
                self.assertTrue(entered.wait(2))
                instance.dispatch_concurrent({"id": 2, "op": "spi", "p": {}})
                self.assertIn("concurrent", str(owner.answers[2][1]["error"]))
            finally: release.set(); owner.cleanup()

    def test_stop_has_independent_admission_reserve_while_long_reads_are_pending(self):
        owner = Owner(); entered = threading.Event(); release = threading.Event(); stopped = threading.Event()
        instance = host.Instance(owner, "stage"); owner.instances["stage"] = instance
        def call(instance, params):
            if params.get("method") == "stop": stopped.set(); release.set()
            else: entered.set(); release.wait(2)
            return {}
        with patch.object(host, "_CONCURRENT_CALLS", 1), patch.dict(host._HANDLERS, {"spi": call}):
            try:
                instance.dispatch_concurrent({"id": 1, "op": "spi", "p": {"method": "read"}})
                self.assertTrue(entered.wait(2))
                instance.dispatch_concurrent({"id": 2, "op": "spi", "p": {"method": "read"}})
                self.assertIn("concurrent", str(owner.answers[2][1]["error"]))
                instance.shutdown(wait=False)
                instance.dispatch_concurrent({"id": 3, "op": "spi", "p": {"method": "stop"}})
                self.assertTrue(stopped.wait(2), "ordinary reads exhausted the emergency reserve")
                self.assertTrue(owner.wait(3))
                instance.shutdown()
                self.assertEqual((instance._active_calls, instance._active_control_calls), (0, 0))
            finally: release.set(); owner.cleanup()

    def test_terminal_cleanup_follows_core_decision_and_finishes_with_token(self):
        from zlc_plugin.context import RunContext
        from zlc_plugin.declare import NodeKind
        for outcome, writer, raises in (("completed", False, False), ("stopped", True, False), ("failed", True, True)):
            with self.subTest(outcome=outcome):
                calls = []
                def request(op, params):
                    calls.append((op, params))
                    if op == "run.begin_terminal":
                        return {"token": "core-owned-token", "outcome": outcome, "run_exit_writer": writer,
                                "stop_accepted": outcome == "stopped", "partial": False}
                    self.assertEqual(op, "run.finish")
                    self.assertNotIn("run-1", instance.runs, "core finish must observe a reusable local runner")
                    self.assertTrue(run.ended, "delivery cleanup precedes completion")
                    self.assertEqual(params["token"], "core-owned-token")
                    self.assertNotIn("outcome", params)
                    return {"outcome": outcome}
                process = types.SimpleNamespace(request=request)
                instance = types.SimpleNamespace(name="task", process=process, runs={})
                run = object.__new__(host.Run)
                run.instance, run.run_id, run.params = instance, "run-1", {}
                instance.runs[run.run_id] = run
                run.declared = types.SimpleNamespace(kind=NodeKind.MEASUREMENT)
                run.lock, run.cancel = threading.Lock(), threading.Event()
                run.partial = run.ended = False
                run.exit_failure = None
                run.partial_exit_writer = lambda status, error: calls.append(("cleanup", status))
                run.deliveries = __import__("queue").Queue()
                def execute(context):
                    if raises: raise RuntimeError("work failed")
                    return {"count": 1}
                run.prepared_node = types.SimpleNamespace(execute=execute)
                run.context = RunContext(run)
                run._main()
                self.assertEqual([op for op, _ in calls], ["run.begin_terminal", *(["cleanup"] if writer else []), "run.finish"])
                self.assertNotIn(run.run_id, instance.runs)

    def test_terminal_seal_only_caches_the_core_acknowledgement(self):
        from zlc_plugin.context import RunContext
        calls = []
        def request(op, params):
            calls.append((op, params))
            return {"token": "core-token", "stop_accepted": True, "partial": True}
        run = types.SimpleNamespace(instance=types.SimpleNamespace(name="task", process=types.SimpleNamespace(request=request)),
                                    run_id="run-1", lock=threading.Lock())
        RunContext(run).seal_terminal(accept_stop=True, partial=True)
        self.assertEqual(run.terminal_token, "core-token")
        self.assertTrue(run.sealed and run.stop_accepted and run.partial)
        self.assertEqual(calls[0][0], "run.begin_terminal")

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
