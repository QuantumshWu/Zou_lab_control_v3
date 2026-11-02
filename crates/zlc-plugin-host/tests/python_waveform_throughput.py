"""Waveform Measurement keeps up with a fast free-running source.

The sampler a run holds is a device reached through the core: every
attribute read and every call is a round trip.  These tests put the virtual
IMU behind a proxy that charges each such access, and check that a capture
reaches the source once per batch (never once per record) and catches up
after a stall instead of overflowing the source's 2 s receive buffer.
"""
import pathlib
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'science' / 'src'))
from zlc_atom.devices import RecordQueue
from zlc_atom.devices.simulation.waveform.source import VirtualWaveformConfig, VirtualWaveformSource
from zlc_atom.devices.waveform.wheeltec_n100 import N100_OUTPUTS
from zlc_atom.nodes.waveform_measurement.measurement import (
    FiniteCapture, MonitorCapture, WaveformMeasurementNode, WaveformMeasurementRequest)

#: What one round trip through the core costs a node (measured: about 0.7 ms).
ROUND_TRIP_SECONDS = 0.001


class Remote:
    """A source as a run sees it (zlc_plugin's device proxy): each property
    read and each call is a charged round trip, and counted."""

    def __init__(self, source):
        self._source = source
        self.accesses = {}

    def _reach(self, name):
        self.accesses[name] = self.accesses.get(name, 0) + 1
        time.sleep(ROUND_TRIP_SECONDS)
        return getattr(self._source, name)


for _name in ('timeout', 'outputs', 'record_samples'):
    setattr(Remote, _name, property(lambda self, name=_name: self._reach(name)))
for _name in ('working_point', 'arm', 'read_records', 'finish_record_capture', 'capture_state'):
    setattr(Remote, _name, lambda self, *args, name=_name, **kwargs: self._reach(name)(*args, **kwargs))


def imu(rate_hz):
    rng = np.random.default_rng(1)
    return VirtualWaveformSource(VirtualWaveformConfig(rate_hz, 1, N100_OUTPUTS),
                                 sample_source=lambda times: rng.normal(size=(times.size, 10)).astype(np.float32))


def node_for(remote, repeat=0):
    node = WaveformMeasurementNode(sampler=remote, request=WaveformMeasurementRequest('imu', repeat), producer='wf')
    node._generation = 'run'
    node._configure()
    node._arm()
    remote.accesses.clear()
    return node


class Committed:
    """commit_live that costs ``seconds`` per event and keeps the ordinals, once stalled."""

    def __init__(self, seconds, stall_at=None, stall_seconds=0.0):
        self.seconds, self.stall_at, self.stall_seconds = seconds, stall_at, stall_seconds
        self.ordinals = []

    def __call__(self, outputs):
        (first, *_rest) = outputs.values()
        self.ordinals.append(int(next(iter(first.event_record['record_timing']['wf']))))
        if self.stall_at is not None and len(self.ordinals) == self.stall_at:
            time.sleep(self.stall_seconds)
        deadline = time.perf_counter() + self.seconds
        while time.perf_counter() < deadline:
            pass


class RecordQueueReads(unittest.TestCase):
    def test_a_loose_read_takes_what_is_there_at_once_and_an_exact_one_waits(self):
        class Record:
            def __init__(self, ordinal): self.source_ordinal = ordinal
        queue = RecordQueue('test', join_timeout_seconds=1.0)
        queue.arm(None, buffer_record_count=10)
        for ordinal in range(3):
            self.assertTrue(queue.push(Record(ordinal)))
        started = time.monotonic()
        self.assertEqual([r.source_ordinal for r in queue.read(10, timeout=1.0, exact=False)], [0, 1, 2])
        self.assertLess(time.monotonic() - started, 0.5, 'a loose read must not wait for the full count')
        self.assertEqual(queue.read(10, timeout=0.05, exact=False), [], 'an empty queue answers nothing at the timeout')
        threading.Timer(0.05, lambda: queue.push(Record(3))).start()
        self.assertEqual([r.source_ordinal for r in queue.read(10, timeout=1.0, exact=False)], [3])
        queue.push(Record(4))
        with self.assertRaises(TimeoutError):
            queue.read(2, timeout=0.05, exact=True)
        self.assertEqual([r.source_ordinal for r in queue.read(1, timeout=0.0, exact=True)], [4])
        queue.finish()


class MonitorThroughput(unittest.TestCase):
    def run_monitor(self, rate_hz, seconds, committed):
        remote = Remote(imu(rate_hz))
        node = node_for(remote)
        capture = MonitorCapture(node, commit_live=committed)
        polls = 0
        stop = time.monotonic() + seconds
        try:
            while time.monotonic() < stop:
                capture.poll()
                polls += 1
        finally:
            terminal = capture.close()
            remote._source.close()
        return remote, capture, terminal, polls

    def test_a_backlog_is_one_read_and_the_capture_catches_up_after_a_stall(self):
        # 2 kHz, 0.25 ms of publishing per record, and one 1.5 s stall: the
        # backlog (3000 records, inside the 4000-record buffer) is taken in a
        # few reads and published, and the capture is current again.
        committed = Committed(0.00025, stall_at=200, stall_seconds=1.5)
        remote, capture, terminal, polls = self.run_monitor(2000.0, 3.5, committed)
        self.assertEqual(committed.ordinals, list(range(terminal.produced_count)), 'every record, once, in order')
        self.assertTrue(terminal.no_more_records)
        self.assertGreater(terminal.produced_count, 3000)
        reads = remote.accesses.get('read_records', 0)
        self.assertLess(reads, terminal.produced_count / 2, 'the source is read per batch, not per record')
        # The facts of the source were read when the run was configured, not per record.
        for name in ('record_samples', 'timeout', 'outputs', 'working_point'):
            self.assertNotIn(name, remote.accesses, f'{name} read during the capture')

    def test_the_cost_per_record_stays_flat_over_a_long_capture(self):
        committed = Committed(0.0002)
        started = time.perf_counter()
        remote, capture, terminal, polls = self.run_monitor(1000.0, 6.0, committed)
        self.assertEqual(committed.ordinals, list(range(terminal.produced_count)))
        # Kept pace with the source: everything it produced while running was published.
        self.assertGreater(terminal.produced_count, 0.9 * 1000.0 * (time.perf_counter() - started - 1.0))


class FiniteThroughput(unittest.TestCase):
    def test_a_finite_capture_takes_batches_and_keeps_every_index(self):
        remote = Remote(imu(2000.0))
        node = node_for(remote, repeat=3000)
        committed = []
        stalled = []

        def commit_shot(record, index):
            if index == 100 and not stalled:
                stalled.append(True)
                time.sleep(0.8)
            committed.append((index, record.source_ordinal))

        capture = FiniteCapture(node, should_stop=None)
        try:
            self.assertEqual(capture.collect(commit_shot=commit_shot), 3000)
        finally:
            remote._source.close()
        self.assertEqual(committed, [(index, index) for index in range(3000)])
        self.assertTrue(capture.terminal.no_more_records)
        self.assertLess(remote.accesses.get('read_records', 0), 1500)

    def test_a_stopped_finite_capture_publishes_what_it_had_taken_and_what_the_source_held(self):
        remote = Remote(imu(2000.0))
        node = node_for(remote, repeat=100000)
        stop = threading.Event()
        committed = []

        def commit_shot(record, index):
            committed.append((index, record.source_ordinal))
            if index == 500:
                stop.set()
                time.sleep(0.3)  # records pile up, some taken, some still in the source

        capture = FiniteCapture(node, should_stop=stop.is_set)
        try:
            kept = capture.collect(commit_shot=commit_shot)
        finally:
            remote._source.close()
        self.assertEqual(kept, capture.terminal.produced_count)
        self.assertEqual(committed, [(index, index) for index in range(kept)])
        self.assertTrue(capture.terminal.no_more_records)


if __name__ == '__main__':
    unittest.main()
