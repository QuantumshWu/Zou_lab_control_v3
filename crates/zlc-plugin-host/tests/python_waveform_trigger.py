"""The software trigger of Waveform Measurement, at the rates it serves.

The trigger holds nothing: every sample is published as it comes, and an edge is a
mark on the event it fell on. A virtual source samples at 400 Hz to 1 kHz, with one
sample or a block in each of its records and one or several quantities, and streams
signals of 1-50 Hz with 1-5% noise. The same through the real host and Runtime is
tests/waveform_commit.rs. Run directly with Python.
"""
import pathlib
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'science' / 'src'))
from zlc_atom.devices.simulation.waveform.source import VirtualWaveformConfig, VirtualWaveformSource
from zlc_atom.devices.waveform.contract import WaveformAcquisitionMode, WaveformCaptureTerminalRecord, WaveformOutput, WaveformWorkingPoint
from zlc_atom.devices.waveform.wheeltec_n100 import N100_OUTPUTS
from zlc_atom.nodes.waveform_measurement import logic_node
from zlc_atom.nodes.waveform_measurement.measurement import (
    WAVEFORM_MEASUREMENT_SCHEMA, FiniteCapture, MonitorCapture, WaveformMeasurementNode, WaveformMeasurementRequest,
    waveform_preview)
from zlc_atom.nodes.waveform_measurement.trigger import TriggerDetector, TriggerSettings, WaveformTrigger, channel_choices, unit_text
from zlc_plugin import capabilities
from zlc_plugin.declare import QUERY_PARAMETERS, field_json

RATE_HZ = 1000.0
VOLTAGE = WaveformOutput('voltage', 'V', ('CH1', 'CH2'), (0, 1))


def sine(frequency, noise=0.0, seed=1, phase=0.0):
    """Two channels: a sine on the first, its negative on the second; noise relative to amplitude."""

    rng = np.random.default_rng(seed)

    def samples(times):
        wave = np.sin(2 * np.pi * frequency * times + phase)
        return np.stack((wave + noise * rng.standard_normal(times.size), -wave), axis=1).astype(np.float32)
    return samples


def source_of(signal, *, rate=RATE_HZ, samples=1, outputs=(VOLTAGE,)):
    return VirtualWaveformSource(VirtualWaveformConfig(rate, samples, outputs), sample_source=signal)


def node_of(source, *, repeat=0, trigger=None, producer='wf'):
    node = WaveformMeasurementNode(sampler=source, request=WaveformMeasurementRequest('s', repeat, trigger), producer=producer)
    node._generation = 'run'
    node._configure()
    node._arm()
    return node


def detector(level=0.0, slope='rising', rate=RATE_HZ):
    return TriggerDetector(WaveformTrigger(0, level, slope, 'x', 'V'), 1.0 / rate)


class Commits:
    """What the context's commit_events was given, as the events it says."""

    def __init__(self):
        self.ordinals, self.shot_times, self.shown, self.records, self.values = [], [], [], [], {}
        self.commits = 0
        self.origins = []

    def __call__(self, outputs, *, ordinals, shot_times=None, origins=None, shown=None, records=None):
        self.commits += 1
        count = len(ordinals)
        assert all(output.values.shape[0] == count for output in outputs.values())
        assert len(records) == count
        self.ordinals += [int(item) for item in ordinals]
        self.shot_times += [] if shot_times is None else [float(item) for item in shot_times]
        self.shown += [True] * count if shown is None else [bool(item) for item in shown]
        self.records += records
        self.origins += [] if origins is None else [tuple(int(cell) for cell in row) for row in origins]
        for name, output in outputs.items():
            self.values.setdefault(name, []).append(output.values)
            self.canonical = output.canonical_schema

    def column(self, name, channel=0):
        return np.concatenate(self.values[name]).reshape(len(self.ordinals), -1, self.values[name][0].shape[-1])[:, 0, channel]


def monitored(signal, seconds, *, trigger=None, **source):
    """A monitor over a live source for ``seconds``, then Stop."""

    device = source_of(signal, **source)
    node = node_of(device, trigger=trigger)
    commits = Commits()
    capture = MonitorCapture(node, commit_events=commits)
    stop = time.monotonic() + seconds
    try:
        while time.monotonic() < stop:
            capture.poll()
        terminal = capture.close()
    finally:
        device.close()
    return node, commits, terminal


class Detection(unittest.TestCase):
    def edges(self, frequency, noise, seconds, *, slope='rising', level=0.0, cut=1, seed=3):
        times = np.arange(int(seconds * RATE_HZ)) / RATE_HZ
        signal = sine(frequency, noise, seed)(times)[:, 0]
        scan = detector(level, slope)
        marks = np.concatenate([scan.scan(part) for part in np.array_split(signal, max(1, signal.size // cut))])
        return signal, np.flatnonzero(marks)

    def test_ordinary_signals_fire_once_a_period_on_the_chosen_edge(self):
        for frequency in (1.0, 2.0, 5.0, 23.7, 50.0):
            for noise in (0.0, 0.01, 0.05):
                with self.subTest(frequency=frequency, noise=noise):
                    seconds = max(4.0, 20.0 / frequency)
                    signal, edges = self.edges(frequency, noise, seconds)
                    # The first swing of a slow noisy signal has seen no extreme yet (a known limit).
                    edges = edges[edges >= RATE_HZ * 1.5 / frequency]
                    periods = frequency * seconds - 1.5
                    self.assertGreaterEqual(edges.size, periods - 2)
                    self.assertLessEqual(edges.size, periods + 1)
                    self.assertTrue(np.all(signal[edges] >= 0.0) and np.all(signal[edges - 1] < 0.0))
                    # On the rising side of the clean wave: the phase is the lock.
                    phase = np.mod(frequency * edges / RATE_HZ, 1.0)
                    phase = np.minimum(phase, 1.0 - phase)
                    self.assertLess(phase.max(), 0.08 + 2 * noise, 'fired off the rising zero crossing')

    def test_falling_is_the_rising_of_the_negated_signal(self):
        signal, edges = self.edges(10.0, 0.02, 4.0, slope='falling')
        self.assertTrue(np.all(signal[edges] <= 0.0) and np.all(signal[edges - 1] > 0.0))
        self.assertGreater(edges.size, 35)

    def test_how_reads_cut_the_stream_changes_nothing(self):
        _, whole = self.edges(23.7, 0.03, 4.0, cut=4000)
        for cut in (1, 7, 50):
            self.assertEqual(self.edges(23.7, 0.03, 4.0, cut=cut)[1].tolist(), whole.tolist())

    def test_a_level_nothing_reaches_fires_nothing_and_the_range_says_so(self):
        scan = detector(5.0)
        self.assertFalse(scan.scan(np.sin(np.arange(1000) / 10.0)).any())
        low, high = scan.take_range()
        self.assertAlmostEqual(low, -1.0, 2)
        self.assertAlmostEqual(high, 1.0, 2)
        self.assertIsNone(scan.take_range())

    def test_samples_that_could_not_be_read_fire_nothing(self):
        scan = detector(0.0)
        values = np.sin(np.arange(1000) / 10.0)
        values[::7] = np.nan
        marks = scan.scan(values)
        self.assertTrue(np.all(np.isfinite(values[marks])))


class Publication(unittest.TestCase):
    """Only the marks differ from trigger off: the same events, published as they arrive."""

    def test_every_sample_is_published_as_it_comes_and_edges_are_marked(self):
        for samples, outputs in ((1, (VOLTAGE,)), (25, (VOLTAGE,)), (1, N100_OUTPUTS), (25, N100_OUTPUTS)):
            with self.subTest(samples=samples, outputs=len(outputs)):
                columns = 1 + max(column for output in outputs for column in output.columns)

                def signal(times, inner=sine(23.7, 0.03)):
                    return np.tile(inner(times)[:, :1], (1, columns)).astype(np.float32)

                channel = f'{outputs[0].name}/{outputs[0].channel_labels[0]}'
                node, commits, terminal = monitored(signal, 1.5, trigger=TriggerSettings(channel, 0.0, 'rising'),
                                                    samples=samples, outputs=outputs)
                self.assertEqual(commits.ordinals, list(range(terminal.produced_count * samples)), 'Stop leaves nothing unpublished')
                expected = np.arange(len(commits.ordinals)) / RATE_HZ
                np.testing.assert_allclose(commits.shot_times, expected, atol=1e-9)
                self.assertTrue(np.all(np.diff(commits.shot_times) > 0))
                values = commits.column(outputs[0].name)
                marks = np.flatnonzero(commits.shown)
                self.assertGreater(marks.size, 20)
                self.assertTrue(np.all(values[marks] >= 0.0) and np.all(values[marks - 1] < 0.0))
                # Single-sample events, the shape of a single-sample source off.
                self.assertEqual(commits.values[outputs[0].name][0].shape[1:3], (1, 1))
                # Reads follow the slice, not the sample.
                self.assertLess(commits.commits, 40, 'one commit a read')
                timing = commits.records[10]['record_timing']['wf']['10']
                self.assertAlmostEqual(timing['record_time_seconds'], commits.shot_times[10])
                self.assertEqual(timing['time_basis'], 'sample_clock')

    def test_trigger_off_marks_every_event_and_keeps_the_records_whole(self):
        for samples in (1, 25):
            node, commits, terminal = monitored(sine(23.7, 0.03), 0.8, samples=samples)
            self.assertEqual(commits.ordinals, list(range(terminal.produced_count)))
            self.assertTrue(all(commits.shown))
            self.assertEqual(commits.values['voltage'][0].shape[1:], (1, samples, 2))
            np.testing.assert_allclose(commits.shot_times, np.arange(terminal.produced_count) * samples / RATE_HZ, atol=1e-9)
            self.assertLess(commits.commits, 30)

    def test_invalid_samples_stay_invalid_through_a_commit(self):
        def signal(times):
            out = np.ones((times.size, 2), dtype=np.float32)
            out[times > 0.1] = np.nan
            return out
        device = source_of(signal)
        node = node_of(device)
        seen = []

        def commit(outputs, **_):
            seen.append(outputs['voltage'].validity)
        capture = MonitorCapture(node, commit_events=commit)
        stop = time.monotonic() + 0.4
        while time.monotonic() < stop:
            capture.poll()
        capture.close()
        device.close()
        self.assertTrue(any(valid is True for valid in seen))
        masks = [valid for valid in seen if valid is not True]
        self.assertTrue(masks and masks[-1].shape[1:] == (1, 1, 2) and not masks[-1].all())


class FiniteRuns(unittest.TestCase):
    def collect(self, signal, repeat, *, trigger=None, samples=1, stop=None):
        device = source_of(signal, samples=samples)
        node = node_of(device, repeat=repeat, trigger=trigger)
        commits = Commits()
        capture = FiniteCapture(node, should_stop=stop)
        try:
            kept = capture.collect(commit=commits)
        finally:
            device.close()
        return kept, commits, capture

    def test_repeat_is_n_consecutive_samples_from_the_first_edge(self):
        for samples in (1, 25):
            trigger = TriggerSettings('voltage/CH1', 0.0, 'rising')
            # Starts high and falling, so its first samples are well before the first edge.
            kept, commits, capture = self.collect(sine(5.0, 0.0, phase=np.pi / 2), 300, trigger=trigger, samples=samples)
            self.assertEqual(kept, 300)
            self.assertEqual(commits.ordinals, list(range(300)))
            self.assertEqual(commits.origins, [(index, 0) for index in range(300)])
            values = commits.column('voltage')
            self.assertGreaterEqual(values[0], 0.0)
            self.assertLess(values[0], 0.1)
            self.assertTrue(np.all(np.diff(values[:10]) > 0))
            self.assertEqual(commits.canonical.repeat_domain.size, 300)
            self.assertTrue(capture.terminal.source_stopped)
            first = commits.records[0]['record_timing']['wf']['0']
            self.assertEqual(first['record_time_seconds'], 0.0)
            self.assertGreater(first['source_time_seconds'], 0.05, 'nothing before the first trigger was kept')

    def test_stopped_before_any_edge_keeps_nothing(self):
        stop = threading.Event()
        threading.Timer(0.3, stop.set).start()
        kept, commits, capture = self.collect(lambda times: np.ones((times.size, 2), dtype=np.float32), 100,
                                              trigger=TriggerSettings('voltage/CH1', 5.0, 'rising'), stop=stop.is_set)
        self.assertEqual((kept, commits.commits), (0, 0))
        self.assertTrue(capture.terminal.no_more_records)

    def test_repeat_without_a_trigger_is_as_it_was(self):
        kept, commits, capture = self.collect(sine(5.0), 60)
        self.assertEqual(kept, 60)
        self.assertEqual(commits.ordinals, list(range(60)))
        self.assertTrue(capture.terminal.no_more_records)
        kept, commits, capture = self.collect(sine(5.0), 20, samples=10)
        self.assertEqual(commits.values['voltage'][0].shape[1:], (1, 10, 2))
        self.assertEqual(commits.origins, [(index, 0) for index in range(20)])


class Triggered(VirtualWaveformSource):
    """A source with its own trigger that no pulse fires."""

    def working_point(self):
        return WaveformWorkingPoint(WaveformAcquisitionMode.EXTERNAL_TRIGGERED, 1e-3, {}, time_basis='device_clock')

    def arm(self, records, *, buffer_record_count):
        pass

    def read_records(self, n, *, timeout, exact):
        time.sleep(timeout)
        return []

    def finish_record_capture(self):
        return WaveformCaptureTerminalRecord(0, True, True, True)


class Quiet(unittest.TestCase):
    def test_waiting_for_an_edge_is_a_status_that_says_what_it_saw(self):
        def centred(times):
            return (30.0 + np.sin(times)[:, None] * np.ones((1, 2))).astype(np.float32)
        device = source_of(centred, outputs=(WaveformOutput('field', 'uT', ('z', 'y'), (0, 1)),), rate=400.0)
        node = node_of(device, trigger=TriggerSettings('field/z', 0.0, 'rising'))
        said = []
        node._report_status = lambda message, **_: said.append(message)
        commits = Commits()
        capture = MonitorCapture(node, commit_events=commits)
        stop = time.monotonic() + 2.3
        try:
            while time.monotonic() < stop:
                capture.poll()
            capture.close()
        finally:
            device.close()
        self.assertGreaterEqual(len(said), 2, 'it keeps saying what it sees')
        self.assertRegex(said[0], r'^Waiting for trigger: field z ranged 3\d(\.\d+)? to 3\d(\.\d+)? µT, level 0$')
        self.assertFalse(any(commits.shown), 'no view may stand anywhere')
        self.assertEqual(commits.ordinals, list(range(len(commits.ordinals))), 'and every sample is published')

    def test_the_first_edge_ends_the_wait(self):
        def signal(times):
            out = np.full((times.size, 2), -1.0, dtype=np.float32)
            out[times > 1.4, 0] = 1.0
            return out
        device = source_of(signal)
        node = node_of(device, trigger=TriggerSettings('voltage/CH1', 0.0, 'rising'))
        said = []
        node._report_status = lambda message, **_: said.append(message)
        capture = MonitorCapture(node, commit_events=Commits())
        stop = time.monotonic() + 1.9
        try:
            while time.monotonic() < stop:
                capture.poll()
            capture.close()
        finally:
            device.close()
        self.assertTrue(said[0].startswith('Waiting for trigger'))
        self.assertEqual(said[-1], 'Triggered')

    def test_a_source_that_stops_delivering_is_a_fault_trigger_or_not(self):
        class Dead(VirtualWaveformSource):
            def read_records(self, n, *, timeout, exact):
                time.sleep(timeout)
                return []

            @property
            def timeout(self):
                return 0.2

        for trigger in (None, TriggerSettings('voltage/CH1', 0.0, 'rising')):
            device = Dead(VirtualWaveformConfig(RATE_HZ, 1, (VOLTAGE,)), sample_source=sine(5.0))
            node = node_of(device, trigger=trigger)
            capture = MonitorCapture(node, commit_events=Commits())
            with self.assertRaisesRegex(TimeoutError, 'stopped delivering'):
                until = time.monotonic() + 3.0
                while time.monotonic() < until:
                    capture.poll()
            capture.close(drain=False)
            device.close()

    def test_a_finite_capture_on_a_source_with_its_own_trigger_waits_for_it(self):
        device = Triggered(VirtualWaveformConfig(RATE_HZ, 5, (VOLTAGE,)), sample_source=sine(5.0))
        stop = threading.Event()
        node = node_of(device, repeat=3)
        said = []
        node._report_status = lambda message, **_: said.append(message)
        threading.Timer(1.3, stop.set).start()
        capture = FiniteCapture(node, should_stop=stop.is_set)
        try:
            self.assertEqual(capture.collect(commit=Commits()), 0)
        finally:
            device.close()
        self.assertEqual(said[0], 'Waiting for trigger')

    def test_a_software_trigger_on_a_source_with_its_own_is_refused(self):
        device = Triggered(VirtualWaveformConfig(RATE_HZ, 5, (VOLTAGE,)), sample_source=sine(5.0))
        node = WaveformMeasurementNode(sampler=device, request=WaveformMeasurementRequest('s', 0, TriggerSettings('voltage/CH1', 0.0, 'rising')), producer='wf')
        with self.assertRaisesRegex(ValueError, 'hardware trigger'):
            node._configure()
        device.close()


class Form(unittest.TestCase):
    def source(self):
        return source_of(sine(5.0), outputs=N100_OUTPUTS)

    def test_the_trigger_has_three_parameters_on_the_form(self):
        rows = {row['name']: row for row in map(field_json, WAVEFORM_MEASUREMENT_SCHEMA.fields)}
        self.assertEqual(list(rows), ['repeat', 'trigger_slope', 'trigger_channel', 'trigger_level'])
        self.assertEqual(rows['trigger_slope']['label'], 'Slope')
        self.assertEqual(rows['trigger_slope']['default'], 'off')
        self.assertEqual(rows['trigger_channel']['group'], 'Trigger')

    def test_channels_and_the_level_read_in_display_units(self):
        self.assertEqual([unit_text(unit) for unit in ('uT', 'rad*s^-1', 'm*s^-2', 'K', 'V')], ['µT', 'rad/s', 'm/s²', 'K', 'V'])
        choices = channel_choices(N100_OUTPUTS, 'sample_clock')
        self.assertIn(['magnetic_field/z', 'magnetic_field z (µT)'], choices)
        self.assertIn(['angular_rate/x', 'angular_rate x (rad/s)'], choices)
        answer = logic_node.LOGIC_NODE.resolve_field_units
        self.assertEqual(QUERY_PARAMETERS['resolve_field_units'], ('values', 'devices'))
        self.assertEqual(answer({'trigger_channel': 'angular_rate/y'}, {'sampler': self.source()}), {'trigger_level': 'rad/s'})
        self.assertEqual(answer({'trigger_channel': ''}, {'sampler': self.source()}), {})
        self.assertEqual(answer({}, {}), {})

    def test_the_preview_is_a_rolling_view_of_the_trigger_channel(self):
        source = self.source()
        free = waveform_preview(source)
        self.assertEqual((free.output.name, free.plot_kind, dict(free.semantic)), ('magnetic_field', 'rolling', {}))
        triggered = waveform_preview(source, trigger_channel='angular_rate/z', node='n1')
        self.assertEqual((triggered.output.name, triggered.plot_kind), ('angular_rate', 'rolling'))
        self.assertEqual(dict(triggered.semantic), {'fate:cell_data:n1.angular_rate.0.component': ['scope-value', 2]})
        self.assertEqual(waveform_preview(source, trigger_channel='', node='n1').output.name, 'magnetic_field')
        # Block records are single samples when triggered, and their own trace when not.
        block = source_of(sine(5.0), samples=50)
        self.assertEqual(waveform_preview(block).plot_kind, 'curve')
        self.assertEqual(waveform_preview(block, trigger_channel='voltage/CH2', node='n').plot_kind, 'rolling')

    def test_a_source_with_its_own_trigger_says_so_on_the_slope_row(self):
        device = Triggered(VirtualWaveformConfig(RATE_HZ, 5, (VOLTAGE,)), sample_source=sine(5.0))
        answer = logic_node.LOGIC_NODE.resolve_field_availability({'sampler': device})
        self.assertEqual(list(answer), ['trigger_slope'])
        self.assertIn('hardware trigger', answer['trigger_slope'])
        self.assertEqual(logic_node.LOGIC_NODE.resolve_field_availability({'sampler': self.source()}), {})

    def test_a_draft_query_may_read_the_working_point(self):
        self.assertIn('working_point', capabilities.READS)


if __name__ == '__main__':
    unittest.main()
