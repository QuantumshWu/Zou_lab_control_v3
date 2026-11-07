"""The software trigger of Waveform Measurement, at the rates it serves.

Each edge gives one trace with the crossing in its middle, on a time axis that is zero
there. A virtual source samples at 400 Hz to 1 kHz, with one sample or a block in each
of its records and one or several quantities, and streams signals of 1-50 Hz with 1-5%
noise. The same through the real host and Runtime is
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
from zlc_atom.nodes.waveform_measurement.trigger import TraceAssembler, TriggerDetector, TriggerSettings, WaveformTrigger, channel_choices, unit_text
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
    return TriggerDetector(WaveformTrigger(0, level, slope, 'x', 'V', 2), 1.0 / rate)


class Commits:
    """What the context's commit_events was given, as the events it says."""

    def __init__(self):
        self.ordinals, self.shot_times, self.records, self.values = [], [], [], {}
        self.commits = 0
        self.origins = []

    def __call__(self, outputs, *, ordinals, shot_times=None, origins=None, records=None):
        self.commits += 1
        count = len(ordinals)
        assert all(output.values.shape[0] == count for output in outputs.values())
        assert len(records) == count
        self.ordinals += [int(item) for item in ordinals]
        self.shot_times += [] if shot_times is None else [float(item) for item in shot_times]
        self.records += records
        self.origins += [] if origins is None else [tuple(int(cell) for cell in row) for row in origins]
        for name, output in outputs.items():
            self.values.setdefault(name, []).append(output.values)
            self.canonical = output.canonical_schema

    def traces(self, name, channel=0):
        """``(traces, samples)`` of one channel."""

        return np.concatenate(self.values[name])[:, 0, 0, channel, :]


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


def square(times):
    """A square-like wave of unit swing with a 10 ms edge, not locked to the sampling."""

    phase = np.mod(times, 0.5137)
    return np.stack((np.clip(0.5 + (phase - 0.1) / 0.010, 0.0, 1.0) - np.clip(0.5 + (phase - 0.35) / 0.010, 0.0, 1.0),) * 2, axis=1).astype(np.float32)


def pushed(signal, *, rate=RATE_HZ, samples=100, level=0.0, slope='rising', seconds=4.0, cut=50):
    """A stream through an assembler, cut as reads cut it."""

    assembler = TraceAssembler(WaveformTrigger(0, level, slope, 'x', 'V', samples), 1.0 / rate)
    times = np.arange(int(seconds * rate)) / rate
    values = signal(times)
    received = np.arange(times.size, dtype=np.int64)
    parts = [assembler.push(values[at:at + cut], times[at:at + cut], received[at:at + cut]) for at in range(0, times.size, cut)]
    return assembler, np.concatenate([part.samples for part in parts]), np.concatenate([part.times for part in parts]), sum(part.edges for part in parts)


class Traces(unittest.TestCase):
    def test_a_trace_is_centred_on_its_crossing_and_passes_the_level_there(self):
        for level, slope in ((0.0, 'rising'), (0.3, 'rising'), (-0.2, 'falling')):
            assembler, traces, times, _ = pushed(sine(23.7, 0.02), samples=100, level=level, slope=slope)
            self.assertEqual(assembler.before, 50)
            self.assertEqual(traces.shape[1:], (100, 2))
            self.assertGreater(traces.shape[0], 80)
            np.testing.assert_allclose(traces[:, 50, 0], level, atol=1e-5)
            # Before the crossing it is on one side, after it on the other.
            sign = 1.0 if slope == 'rising' else -1.0
            self.assertLess(np.median(sign * traces[:, 45, 0]), sign * level)
            self.assertGreater(np.median(sign * traces[:, 55, 0]), sign * level)
            periods = np.diff(times) * 23.7
            self.assertLess(np.abs(periods[np.abs(periods - 1.0) < 0.5] - 1.0).max(), 0.05)

    def test_the_sample_phase_does_not_move_a_trace(self):
        # 400 Hz over a 10 ms edge: four samples on it.
        _, traces, times, _ = pushed(square, rate=400.0, samples=40, level=0.5, seconds=30.0, cut=7)
        self.assertGreater(traces.shape[0], 50)
        # Every point of the edge, in every trace, sits where the exact edge is.
        for index, expected in ((19, 0.5 - 2.5 / 10), (20, 0.5), (21, 0.5 + 2.5 / 10)):
            np.testing.assert_allclose(traces[:, index, 0], expected, atol=1e-5)
        # What aligning on the first sample past the level would have left: a quarter of the swing.
        first_past = np.ceil(times * 400.0 - 1e-9) / 400.0 - times
        self.assertGreater(np.ptp(first_past) * 400.0, 0.9)

    def test_how_reads_cut_the_stream_changes_nothing(self):
        whole = pushed(sine(23.7, 0.03), samples=64, cut=4000)
        for cut in (1, 7, 50, 333):
            part = pushed(sine(23.7, 0.03), samples=64, cut=cut)
            np.testing.assert_array_equal(part[1], whole[1])
            np.testing.assert_array_equal(part[2], whole[2])

    def test_every_edge_gives_a_trace_and_traces_overlap(self):
        _, traces, times, edges = pushed(sine(23.7), samples=500, seconds=4.0)
        # A period is about 42 samples; a trace is about twelve periods long.
        self.assertGreater(traces.shape[0], 80)
        self.assertLessEqual(edges - traces.shape[0], 13, 'only the ends lack samples')
        self.assertTrue(np.all(np.diff(times) < 0.06))

    def test_an_edge_without_enough_samples_ahead_gives_no_trace(self):
        def late(times):
            out = np.full((times.size, 2), -1.0, dtype=np.float32)
            out[times >= 0.03, 0] = 1.0
            return out
        _, traces, _, edges = pushed(late, samples=100, seconds=1.0)
        self.assertEqual((edges, traces.shape[0]), (1, 0), 'its crossing is 30 samples in, a trace asks for 50 ahead')

    def test_it_holds_a_bounded_number_of_samples(self):
        assembler, *_ = pushed(lambda times: np.ones((times.size, 2), dtype=np.float32), samples=64, seconds=10.0, level=5.0)
        self.assertLessEqual(assembler._held.shape[0], 65)

    def test_a_sample_that_could_not_be_read_stays_invalid_in_its_trace(self):
        def holes(times, inner=sine(5.0)):
            out = inner(times)
            out[::37] = np.nan
            return out
        _, traces, _, _ = pushed(holes, samples=60, seconds=4.0)
        self.assertGreater(traces.shape[0], 5)
        self.assertTrue(np.isnan(traces).any())
        self.assertTrue(np.isfinite(traces[:, 30, 0]).all(), 'a crossing lies between two good samples')


class Publication(unittest.TestCase):
    """Each trace is one event, the shape of a record of a source triggered in hardware."""

    def test_each_edge_is_one_event_from_any_source_layout(self):
        for samples, outputs in ((1, (VOLTAGE,)), (25, (VOLTAGE,)), (1, N100_OUTPUTS), (25, N100_OUTPUTS)):
            with self.subTest(samples=samples, outputs=len(outputs)):
                columns = 1 + max(column for output in outputs for column in output.columns)

                def signal(times, inner=sine(23.7, 0.03)):
                    return np.tile(inner(times)[:, :1], (1, columns)).astype(np.float32)

                channel = f'{outputs[0].name}/{outputs[0].channel_labels[0]}'
                node, commits, _ = monitored(signal, 1.5, trigger=TriggerSettings(channel, 0.0, 'rising', 40),
                                             samples=samples, outputs=outputs)
                self.assertEqual(commits.ordinals, list(range(len(commits.ordinals))))
                self.assertGreater(len(commits.ordinals), 20)
                np.testing.assert_allclose(commits.traces(outputs[0].name)[:, 20], 0.0, atol=1e-5)
                self.assertTrue(np.all(np.diff(commits.shot_times) > 0))
                self.assertLess(commits.commits, 40, 'one commit a read')
                timing = commits.records[3]['record_timing']['wf']['3']
                self.assertAlmostEqual(timing['record_time_seconds'], commits.shot_times[3])

    def test_the_time_axis_is_zero_at_the_crossing_in_a_continuous_and_a_finite_run(self):
        for repeat in (0, 5):
            device = source_of(sine(23.7), samples=1)
            node = node_of(device, repeat=repeat, trigger=TriggerSettings('voltage/CH1', 0.0, 'rising', 40))
            (event, complete) = node._event_schemas(40, 2)['voltage']
            device.close()
            (channel, record) = event.cell_domain.axes
            self.assertEqual((channel.name, record.name, record.size, record.unit), ('channel', 'record time', 40, 's'))
            np.testing.assert_allclose(record.coordinates[:21:20], (-0.02, 0.0))
            self.assertEqual(event.point_domain.axes, ())
            if repeat:
                self.assertEqual(complete.repeat_domain.size, 5)
                self.assertEqual(complete.cell_domain, event.cell_domain)
            else:
                self.assertIsNone(complete)

    def test_trigger_off_keeps_the_records_whole(self):
        for samples in (1, 25):
            node, commits, terminal = monitored(sine(23.7, 0.03), 0.8, samples=samples)
            self.assertEqual(commits.ordinals, list(range(terminal.produced_count)))
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

    def test_repeat_is_one_trace_for_each_of_that_many_edges(self):
        for samples in (1, 25):
            trigger = TriggerSettings('voltage/CH1', 0.0, 'rising', 200)
            kept, commits, capture = self.collect(sine(5.0, 0.0, phase=np.pi / 2), 5, trigger=trigger, samples=samples)
            self.assertEqual(kept, 5)
            self.assertEqual(commits.ordinals, list(range(5)))
            self.assertEqual(commits.origins, [(index, 0) for index in range(5)])
            self.assertEqual(commits.canonical.repeat_domain.size, 5)
            traces = commits.traces('voltage')
            self.assertEqual(traces.shape, (5, 200))
            np.testing.assert_allclose(traces[:, 100], 0.0, atol=1e-5)
            # The same wave at the same phase: one trace lies on the next.
            self.assertLess(np.abs(traces - traces[0]).max(), 1e-3)
            self.assertTrue(capture.terminal.source_stopped)
            crossings = [commits.records[index]['record_timing']['wf'][str(index)]['source_time_seconds'] for index in range(5)]
            np.testing.assert_allclose(np.diff(crossings), 0.2, atol=1e-3)

    def test_stopped_before_any_edge_keeps_nothing(self):
        stop = threading.Event()
        threading.Timer(0.3, stop.set).start()
        kept, commits, capture = self.collect(lambda times: np.ones((times.size, 2), dtype=np.float32), 100,
                                              trigger=TriggerSettings('voltage/CH1', 5.0, 'rising', 20), stop=stop.is_set)
        self.assertEqual((kept, commits.commits), (0, 0))
        self.assertTrue(capture.terminal.no_more_records)

    def test_stopped_part_way_keeps_the_traces_it_has(self):
        stop = threading.Event()
        threading.Timer(0.8, stop.set).start()
        kept, commits, _ = self.collect(sine(5.0), 100, trigger=TriggerSettings('voltage/CH1', 0.0, 'rising', 40), stop=stop.is_set)
        self.assertGreater(kept, 0)
        self.assertLess(kept, 100)
        self.assertEqual(commits.ordinals, list(range(kept)))

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
        node = node_of(device, trigger=TriggerSettings('field/z', 0.0, 'rising', 20))
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
        self.assertEqual(commits.commits, 0, 'no edge, no trace')

    def test_the_first_edge_ends_the_wait(self):
        def signal(times):
            out = np.full((times.size, 2), -1.0, dtype=np.float32)
            out[times > 1.4, 0] = 1.0
            return out
        device = source_of(signal)
        node = node_of(device, trigger=TriggerSettings('voltage/CH1', 0.0, 'rising', 20))
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

        for trigger in (None, TriggerSettings('voltage/CH1', 0.0, 'rising', 20)):
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
        node = WaveformMeasurementNode(sampler=device, request=WaveformMeasurementRequest('s', 0, TriggerSettings('voltage/CH1', 0.0, 'rising', 20)), producer='wf')
        with self.assertRaisesRegex(ValueError, 'hardware trigger'):
            node._configure()
        device.close()


class Form(unittest.TestCase):
    def source(self):
        return source_of(sine(5.0), outputs=N100_OUTPUTS)

    def test_the_trigger_has_four_parameters_on_the_form(self):
        rows = {row['name']: row for row in map(field_json, WAVEFORM_MEASUREMENT_SCHEMA.fields)}
        self.assertEqual(list(rows), ['repeat', 'trigger_slope', 'trigger_channel', 'trigger_level', 'trigger_samples'])
        self.assertEqual(rows['trigger_slope']['label'], 'Slope')
        self.assertEqual(rows['trigger_slope']['default'], 'off')
        self.assertEqual(rows['trigger_channel']['group'], 'Trigger')
        self.assertEqual(rows['trigger_samples']['group'], 'Trigger')

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

    def test_the_preview_draws_the_traces_of_the_trigger_channel_one_on_the_other(self):
        source = self.source()
        free = waveform_preview(source)
        self.assertEqual((free.output.name, free.plot_kind, dict(free.semantic)), ('magnetic_field', 'rolling', {}))
        triggered = waveform_preview(source, trigger_channel='angular_rate/z', node='n1')
        self.assertEqual((triggered.output.name, triggered.plot_kind), ('angular_rate', 'curve'))
        self.assertEqual(dict(triggered.semantic), {'fate:cell_data:n1.angular_rate.1.readout-event': 'x',
                                                    'fate:cell_data:n1.angular_rate.0.component': ['scope-value', 2],
                                                    'fate:repeat:n1.angular_rate.repeat': 'group'})
        draft = waveform_preview(source, trigger_channel='', node='n1')
        self.assertEqual(draft.output.name, 'magnetic_field')
        self.assertEqual(draft.semantic['fate:cell_data:n1.magnetic_field.1.readout-event'], 'x')
        block = source_of(sine(5.0), samples=50)
        self.assertEqual(waveform_preview(block).plot_kind, 'curve')
        self.assertEqual(waveform_preview(block, trigger_channel='voltage/CH2', node='n').plot_kind, 'curve')

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
