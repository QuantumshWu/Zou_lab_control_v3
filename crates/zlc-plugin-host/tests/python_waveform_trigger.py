"""The software trigger of Waveform Measurement, at the rates it serves.

A virtual source samples at 400 Hz to 1 kHz, with one sample or a block in each
of its records and one or several quantities, and streams signals of 1-50 Hz with 1-5%
noise; the trigger cuts records of tens of points out of its records.  Run
directly with Python.  Set ZLC_TRIGGER_PERFORMANCE=1 to also print what keeping
up costs.
"""
import os
import pathlib
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'science' / 'src'))
from zlc_atom.devices.simulation.waveform.source import VirtualWaveformConfig, VirtualWaveformSource
from zlc_atom.devices.waveform.contract import (
    WaveformAcquisitionMode, WaveformCaptureTerminalRecord, WaveformOutput, WaveformRecord, WaveformWorkingPoint)
from zlc_atom.devices.waveform.wheeltec_n100 import N100_OUTPUTS
from zlc_plugin import capabilities
from zlc_plugin.declare import QUERY_PARAMETERS, field_json
from zlc_plugin.wire import Decoder, Encoder
from zlc_atom.nodes.waveform_measurement import logic_node
from zlc_atom.nodes.waveform_measurement.measurement import (
    WAVEFORM_MEASUREMENT_SCHEMA, FiniteCapture, MonitorCapture, WaveformMeasurementNode, WaveformMeasurementRequest,
    waveform_preview)
from zlc_atom.nodes.waveform_measurement.trigger import (
    TriggerDetector, TriggerSettings, WaveformTrigger)

RATE_HZ = 1000.0
DT = 1.0 / RATE_HZ
VOLTAGE = WaveformOutput('voltage', 'V', ('CH1', 'CH2'), (0, 1))


def sine(frequency, noise=0.0, seed=1, phase=0.0):
    """Two channels: a sine on the first, its negative on the second; noise relative to amplitude."""

    rng = np.random.default_rng(seed)

    def samples(times):
        wave = np.sin(2 * np.pi * frequency * times + phase)
        return np.stack((wave + noise * rng.standard_normal(times.size), -wave), axis=1).astype(np.float32)
    return samples


def stream(signal, seconds, block):
    """A signal as a source's consecutive records of ``block`` samples."""

    count = int(seconds * RATE_HZ) // block
    return [WaveformRecord(signal(index * block * DT + np.arange(block) * DT), index, index * block * DT, index + 1)
            for index in range(count)]


def detect(signal, seconds, *, samples=30, pre=6, slope='rising', level=0.0, block=10):
    detector = TriggerDetector(WaveformTrigger(0, level, slope, samples, pre), DT)
    records = stream(signal, seconds, block)
    return detector, [shot for at in range(0, len(records), 7) for shot in detector.feed(records[at:at + 7])]


def wrong_edges(shots, frequency, slope='rising'):
    """How many crossings lie on the other slope of the clean signal."""

    derivative = np.cos(2 * np.pi * frequency * np.array([shot.time_seconds for shot in shots]))
    return int(np.sum(derivative < 0 if slope == 'rising' else derivative > 0))


def true_edges(frequency, phase, level, slope, end):
    """When the clean signal crosses the level in the chosen direction."""

    angle = np.arcsin(level) if slope == 'rising' else np.pi - np.arcsin(level)
    first = np.floor((-phase - angle) / (2 * np.pi))
    times = (angle + phase * -1 + 2 * np.pi * np.arange(first, first + end * frequency + 3)) / (2 * np.pi * frequency)
    return times[(times > 0.0) & (times < end)]


def counted(frequency, noise, rate, samples, slope, level, seed, block=1):
    """(true edges, records fired for them): every edge from the second period on,
    in a stream read twenty of the source's records at a time."""

    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2 * np.pi)
    seconds = max(6.0, 8.0 / frequency)
    count = int(seconds * rate)
    times = np.arange(count) / rate
    values = (np.sin(2 * np.pi * frequency * times + phase) + noise * rng.standard_normal(count)).astype(np.float32)[:, None]
    records = [WaveformRecord(values[at:at + block], at // block, times[at], 1)
               for at in range(0, count - block + 1, block)]
    detector = TriggerDetector(WaveformTrigger(0, level, slope, samples, samples // 5), 1.0 / rate)
    shots = [shot for at in range(0, len(records), 20) for shot in detector.feed(records[at:at + 20])]
    fired = np.array([shot.time_seconds for shot in shots])
    period = 1.0 / frequency
    edges = true_edges(frequency, phase, level, slope, times[-1])
    # Past the run's start (its cost is measured on its own) and the first pre-trigger share,
    # and whole by the end of the stream.
    edges = edges[(edges > max(period, samples / rate)) & (edges < times[-1] - samples / rate - 2 * block / rate)]
    return edges, fired[(fired > edges[0] - period / 2) & (fired < edges[-1] + period / 2)]


def lengths_around_the_period(frequency, rate):
    """Below, at and just above a period and a multiple of it."""

    period = rate / frequency
    return tuple(max(4, length) for length in (int(0.6 * period), int(period), int(2 * period), int(2 * period) + 1))


def settings(channel='voltage/CH1', level=0.0, slope='rising', samples=30, position=20.0):
    return TriggerSettings(channel, level, slope, samples, position)


def virtual(signal, block=10):
    return VirtualWaveformSource(VirtualWaveformConfig(RATE_HZ, block, (VOLTAGE,)), sample_source=signal)


def armed(source, repeat=0, trigger=None):
    node = WaveformMeasurementNode(sampler=source, request=WaveformMeasurementRequest('s', repeat, trigger), producer='wf')
    node._generation = 'run'
    node._configure()
    node._arm()
    return node


class Detection(unittest.TestCase):
    def assert_every_edge_fires_once(self, edges, fired, frequency):
        self.assertEqual(fired.size, edges.size, 'as many records as true edges')
        self.assertLess(np.max(np.abs(fired - edges)), 0.2 / frequency, 'each at its own edge')

    def test_every_edge_fires_once_at_a_centred_level_whatever_the_record_length(self):
        for rate in (400.0, 1000.0):
            for frequency in (1.0, 4.7, 11.0, 24.0, 50.0):
                for noise in (0.01, 0.05):
                    for samples in lengths_around_the_period(frequency, rate):
                        for slope in ('rising', 'falling'):
                            with self.subTest(rate=rate, frequency=frequency, noise=noise, samples=samples, slope=slope):
                                self.assert_every_edge_fires_once(
                                    *counted(frequency, noise, rate, samples, slope, 0.0, int(rate + samples)), frequency)

    def test_every_edge_fires_once_in_the_outer_parts_of_the_range_at_low_noise(self):
        for rate in (400.0, 1000.0):
            for frequency in (1.0, 4.7, 11.0, 24.0, 50.0):
                for level in (-0.8, 0.8):
                    for slope in ('rising', 'falling'):
                        with self.subTest(rate=rate, frequency=frequency, level=level, slope=slope):
                            samples = lengths_around_the_period(frequency, rate)[1]
                            self.assert_every_edge_fires_once(
                                *counted(frequency, 0.01, rate, samples, slope, level, 7), frequency)

    @unittest.expectedFailure
    def test_the_measured_limit_of_the_arming_rule_5_percent_noise_at_a_level_80_percent_out(self):
        # The level is then about four noise widths from the extreme: slow signals
        # fire extra records and 24-50 Hz ones miss edges.  Stated, not hidden.
        for rate, frequency, level, slope in ((400.0, 1.0, -0.8, 'rising'), (400.0, 50.0, -0.8, 'rising')):
            self.assert_every_edge_fires_once(*counted(frequency, 0.05, rate, 40, slope, level, 3), frequency)

    def test_edges_inside_the_record_before_them_fire_and_each_record_holds_its_own_raw_samples(self):
        records = stream(sine(50.0, 0.02), 6.0, 10)
        raw = np.concatenate([record.samples for record in records])
        shots = TriggerDetector(WaveformTrigger(0, 0.0, 'rising', 50, 10), DT).feed(records)
        gaps = np.diff([shot.time_seconds for shot in shots])
        self.assertTrue(np.allclose(gaps, 0.02, atol=2e-3), 'an edge every period: 2.5 records overlap')
        self.assertGreater(len(shots), 295)
        for shot in shots:
            at = int(round((shot.time_seconds - shot.crossing_offset_seconds) * RATE_HZ))
            np.testing.assert_array_equal(shot.samples, raw[at - 10:at + 40])
        self.assertFalse(np.shares_memory(shots[0].samples, shots[1].samples))

    def test_the_run_s_first_samples_are_compared_with_what_has_been_seen_so_far(self):
        # A level near the signal's edge fires at once: no fill, no wait.
        _, shots = detect(sine(5.0, 0.01, phase=-np.pi / 2), 2.0, level=-0.8, pre=0)
        self.assertEqual(len(shots), 10)
        self.assertLess(shots[0].time_seconds, 0.2)

    def test_the_crossing_time_is_sub_sample_and_the_samples_are_the_raw_ones(self):
        records = stream(sine(5.0), 4.0, 10)
        raw = np.concatenate([record.samples for record in records])
        detector = TriggerDetector(WaveformTrigger(0, 0.1, 'rising', 30, 6), DT)
        shots = detector.feed(records)
        self.assertGreater(len(shots), 10)
        for shot in shots:
            at = int(round((shot.time_seconds - shot.crossing_offset_seconds) * RATE_HZ))
            self.assertTrue(-DT <= shot.crossing_offset_seconds <= 0.0)
            np.testing.assert_array_equal(shot.samples, raw[at - 6:at + 24])
            self.assertLess(raw[at - 1, 0], 0.1)
            self.assertGreaterEqual(raw[at, 0], 0.1)
            # A clean sine: the linear interpolation lands on the true crossing.
            phase = np.arcsin(0.1) / (2 * np.pi * 5.0)
            self.assertLess(abs(((shot.time_seconds - phase + 0.1) % 0.2) - 0.1), 2e-5)

    def test_pre_trigger_and_records_span_blocks_of_any_cut(self):
        records = stream(sine(7.0, 0.02), 6.0, 1)
        whole = TriggerDetector(WaveformTrigger(0, 0.0, 'rising', 40, 25), DT).feed(records)
        self.assertGreater(len(whole), 35)
        for block in (1, 3, 10, 64):
            detector = TriggerDetector(WaveformTrigger(0, 0.0, 'rising', 40, 25), DT)
            cut = [shot for at in range(0, len(records), block) for shot in detector.feed(records[at:at + block])]
            self.assertEqual([shot.time_seconds for shot in cut], [shot.time_seconds for shot in whole])
            for shot, reference in zip(cut, whole):
                np.testing.assert_array_equal(shot.samples, reference.samples)

    def test_an_ordinal_gap_is_never_stitched_and_the_broken_record_is_counted(self):
        records = stream(sine(5.0), 4.0, 10)
        detector = TriggerDetector(WaveformTrigger(0, 0.0, 'rising', 60, 10), DT)
        # Find a record boundary in the middle of a triggered record, then lose the next ten records.
        probe = TriggerDetector(WaveformTrigger(0, 0.0, 'rising', 60, 10), DT)
        shots = probe.feed(records)
        trigger_sample = int(round(shots[3].time_seconds * RATE_HZ))
        cut = trigger_sample // 10 + 2
        kept = records[:cut] + records[cut + 10:]
        got = detector.feed(kept[:cut]) + detector.feed(kept[cut:])
        self.assertEqual(detector.broken_records, 1)
        self.assertEqual([shot.source_ordinal for shot in got], list(range(len(got))))
        raw = {}
        for record in kept:
            for offset in range(10):
                raw[record.source_ordinal * 10 + offset] = record.samples[offset]
        for shot in got:
            at = int(round((shot.time_seconds - shot.crossing_offset_seconds) * RATE_HZ))
            for index in range(at - 10, at + 50):
                self.assertIn(index, raw, 'a record holds only samples that follow one another')

    def test_memory_stays_about_one_record_however_many_are_being_filled(self):
        for level in (5.0, 0.0):  # never fires; fires with several records being filled
            detector = TriggerDetector(WaveformTrigger(0, level, 'rising', 300, 60), DT)
            for record in stream(sine(50.0), 3.0, 10):
                detector.feed([record])
                self.assertLessEqual(detector._samples.shape[0], 300)

    def test_missing_samples_fire_nothing_and_do_not_break_the_stream(self):
        def holes(times):
            values = sine(5.0)(times)
            values[np.floor(times * 1000) % 7 == 0, 0] = np.nan
            return values
        _, shots = detect(holes, 8.0)
        self.assertGreater(len(shots), 25)
        self.assertEqual(wrong_edges(shots, 5.0), 0)
        self.assertTrue(any(np.isnan(shot.samples).any() for shot in shots), 'never filled in: NaN stays NaN')


class Settings(unittest.TestCase):
    def test_the_form_offers_the_trigger_off_by_default_with_the_sources_channels(self):
        values = WAVEFORM_MEASUREMENT_SCHEMA.project_values({})
        self.assertEqual(values['trigger_slope'], 'off')
        self.assertEqual(WAVEFORM_MEASUREMENT_SCHEMA.project_values({'repeat': 3})['repeat'], 3)
        with self.assertRaisesRegex(ValueError, 'Channel'):
            WAVEFORM_MEASUREMENT_SCHEMA.project_values({'trigger_slope': 'rising'})
        source = virtual(sine(5.0))
        answer = logic_node.LOGIC_NODE.resolve_field_choices({}, {}, {'sampler': source})
        self.assertEqual(answer['trigger_channel'], [['voltage/CH1', 'voltage CH1 (V)'], ['voltage/CH2', 'voltage CH2 (V)']])
        self.assertEqual(logic_node.LOGIC_NODE.resolve_field_availability({'sampler': source}), {})
        self.assertEqual(logic_node.LOGIC_NODE.resolve_field_choices({}, {}, {}), {})
        built = logic_node.LOGIC_NODE.build(sampler=source, sampler_key='s', trigger_slope='falling',
                                            trigger_channel='voltage/CH2', trigger_level=0.5,
                                            trigger_record_samples=40, trigger_position=25.0)
        self.assertEqual(built.request.trigger, TriggerSettings('voltage/CH2', 0.5, 'falling', 40, 25.0))

    def test_a_draft_query_reaches_what_the_form_needs_through_the_plugin_interface(self):
        source = virtual(sine(5.0))
        self.assertIn('working_point', capabilities.reads(source), 'a draft query may call it')
        point = Decoder(None)(Encoder(None)(source.working_point()))
        self.assertEqual(point.acquisition_mode, WaveformAcquisitionMode.FREE_RUNNING.value)
        node = logic_node.LOGIC_NODE
        self.assertEqual(QUERY_PARAMETERS['resolve_field_choices'], ('values', 'resources', 'devices'))
        # A node just added: no devices yet, answered for the catalogue.
        answers = node.first_answers([field_json(item) for item in WAVEFORM_MEASUREMENT_SCHEMA.fields])
        self.assertEqual(answers['resolve_field_choices'], {})
        self.assertEqual(answers['resolve_field_availability'], {})
        self.assertEqual(answers['declare_previews'], [])

    def test_a_host_timed_channel_says_its_precision_is_limited(self):
        class HostTimed:
            outputs = (VOLTAGE,)

            def working_point(self):
                return WaveformWorkingPoint(WaveformAcquisitionMode.FREE_RUNNING, DT, {}, time_basis='host_receive')
        (first, *_) = logic_node.LOGIC_NODE.resolve_field_choices({}, {}, {'sampler': HostTimed()})['trigger_channel']
        self.assertIn('limited precision', first[1])

    def test_a_source_with_its_own_trigger_does_not_offer_it_and_refuses_it(self):
        source = Scope()
        unavailable = logic_node.LOGIC_NODE.resolve_field_availability({'sampler': source})
        self.assertEqual(set(unavailable), {'trigger_slope', 'trigger_channel', 'trigger_level',
                                            'trigger_record_samples', 'trigger_position'})
        node = WaveformMeasurementNode(sampler=source, request=WaveformMeasurementRequest('s', 1, settings()), producer='wf')
        with self.assertRaisesRegex(ValueError, 'own hardware trigger'):
            node._configure()

    def test_a_channel_the_source_lacks_and_a_record_too_long_are_refused_before_arming(self):
        with self.assertRaisesRegex(ValueError, 'no channel'):
            settings(channel='voltage/CH9').resolve((VOLTAGE,))
        with self.assertRaisesRegex(ValueError, 'too long'):
            settings(samples=600_000).resolve((VOLTAGE,))
        with self.assertRaisesRegex(ValueError, 'at least 2'):
            settings(samples=1)

    def test_the_preview_follows_the_trigger(self):
        one_sample = VirtualWaveformSource(VirtualWaveformConfig(RATE_HZ, 1, (
            WaveformOutput('field', 'uT', ('X',), (0,)), VOLTAGE_LATER)), sample_source=sine(5.0))
        self.assertEqual(waveform_preview(one_sample).plot_kind, 'rolling')
        (triggered,) = logic_node.LOGIC_NODE.declare_previews(
            {'trigger_slope': 'rising', 'trigger_channel': 'voltage/CH1'}, {'sampler': one_sample})
        self.assertEqual((triggered.plot_kind, triggered.output.name), ('curve', 'voltage'))
        (off,) = logic_node.LOGIC_NODE.declare_previews({'trigger_slope': 'off'}, {'sampler': one_sample})
        self.assertEqual((off.plot_kind, off.output.name), ('rolling', 'field'))


VOLTAGE_LATER = WaveformOutput('voltage', 'V', ('CH1',), (1,))


class Scope:
    """A source with its own trigger, which has not fired; a Stop must still end its wait."""

    timeout = 0.05
    outputs = (VOLTAGE,)
    record_samples = 20

    def __init__(self):
        self.armed = None

    def working_point(self):
        return WaveformWorkingPoint(WaveformAcquisitionMode.EXTERNAL_TRIGGERED, DT, {}, time_basis='host_receive')

    def arm(self, records, *, buffer_record_count):
        self.armed = records

    def read_records(self, n, *, timeout, exact):
        time.sleep(timeout)
        return []

    def finish_record_capture(self):
        return WaveformCaptureTerminalRecord(0, True, True, True)

    def capture_state(self):
        return self.armed is not None


class Captures(unittest.TestCase):
    def test_a_finite_capture_of_overlapping_records_stops_at_exactly_n(self):
        source = virtual(sine(50.0, 0.02))
        node = armed(source, 12, settings(samples=70, position=20.0))
        committed = []
        capture = FiniteCapture(node, should_stop=None)
        try:
            self.assertEqual(capture.collect(commit_shot=lambda record, index: committed.append(record)), 12)
        finally:
            source.close()
        self.assertEqual([record.source_ordinal for record in committed], list(range(12)))
        self.assertTrue(np.allclose(np.diff([record.time_seconds for record in committed]), 0.02, atol=2e-3))

    def test_a_finite_capture_gets_n_records_in_order_with_the_trigger_at_time_zero(self):
        source = virtual(sine(20.0, 0.02))
        node = armed(source, 12, settings(samples=30, position=20.0))
        committed = []

        def commit(record, index):
            committed.append((index, node._shot_outputs(record, revision=index + 1, index=index)))

        capture = FiniteCapture(node, should_stop=None)
        try:
            self.assertEqual(capture.collect(commit_shot=commit), 12)
        finally:
            source.close()
        self.assertEqual([index for index, _ in committed], list(range(12)))
        self.assertTrue(capture.terminal.no_more_records)
        times = []
        for index, outputs in committed:
            output = outputs['voltage']
            self.assertEqual(output.cell_origin, (index, 0))
            schema = output.snapshot.block.schema
            (channel, sample) = schema.cell_domain.axes
            self.assertEqual((channel.name, sample.name, sample.unit), ('channel', 'record time', 's'))
            self.assertEqual(sample.size, 30)
            self.assertAlmostEqual(sample.coordinates[0], -6 * DT)
            self.assertEqual(sample.coordinates[6], 0.0)
            self.assertAlmostEqual(sample.coordinates[-1], 23 * DT)
            timing = output.event_record['record_timing']['wf'][str(index)]
            self.assertTrue(-DT <= timing['crossing_offset_seconds'] <= 0.0)
            self.assertEqual(timing['time_basis'], 'sample_clock')
            times.append(timing['source_time_seconds'])
            self.assertEqual(output.canonical_schema.repeat_domain.size, 12)
        self.assertTrue(np.all(np.diff(times) > 0))
        self.assertEqual(wrong_edges([type('S', (), {'time_seconds': t}) for t in times], 20.0), 0)
        self.assertEqual(node.run_record['parameters']['trigger']['pre_trigger_samples'], 6)
        self.assertEqual(node.run_record['parameters']['trigger']['channel'], 'voltage/CH1')

    def test_a_monitor_publishes_every_triggered_record_in_order(self):
        source = virtual(sine(20.0, 0.02))
        node = armed(source, 0, settings(samples=30))
        committed = []
        capture = MonitorCapture(node, commit_live=committed.append)
        stop = time.monotonic() + 2.0
        try:
            while time.monotonic() < stop:
                capture.poll()
            terminal = capture.close()
        finally:
            source.close()
        ordinals = [int(next(iter(outputs['voltage'].event_record['record_timing']['wf']))) for outputs in committed]
        self.assertEqual(ordinals, list(range(len(ordinals))))
        times = [outputs['voltage'].event_record['record_timing']['wf'][str(ordinal)]['source_time_seconds']
                 for ordinal, outputs in zip(ordinals, committed)]
        self.assertGreater(len(ordinals), 30, 'about 20/s for 2 s, every edge, nothing skipped')
        self.assertTrue(np.allclose(np.diff(times), 0.05, atol=2e-3), 'no trigger missing')
        self.assertTrue(terminal.no_more_records)
        (channel, sample) = committed[0]['voltage'].snapshot.block.schema.cell_domain.axes
        self.assertEqual(sample.name, 'record time')

    def test_stop_while_waiting_ends_promptly_and_a_quiet_trigger_is_no_fault(self):
        source = virtual(sine(5.0, 0.01))
        # Longer than the source's silence limit, with a level the signal never reaches.
        node = armed(source, 5, settings(level=9.0))
        status = []
        node._report_status = lambda message, **kwargs: status.append((message, kwargs))
        stop = threading.Event()
        capture = FiniteCapture(node, should_stop=stop.is_set)
        threading.Timer(3.0, stop.set).start()
        started = time.monotonic()
        try:
            self.assertEqual(capture.collect(commit_shot=lambda record, index: self.fail('nothing triggered')), 0)
        finally:
            source.close()
        self.assertLess(time.monotonic() - started, 3.5, 'a Stop is honoured within a slice')
        self.assertEqual(status[0], ('Waiting for trigger', {'current': 0, 'total': 5}))

    def test_a_dead_source_is_a_fault_under_the_trigger_too(self):
        alive = threading.Event()
        alive.set()
        signal = sine(5.0)

        def dying(times):
            while not alive.is_set():
                time.sleep(0.01)
            if times[0] > 0.3:
                alive.clear()
            return signal(times)
        source = virtual(dying)
        node = armed(source, 5, settings(level=9.0))
        capture = FiniteCapture(node, should_stop=None)
        started = time.monotonic()
        try:
            with self.assertRaisesRegex(TimeoutError, 'stopped delivering'):
                capture.collect(commit_shot=lambda record, index: None)
        finally:
            alive.set()
            source.close()
        self.assertLess(time.monotonic() - started, 6.0, 'the silence limit, then the join of the stuck producer')

    def test_a_finite_scope_waits_for_its_own_trigger_past_the_silence_limit_and_stops(self):
        source = Scope()
        node = armed(source, 3)
        status = []
        node._report_status = lambda message, **kwargs: status.append(message)
        self.assertLess(node.silence_seconds, 0.2)
        stop = threading.Event()
        capture = FiniteCapture(node, should_stop=stop.is_set)
        threading.Timer(1.5, stop.set).start()
        started = time.monotonic()
        self.assertEqual(capture.collect(commit_shot=lambda record, index: self.fail('never triggered')), 0)
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertEqual(status, ['Waiting for trigger'])
        self.assertEqual(source.armed, 3)

    def test_a_stop_keeps_what_the_stopped_source_still_held(self):
        source = virtual(sine(20.0, 0.02))
        node = armed(source, 1000, settings(samples=30))
        committed = []
        stop = threading.Event()

        def commit(record, index):
            committed.append((index, record.source_ordinal))
            if index == 5:
                stop.set()
                time.sleep(0.3)  # triggers pile up in what the source holds

        capture = FiniteCapture(node, should_stop=stop.is_set)
        try:
            kept = capture.collect(commit_shot=commit)
        finally:
            source.close()
        self.assertGreater(kept, 6)
        self.assertEqual(committed, [(index, index) for index in range(kept)])

    def test_with_the_trigger_off_nothing_changes(self):
        source = virtual(sine(5.0))
        node = armed(source, 4)
        self.assertEqual(node.run_record['parameters'], {'repeat': 4})
        committed = []

        def commit(record, index):
            committed.append(node._shot_outputs(record, revision=index + 1, index=index))

        capture = FiniteCapture(node, should_stop=None)
        try:
            self.assertEqual(capture.collect(commit_shot=commit), 4)
        finally:
            source.close()
        self.assertEqual(capture.terminal.produced_count, 4, 'the source was armed for exactly the records')
        schema = committed[0]['voltage'].snapshot.block.schema
        self.assertEqual(schema.point_domain.axes[0].name, 'sample time')
        self.assertEqual(schema.point_domain.axes[0].coordinates[0], 0.0)
        timing = committed[0]['voltage'].event_record['record_timing']['wf']['0']
        self.assertNotIn('crossing_offset_seconds', timing)
        self.assertEqual(waveform_preview(virtual(sine(5.0))).plot_kind, 'curve')


SLOW_RATE_HZ = 400.0
SLOW_DT = 1.0 / SLOW_RATE_HZ
FIELD_Z = 'magnetic_field/z'
FIELD_Z_COLUMN = 2


def quantities_signal(frequency, noise=0.0, seed=1, amplitude=5.0, centre=30.0):
    """Ten columns: the field's z axis swings in uT, every other column is its own."""

    rng = np.random.default_rng(seed)

    def samples(times):
        out = np.empty((times.size, 10), dtype=np.float32)
        wave = np.sin(2 * np.pi * frequency * times)
        out[:, FIELD_Z_COLUMN] = centre + amplitude * wave + amplitude * noise * rng.standard_normal(times.size)
        for column in range(10):
            if column != FIELD_Z_COLUMN:
                out[:, column] = 100.0 * column + times  # says which column and which sample it is
        return out
    return samples


def quantities_source(signal, block=1):
    return VirtualWaveformSource(VirtualWaveformConfig(SLOW_RATE_HZ, block, N100_OUTPUTS), sample_source=signal)


def quantities_records(signal, seconds, block=1):
    count = int(seconds * SLOW_RATE_HZ) // block
    return [WaveformRecord(signal(index * block * SLOW_DT + np.arange(block) * SLOW_DT), index, index * block * SLOW_DT, index + 1)
            for index in range(count)]


def field_trigger(slope='rising', samples=40, pre=8, level=30.0):
    return WaveformTrigger(FIELD_Z_COLUMN, level, slope, samples, pre)


class SeveralQuantities(unittest.TestCase):
    """Several outputs of different units in every record, at 400 Hz, in records of any length."""

    def test_every_edge_fires_once_from_records_of_one_sample_or_a_block(self):
        for block in (1, 16):
            for frequency in (1.0, 11.0, 50.0):
                for samples in (20, 40, 80):
                    for slope in ('rising', 'falling'):
                        with self.subTest(block=block, frequency=frequency, samples=samples, slope=slope):
                            edges, fired = counted(frequency, 0.05, SLOW_RATE_HZ, samples, slope, 0.0, 11, block=block)
                            self.assertEqual(fired.size, edges.size)

    def test_at_50_hz_the_crossing_time_beats_the_sample_period(self):
        # 8 samples a period, 40-point records, five periods long: every edge still fires.
        frequency = 50.0
        results = {}
        for noise in (0.01, 0.05):
            detector = TriggerDetector(field_trigger(), SLOW_DT)
            shots = detector.feed(quantities_records(quantities_signal(frequency, noise), 20.0))
            # 1000 rising edges, less the one before the 8 pre-trigger samples and the 4 in the last 32.
            self.assertAlmostEqual(len(shots), 995, delta=1)
            self.assertEqual(wrong_edges(shots, frequency), 0)
            crossing = np.array([shot.time_seconds for shot in shots])
            sample = crossing - np.array([shot.crossing_offset_seconds for shot in shots])
            error = lambda times: np.std((times * frequency + 0.5) % 1.0 - 0.5) / frequency
            results[noise] = (error(crossing), error(sample))
        print(f'\n50 Hz at 400 Hz sampling, rms time of the trigger: '
              + ', '.join(f'{100 * noise:.0f}% noise {1e3 * got:.3f} ms interpolated, {1e3 * raw:.3f} ms at the sample'
                          for noise, (got, raw) in results.items()))
        for got, raw in results.values():
            self.assertLess(got, 0.3e-3)
            self.assertGreater(raw, 0.5e-3)

    def test_every_quantity_of_a_record_is_in_one_event_aligned_to_the_same_trigger(self):
        source = quantities_source(quantities_signal(20.0, 0.02))
        trigger = settings(channel=FIELD_Z, level=30.0, samples=40, position=20.0)
        node = WaveformMeasurementNode(sampler=source, request=WaveformMeasurementRequest('s', 6, trigger), producer='wf')
        events = []

        class Context:
            instance_id, generation = 'wf', 'run'
            cancel_requested = staticmethod(lambda: False)
            record_devices = staticmethod(lambda: None)
            set_run_record = staticmethod(lambda record: None)
            report_ready = staticmethod(lambda: None)
            report_progress = staticmethod(lambda message, **kwargs: None)
            commit_live = staticmethod(events.append)

        try:
            self.assertEqual(node.execute(Context)['shots'], 6)
        finally:
            source.close()
        self.assertEqual(node.run_record['parameters']['trigger']['column'], FIELD_Z_COLUMN)
        self.assertEqual(len(events), 6, 'one event per trigger')
        expected = {'magnetic_field': ('uT', 3, (0, 1, 2)), 'angular_rate': ('rad*s^-1', 3, (3, 4, 5)),
                    'acceleration': ('m*s^-2', 3, (6, 7, 8)), 'temperature': ('K', 1, (9,))}
        for index, outputs in enumerate(events):
            self.assertEqual(set(outputs), set(expected))
            timings = {name: output.event_record['record_timing']['wf'][str(index)] for name, output in outputs.items()}
            self.assertEqual(len({timing['source_time_seconds'] for timing in timings.values()}), 1)
            for name, (unit, channels, columns) in expected.items():
                schema = outputs[name].snapshot.block.schema
                (channel, sample) = schema.cell_domain.axes
                self.assertEqual((schema.value_schema.value_unit, channel.size, sample.size), (unit, channels, 40))
                self.assertAlmostEqual(sample.coordinates[0], -8 * SLOW_DT)
                self.assertEqual(sample.coordinates[8], 0.0)
                self.assertEqual(outputs[name].cell_origin, (index, 0))
                values = np.asarray(outputs[name].snapshot.block.as_segment()[0])[0, 0]
                self.assertEqual(values.shape, (channels, 40))
                # Each published channel is its own source column, at the same samples.
                for row, column in enumerate(columns):
                    if column == FIELD_Z_COLUMN:
                        self.assertLess(values[row, 7], 30.5)
                        self.assertGreaterEqual(values[row, 8], 29.5)
                    else:
                        at = timings[name]['source_time_seconds'] - timings[name]['crossing_offset_seconds']
                        np.testing.assert_allclose(values[row], 100.0 * column + at + (np.arange(40) - 8) * SLOW_DT,
                                                   atol=3e-2 if column > 2 else 1e-3)

    def test_the_form_offers_every_channel_of_every_quantity_with_its_unit(self):
        class Several:
            outputs = N100_OUTPUTS

            def working_point(self):
                return WaveformWorkingPoint(WaveformAcquisitionMode.FREE_RUNNING, SLOW_DT, {}, time_basis='device_clock')
        choices = logic_node.LOGIC_NODE.resolve_field_choices({}, {}, {'sampler': Several()})['trigger_channel']
        self.assertEqual(len(choices), 10)
        self.assertIn([FIELD_Z, 'magnetic_field z (uT)'], choices)
        self.assertIn(['temperature/imu', 'temperature imu (K)'], choices)
        (preview,) = logic_node.LOGIC_NODE.declare_previews(
            {'trigger_slope': 'rising', 'trigger_channel': 'acceleration/x'},
            {'sampler': type('S', (), {'outputs': N100_OUTPUTS, 'record_samples': 1})()})
        self.assertEqual((preview.plot_kind, preview.output.name), ('curve', 'acceleration'))


def round_trip():
    """About 0.7 ms through the core, spun: a Windows sleep this short takes up to a timer tick."""
    deadline = time.perf_counter() + 0.0007
    while time.perf_counter() < deadline:
        pass


@unittest.skipUnless(os.environ.get('ZLC_TRIGGER_PERFORMANCE'), 'ZLC_TRIGGER_PERFORMANCE is not set')
class Performance(unittest.TestCase):
    def measure(self, source, trigger):
        # Each source read and each commit charged one round trip through the core (about 0.7 ms).
        node = armed(source, 0, trigger)
        reads = [0]
        read = source.read_records

        def charged(*args, **kwargs):
            reads[0] += 1
            round_trip()
            return read(*args, **kwargs)

        source.read_records = charged
        committed = []

        def commit(outputs):
            committed.append(len(outputs))
            round_trip()

        capture = MonitorCapture(node, commit_live=commit)
        wall, cpu = time.perf_counter(), time.process_time()
        stop = time.monotonic() + 10.0
        waiting = 0
        try:
            while time.monotonic() < stop:
                capture.poll()
                waiting = max(waiting, source._records.pending_count)
            cpu = time.process_time() - cpu
            wall = time.perf_counter() - wall
            capture.close()
        finally:
            source.close()
        return len(committed) / wall, cpu / wall, waiting, reads[0] / wall

    def test_it_keeps_up(self):
        for name, source, trigger, wanted in (
            ('400 Hz, 1 sample/record, 4 outputs, 40-point records, 50 Hz signal',
             quantities_source(quantities_signal(50.0, 0.02)), settings(channel=FIELD_Z, level=30.0, samples=40), 50.0),
            ('1 kHz, 10 samples/record, 1 output, 40-point records, 50 Hz signal',
             virtual(sine(50.0, 0.02), block=10), settings(samples=40), 50.0),
            ('1 kHz, 100 samples/record, 1 output, 40-point records, 50 Hz signal',
             virtual(sine(50.0, 0.02), block=100), settings(samples=40), 50.0),
        ):
            rate, cpu, waiting, reads = self.measure(source, trigger)
            print(f'\n{name}: {rate:5.1f} records/s, {reads:4.1f} reads/s, process CPU {100 * cpu:4.1f}% of one core, '
                  f'at most {waiting} source records waiting')
            self.assertGreater(rate, 0.95 * wanted)


if __name__ == '__main__':
    unittest.main()
