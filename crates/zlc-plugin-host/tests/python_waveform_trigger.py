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
    HYSTERESIS_SHARE, TriggerDetector, TriggerSettings, WaveformTrigger)

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
    def test_no_wrong_edge_over_slow_and_fast_signals_noise_lengths_and_both_slopes(self):
        for frequency in (1.3, 4.7, 11.0, 24.0, 47.0):
            for noise in (0.01, 0.05):
                for samples in (20, 30, 90):
                    for slope in ('rising', 'falling'):
                        with self.subTest(frequency=frequency, noise=noise, samples=samples, slope=slope):
                            _, shots = detect(sine(frequency, noise), 12.0, samples=samples, slope=slope)
                            self.assertGreater(len(shots), 5)
                            self.assertEqual(wrong_edges(shots, frequency, slope), 0)

    def test_the_hysteresis_is_what_keeps_the_slow_wrong_edges_out(self):
        original = HYSTERESIS_SHARE
        import zlc_atom.nodes.waveform_measurement.trigger as module
        module.HYSTERESIS_SHARE = 0.0
        try:
            # No hysteresis and no clamp distance either: any sample below the level arms it.
            _, shots = detect(sine(4.7, 0.05), 12.0)
        finally:
            module.HYSTERESIS_SHARE = original
        self.assertGreater(wrong_edges(shots, 4.7) / len(shots), 0.05, 'a wrong-edge rate the hysteresis has to remove')

    def test_a_record_shorter_than_a_period_catches_every_edge_and_one_period_about_every_second(self):
        _, every = detect(sine(10.0, 0.01), 10.0, samples=50, pre=10)
        rising_edges = 10.0 * 10.0
        self.assertGreater(len(every), 0.95 * rising_edges)
        self.assertTrue(np.allclose(np.diff([shot.time_seconds for shot in every]), 0.1, atol=2e-3))
        _, second = detect(sine(10.0, 0.01), 10.0, samples=100, pre=0)
        gaps = np.diff([shot.time_seconds for shot in second])
        self.assertLess(len(second), 0.75 * rising_edges)
        self.assertGreater(np.median(gaps), 0.19)

    def test_a_level_in_the_outer_quarter_of_the_signal_still_fires(self):
        for level, slope in ((-0.9, 'rising'), (0.9, 'falling')):
            with self.subTest(level=level, slope=slope):
                _, shots = detect(sine(5.0, 0.01), 8.0, level=level, slope=slope)
                self.assertGreaterEqual(len(shots), 25, "after the look-back fills, one per period")
                # All at that crossing, one per period.
                self.assertTrue(np.allclose(np.diff([shot.time_seconds for shot in shots]), 0.2, atol=5e-3))

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
        self.assertGreater(len(whole), 20)
        for block in (1, 3, 10, 64):
            detector = TriggerDetector(WaveformTrigger(0, 0.0, 'rising', 40, 25), DT)
            cut = [shot for at in range(0, len(records), block) for shot in detector.feed(records[at:at + block])]
            # A read's threshold differs by a hair; the crossings found do not.
            self.assertEqual([round(shot.time_seconds, 6) for shot in cut],
                             [round(shot.time_seconds, 6) for shot in whole][:len(cut)])
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

    def test_memory_stays_about_one_record(self):
        detector = TriggerDetector(WaveformTrigger(0, 5.0, 'rising', 30, 6), DT)  # never fires
        for at, record in enumerate(stream(sine(5.0), 5.0, 10)):
            self.assertEqual(detector.feed([record]), [])
            self.assertLessEqual(detector._samples.shape[0], 30)

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

    def test_no_wrong_edge_over_1_to_50_hz_from_records_of_one_sample_or_a_block(self):
        for frequency in (1.0, 4.7, 11.0, 24.0, 50.0):
            for noise in (0.01, 0.05):
                for samples in (20, 40, 80):
                    for slope, block in (('rising', 1), ('falling', 1), ('rising', 16)):
                        with self.subTest(frequency=frequency, noise=noise, samples=samples, slope=slope, block=block):
                            detector = TriggerDetector(field_trigger(slope, samples, samples // 5), SLOW_DT)
                            records = quantities_records(quantities_signal(frequency, noise), 12.0, block)
                            # Read as the capture reads: a slice's worth of records at a time.
                            step = max(1, 20 // block)
                            shots = [shot for at in range(0, len(records), step) for shot in detector.feed(records[at:at + step])]
                            self.assertGreater(len(shots), 5)
                            self.assertEqual(wrong_edges(shots, frequency, slope), 0)
                            self.assertTrue(all(shot.samples.shape == (samples, 10) for shot in shots))

    def test_at_50_hz_the_crossing_time_beats_the_sample_period(self):
        # 8 samples a period, 40-point records: re-armed on a crossing, so the next
        # one often has to wait for the edge after it: every 5 or 6 periods.
        frequency = 50.0
        results = {}
        for noise in (0.01, 0.05):
            detector = TriggerDetector(field_trigger(), SLOW_DT)
            shots = detector.feed(quantities_records(quantities_signal(frequency, noise), 20.0))
            self.assertTrue(8.0 <= len(shots) / 20.0 <= 10.0, len(shots))
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
             quantities_source(quantities_signal(50.0, 0.02)), settings(channel=FIELD_Z, level=30.0, samples=40), 8.0),
            ('1 kHz, 10 samples/record, 1 output, 15-point records, 50 Hz signal',
             virtual(sine(50.0, 0.02), block=10), settings(samples=15), 50.0),
            ('1 kHz, 100 samples/record, 1 output, 15-point records, 50 Hz signal',
             virtual(sine(50.0, 0.02), block=100), settings(samples=15), 50.0),
        ):
            rate, cpu, waiting, reads = self.measure(source, trigger)
            print(f'\n{name}: {rate:5.1f} records/s, {reads:4.1f} reads/s, process CPU {100 * cpu:4.1f}% of one core, '
                  f'at most {waiting} source records waiting')
            self.assertGreater(rate, 0.9 * wanted)


if __name__ == '__main__':
    unittest.main()
