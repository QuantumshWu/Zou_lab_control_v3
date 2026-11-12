"""A calibration's thresholds were trained on one light window and one readout: frames exposed
shorter than that window (by a camera step or more), or read out at another readout speed /
sensor mode / pixel format, are read anyway and said once a run -- on the node and in the
journal, naming both values; an exposure that covers the window, or another trigger line, is
not said."""
import pathlib
import sys
import types
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'science' / 'src'))
from zlc_plugin import host
from zlc_atom.devices.camera.contract import counting_readout
from zlc_atom.nodes.calibration.calibration import FrameContract, acquisition_cautions, trained_light_window_seconds
from zlc_atom.nodes.occupancy.processor import CalibrationReading, OccupancyProcessor

READOUT = 'dcam:readout=1;sensor=1;global_exposure=2'


def calibration(*, camera_exposure=0.02003, windows=(0.02, 0.005, 0.02), readout_mode=READOUT):
    """The lab's calibration: camera exposure 20.03 ms (the qCMOS's readback of 20 ms), reference
    windows 20 ms, readout window 5 ms, as the calibration task records its pulse facts."""
    report = {} if windows is None else {'run_record': {'pulse': {
        'frame_exposures': list(windows), 'reference_frame_indices': [0, 2], 'readout_frame_index': 1}}}
    contract = FrameContract((10, 10), sensor_shape=(2304, 4096), exposure_seconds=camera_exposure, readout_mode=readout_mode)
    return types.SimpleNamespace(frame_contract=contract, report=report)


def kinds(*args, **kwargs):
    return [caution.kind for caution in acquisition_cautions(*args, **kwargs)]


class Core:
    def __init__(self):
        self.lines = []

    def event(self, op, params, *, plain=False):
        if op == 'journal':
            self.lines.extend(params['lines'])


class Context:
    """A run's context as far as a caution reaches it: the node's progress line."""

    def __init__(self):
        self.progress = []

    def report_progress(self, message, *, current=None, total=None):
        self.progress.append(message)


class AcquisitionCautions(unittest.TestCase):
    def test_the_light_window_is_the_readout_frames(self):
        self.assertEqual(trained_light_window_seconds(calibration()), 0.005)
        self.assertIsNone(trained_light_window_seconds(calibration(windows=None)))

    def test_an_exposure_covering_the_window_says_nothing_shorter_or_longer_than_the_calibrations(self):
        for exposure in (0.005, 0.01, 0.02003, 0.05):
            self.assertEqual(kinds(calibration(), exposure, READOUT), [])

    def test_an_exposure_cutting_the_window_is_said_with_both_values_and_what_follows(self):
        (caution,) = acquisition_cautions(calibration(), 0.003, READOUT, name='readout.json')
        self.assertEqual(caution.kind, 'exposure_short')
        self.assertIn('exposure 3 ms is shorter than the 5 ms light window readout.json was trained on', caution.message)
        self.assertIn('expect 5 ms of light and these frames received at most 3 ms', caution.message)
        self.assertEqual((caution.facts['exposure'], caution.facts['light_window'], caution.facts['calibration']),
                         (0.003, 0.005, 'readout.json'))

    def test_a_readback_within_one_camera_step_of_the_window_is_not_a_cut(self):
        # 4.9998 ms read back for a 5 ms request, on a camera that sets its exposure in 1 us steps.
        self.assertEqual(kinds(calibration(), 0.0049998, READOUT, exposure_step_seconds=1e-6), [])
        self.assertEqual(kinds(calibration(), 0.0049998, READOUT), ['exposure_short'])  # no step reported
        self.assertEqual(kinds(calibration(), 0.003, READOUT, exposure_step_seconds=1e-6), ['exposure_short'])

    def test_a_calibration_recording_no_window_compares_no_exposure(self):
        self.assertEqual(kinds(calibration(windows=None), 0.001, READOUT), [])

    def test_only_what_changes_a_count_is_compared_of_the_readout(self):
        self.assertEqual(counting_readout(READOUT), {'readout': '1', 'sensor': '1'})
        self.assertEqual(counting_readout('pylon:format=Mono8;trigger=Line1;grab=OneByOne'), {'format': 'Mono8'})
        self.assertEqual(counting_readout('virtual:trigger=external'), {})
        self.assertIsNone(counting_readout(None))
        # How the exposure is triggered is not how a count is read.
        self.assertEqual(kinds(calibration(), 0.02, 'dcam:readout=1;sensor=1;global_exposure=5'), [])
        (speed,) = acquisition_cautions(calibration(), 0.02, 'dcam:readout=2;sensor=1;global_exposure=2')
        self.assertEqual(speed.kind, 'readout_differs')
        self.assertIn('read out at readout=2 (trained at readout=1)', speed.message)
        (mode,) = acquisition_cautions(calibration(), 0.02, 'dcam:readout=1;sensor=18;global_exposure=2')
        self.assertIn('sensor=18 (trained at sensor=1)', mode.message)

    def test_a_basler_on_another_trigger_line_says_nothing(self):
        basler = calibration(readout_mode='pylon:format=Mono8;trigger=Line1;grab=OneByOne')
        self.assertEqual(kinds(basler, 0.02, 'pylon:format=Mono8;trigger=Line3;grab=OneByOne'), [])
        self.assertEqual(kinds(basler, 0.02, 'pylon:format=Mono8;trigger=free-running;grab=OneByOne'), [])

    def test_calibrations_of_different_camera_exposures_share_one_run_that_covers_both_windows(self):
        load = calibration(camera_exposure=0.05, windows=(0.05, 0.03, 0.05))
        for each in (load, calibration()):
            self.assertEqual(kinds(each, 0.03, READOUT), [])

    def test_a_camera_reports_its_exposure_step_when_the_exposure_is_set(self):
        from zlc_atom.devices.camera.pylon import PylonCameraAdapter, PylonCameraConfig
        from python_pylon_lifecycle import Camera

        class Exposure:
            def __init__(self, increment):
                self.increment, self.reads = increment, 0

            def SetValue(self, value):
                self.value = value

            def HasInc(self):
                return self.increment is not None

            def GetInc(self):
                self.reads += 1
                return self.increment

        adapter = PylonCameraAdapter(PylonCameraConfig('123'), camera=Camera())
        adapter._camera.ExposureTime = Exposure(1.0)  # microseconds, as Basler states it
        adapter._apply_exposure(0.005)
        self.assertEqual((adapter._exposure_step_seconds, adapter._camera.ExposureTime.reads), (1e-6, 1))
        adapter._camera.ExposureTime = Exposure(None)
        adapter._apply_exposure(0.005)
        self.assertIsNone(adapter._exposure_step_seconds)
        adapter.close()


class OccupancyExposure(unittest.TestCase):
    """Occupancy's own evaluate path; the classification itself is stood in for (the calibration
    here is only its frame contract and record), so what is shown is that the frames go on to it."""

    def setUp(self):
        self.core = Core()
        host._PROCESS = self.core
        self.addCleanup(setattr, host, '_PROCESS', None)
        self.processor = object.__new__(OccupancyProcessor)
        self.processor.readings = (CalibrationReading(calibration(), ('rest',)),)
        self.processor._names = ('readout.json',)
        self.read = []
        self.processor.process = lambda snapshot: self.read.append(snapshot) or 'classified'
        self.processor._live_outputs = lambda result, *, source: {'counts': result}
        self.context = Context()
        self.processor.bind_run(self.context)

    @staticmethod
    def frames(exposure_seconds, step=None):
        schema = types.SimpleNamespace(value_schema=types.SimpleNamespace(value_unit='count'),
                                       cell_domain=types.SimpleNamespace(axes=()))
        camera = {'exposure_seconds': exposure_seconds, 'exposure_step_seconds': step, 'readout_mode': READOUT,
                  'sensor_shape_yx': [2304, 4096]}
        return types.SimpleNamespace(name='camera/frames', snapshot=types.SimpleNamespace(block=types.SimpleNamespace(schema=schema)),
                                     run_record={'device_snapshots': {'camera': camera}})

    def test_the_labs_run_reads_and_says_nothing(self):
        # Measured at 5 ms against a calibration whose camera exposed 20.03 ms around a 5 ms window,
        # and a readback a hair under the window on a camera that states its step.
        for frames in (self.frames(0.005), self.frames(0.0049998, step=1e-6)):
            self.assertEqual(self.processor.evaluate(frames), {'counts': 'classified'})
        self.assertEqual((self.core.lines, self.context.progress), ([], []))

    def test_an_exposure_cutting_the_window_is_read_and_said_once_a_run(self):
        for _ in range(3):
            self.assertEqual(self.processor.evaluate(self.frames(0.003)), {'counts': 'classified'})
        self.assertEqual(len(self.read), 3, 'every delivery is classified')
        (line,) = self.core.lines
        self.assertEqual((line['src'], line['ev']), ('occupancy', 'exposure_short'))
        self.assertEqual({key: line['facts'][key] for key in ('signal', 'exposure', 'light_window', 'calibration')},
                         {'signal': 'camera/frames', 'exposure': 0.003, 'light_window': 0.005, 'calibration': 'readout.json'})
        (said,) = self.context.progress
        self.assertIn('expect 5 ms of light and these frames received at most 3 ms', said)
        # The next run says it again, once.
        self.processor.bind_run(Context())
        self.processor.evaluate(self.frames(0.003))
        self.assertEqual(len(self.core.lines), 2)


if __name__ == '__main__':
    unittest.main()
