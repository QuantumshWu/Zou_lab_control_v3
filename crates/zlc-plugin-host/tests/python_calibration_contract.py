"""A calibration's thresholds hold while the camera collects the light they were trained on and
counts it the same way: an exposure shorter than the trained light window, or another readout
speed / sensor mode / pixel format, is refused naming both; an exposure that covers the window,
or another trigger line, is not. Occupancy says a refusal in the journal."""
import pathlib
import sys
import types
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'science' / 'src'))
from zlc_plugin import host
from zlc_atom.devices.camera.contract import counting_readout
from zlc_atom.nodes.calibration.calibration import FrameContract, refuse_other_acquisition, trained_light_window_seconds
from zlc_atom.nodes.occupancy.processor import CalibrationReading, OccupancyProcessor

READOUT = 'dcam:readout=1;sensor=1;global_exposure=2'


def calibration(*, camera_exposure=0.02003, windows=(0.02, 0.005, 0.02), readout_mode=READOUT):
    """The lab's calibration: camera exposure 20.03 ms (the qCMOS's readback of 20 ms), reference
    windows 20 ms, readout window 5 ms, as the calibration task records its pulse facts."""
    report = {} if windows is None else {'run_record': {'pulse': {
        'frame_exposures': list(windows), 'reference_frame_indices': [0, 2], 'readout_frame_index': 1}}}
    contract = FrameContract((10, 10), sensor_shape=(2304, 4096), exposure_seconds=camera_exposure, readout_mode=readout_mode)
    return types.SimpleNamespace(frame_contract=contract, report=report)


class Core:
    def __init__(self):
        self.lines = []

    def event(self, op, params, *, plain=False):
        if op == 'journal':
            self.lines.extend(params['lines'])


class AcquisitionContract(unittest.TestCase):
    def test_the_light_window_is_the_readout_frames(self):
        self.assertEqual(trained_light_window_seconds(calibration()), 0.005)
        self.assertIsNone(trained_light_window_seconds(calibration(windows=None)))

    def test_an_exposure_covering_the_window_passes_shorter_or_longer_than_the_calibrations(self):
        for exposure in (0.005, 0.01, 0.02003, 0.05):
            refuse_other_acquisition(calibration(), exposure, READOUT)

    def test_an_exposure_shorter_than_the_window_is_refused_naming_both(self):
        with self.assertRaises(ValueError) as refused:
            refuse_other_acquisition(calibration(), 0.003, READOUT, name='readout.json')
        self.assertIn('exposed 3 ms, shorter than the 5 ms light window readout.json was trained on', str(refused.exception))

    def test_a_readback_within_one_camera_step_of_the_window_is_not_a_cut(self):
        # 4.9998 ms read back for a 5 ms request, on a camera that sets its exposure in 1 us steps.
        refuse_other_acquisition(calibration(), 0.0049998, READOUT, exposure_step_seconds=1e-6)
        with self.assertRaisesRegex(ValueError, 'exposed 4.9998 ms, shorter than the 5 ms light window'):
            refuse_other_acquisition(calibration(), 0.0049998, READOUT)  # no step reported: as recorded
        with self.assertRaisesRegex(ValueError, 'exposed 3 ms, shorter than the 5 ms light window'):
            refuse_other_acquisition(calibration(), 0.003, READOUT, exposure_step_seconds=1e-6)

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

    def test_a_calibration_recording_no_window_compares_no_exposure(self):
        refuse_other_acquisition(calibration(windows=None), 0.001, READOUT)

    def test_only_what_changes_a_count_is_compared_of_the_readout(self):
        self.assertEqual(counting_readout(READOUT), {'readout': '1', 'sensor': '1'})
        self.assertEqual(counting_readout('pylon:format=Mono8;trigger=Line1;grab=OneByOne'), {'format': 'Mono8'})
        self.assertEqual(counting_readout('virtual:trigger=external'), {})
        self.assertIsNone(counting_readout(None))
        # How the exposure is triggered is not how a count is read.
        refuse_other_acquisition(calibration(), 0.02, 'dcam:readout=1;sensor=1;global_exposure=5')
        with self.assertRaisesRegex(ValueError, r'read out at readout=2 \(trained at readout=1\)'):
            refuse_other_acquisition(calibration(), 0.02, 'dcam:readout=2;sensor=1;global_exposure=2')
        with self.assertRaisesRegex(ValueError, r'sensor=18 \(trained at sensor=1\)'):
            refuse_other_acquisition(calibration(), 0.02, 'dcam:readout=1;sensor=18;global_exposure=2')

    def test_a_basler_on_another_trigger_line_is_not_refused(self):
        basler = calibration(readout_mode='pylon:format=Mono8;trigger=Line1;grab=OneByOne')
        refuse_other_acquisition(basler, 0.02, 'pylon:format=Mono8;trigger=Line3;grab=OneByOne')
        refuse_other_acquisition(basler, 0.02, 'pylon:format=Mono8;trigger=free-running;grab=OneByOne')

    def test_calibrations_of_different_camera_exposures_share_one_run_that_covers_both_windows(self):
        load = calibration(camera_exposure=0.05, windows=(0.05, 0.03, 0.05))
        readout = calibration()
        for each in (load, readout):
            refuse_other_acquisition(each, 0.03, READOUT)


class OccupancyExposure(unittest.TestCase):
    def setUp(self):
        self.core = Core()
        host._PROCESS = self.core
        self.addCleanup(setattr, host, '_PROCESS', None)
        # The processor's own placement path, over a calibration that is only its contract and record.
        self.processor = object.__new__(OccupancyProcessor)
        self.processor.readings = (CalibrationReading(calibration(), ('rest',)),)
        self.processor._names = ('readout.json',)

    @staticmethod
    def frames(exposure_seconds, step=None):
        schema = types.SimpleNamespace(value_schema=types.SimpleNamespace(value_unit='count'),
                                       cell_domain=types.SimpleNamespace(axes=()))
        camera = {'exposure_seconds': exposure_seconds, 'exposure_step_seconds': step, 'readout_mode': READOUT,
                  'sensor_shape_yx': [2304, 4096]}
        return types.SimpleNamespace(name='camera/frames', snapshot=types.SimpleNamespace(block=types.SimpleNamespace(schema=schema)),
                                     run_record={'device_snapshots': {'camera': camera}})

    def test_the_labs_run_reads_and_says_nothing(self):
        # Measured at 5 ms against a calibration whose camera exposed 20.03 ms around a 5 ms window.
        self.processor._place_against_the_frames(self.frames(0.005))
        # A readback a hair under the window, on a camera that states its step.
        self.processor._place_against_the_frames(self.frames(0.0049998, step=1e-6))
        self.assertEqual(self.core.lines, [])

    def test_an_exposure_cutting_the_window_is_refused_and_journaled(self):
        with self.assertRaises(ValueError) as refused:
            self.processor.evaluate(self.frames(0.003))
        self.assertIn('exposed 3 ms, shorter than the 5 ms light window readout.json was trained on', str(refused.exception))
        (line,) = self.core.lines
        self.assertEqual((line['src'], line['ev'], line['facts']['signal']), ('occupancy', 'frames_refused', 'camera/frames'))
        self.assertEqual(line['facts']['reason'], str(refused.exception))


if __name__ == '__main__':
    unittest.main()
