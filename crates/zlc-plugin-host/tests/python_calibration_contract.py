"""A calibration's thresholds hold for the acquisition it was trained on: frames the camera took at
another exposure or readout are refused, naming both, and Occupancy says so in the journal."""
import pathlib
import sys
import types
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'science' / 'src'))
from zlc_plugin import host
from zlc_atom.nodes.calibration.calibration import FrameContract
from zlc_atom.nodes.occupancy.processor import CalibrationReading, OccupancyProcessor

READOUT = 'dcam:readout=1;sensor=1;global_exposure=2'
#: What the lab's calibration recorded: the qCMOS's readback of a 20 ms request.
TRAINED = FrameContract((10, 10), sensor_shape=(2304, 4096), exposure_seconds=0.02003, readout_mode=READOUT)


class Core:
    def __init__(self):
        self.lines = []

    def event(self, op, params, *, plain=False):
        if op == 'journal':
            self.lines.extend(params['lines'])


class AcquisitionContract(unittest.TestCase):
    def test_the_same_acquisition_passes(self):
        TRAINED.refuse_other_acquisition(0.02003, READOUT)

    def test_another_exposure_is_refused_naming_both(self):
        with self.assertRaises(ValueError) as refused:
            TRAINED.refuse_other_acquisition(0.005, READOUT, name='calibration.json')
        self.assertIn('exposure 5 ms and calibration.json was trained at 20.03 ms', str(refused.exception))

    def test_equal_is_equal_as_recorded(self):
        # The contract keeps no exposure resolution, so no tolerance is invented for it.
        with self.assertRaisesRegex(ValueError, 'exposure 20.0300000001 ms'):
            TRAINED.refuse_other_acquisition(0.0200300000001, READOUT)

    def test_another_readout_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'readout mode dcam:readout=2;sensor=1;global_exposure=2 and the calibration'):
            TRAINED.refuse_other_acquisition(0.02003, 'dcam:readout=2;sensor=1;global_exposure=2')

    def test_a_fact_either_side_does_not_record_is_not_compared(self):
        TRAINED.refuse_other_acquisition(None, None)
        FrameContract((10, 10)).refuse_other_acquisition(0.005, READOUT)


class OccupancyExposure(unittest.TestCase):
    """The lab's case: frames at 5 ms read with a calibration trained at 20.03 ms."""

    def setUp(self):
        self.core = Core()
        host._PROCESS = self.core
        self.addCleanup(setattr, host, '_PROCESS', None)
        # The processor's own placement path, over a calibration that is only its frame contract.
        self.processor = object.__new__(OccupancyProcessor)
        calibration = types.SimpleNamespace(frame_contract=TRAINED, report={})
        self.processor.readings = (CalibrationReading(calibration, ('rest',)),)
        self.processor._names = ('readout.json',)

    @staticmethod
    def frames(exposure_seconds):
        schema = types.SimpleNamespace(value_schema=types.SimpleNamespace(value_unit='count'),
                                       cell_domain=types.SimpleNamespace(axes=()))
        camera = {'exposure_seconds': exposure_seconds, 'readout_mode': READOUT, 'sensor_shape_yx': [2304, 4096]}
        return types.SimpleNamespace(name='camera/frames', snapshot=types.SimpleNamespace(block=types.SimpleNamespace(schema=schema)),
                                     run_record={'device_snapshots': {'camera': camera}})

    def test_frames_at_another_exposure_are_refused_and_journaled(self):
        with self.assertRaises(ValueError) as refused:
            self.processor.evaluate(self.frames(0.005))
        self.assertIn('exposure 5 ms and readout.json was trained at 20.03 ms', str(refused.exception))
        (line,) = self.core.lines
        self.assertEqual((line['src'], line['ev'], line['facts']['signal']), ('occupancy', 'frames_refused', 'camera/frames'))
        self.assertEqual(line['facts']['reason'], str(refused.exception))

    def test_frames_at_the_trained_exposure_say_nothing(self):
        self.processor._place_against_the_frames(self.frames(0.02003))
        self.assertEqual(self.core.lines, [])


if __name__ == '__main__':
    unittest.main()
