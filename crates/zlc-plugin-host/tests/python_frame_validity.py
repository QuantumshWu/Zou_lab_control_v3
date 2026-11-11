"""A camera cycle with a frame that was never taken is one dataset whose validity says so: the frames
that came are valid, the stand-in for the one that did not is invalid, every pixel of it."""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "science" / "src"))
from zlc_atom.devices.camera.contract import CameraFrameRecord
from zlc_atom.nodes.camera.measurement import frames_snapshot
from zlc_data.value import dataset_validity_storage


def record(ordinal, value):
    image = np.full((4, 6), value, dtype="<u2")
    image.setflags(write=False)
    return CameraFrameRecord(image=image, source_ordinal=ordinal)


class FrameValidity(unittest.TestCase):
    def test_a_missing_frame_is_invalid_and_the_others_stand(self):
        first, last = record(0, 5), record(1, 9)
        snapshot = frames_snapshot(((first, first, last),), producer="camera_measurement", generation="g", revision=1, value_unit="count",
                                   present=((True, False, True),))
        self.assertEqual(snapshot.block.schema.point_domain.shape, (3,))
        valid = np.asarray(dataset_validity_storage(snapshot.block.validity, snapshot.block.schema))
        per_frame = valid.reshape(1, 3, -1)
        self.assertEqual([bool(per_frame[0, frame].all()) for frame in range(3)], [True, False, True])
        self.assertFalse(per_frame[0, 1].any(), "every pixel of the frame that never came")

    def test_a_whole_cycle_of_an_aligned_capture_has_the_same_schema_as_one_with_a_gap(self):
        frames = (record(0, 1), record(1, 2))
        whole = frames_snapshot((frames,), producer="camera_measurement", generation="g", revision=1, value_unit="count", present=((True, True),))
        gap = frames_snapshot((frames,), producer="camera_measurement", generation="g", revision=2, value_unit="count", present=((True, False),))
        self.assertEqual(whole.block.schema, gap.block.schema, "one signal, one schema, whichever shot lacks a frame")
        counted = frames_snapshot((frames,), producer="camera_measurement", generation="g", revision=1, value_unit="count")
        self.assertNotEqual(counted.block.schema, whole.block.schema, "a counted capture states no validity")


if __name__ == "__main__": unittest.main()
