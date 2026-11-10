"""A camera rests between the end of one capture and the start of the next, so a frame the
stopped capture still had on its way is never counted as the next capture's first."""
import pathlib
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "science" / "src"))
from zlc_atom.devices.camera.contract import CAPTURE_REST_SECONDS
from zlc_atom.devices.simulation.camera.adapter import VirtualCamera


def camera():
    return VirtualCamera(frame_source=lambda _time, roi: np.zeros((roi[3], roi[2]), dtype="<u2"))


def arm(device):
    began = time.monotonic()
    device.arm(1, source_group_sizes=(1,), buffer_frame_count=4, timeout=1.0)
    return began, time.monotonic()


class CaptureRest(unittest.TestCase):
    def test_a_capture_begins_only_after_the_rest_since_the_last_one_ended(self):
        device = camera()
        began, armed = arm(device)
        self.assertLess(armed - began, CAPTURE_REST_SECONDS / 2, "the first capture waits for nothing")
        device.finish_record_capture()
        ended = time.monotonic()
        _, armed = arm(device)
        self.assertGreaterEqual(armed - ended, CAPTURE_REST_SECONDS - 0.02)
        device.finish_record_capture()

    def test_a_rest_already_over_and_a_finish_of_nothing_cost_no_time(self):
        device = camera()
        arm(device)
        device.finish_record_capture()
        time.sleep(CAPTURE_REST_SECONDS)
        device.finish_record_capture()  # a cleanup's second finish: no capture ended here
        began, armed = arm(device)
        self.assertLess(armed - began, CAPTURE_REST_SECONDS / 2)
        device.finish_record_capture()


if __name__ == "__main__": unittest.main()
