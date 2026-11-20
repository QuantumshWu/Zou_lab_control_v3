"""A camera rests between the end of one capture and the start of the next, so a frame the
stopped capture still had on its way is never counted as the next capture's first."""
import pathlib
import sys
import time
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "science" / "src"))
import zlc_plugin.journal  # as in a plugin process, where the SDK is loaded before any device: its import is not a capture's time
from zlc_atom.devices.camera.contract import CAPTURE_REST_SECONDS
from zlc_atom.devices.simulation.camera.adapter import VirtualCamera


def camera():
    return VirtualCamera(frame_source=lambda _time, roi: np.zeros((roi[3], roi[2]), dtype="<u2"))


def arm(device):
    """Arm once: (when the arm returned, how long it rested -- as its own journal line says)."""

    said = []
    with mock.patch.object(zlc_plugin.journal, "note", lambda src, ev, **facts: said.append((src, ev, facts))):
        device.arm(1, source_group_sizes=(1,), buffer_frame_count=4, timeout=1.0)
    (rested,) = [facts["rested_seconds"] for src, ev, facts in said if (src, ev) == ("camera", "arm")]
    return time.monotonic(), rested


class CaptureRest(unittest.TestCase):
    def test_a_capture_begins_only_after_the_rest_since_the_last_one_ended(self):
        device = camera()
        _, rested = arm(device)
        self.assertEqual(rested, 0.0, "the first capture waits for nothing")
        device.finish_record_capture()
        # The rest runs from when the capture came to its end (the camera's own mark), not from when
        # finish returned; the clock's tick on Windows (~15.6 ms) is the only slack.
        ended = device._capture_ended_at
        armed, rested = arm(device)
        self.assertGreater(rested, 0.0)
        self.assertGreaterEqual(armed - ended, CAPTURE_REST_SECONDS - 0.02)
        device.finish_record_capture()

    def test_a_rest_already_over_and_a_finish_of_nothing_cost_no_time(self):
        device = camera()
        arm(device)
        device.finish_record_capture()
        time.sleep(CAPTURE_REST_SECONDS)
        device.finish_record_capture()  # a cleanup's second finish: no capture ended here
        _, rested = arm(device)
        self.assertEqual(rested, 0.0)
        device.finish_record_capture()


if __name__ == "__main__": unittest.main()
