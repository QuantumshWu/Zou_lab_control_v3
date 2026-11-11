"""What a plugin says for the application's flight journal: one stamped line per fact, sent to the
core as a `journal` event; never kept outside a plugin process, and never a reason to fail."""
import pathlib
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "science" / "src"))
from zlc_plugin import host, journal
from zlc_atom.devices.simulation.camera.adapter import VirtualCamera


class Core:
    """The plugin's channel to the core, keeping what was said on it."""

    def __init__(self, broken=False):
        self.events, self.broken = [], broken

    def event(self, op, params, *, plain=False):
        if self.broken:
            raise OSError("the channel is closing")
        self.events.append((op, params, plain))

    def lines(self):
        return [line for op, params, _ in self.events if op == "journal" for line in params["lines"]]


class PluginJournal(unittest.TestCase):
    def setUp(self):
        self.core = Core()
        host._PROCESS = self.core
        self.addCleanup(setattr, host, "_PROCESS", None)

    def test_a_note_is_one_stamped_line_of_plain_facts(self):
        before = time.time_ns() // 1000
        journal.note("scan", "segment_fired", command_id=7, outer_row=[1.5])
        ((op, params, plain),) = self.core.events
        (line,) = params["lines"]
        self.assertEqual((op, plain, line["src"], line["ev"], line["facts"]), ("journal", True, "scan", "segment_fired", {"command_id": 7, "outer_row": [1.5]}))
        self.assertTrue(before <= line["us"] <= time.time_ns() // 1000)

    def test_outside_a_plugin_process_and_on_a_broken_channel_nothing_is_kept_and_nothing_fails(self):
        host._PROCESS = None
        journal.note("scan", "segment_fired", command_id=7)
        host._PROCESS = Core(broken=True)
        journal.note("scan", "segment_fired", command_id=7)
        self.assertEqual(self.core.events, [])

    def test_a_camera_says_each_capture_it_begins_and_ends(self):
        camera = VirtualCamera(frame_source=lambda _time, roi: np.zeros((roi[3], roi[2]), dtype="<u2"))
        camera.finish_record_capture()  # nothing armed: no capture ended, nothing to say
        camera.arm(4, source_group_sizes=(2, 2), buffer_frame_count=8, timeout=1.0)
        camera.finish_record_capture()
        said = [(line["ev"], line["facts"]) for line in self.core.lines()]
        self.assertEqual([event for event, _ in said], ["arm", "finish"])
        self.assertEqual(said[0][1], {"camera": "VirtualCamera", "frames": 4, "cycle_size": 2, "cycles": 2, "rested_seconds": 0.0})
        self.assertEqual(said[1][1]["camera"], "VirtualCamera")
        self.assertIn("produced", said[1][1])


if __name__ == "__main__": unittest.main()
