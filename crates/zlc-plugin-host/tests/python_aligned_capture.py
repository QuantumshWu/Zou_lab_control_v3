"""A camera measurement that names its trigger lane follows what the board plays: nothing is
published until a pulse plays; each FIRE is a play of its own; a frame at no trigger's time is
dropped; a trigger with no frame leaves its shot published invalid; a pulse with another cycle
size is refused rather than published under the old one."""
import pathlib
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "science" / "src"))
from zlc_atom.nodes.camera import measurement
from zlc_atom.nodes.camera.alignment import AlignmentError

RISING = (0.82, 1.091, 1.134)
SHOT = 1.154
#: When the camera was armed, on this computer's clock (us).
ARMED = 1_000_000_000


def frame(ordinal, camera_seconds, host_us=None):
    """A frame the camera stamped at ``camera_seconds`` and this computer received then (or at ``host_us``)."""
    micro = round(camera_seconds * 1e6)
    received = ARMED + 2_000_000 + micro if host_us is None else host_us
    return types.SimpleNamespace(source_ordinal=ordinal, frame_stamp=ordinal, camera_stamp=0, timestamp_seconds=micro // 1_000_000,
                                 timestamp_microseconds=micro % 1_000_000, host_received_at_ns=received * 1000)


def answer(revision, state, command_id=None, sent_us=0, points=((RISING, SHOT),), run_repeats=0):
    return {"revision": revision, "state": state, "lane": "ch11", "command_id": command_id, "run_repeats": run_repeats, "scan_repeats": 1,
            "sent_us": sent_us, "points": [{"rising": list(rising), "duration": duration} for rising, duration in points]}


class Bench:
    """A measurement with its camera and board stood in for: frames are handed in batches, the board answers from a script."""

    def __init__(self, repeat=0):
        self.node = node = object.__new__(measurement.CameraMeasurementNode)
        node._request = types.SimpleNamespace(frames_per_cycle=1, photoelectrons=False, camera_key="qcmos", trigger_channel="ch11", repeat=repeat)
        node._actual_working_point = types.SimpleNamespace(electrons_per_count=None, exposure_seconds=0.005)
        node._record_received = None
        node._next_record_ordinal = 0
        node._capture_group_ends = frozenset()
        node._aligned_frames = None
        self.batches, self.board, self.asked, self.published, self.said = [], answer(0, "none"), [], [], []
        node.camera = types.SimpleNamespace(read_frame_records=lambda count, **options: self.batches.pop(0) if self.batches else ())
        self.capture = measurement.AlignedCapture(node, commit_live=lambda outputs: self.published.append(outputs["frames"]),
                                                  schedule=self.schedule, armed_us=ARMED, say=lambda text: self.said.append(text))

    def schedule(self, known):
        self.asked.append(known)
        return {"revision": self.board["revision"], "unchanged": True} if known == self.board["revision"] else self.board

    def poll(self, *frames, now_us=0):
        """Read these frames; ``now_us`` is this computer's clock at the read (0: no time has passed)."""
        self.batches.append(tuple(frames))
        with mock.patch.object(measurement, "time_ns", return_value=now_us * 1000),              mock.patch.object(measurement, "validate_batch", side_effect=lambda ordinals, **options: {"next": 0, "group_ends": []}), \
             mock.patch.object(measurement, "_monitor_cycle_output",
                               side_effect=lambda node, cycle, revision, present, record: (tuple(r.source_ordinal for r in cycle), present, record["trigger_alignment"])), \
             mock.patch.object(measurement, "_finite_cycle_output",
                               side_effect=lambda node, cycle, index, present, record: (index, tuple(r.source_ordinal for r in cycle), present)):
            self.capture.poll()


class AlignedCapture(unittest.TestCase):
    def test_nothing_is_published_until_a_pulse_plays_and_each_fire_is_its_own_play(self):
        bench = Bench()
        bench.poll(frame(0, 0.5))
        self.assertEqual((bench.published, bench.capture.dropped), ([], 1), "a frame before any pulse belongs to no shot")
        self.assertIn("no pulse was playing", bench.said[-1])
        # LOAD, then FIRE: the first frame after it is the first trigger; a stray between two triggers is dropped.
        bench.board = answer(1, "loaded")
        bench.poll()
        bench.board = answer(2, "playing", command_id=7, sent_us=ARMED + 1_000_000)
        bench.poll(frame(1, 10 + RISING[0]), frame(2, 10 + RISING[1]), frame(3, 10 + 1.11), frame(4, 10 + RISING[2]))
        self.assertEqual(bench.published, [((1, 2, 4), (True, True, True), {"shot": 0, "missing": []})])
        self.assertEqual((bench.capture.dropped, bench.node._aligned_frames), (2, 3))
        self.assertEqual(bench.asked[:3], [None, 0, 1], "the board is asked only for what changed")
        # The second shot lacks its middle frame: published, with that frame marked invalid.
        bench.poll(frame(5, 10 + SHOT + RISING[0]), frame(6, 10 + SHOT + RISING[2]), frame(7, 10 + 2 * SHOT + RISING[0]))
        self.assertEqual(bench.published[1], ((5, 5, 6), (True, False, True), {"shot": 1, "missing": [1]}))
        self.assertEqual(bench.capture.incomplete, 1)
        # Stop and FIRE again: the shot the old play stopped in is no shot; the new play starts at its own first trigger.
        bench.board = answer(4, "playing", command_id=8, sent_us=ARMED + 60_000_000)
        late = ARMED + 61_000_000
        bench.poll(frame(8, 10 + 2 * SHOT + RISING[1], host_us=ARMED + 59_000_000),
                   frame(9, 90 + RISING[0], host_us=late), frame(10, 90 + RISING[1], host_us=late), frame(11, 90 + RISING[2], host_us=late))
        self.assertEqual(len(bench.published), 3)
        self.assertEqual(bench.published[2], ((9, 10, 11), (True, True, True), {"shot": 0, "missing": []}))

    def test_a_shot_whose_last_frames_never_come_is_published_once_its_length_has_passed(self):
        bench = Bench()
        bench.board = answer(1, "playing", command_id=1, sent_us=ARMED + 1)
        first = frame(0, RISING[0])
        received = first.host_received_at_ns // 1000
        bench.poll(first, now_us=received + 1000)
        bench.poll(now_us=received + round(SHOT * 0.9e6))
        self.assertEqual(bench.published, [], "still inside the shot")
        bench.poll(now_us=received + round(SHOT * 1.1e6))
        self.assertEqual(bench.published, [((0, 0, 0), (True, False, False), {"shot": 0, "missing": [1, 2]})])

    def test_a_pulse_with_another_cycle_size_is_refused(self):
        bench = Bench()
        bench.board = answer(1, "playing", command_id=1, sent_us=ARMED + 1)
        bench.poll(frame(0, RISING[0]), frame(1, RISING[1]), frame(2, RISING[2]))
        bench.board = answer(2, "playing", command_id=2, sent_us=ARMED + 9_000_000, points=(((0.1, 0.2), 1.0),))
        with self.assertRaises(AlignmentError) as refusal:
            bench.poll()
        self.assertIn("restart the measurement", str(refusal.exception))

    def test_a_finite_measurement_ends_with_the_shots_it_asked_for(self):
        bench = Bench(repeat=2)
        bench.board = answer(1, "playing", command_id=1, sent_us=ARMED + 1)
        times = [shot * SHOT + edge for shot in range(3) for edge in RISING]
        bench.poll(*(frame(ordinal, time) for ordinal, time in enumerate(times)))
        self.assertEqual(bench.published, [(0, (0, 1, 2), (True, True, True)), (1, (3, 4, 5), (True, True, True))])
        self.assertTrue(bench.capture.finished)

    def test_a_scan_that_changes_a_shots_length_is_followed_point_by_point(self):
        bench = Bench()
        points = (((0.10, 0.15), 1.0), ((0.10, 0.35), 1.2))
        bench.board = answer(1, "playing", command_id=1, sent_us=ARMED + 1, points=points, run_repeats=1)
        starts = [0.0, 1.0]
        times = [start + edge for start, point in zip(starts, points) for edge in point[0]]
        bench.poll(*(frame(ordinal, time) for ordinal, time in enumerate(times)))
        self.assertEqual([published[0] for published in bench.published], [(0, 1), (2, 3)])


if __name__ == "__main__": unittest.main()
