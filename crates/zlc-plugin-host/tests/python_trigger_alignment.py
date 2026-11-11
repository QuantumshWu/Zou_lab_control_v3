"""Frames are sorted into shots by when they arrive, against when the pulse triggers: a frame at no
trigger's time is dropped and the shot it fell in is whole, a trigger with no frame leaves its
shot marked, and every later shot stays in step. Checked on the frame stamps the lab recorded when
extra triggers reached the camera, and on plays whose shots change length."""
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "science" / "src"))
from zlc_atom.nodes.camera.alignment import AlignmentError, Extra, Shot, TriggerAlignment, TriggerSchedule

RECORDED = json.loads((pathlib.Path(__file__).with_name("recorded_frame_stamps.json")).read_text(encoding="utf-8"))
#: The recorded pulse as its triggers were measured: three exposures after an 820 ms load, then the tail.
RISING = (0.82, 0.82 + 0.2710287, 0.82 + 0.2710287 + 0.0430904)
SHOT = 1.1541278
#: The recorded camera's exposure: no two of its frames are nearer than this.
EXPOSURE = 0.005


def schedule(points=((RISING, SHOT),), run_repeats=0, scan_repeats=1, state="playing"):
    return TriggerSchedule(revision=1, state=state, lane="ch11", points=tuple(points), run_repeats=run_repeats, scan_repeats=scan_repeats)


def sort(alignment, times, start=0):
    """(shots, the positions of the frames found extra): frames are named by their position in ``times``."""
    shots, extras = [], []
    for position, time in enumerate(times, start):
        for found in alignment.accept(time, position):
            (shots if isinstance(found, Shot) else extras).append(found)
    return shots, [extra.frame for extra in extras]


class RecordedSessions(unittest.TestCase):
    def test_the_four_extra_triggers_of_the_first_session_are_dropped_and_no_shot_is_lost(self):
        times = [stamp / 1e6 for stamp in RECORDED["first"]]
        shots, extras = sort(TriggerAlignment(schedule(), from_first_trigger=True, separation=EXPOSURE), times)
        self.assertEqual(sorted(extras), [521, 523, 527, 528])
        self.assertEqual([shot.index for shot in shots], list(range(186)), "186 shots, each once, in order")
        self.assertTrue(all(shot.complete for shot in shots))
        real = [position for position in range(len(times)) if position not in extras]
        self.assertEqual([frame for shot in shots for frame in shot.frames], real, "every real frame, in its own shot and place")
        # The shots around the extras are the ones a count would have torn apart.
        self.assertEqual([shots[173].frames, shots[174].frames, shots[175].frames], [(519, 520, 522), (524, 525, 526), (529, 530, 531)])

    def test_the_extra_trigger_of_the_second_session_is_found_by_a_camera_started_mid_play(self):
        times = [stamp / 1e6 for stamp in RECORDED["second"]]
        shots, extras = sort(TriggerAlignment(schedule(), from_first_trigger=False, separation=EXPOSURE), times)
        self.assertEqual(extras, [210])
        self.assertEqual(len(shots), 84)
        self.assertTrue(all(shot.complete for shot in shots))
        self.assertEqual(shots[0].frames, (0, 1, 2), "its first frames' spacing says they open a shot")
        self.assertEqual([shot.index for shot in shots], list(range(shots[0].index, shots[0].index + 84)))

    def test_a_camera_started_in_the_middle_of_a_shot_begins_with_the_next_whole_one(self):
        times = [stamp / 1e6 for stamp in RECORDED["second"]][1:]
        shots, extras = sort(TriggerAlignment(schedule(), from_first_trigger=False, separation=EXPOSURE), times, start=1)
        self.assertEqual(extras, [210])
        self.assertEqual(shots[0].frames, (3, 4, 5), "the two frames of the shot it began inside are no shot")
        self.assertTrue(all(shot.complete for shot in shots))
        self.assertEqual(len(shots), 83)


class Plays(unittest.TestCase):
    def test_a_trigger_with_no_frame_marks_its_shot_and_the_next_is_in_step(self):
        times = [shot * SHOT + edge for shot in range(4) for edge in RISING]
        for missing, expected in ((4, (3, None, 5)), (3, (None, 4, 5)), (5, (3, 4, None))):
            kept = [(position, time) for position, time in enumerate(times) if position != missing]
            alignment = TriggerAlignment(schedule(), from_first_trigger=True)
            found = [item for position, time in kept for item in alignment.accept(time, position)]
            self.assertEqual([(shot.index, shot.frames) for shot in found],
                             [(0, (0, 1, 2)), (1, expected), (2, (6, 7, 8)), (3, (9, 10, 11))], f"frame {missing} missing")

    def test_shots_that_change_length_from_point_to_point_are_followed(self):
        # A scan of a wait: two points, two shots each, one sweep; the camera's clock runs 50 ppm fast.
        points = (((0.10, 0.15), 1.0), ((0.10, 0.35), 1.2))
        play = schedule(points, run_repeats=2, scan_repeats=1)
        self.assertEqual((play.frames, play.cyclic), (2, False))
        self.assertAlmostEqual(play.tolerance, 0.025)
        starts = [0.0, 1.0, 2.0, 3.2]
        true = [start + edge for start, point in zip(starts, (0, 0, 1, 1)) for edge in points[point][0]]
        times = [time * 1.00005 for time in true]
        times.insert(5, 2.2 * 1.00005)   # an extra between the two triggers of the third shot
        times.append(4.9)                # and one after the play's last trigger
        alignment = TriggerAlignment(play, from_first_trigger=True)
        shots, extras = sort(alignment, times)
        self.assertEqual([(shot.index, shot.frames) for shot in shots], [(0, (0, 1)), (1, (2, 3)), (2, (4, 6)), (3, (7, 8))])
        self.assertEqual(extras, [5, 9])
        self.assertTrue(alignment.ended)
        with self.assertRaises(AlignmentError):
            TriggerAlignment(play, from_first_trigger=False)

    def test_evenly_spaced_triggers_cannot_place_a_camera_started_mid_play(self):
        even = schedule((((0.1, 0.6), 1.0),))
        alignment = TriggerAlignment(even, from_first_trigger=False)
        with self.assertRaises(AlignmentError) as refusal:
            sort(alignment, [0.1 + 0.5 * step for step in range(8)])
        self.assertIn("start the camera measurement before the pulse", str(refusal.exception))
        # Armed before the play there is nothing to guess.
        shots, extras = sort(TriggerAlignment(even, from_first_trigger=True), [0.1 + 0.5 * step for step in range(8)])
        self.assertEqual(([shot.frames for shot in shots], extras), ([(0, 1), (2, 3), (4, 5), (6, 7)], []))

    def test_a_stray_first_frame_of_a_camera_started_mid_play_is_set_aside(self):
        times = [3.3] + [shot * SHOT + edge for shot in range(3, 6) for edge in RISING]
        shots, extras = sort(TriggerAlignment(schedule(), from_first_trigger=False, separation=EXPOSURE), times)
        self.assertEqual((extras, [shot.frames for shot in shots]), ([0], [(1, 2, 3), (4, 5, 6), (7, 8, 9)]))

    def test_the_shot_in_hand_is_given_as_it_stands_when_the_play_is_over(self):
        alignment = TriggerAlignment(schedule(), from_first_trigger=True)
        self.assertEqual(sort(alignment, [RISING[0], RISING[1]]), ([], []))
        self.assertEqual(alignment.overdue_after, SHOT)
        (last,) = alignment.flush()
        self.assertEqual((last.index, last.frames, last.complete), (0, (0, 1, None), False))
        self.assertEqual((alignment.flush(), alignment.overdue_after), ([], None))

    def test_of_two_frames_near_one_trigger_the_nearer_is_its_frame(self):
        # An extra 17.5 ms before the third trigger, as recorded; and one 4 ms before the second (nearer than
        # half an exposure the camera could not have taken both).
        times = [RISING[0], RISING[1] - 0.004, RISING[1], RISING[2] - 0.0175, RISING[2], SHOT + RISING[0]]
        for separation in (EXPOSURE, 0.0):
            alignment = TriggerAlignment(schedule(), from_first_trigger=True, separation=separation)
            shots, extras = sort(alignment, times)
            self.assertEqual(([shot.frames for shot in shots], sorted(extras)), ([(0, 2, 4)], [1, 3]), f"separation {separation}")
        # With the camera's exposure known the shot is whole the moment its last frame is in; without it, at the next frame.
        known = TriggerAlignment(schedule(), from_first_trigger=True, separation=EXPOSURE)
        self.assertEqual(len(sort(known, [RISING[0], RISING[1], RISING[2] - 1e-5])[0]), 1)
        unknown = TriggerAlignment(schedule(), from_first_trigger=True)
        self.assertEqual(len(sort(unknown, [RISING[0], RISING[1], RISING[2] - 1e-5])[0]), 0)
        self.assertEqual([shot.frames for shot in unknown.flush()], [(0, 1, 2)])

    def test_a_schedule_says_what_it_cannot_align(self):
        with self.assertRaises(AlignmentError):
            schedule((((0.1,), 1.0), ((0.1, 0.2), 1.0)), run_repeats=1).frames
        with self.assertRaises(AlignmentError):
            schedule((((), 1.0),)).frames
        answer = {"revision": 7, "state": "playing", "lane": "ch11", "command_id": 3, "run_repeats": 2, "scan_repeats": 0, "sent_us": 5,
                  "points": [{"rising": [0.1, 0.2], "duration": 1.0}]}
        play = TriggerSchedule.from_answer(answer)
        self.assertEqual((play.revision, play.playing, play.frames, play.command_id, play.shot(10 ** 6)), (7, True, 2, 3, ((0.1, 0.2), 1.0)))
        self.assertIsNone(TriggerSchedule.from_answer({**answer, "scan_repeats": 1}).shot(2))


if __name__ == "__main__": unittest.main()
