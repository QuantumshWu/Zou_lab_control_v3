"""Frame Survival on a cycle that carries no pair: the shots are published, every value invalid, and said once.

A one-frame cycle, or one that carries none of the selected pairs, is data and not a fault: each shot is
published with the pair axis holding one ``no pair`` entry and validity all false, so a rolling panel keeps
rolling and a same-shot group is answered.  Wiring mistakes (counts or camera frames picked, no single
readout-event axis) stay errors.  Occupancy's post-selection, a condition on two frames, follows the same rule.
"""
import pathlib
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "science" / "src"))
from zlc_data import DatasetSchema, DomainSpec, ValidityContract, ValueSchema, owned_snapshot_from_arrays
from zlc_data.axis import READOUT_EVENT, REPEAT, SCAN_POINT, SITE, AxisId, AxisSpec
from zlc_atom.nodes.frame_survival.processor import NO_PAIR, ONE_FRAME, FrameSurvivalProcessor
from zlc_atom.nodes.occupancy.post_selection import post_select_step
from zlc_atom.nodes.occupancy.processor import OccupancyResult

SITES = 4
REPEATS = AxisSpec(AxisId("shot"), "shot", REPEAT, 1, coordinates=(0,))
SITE_AXIS = AxisSpec(AxisId("site"), "site", SITE, SITES, coordinates=tuple(range(SITES)))
SCAN = AxisSpec(AxisId("scan.duration"), "duration", SCAN_POINT, 2, coordinates=(1.0, 2.0), unit="ms")


def frame_axis(frames, axis_id="camera.frame"):
    return AxisSpec(AxisId(axis_id), "frame", READOUT_EVENT, frames, coordinates=tuple(range(frames)))


def occupied_schema(frames, *, dtype="?", point=None, cells=(SITE_AXIS,)):
    point = point or DomainSpec((frames,), (frame_axis(frames),), (range(frames),))
    return DatasetSchema(DomainSpec((1,), (REPEATS,), (range(1),)), point,
                         DomainSpec(tuple(axis.size for axis in cells), cells),
                         ValueSchema(ValidityContract.components(*(axis.axis_id for axis in cells)), np.dtype(dtype), "1",
                                     name="occupied"))


def delivered(schema, occupied, valid=None, *, canonical=None, origin=None):
    shape = (1, schema.point_domain.size, *schema.cell_domain.shape)
    occupied = np.asarray(occupied, dtype=schema.value_schema.dtype).reshape(shape)
    valid = np.ones(occupied.shape, bool) if valid is None else np.asarray(valid, bool).reshape(occupied.shape)
    snapshot = owned_snapshot_from_arrays(schema, occupied, 1, validity=valid, stream_generation="source")
    return types.SimpleNamespace(name="@logic/occupancy1/occupied", snapshot=snapshot, canonical_schema=canonical,
                                 cell_origin=origin)


class Context:
    """The run's status line: what report_progress said, in order."""

    def __init__(self):
        self.said = []

    def report_progress(self, message, *, current=None, total=None):
        self.said.append(message)


def published(outputs):
    survival = outputs["survival"].snapshot
    values = np.asarray(survival.materialize().block.values, dtype=bool)
    valid = np.asarray(survival.expanded_validity(), dtype=bool)
    pair = next(axis for axis in survival.block.schema.point_domain.axes if axis.role == READOUT_EVENT)
    return pair.labels(), values, valid


class NoPairCycles(unittest.TestCase):
    def processor(self, **options):
        processor = FrameSurvivalProcessor(source_signal="@logic/occupancy1/occupied", **options)
        context = Context()
        processor.bind_run(context)
        return processor, context

    def test_a_one_frame_source_publishes_every_shot_invalid_under_one_no_pair_entry(self):
        processor, context = self.processor()
        schema = occupied_schema(1)
        for _shot in range(3):
            labels, values, valid = published(processor.evaluate(delivered(schema, [1, 0, 1, 1])))
            self.assertEqual(labels, (NO_PAIR,))
            self.assertEqual(valid.shape, (1, 1, SITES), "(shots) x (one entry) x (sites)")
            self.assertFalse(valid.any(), "no site of a one-frame cycle is a trial")
            self.assertFalse(values.any())
        self.assertEqual(len(context.said), 1, "said once a run, not once a shot")
        self.assertTrue(context.said[0].startswith(ONE_FRAME), context.said[0])
        record = processor.describe_run({"occupied": delivered(schema, [1, 0, 1, 1])})["parameters"]
        self.assertEqual((record["frames"], record["pairs"]), (1, ()))

    def test_an_explicit_first_to_last_pair_is_absent_in_a_one_frame_cycle(self):
        processor, context = self.processor(pairs=((0, "last"),))
        labels, _values, valid = published(processor.evaluate(delivered(occupied_schema(1), [1, 1, 1, 1])))
        self.assertEqual(labels, (NO_PAIR,), "first and last are one frame: no pair, not 0→0")
        self.assertFalse(valid.any())
        record = processor.describe_run({"occupied": delivered(occupied_schema(1), [1, 1, 1, 1])})["parameters"]
        self.assertEqual(record["absent_pairs"], ((0, "last"),))

    def test_a_source_going_from_one_to_three_frames_publishes_its_pairs_without_a_restart(self):
        processor, context = self.processor()
        published(processor.evaluate(delivered(occupied_schema(1), [1, 1, 1, 1])))
        # Rows are frames 0, 1, 2; site 3 is unjudged in frame 2.
        occupied = np.array([[1, 1, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]])
        valid = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 0]])
        labels, values, kept = published(processor.evaluate(delivered(occupied_schema(3), occupied, valid)))
        self.assertEqual(labels, ("0→1", "0→2", "1→2"))
        self.assertEqual(kept[0].astype(int).tolist(), [[1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 0]])
        self.assertEqual(values[0].astype(int).tolist(), [[1, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0]])
        self.assertEqual(context.said[1:], [""], "once it has pairs the status line reads running again")

    def test_selected_pairs_the_cycle_lacks_publish_invalid_and_name_them(self):
        processor, context = self.processor(pairs=((1, 2),))
        labels, _values, valid = published(processor.evaluate(delivered(occupied_schema(3), np.ones((3, SITES)))))
        self.assertEqual(labels, (NO_PAIR,))
        self.assertFalse(valid.any())
        self.assertIn("kept for a longer cycle: 1→2", context.said[0])
        labels, _values, valid = published(processor.evaluate(delivered(occupied_schema(4), np.ones((4, SITES)))))
        self.assertEqual(labels, ("1→2",), "a cycle that carries it publishes it")
        self.assertTrue(valid.all())

    def test_a_finite_run_places_one_no_pair_entry_per_cycle(self):
        processor, _context = self.processor()
        frame = frame_axis(1)
        canonical = occupied_schema(1, point=DomainSpec((2,), (SCAN, frame), (range(2), (0, 0))))
        for point in range(2):
            event = occupied_schema(1, point=DomainSpec((1,), (SCAN, frame), ((point,), (0,))))
            outputs = processor.evaluate(delivered(event, [1, 1, 1, 1], canonical=canonical, origin=(0, point)))
            self.assertEqual(outputs["survival"].cell_origin, (0, point))
            self.assertEqual(outputs["survival"].canonical_schema.point_domain.shape, (2,))
            self.assertFalse(published(outputs)[2].any())

    def test_wiring_mistakes_still_stop_the_run(self):
        processor, _context = self.processor()
        with self.assertRaisesRegex(ValueError, "'counts'"):
            processor.evaluate(delivered(occupied_schema(1, dtype="<f4"), [1.0] * SITES))
        pixels = (AxisSpec(AxisId("y"), "y", SITE, 2, coordinates=(0, 1)), AxisSpec(AxisId("x"), "x", SITE, 2, coordinates=(0, 1)))
        with self.assertRaisesRegex(ValueError, "camera frames"):
            processor.evaluate(delivered(occupied_schema(1, cells=pixels), np.ones((1, 4))))
        two = DomainSpec((1,), (frame_axis(1), frame_axis(1, "camera.other")), ((0,), (0,)))
        with self.assertRaisesRegex(ValueError, "declares 2"):
            processor.evaluate(delivered(occupied_schema(1, point=two), [1] * SITES))


class PostSelectionWithoutItsFrames(unittest.TestCase):
    def result(self, frames):
        schema = occupied_schema(frames)
        occupied = owned_snapshot_from_arrays(schema, np.ones((1, frames, SITES), bool), 1,
                                              validity=np.ones((1, frames, SITES), bool), stream_generation="g")
        counts_schema = DatasetSchema(schema.repeat_domain, schema.point_domain, schema.cell_domain,
                                      ValueSchema(ValidityContract.components(SITE_AXIS.axis_id), np.dtype("<f4"), "count",
                                                  name="counts"))
        counts = owned_snapshot_from_arrays(counts_schema, np.full((1, frames, SITES), 7.0, "<f4"), 1,
                                            validity=np.ones((1, frames, SITES), bool), stream_generation="g")
        return OccupancyResult({"counts": counts, "occupied": occupied, "frame_judged": occupied}), schema

    def test_a_one_frame_cycle_keeps_nothing_and_is_published(self):
        step = post_select_step({"frames": ("first", "last"), "mode": "relation", "relation": "=="})
        result, schema = self.result(1)
        kept = step["post_select"](result, schema.point_domain.axes[0])
        for member in ("counts", "occupied"):
            self.assertFalse(np.asarray(kept.artifacts[member].expanded_validity()).any(), member)
            self.assertEqual(kept.artifacts[member].block.schema, result.artifacts[member].block.schema)
        self.assertIs(kept.artifacts["frame_judged"], result.artifacts["frame_judged"])
        self.assertIn("at least two frames", step["post_select_problem"](1))
        self.assertEqual(step["post_select_problem"](2), "")

    def test_an_absent_middle_frame_keeps_nothing(self):
        step = post_select_step({"frames": ("first", 3), "mode": "relation", "relation": "=="})
        result, schema = self.result(3)
        kept = step["post_select"](result, schema.point_domain.axes[0])
        self.assertFalse(np.asarray(kept.artifacts["occupied"].expanded_validity()).any())
        self.assertIn("middle frame 3", step["post_select_problem"](3))


if __name__ == "__main__":
    unittest.main()
