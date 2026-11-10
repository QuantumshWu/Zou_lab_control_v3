"""A region drawn on a scan's plot sets that scan's next range: each wall by itself, a wall
past the data widening the range, one in the outermost point's own cell keeping the end."""
import json
import pathlib
import sys
import types
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "science" / "src"))
from zlc_atom.nodes.scan.plan import _selected_plan, port_label, scan_axis_ids

PORT = "pulse:param:detuning"
AXIS = scan_axis_ids([port_label(PORT)])[0]
#: The scan played 100, 105, 110, 115, 120 (a step of 5: its outermost cells reach 97.5 and 122.5).
PLAYED = [100.0, 105.0, 110.0, 115.0, 120.0]


def selected(lower, upper, row=None):
    row = row or {"port": PORT, "unit": "MHz", "mode": "range", "value_text": "", "values": [], "range": {"start": 100, "stop": 120, "count": 5}}
    selection = types.SimpleNamespace(ranges=(types.SimpleNamespace(axis=AXIS, lower=lower, upper=upper),))
    context = {"run_record": {"plan": {"axes": [{"port": PORT, "values": PLAYED, "unit": "MHz"}]}}, "axis_units": {AXIS: "MHz"}}
    draft = {"plan": json.dumps({"axes": [row]})}
    patch = _selected_plan(selection, draft, context)
    return json.loads(patch["plan"])["axes"][0], patch["plan"] == draft["plan"]


class SelectedRange(unittest.TestCase):
    def ends(self, lower, upper, **row):
        axis, _ = selected(lower, upper, **row)
        return axis["range"]["start"], axis["range"]["stop"], axis["range"]["count"]

    def test_a_region_inside_the_data_narrows_the_range_and_keeps_its_points(self):
        self.assertEqual(self.ends(106, 114), (106, 114, 5))

    def test_a_wall_past_the_data_widens_the_range(self):
        self.assertEqual(self.ends(80, 114), (80, 114, 5), "below the first point")
        self.assertEqual(self.ends(106, 150), (106, 150, 5), "above the last point")
        self.assertEqual(self.ends(60, 180), (60, 180, 5), "both")
        self.assertEqual(self.ends(130, 150), (130, 150, 5), "wholly beyond what was scanned")

    def test_a_wall_in_the_outermost_cell_keeps_the_rows_end(self):
        # The half pixel an image draws past its last column, or a little inside it.
        self.assertEqual(self.ends(97.5, 114), (100, 114, 5))
        self.assertEqual(self.ends(102, 114), (100, 114, 5))
        self.assertEqual(self.ends(106, 122.5), (106, 120, 5))
        _, untouched = selected(97.5, 122.5)
        self.assertTrue(untouched, "the whole picture: the plan as written")
        _, untouched = selected(98, 121)
        self.assertTrue(untouched)

    def test_a_descending_range_keeps_its_direction(self):
        row = {"port": PORT, "unit": "MHz", "mode": "range", "value_text": "", "values": [], "range": {"start": 120, "stop": 100, "count": 5}}
        self.assertEqual(self.ends(80, 114, row=row), (114, 80, 5))

    def test_a_values_row_still_keeps_the_listed_values_the_region_holds(self):
        row = {"port": PORT, "unit": "MHz", "mode": "values", "value_text": "100, 105, 110, 115, 120", "values": PLAYED}
        axis, _ = selected(80, 111, row=row)
        self.assertEqual(axis["value_text"], "100, 105, 110")
        with self.assertRaises(ValueError):
            selected(130, 150, row=row)


if __name__ == "__main__": unittest.main()
