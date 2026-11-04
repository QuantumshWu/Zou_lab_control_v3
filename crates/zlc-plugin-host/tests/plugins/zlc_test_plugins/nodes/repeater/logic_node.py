"""A source for Derive's groups through the real host: repeat x duration x Site, one
cell per event, every value r * 10 + p + site / 10 (+ 100 per shot); or a monitor of the same shots.
``shots`` puts a second Repeat axis beside repeat, slowest when ``shots_slow``; ``places``
(JSON, [[row, point, rows, points], ...]) publishes those rectangles instead of a cell at a time."""

import json
import time

import numpy as np
from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_data import AxisId, AxisSpec, DatasetSchema, DomainSpec, SITE, ValidityContract, ValueSchema, owned_snapshot_from_arrays
from zlc_data.axis import REPEAT, SCAN_POINT
from zlc_plugin import DatasetOutputDeclaration, LiveDatasetOutput, NodeKind, NodeType

COUNTS = DatasetOutputDeclaration("counts", "zlc.test.counts")
SCHEMA = AuthoringSchema((AuthoringField("repeats", "int", "Repeats", 5, minimum=1), AuthoringField("points", "int", "Points", 2, minimum=1),
                          AuthoringField("monitor", "bool", "Monitor", False), AuthoringField("pause", "float", "Pause", 0.05),
                          AuthoringField("shots", "int", "Shots", 1, minimum=1), AuthoringField("shots_slow", "bool", "Shots slowest", False),
                          AuthoringField("places", "text", "Places", "")))


def _rows(repeats, shots, slow):
    """(repeat, shot) of each Repeat row."""

    if slow:
        return [(r, s) for s in range(shots) for r in range(repeats)]
    return [(r, s) for r in range(repeats) for s in range(shots)]


def _schema(repeats, points, sites, coordinates, shots=1, slow=False):
    site = AxisSpec(AxisId("site"), "Site", SITE, sites, coordinates=np.arange(sites))
    repeat = AxisSpec(AxisId("repeat"), "repeat", REPEAT, repeats, coordinates=np.arange(coordinates[0], coordinates[0] + repeats))
    if shots == 1:
        repeat_domain = DomainSpec((repeats,), (repeat,), (range(repeats),))
    else:
        shot = AxisSpec(AxisId("shot"), "shot", REPEAT, shots, coordinates=np.arange(shots))
        rows = _rows(repeats, shots, slow)
        repeat_domain = DomainSpec((len(rows),), (repeat, shot), (tuple(row[0] for row in rows), tuple(row[1] for row in rows)))
    return DatasetSchema(
        repeat_domain,
        DomainSpec((points,), (AxisSpec(AxisId("duration"), "duration", SCAN_POINT, points, coordinates=np.arange(coordinates[1], coordinates[1] + points) * 2.0 + 1.0, unit="µs"),), (range(points),)),
        DomainSpec((sites,), (site,)), ValueSchema(ValidityContract.components(site.axis_id), np.dtype("<f8"), "V", name="counts"))


class Repeater:
    def __init__(self, repeats, points, monitor, pause, shots, shots_slow, places):
        self.repeats, self.points, self.monitor, self.pause = repeats, points, monitor, pause
        self.shots, self.slow, self.places = shots, shots_slow, json.loads(places) if places else None

    def _finite(self, context, sites):
        whole = _schema(self.repeats, self.points, sites, (0, 0), self.shots, self.slow)
        rows = _rows(self.repeats, self.shots, self.slow)
        repeat = np.array([row[0] * 10.0 + row[1] * 100.0 for row in rows])[:, None, None]
        values = repeat + np.arange(self.points)[None, :, None] + np.arange(sites)[None, None, :] / 10.0
        places = self.places or [[row, p, 1, 1] for row in range(len(rows)) for p in range(self.points)]
        for k, (row, p, height, width) in enumerate(places):
            snapshot = owned_snapshot_from_arrays(_schema(height, width, sites, (0, 0)), values[row:row + height, p:p + width], k)
            context.commit_live({"counts": LiveDatasetOutput(COUNTS, snapshot, canonical_schema=whole, cell_origin=(row, p))})
            time.sleep(self.pause)

    def execute(self, context):
        sites = 3
        context.report_ready()
        if not self.monitor:
            self._finite(context, sites)
            return {}
        for r in range(self.repeats):
            values = np.full((1, 1, sites), r * 10.0) + np.arange(sites) / 10.0
            snapshot = owned_snapshot_from_arrays(_schema(1, 1, sites, (0, 0)), values, r)
            context.commit_live({"counts": LiveDatasetOutput(COUNTS, snapshot)})
            time.sleep(self.pause)
        return {}


LOGIC_NODE = NodeType("repeater", NodeKind.MEASUREMENT, SCHEMA, description="Publishes test shots, finite or as a monitor.", outputs=(COUNTS,),
                      build=lambda repeats, points, monitor, pause, shots, shots_slow, places: Repeater(repeats, points, monitor, pause, shots, shots_slow, places),
                      reports_ready=True)
