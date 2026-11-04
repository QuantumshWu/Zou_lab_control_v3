"""A source for Derive's groups through the real host: repeat x duration x Site, one
cell per event, every value r * 10 + p + site / 10; or a monitor of the same shots."""

import time

import numpy as np
from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_data import AxisId, AxisSpec, DatasetSchema, DomainSpec, SITE, ValidityContract, ValueSchema, owned_snapshot_from_arrays
from zlc_data.axis import REPEAT, SCAN_POINT
from zlc_plugin import DatasetOutputDeclaration, LiveDatasetOutput, NodeKind, NodeType

COUNTS = DatasetOutputDeclaration("counts", "zlc.test.counts")
SCHEMA = AuthoringSchema((AuthoringField("repeats", "int", "Repeats", 5, minimum=1), AuthoringField("points", "int", "Points", 2, minimum=1),
                          AuthoringField("monitor", "bool", "Monitor", False), AuthoringField("pause", "float", "Pause", 0.05)))


def _schema(repeats, points, sites, coordinates):
    site = AxisSpec(AxisId("site"), "Site", SITE, sites, coordinates=np.arange(sites))
    return DatasetSchema(
        DomainSpec((repeats,), (AxisSpec(AxisId("repeat"), "repeat", REPEAT, repeats, coordinates=np.arange(coordinates[0], coordinates[0] + repeats)),), (range(repeats),)),
        DomainSpec((points,), (AxisSpec(AxisId("duration"), "duration", SCAN_POINT, points, coordinates=np.arange(coordinates[1], coordinates[1] + points) * 2.0 + 1.0, unit="µs"),), (range(points),)),
        DomainSpec((sites,), (site,)), ValueSchema(ValidityContract.components(site.axis_id), np.dtype("<f8"), "V", name="counts"))


class Repeater:
    def __init__(self, repeats, points, monitor, pause):
        self.repeats, self.points, self.monitor, self.pause = repeats, points, monitor, pause

    def execute(self, context):
        sites = 3
        whole = _schema(self.repeats, self.points, sites, (0, 0))
        context.report_ready()
        for r in range(self.repeats):
            for p in range(1 if self.monitor else self.points):
                values = np.full((1, 1, sites), r * 10.0 + p) + np.arange(sites) / 10.0
                if self.monitor:
                    snapshot = owned_snapshot_from_arrays(_schema(1, 1, sites, (0, 0)), values, r)
                    context.commit_live({"counts": LiveDatasetOutput(COUNTS, snapshot)})
                else:
                    snapshot = owned_snapshot_from_arrays(_schema(1, 1, sites, (0, 0)), values, r * self.points + p)
                    context.commit_live({"counts": LiveDatasetOutput(COUNTS, snapshot, canonical_schema=whole, cell_origin=(r, p))})
                time.sleep(self.pause)
        return {}


LOGIC_NODE = NodeType("repeater", NodeKind.MEASUREMENT, SCHEMA, description="Publishes test shots, finite or as a monitor.",
                      outputs=(COUNTS,), build=lambda repeats, points, monitor, pause: Repeater(repeats, points, monitor, pause), reports_ready=True)
