"""A source for Derive through the real host: a saved dataset (``data.npz`` with ``info``,
``data``, ``data.validity``), or a Rabi scan drifting by a known law, of booleans with
validity like a survival, published ``block`` points at a time (0: whole, in one commit)
at most every ``pause`` seconds, the first ``limit`` of them when it says (0: all)."""

import json
import math
import time

import numpy as np
from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.nodes.derive.groups import restricted
from zlc_data import AxisId, AxisSpec, DatasetSchema, DomainSpec, SITE, ValidityContract, ValueSchema, owned_snapshot_from_arrays
from zlc_data.axis import REPEAT, SCAN_POINT
from zlc_data.codec import dataset_schema_from_tree
from zlc_plugin import DatasetOutputDeclaration, LiveDatasetOutput, NodeKind, NodeType

SURVIVAL = DatasetOutputDeclaration("survival", "zlc.test.survival")
SCHEMA = AuthoringSchema((AuthoringField("path", "text", "File", ""), AuthoringField("repeats", "int", "Repeats", 200, minimum=1),
                          AuthoringField("points", "int", "Points", 60, minimum=1), AuthoringField("sites", "int", "Sites", 35, minimum=1),
                          AuthoringField("block", "int", "Block", 0, minimum=0), AuthoringField("pause", "float", "Pause", 0.0),
                          AuthoringField("limit", "int", "Publications", 0, minimum=0)))


def rabi_frequency(progress):
    """The law the synthetic scan drifts by, in MHz over its axis in µs; ``progress`` from 0 to 1."""

    return 0.026 + 0.009 * progress


def synthetic(repeats, points, sites):
    repeat = AxisSpec(AxisId("repeat"), "repeat", REPEAT, repeats, coordinates=np.arange(repeats))
    duration = AxisSpec(AxisId("duration"), "duration", SCAN_POINT, points, coordinates=np.linspace(0.1, 150.0, points), unit="µs")
    site = AxisSpec(AxisId("site"), "Site", SITE, sites, coordinates=np.arange(sites))
    schema = DatasetSchema(DomainSpec((repeats,), (repeat,), (range(repeats),)), DomainSpec((points,), (duration,), (range(points),)), DomainSpec((sites,), (site,)),
                           ValueSchema(ValidityContract.components(site.axis_id), np.dtype(bool), "1", name="survival"))
    rng = np.random.default_rng(20261104)
    t = np.asarray(duration)[None, :, None]
    f = rabi_frequency(np.arange(repeats)[:, None, None] / max(repeats - 1, 1))
    survive = 0.09 + 0.06 * np.exp(-t / 300.0) * np.cos(2.0 * math.pi * f * t + 0.4)
    return schema, rng.random(schema.physical_shape) < survive, rng.random(schema.physical_shape) < 0.32


def saved(path):
    with np.load(path) as stored:
        schema = dataset_schema_from_tree(json.loads(str(stored["info"]))["sections"]["dataset"]["data"]["schema"])
        return schema, stored["data"], stored["data.validity"]


class Scanner:
    def __init__(self, path, repeats, points, sites, block, pause, limit):
        self.path, self.repeats, self.points, self.sites, self.block, self.pause, self.limit = path, repeats, points, sites, block, pause, limit

    def execute(self, context):
        whole, values, valid = saved(self.path) if self.path else synthetic(self.repeats, self.points, self.sites)
        repeats, points = whole.repeat_domain.size, whole.point_domain.size
        places = [(r, p, self.block) for r in range(repeats) for p in range(0, points, self.block)] if self.block else [(0, 0, points)]
        places = places[:self.limit or len(places)]
        rows = 1 if self.block else repeats
        context.report_ready()
        started = time.perf_counter()
        for k, (r, p, size) in enumerate(places):
            time.sleep(max(0.0, started + k * self.pause - time.perf_counter()))
            size = min(size, points - p)
            schema = restricted(whole, (r, p), (rows, size))
            snapshot = owned_snapshot_from_arrays(schema, values[r:r + rows, p:p + size], k, validity=valid[r:r + rows, p:p + size])
            context.commit_live({"survival": LiveDatasetOutput(SURVIVAL, snapshot, canonical_schema=whole, cell_origin=(r, p))})
        return {"publications": len(places), "seconds": time.perf_counter() - started}


LOGIC_NODE = NodeType("scanner", NodeKind.MEASUREMENT, SCHEMA, description="Publishes a scan, saved or synthetic.", outputs=(SURVIVAL,),
                      build=lambda path, repeats, points, sites, block, pause, limit: Scanner(path, repeats, points, sites, block, pause, limit), reports_ready=True)
