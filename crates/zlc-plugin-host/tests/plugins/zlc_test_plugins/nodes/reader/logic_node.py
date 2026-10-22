"""A node for the host's own tests: started on a source over its 'latest' or 'exact' input
(whichever `mode` enables), it never takes a delivery; over 'latest' it publishes a reading
of its own, without a parent, as a node reading its source by name does (SLM feedback)."""

import time

import numpy as np
from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_data import AxisId, AxisSpec, DomainSpec, SPATIAL_X, SPATIAL_Y, ValidityContract, ValidityMode, ValueSchema, owned_snapshot_from_arrays
from zlc_plugin import DatasetInput, DatasetOutputDeclaration, LiveDatasetOutput, NodeKind, NodeType

READING = DatasetOutputDeclaration("reading", "zlc.test.reading")
SCHEMA = AuthoringSchema((AuthoringField("mode", "str", "Mode", "latest"),))


def cells(size):
    return DomainSpec((size, size), (AxisSpec(AxisId("test.y"), "y", SPATIAL_Y, size), AxisSpec(AxisId("test.x"), "x", SPATIAL_X, size)))


class Reader:
    def __init__(self, mode):
        self.mode = mode

    def execute(self, context):
        if self.mode == "latest":
            value = ValueSchema(ValidityContract(ValidityMode.VALUE), np.dtype("<u2"), "count")
            snapshot = owned_snapshot_from_arrays(value, np.ones((1, 1, 2, 2), dtype="<u2"), 0, cell_domain=cells(2))
            context.commit_live({"reading": LiveDatasetOutput(READING, snapshot)})
        time.sleep(1.0)
        return {}


LOGIC_NODE = NodeType("reader", NodeKind.TASK, SCHEMA, description="Never takes its input.",
                      input_specs=(DatasetInput("latest", None, "latest", enabled_when=("mode", ("latest",))),
                                   DatasetInput("exact", None, "exact", enabled_when=("mode", ("exact",)))),
                      outputs=(READING,), build=lambda mode, source_signal: Reader(mode))
