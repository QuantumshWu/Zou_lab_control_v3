"""A node for the host's own tests: publishes one frame on a block, then runs until stopped."""

import time

import numpy as np
from zlc_atom.authoring import AuthoringSchema
from zlc_data import AxisId, AxisSpec, DomainSpec, SPATIAL_X, SPATIAL_Y, ValidityContract, ValidityMode, ValueSchema, owned_snapshot_from_arrays
from zlc_plugin import DatasetOutputDeclaration, LiveDatasetOutput, NodeKind, NodeType, allocate

FRAMES = DatasetOutputDeclaration("frames", "zlc.test.frames")


def cells(size):
    return DomainSpec((size, size), (AxisSpec(AxisId("test.y"), "y", SPATIAL_Y, size), AxisSpec(AxisId("test.x"), "x", SPATIAL_X, size)))


class Holder:
    def execute(self, context):
        frame = allocate((1, 1, 128, 128), "<u2")
        frame[...] = 5
        value = ValueSchema(ValidityContract(ValidityMode.VALUE), np.dtype("<u2"), "count")
        context.commit_live({"frames": LiveDatasetOutput(FRAMES, owned_snapshot_from_arrays(value, frame, 0, cell_domain=cells(128)))})
        while not context.cancel_requested():
            time.sleep(0.02)
        return {}


LOGIC_NODE = NodeType("holder", NodeKind.MEASUREMENT, AuthoringSchema(()), description="Publishes one frame and runs until stopped.",
                      outputs=(FRAMES,), build=lambda: Holder())
