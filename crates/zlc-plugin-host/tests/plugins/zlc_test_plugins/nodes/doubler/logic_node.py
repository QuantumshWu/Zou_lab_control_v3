"""A processor for the host's own tests: doubles each delivered frame."""

import numpy as np
from zlc_atom.authoring import AuthoringSchema
from zlc_data import ValidityContract, ValidityMode, ValueSchema, owned_snapshot_from_arrays
from zlc_plugin import DatasetInput, DatasetOutputDeclaration, LiveDatasetOutput, NodeKind, NodeType

DOUBLED = DatasetOutputDeclaration("doubled", "zlc.test.frames")


class Doubler:
    def describe_run(self, inputs):
        source = inputs["source"]
        return {"source": source.name, "source_count": source.run_record.get("count")}

    def evaluate(self, value):
        frames = value.snapshot.materialize().block
        doubled = frames.values.astype("<u4") * 2
        schema = ValueSchema(ValidityContract(ValidityMode.VALUE), np.dtype("<u4"), "count")
        snapshot = owned_snapshot_from_arrays(schema, doubled, int(value.key["revision"]),
                                              repeat_domain=frames.schema.repeat_domain,
                                              point_domain=frames.schema.point_domain,
                                              cell_domain=frames.schema.cell_domain)
        return {"doubled": LiveDatasetOutput(DOUBLED, snapshot, event_record={"from": value.event_record})}


LOGIC_NODE = NodeType("doubler", NodeKind.PROCESSOR, AuthoringSchema(()), description="Doubles the frames it follows.",
                      input_specs=(DatasetInput("source", "zlc.test.frames", "exact"),), outputs=(DOUBLED,),
                      build=lambda source_signal: Doubler())
