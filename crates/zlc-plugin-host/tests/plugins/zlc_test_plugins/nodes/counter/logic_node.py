"""A node for the host's own tests: publishes device frames through Runtime."""

import numpy as np
from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_data import AxisId, AxisSpec, DomainSpec, SPATIAL_X, SPATIAL_Y, ValidityContract, ValidityMode, ValueSchema
from zlc_data import owned_snapshot_from_arrays
from zlc_plugin import DatasetOutputDeclaration, DeviceRequirement, LiveDatasetOutput, NodeKind, NodeType

import zlc_test_plugins.devices.test.contract  # noqa: F401 -- registers the capability

FRAMES = DatasetOutputDeclaration("frames", "zlc.test.frames")
SCHEMA = AuthoringSchema((AuthoringField("count", "int", "Count", 3, minimum=1), AuthoringField("size", "int", "Size", 128, minimum=1)))


class Counter:
    def __init__(self, count, size, echo):
        if size == 13:
            raise ValueError("unsupported test frame shape")
        self.count, self.size, self.echo = count, size, echo

    def execute(self, context):
        from zlc_atom.nodes.scan.plan import DeviceKnob
        from zlc_atom.nodes.scan.devices import ScanDeviceKnobs
        knobs = ScanDeviceKnobs({"echo": DeviceKnob(self.echo)})
        knobs.move("device:echo:amplitude", 1.0, "Vpp")
        tuned = self.echo.tunable_values()["amplitude"]
        knobs.restore()
        context.set_run_record({"count": self.count, "opened_with": self.echo.opened_with,
                                "converted": list(self.echo.convert_tunable_value("gain", [1.0, 2.0], "1", "1")),
                                "tuned_dbm": tuned, "restored_dbm": self.echo.tunable_values()["amplitude"],
                                "node": "wrong-plugin-id", "node_name": "wrong-plugin-name"})
        context.report_ready()
        cells = DomainSpec((self.size, self.size), (AxisSpec(AxisId("test.y"), "y", SPATIAL_Y, self.size),
                                                    AxisSpec(AxisId("test.x"), "x", SPATIAL_X, self.size)))
        value = ValueSchema(ValidityContract(ValidityMode.VALUE), np.dtype("<u2"), "count")
        for index in range(self.count):
            if context.cancel_requested():
                break
            frame = self.echo.frame(self.size, index + 1)
            snapshot = owned_snapshot_from_arrays(value, frame.reshape(1, 1, self.size, self.size), index, cell_domain=cells)
            context.commit_live({"frames": LiveDatasetOutput(FRAMES, snapshot, event_record={"index": index})})
            context.report_progress("frames", current=index + 1, total=self.count)
        answer = context.core("probe", value=self.count)
        return {"published": self.count, "device_total": self.echo.total(frame), "core": answer}


def _editor_facts(values, resources, devices, source, bundle, artifacts):
    """A draft query reads its device (a conversion) and may change nothing (a fire is refused)."""
    echo = devices["echo"]
    try:
        echo.fire()
        refused = ""
    except Exception as error:  # noqa: BLE001 -- the refusal is the answer
        refused = str(error)
    return {"editor": "test", "manages": [], "converted": list(echo.convert_tunable_value("gain", [1.0, 2.0], "1", "1")),
            "refused": refused}


LOGIC_NODE = NodeType("counter", NodeKind.MEASUREMENT, SCHEMA, description="Publishes test frames from its echo device.",
                      outputs=(FRAMES,), device_requirements=(DeviceRequirement("test.echo", "echo"),),
                      build=lambda count, size, echo, echo_key: Counter(count, size, echo), reports_ready=True,
                      editor_facts=_editor_facts)
