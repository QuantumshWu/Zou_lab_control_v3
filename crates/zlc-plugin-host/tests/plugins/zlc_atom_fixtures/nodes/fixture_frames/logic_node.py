"""A finite camera-like run replayed from a recorded NPZ: (cycles, frames, y, x) site frames.

The v2 known-truth fixture stores two reference frames and one short frame per
cycle; each cycle is published as [reference 0, short, reference 1], on a
READOUT_EVENT frame axis and sensor-pixel axes, placed in the run's geometry,
as camera_measurement publishes a finite run.
"""

from dataclasses import replace
import time

import numpy as np
from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.data import cell_axis_id, snapshot_from_array
from zlc_data import AxisId, AxisSpec, CoordinateFrameId, DomainSpec, READOUT_EVENT, SPATIAL_X, SPATIAL_Y
from zlc_plugin import DatasetOutputDeclaration, LiveDatasetOutput, NodeKind, NodeType

FRAMES = DatasetOutputDeclaration("frames", "camera.frames")
SCHEMA = AuthoringSchema((
    AuthoringField("path", "str", "Fixture NPZ", "", required=True),
    # Followers subscribe from the current event on: the first cycle is held so
    # they can join before the rest arrives.
    AuthoringField("hold_seconds", "float", "Hold after the first cycle", 0.0, minimum=0.0),
))


class FixtureFrames:
    def __init__(self, path, hold_seconds):
        self.hold_seconds = float(hold_seconds)
        with np.load(path, allow_pickle=False) as archive:
            reference = np.array(archive["input_reference_frames"])
            short = np.array(archive["input_short_frames"])
        self.cycles = np.stack((reference[:, 0], short, reference[:, 1]), axis=1)

    def execute(self, context):
        producer = context.instance_id
        repeat, frames, height, width = self.cycles.shape
        frame_axis = AxisSpec(AxisId(f"{producer}.frames.frame"), "frame", READOUT_EVENT, frames, tuple(range(frames)))

        def pixel(index, role, size):
            return AxisSpec(cell_axis_id(producer, "frames", index, role), role.value, role, size,
                            coordinates=tuple(range(size)), unit="pixel",
                            coordinate_frame=CoordinateFrameId("sensor_pixel_xy"))

        cells = (pixel(0, SPATIAL_Y, height), pixel(1, SPATIAL_X, width))
        context.set_run_record({"device_snapshots": {"camera": {"sensor_shape_yx": [height, width]}}})
        context.report_ready()
        canonical = None
        for index in range(repeat):
            event = snapshot_from_array(self.cycles[index:index + 1], producer=producer, signal="frames",
                                        point_axes=(frame_axis,), cell_axes=cells, value_unit=None,
                                        generation=context.generation, revision=index + 1)
            if canonical is None:
                schema = event.block.schema
                (repeat_axis,) = schema.repeat_domain.axes
                canonical = replace(schema, repeat_domain=DomainSpec(
                    (repeat,), (replace(repeat_axis, size=repeat),), (tuple(range(repeat)),)))
            context.commit_live({"frames": LiveDatasetOutput(FRAMES, event, canonical_schema=canonical,
                                                             cell_origin=(index, 0))})
            if index == 0:
                time.sleep(self.hold_seconds)
        return {"cycles": repeat}


LOGIC_NODE = NodeType("fixture_frames", NodeKind.MEASUREMENT, SCHEMA, description="Replays recorded site frames.",
                      outputs=(FRAMES,), build=lambda path, hold_seconds: FixtureFrames(path, hold_seconds), reports_ready=True)
