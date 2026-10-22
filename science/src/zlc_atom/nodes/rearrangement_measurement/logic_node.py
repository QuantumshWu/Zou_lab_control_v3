"""A camera-compatible acquisition with internal first-frame rearrangement."""
from dataclasses import replace

from zlc_atom.authoring import AuthoringSchema
from zlc_atom.devices.camera import CAMERA_PROTECTED_FIELDS
from zlc_atom.devices.camera.photoelectrons import resolve_photoelectron_availability
from zlc_atom.nodes._framework.descriptor import NodePreviewSpec
from zlc_atom.nodes.camera.authoring import CAMERA_MEASUREMENT_SCHEMA, IMAGE_AREA_TO_ROI, ROI_FIELD_NAMES
from zlc_atom.nodes.camera.measurement import CameraMeasurementRequest, CAMERA_FRAMES_OUTPUT, describe_camera_frames
from zlc_atom.nodes.slm import CALIBRATION_INPUT, END_TARGET_INPUT, SCIENCE_CONTEXT_INPUT, SLM_PROTECTED_FIELDS
from zlc_atom.nodes.slm.rearrangement import REARRANGEMENT_FIELDS, rearrangement_defaults
from zlc_plugin import DeviceRequirement, NodeKind, NodeType

from .measurement import RearrangementMeasurement


_MOVEMENT_FIELDS = {"target_rows", "target_columns", "frame_mode", "motion_frames",
                    "max_camera_step", "frame_rate_hz", "minimum_separation", "nominal_playback_seconds",
                    "intensity_error_percent", "motion_intensity_error_percent"}
REARRANGEMENT_MEASUREMENT_SCHEMA = AuthoringSchema(
    tuple(replace(field, default=2, minimum=2,
                  description="First frame starts rearrangement; the last frame restores the initial SLM state. Pulse timing and any verification image remain operator-controlled.")
          if field.name == "frames_per_cycle" else field for field in CAMERA_MEASUREMENT_SCHEMA.fields)
    + tuple(field for field in REARRANGEMENT_FIELDS if field.name in _MOVEMENT_FIELDS),
)

def _build(*, camera, camera_key, slm, slm_key, calibration_path, science_context_path,
           end_target_path=None, **values):
    authored = REARRANGEMENT_MEASUREMENT_SCHEMA.project_values(values)
    calibration = CALIBRATION_INPUT.resolve_value(calibration_path)
    science_context = SCIENCE_CONTEXT_INPUT.resolve_value(science_context_path)
    end_target = END_TARGET_INPUT.resolve_value(end_target_path)
    destination = None
    if end_target is not None:
        destination, kind = end_target.value
        if kind != "spots":
            raise ValueError("The end Target must be a spots Target")
    return RearrangementMeasurement(camera=camera, slm=slm, slm_key=slm_key,
        request=CameraMeasurementRequest(camera_key=camera_key, exposure_seconds=authored["exposure_seconds"],
            roi_xywh=tuple(authored[name] for name in ROI_FIELD_NAMES),
            repeat=authored["repeat"], frames_per_cycle=authored["frames_per_cycle"],
            photoelectrons=authored["photoelectrons"]),
        calibration=calibration.value, calibration_path=calibration.path,
        science_context=science_context.value, science_context_path=science_context.path,
        target_intensity=destination, target_path=None if end_target is None else str(end_target.path),
        intensity_tolerance=1. + authored["intensity_error_percent"] / 100.,
        motion_intensity_tolerance=1. + authored["motion_intensity_error_percent"] / 100.,
        **{key: value for key, value in authored.items() if key in _MOVEMENT_FIELDS
           - {"intensity_error_percent", "motion_intensity_error_percent", "nominal_playback_seconds"}})


LOGIC_NODE = NodeType("rearrangement_measurement", NodeKind.MEASUREMENT,
    REARRANGEMENT_MEASUREMENT_SCHEMA, reports_ready=True,
    description=(
        "Acquire camera cycles, classify their first frame with a Calibration and "
        "rearrange the occupied atoms using the selected SLM Science Context and target. "
        "Restore the initial Context after the last frame; publish only ordinary camera "
        "frames, compatible with Occupancy and Frame Survival. The caller supplies "
        "Pulse triggers; this measurement does not Fire."
    ),
    outputs=(CAMERA_FRAMES_OUTPUT,), node_previews=(NodePreviewSpec(CAMERA_FRAMES_OUTPUT, "facet_grid"),),
    describe_output_frames=describe_camera_frames,
    input_specs=(CALIBRATION_INPUT, SCIENCE_CONTEXT_INPUT, END_TARGET_INPUT),
    device_requirements=(DeviceRequirement("camera.adapter", "camera", CAMERA_PROTECTED_FIELDS),
                         DeviceRequirement("slm.phase", "slm", SLM_PROTECTED_FIELDS)),
    resolve_defaults=rearrangement_defaults, build=_build,
    selection_mappings=(IMAGE_AREA_TO_ROI,), resolve_field_availability=resolve_photoelectron_availability,
)

__all__ = ["LOGIC_NODE", "REARRANGEMENT_MEASUREMENT_SCHEMA"]
