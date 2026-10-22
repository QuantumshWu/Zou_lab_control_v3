"""One imaging Pulse and an explicitly authored SLM rearrangement Task."""

from __future__ import annotations

from dataclasses import replace

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.camera import CAMERA_PROTECTED_FIELDS, EXPOSURE_STEP_SECONDS
from zlc_atom.nodes._framework.descriptor import ArtifactOutputSpec, NodePreviewSpec, WorkspaceResourceSpec
from zlc_atom.nodes._framework.pulse import pulse_description
from zlc_atom.nodes.calibration import TrapCalibration
from zlc_atom.nodes.slm import CALIBRATION_INPUT, END_TARGET_INPUT, SCIENCE_CONTEXT_INPUT, SLM_PROTECTED_FIELDS
from zlc_atom.nodes.slm.rearrangement import REARRANGEMENT_FIELDS, rearrangement_defaults
from zlc_plugin import DeviceRequirement, NodeKind, NodeType

from .task import (
    AFTER_FRAME_OUTPUT,
    AFTER_OCCUPIED_OUTPUT,
    BEFORE_FRAME_OUTPUT,
    BEFORE_OCCUPIED_OUTPUT,
    PHASE_OUTPUT,
    REARRANGEMENT_ARTIFACT_CONTRACT,
    SlmRearrangementTask,
    rearrangement_outputs,
)


#: The core describes the chosen pulse (_framework.pulse); the build, the
#: period choices and the run read that description.
PULSE_RESOURCE = WorkspaceResourceSpec(
    "pulse_template",
    "zlc.pulse/slm-rearrangement",
    "pulses",
    (".json",),
    None,
    argument_name="pulse_resource",
)
#: The fields naming the chosen pulse's imaging Periods.
PERIOD_FIELDS = ("before_period", "after_period")


def _validate_rearrangement(values):
    if values["before_period"] == values["after_period"]:
        raise ValueError("Before and after imaging must select different Periods")


#: The form's sections, as v2's rearrangement form grouped it; the pulse
#: picker stands above them.
_GROUPS = {
    **dict.fromkeys(("target_rows", "target_columns"), "Target grid"),
    **dict.fromkeys(("before_period", "after_period", "exposure_seconds", "shots"), "Imaging"),
    **dict.fromkeys(("frame_mode", "motion_frames", "max_camera_step", "frame_rate_hz", "nominal_playback_seconds"),
                    "Movement"),
    **dict.fromkeys(("minimum_separation", "motion_intensity_error_percent", "intensity_error_percent",
                     "save_phase_sequence"), "Quality and output"),
}


def _grouped(fields):
    """``fields`` each in its section of the rearrangement form."""

    return tuple(replace(field, group=_GROUPS.get(field.name, "")) for field in fields)


SLM_REARRANGEMENT_SCHEMA = AuthoringSchema(
    _grouped((
        AuthoringField("pulse_template", "resource", "Imaging pulse", "", required=True),
        *REARRANGEMENT_FIELDS[:2],
        AuthoringField("before_period", "text", "Before imaging Period", "", required=True),
        AuthoringField("after_period", "text", "After imaging Period", "", required=True),
        AuthoringField("exposure_seconds", "float", "Camera exposure", None, required=False, minimum=1e-9, unit="s",
                       step=EXPOSURE_STEP_SECONDS,
                       description="Blank keeps the camera's current exposure; the Task does not compare it with Pulse Period lengths."),
        AuthoringField("shots", "int", "Shots", 1, minimum=1,
                       description="Pulses fired one after another, each photographed, rearranged and verified on its own. "
                                   "Shots after the first completed one keep their atom-by-atom data, not the phase sequence, optical diagnostics or figures."),
        *REARRANGEMENT_FIELDS[2:],
        AuthoringField("save_phase_sequence", "bool", "Save phase sequence", True),
    )),
    validator=_validate_rearrangement,
)


def _period_choices(values, resources):
    """The chosen pulse's Periods, each by its id with its operator label, for
    every field that names an imaging Period."""

    del values
    try:
        pulse = pulse_description((resources or {}).get(PULSE_RESOURCE.field_name))
    except TypeError:
        return {}
    choices = [[period["period_id"], period["label"]] for period in pulse.get("periods", ())]
    return {name: choices for name in PERIOD_FIELDS}


def _build(
    *,
    camera: object,
    camera_key: str,
    slm: object,
    slm_key: str,
    calibration_path: object,
    science_context_path: object,
    pulse_resource: object,
    end_target_path: object = None,
    **values: object,
) -> SlmRearrangementTask:
    authored = SLM_REARRANGEMENT_SCHEMA.project_values(values)
    calibration = CALIBRATION_INPUT.resolve_value(calibration_path)
    science_context = SCIENCE_CONTEXT_INPUT.resolve_value(science_context_path)
    end_target = END_TARGET_INPUT.resolve_value(end_target_path)
    if not isinstance(calibration.value, TrapCalibration):
        raise TypeError("calibration must be a resolved calibration artifact")
    pulse = pulse_description(pulse_resource)
    period_ids = tuple(period["period_id"] for period in pulse["periods"])
    for key in PERIOD_FIELDS:
        if authored[key] not in period_ids:
            raise ValueError(f"{key} names a Period absent from the selected imaging pulse")
    if period_ids.index(authored["before_period"]) >= period_ids.index(authored["after_period"]):
        raise ValueError("Before imaging Period must precede after imaging Period")
    destination = None
    if end_target is not None:
        destination, kind = end_target.value
        if kind != "spots":
            raise ValueError("The end Target must be a spots Target")
    return SlmRearrangementTask(
        camera=camera,
        camera_key=camera_key,
        slm=slm,
        slm_key=slm_key,
        calibration=calibration.value,
        calibration_path=calibration.path,
        science_context=science_context.value,
        science_context_path=science_context.path,
        target_intensity=destination,
        target_path=None if end_target is None else end_target.path,
        pulse=pulse,
        intensity_tolerance=1.0 + authored["intensity_error_percent"] / 100.0,
        motion_intensity_tolerance=1.0 + authored["motion_intensity_error_percent"] / 100.0,
        **{key: value for key, value in authored.items()
           if key not in {"pulse_template", "nominal_playback_seconds", "intensity_error_percent", "motion_intensity_error_percent"}},
    )


#: The before and after photographs with their occupancy, and the phase on the SLM.
PREVIEWS = (
    NodePreviewSpec(BEFORE_FRAME_OUTPUT, "image", overlay=BEFORE_OCCUPIED_OUTPUT),
    NodePreviewSpec(AFTER_FRAME_OUTPUT, "image", overlay=AFTER_OCCUPIED_OUTPUT),
    NodePreviewSpec(PHASE_OUTPUT, "image"),
)


LOGIC_NODE = NodeType(
    "slm_rearrangement",
    NodeKind.TASK,
    SLM_REARRANGEMENT_SCHEMA,
    description=(
        "Use a Calibration, SLM Science Context and authored Pulse to image occupancy, "
        "plan collision-constrained atom transport to a grid or saved target, play GPU "
        "phase maps and image the result. This Task owns camera preparation and Pulse "
        "Fire, publishing before/after images, occupancy and phase diagnostics and "
        "saving the native rearrangement artifacts."
    ),
    input_specs=(CALIBRATION_INPUT, SCIENCE_CONTEXT_INPUT, END_TARGET_INPUT),
    declare_outputs=lambda values, devices: rearrangement_outputs(str(values.get("frame_mode", "fixed"))),
    node_previews=PREVIEWS,
    artifact_outputs=(ArtifactOutputSpec("artifact_path", REARRANGEMENT_ARTIFACT_CONTRACT),),
    device_requirements=(
        DeviceRequirement("camera.adapter", "camera", CAMERA_PROTECTED_FIELDS),
        # The core's native board, driven through pulse_* (claimed, as v2 did).
        DeviceRequirement("sequencer.streamer", "sequencer", ("program",)),
        DeviceRequirement("slm.phase", "slm", SLM_PROTECTED_FIELDS),
    ),
    build=_build,
    workspace_resources=(PULSE_RESOURCE,),
    resolve_field_choices=_period_choices,
    resolve_defaults=rearrangement_defaults,
)


__all__ = ["LOGIC_NODE", "PULSE_RESOURCE", "SLM_REARRANGEMENT_SCHEMA"]
