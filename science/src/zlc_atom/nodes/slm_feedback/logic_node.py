"""SLM feedback from camera populations or a configured acquisition's fit."""

from __future__ import annotations

from zlc_plugin.figures import AxisRef, Reduction

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
from zlc_atom.devices.camera import CAMERA_PROTECTED_FIELDS
from zlc_atom.nodes._framework.descriptor import (
    ArtifactOutputSpec,
    NodePreviewSpec,
    WorkspaceResourceSpec,
)
from zlc_atom.nodes.calibration import TrapCalibration
from zlc_atom.nodes.camera.measurement import CAMERA_FRAMES_OUTPUT
from zlc_atom.nodes.slm import CALIBRATION_INPUT, SCIENCE_CONTEXT_INPUT, SLM_PROTECTED_FIELDS
from zlc_plugin import DatasetInput, DeviceRequirement, NodeKind, NodeType

from .task import (
    CANDIDATE_PHASE_OUTPUT,
    FIT_PARAMETER_CONTRACT,
    OBSERVABLE_UNIFORMITY_HISTORY_OUTPUT,
    SITE_SIGNAL_HISTORY_OUTPUT,
    SLM_PHASE_ARTIFACT_CONTRACT,
    SlmFeedbackTask,
    TARGET_SHARE_HISTORY_OUTPUT,
    UNIFORMITY_HISTORY_OUTPUT,
)


#: The core describes the chosen pulse (_framework.pulse); the build reads that description.
_PULSE_RESOURCE = WorkspaceResourceSpec(
    "pulse_template",
    "zlc.pulse/slm-feedback",
    "pulses",
    (".json",),
    None,
    argument_name="pulse_resource",
)
_CAMERA_MODES = ("feedback_mode", ("qcmos_bright_dark", "qcmos_loading_rate"))
_EXTERNAL_MODE = ("feedback_mode", ("ramsey_frequency",))


def _validate_feedback(values: dict[str, object]) -> None:
    if values["feedback_mode"] == "ramsey_frequency":
        return
    factors = tuple(float(value) for value in values["probe_factors"])
    if len(set(factors)) != len(factors) or any(
        value <= 0.0 or value == 1.0 for value in factors
    ):
        raise ValueError("probe_factors must be unique positive numbers excluding 1")

SLM_FEEDBACK_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "feedback_mode",
            "choice",
            "Feedback mode",
            "qcmos_bright_dark",
            choices=(
                AuthoringChoice(
                    "qcmos_bright_dark",
                    "qCMOS fluorescence (bright - dark)",
                ),
                AuthoringChoice(
                    "qcmos_loading_rate",
                    "qCMOS loading rate (bright shots / all shots)",
                ),
                AuthoringChoice(
                    "ramsey_frequency", "Ramsey frequency (external acquisition)"
                ),
            ),
        ),
        AuthoringField(
            "pulse_template", "resource", "Imaging pulse", "", required=True,
            enabled_when=_CAMERA_MODES,
        ),
        AuthoringField(
            "acquisition_logic", "str", "Acquisition logic", "", required=True,
            enabled_when=_EXTERNAL_MODE,
            description="Choose a finite acquisition; run it to completion for each SLM phase without changing its settings.",
        ),
        AuthoringField(
            "exposure_seconds",
            "float",
            "Camera exposure seconds",
            None,
            minimum=1e-9,
            enabled_when=_CAMERA_MODES,
            description="Blank keeps the camera's current exposure.",
        ),
        AuthoringField(
            "shots_per_candidate", "int", "qCMOS shots per candidate", 100, minimum=10,
            enabled_when=_CAMERA_MODES,
        ),
        AuthoringField(
            "probe_factors",
            "numeric_tuple",
            "Single-population probe factors",
            (0.5, 2.0),
            enabled_when=_CAMERA_MODES,
        ),
        # LOOP gain: the fraction of each site's residual removed per
        # candidate, once the plant slope has been measured.  Not a weight
        # multiplier -- the task divides by the measured response so that
        # 0.3 means 30% of the residual regardless of how hard the traps
        # answer; before a slope is trusted it steps at half this.
        AuthoringField(
            "feedback_gain",
            "float",
            "Loop gain (residual fraction removed per candidate)",
            0.3,
            minimum=0.0,
        ),
        AuthoringField(
            "maximum_weight_change",
            "float",
            "Maximum ordinary weight change",
            0.5,
            minimum=0.0,
        ),
        AuthoringField("max_updates", "int", "Maximum feedback updates", 12, minimum=1),
    ),
    validator=_validate_feedback,
)


def _build(
    *,
    camera: object = None,
    camera_key: str = "",
    sequencer_key: str = "",
    slm: object,
    slm_key: str,
    calibration_path: object,
    science_context_path: object,
    pulse_resource: object = None,
    source_signal: str = "",
    save_figure_artifact: object = None,
    **values: object,
) -> SlmFeedbackTask:
    authored = SLM_FEEDBACK_SCHEMA.project_values(values)
    calibration = CALIBRATION_INPUT.resolve_value(calibration_path)
    if not isinstance(calibration.value, TrapCalibration):
        raise TypeError("calibration must be a resolved calibration artifact")
    science_context = SCIENCE_CONTEXT_INPUT.resolve_value(science_context_path)
    external = authored["feedback_mode"] == "ramsey_frequency"
    return SlmFeedbackTask(
        camera=camera,
        camera_key=camera_key,
        sequencer_key=sequencer_key,
        slm=slm,
        slm_key=slm_key,
        calibration=calibration.value,
        calibration_path=calibration.path,
        science_context=science_context.value,
        science_context_path=science_context.path,
        pulse=None if external else pulse_resource,
        acquisition_logic=str(authored.get("acquisition_logic", "")),
        source_signal=source_signal,
        feedback_mode=str(authored["feedback_mode"]),
        exposure_seconds=None if external else authored["exposure_seconds"],
        shots_per_candidate=0 if external else int(authored["shots_per_candidate"]),
        probe_factors=() if external else tuple(float(value) for value in authored["probe_factors"]),
        feedback_gain=float(authored["feedback_gain"]),
        maximum_weight_change=float(authored["maximum_weight_change"]),
        max_updates=int(authored["max_updates"]),
        save_figure_artifact=save_figure_artifact,
    )


_PREVIEWS = (
        # The latest frame of the batch being shot, the site overlay in its record.
        NodePreviewSpec(
            CAMERA_FRAMES_OUTPUT,
            "image",
            semantic={
                "reduction": Reduction.MEAN,
            },
        ),
        NodePreviewSpec(OBSERVABLE_UNIFORMITY_HISTORY_OUTPUT, "curve"),
        NodePreviewSpec(
            SITE_SIGNAL_HISTORY_OUTPUT,
            "curve",
            semantic={
                AxisRef.point("slm_feedback.candidate").fate_field: "x",
                AxisRef.cell_data("calibration.site").fate_field: "group",
            },
        ),
        NodePreviewSpec(
            TARGET_SHARE_HISTORY_OUTPUT,
            "curve",
            semantic={
                AxisRef.point("slm_feedback.candidate").fate_field: "x",
                AxisRef.cell_data("calibration.site").fate_field: "group",
            },
        ),
)


def _previews(values, devices=None):
    """Frequency mode shoots no camera frames of its own."""

    del devices
    return _PREVIEWS[1:] if values.get("feedback_mode") == "ramsey_frequency" else _PREVIEWS


LOGIC_NODE = NodeType(
    "slm_feedback",
    NodeKind.TASK,
    SLM_FEEDBACK_SCHEMA,
    description=(
        "Adjust a saved SLM Science Context's target weights using camera readout or "
        "an external fitted-frequency signal to improve the selected per-site "
        "uniformity measure. Publishes candidate phase and convergence/site histories "
        "and saves the best measured Context; camera mode owns acquisition and Fire, "
        "while frequency mode requests the selected finite acquisition logic."
    ),
    input_specs=(
        DatasetInput("frequency", FIT_PARAMETER_CONTRACT, "latest", enabled_when=_EXTERNAL_MODE),
        CALIBRATION_INPUT,
        SCIENCE_CONTEXT_INPUT,
    ),
    # Camera mode's frames are this run's own output, one latest event per
    # shot with the site overlay in its record.
    outputs=(
        CANDIDATE_PHASE_OUTPUT,
        UNIFORMITY_HISTORY_OUTPUT,
        OBSERVABLE_UNIFORMITY_HISTORY_OUTPUT,
        SITE_SIGNAL_HISTORY_OUTPUT,
        TARGET_SHARE_HISTORY_OUTPUT,
        CAMERA_FRAMES_OUTPUT,
    ),
    node_previews=(),
    declare_previews=_previews,
    artifact_outputs=(
        ArtifactOutputSpec("artifact_path", SLM_PHASE_ARTIFACT_CONTRACT),
    ),
    device_requirements=(
        DeviceRequirement("camera.adapter", "camera", CAMERA_PROTECTED_FIELDS, enabled_when=_CAMERA_MODES),
        # The core's native board, driven through pulse_* (claimed, as v2 did).
        DeviceRequirement("sequencer.streamer", "sequencer", ("program",), enabled_when=_CAMERA_MODES),
        DeviceRequirement("slm.phase", "slm", SLM_PROTECTED_FIELDS),
    ),
    build=_build,
    workspace_resources=(_PULSE_RESOURCE,),
    acquisition_input="acquisition_logic",
    acquisition_completion="terminal",
)

__all__ = [
    "LOGIC_NODE",
    "SLM_FEEDBACK_SCHEMA",
]
