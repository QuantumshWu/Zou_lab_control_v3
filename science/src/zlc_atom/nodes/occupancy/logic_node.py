"""Discoverable occupancy processor descriptor."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.nodes._framework.descriptor import ArtifactInputSpec, ResolvedArtifact
from zlc_atom.nodes._framework.frames import FrameClaim
from zlc_atom.nodes.calibration import (
    CALIBRATION_ARTIFACT_CODEC,
    DEFAULT_READOUT_MODEL_CHOICE,
    READOUT_MODEL_CHOICES,
    TrapCalibration,
    readout_model_kind_from_choice,
)
from zlc_plugin import DatasetInput, NodeKind, NodeType

from .editor import calibrations_facts, post_selection_facts
from .post_selection import POST_SELECTION, POST_SELECTION_FIELDS, condition, post_select_step
from .processor import OCCUPANCY_OUTPUTS, CalibrationReading, OccupancyProcessor


OCCUPANCY_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "model_kind",
            "choice",
            "Readout model",
            DEFAULT_READOUT_MODEL_CHOICE,
            choices=READOUT_MODEL_CHOICES,
            segmented=True,
            description="Default reads each file with its calibration's own default model",
        ),
        *POST_SELECTION_FIELDS,
    ),
    validator=condition,
)

#: Each calibration claims the frames it reads -- frame numbers from 0,
#: first, last, rest -- so a load frame and a readout frame taken under
#: different exposures are read with calibrations trained on each.
CALIBRATIONS_INPUT = ArtifactInputSpec(
    "calibrations",
    "Calibrations",
    CALIBRATION_ARTIFACT_CODEC,
    frame_claims=True,
)


def _reading(claim: object) -> CalibrationReading:
    artifact = getattr(claim, "value", None)
    if (
        not isinstance(claim, FrameClaim)
        or not isinstance(artifact, ResolvedArtifact)
        or artifact.contract_id != CALIBRATION_ARTIFACT_CODEC.contract_id
        or not isinstance(artifact.value, TrapCalibration)
    ):
        raise TypeError("occupancy calibrations must claim frames with resolved calibration artifacts")
    return CalibrationReading(artifact.value, claim.frames, artifact.path)


def _build(*, calibrations: object, source_signal: str, **values: object) -> OccupancyProcessor:
    authored = OCCUPANCY_SCHEMA.project_values(values)
    selected_source = str(source_signal).strip()
    if not selected_source:
        raise ValueError("source_signal must be non-empty")
    claims = CALIBRATIONS_INPUT.resolve_value(calibrations)
    # A row claiming no frame is kept, not read: only its file goes on, for
    # the run record -- whether or not that file could be decoded.
    return OccupancyProcessor(
        tuple(_reading(claim) for claim in claims if claim.frames),
        parked=tuple(
            None if claim.value is None else claim.value.path
            for claim in claims if not claim.frames
        ),
        source_signal=selected_source,
        model_kind=readout_model_kind_from_choice(authored["model_kind"]),
        **post_select_step(condition(authored)),
    )


def _editor_facts(values, resources, devices, source, bundle, artifacts):
    """The calibrations grid over the frames the source carries, then the
    post-selection section (occupancy.editor)."""

    del resources, devices, bundle
    first = calibrations_facts(values, artifacts, source, codec=CALIBRATION_ARTIFACT_CODEC, label=CALIBRATIONS_INPUT.label)
    try:
        drafted = OCCUPANCY_SCHEMA.draft_values(values)
    except (TypeError, ValueError):
        drafted = {field.name: values.get(field.name, field.default) for field in OCCUPANCY_SCHEMA.fields}
    second = post_selection_facts(drafted, source)
    return {"editor": "stack", "manages": [*first["manages"], *second["manages"]],
            "manages_artifacts": first["manages_artifacts"], "editors": [first, second]}


def _describe_output_frames(values, output_name, source):
    """Every Occupancy output preserves its camera's readout-event axis, name and labels."""
    return source if any(item.name == output_name for item in OCCUPANCY_OUTPUTS) else None


LOGIC_NODE = NodeType(
    "occupancy",
    NodeKind.PROCESSOR,
    OCCUPANCY_SCHEMA,
    description=(
        "Read camera image data using saved Calibrations and publish per-site counts, "
        "occupied verdicts and frame_judged diagnostics while preserving frame and scan "
        "axes. Each calibration claims the frames it reads: 'first' (frame 0), 'last', "
        "positive numbers for middle frames between them (a middle number the cycle no "
        "longer has between its ends is absent: kept, noted and reading nothing, never the "
        "last frame), and 'rest' (every frame no other calibration names); one calibration "
        "with ['rest'] reads every frame. A calibration with frames [] is kept but not read, "
        "and its file is not checked against the others. Use occupied for loading-rate "
        "plots, camera overlays or Frame Survival; the processor follows the source and "
        "resumes when compatible data returns. post_selection (off by default) keeps counts and "
        "occupied valid only where a condition on two frames of the cycle holds for that shot and "
        "site (on every frame); with its defaults it equals Occupancy followed by Derive same = "
        "occupied[frame first] == occupied[frame last] (internal), kept = occupied where same. "
        "frame_a and frame_b are 'first' (default for A), 'last' (default for B) or a middle frame "
        "number, as the frame claims name frames. mode is either 'relation' (default): kept where "
        "occupied at frame_a <relation> occupied at frame_b, relation one of Derive's '==' "
        "(default), '!=', '>', '<', '>=', '<=' on booleans false < true ('>' keeps loaded at A and "
        "empty at B), both frames judged and different; or 'values': kept where each frame's verdict "
        "is value_a / value_b, 'true' (1), 'false' (0) or 'any' (not asked; its frame need not be "
        "judged), at least one not 'any' -- e.g. value_a 'true', value_b 'false' keeps loaded at A "
        "and empty at B. The other mode's fields stay in the draft unused, as do all of them while "
        "post_selection is off. The condition is evaluated by the Derive engine; frame_judged stays "
        "the evidence frames (pixels have no site). With post_selection on, a cycle of fewer than two "
        "frames, or a chosen middle frame the cycle does not have, is refused and the processor waits."
    ),
    input_specs=(
        DatasetInput("frames", None, "exact"),
        CALIBRATIONS_INPUT,
    ),
    outputs=OCCUPANCY_OUTPUTS,
    describe_output_frames=_describe_output_frames,
    editor_facts=_editor_facts,
    build=_build,
    replaces={"post_selected_occupancy": {POST_SELECTION.name: True}},
)

__all__ = ["CALIBRATIONS_INPUT", "LOGIC_NODE", "OCCUPANCY_SCHEMA"]
