"""What the SLM nodes share: their saved-artifact codecs and the SLM fields a run holds.

The rearrangement and feedback nodes read the same Science Context and Target
files and drive the same SLM, so these are said once. This package declares
no Logic Node.
"""

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from zlc_atom.devices.slm.solver import SCIENCE_CONTEXT_ARTIFACT_CONTRACT, load_science_context, load_target
from zlc_atom.nodes._framework.descriptor import ArtifactCodec, ArtifactInputSpec
from zlc_atom.nodes.calibration.artifact import CALIBRATION_ARTIFACT_CODEC

SCIENCE_CONTEXT_CODEC = ArtifactCodec(
    SCIENCE_CONTEXT_ARTIFACT_CONTRACT,
    "SLM Science Contexts (*.npz)",
    (".npz",),
    load_science_context,
)
TARGET_CODEC = ArtifactCodec("zlc.slm.target", "SLM Targets (*.json)", (".json",), load_target)

#: The saved files an SLM node starts from: the trap Calibration that reads
#: the camera, the Science Context the SLM shows, and an optional end Target.
CALIBRATION_INPUT = ArtifactInputSpec("calibration_path", "Calibration", CALIBRATION_ARTIFACT_CODEC,
                                      argument_name="calibration")
SCIENCE_CONTEXT_INPUT = ArtifactInputSpec("science_context_path", "SLM Science Context", SCIENCE_CONTEXT_CODEC,
                                          argument_name="science_context")
END_TARGET_INPUT = ArtifactInputSpec("end_target_path", "End Target (blank = grid)", TARGET_CODEC,
                                     required=False, argument_name="end_target")

#: The SLM settings a run holds for its length: the phase it commands and
#: the device facts that phase is built for.
SLM_PROTECTED_FIELDS = (
    "phase", "wavelength_nm", "display_name", "width", "height", "correction_path", "flip_x", "flip_y",
)



def plain_json(value: object) -> object:
    """Freeze the small metadata side of an SLM run's archive: text keys, plain
    numbers (a non-finite one is null), text, paths as text, lists."""

    if value is None or type(value) in (str, bool, int):
        return value
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("SLM run metadata keys must be text")
        return {key: plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain_json(item) for item in value]
    raise TypeError(f"SLM run metadata contains unsupported {type(value).__name__}")


__all__ = ["CALIBRATION_INPUT", "END_TARGET_INPUT", "SCIENCE_CONTEXT_CODEC", "SCIENCE_CONTEXT_INPUT", "SLM_PROTECTED_FIELDS",
           "TARGET_CODEC", "plain_json"]
