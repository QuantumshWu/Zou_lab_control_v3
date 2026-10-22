"""The readout mathematics a calibration produces and every readout node reads.

``TrapCalibration.signals``/``detect`` is the single readout contract. The
Calibration task itself is its leaf ``logic_node.py``; importing this package
does not import the task or its device and pulse dependencies.
"""

from .bimodal import (
    BimodalFit,
    fit_bimodal,
    gaussian_fidelity,
    normal_cdf,
    optimal_gaussian_threshold,
    per_site_fidelity,
)
from .artifact import CALIBRATION_ARTIFACT_CODEC
from .calibration import (
    AtomDetection,
    CalibrationResult,
    DEFAULT_READOUT_MODEL_CHOICE,
    FrameContract,
    READOUT_MODEL_CHOICES,
    ReadoutModel,
    ReadoutModelKind,
    SiteMap,
    TrapCalibration,
    calibrate,
    classify_threshold,
    detect_sites,
    extract_box_signals,
    extract_psf_signals,
    readout_model_kind_from_choice,
)
from .psf import gaussian_psf_kernel, normalized_psf_kernel

__all__ = [
    "BimodalFit",
    "AtomDetection",
    "CalibrationResult",
    "CALIBRATION_ARTIFACT_CODEC",
    "DEFAULT_READOUT_MODEL_CHOICE",
    "FrameContract",
    "READOUT_MODEL_CHOICES",
    "ReadoutModel",
    "ReadoutModelKind",
    "SiteMap",
    "TrapCalibration",
    "calibrate",
    "classify_threshold",
    "detect_sites",
    "extract_box_signals",
    "extract_psf_signals",
    "fit_bimodal",
    "gaussian_fidelity",
    "gaussian_psf_kernel",
    "normal_cdf",
    "normalized_psf_kernel",
    "optimal_gaussian_threshold",
    "per_site_fidelity",
    "readout_model_kind_from_choice",
]
