"""The phase-only SLM contract and its phase mathematics; each SLM is a folder beside this one."""

from .device import CONCURRENT_SLM_METHODS, SlmAdapter, opened_slm
from .phase import canonical_phase

__all__ = ["CONCURRENT_SLM_METHODS", "SlmAdapter", "canonical_phase", "opened_slm"]
