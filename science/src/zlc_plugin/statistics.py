"""Proportions' uncertainty by the native rule (docs/UNCERTAINTY.md); no Python copy.

A proportion of ``successes`` out of ``trials`` Bernoulli trials has the
68.27 % Wilson score interval as its interval and that interval's half-width
as its one symmetric standard uncertainty: neither is ever zero, and neither
leaves [0, 1] -- unlike ``sqrt(p (1 - p) / n)``, which is zero at 0 % and 100 %.
"""
from __future__ import annotations

import numpy as np

from .services import call


def _proportion(successes, trials) -> tuple[dict, tuple[int, ...]]:
    k, n = np.broadcast_arrays(np.asarray(successes, dtype="<f8"), np.asarray(trials, dtype="<f8"))
    answer = call("scientific.proportion", successes=np.ascontiguousarray(k.reshape(-1)),
                  trials=np.ascontiguousarray(n.reshape(-1)))
    return answer, k.shape


def proportion_interval(successes, trials) -> tuple[np.ndarray, np.ndarray]:
    """``(low, high)``: the Wilson interval of each proportion, shaped as the inputs."""
    answer, shape = _proportion(successes, trials)
    return np.asarray(answer["low"]).reshape(shape), np.asarray(answer["high"]).reshape(shape)


def proportion_uncertainty(successes, trials) -> np.ndarray:
    """The half-width of each proportion's Wilson interval, shaped as the inputs."""
    answer, shape = _proportion(successes, trials)
    return np.asarray(answer["uncertainty"]).reshape(shape)


__all__ = ["proportion_interval", "proportion_uncertainty"]
