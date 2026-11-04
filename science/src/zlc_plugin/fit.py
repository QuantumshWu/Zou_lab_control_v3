"""A run's fits, by the core's fit engine (PLUGIN_PROTOCOL.md section 5, ``run.fit``).

Python holds no model table: ``fit_models`` reads the core's catalogue once per process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

_MODELS: dict[str, Mapping[str, Any]] | None = None


def fit_models(ask: Callable[..., Any] | None = None) -> Mapping[str, Mapping[str, Any]]:
    """The series fit models by id: ``{id, label, parameters: [{name, unit_relation}]}``.

    ``ask`` is how to reach the core: ``RunContext.core`` in a run, omitted in a draft query.
    """

    global _MODELS
    if _MODELS is None:
        if ask is None:
            from .host import core as ask
        _MODELS = {row["id"]: row for row in ask("fit_models")["models"]}
    return _MODELS


@dataclass(frozen=True)
class FitAnswer:
    """One row per cell. A failed cell has NaN ``values`` and ``errors``, ``success``
    False and its reason in ``messages``; a parameter the data do not determine has
    a NaN error in a successful cell."""

    parameters: tuple[str, ...]
    values: np.ndarray
    errors: np.ndarray
    success: np.ndarray
    reduced_chi_square: np.ndarray
    messages: tuple[str, ...]


def fit_request(model_id: str, x: Any, y: Any, sigma: Any = None, warm: Any = None) -> dict[str, Any]:
    """The ``run.fit`` request: ``y`` ``(cells, n)`` over ``x`` ``(n)``, ``sigma`` as ``y``,
    ``warm`` ``(cells, parameters)``."""

    x, y = np.asarray(x, dtype="<f8"), np.asarray(y, dtype="<f8")
    if x.ndim != 1 or y.ndim != 2 or y.shape[1] != x.size:
        raise ValueError(f"a fit takes y (cells, n) over x (n), not {y.shape} over {x.shape}")
    if sigma is not None:
        sigma = np.asarray(sigma, dtype="<f8")
        if sigma.shape != y.shape:
            raise ValueError(f"sigma has shape {sigma.shape}, not y's {y.shape}")
    if warm is not None:
        warm = np.asarray(warm, dtype="<f8")
        if warm.ndim != 2 or warm.shape[0] != y.shape[0]:
            raise ValueError(f"warm has shape {warm.shape}, not (cells, parameters) for {y.shape[0]} cells")
    return {"model": str(model_id), "x": x, "y": y, "sigma": sigma, "warm": warm}


def fit_answer(reply: Mapping[str, Any]) -> FitAnswer:
    """The core's reply as arrays (null is NaN there)."""

    parameters = tuple(reply["parameters"])
    cells = len(reply["success"])
    shape = (cells, len(parameters))
    return FitAnswer(parameters, np.asarray(reply["values"], dtype=np.float64).reshape(shape),
                     np.asarray(reply["errors"], dtype=np.float64).reshape(shape),
                     np.asarray(reply["success"], dtype=bool).reshape(cells),
                     np.asarray(reply["reduced_chi_square"], dtype=np.float64).reshape(cells), tuple(reply["messages"]))


__all__ = ["FitAnswer", "fit_answer", "fit_models", "fit_request"]
