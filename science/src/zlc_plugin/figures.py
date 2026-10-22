"""A run's saved figures, described as data for the core to write and draw.

Plotting and figure archives are native: a plugin never loads a plotting
package. It says what a figure IS -- its dataset, the plot that shows it
(a small spec tree: kind, axes, labels) and any point overlay -- and
``RunContext.save_figure`` hands that to the core, which completes the plot
recipe with the kind's defaults, writes the strict ``zlc.figure`` archive and
renders its image (PLUGIN_PROTOCOL.md section 5, ``run.figure``).

The spec vocabulary below writes exactly the tree a figure archive's recipe
stores under ``spec``; the names follow the plotting package's so a figure
reads the same in both.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

class AxisDomain(str, Enum):
    """The three domains a dataset's axes live in."""

    REPEAT = "repeat"
    POINT = "point"
    CELL_DATA = "cell_data"


#: The fate-field prefix of a preview's semantic assignment (``fate:<domain>:<axis_id>``).
FATE_PREFIX = "fate:"


@dataclass(frozen=True)
class AxisRef:
    """One producer-declared axis as a plot refers to it."""

    domain: AxisDomain
    axis_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", AxisDomain(self.domain))
        if not isinstance(self.axis_id, str) or not self.axis_id.strip():
            raise ValueError(f"{self.domain.value} requires a non-empty axis_id")
        object.__setattr__(self, "axis_id", self.axis_id.strip())

    @classmethod
    def repeat(cls, axis_id: str) -> "AxisRef":
        return cls(AxisDomain.REPEAT, axis_id)

    @classmethod
    def point(cls, axis_id: str) -> "AxisRef":
        return cls(AxisDomain.POINT, axis_id)

    @classmethod
    def cell_data(cls, axis_id: str) -> "AxisRef":
        return cls(AxisDomain.CELL_DATA, axis_id)

    def tree(self) -> dict[str, str]:
        return {"domain": self.domain.value, "axis_id": self.axis_id}

    @property
    def fate_field(self) -> str:
        """This axis's key in a preview's semantic assignment."""

        return f"{FATE_PREFIX}{self.domain.value}:{self.axis_id}"


class Reduction(str, Enum):
    """How samples that share a plotted coordinate are combined."""

    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    FIRST = "first"
    LAST = "last"


@dataclass(frozen=True)
class PlotLabels:
    title: str | None = None
    x: str | None = None
    y: str | None = None
    value: str | None = None

    def tree(self) -> dict[str, str | None]:
        return {"title": self.title, "x": self.x, "y": self.y, "value": self.value}


def _axis(ref: AxisRef | None) -> dict[str, str] | None:
    return None if ref is None else ref.tree()


@dataclass(frozen=True)
class _Spec:
    def _common(self) -> dict[str, Any]:
        return {"kind": self.kind, "labels": self.labels.tree(),
                "scope": [{"axis": axis.tree(), "coordinate": {"kind": "value", "value": coordinate}}
                          for axis, coordinate in self.scope],
                "coordinates": [ref.tree() for ref in self.coordinates]}


@dataclass(frozen=True)
class CurvePlot(_Spec):
    x: AxisRef
    group: AxisRef | None = None
    reduction: Reduction = Reduction.MEAN
    labels: PlotLabels = field(default_factory=PlotLabels)
    scope: tuple[tuple[AxisRef, object], ...] = ()
    coordinates: tuple[AxisRef, ...] = ()
    kind = "curve"

    def tree(self) -> dict[str, Any]:
        return {**self._common(), "reduction": Reduction(self.reduction).value, "group": _axis(self.group),
                "x": self.x.tree()}


@dataclass(frozen=True)
class ImagePlot(_Spec):
    x: AxisRef
    y: AxisRef
    reduction: Reduction = Reduction.MEAN
    labels: PlotLabels = field(default_factory=PlotLabels)
    scope: tuple[tuple[AxisRef, object], ...] = ()
    coordinates: tuple[AxisRef, ...] = ()
    kind = "image"

    def tree(self) -> dict[str, Any]:
        return {**self._common(), "reduction": Reduction(self.reduction).value, "x": self.x.tree(), "y": self.y.tree()}


@dataclass(frozen=True)
class HistogramPlot(_Spec):
    reduction: Reduction = Reduction.MEAN
    reduced: tuple[AxisRef, ...] = ()
    labels: PlotLabels = field(default_factory=PlotLabels)
    scope: tuple[tuple[AxisRef, object], ...] = ()
    coordinates: tuple[AxisRef, ...] = ()
    group: AxisRef | None = None
    kind = "histogram"

    def tree(self) -> dict[str, Any]:
        return {**self._common(), "reduction": Reduction(self.reduction).value,
                "reduced": [ref.tree() for ref in self.reduced], "group": _axis(self.group)}


@dataclass(frozen=True)
class FacetGridPlot(_Spec):
    facet: AxisRef | None
    cell: CurvePlot | ImagePlot | HistogramPlot
    labels: PlotLabels = field(default_factory=PlotLabels)
    scope: tuple[tuple[AxisRef, object], ...] = ()
    coordinates: tuple[AxisRef, ...] = ()
    kind = "facet_grid"

    def tree(self) -> dict[str, Any]:
        return {**self._common(), "facet": _axis(self.facet), "cell": self.cell.tree()}


@dataclass(frozen=True)
class ImagePointOverlay:
    """Points drawn over an image figure: x/y centres in the image's own
    coordinates, their ids and labels, and either a fixed judgement per point
    (``static_statuses``, zlc_data.point_overlay.PointStatus) or a measured
    status dataset (``status``); ``paths_xy`` optionally draws each point's path."""

    revision: int
    coordinates: object
    point_ids: tuple[str, ...] | None = None
    labels: tuple[str | None, ...] | None = None
    static_statuses: tuple[object, ...] | None = None
    status: object | None = None
    paths_xy: object | None = None

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1:] != (2,) or not np.all(np.isfinite(coordinates)):
            raise ValueError("overlay coordinates must be finite with shape (N, 2)")
        if self.static_statuses is not None and self.status is not None:
            raise ValueError("static statuses and a status dataset are mutually exclusive")
        object.__setattr__(self, "coordinates", coordinates)

    def wire(self) -> dict[str, Any]:
        return {
            "revision": int(self.revision), "coordinates": self.coordinates,
            "point_ids": None if self.point_ids is None else list(self.point_ids),
            "labels": None if self.labels is None else list(self.labels),
            "static_statuses": (None if self.static_statuses is None
                                else [getattr(status, "value", status) for status in self.static_statuses]),
            "status": None if self.status is None else dataset_to_wire(self.status),
            "paths_xy": None if self.paths_xy is None else np.asarray(self.paths_xy, dtype=np.float64),
        }


@dataclass(frozen=True)
class ImageFrame:
    """An image dataset and the points drawn over it, saved as one figure."""

    snapshot: object
    overlay: ImagePointOverlay


def image_point_overlay_from_signal(geometry: Mapping[str, Any], status: object, image: object, *,
                                    revision: int) -> ImagePointOverlay:
    """The overlay a run's geometry document and its status signal describe on this image."""

    from zlc_data.point_overlay import validated_overlay_geometry

    ids, coordinates, labels = validated_overlay_geometry(geometry, image, status)
    return ImagePointOverlay(revision=revision, coordinates=coordinates, point_ids=ids, labels=labels,
                             status=status, paths_xy=geometry.get("paths_xy"))


# ------------------------------------------------------------------- the wire

def dataset_to_wire(snapshot: object) -> dict[str, Any]:
    """One complete dataset as the protocol carries it: schema tree, planes, identity."""

    from zlc_data.codec import dataset_schema_to_tree

    block = snapshot.materialize().block if hasattr(snapshot, "materialize") else snapshot.block
    values, validity, sigma = block.as_segment()
    return {"schema": dataset_schema_to_tree(block.schema),
            "data": {"values": values, "validity": validity, "sigma": sigma},
            "block_id": str(block.block_id), "revision": int(block.revision.value),
            "generation": str(snapshot.ref.stream_generation.value)}


def dataset_from_wire(wire: Mapping[str, Any]) -> object:
    """The zlc_data snapshot ``dataset_to_wire`` describes."""

    from zlc_data import INVALID, VALID, CellValidity, DatasetComponentValidity, ValidityMode
    from zlc_data import owned_snapshot_from_arrays
    from zlc_data.codec import dataset_schema_from_tree

    schema = dataset_schema_from_tree(wire["schema"])
    data = wire["data"]
    validity = data["validity"]
    if validity is True or validity is False:
        validity = VALID if validity else INVALID
    else:
        # The storage mask as the dataset held it: an all-true mask stays a mask.
        contract = schema.value_schema.validity_contract
        mask = np.asarray(validity, dtype=np.bool_)
        validity = (DatasetComponentValidity(contract.component_axis_ids, mask)
                    if contract.mode is ValidityMode.COMPONENTS else CellValidity(mask))
    return owned_snapshot_from_arrays(schema, data["values"], int(wire["revision"]),
                                      validity=validity, sigma=data["sigma"], block_id=str(wire["block_id"]),
                                      stream_generation=str(wire["generation"]))


#: The parts of a plot recipe besides spec, parameters and size a figure may state.
RECIPE_PARTS = ("viewport", "classifier_thresholds", "facet_focus", "fit", "selectors", "interaction", "presentation")


def figure_request(path: object, *, plot_input: object, spec: _Spec, parameters: Mapping[str, Any] | None,
                   size: str, source: Mapping[str, Any] | None, lineage: Mapping[str, Any] | None,
                   image: bool, configuration: Mapping[str, Any]) -> dict[str, Any]:
    """The ``run.figure`` request for one figure."""

    unknown = set(configuration) - set(RECIPE_PARTS)
    if unknown:
        raise TypeError(f"a figure recipe has no part {sorted(unknown)}")
    if not isinstance(spec, _Spec):
        raise TypeError("a figure's spec must be one of zlc_plugin.figures' plots")
    frame = plot_input if isinstance(plot_input, ImageFrame) else None
    snapshot = plot_input.snapshot if frame is not None else plot_input
    recipe = {"spec": spec.tree(), "parameters": dict(parameters or {}), "size": str(size)}
    for name, value in configuration.items():
        recipe[name] = _plain(value)
    return {"path": str(path), "data": dataset_to_wire(snapshot),
            "overlay": None if frame is None else frame.overlay.wire(), "recipe": recipe,
            "source": None if source is None else _plain(source),
            "lineage": None if lineage is None else _plain(lineage), "image": bool(image)}


def _plain(value: object) -> object:
    """Mappings and sequences as JSON objects and arrays (AxisRefs as their trees)."""

    if isinstance(value, AxisRef):
        return value.tree()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    return value


__all__ = [
    "AxisDomain",
    "AxisRef",
    "CurvePlot",
    "FacetGridPlot",
    "HistogramPlot",
    "ImageFrame",
    "ImagePlot",
    "ImagePointOverlay",
    "PlotLabels",
    "RECIPE_PARTS",
    "Reduction",
    "dataset_from_wire",
    "dataset_to_wire",
    "figure_request",
    "image_point_overlay_from_signal",
]
