"""Image point overlays as data: the geometry document a producer pairs with
its per-point status signal, and the status vocabulary.

A producer of camera images (a calibration, an occupancy readout, an SLM
task) says WHERE its points are in the image's own coordinates and, as a
separate typed signal, whether each is occupied. Plots, figure archives and
the native renderer read the same document; it is data, so it lives here
and not in a plotting package.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum

import numpy as np

from .axis import SPATIAL_X, SPATIAL_Y, AxisSpec
from .value import OwnedSnapshot


class PointStatus(str, Enum):
    UNKNOWN = "unknown"
    EMPTY = "empty"
    OCCUPIED = "occupied"
    INVALID = "invalid"


#: The contract of a per-point status signal paired with an image.
IMAGE_POINT_OVERLAY_CONTRACT = "zlc_plot.image-point-overlay-status"
#: The run/event record key the geometry document travels under.
IMAGE_POINT_OVERLAY_GEOMETRY_RECORD = "image_point_overlay_geometry"

_GEOMETRY_FIELDS = frozenset({
    "coordinate_frame", "x_axis_id", "y_axis_id", "point_ids", "coordinates_xy", "labels",
    "status_axis_id", "status_coordinates",
})
_OPTIONAL_GEOMETRY_FIELDS = frozenset({"paths_xy", "static_statuses"})


def image_axes(snapshot: OwnedSnapshot) -> tuple[object, object]:
    """The image's (y, x) cell axes, which must share one coordinate frame."""

    axes = tuple(snapshot.block.schema.cell_domain.axes)
    if len(axes) != 2 or tuple(axis.role for axis in axes) != (SPATIAL_Y, SPATIAL_X):
        raise ValueError("image overlay geometry requires spatial-y, spatial-x")
    y_axis, x_axis = axes
    if y_axis.coordinate_frame != x_axis.coordinate_frame:
        raise ValueError("image overlay axes must share one coordinate frame")
    return y_axis, x_axis


def _axis_coordinates(axis: object, positions: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(tuple(axis.coordinate_at(index) for index in range(axis.size)), dtype=np.float64)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("image overlay axis coordinates must be finite")
    if coordinates.size == 1:
        if not np.all(np.asarray(positions) == 0):
            raise ValueError("a single-pixel image axis cannot infer an affine step")
        return np.full(np.asarray(positions).shape, coordinates[0])
    differences = np.diff(coordinates)
    if not np.allclose(differences, differences[0]):
        raise ValueError("image overlay axes must be affine")
    return float(coordinates[0]) + float(differences[0]) * positions


def point_paths(paths_xy: object, count: int) -> np.ndarray:
    """One ordered x/y polyline per point, shape ``(N, K, 2)`` with ``K >= 1``."""

    paths = np.asarray(paths_xy)
    if paths.ndim != 3 or paths.shape[0] != count or paths.shape[1] < 1 or paths.shape[2] != 2:
        raise ValueError("paths_xy must have shape (N, K, 2), K >= 1")
    if paths.dtype.kind not in "biuf":
        raise TypeError("paths_xy must be numeric")
    paths = np.asarray(paths, dtype=np.float64)
    if not np.all(np.isfinite(paths)):
        raise ValueError("paths_xy must be finite")
    return paths


def static_point_statuses(statuses: object, count: int) -> tuple[PointStatus, ...]:
    """A fixed judgement per point (markers with no run axes)."""

    values = tuple(statuses)
    if len(values) != count:
        raise ValueError("static_statuses must have one item per coordinate")
    if any(not isinstance(status, PointStatus) for status in values):
        raise TypeError("static_statuses must contain PointStatus values")
    return values


def image_point_overlay_geometry(
    image: OwnedSnapshot,
    coordinates_xy: object,
    point_ids: object,
    *,
    status_axis: AxisSpec,
    labels: object | None = None,
    coordinates_are_indices: bool = False,
    paths_xy: object | None = None,
    static_statuses: object | None = None,
) -> dict[str, object]:
    """Build the strict plain geometry document paired with an overlay signal."""

    if not isinstance(image, OwnedSnapshot):
        raise TypeError("overlay image must be an OwnedSnapshot")
    if not isinstance(status_axis, AxisSpec):
        raise TypeError("overlay status_axis must be an AxisSpec")
    if type(coordinates_are_indices) is not bool:
        raise TypeError("coordinates_are_indices must be bool")
    ids = tuple(str(value).strip() for value in tuple(point_ids))
    if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("overlay point ids must be unique non-empty text")
    centers = np.asarray(coordinates_xy, dtype=np.float64)
    if centers.shape != (len(ids), 2) or not np.all(np.isfinite(centers)):
        raise ValueError("overlay coordinates must have finite shape (points, 2)")
    resolved_labels = (
        tuple(str(index) for index in range(1, len(ids) + 1))
        if labels is None else tuple(str(value).strip() for value in tuple(labels))
    )
    if len(resolved_labels) != len(ids) or any(not value for value in resolved_labels):
        raise ValueError("overlay labels must match point ids")
    if int(status_axis.size) != len(ids):
        raise ValueError("overlay status_axis must match point ids")
    y_axis, x_axis = image_axes(image)
    if coordinates_are_indices:
        centers = np.column_stack((_axis_coordinates(x_axis, centers[:, 0]), _axis_coordinates(y_axis, centers[:, 1])))
    document = {
        "coordinate_frame": None if x_axis.coordinate_frame is None else str(x_axis.coordinate_frame),
        "x_axis_id": str(x_axis.axis_id),
        "y_axis_id": str(y_axis.axis_id),
        "point_ids": list(ids),
        "coordinates_xy": centers.tolist(),
        "labels": list(resolved_labels),
        "status_axis_id": str(status_axis.axis_id),
        "status_coordinates": [status_axis.coordinate_at(index) for index in range(status_axis.size)],
    }
    if paths_xy is not None:
        paths = point_paths(paths_xy, len(ids))
        if coordinates_are_indices:
            paths = np.stack((_axis_coordinates(x_axis, paths[..., 0]), _axis_coordinates(y_axis, paths[..., 1])), axis=-1)
        document["paths_xy"] = paths.tolist()
    if static_statuses is not None:
        document["static_statuses"] = [status.value for status in static_point_statuses(static_statuses, len(ids))]
    return document


def image_point_overlay_geometry_matches(image: OwnedSnapshot, geometry: Mapping[str, object]) -> bool:
    """Whether shared run geometry belongs to this image's coordinates."""

    try:
        y_axis, x_axis = image_axes(image)
    except ValueError:
        return False
    frame = None if x_axis.coordinate_frame is None else str(x_axis.coordinate_frame)
    return (geometry.get("coordinate_frame") == frame and geometry.get("x_axis_id") == str(x_axis.axis_id)
            and geometry.get("y_axis_id") == str(y_axis.axis_id))


def validated_overlay_geometry(
    geometry: object, image: OwnedSnapshot, status: OwnedSnapshot,
) -> tuple[tuple[str, ...], np.ndarray, tuple[str, ...]]:
    """The point ids, x/y centres and labels of a geometry document, checked
    against the image it annotates and the status signal it pairs with."""

    if not isinstance(geometry, Mapping):
        raise TypeError("image overlay geometry must be a mapping")
    if not _GEOMETRY_FIELDS <= set(geometry) or set(geometry) - _GEOMETRY_FIELDS - _OPTIONAL_GEOMETRY_FIELDS:
        raise ValueError("image overlay geometry fields are not canonical")
    ids = tuple(str(value).strip() for value in tuple(geometry["point_ids"]))
    labels = tuple(str(value).strip() for value in tuple(geometry["labels"]))
    if (not ids or any(not value for value in ids) or len(set(ids)) != len(ids)
            or len(labels) != len(ids) or any(not value for value in labels)):
        raise ValueError("image overlay geometry identities are invalid")
    centers = np.asarray(geometry["coordinates_xy"], dtype=np.float64)
    if centers.shape != (len(ids), 2) or not np.all(np.isfinite(centers)):
        raise ValueError("image overlay geometry coordinates are invalid")
    if not image_point_overlay_geometry_matches(image, geometry):
        raise ValueError("image overlay geometry differs from image axes")
    image_schema, status_schema = image.block.schema, status.block.schema
    if (status_schema.repeat_domain != image_schema.repeat_domain
            or status_schema.point_domain != image_schema.point_domain):
        raise ValueError("image overlay status leading geometry differs from image")
    status_axes = tuple(status_schema.cell_domain.axes)
    if len(status_axes) != 1:
        raise ValueError("image overlay status must declare one trailing axis")
    status_axis = status_axes[0]
    status_coordinates = tuple(status_axis.coordinate_at(index) for index in range(status_axis.size))
    if (str(geometry["status_axis_id"]) != str(status_axis.axis_id)
            or tuple(geometry["status_coordinates"]) != status_coordinates):
        raise ValueError("image overlay geometry differs from status axis")
    return ids, centers, labels


__all__ = [
    "IMAGE_POINT_OVERLAY_CONTRACT",
    "IMAGE_POINT_OVERLAY_GEOMETRY_RECORD",
    "PointStatus",
    "image_axes",
    "image_point_overlay_geometry",
    "image_point_overlay_geometry_matches",
    "point_paths",
    "static_point_statuses",
    "validated_overlay_geometry",
]
