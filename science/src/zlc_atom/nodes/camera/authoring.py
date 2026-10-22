"""Camera-measurement authoring and the image-area -> ROI selection mapping.

These are node facts (what a camera run asks for, and how a box drawn on its
frames becomes its ROI), shared by the camera and rearrangement measurements.
"""

from collections.abc import Mapping
import math
from numbers import Integral

from zlc_data import SPATIAL_X, SPATIAL_Y
from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.camera.contract import EXPOSURE_STEP_SECONDS
from zlc_atom.devices.camera.photoelectrons import photoelectron_switch
from zlc_atom.nodes._framework.descriptor import SelectionMapping


#: Each blank keeps the camera's current value for that one field: a
#: measurement that only changes the ROI height leaves x, y and width where
#: the camera has them (``CameraMeasurementNode`` completes the rectangle).
ROI_FIELD_NAMES = ("roi_x", "roi_y", "roi_width", "roi_height")


def _spatial_range(selection: object, role: str) -> object | None:
    matches = tuple(
        value
        for value in selection.ranges
        if value.axis == role or value.axis.endswith(f".{role}")
    )
    return matches[0] if len(matches) == 1 else None


def _selected_sensor_interval(
    value: object,
    *,
    binning: int,
    sensor_size: int,
) -> tuple[int, int] | None:
    # A frame's axes carry the sensor pixels it covers, so a region drawn on
    # it is already stated in sensor pixels and needs no conversion: it is the
    # ROI.  Producer ROI follows the actual canvas rectangle, including
    # viewport or Area bounds outside the current frame.  Only the physical
    # sensor clips it; data-derived ROI/Fit signals keep their separate
    # data-domain slicing.  Each coordinate names the FIRST sensor pixel of
    # its (possibly binned) sample, so the selected rectangle ends one whole
    # sample past the last.
    start = max(0, math.ceil(value.lower))
    stop = min(sensor_size, math.floor(value.upper) + binning)
    if stop <= start:
        # A region drawn entirely off the sensor -- in the band beside the
        # picture, or past its edge -- names no crop: an answer, not a fault.
        return None
    return start, stop


def _positive_pair(context: Mapping[str, object], name: str) -> tuple[int, int]:
    """One of the camera's current facts the core supplies: ``sensor_shape_yx`` or ``binning_yx``."""

    raw = context.get(name)
    if raw is None:
        raise ValueError(f"camera image selection requires current {name}")
    try:
        values = tuple(raw)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{name} must contain two positive integers") from error
    if (
        len(values) != 2
        or any(isinstance(value, bool) or not isinstance(value, Integral) for value in values)
        or any(int(value) <= 0 for value in values)
    ):
        raise ValueError(f"{name} must contain two positive integers")
    return int(values[0]), int(values[1])


def _image_area_to_roi_patch(
    selection: object,
    draft: Mapping[str, object],
    context: Mapping[str, object],
) -> dict[str, int] | None:
    x_range = _spatial_range(selection, SPATIAL_X.value)
    y_range = _spatial_range(selection, SPATIAL_Y.value)
    if x_range is None or y_range is None:
        # An image whose axes are not the sensor plane -- a frame or a site
        # put on one of them -- has an Area that names no crop.
        return None
    del draft
    sensor_height, sensor_width = _positive_pair(context, "sensor_shape_yx")
    binning_y, binning_x = _positive_pair(context, "binning_yx")
    horizontal = _selected_sensor_interval(
        x_range,
        binning=binning_x,
        sensor_size=sensor_width,
    )
    vertical = _selected_sensor_interval(
        y_range,
        binning=binning_y,
        sensor_size=sensor_height,
    )
    if horizontal is None or vertical is None:
        return None
    x_start, x_stop = horizontal
    y_start, y_stop = vertical
    return {
        "roi_x": x_start,
        "roi_y": y_start,
        "roi_width": x_stop - x_start,
        "roi_height": y_stop - y_start,
    }


IMAGE_AREA_TO_ROI = SelectionMapping(
    plot_kind="image",
    selector_kind="area",
    draft_fields=ROI_FIELD_NAMES,
    map_patch=_image_area_to_roi_patch,
)


CAMERA_MEASUREMENT_SCHEMA = AuthoringSchema(
    (
        # Blank keeps the camera's current setting.
        AuthoringField(
            "exposure_seconds", "float", "Exposure seconds", None, required=False, minimum=1e-9,
            step=EXPOSURE_STEP_SECONDS,
            description="Blank keeps the camera's current exposure.",
            blank_label="camera's current",
        ),
        AuthoringField("roi_x", "int", "ROI x", None, required=False, minimum=0,
                       blank_label="camera's current"),
        AuthoringField("roi_y", "int", "ROI y", None, required=False, minimum=0,
                       blank_label="camera's current"),
        AuthoringField("roi_width", "int", "ROI width", None, required=False, minimum=1,
                       blank_label="camera's current"),
        AuthoringField("roi_height", "int", "ROI height", None, required=False, minimum=1,
                       blank_label="camera's current"),
        AuthoringField("repeat", "int", "Repeat", 0, minimum=0),
        AuthoringField("frames_per_cycle", "int", "Frames per cycle", 1, minimum=1),
        photoelectron_switch(),
    ),
)


__all__ = ["CAMERA_MEASUREMENT_SCHEMA", "IMAGE_AREA_TO_ROI", "ROI_FIELD_NAMES"]
