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


#: The switch: off, frames are counted and no sequencer is asked for; on, a lane must be chosen.
TRIGGERED = "trigger_by_channel"
#: While the switch is on (a field's or a device requirement's ``enabled_when``).
TRIGGER_ON = (TRIGGERED, (True,))
#: The draft field naming the sequencer lane that triggers the camera (a raw lane, ``ch11``).
TRIGGER_CHANNEL = "trigger_channel"
#: The build argument of the device that states the lanes and says what each plays.
TRIGGER_SOURCE = "trigger_source"


def trigger_lanes(devices: object) -> list[dict]:
    """The lanes the draft's trigger source states: ``{lane, port, label, frames}`` each."""

    source = (devices or {}).get(TRIGGER_SOURCE) if isinstance(devices, dict) else None
    lanes = (getattr(source, "facts", None) or {}).get("trigger_lanes") or []
    return [dict(lane) for lane in lanes]


def trigger_frames(values: object, devices: object) -> int | None:
    """How many frames one shot of the chosen lane makes; None while the board holds no pulse, or no lane is chosen."""

    chosen = str((values or {}).get(TRIGGER_CHANNEL) or "").strip() if (values or {}).get(TRIGGERED) else ""
    for lane in trigger_lanes(devices) if chosen else ():
        if lane.get("lane") == chosen:
            return None if lane.get("frames") is None else int(lane["frames"])
    return None


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
        AuthoringField("frames_per_cycle", "int", "Frames per cycle", 1, minimum=1,
                       # A cycle's size is the trigger lane's to say while the switch is on.
                       enabled_when=(TRIGGERED, (False,)),
                       description="Typed while Trigger by channel is off; on, it is the chosen lane's triggers in "
                                   "one shot, shown here once the board holds its pulse."),
        AuthoringField(
            TRIGGERED, "bool", "Trigger by channel", False,
            description=(
                "Off: frames are counted, as the number above says, and no sequencer is needed. On: a sequencer "
                "lane must be chosen below, and each frame is the trigger it arrives at."
            ),
        ),
        AuthoringField(
            TRIGGER_CHANNEL, "str", "Trigger channel", None, required=True, enabled_when=TRIGGER_ON,
            description=(
                "The sequencer lane that triggers this camera: each frame is the trigger it arrives at, a cycle is "
                "the lane's triggers of one shot, a frame at no trigger's time is dropped, and a trigger with no "
                "frame leaves its shot marked invalid."
            ),
        ),
        photoelectron_switch(),
    ),
)


__all__ = ["CAMERA_MEASUREMENT_SCHEMA", "IMAGE_AREA_TO_ROI", "ROI_FIELD_NAMES", "TRIGGERED", "TRIGGER_CHANNEL", "TRIGGER_ON", "TRIGGER_SOURCE",
           "trigger_frames", "trigger_lanes"]
