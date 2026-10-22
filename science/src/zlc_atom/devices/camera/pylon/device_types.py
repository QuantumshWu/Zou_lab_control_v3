"""The Basler this bench can install, and how to find one."""

from __future__ import annotations

from typing import Any, Mapping

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices import discovered
from zlc_atom.devices.camera import EXPOSURE_STEP_SECONDS
from zlc_atom.devices.camera.roi_grid import ROI_FIELDS, authored_roi_xywh
from zlc_plugin import DeviceType

from .adapter import PylonCameraAdapter, PylonCameraConfig


#: A Basler: which one by serial, how long, and which external line gates finite
#: acquisition.  Mono8 and the SDK timeout are adapter policy, not authoring.
PYLON_CAMERA_SCHEMA = AuthoringSchema(
    (
        AuthoringField("serial", "str", "Serial number", "", required=True),
        AuthoringField(
            "exposure_seconds", "float", "Exposure seconds", 0.1,
            minimum=1e-9, step=EXPOSURE_STEP_SECONDS,
        ),
        # No bounds written down: the sensor's own gain limits differ by model
        # and pixel format, and the camera refuses what it cannot do.
        AuthoringField("gain_db", "float", "Gain (dB)", 0.0),
        AuthoringField("trigger_source", "str", "Trigger source", "Line1"),
        *ROI_FIELDS,
        # Unset by default, which is the honest answer for most machine-vision
        # sensors: no conversion stated, so its frames are the counts they are.
        AuthoringField(
            "offset_counts",
            "float",
            "Offset (counts)",
            None,
            required=False,
        ),
        AuthoringField(
            "electrons_per_count",
            "float",
            "Electrons per count",
            None,
            required=False,
            minimum=1e-12,
        ),
    )
)


def _discover_pylon() -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    from pypylon import pylon

    factory = pylon.TlFactory.GetInstance()
    return tuple(
        discovered(f"pylon_{serial.replace('/', '_')}", PYLON_CAMERA_SCHEMA.project_values({"serial": serial}))
        for info in factory.EnumerateDevices()
        for serial in (str(info.GetSerialNumber()),)
    ), ()


def _pylon_factory(key: str, values: Mapping[str, Any]) -> PylonCameraAdapter:
    """Open a Basler from a written-down configuration; the serial selects exactly one camera."""

    camera = PylonCameraAdapter(
        PylonCameraConfig(
            serial=str(values["serial"]),
            exposure_seconds=float(values["exposure_seconds"]),
            gain_db=float(values["gain_db"]),
            trigger_source=str(values["trigger_source"]),
            roi_xywh=authored_roi_xywh(dict(values)),
            offset_counts=values["offset_counts"],
            electrons_per_count=values["electrons_per_count"],
        ),
    )
    camera.open()
    return camera


DEVICE_TYPES = (
    DeviceType(
        "camera.pylon",
        "camera",
        PYLON_CAMERA_SCHEMA,
        ("camera.adapter",),
        factory=_pylon_factory,
        label="Basler camera (pylon)",
        description=(
            "Basler machine-vision camera over pypylon, Mono8: FrameStart on the configured "
            "trigger line for finite runs, free-running for a monitor; gain is tunable."
        ),
        discover=_discover_pylon,
    ),
)

__all__ = ["DEVICE_TYPES", "PYLON_CAMERA_SCHEMA"]
