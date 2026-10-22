"""The Hamamatsu qCMOS this bench can install, and how to find one."""

from __future__ import annotations

from typing import Any, Mapping

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices import discovered
from zlc_atom.devices.camera import EXPOSURE_STEP_SECONDS
from zlc_atom.devices.camera.roi_grid import ROI_FIELDS, authored_roi_xywh
from zlc_plugin import DeviceType

from ._dcam_driver import DcamSdkDriver
from .adapter import DcamCameraAdapter, DcamCameraConfig


#: A physical sensor: which one, how long, how fast, and which corner of it.
#: ROI is four independently optional numbers, and leaving them unset means the
#: full sensor -- an operator who only wants a shorter exposure should not have
#: to state the geometry as well.
DCAM_CAMERA_SCHEMA = AuthoringSchema(
    (
        AuthoringField("device_index", "int", "Device index", 0, minimum=0),
        AuthoringField(
            "exposure_seconds", "float", "Exposure seconds", 0.02,
            minimum=1e-9, step=EXPOSURE_STEP_SECONDS,
        ),
        AuthoringField("readout_speed", "int", "Readout speed", 1, minimum=1),
        *ROI_FIELDS,
        # What one count is worth, from this sensor's datasheet: the
        # ORCA-Quest's ultra-quiet readout is 0.107 electrons per count over
        # an offset of 200, which is also what the virtual sensor applies
        # going the other way.  It is authored rather than read back because
        # it is a property of the SENSOR, not of the session, and a
        # measurement has to be able to offer the unit before anything is
        # open.  Cleared, this camera publishes counts and nothing else.
        AuthoringField(
            "offset_counts",
            "float",
            "Offset (counts)",
            200.0,
            required=False,
        ),
        AuthoringField(
            "electrons_per_count",
            "float",
            "Electrons per count",
            0.107,
            required=False,
            minimum=1e-12,
        ),
    )
)


def _discover_dcam() -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    # A count read: the DCAM runtime belongs to the process (the "dcam" group),
    # so a scan does not start and tear it down beside an open camera.
    driver = DcamSdkDriver()
    driver.initialize()
    return tuple(
        discovered(f"dcam_{index}", DCAM_CAMERA_SCHEMA.project_values({"device_index": index}))
        for index in range(driver.device_count)
    ), ()


def _dcam_factory(key: str, values: Mapping[str, Any]) -> DcamCameraAdapter:
    """Open a Hamamatsu qCMOS from a written-down configuration.

    The adapter opens the SDK itself from the authored device index: a saved
    apparatus has to be reopenable tomorrow, and a driver is not something a
    JSON file can hold.
    """

    return DcamCameraAdapter(
        DcamCameraConfig(
            exposure_seconds=float(values["exposure_seconds"]),
            readout_speed=int(values["readout_speed"]),
            binning=1,
            roi_xywh=authored_roi_xywh(dict(values)),
            device_index=int(values["device_index"]),
            offset_counts=values["offset_counts"],
            electrons_per_count=values["electrons_per_count"],
        ),
    )


DEVICE_TYPES = (
    DeviceType(
        "camera.dcam",
        "camera",
        DCAM_CAMERA_SCHEMA,
        ("camera.adapter",),
        factory=_dcam_factory,
        identity="dcam-camera:index={device_index}",
        label="Hamamatsu qCMOS (DCAM)",
        description=(
            "Hamamatsu ORCA-Quest qCMOS over DCAM-API: externally triggered (rising edge) "
            "Mono16 frames, exposure, readout speed and ROI from the configuration; "
            "offset and electrons per count state its photoelectron conversion."
        ),
        discover=_discover_dcam,
    ),
)

__all__ = ["DCAM_CAMERA_SCHEMA", "DEVICE_TYPES"]
