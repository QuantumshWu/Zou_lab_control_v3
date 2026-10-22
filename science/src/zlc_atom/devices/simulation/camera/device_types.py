"""The virtual cameras this bench can install: the site sensor and the MOT monitor."""

from __future__ import annotations

from typing import Any, Mapping

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.camera.contract import EXPOSURE_STEP_SECONDS
from zlc_atom.devices.camera.roi_grid import ROI_FIELDS, authored_roi_xywh
from zlc_plugin import DeviceType, Opened

from ..authoring import acquire_world
from ..world import DEFAULT_SIMULATION_MOT_IMAGE_SHAPE_YX
from .adapter import VirtualCamera, VirtualCameraConfig

VIRTUAL_CAMERA_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "exposure_seconds",
            "float",
            "Exposure seconds",
            0.02,
            minimum=1e-9,
            step=EXPOSURE_STEP_SECONDS,
        ),
        *ROI_FIELDS,
    )
)

VIRTUAL_MOT_CAMERA_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "frame_shape_yx",
            "pair",
            "Frame shape (Y, X)",
            DEFAULT_SIMULATION_MOT_IMAGE_SHAPE_YX,
        ),
        AuthoringField(
            "exposure_seconds",
            "float",
            "Exposure seconds",
            0.1,
            minimum=1e-9,
            step=EXPOSURE_STEP_SECONDS,
        ),
        *ROI_FIELDS,
    )
)


def _site_camera(key: str, values: Mapping[str, Any]) -> Opened:
    world, release = acquire_world()
    try:
        config = VirtualCameraConfig(
            frame_shape_yx=world.geometry.image_shape_yx,
            exposure_seconds=float(values["exposure_seconds"]),
            roi_xywh=authored_roi_xywh(dict(values)),
            # The world's own numbers, read from it rather than restated here:
            # it converts electrons to counts to make the frame, and this is the
            # same conversion said in the direction a reader needs it.
            offset_counts=world.offset_counts,
            electrons_per_count=world.conversion_e_per_count,
        )
        camera = VirtualCamera(
            config,
            frame_source=lambda exposure, roi: world.render_frame(exposure_seconds=exposure, roi_xywh=roi),
        )
        # The world walks its registered cameras on every shot it fires.
        world.register_camera(camera)
    except BaseException:
        release()
        raise

    def close() -> None:
        try:
            world.unregister_camera(camera)
            camera.close()
        finally:
            release()

    return Opened(camera, close)


def _mot_camera(key: str, values: Mapping[str, Any]) -> Opened:
    world, release = acquire_world()
    try:
        # The real MOT monitor is a Basler read out as Mono8: the pylon adapter
        # declares and enforces uint8 frames, so the virtual stand-in does too.
        config = VirtualCameraConfig(
            frame_shape_yx=tuple(values["frame_shape_yx"]),
            exposure_seconds=float(values["exposure_seconds"]),
            roi_xywh=authored_roi_xywh(dict(values)),
            roi_step=1,
            frame_dtype="|u1",
        )

        def window(exposure: float, roi: tuple[int, int, int, int]):
            x, y, width, height = roi
            frame = world.render_mot_frame(exposure_seconds=exposure, frame_shape_yx=config.frame_shape_yx)
            return frame[y : y + height, x : x + width]

        camera = VirtualCamera(
            config,
            frame_source=window,
            free_running=True,
        )
    except BaseException:
        release()
        raise

    def close() -> None:
        try:
            camera.close()
        finally:
            release()

    return Opened(camera, close)


DEVICE_TYPES = (
    DeviceType(
        "camera.virtual",
        "camera",
        VIRTUAL_CAMERA_SCHEMA,
        ("camera.adapter",),
        factory=_site_camera,
        label="Virtual site camera",
        description=(
            "Simulated qCMOS over the simulated trap array: a frame per trigger the "
            "simulated pulse board fires on its camera channel, in counts with the "
            "world's offset and electrons per count."
        ),
        simulation=True,
    ),
    DeviceType(
        "camera.virtual_mot",
        "camera",
        VIRTUAL_MOT_CAMERA_SCHEMA,
        ("camera.adapter",),
        factory=_mot_camera,
        label="Virtual MOT camera",
        description="Simulated Mono8 MOT monitor camera: free-running frames of the simulated cloud.",
        simulation=True,
    ),
)

__all__ = ["DEVICE_TYPES", "VIRTUAL_CAMERA_SCHEMA", "VIRTUAL_MOT_CAMERA_SCHEMA"]
