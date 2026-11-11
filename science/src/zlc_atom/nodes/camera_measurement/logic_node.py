"""Discoverable camera measurement descriptor."""

from __future__ import annotations

from zlc_atom.devices.camera import CAMERA_PROTECTED_FIELDS
from zlc_atom.devices.camera.photoelectrons import (
    PHOTOELECTRONS,
    resolve_photoelectron_availability,
)
from zlc_atom.nodes._framework.descriptor import NodePreviewSpec
from zlc_atom.nodes.camera.authoring import (
    CAMERA_MEASUREMENT_SCHEMA,
    IMAGE_AREA_TO_ROI,
    ROI_FIELD_NAMES,
    TRIGGER_CHANNEL,
    TRIGGER_SOURCE,
    trigger_frames,
    trigger_lanes,
)
from zlc_atom.nodes.camera.measurement import (
    CAMERA_FRAMES_OUTPUT,
    CameraMeasurementNode,
    CameraMeasurementRequest,
    describe_camera_frames,
)
from zlc_plugin import DeviceRequirement, NodeKind, NodeType


def _trigger_choices(values: object, resources: object, devices: object = None) -> dict[str, list]:
    """The trigger source's lanes: each by its raw lane, which is what the draft keeps, then the port's
    name -- a port renamed in the Pulse Editor is still the lane it was."""

    del values, resources
    return {TRIGGER_CHANNEL: [[lane["lane"], f"{lane['lane']} · {lane.get('label') or lane.get('port') or ''}".rstrip(" ·")]
                              for lane in trigger_lanes(devices)]}


def _availability(devices: object) -> dict[str, str]:
    """What this draft's devices cannot do: photoelectrons without a stated conversion, and a trigger
    channel without a camera that stamps its frames or a sequencer that says what it plays."""

    unavailable = dict(resolve_photoelectron_availability(devices))
    camera = (devices or {}).get("camera")
    if camera is not None and not bool(getattr(camera, "stamps_frames", False)):
        unavailable[TRIGGER_CHANNEL] = (
            "the selected camera does not stamp its frames with its own clock, so a frame cannot be told from "
            "the next by when it came: its frames are counted"
        )
    elif not trigger_lanes(devices):
        unavailable[TRIGGER_CHANNEL] = "no sequencer is open in the apparatus to say when its lanes trigger"
    return unavailable


def _describe_frames(values, output_name, source=None, devices=None):
    """The frame axis of the draft: as many as the chosen lane triggers a shot, else as many as counted."""

    if str((values or {}).get(TRIGGER_CHANNEL) or "").strip():
        frames = trigger_frames(values, devices)
        # While the board holds no pulse nobody knows the cycle's size: say so rather than a guess.
        return None if frames is None else describe_camera_frames({"frames_per_cycle": frames}, output_name, source)
    return describe_camera_frames(values, output_name, source)


def _build(
    *,
    camera: object,
    camera_key: str,
    **values: object,
) -> CameraMeasurementNode:
    # The trigger source is known by its key alone (a device the core implements is not called).
    values.pop(TRIGGER_SOURCE, None)
    values.pop(f"{TRIGGER_SOURCE}_key", None)
    authored = CAMERA_MEASUREMENT_SCHEMA.project_values(values)
    return CameraMeasurementNode(
        camera=camera,  # type: ignore[arg-type]
        request=CameraMeasurementRequest(
            camera_key=camera_key,
            exposure_seconds=authored["exposure_seconds"],
            roi_xywh=tuple(authored[name] for name in ROI_FIELD_NAMES),
            repeat=int(authored["repeat"]),
            frames_per_cycle=int(authored["frames_per_cycle"]),
            photoelectrons=bool(authored[PHOTOELECTRONS]),
            trigger_channel=authored.get(TRIGGER_CHANNEL),
        ),
    )


LOGIC_NODE = NodeType(
    "camera_measurement",
    NodeKind.MEASUREMENT,
    CAMERA_MEASUREMENT_SCHEMA,
    description=(
        "Acquire complete camera cycles and publish frames with repeat, frame and "
        "pixel coordinates for Image/Facet plots or Occupancy. Start arms the camera "
        "and reports ready; the external trigger source or experiment owner supplies "
        "the triggers, not this measurement."
    ),
    reports_ready=True,
    # One output whatever the cycle size: the frames live on the dataset's
    # READOUT_EVENT axis, so the signal vocabulary does not change with the
    # acquisition configuration.
    outputs=(CAMERA_FRAMES_OUTPUT,),
    describe_output_frames=_describe_frames,
    node_previews=(NodePreviewSpec(CAMERA_FRAMES_OUTPUT, "facet_grid"),),
    device_requirements=(
        DeviceRequirement("camera.adapter", "camera", CAMERA_PROTECTED_FIELDS),
        # Whoever fires the camera: read for its lanes and what they play, never claimed -- a scan or the
        # Pulse Editor holds the sequencer while this measurement watches it.
        DeviceRequirement("trigger.source", TRIGGER_SOURCE, (), reads_only=True, optional=True),
    ),
    build=_build,
    selection_mappings=(IMAGE_AREA_TO_ROI,),
    # Whether this camera can be read that way is the camera's answer, not
    # this node's: the conversion is configured on the device.
    resolve_field_availability=_availability,
    resolve_field_choices=_trigger_choices,
)

__all__ = ["LOGIC_NODE"]
