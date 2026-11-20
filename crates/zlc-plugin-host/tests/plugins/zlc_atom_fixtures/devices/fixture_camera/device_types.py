"""A camera no trigger ever reaches, which says what its own close found.

Armed, it is a qCMOS waiting for an external trigger: every read waits out its
timeout and comes back empty, as DCAM's wait does while the board plays nothing.
Each line of ``log`` is one fact: ``finish`` when a capture ends, ``close armed`` or
``close idle`` as the device's own close finds it. A measurement that ended its own
capture leaves ``close idle``; ``close armed`` is a camera still capturing when its
device closed under the run.
"""

import pathlib

import numpy as np
from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.simulation.camera.adapter import VirtualCamera, VirtualCameraConfig
from zlc_plugin import DeviceType, Opened

SCHEMA = AuthoringSchema((AuthoringField("log", "str", "Log file", "", required=True),))


def _untriggered(key, values):
    log = pathlib.Path(str(values["log"]))

    def say(line):
        with log.open("a", encoding="utf-8") as file:
            file.write(f"{line}\n")

    class Untriggered(VirtualCamera):
        def finish_record_capture(self):
            terminal = super().finish_record_capture()
            say("finish")
            return terminal

    camera = Untriggered(VirtualCameraConfig(frame_shape_yx=(64, 64)), frame_source=lambda exposure, roi: np.zeros((roi[3], roi[2]), np.uint16))

    def close():
        say("close armed" if camera.capture_state() else "close idle")
        camera.close()

    return Opened(camera, close)


DEVICE_TYPES = (
    DeviceType("fixture.untriggered_camera", "camera", SCHEMA, ("camera.adapter",), factory=_untriggered),
)
