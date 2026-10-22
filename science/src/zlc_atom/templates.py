"""Named starting points for an apparatus (Device Manager New...).

A template is a convenience, not a constraint: devices are installed one by
one, so any mixture is legal.  The two templates are the ends of that
spectrum (v2 ``zlc_atom.install.templates``); every row is (instance id,
device type).  The sequencer types are the core's native ones.
"""

APPARATUS_TEMPLATES = {
    "virtual": (
        ("camera", "camera.virtual"),
        ("sequencer", "sequencer.virtual"),
        ("mot_camera", "camera.virtual_mot"),
        ("slm", "slm.virtual"),
    ),
    "hardware": (
        ("camera", "camera.dcam"),
        ("sequencer", "sequencer.hardware"),
    ),
}

__all__ = ["APPARATUS_TEMPLATES"]
