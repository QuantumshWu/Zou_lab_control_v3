"""The virtual SLM this bench can install."""

from __future__ import annotations

from typing import Any, Mapping

from zlc_atom.authoring import AuthoringSchema
from zlc_atom.devices.slm import CONCURRENT_SLM_METHODS, opened_slm
from zlc_atom.devices.slm.editor import CONTROL_WINDOW, EDITOR_COMMANDS
from zlc_plugin import DeviceType, Opened

from ..authoring import acquire_world
from .device import VirtualSLM


# The virtual panel geometry belongs to its one SimulationWorld.
VIRTUAL_SLM_SCHEMA = AuthoringSchema(())


def _slm_factory(key: str, values: Mapping[str, Any]) -> Opened:
    world, release = acquire_world()
    try:
        # Named after its instance: two virtual panels are two devices.
        slm = opened_slm(VirtualSLM(world, identity=f"virtual-slm:{key}"))
    except BaseException:
        release()
        raise

    def close() -> None:
        try:
            slm.close()
        finally:
            release()

    return Opened(slm, close)


DEVICE_TYPES = (
    DeviceType(
        "slm.virtual",
        "slm",
        VIRTUAL_SLM_SCHEMA,
        ("slm.phase",),
        factory=_slm_factory,
        label="Virtual SLM",
        description=(
            "Phase-only SLM of the simulated world: a commanded phase re-forms the world's "
            "trap roster (its FFT peaks), which the virtual cameras then image."
        ),
        simulation=True,
        concurrent=CONCURRENT_SLM_METHODS,
        commands=EDITOR_COMMANDS,
        control_window=CONTROL_WINDOW,
    ),
)

__all__ = ["DEVICE_TYPES", "VIRTUAL_SLM_SCHEMA"]
