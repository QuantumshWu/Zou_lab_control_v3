"""The virtual stage this bench can install."""

from __future__ import annotations

from typing import Any, Mapping

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.motion.contract import MOTION_OUTPUT
from zlc_atom.devices.motion.newport_smc100.device_types import SMC100_COMMANDS
from zlc_plugin import DeviceType, PhysicalOutput

VIRTUAL_SMC100_SCHEMA = AuthoringSchema(
    (AuthoringField("address", "int", "Controller address", 1, minimum=1, maximum=31),)
)


def _factory(key: str, values: Mapping[str, Any]) -> object:
    from zlc_atom.devices.motion.newport_smc100 import Smc100Axis, Smc100Config

    from .controller import SimulatedSerialPort, SimulatedSmc100

    # A controller at power-up per Init: NOT REFERENCED, as a real one is.
    address = int(values["address"])
    return Smc100Axis(
        Smc100Config(port=f"virtual-{key}", address=address, unit="mm"),
        serial_port=SimulatedSerialPort(SimulatedSmc100(address)),
    )


DEVICE_TYPES = (
    DeviceType(
        "motion.virtual_smc100",
        "motion",
        VIRTUAL_SMC100_SCHEMA,
        ("motion.axis",),
        factory=_factory,
        identity="virtual-smc100:{address}",
        label="Virtual SMC100 stage",
        description=(
            "The Newport SMC100 driver over a simulated controller and 25 mm stage (5 mm/s, 20 mm/s², "
            "home at 0): NOT REFERENCED at power-up, homing, trapezoidal moves, travel limits, Stop "
            "and the controller's own refusals, as the real one."
        ),
        physical_outputs=(PhysicalOutput(MOTION_OUTPUT),),
        commands=SMC100_COMMANDS,
        concurrent=("stop", "make_safe"),
    ),
)

__all__ = ["DEVICE_TYPES", "VIRTUAL_SMC100_SCHEMA"]
