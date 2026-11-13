"""The Newport SMC100CC/PP this bench can install, and how to find one."""

from __future__ import annotations

from typing import Any, Mapping

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
from zlc_atom.devices import discovered, free_serial_ports, underscored
from zlc_atom.devices.motion.contract import AXIS_UNITS, MOTION_OUTPUT
from zlc_plugin import Command, DeviceType, PhysicalOutput

from .axis import Smc100Axis, Smc100Config, discover_smc100

#: Which port, which controller address on it, and what unit its stage counts
#: in.  Everything else -- travel, velocity, state, position -- is read from the
#: controller.  The unit is authored because the controller does not name it:
#: its numbers are in whatever unit its stage configuration chose.
NEWPORT_SMC100_SCHEMA = AuthoringSchema(
    (
        AuthoringField("port", "str", "Serial port", "", blank_label="found on Init"),
        AuthoringField(
            "address", "int", "Controller address", 1, minimum=1, maximum=31,
            description="the controller's RS-485 address (1 unless changed with Newport's utility)",
        ),
        AuthoringField(
            "unit", "choice", "Stage unit", "mm",
            choices=(AuthoringChoice("mm", "mm (linear)"), AuthoringChoice("deg", "deg (rotation)")),
            segmented=True,
            description="the unit the stage's configuration counts in: mm for a linear stage, deg for a rotation stage",
        ),
    )
)


def _factory(key: str, values: Mapping[str, Any]) -> Smc100Axis:
    return Smc100Axis(Smc100Config(port=str(values["port"]), address=int(values["address"]), unit=str(values["unit"])))


def _discover(excluded: frozenset[str] = frozenset()) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    # Each free port is asked VE at address 1, the factory address; a controller
    # set to another address is added by its port and address.  A port the
    # apparatus binds is passed over, never opened.
    return tuple(
        discovered(f"smc100_{underscored(port)}", NEWPORT_SMC100_SCHEMA.project_values({"port": port, "address": address}))
        for port, address, _version, _stage in discover_smc100(excluded=excluded)
    ), ()


def _resolve(values: Mapping[str, Any], excluded: frozenset[str]) -> dict[str, Any]:
    """A row left without its port, on Init: the first free port a controller answers VE
    on at the row's address, pinned into the apparatus."""

    free, passed = free_serial_ports(excluded)
    address = int(values["address"])
    unopened: list[str] = []
    found = discover_smc100(addresses=(address,), ports=free, first=True, unopened=unopened)
    bound = [f"passed over {', '.join(passed)}: bound"] if passed else []
    if not found:
        said = "; ".join([f"asked {', '.join(free) or 'no free port'}", *unopened, *bound])
        raise RuntimeError(
            f"no Newport SMC100 answered at address {address} on the free serial ports ({said}); "
            "set its port and address, or power it and Init again"
        )
    port, _address, version, stage = found[0]
    return {
        "found": {"port": port},
        "note": f"SMC100 at address {address} on {port} ({version}{f', stage {stage}' if stage else ''})"
        + "".join(f" ({said})" for said in bound),
    }


#: The operator's actions.  Stop is not ``drives``: it must reach the stage
#: while a run holds it, and it is ``concurrent`` so a move holding the lane
#: does not keep it waiting (PLUGIN_PROTOCOL §4).
SMC100_COMMANDS = (
    Command("home", "Find the stage's reference (OR); only from NOT REFERENCED. Answers once it is READY.",
            "home", drives=True),
    Command("stop", "Stop any motion now (ST), even while a move or a run holds the stage.", "stop"),
    Command("enable", "Energize the motor and close the loop (MM1: DISABLE to READY).", "enable", drives=True),
    Command("disable", "De-energize the motor (MM0: READY to DISABLE); the position is still read.",
            "disable", drives=True),
    Command("reset", "Reboot the controller (RS): it comes back NOT REFERENCED, with its stored settings.",
            "reset", drives=True),
    Command("status", "The controller's state, fault, position, travel and velocity, read now.", "status",
            read_only=True),
)


DEVICE_TYPES = (
    DeviceType(
        "motion.newport_smc100",
        "motion",
        NEWPORT_SMC100_SCHEMA,
        ("motion.axis",),
        factory=_factory,
        identity="newport-smc100:{port}/{address}",
        resources=("serial:{port}",),
        label="Newport SMC100 stage",
        description=(
            "Newport SMC100CC/PP single-axis controller over its RS-232/USB link (57600 baud, Xon/Xoff): "
            "absolute position in the stage's unit (" + " or ".join(AXIS_UNITS) + ") inside the controller's "
            "travel, read back where the stage stopped; Home, Stop, Enable, Disable, Reset. A move is refused "
            "until the stage is homed."
        ),
        # Stop all stops the motion (Smc100Axis.make_safe).
        physical_outputs=(PhysicalOutput(MOTION_OUTPUT),),
        commands=SMC100_COMMANDS,
        concurrent=("stop", "make_safe"),
        discover=_discover,
        resolve=_resolve,
    ),
)

__all__ = ["DEVICE_TYPES", "NEWPORT_SMC100_SCHEMA", "SMC100_COMMANDS"]
