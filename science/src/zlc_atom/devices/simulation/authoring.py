"""What every virtual device is told about the one world it stands in.

The apparatus has ONE simulated world. Every virtual device type declares
``process_group = SIMULATION_GROUP``, so all of them run in one plugin host
process, and each takes the world from ``acquire_world``: the first device to
open builds it from the installation's simulation mapping (the process group's
parameters), the last one to close drops it.
"""

from __future__ import annotations

from typing import Callable

import zlc_plugin
from zlc_atom.authoring import AuthoringField, AuthoringSchema

from .world import (
    DEFAULT_SIMULATION_GRID_SHAPE_YX,
    DEFAULT_SIMULATION_IMAGE_SHAPE_YX,
    SimulationGeometry,
    SimulationWorld,
    SimulationWorldConfig,
)

#: The process group every virtual device type declares.
SIMULATION_GROUP = "simulation"

SIMULATION_WORLD_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "image_shape_yx",
            "pair",
            "Image shape (Y, X)",
            DEFAULT_SIMULATION_IMAGE_SHAPE_YX,
        ),
        AuthoringField(
            "grid_shape_yx",
            "pair",
            "Grid shape (Y, X)",
            DEFAULT_SIMULATION_GRID_SHAPE_YX,
        ),
        AuthoringField("seed", "int", "Seed", 0, minimum=0),
        AuthoringField(
            "world_profile",
            "str",
            "Simulation world profile (JSON)",
            "",
        ),
    )
)


def simulation_world_config(values: dict) -> SimulationWorldConfig:
    authored = SIMULATION_WORLD_SCHEMA.project_values(values)
    geometry = SimulationGeometry(
        grid_shape_yx=tuple(authored["grid_shape_yx"]),
        image_shape_yx=tuple(authored["image_shape_yx"]),
    )
    profile_value = authored["world_profile"]
    if not isinstance(profile_value, str):
        raise TypeError("Simulation world profile (JSON) must be text")
    profile = profile_value.strip()
    if profile:
        return SimulationWorldConfig.from_profile(
            profile,
            geometry=geometry,
            seed=int(authored["seed"]),
        )
    return SimulationWorldConfig(geometry, seed=int(authored["seed"]))


def acquire_world() -> tuple[SimulationWorld, Callable[[], None]]:
    """This process's one world and the release its caller owes at close."""

    return zlc_plugin.share(
        "simulation.world",
        lambda: SimulationWorld(simulation_world_config(zlc_plugin.group())),
    )


__all__ = ["SIMULATION_GROUP", "SIMULATION_WORLD_SCHEMA", "acquire_world", "simulation_world_config"]
