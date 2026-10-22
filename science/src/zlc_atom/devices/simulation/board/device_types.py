"""The physical side of the virtual pulse board: what each played run does to the world.

The virtual sequencer is the core's (the native memory twin and its playback
clock). Every Pulse run it plays, it hands to this device as a timeline --
``play_run(timeline)``, blocking for the run's physical duration -- and a SAFE
it performs ends with ``safe()``. The timeline grammar is the one
``SimulationWorld.fire`` reads::

    {"clock_hz": 50e6, "duration_ticks": N,
     "windows": {"cooling": [[start, end], ...], "probe": [...], "trap": [...], "emCCD": [...]},
     "buses": [{"name": "da_bias_x", "safe": code, "delay_ticks": d, "final": code,
                "steps": [[start_tick, row_span_ticks, "edge" | "ramp", target_code], ...]}]}

``windows`` are each lane's high windows in board ticks for that run (and that
scan point), ``steps`` a bus's actions in the order the run enters their rows,
``final`` the code its last-row action leaves (v2's ``_final_dac_values``).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Mapping

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_plugin import DeviceType, Opened

from ..authoring import acquire_world

#: The board line wired to the virtual site camera's trigger, as ``board.xdc`` names it.
CAMERA_TRIGGER_CHANNEL = "emCCD"

SIMULATION_BOARD_SCHEMA = AuthoringSchema((
    AuthoringField("camera_channel", "str", "Camera trigger line", CAMERA_TRIGGER_CHANNEL, required=True,
                   description="the board output whose high windows trigger the virtual site camera"),
))


class SimulationBoard:
    """Plays runs into the shared world; a SAFE ends the one playing."""

    def __init__(self, world: object, camera_channel: str) -> None:
        self._world = world
        self._camera_channel = str(camera_channel)
        self._lock = threading.Lock()
        self._stop: threading.Event | None = None

    def play_run(self, timeline: Mapping[str, Any]) -> dict[str, object]:
        """Play one run from now; answers when its physical duration has passed or SAFE stopped it."""

        stop = threading.Event()
        with self._lock:
            self._stop = stop
        started = time.monotonic()
        try:
            self._world.fire(timeline, camera_channel=self._camera_channel, started_at=started, stop_event=stop)
        finally:
            with self._lock:
                if self._stop is stop:
                    self._stop = None
        return {"stopped": stop.is_set(), "played_seconds": time.monotonic() - started}

    def safe(self) -> None:
        """The board went SAFE: stop the run being played, then the world's outputs are safe."""

        with self._lock:
            stop = self._stop
        if stop is not None:
            stop.set()
        self._world.safe()

    def close(self) -> None:
        self.safe()


def _board_factory(key: str, values: Mapping[str, Any]) -> Opened:
    world, release = acquire_world()
    board = SimulationBoard(world, values["camera_channel"])

    def close() -> None:
        try:
            board.close()
        finally:
            release()

    return Opened(board, close)


DEVICE_TYPES = (
    DeviceType(
        "simulation.board",
        "simulation",
        SIMULATION_BOARD_SCHEMA,
        (),
        factory=_board_factory,
        label="Virtual board (world side)",
        description=(
            "What the virtual pulse board's played runs do to the simulated world: cooling, "
            "probe and trap windows load and lose atoms, camera-trigger windows expose the "
            "virtual site camera, DAC buses set the bias field. Driven by the core's virtual sequencer."
        ),
        simulation=True,
        # SAFE must reach the run being played, which holds the lane.
        concurrent=("safe",),
        # Not an apparatus row: the core's virtual sequencer opens it as its world
        # side (one virtual sequencer, as in v2).
        addable=False,
    ),
)

__all__ = ["CAMERA_TRIGGER_CHANNEL", "DEVICE_TYPES", "SIMULATION_BOARD_SCHEMA", "SimulationBoard"]
