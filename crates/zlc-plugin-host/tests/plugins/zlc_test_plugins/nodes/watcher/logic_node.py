"""A node for the host's own tests: watches one lane's pulse schedule and keeps each answer it was
told, until the board plays a FIRE other than the one it held when the watch began (or it is stopped)."""

import time

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_plugin import NodeKind, NodeType

SCHEMA = AuthoringSchema((AuthoringField("lane", "str", "Trigger lane", ""),))


class Watcher:
    def __init__(self, lane):
        self.lane = lane

    def execute(self, context):
        board = context.watch("pulse_schedule", lane=self.lane)
        told = [board.latest]
        context.report_ready()
        while not context.cancel_requested() and not (told[-1]["state"] == "playing" and told[-1]["command_id"] != told[0]["command_id"]):
            if board.latest is not told[-1]:
                told.append(board.latest)
            time.sleep(0.002)
        return {"states": [answer["state"] for answer in told], "revision": told[-1]["revision"],
                "command_id": told[-1]["command_id"], "points": len(told[-1]["points"])}


LOGIC_NODE = NodeType("watcher", NodeKind.MEASUREMENT, SCHEMA, description="Watches one lane's pulse schedule.",
                      build=lambda lane: Watcher(lane), reports_ready=True)
