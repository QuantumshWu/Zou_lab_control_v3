"""A node for the host's own tests: plays one pulse and watches its report (PulseBoard.watch_done),
told by the core when the board has it -- or, stopped first, when its FIRE can no longer report."""

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.nodes._framework.pulse import PulseBoard
from zlc_plugin import NodeKind, NodeType

SCHEMA = AuthoringSchema((AuthoringField("document", "str", "Pulse document", ""),
                          AuthoringField("repeats", "int", "Run repeats", 1, minimum=0)))


class FireWatcher:
    def __init__(self, document, repeats):
        self.document, self.repeats = document, repeats

    def execute(self, context):
        board = PulseBoard(context)
        receipt = board.run(self.document, run_repeats=self.repeats)
        done = board.watch_done(receipt["command_id"])
        first = done.latest
        context.report_ready()
        try:
            report = board.wait_report(done, context.cancel_requested)
        except RuntimeError as error:
            board.stop()
            return {"command_id": receipt["command_id"], "first_done": bool(first.get("done")), "error": str(error)}
        return {"command_id": receipt["command_id"], "first_done": bool(first.get("done")), "status": report["status"]}


LOGIC_NODE = NodeType("fire_watcher", NodeKind.MEASUREMENT, SCHEMA, description="Plays a pulse and watches its report.",
                      build=lambda document, repeats: FireWatcher(document, repeats), reports_ready=True)
