"""A node for the host's own tests: asks the application for batches of fits."""

import numpy as np
from zlc_atom.authoring import AuthoringSchema
from zlc_plugin import NodeKind, NodeType


class Fitter:
    def execute(self, context):
        x = np.arange(60, dtype="<f4")
        y = np.ones((300, 60))
        answer = context.fit("damped_sine", x, y, sigma=np.full(y.shape, 0.5), warm=np.zeros((300, 2)))
        short = context.fit("damped_sine", x[:3], y[:2, :3])
        return {"values": list(answer.values.shape), "errors": list(answer.errors.shape), "nan_error": bool(np.isnan(answer.errors[0, 1])),
                "success": answer.success.tolist()[:2], "chi": list(answer.reduced_chi_square.shape), "messages": list(answer.messages[:2]),
                "parameters": list(answer.parameters), "short": list(short.values.shape)}


LOGIC_NODE = NodeType("fitter", NodeKind.MEASUREMENT, AuthoringSchema(()), description="Asks for batches of fits.", build=lambda: Fitter())
