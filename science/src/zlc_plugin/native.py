"""SDK adapter for the independent native client."""
from zlc_native import call, scope, current_context, executable, install_transport


def install_host_transport(process):
    def request(operation, params, context):
        payload = {"operation": operation, "params": params}
        if context is not None:
            payload.update(instance=context._run.instance.name, run=context.generation)
        return process.request("native", payload)
    install_transport(request)


__all__ = ["call", "scope", "current_context", "executable"]
