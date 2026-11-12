"""The plugin SDK: declare devices and logic nodes that run outside the Rust core.

Contract: docs/PLUGIN_PROTOCOL.md. A device folder exports
``DEVICE_TYPES = (DeviceType(...), ...)`` from ``device_types.py``; a node
folder exports ``LOGIC_NODE = NodeType(...)`` from ``logic_node.py``. The core
runs every instance in one ``python -m zlc_plugin host`` process (a type that
declares ``isolated`` gets one of its own) and talks to it over the framed
channel; plugin code never sees the channel.

    DeviceType(type_id, family, parameters, capabilities, factory=lambda key, values: Adapter(values))
    NodeType(api_name, NodeKind.MEASUREMENT, schema, description=..., build=..., outputs=(...))

A node's ``execute(context)`` receives a ``RunContext`` with v2's
NodeExecutionContext methods (commit_live, report_ready, cancel_requested, ...);
its device arguments are proxies implementing the capability's Protocol.
"""

# Keep the SDK namespace light: importing its data/service adapters must not
# import the plugin server or start an interpreter/engine process.
import importlib

_EXPORTS = {
    **{name: "context" for name in ("LiveDatasetOutput", "LiveEvents", "Read", "RunContext")},
    **{name: "declare" for name in ("Command", "DatasetInput", "DatasetOutputDeclaration", "DeviceRequirement", "DeviceType", "FileParam", "NodeKind", "NodeType", "Opened", "PhysicalOutput")},
    **{name: "fit" for name in ("FitAnswer", "fit_models")},
    **{name: "host" for name in ("allocate", "core", "notifier", "share", "simulation_parameters")},
    **{name: "wire" for name in ("DeviceHeld", "PluginError", "Refused")},
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(name)
    value = getattr(importlib.import_module("." + module, __name__), name)
    globals()[name] = value
    return value
