"""The plugin SDK: declare devices and logic nodes that run outside the Rust core.

Contract: docs/PLUGIN_PROTOCOL.md. A device folder exports
``DEVICE_TYPES = (DeviceType(...), ...)`` from ``device_types.py``; a node
folder exports ``LOGIC_NODE = NodeType(...)`` from ``logic_node.py``. The core
starts one ``python -m zlc_plugin.host`` per instance and talks to it over the
framed channel; plugin code never sees the channel.

    DeviceType(type_id, family, parameters, capabilities, factory=lambda key, values: Adapter(values))
    NodeType(api_name, NodeKind.MEASUREMENT, schema, description=..., build=..., outputs=(...))

A node's ``execute(context)`` receives a ``RunContext`` with v2's
NodeExecutionContext methods (commit_live, report_ready, cancel_requested, ...);
its device arguments are proxies implementing the capability's Protocol.
"""

from .context import LiveDatasetOutput, Read, RunContext
from .declare import (Command, DatasetInput, DatasetOutputDeclaration, DeviceRequirement, DeviceType, FileParam,
                      NodeKind, NodeType, Opened, PhysicalOutput)
from .host import allocate, core, group, notifier, share
from .wire import PluginError, Refused

__all__ = [
    "Command", "DatasetInput", "DatasetOutputDeclaration", "DeviceRequirement", "DeviceType", "FileParam", "LiveDatasetOutput",
    "NodeKind", "NodeType", "Opened", "PhysicalOutput", "PluginError", "Read", "Refused", "RunContext", "allocate", "core",
    "group", "notifier", "share",
]
