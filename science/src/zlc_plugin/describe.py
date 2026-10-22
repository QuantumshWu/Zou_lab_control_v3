"""Plugin discovery: ``python -m zlc_plugin describe <package>...``.

Imports every ``device_types.py`` (``DEVICE_TYPES``) and ``logic_node.py``
(``LOGIC_NODE``) under the named packages' ``devices``/``nodes`` folders and
writes one JSON document to stdout. A leaf that fails to import is reported
with its traceback; the others stay available (PLUGIN_PROTOCOL.md §2).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import pathlib
import sys
import traceback
from typing import Any, Iterator

from .declare import DeviceType, NodeType
from .wire import PROTOCOL


def leaf_modules(package: str) -> Iterator[str]:
    root = pathlib.Path(importlib.import_module(package).__file__).resolve().parent
    for folder, leaf in (("devices", "device_types.py"), ("nodes", "logic_node.py")):
        base = root / folder
        if not base.is_dir():
            continue
        for path in sorted(base.rglob(leaf)):
            relative = path.relative_to(root).with_suffix("")
            if any(part.startswith("_") or part == "__pycache__" for part in relative.parts[:-1]):
                continue
            yield ".".join((package, *relative.parts))


def types_of(module_name: str) -> tuple[DeviceType | NodeType, ...]:
    module = importlib.import_module(module_name)
    if module_name.endswith(".device_types"):
        declared = getattr(module, "DEVICE_TYPES", None)
        if not isinstance(declared, tuple) or not all(isinstance(item, DeviceType) for item in declared):
            raise TypeError(f"{module_name} must export DEVICE_TYPES, a tuple of zlc_plugin.DeviceType")
        return declared
    declared = getattr(module, "LOGIC_NODE", None)
    if not isinstance(declared, NodeType):
        raise TypeError(f"{module_name} must export LOGIC_NODE, a zlc_plugin.NodeType")
    return (declared,)


def find(type_id: str, module_name: str) -> DeviceType | NodeType:
    for item in types_of(module_name):
        if item.type_id == type_id:
            return item
    raise LookupError(f"{module_name} declares no plugin type {type_id!r}")


def describe(packages: list[str]) -> dict[str, Any]:
    types: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    templates: dict[str, list[dict[str, str]]] = {}
    seen: dict[str, str] = {}
    for package in packages:
        for module_name in leaf_modules(package):
            try:
                for item in types_of(module_name):
                    if item.type_id in seen:
                        raise ValueError(f"type {item.type_id!r} is also declared by {seen[item.type_id]}")
                    seen[item.type_id] = module_name
                    types.append(item.descriptor(module_name))
            except Exception as error:
                errors.append({"module": module_name, "error": f"{type(error).__name__}: {error}",
                               "traceback": "".join(traceback.format_exception(error))})
        # Apparatus templates a package offers (``<package>.templates.APPARATUS_TEMPLATES``).
        try:
            spec = importlib.util.find_spec(f"{package}.templates")
            offered = {} if spec is None else getattr(importlib.import_module(f"{package}.templates"), "APPARATUS_TEMPLATES", {})
            for name, rows in offered.items():
                templates[str(name)] = [{"instance_id": str(instance), "type_id": str(type_id)} for instance, type_id in rows]
        except Exception as error:
            errors.append({"module": f"{package}.templates", "error": f"{type(error).__name__}: {error}",
                           "traceback": "".join(traceback.format_exception(error))})
    return {"protocol": list(PROTOCOL), "types": types, "errors": errors, "templates": templates}


def main(packages: list[str]) -> int:
    channel = sys.stdout
    sys.stdout = sys.stderr
    document = describe(packages or ["zlc_atom"])
    channel.write(json.dumps(document, allow_nan=False))
    channel.flush()
    return 0
