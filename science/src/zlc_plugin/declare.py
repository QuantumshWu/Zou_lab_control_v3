"""Plugin declarations and their descriptor JSON (docs/PLUGIN_PROTOCOL.md §2, §8).

Fields are the authoring fields a plugin package already declares (v2's
``AuthoringField``/``AuthoringSchema``), read by attribute; this SDK does not
define a second field grammar. Everything here is data plus the callables the
plugin's own process calls; the core sees only ``descriptor()``.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .wire import Encoder, plain


def _fields(schema: object) -> tuple[object, ...]:
    if schema is None:
        return ()
    fields = getattr(schema, "fields", schema)
    return tuple(fields)


def field_json(item: object) -> dict[str, Any]:
    """One authoring field as descriptor data."""

    enabled = getattr(item, "enabled_when", None)
    return {
        "name": str(item.name),
        "type": str(getattr(item, "value_type")),
        "label": str(item.label),
        "default": plain(getattr(item, "default", None)),
        "required": bool(getattr(item, "required", False)),
        "minimum": getattr(item, "minimum", None),
        "maximum": getattr(item, "maximum", None),
        "step": getattr(item, "step", None),
        "unit": getattr(item, "unit", None),
        "choices": [{"value": plain(choice.value), "label": str(choice.label)} for choice in getattr(item, "choices", ())],
        "segmented": bool(getattr(item, "segmented", False)),
        "blank_label": str(getattr(item, "blank_label", "")),
        "description": str(getattr(item, "description", "")),
        "enabled_when": None if enabled is None else [str(enabled[0]), plain(list(enabled[1]))],
        "derived": bool(getattr(item, "derived", False)),
        "group": str(getattr(item, "group", "")),
        "columns": [field_json(column) for column in getattr(item, "columns", ())],
    }


def _value_schema(item: dict[str, Any]) -> dict[str, Any]:
    kind = item["type"]
    if item["choices"]:
        schema: dict[str, Any] = {"enum": [choice["value"] for choice in item["choices"]]}
    elif kind == "int":
        schema = {"type": "integer"}
    elif kind == "float":
        schema = {"type": "number"}
    elif kind == "bool":
        schema = {"type": "boolean"}
    elif kind == "pair":
        schema = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
    elif kind == "rows":
        schema = {"type": "array", "items": object_schema(item["columns"])}
    else:
        schema = {"type": "string"}
    for bound in ("minimum", "maximum"):
        if item[bound] is not None and kind in ("int", "float") and not item["choices"]:
            schema[bound] = item[bound]
    words = [item["label"]]
    if item["unit"]:
        words.append(f"in {item['unit']}")
    if item["choices"]:
        words.append("; ".join(f"{choice['value']!r} = {choice['label']}" for choice in item["choices"]))
    if item["description"]:
        words.append(item["description"])
    if item["blank_label"]:
        words.append(f"null: {item['blank_label']}")
    schema["description"] = ". ".join(words)
    return schema if item["required"] else {"anyOf": [schema, {"type": "null"}], "description": schema["description"]}


def object_schema(fields: list[dict[str, Any]]) -> dict[str, Any]:
    """The JSON Schema of a parameter object, generated from its fields."""

    usable = [item for item in fields if not item["derived"]]
    return {"type": "object", "properties": {item["name"]: _value_schema(item) for item in usable},
            "required": [item["name"] for item in usable if item["required"]], "additionalProperties": False}


@dataclass(frozen=True)
class FileParam:
    """A command parameter that names a file on the operator's computer. The
    core reads (``"read"``) or writes (``"write"``) that file itself, so a
    device another computer serves reads and writes the operator's files; the
    handler gets the parameter as the path of a private copy with the
    operator's file name: the file to read, or the file to write."""

    param: str
    direction: str

    def __post_init__(self) -> None:
        if self.direction not in ("read", "write"):
            raise ValueError(f"file direction must be 'read' or 'write', not {self.direction!r}")


@dataclass(frozen=True)
class Command:
    """One operator/AI action of an open device; becomes catalogue entry ``<instance>__<name>``
    (its MCP tool name is spelled from both, PLUGIN_PROTOCOL §8).

    ``handler`` is the adapter method name to call with the parameters as
    keywords, or ``callable(adapter, **parameters)``. ``read_only``: it only
    reads, so a read-only MCP client is offered it too. ``drives``: it
    commands the device itself (an SLM Editor's Send), so the core refuses
    it while a run claims the device (v2's device-command lease). ``file``:
    the parameter naming a file the core reads or writes (``FileParam``).
    """

    name: str
    summary: str
    handler: str | Callable[..., Any]
    params: object = ()
    read_only: bool = False
    drives: bool = False
    file: FileParam | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or self.name != self.name.strip():
            raise ValueError(f"command name {self.name!r} must be non-empty text with no surrounding space")
        if not self.summary.strip():
            raise ValueError(f"command {self.name} needs a summary")
        object.__setattr__(self, "read_only", bool(self.read_only))
        object.__setattr__(self, "drives", bool(self.drives))
        if self.file is not None and self.file.param not in {str(item.name) for item in _fields(self.params)}:
            raise ValueError(f"command {self.name}'s file {self.file.param!r} is not one of its parameters")

    def descriptor(self) -> dict[str, Any]:
        fields = [field_json(item) for item in _fields(self.params)]
        file = None if self.file is None else {"param": self.file.param, "direction": self.file.direction}
        return {"name": self.name, "summary": self.summary.strip(), "read_only": self.read_only,
                "drives": self.drives, "params": fields, "input_schema": object_schema(fields), "file": file}


@dataclass(frozen=True)
class PhysicalOutput:
    """An output Stop all drives safe through the adapter's ``make_safe(name)``."""

    name: str


@dataclass(frozen=True)
class Opened:
    """A factory's result when closing needs more than ``adapter.close()``.
    ``admit_peers(bool)``: the server of a type that announces one (DeviceType
    ``announce``) admits other machines while the device is published."""

    adapter: object
    close: Callable[[], None]
    admit_peers: Callable[[bool], None] | None = None


@dataclass(frozen=True)
class DeviceType:
    """One device type: v2's DeviceTypeDescriptor without the installation context.

    ``factory(key, values)`` receives the instance key and the projected
    parameters and returns the opened adapter; ``adapter.close()`` closes it.
    State shared between instances of one ``process_group`` is a module-level
    owner (``zlc_plugin.shared``), not a context argument.
    """

    type_id: str
    family: str
    parameters: object
    capabilities: tuple[str, ...]
    factory: Callable[[str, Mapping[str, Any]], object]
    label: str = ""
    description: str = ""
    process_group: str | None = None
    physical_outputs: tuple[PhysicalOutput, ...] = ()
    commands: tuple[Command, ...] = ()
    #: Hardware scan, answered by ``query discover``: (found entries
    #: ``{instance_id, role, parameters}``, notes about what could not be asked).
    discover: Callable[[], tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]] | None = None
    addable: bool = True
    #: SPI methods answered on their own thread instead of the lane, for an
    #: adapter that is thread-safe for them: a phase-movie submit or cancel
    #: while play_phase_sequence holds the lane.
    concurrent: tuple[str, ...] = ()
    #: The window a device's Control opens instead of the generic field form
    #: (v2 control_factory): ``"slm-editor"`` for an SLM; empty for the form.
    control_window: str = ""
    #: Logger-name prefixes this type's code narrates under (v2 log_channels): their
    #: INFO lines are its device log, beside the lines ending in ``device=<identity>``.
    log_channels: tuple[str, ...] = ()
    #: A device with a server of its own (v2 announce: a local SLM head, a local pulse
    #: board): ``(client type, endpoint(values))`` -- the type another machine opens and
    #: where (``{"port": ...}``; the peer dials the address it found this machine at).
    #: Remote on then announces that, and the factory's ``Opened.admit_peers`` opens the
    #: server to other machines; any other device is served by the bench fabric itself.
    announce: tuple[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]] | None = None

    def __post_init__(self) -> None:
        if not self.type_id or not self.family:
            raise ValueError("a device type needs type_id and family")
        if not callable(self.factory):
            raise TypeError("a device type needs a callable factory")
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        object.__setattr__(self, "physical_outputs", tuple(self.physical_outputs))
        object.__setattr__(self, "commands", tuple(self.commands))
        object.__setattr__(self, "concurrent", tuple(str(name) for name in self.concurrent))
        object.__setattr__(self, "log_channels", tuple(str(name) for name in self.log_channels))
        if any(not name for name in self.log_channels):
            raise ValueError(f"{self.type_id}: log_channels must be non-empty logger names")
        names = [command.name for command in self.commands]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.type_id}: command names must be unique")

    def project(self, values: Mapping[str, Any] | None) -> dict[str, Any]:
        project = getattr(self.parameters, "project_values", None)
        return dict(values or {}) if project is None else project(values or {})

    def descriptor(self, module: str) -> dict[str, Any]:
        return {
            "type_id": self.type_id, "kind": "device", "family": self.family,
            "label": self.label or self.type_id, "description": self.description, "module": module,
            "process_group": self.process_group, "capabilities": list(self.capabilities),
            "parameters": [field_json(item) for item in _fields(self.parameters)],
            "physical_outputs": [{"name": item.name} for item in self.physical_outputs],
            "commands": [command.descriptor() for command in self.commands],
            "queries": ["discover"] if self.discover is not None else [], "addable": self.addable,
            "concurrent": list(self.concurrent), "control_window": self.control_window,
            "announces": self.announce[0] if self.announce is not None else "",
        }


class NodeKind(str, enum.Enum):
    """What a node is to the experiment: how it is layered, not how it runs."""

    MEASUREMENT = "measurement"
    TASK = "task"
    PROCESSOR = "processor"


@dataclass(frozen=True, slots=True)
class DatasetOutputDeclaration:
    """One owner-paired public output (v2's zlc_runtime declaration)."""

    name: str
    contract_id: str
    index_by_source: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name or self.name.startswith("@"):
            raise ValueError("a dataset output name must be bare")
        if not self.contract_id:
            raise ValueError("a dataset output needs a contract id")


@dataclass(frozen=True)
class DatasetInput:
    """One Runtime input of a node (v2's DatasetInputSpec)."""

    name: str
    contract_id: str | None
    delivery: str
    select_bundle: bool = False
    enabled_when: tuple[str, tuple[Any, ...]] | None = None

    def __post_init__(self) -> None:
        if self.delivery not in ("exact", "latest"):
            raise ValueError("dataset input delivery must be 'exact' or 'latest'")


@dataclass(frozen=True)
class DeviceRequirement:
    """A device a node drives during its run (v2's DeviceRequirement)."""

    capability_token: str
    argument_name: str
    protected_fields: tuple[str, ...] | None = ()
    enabled_when: tuple[str, tuple[Any, ...]] | None = None


#: NodeType callables answered through ``query`` (PLUGIN_PROTOCOL.md §2), each
#: with its v2 parameters in v2's positional order. The core states the facts
#: by name (``{"values", "devices", ...}``); the SDK, not the core, knows how a
#: v2 callable takes them.
QUERY_PARAMETERS: dict[str, tuple[str, ...]] = {
    "declare_outputs": ("values", "devices"),
    "declare_previews": ("values", "devices"),
    "resolve_field_availability": ("devices",),
    "resolve_field_choices": ("values", "resources"),
    "resolve_device_claims": ("values",),
    "editor_facts": ("values", "resources", "devices", "source", "bundle", "artifacts"),
    "resolve_defaults": ("values", "resources"),
    "admit_source": ("values", "bundle"),
    "admit_resources": ("values", "resources", "installation"),
    "estimate_run": ("values", "devices", "resources"),
    "describe_output_frames": ("values", "output_name", "source"),
}
NODE_QUERIES = tuple(QUERY_PARAMETERS)
#: The draft queries a node type answers for its default draft in its catalogue entry
#: (``node.answers``): a node just added is drawn from them at once, its own process's
#: answers replacing them when they land.
FIRST_ANSWERS = ("declare_outputs", "declare_previews", "resolve_field_availability", "resolve_field_choices",
                 "resolve_defaults", "resolve_device_claims", "editor_facts")


@dataclass(frozen=True)
class NodeType:
    """One logic node: v2's LogicNodeDescriptor with the same field names.

    ``build(**values, **devices, **{argument + "_key": instance}, ...)`` makes
    the node object; its ``execute(context)`` runs on the run thread with a
    ``RunContext``. Previews, selection mappings, artifacts and workspace
    resources are carried as the plugin declares them; the descriptor lists
    them by name for the Workbench.
    """

    api_name: str
    kind: NodeKind
    authoring_schema: object
    description: str = field(kw_only=True)
    input_specs: tuple[object, ...] = ()
    outputs: tuple[DatasetOutputDeclaration, ...] = ()
    declare_outputs: Callable[..., Any] | None = None
    describe_output_frames: Callable[..., Any] | None = None
    device_requirements: tuple[DeviceRequirement, ...] = ()
    build: Callable[..., object] | None = None
    node_previews: tuple[object, ...] | None = None
    declare_previews: Callable[..., Any] | None = None
    artifact_outputs: tuple[object, ...] = ()
    selection_mappings: tuple[object, ...] = ()
    workspace_resources: tuple[object, ...] = ()
    resolve_field_availability: Callable[..., Any] | None = None
    #: ``(values, resources) -> {field: [(value, label), ...]}``: the choices of
    #: a field whose options come from the draft (a chosen pulse's periods).
    resolve_field_choices: Callable[..., Any] | None = None
    #: ``(values) -> {argument: {"instance": key, "fields": [...]}}``: devices a
    #: draft names (a scan's device axes) beside the declared requirements;
    #: the core claims those fields for the run and passes each as ``argument``.
    resolve_device_claims: Callable[..., Any] | None = None
    #: ``(values, resources, devices, source, bundle, artifacts) -> {"editor": kind,
    #: "manages": [field, ...], ...}``: the model of this node's own editor,
    #: drawn by the UI's editor of that kind, which writes back the fields it
    #: manages (a scan's plan from every knob the pulse and the lent devices
    #: offer; a pair matrix over the frames ``source`` carries; Derive's
    #: variables typed against the source ``bundle``'s schemas; Occupancy's frame
    #: claims over the files ``artifacts`` names).  ``{"editor": "stack",
    #: "editors": [...]}`` stacks several.
    editor_facts: Callable[..., Any] | None = None
    resolve_defaults: Callable[..., Any] | None = None
    admit_source: Callable[..., Any] | None = None
    admit_resources: Callable[..., Any] | None = None
    estimate_run: Callable[..., Any] | None = None
    acquisition_input: str = ""
    acquisition_completion: str = "ready"
    reports_ready: bool = False
    label: str = ""
    #: ``{type_id: values}``: node types this one took over; a saved row naming
    #: one loads as this type, its values with these over them.
    replaces: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.api_name or not isinstance(self.kind, NodeKind):
            raise ValueError("a node needs api_name and a NodeKind")
        if not self.description.strip():
            raise ValueError(f"node {self.api_name} needs a description of its purpose")
        for name in ("input_specs", "outputs", "device_requirements", "artifact_outputs",
                     "selection_mappings", "workspace_resources"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.node_previews is not None:
            object.__setattr__(self, "node_previews", tuple(self.node_previews))
        if self.declare_outputs is not None and self.outputs:
            raise ValueError("a node declares its outputs once: by name, or by draft")
        names = [item.name for item in self.outputs]
        if len(set(names)) != len(names):
            raise ValueError("output names must be unique")
        if self.acquisition_completion not in ("ready", "terminal"):
            raise ValueError("acquisition_completion must be 'ready' or 'terminal'")

    def project(self, values: Mapping[str, Any] | None) -> dict[str, Any]:
        project = getattr(self.authoring_schema, "project_values", None)
        return dict(values or {}) if project is None else project(values or {})

    @property
    def type_id(self) -> str:
        return self.api_name

    def first_answers(self, parameters: list[dict[str, Any]]) -> dict[str, Any]:
        """Each draft query's answer for the default draft, as no bench yet: no devices, files,
        source or bundle. A query that cannot answer it is left to the node's own process."""

        facts = {"values": {item["name"]: item["default"] for item in parameters}, "devices": {}, "resources": {},
                 "source": None, "bundle": [], "artifacts": {"inputs": {}, "resolved": {}}}
        answers: dict[str, Any] = {}
        for name in FIRST_ANSWERS:
            function = getattr(self, name)
            if function is None:
                continue
            try:
                # As the node's process sends it to the core (a plain query reply).
                answer = Encoder(None, plain=True)(function(*(facts[parameter] for parameter in QUERY_PARAMETERS[name])))
                json.dumps(answer, allow_nan=False)
            except Exception:  # noqa: BLE001 -- a default draft it cannot answer waits for the node's own
                continue
            answers[name] = answer
        return answers

    def descriptor(self, module: str) -> dict[str, Any]:
        inputs, artifact_inputs = [], []
        for item in self.input_specs:
            if isinstance(item, DatasetInput) or hasattr(item, "delivery"):
                inputs.append({"name": item.name, "contract": item.contract_id,
                               "delivery": item.delivery, "select_bundle": bool(getattr(item, "select_bundle", False)),
                               "enabled_when": plain(getattr(item, "enabled_when", None))})
            else:
                codec = getattr(item, "codec", None)
                artifact_inputs.append({"name": item.name, "label": getattr(item, "label", item.name),
                                        "contract_id": getattr(item, "contract_id", None),
                                        "file_filter": str(getattr(codec, "file_filter", "")),
                                        "required": bool(getattr(item, "required", True)),
                                        "frame_claims": bool(getattr(item, "frame_claims", False))})
        parameters = [field_json(item) for item in _fields(self.authoring_schema)]
        previews = [{"output": item.output.name, "plot_kind": item.plot_kind, "semantic": plain(dict(item.semantic)),
                     "producer": item.producer, "overlay": None if item.overlay is None else item.overlay.name}
                    for item in (self.node_previews or ())]
        return {
            "type_id": self.api_name, "kind": "node", "family": self.kind.value,
            "label": self.label or self.api_name, "description": self.description.strip(), "module": module,
            "process_group": None, "capabilities": [],
            "parameters": parameters,
            "physical_outputs": [], "commands": [],
            "node": {
                "node_kind": self.kind.value, "inputs": inputs, "artifact_inputs": artifact_inputs,
                "outputs": [{"name": item.name, "contract": item.contract_id, "index_by_source": item.index_by_source}
                            for item in self.outputs],
                "devices": [{"capability": item.capability_token, "argument": item.argument_name,
                             "protected_fields": None if item.protected_fields is None else list(item.protected_fields),
                             "enabled_when": plain(item.enabled_when)} for item in self.device_requirements],
                "previews": previews, "reports_ready": self.reports_ready,
                "acquisition_input": self.acquisition_input, "acquisition_completion": self.acquisition_completion,
                "artifact_outputs": [{"name": item.name, "contract": item.contract_id} for item in self.artifact_outputs],
                "selection_mappings": [{"plot_kind": item.plot_kind, "selector_kind": item.selector_kind,
                                        "draft_fields": list(item.draft_fields)} for item in self.selection_mappings],
                "workspace_resources": [{"field": item.field_name, "contract": item.contract_id,
                                         "directory": item.directory, "suffixes": list(item.suffixes),
                                         "argument": getattr(item, "argument_name", "") or item.field_name}
                                        for item in self.workspace_resources],
                "replaces": {name: plain(dict(values)) for name, values in self.replaces.items()},
                "answers": self.first_answers(parameters),
            },
            "queries": [name for name in NODE_QUERIES if getattr(self, name) is not None]
                       + (["selection_patch"] if self.selection_mappings else []),
        }


__all__ = ["Command", "DatasetInput", "DatasetOutputDeclaration", "DeviceRequirement", "DeviceType",
           "NODE_QUERIES", "QUERY_PARAMETERS", "NodeKind", "NodeType", "Opened", "PhysicalOutput",
           "field_json", "object_schema"]
