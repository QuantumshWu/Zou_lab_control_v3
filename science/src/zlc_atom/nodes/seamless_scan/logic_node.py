"""The seamless scan Measurement: a plan the BOARD plays from its scan table.

The board's axes play from one load, every point back to back; a manual,
device or API axis is moved by the host between fires, one fire per host
point.  What the operator authors is the template, the plan, how many whole
sweeps and how many in-place shots per point.  There is nothing to say about
how a fresh value is taken: the fired cycle drives the source, so its
publications ARE the played rows, in order.

The loop this node offers is ``scan.SeamlessScanMeasurement``; this module
declares its form and published output.
"""

from __future__ import annotations

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.nodes._framework.descriptor import NodePreviewSpec
from zlc_atom.nodes._framework.pulse import pulse_description
from zlc_atom.nodes.scan.editor import scan_editor_facts
from zlc_atom.nodes.scan import (
    API_PARAM_FAMILY,
    MANUAL_PARAM_FAMILY,
    PULSE_PARAM_FAMILY,
    SCAN_OUTPUT,
    SCAN_PLAN_SELECTIONS,
    SEAMLESS_PULSE_RESOURCE,
    DeviceKnob,
    ScanPlan,
    SeamlessScanMeasurement,
    api_overrides_from_authored,
    api_scan_ports_for,
    bind_plan,
    device_claims,
    estimate_scan_time,
    hardware_scan_ports_for,
    plan_from_authored,
    resolve_api_overrides,
    scan_ports_for_devices,
    unscannable_api_ports,
)
import zlc_plugin
from zlc_plugin import DatasetInput, DeviceRequirement, NodeKind, NodeType

#: The argument a claimed device of a device axis arrives as (``tunable:<key>``).
_TUNABLE = "tunable:"


def _validate(values) -> None:
    """The plan and the API values read as the build reads them, while the
    draft is still being edited: an empty Values list, an empty plan or a
    duplicated port is said on the Edit page, not at Start."""

    plan_from_authored(values["plan"])
    api_overrides_from_authored(values.get("api_values"))


SEAMLESS_SCAN_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "pulse_template",
            "resource",
            "Pulse template",
            "",
            required=True,
            description="Pulse filename relative to workspace/pulses (for example imaging.json), or an absolute path.",
        ),
        AuthoringField(
            "plan",
            "text",
            "Scan plan",
            "",
            required=True,
            description=(
                'JSON text: {"axes":[{"port":"device:rf:frequency","values":[500,501,502],"unit":"MHz"}]}. '
                'Axes are outermost first; values is the explicit ordered numeric point list, not [start,stop,count]. '
                'Ports: device:<device_id>:<field>, pulse:param:<scan field_id>, api:<API field_id>, manual:<name>. '
                'Use device Control metadata or the Pulse field/binding catalog for IDs and units. '
                'Only authored axes are scanned; omitted Pulse scan slots retain their authored value. '
                'The optional Values input bank uses mode="values" and value_text="500,501,502" '
                'while values retains the independent Range bank. Manual axes require the human operator. '
                'Every listed axis is scanned; remove an axis to stop scanning it.'
            ),
        ),
        AuthoringField(
            "acquisition_logic",
            "str",
            "Acquisition logic",
            "",
            description="Optional acquisition Logic to restart once and wait until ready before the scan.",
        ),
        # The pulse's API slots, set once for this run.  Only what differs
        # from the pulse is written here, so a recalibration that lands in
        # the workspace's current set is not outranked by a stale copy.
        AuthoringField(
            "api_values",
            "text",
            "API values",
            "",
            description=(
                "Optional fixed API overrides, one <field_id> = <number> per line in that field's "
                "authored unit; empty keeps the Pulse defaults. Keep lines you are not using: a "
                "value for a field the plan sweeps is overridden point by point and comes back "
                "when the axis is removed, and a name this Pulse does not declare is applied to "
                "nothing and listed in the run record as unapplied_api_values. A label two of the "
                "Pulse's fields share is refused at the draft; write the field id instead."
            ),
        ),
        # Repeats and shots both land on the dataset's repeat axis (size
        # repeats x shots): shots override the Pulse's Run repeats while the
        # row is held; repeats override complete scan-table sweeps. The
        # independent PulseBracket stays inside every shot.
        AuthoringField("repeats", "int", "Repeats (full sweeps)", 1, minimum=1),
        AuthoringField("shots_per_point", "int", "Shots per point", 1, minimum=1),
    ),
    validator=_validate,
)


def _pulse_ports(pulse: object) -> tuple:
    return (*hardware_scan_ports_for(pulse), *api_scan_ports_for(pulse))


def _admit_resources(values, resources, installation=None) -> tuple[str, ...]:
    """The API values and the pulse axes against what the chosen pulse
    offers, so an API value the pulse cannot take (a label two of its fields
    answer to) or an axis on a port it does not offer is a draft issue the
    Edit page says and Start waits on.  A device axis on a device that is not
    open is the core's to refuse: it cannot claim it."""

    del installation
    plan = plan_from_authored(values["plan"])
    issues = []
    try:
        pulse = pulse_description((resources or {}).get(SEAMLESS_PULSE_RESOURCE.field_name))
    except TypeError:
        return ()
    try:
        resolve_api_overrides(pulse, api_overrides_from_authored(values.get("api_values")))
    except ValueError as error:
        issues.append(str(error))
    pulse_axes = tuple(axis for axis in plan.axes if axis.port.startswith((PULSE_PARAM_FAMILY, API_PARAM_FAMILY)))
    if pulse_axes:
        try:
            bind_plan(ScanPlan(pulse_axes), _pulse_ports(pulse), withheld=unscannable_api_ports(pulse))
        except ValueError as error:
            issues.append(str(error))
    return tuple(issues)


def _tunables(devices) -> dict[str, DeviceKnob]:
    """The claimed devices of the plan's device axes, by device key."""

    return {argument[len(_TUNABLE):]: DeviceKnob(device) for argument, device in dict(devices or {}).items()
            if argument.startswith(_TUNABLE) and not argument.endswith("_key")}


def _build(
    *,
    source_signal: str,
    pulse_resource: object,
    plan: object,
    api_values: object = "",
    repeats: int = 1,
    shots_per_point: int = 1,
    acquisition_logic: str = "",
    sequencer_key: str = "",
    **devices: object,
) -> SeamlessScanMeasurement:
    pulse = pulse_description(pulse_resource)
    # This run's own API values over whatever the pulse carries.  The plan
    # still overrides the ones it sweeps; a value for a name this pulse does
    # not declare is kept in the draft and named in the run record as not
    # applied.
    applied, unapplied = resolve_api_overrides(pulse, api_overrides_from_authored(api_values))
    parsed = plan_from_authored(plan)
    tunables = _tunables(devices)
    # An operator's axis binds to no port at all: the run stops and asks
    # for it.  A device axis binds to a claimed knob the HOST moves between
    # fires, and an API axis to a parameter of the pulse the host plays with
    # another value between fires.  Everything left under them is the table
    # the board plays, if there are board axes.  A host-only plan plays a
    # fixed Pulse.
    bindable = tuple(axis for axis in parsed.axes if not axis.port.startswith(MANUAL_PARAM_FAMILY))
    ports = bind_plan(
        ScanPlan(bindable),
        (*_pulse_ports(pulse),
         *scan_ports_for_devices(tunables, units={axis.port: axis.unit for axis in parsed.axes})),
        withheld=unscannable_api_ports(pulse),
    ) if bindable else ()
    return SeamlessScanMeasurement(
        pulse=pulse,
        plan=parsed,
        ports=ports,
        source_signal=source_signal,
        tunables=tunables,
        repeats=int(repeats),
        shots_per_point=int(shots_per_point),
        acquisition_logic=acquisition_logic,
        api_values=applied,
        unapplied_api_values=unapplied,
        sequencer_key=sequencer_key,
    )


def _estimate_run(values, devices, resources):
    """This Scan's workload from its draft and pulse description; never Load/Fire."""

    del devices
    resource = (resources or {}).get(SEAMLESS_PULSE_RESOURCE.field_name)
    if resource is None:
        return {"status": "unknown", "reason": "Choose a resolvable Pulse template first."}
    result = estimate_scan_time(
        resource, plan_from_authored(values.get("plan", "")),
        api_values=values.get("api_values", ""),
        repeats=values.get("repeats", 1),
        shots_per_point=values.get("shots_per_point", 1),
        # The attached board compiles the plan's programs (the core's pulse_estimate).
        board_estimate=lambda **request: zlc_plugin.core("pulse_estimate", **request),
    )
    result["pulse_path"] = str(pulse_description(resource)["path"])
    return result


def _device_claims(values):
    try:
        return device_claims(plan_from_authored(values.get("plan", "")))
    except (TypeError, ValueError):
        return {}


def _editor_facts(values, resources, devices, source, bundle, artifacts):
    """The plan editor's model (scan.editor): every knob the pulse and the
    lent devices offer, what the plan amounts to and the API values."""

    del source, bundle, artifacts
    try:
        pulse = pulse_description((resources or {}).get(SEAMLESS_PULSE_RESOURCE.field_name))
    except TypeError:
        pulse = None
    tunables = _tunables(devices)
    return scan_editor_facts(values, pulse, scan_ports_for_devices(tunables), tunables)


LOGIC_NODE = NodeType(
    "seamless_scan",
    NodeKind.MEASUREMENT,
    SEAMLESS_SCAN_SCHEMA,
    description=(
        "Collect a selected signal, including processor or fit-parameter outputs, over "
        "Pulse scan slots, device parameters and/or manual axes. Publishes scan with "
        "the original cell data and explicit scan/repeat coordinates for native "
        "Curve/Image/Facet plots and fitting. This measurement starts the selected "
        "acquisition logic, waits for ready and owns Pulse Fire throughout the scan."
    ),
    acquisition_input="acquisition_logic",
    input_specs=(DatasetInput("signal", None, "exact"),),
    outputs=(SCAN_OUTPUT,),
    # A scan is one measurement per point, so its plot is one cell per
    # point.  A plan that leaves nothing to face -- one axis over one
    # number per point -- opens as the curve it is.
    node_previews=(NodePreviewSpec(SCAN_OUTPUT, "facet_grid"),),
    # The core's native board, driven through pulse_* (claimed, as v2 did).
    device_requirements=(DeviceRequirement("sequencer.streamer", "sequencer", ("program",)),),
    build=_build,
    admit_resources=_admit_resources,
    estimate_run=_estimate_run,
    resolve_device_claims=_device_claims,
    editor_facts=_editor_facts,
    # A region drawn on this scan's own plot is a statement about
    # what to sweep next, in the axes the picture is drawn in.
    selection_mappings=SCAN_PLAN_SELECTIONS,
    workspace_resources=(SEAMLESS_PULSE_RESOURCE,),
)


__all__ = ["LOGIC_NODE", "SEAMLESS_SCAN_SCHEMA"]
