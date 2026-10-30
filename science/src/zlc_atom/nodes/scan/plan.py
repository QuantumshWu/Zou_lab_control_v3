"""What a scan IS: ordered axes over parameter ports, as pure data, and the
template that offers them.

A scan is not a kind of measurement logic.  It is a declaration -- which
knobs, which values, in which nesting order -- and everything else derives
from it: the execution steps, the dataset axes, the editor's form.  The
vocabulary for "which knob" is a PORT, projected from whatever owns the knob;
nothing here invents one, so a plan can never name a parameter its pulse does
not declare.

The port family decides whether the board or host advances each axis:

* ``pulse:param:<field_id>`` -- a stable Pulse field reference. Seamless uses
  fields marked Scan. The board advances the table.
* ``device:<key>:<field>`` -- a runtime knob on an installed device.  Only the
  host can move it, with a ``tune(field, value)`` call before the point fires,
  between board-table segments.

What a pulse offers is the core's description of it (_framework.pulse): its
scan slots (``scan_fields``), its API parameters (``api_fields``) and the API
parameters it offers no plan (``unscannable``), each with the column facts a
port needs -- the plugin decodes no pulse.
"""

from __future__ import annotations

import ast
import operator
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Real

import numpy as np

from zlc_data.units import DEFAULT_UNITS, VoltageIntoLoad, format_quantity
from zlc_durable import strict_json_loads

from zlc_atom.authoring import TunableField, read_tunable_in_unit
from zlc_atom.nodes._framework.descriptor import (
    SelectionMapping,
    WorkspaceResourceSpec,
)
from zlc_atom.nodes._framework.pulse import pulse_description


PULSE_PARAM_FAMILY = "pulse:param:"

#: ``device:<key>:<field>`` -- a runtime knob on an installed device.  Its
#: advance is a ``tune(field, value)`` call before the point fires; the board
#: cannot advance it itself, so a board-advanced plan refuses it.
DEVICE_PARAM_FAMILY = "device:"

#: ``api:<field id>`` -- one of the pulse's API parameters, walked by the
#: HOST the way a caller of the API would walk it: before the point fires
#: the value is written into the pulse and the board is loaded again.  An
#: API parameter is a number in the program, not a slot in the board's
#: table, so like a device knob it stands outside every board-advanced
#: axis.  A parameter that is also a slot is the board's to sweep, from one
#: load, and is not offered here.
API_PARAM_FAMILY = "api:"

#: ``manual:<name>`` -- a knob no machine here can reach.  Nothing advances
#: it: the run stops, the OPERATOR moves it, and the run continues.  That is
#: what makes power, polarization and anything else that lives behind a
#: thumbscrew scannable at all, and it is why such an axis always stands
#: outside every axis a machine advances.
MANUAL_PARAM_FAMILY = "manual:"

#: What a file must be to be a scan's pulse template.
SCAN_PULSE_CONTRACT = "zlc.pulse/scan-template"


@dataclass(frozen=True)
class ScanPort:
    """One knob the bench offers a scan, projected from its owner."""

    port: str
    label: str
    unit: str
    lo: float
    hi: float
    #: Where the knob stands now, in ``unit`` -- the slot's value in the
    #: pulse, the value this run gives an API parameter, the device's
    #: reading -- or None when nothing says.  The editor centres an axis's
    #: first sweep on it.
    current: float | None = None
    #: A whole-number knob (a DAC code, a repeat count, an int field).
    integer: bool = False
    #: The unit family its owner shows it in by default (AuthoringField.display_unit).
    display_unit: str | None = None

    def __post_init__(self) -> None:
        if not str(self.port):
            raise ValueError("a scan port needs a name")
        if not math.isfinite(self.lo) or not math.isfinite(self.hi) or self.lo > self.hi:
            raise ValueError(f"port {self.port!r} has no usable range")
        if self.current is not None and not math.isfinite(self.current):
            raise ValueError(f"port {self.port!r} stands at a non-finite value")


def device_port_parts(port: str) -> tuple[str, str]:
    """``device:<key>:<field>`` as its installed-device key and field name."""

    text = str(port)
    if not text.startswith(DEVICE_PARAM_FAMILY):
        raise ValueError(f"{port!r} is not a device port")
    key, separator, field = text[len(DEVICE_PARAM_FAMILY):].partition(":")
    if not separator or not key or not field:
        raise ValueError(f"{port!r} names no device field")
    return key, field


def port_label(port: str) -> str:
    """The stable axis-identity spelling derived from an authored port.

    Display names belong to ScanPort.label; changing a Period name must
    not change the axis ID selectors and saved plans reference.
    """

    text = str(port)
    if text.startswith(PULSE_PARAM_FAMILY):
        return text[len(PULSE_PARAM_FAMILY):]
    if text.startswith(API_PARAM_FAMILY):
        return text[len(API_PARAM_FAMILY):]
    if text.startswith(DEVICE_PARAM_FAMILY):
        return ".".join(device_port_parts(text))
    if text.startswith(MANUAL_PARAM_FAMILY):
        # The one definition of a manual axis's label, here with the others.
        return manual_axis_name(text)
    raise ValueError(f"{port!r} belongs to no known port family")


def port_group(port: str) -> str:
    """Which branch a port hangs under, derived from the port itself.

    Beside :func:`port_label` for the same reason: what a knob is called and
    where it is found must not be able to drift apart.  A device's knobs
    gather under THE DEVICE, because that is how an operator looks for one --
    "what can the RF source sweep" -- and not in one flat list where a pulse
    parameter and a laser current sit next to each other by accident.
    """

    text = str(port)
    if text.startswith(PULSE_PARAM_FAMILY):
        return "pulse"
    if text.startswith(API_PARAM_FAMILY):
        return "api"
    if text.startswith(DEVICE_PARAM_FAMILY):
        return device_port_parts(text)[0]
    if text.startswith(MANUAL_PARAM_FAMILY):
        return "manual"
    raise ValueError(f"{port!r} belongs to no known port family")


def port_leaf(port: str) -> str:
    """The port's name WITHIN its group -- the label, less the branch."""

    text = str(port)
    if text.startswith(DEVICE_PARAM_FAMILY):
        return device_port_parts(text)[1]
    return port_label(port)


def host_advanced_port(port: str) -> bool:
    """Whether the HOST advances this port between fires of the board.

    Three families qualify, for one structural reason: the board plays its
    whole table from a single load, and neither a hand on a thumbscrew, a
    ``tune()`` call on an installed device nor a new value in the program
    can reach inside that.  All three therefore stand outside every
    board-advanced axis, walked between segments -- the run pauses, the
    knob moves (by hand, by call, or by writing the pulse and loading it
    again), the next segment fires.
    """

    text = str(port)
    return (
        text.startswith(MANUAL_PARAM_FAMILY)
        or text.startswith(DEVICE_PARAM_FAMILY)
        or text.startswith(API_PARAM_FAMILY)
    )


def scan_axis_ids(labels: Sequence[str]) -> tuple[str, ...]:
    """What the dataset calls each axis a plan sweeps, from the plan alone.

    One spelling, shared by the writer that names the axes and by the
    selection that reads a name back off a published dataset.  Two ports
    may carry one human name -- a manual knob and a pulse parameter both
    called ``bias`` -- so the second and later namesakes take their rank
    among the namesakes as a suffix.  The ids are a pure function of the
    plan's labels in order, which is what lets a range drawn on
    ``scan.bias.2`` land on the axis the picture drew and not on the first
    axis that happened to share the word.
    """

    seen: dict[str, int] = {}
    ids = []
    for label in labels:
        base = f"scan.{label}"
        rank = seen.get(base, 0) + 1
        seen[base] = rank
        ids.append(base if rank == 1 else f"{base}.{rank}")
    return tuple(ids)


def _ports_from_columns(columns, *, family: str = PULSE_PARAM_FAMILY) -> tuple[ScanPort, ...]:
    """One port per column of the description (``field_id, label, kind, unit,
    value, is_dac, limit_lo, limit_hi``): the range a plan may play is the
    column's limit, and it stands at the field's value in the pulse."""

    ports = []
    for column in columns:
        # The port's unit becomes the dataset axis's unit, and an axis unit is
        # something the plot's registry RESOLVES -- "s", "ms", "" -- not a
        # phrase.  A DAC code is a count of codes ("code"), not the editor's
        # label for it ("DAC code (0 = 0 V)").
        ports.append(
            ScanPort(
                family + str(column["field_id"]),
                str(column["label"]),
                "code" if column.get("is_dac") else str(column["unit"]),
                float(column["limit_lo"]),
                float(column["limit_hi"]),
                current=float(column["value"]),
                integer=bool(column.get("is_dac")) or column.get("kind") == "bracket_count",
            )
        )
    return tuple(ports)


def hardware_scan_ports_for(description: Mapping[str, object]) -> tuple[ScanPort, ...]:
    """Every hardware-slot port this pulse offers -- the SEAMLESS vocabulary.

    A seamless scan is the BOARD advancing its own slot table, so what it
    can vary is exactly the scan slots the template's author placed, in
    slot order.
    """

    return _ports_from_columns(pulse_description(description).get("scan_fields", ()))


def api_scan_ports_for(description: Mapping[str, object]) -> tuple[ScanPort, ...]:
    """Every API parameter this pulse offers the HOST to walk, in parameter
    order -- the ones no board slot already sweeps.

    The port's range is the column's own limit, the same one the API values
    form prints, because the value is written into the same field.
    """

    description = pulse_description(description)
    slots = {column["field_id"] for column in description.get("scan_fields", ())}
    withheld = unscannable_api_ports(description)
    return _ports_from_columns(
        tuple(
            column
            for column in description["api_fields"]
            if column["field_id"] not in slots and API_PARAM_FAMILY + column["field_id"] not in withheld
        ),
        family=API_PARAM_FAMILY,
    )


def unscannable_api_ports(description: Mapping[str, object]) -> dict[str, str]:
    """The API parameters this pulse carries but offers no plan, each port
    with the sentence that says why (an empty bracket loops nothing, so
    walking its repeat count would play the same pulse at every point).
    :func:`bind_plan` refuses a plan that walks one with this reason."""

    return {API_PARAM_FAMILY + str(field): str(reason)
            for field, reason in dict(pulse_description(description).get("unscannable") or {}).items()}


def label_device_scan_ports(
    ports: Sequence[ScanPort], device_labels: Mapping[str, str] | None,
) -> tuple[ScanPort, ...]:
    """Project accepted Role metadata without reading or renaming a device."""

    labels = device_labels or {}
    result = []
    for port in ports:
        if port.port.startswith(DEVICE_PARAM_FAMILY):
            key, field = device_port_parts(port.port)
            label = f"{labels.get(key, key)}.{field}"
            if label != port.label:
                port = replace(port, label=label)
        result.append(port)
    return tuple(result)


def scan_ports_for_devices(
    tunables: Mapping | None, *, units: Mapping[str, str] | None = None,
    device_labels: Mapping[str, str] | None = None,
) -> tuple[ScanPort, ...]:
    """Every port the bench's tunable devices offer, from their own words.

    ``tunables`` is already the tunable set -- ``tunable_devices`` chose it
    by ``is_tunable`` -- so each device is asked for its
    ``tunable_fields()`` without being judged again.  A scan exposes only a
    bounded, live-writable field whose dependency group is that field alone:
    this executor advances one scalar port at a time and cannot pretend a
    coupled hardware transaction is atomic.  A field whose bounds leave no
    interval -- a knob pinned to one value by policy -- is a control, not
    an axis, and is passed over rather than failing the whole projection.
    The port's unit is the field's own declared unit -- an RF frequency
    axis publishes hertz -- and its label names the device and the knob;
    it stands at the device's reading, in that unit.
    """

    ports: list[ScanPort] = []
    for key in sorted(dict(tunables or {})):
        device = tunables[key]
        for tunable in device.tunable_fields():
            if not isinstance(tunable, TunableField):
                raise TypeError("device tunable_fields must contain TunableField values")
            field = tunable.metadata
            port = f"{DEVICE_PARAM_FAMILY}{key}:{field.name}"
            selected_unit = (units or {}).get(port)
            if selected_unit:
                tunable = read_tunable_in_unit(device, field.name, selected_unit)
                field = tunable.metadata
            if field.minimum is None or field.maximum is None:
                continue
            if float(field.minimum) >= float(field.maximum):
                continue
            if not tunable.live_write or tunable.dependency_group != (field.name,):
                continue
            reading = tunable.current
            ports.append(
                ScanPort(
                    port,
                    port_label(port),
                    field.unit or "",
                    float(field.minimum),
                    float(field.maximum),
                    current=float(reading) if isinstance(reading, Real) and not isinstance(reading, bool)
                    and math.isfinite(reading) else None,
                    integer=field.value_type == "int",
                    display_unit=field.display_unit,
                )
            )
    return label_device_scan_ports(ports, device_labels if device_labels is not None else
                                   {key: getattr(device, "role", "") or key for key, device in dict(tunables or {}).items()})


class DeviceKnob:
    """A claimed device as a scan reads and moves it.

    Reached through the core, a device answers its knobs as data (the rows
    Device Control shows); this reads them back into the plugin's own
    ``TunableField`` and forwards the moves, so the scan's device law
    (zlc_atom.authoring: read in a unit, tune in a unit) is the one every
    knob is moved by.  A device that converts its own units (an RF source's
    amplitude into its load) keeps doing so through ``tune_in_unit``.
    """

    def __init__(self, device: object) -> None:
        self._device = device
        self.role = getattr(device, "role", "")
        native = getattr(device, "tune_in_unit", None)
        if callable(native):
            self.tune_in_unit = native
        # A device's own conversions (an RF amplitude into its channel's load), when it has them.
        convert = getattr(device, "convert_tunable_value", None)
        if callable(convert):
            self.convert_tunable_value = convert
        if callable(getattr(device, "read_tunable_in_unit", None)):
            self.read_tunable_in_unit = lambda name, unit="": self._field(device.read_tunable_in_unit(name, unit))

    def tunable_fields(self) -> tuple[TunableField, ...]:
        return tuple(self._field(item) for item in self._device.tunable_fields())

    @staticmethod
    def _field(item: object) -> TunableField:
        """A knob as the plugin's own TunableField, from the row a device answers through the core."""

        from zlc_atom.authoring import AuthoringChoice, AuthoringField

        if isinstance(item, TunableField):
            return item
        row = dict(item["field"])
        metadata = AuthoringField(
            row["name"], row["type"], row["label"], row.get("default"), required=bool(row.get("required")),
            minimum=row.get("minimum"), maximum=row.get("maximum"), step=row.get("step"), unit=row.get("unit"),
            display_unit=row.get("display_unit"), choices=tuple(AuthoringChoice(choice["value"], choice["label"]) for choice in row.get("choices") or ()),
            description=str(row.get("description") or ""))
        limits = item.get("device_limits")
        return TunableField(metadata, item.get("current"), bool(item["live_write"]),
                            tuple(item["dependency_group"]), None if limits is None else tuple(limits))

    def tune(self, name: str, value: object) -> object:
        return self._device.tune(name, value)

    def tunable_values(self) -> Mapping[str, object]:
        return self._device.tunable_values()

    def settings_provenance(self) -> Mapping[str, object]:
        return self._device.settings_provenance()


@dataclass(frozen=True)
class ScanAxis:
    """One axis: a port and the values it takes, in play order."""

    port: str
    values: tuple[float, ...]
    unit: str = ""
    coordinate_of: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "port", str(self.port))
        values = tuple(float(value) for value in self.values)
        if not values:
            raise ValueError(f"axis {self.port!r} has no values to play")
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"axis {self.port!r} contains a non-finite value")
        object.__setattr__(self, "values", values)
        if not isinstance(self.unit, str):
            raise TypeError("scan axis unit must be text")
        object.__setattr__(self, "unit", self.unit.strip())

    def native_value(self, port: ScanPort, value: float) -> float:
        """Convert an authored coordinate only at the port boundary."""
        unit = self.unit or port.unit
        native = float(value) if unit == port.unit else float(
            DEFAULT_UNITS.convert(value, unit, port.unit)
        )
        if not math.isfinite(native):
            raise ValueError(f"axis {self.port!r} converts to a non-finite port value")
        return native


@dataclass(frozen=True)
class ScanPlan:
    """Ordered axes, outermost first.  The whole scan, as one document."""

    axes: tuple[ScanAxis, ...]

    def __post_init__(self) -> None:
        axes = tuple(self.axes)
        if not axes:
            raise ValueError("a scan plan needs at least one axis")
        if any(not isinstance(axis, ScanAxis) for axis in axes):
            raise TypeError("plan axes must be ScanAxis values")
        names = tuple(axis.port for axis in axes)
        if len(set(names)) != len(names):
            raise ValueError("a port may appear on one axis only")
        object.__setattr__(self, "axes", tuple(sorted(
            axes, key=lambda axis: any(host_advanced_port(item.port) for item in axes if (item.coordinate_of or item.port) == (axis.coordinate_of or axis.port)), reverse=True,
        )))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(axis.values) for axis in self.axes if axis.coordinate_of is None)

    @property
    def point_count(self) -> int:
        return math.prod(self.shape)

    def rows(self) -> tuple[tuple[float, ...], ...]:
        """Every point, one value per axis, the LAST axis advancing fastest.

        That makes the declared order the nesting order: axes[0] is the
        outermost loop, exactly as an operator reads the plan.
        """

        return scan_rows(self.axes)

    def to_tree(self) -> dict:
        return {
            "axes": [
                {"port": axis.port, "values": list(axis.values), "unit": axis.unit, **({"coordinate_of": axis.coordinate_of} if axis.coordinate_of else {})}
                for axis in self.axes
            ]
        }

    @classmethod
    def from_tree(cls, tree: object) -> "ScanPlan":
        """Compile every listed axis from its selected input bank."""

        rows = plan_input_rows(tree)
        axes = []
        for index, entry in enumerate(rows, 1):
            try:
                if entry["port"].startswith(MANUAL_PARAM_FAMILY):
                    manual_axis_name(entry["port"])
                values = (
                    parse_scan_values(entry["value_text"])
                    if entry["mode"] == "values" else entry["values"]
                )
                axes.append(ScanAxis(entry["port"], tuple(values), entry["unit"], entry.get("coordinate_of")))
                if entry.get("link_enabled"):
                    for linked in entry.get("links", ()):
                        if linked["mode"] == "formula":
                            played = formula_values(linked["formula"], values)
                        elif linked["mode"] == "values":
                            played = parse_scan_values(linked["value_text"])
                        else:
                            ends = linked["values"]
                            if not ends:
                                raise ValueError("linked Range needs endpoints")
                            played = tuple(ends[0] + (ends[-1] - ends[0]) * i / max(1, len(values)-1) for i in range(len(values)))
                        if len(played) != len(values):
                            raise ValueError(f"linked parameter {linked['port']!r} needs {len(values)} values")
                        if linked["port"].startswith(MANUAL_PARAM_FAMILY):
                            raise ValueError("a linked parameter must be a scan slot, API or device parameter")
                        axes.append(ScanAxis(linked["port"], tuple(played), linked["unit"], entry["port"]))
            except (TypeError, ValueError) as error:
                raise ValueError(f"scan axis {index} ({entry['port']!r}): {error}") from None
        return cls(tuple(axes))


def formula_values(source: str, values: Sequence[float]) -> tuple[float, ...]:
    """Numeric arithmetic only: x is the parent number; units belong to the result."""
    if len(source) > 2048:
        raise ValueError("formula is too long")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as error:
        raise ValueError("invalid formula") from error
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}
    if len(list(ast.walk(tree))) > 128:
        raise ValueError("formula is too complex")
    def read(node, x):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float): return float(node.value)
        if isinstance(node, ast.Name) and node.id == "x": return float(x)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return read(node.operand, x) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            a, b = read(node.left, x), read(node.right, x)
            if isinstance(node.op, ast.Pow) and abs(b) > 64: raise ValueError("formula exponent exceeds 64")
            return operations[type(node.op)](a, b)
        raise ValueError("formula accepts numbers, x, parentheses and + - * / **")
    try:
        result = tuple(float(read(tree.body, x)) for x in values)
        if not all(math.isfinite(x) for x in result): raise ValueError("formula must produce finite numbers")
        return result
    except (ArithmeticError, TypeError, OverflowError) as error:
        raise ValueError(f"cannot evaluate formula: {error}") from error


def scan_rows(axes: Sequence[ScanAxis]) -> tuple[tuple[float, ...], ...]:
    """Independent groups form a product; parameters of one group advance together."""
    groups = tuple(dict.fromkeys(axis.coordinate_of or axis.port for axis in axes))
    lengths = {group: len(next(axis for axis in axes if (axis.coordinate_of or axis.port) == group).values) for group in groups}
    for axis in axes:
        if len(axis.values) != lengths[axis.coordinate_of or axis.port]: raise ValueError("linked parameters must have equal lengths")
    return tuple(tuple(axis.values[indices[groups.index(axis.coordinate_of or axis.port)]] for axis in axes)
                 for indices in itertools.product(*(range(lengths[group]) for group in groups)))


def manual_axis_name(port: str) -> str:
    """The operator-facing name behind a ``manual:`` port."""

    text = str(port)
    if not text.startswith(MANUAL_PARAM_FAMILY):
        raise ValueError(f"{port!r} is not a manual axis")
    name = text[len(MANUAL_PARAM_FAMILY):].strip()
    if not name:
        raise ValueError("a manual axis carries a name")
    return name


def split_outer_axes(plan: ScanPlan) -> tuple[tuple[ScanAxis, ...], tuple[ScanAxis, ...]]:
    """The host-advanced axes, and the plan the board plays underneath.

    A manual axis is walked by hand and a device axis by a ``tune()`` call,
    both BETWEEN plays of the inner plan -- not a preference, a fact about
    who moves what: the inner plan plays from one load, and neither a hand
    nor a host call can reach inside it. ScanPlan already stably places
    these axes outside board axes; the editor displays that same order.
    An empty board part means a fixed Pulse at each host point, not a fake
    scan axis or an empty public ScanPlan.
    """

    axes = plan.axes
    outer = tuple(axis for axis in axes if any(host_advanced_port(item.port) for item in axes if (item.coordinate_of or item.port) == (axis.coordinate_of or axis.port)))
    return outer, axes[len(outer):]


def bind_plan(
    plan: ScanPlan,
    ports: Sequence[ScanPort],
    *,
    withheld: Mapping[str, str] | None = None,
) -> tuple[ScanPort, ...]:
    """The port behind each axis, or the refusal that names what is wrong.

    Binding is where a plan meets a bench: an axis naming a port this pulse
    does not offer, or promising values outside the port's hard limits, is
    refused HERE, before anything touches a device.  ``withheld`` maps a
    port the pulse has but does not offer to why
    (:func:`unscannable_api_ports`), and an axis naming one is refused with
    that reason.
    """

    by_name = {port.port: port for port in ports}
    bound = []
    for axis in plan.axes:
        port = by_name.get(axis.port)
        if port is None and axis.port in (withheld or {}):
            raise ValueError(f"{withheld[axis.port]}; remove axis {axis.port!r} or put a period inside the bracket")
        if port is None:
            offered = ", ".join(sorted(by_name)) or "nothing"
            raise ValueError(
                f"this pulse offers no scan port named {axis.port!r}; "
                f"it offers {offered}"
            )
        for value in axis.values:
            native = axis.native_value(port, value)
            if native < port.lo or native > port.hi:
                raise ValueError(
                    f"axis {axis.port!r} plays {value!r} {axis.unit or port.unit}, outside the port's "
                    f"range [{port.lo:g}, {port.hi:g}] {port.unit}"
                )
        bound.append(port)
    return tuple(bound)


#: A seamless plan may carry only host axes and play an ordinary fixed Pulse;
#: any declared board slots still bind to the plan in the scan owner.
SEAMLESS_PULSE_RESOURCE = WorkspaceResourceSpec(
    "pulse_template",
    SCAN_PULSE_CONTRACT,
    "pulses",
    (".json",),
    None,
    argument_name="pulse_resource",
)


def api_overrides_from_authored(payload: object) -> dict[str, float]:
    """One run's API parameter values, from the authored ``api_values`` text.

    ``name = number`` a line, in the unit the pulse declares for that name --
    the same one the form prints beside the box.  Only what this run sets
    differently is written down: a parameter left alone keeps whatever the
    pulse carries, which is where the workspace's current set of values has
    already landed.
    """

    text = "" if payload is None else str(payload)
    overrides: dict[str, float] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not name:
            raise ValueError(
                f"API value line {number} must read 'name = number': {raw.strip()!r}"
            )
        try:
            overrides[name] = float(value.strip())
        except ValueError:
            raise ValueError(
                f"API value {name!r} is not a number: {value.strip()!r}"
            ) from None
    return overrides


def resolve_api_overrides(
    description: Mapping[str, object], overrides: Mapping[str, float]
) -> tuple[dict[str, tuple[float, str]], dict[str, float]]:
    """This run's API values as the board takes them -- ``{field_id: (value,
    unit)}`` in each field's declared unit -- and the values it does not
    declare.

    A name may be a field id or a label one field answers to; a label two
    fields answer to is refused (:func:`ambiguous_api_value`).  A name the
    pulse does not declare -- a value kept from another pulse, or for a field
    its Subpulse file no longer offers -- is parked: applied to nothing and
    returned, so the run record names it as not applied, as the editor shows
    it.  The board converts and checks each value (``pulse_run``).
    """

    fields = pulse_description(description)["api_fields"]
    declared = {field["field_id"]: str(field["unit"]) for field in fields}
    labelled: dict[str, list[str]] = {}
    for field in fields:
        labelled.setdefault(str(field["label"]), []).append(field["field_id"])
    applied: dict[str, tuple[float, str]] = {}
    unapplied: dict[str, float] = {}
    for name, value in overrides.items():
        if name in declared:
            field_id = name
        elif len(labelled.get(name, ())) > 1:
            raise ValueError(ambiguous_api_value(name, labelled[name]))
        elif name in labelled:
            field_id = labelled[name][0]
        else:
            unapplied[name] = float(value)
            continue
        applied[field_id] = (float(value), declared[field_id])
    return applied, unapplied


def ambiguous_api_value(name: str, field_ids: Sequence[str]) -> str:
    """Why an API value written by a label several of the pulse's fields
    answer to cannot be applied: the sentence the draft and the build refuse
    it with, and the one the Scan editor writes on its line."""

    return (
        f"API value {name!r} names {len(field_ids)} fields of this pulse "
        f"({', '.join(field_ids)}); write the field id of the one it sets"
    )


def api_overrides_to_authored(overrides: Mapping[str, float]) -> str:
    """The text behind one mapping of API parameter values."""

    lines = [
        f"{name} = {format_quantity(value, '1')}" for name, value in overrides.items()
    ]
    return "\n".join(lines)


def plan_input_rows(payload: object) -> tuple[dict, ...]:
    """Read the two independent input banks without compiling Values text."""
    if isinstance(payload, ScanPlan):
        payload = payload.to_tree()
    elif isinstance(payload, str):
        if not payload.strip():
            raise ValueError("the scan plan is empty; add an axis")
        payload = strict_json_loads(payload, "the scan plan")
    if not isinstance(payload, Mapping) or set(payload) != {"axes"}:
        raise ValueError("a scan plan document carries only its axes")
    entries = payload["axes"]
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise TypeError("scan plan axes must be a list")
    rows = []
    for index, entry in enumerate(entries, 1):
        allowed = {"port", "values", "unit", "mode", "value_text", "formula", "links", "link_enabled", "coordinate_of"}
        if not isinstance(entry, Mapping) or not {"port", "values"} <= set(entry) or set(entry) - allowed:
            raise ValueError("scan axis fields must be port, values, and optional unit, mode, value_text")
        row = {"port": entry["port"], "values": entry["values"], "unit": entry.get("unit", ""),
               "mode": entry.get("mode", "range"), "value_text": entry.get("value_text", "")}
        if "coordinate_of" in entry:
            row["coordinate_of"] = entry["coordinate_of"]
        if "links" in entry or "link_enabled" in entry:
            if not isinstance(entry.get("link_enabled", False), bool):
                raise TypeError("link_enabled must be boolean")
            row["link_enabled"] = entry.get("link_enabled", False)
            raw_links = entry.get("links", [])
            if not isinstance(raw_links, list):
                raise TypeError("linked parameters must be a list")
            row["links"] = []
            for linked in raw_links:
                if not isinstance(linked, Mapping) or set(linked) - {"port", "values", "unit", "mode", "value_text", "formula"}:
                    raise ValueError("invalid linked parameter fields")
                mode = linked.get("mode", "formula")
                if mode not in ("range", "values", "formula"):
                    raise ValueError("linked mode must be range, values or formula")
                clean = plan_input_rows({"axes": [{**{k:v for k,v in linked.items() if k != "formula"}, "mode": "range" if mode == "formula" else mode}]})[0]
                clean["mode"] = mode
                clean["formula"] = linked.get("formula", "x")
                if not isinstance(clean["formula"], str):
                    raise TypeError("formula must be text")
                row["links"].append(clean)
        if any(not isinstance(row[key], str) for key in ("port", "unit", "mode", "value_text")):
            raise TypeError(f"scan axis {index}: port, unit, mode and value_text must be text")
        if row["mode"] not in ("range", "values"):
            raise ValueError(f"scan axis {index} ({row['port']!r}): mode must be range or values")
        values = row["values"]
        if (isinstance(values, (str, bytes)) or not isinstance(values, Sequence)
                or any(isinstance(value, bool) or not isinstance(value, Real) for value in values)):
            raise TypeError(f"scan axis {index} ({row['port']!r}): Range values must be a numeric list")
        row["values"] = list(values)
        rows.append(row)
    return tuple(sorted(rows, key=lambda row: any(host_advanced_port(item["port"]) for item in (row, *(row.get("links", ()) if row.get("link_enabled") else ()))), reverse=True))


def parse_scan_values(text: str) -> tuple[float, ...]:
    """Comma-separated finite numbers, retaining the authored order/repeats."""
    if not isinstance(text, str):
        raise TypeError("Values must be comma-separated text")
    if not text.strip():
        raise ValueError("Values is empty; enter comma-separated numbers")
    values = []
    for index, item in enumerate(text.split(","), 1):
        try:
            value = float(item.strip())
        except ValueError:
            raise ValueError(f"Values item {index} is not a number: {item!r}") from None
        if not math.isfinite(value):
            raise ValueError(f"Values item {index} must be finite")
        values.append(value)
    return tuple(values)


def plan_from_authored(payload: object) -> ScanPlan:
    """Compile the active bank to the existing immutable execution plan."""
    return payload if isinstance(payload, ScanPlan) else ScanPlan.from_tree(payload)


def _scanned_axes(context: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    """Each axis of the dataset a region was drawn on, by its id, as the run
    that wrote it recorded it: the port it swept and the values it played.

    The ids are :func:`scan_axis_ids` of THAT run's plan -- the spelling its
    writer named the axes by -- so a region lands on the port the picture
    drew even after the plan has gained, lost or reordered a namesake.
    """

    record = context.get("run_record")
    plan = record.get("plan") if isinstance(record, Mapping) else None
    axes = plan.get("axes") if isinstance(plan, Mapping) else None
    if not isinstance(axes, Sequence) or isinstance(axes, (str, bytes)):
        raise ValueError("the region was drawn on data whose run record names no scan plan")
    return dict(zip(scan_axis_ids([port_label(axis["port"]) for axis in axes]), axes, strict=True))


def _bounds_in_row_unit(
    axis_id: str, row: Mapping[str, object], bounds: tuple[float, float], source_unit: str,
) -> tuple[float, float]:
    """A region's walls on one axis, from the drawn dataset's unit into the
    row's, low first."""

    unit = row["unit"] or source_unit
    if source_unit == unit:
        return bounds
    # A device's knob converts through the device: an amplitude in Vpp is a
    # power only into the channel's own load and waveform.  Read here at the
    # registry's 50-ohm sine, a region drawn in mVpp over a row since
    # switched to dBm filled another load's power.  A prefix move, or a power
    # read in mW or dBm, is the same number into any load; a move through an
    # amplitude is refused by name.
    families = (DEFAULT_UNITS.family_of(source_unit or "1"), DEFAULT_UNITS.family_of(unit or "1"))
    if (
        row["port"].startswith(DEVICE_PARAM_FAMILY)
        and families[0] != families[1]
        and any(isinstance(family.conversion, VoltageIntoLoad) for family in families)
    ):
        raise ValueError(
            f"selected scan axis {axis_id!r} row is in {unit}, region drawn in "
            f"{source_unit}: that conversion is the device's; set the row to "
            f"{source_unit} to draw it"
        )
    converted = DEFAULT_UNITS.convert(bounds, source_unit or "1", unit or "1")
    # A wall the row's unit has no value for -- below 0 mW or 0 Vpp, for a
    # row now in dBm -- is refused by name.  Written as NaN it made the draft
    # no plan at all, and the editor reading it back retired every row the
    # operator had authored.
    for name, bound, value in zip(("low", "high"), bounds, converted):
        if not math.isfinite(float(value)):
            raise ValueError(
                f"selected scan axis {axis_id!r} {name} {bound:g} {source_unit} "
                f"has no value in {unit}: draw the region where it has one"
            )
    low, high = sorted(float(value) for value in converted)
    return low, high


def _selected_plan(
    selection: object,
    draft: Mapping[str, object],
    context: Mapping[str, object],
) -> dict[str, object] | None:
    """The plan the region was drawn over, narrowed to it.

    A box or an x range on a scan's own plot names SCANNED axes -- that is
    what the dataset's axes are -- so it says "sweep this part next".  The
    run that wrote the dataset says which port each axis swept and what it
    played (its run record, :func:`_scanned_axes`); the region lands on the
    draft's row for THAT port.  An axis the region does not name is
    untouched, and so is one it spans whole -- a box the full height of an
    image narrows its columns only -- or one of a single point.  A region
    that names no scanned axis -- a camera ROI drawn on the frames a scan
    captured -- sets nothing: the frames belong to the camera.  Nor does a
    facet cell or a Scope pin the axis it shows one coordinate of: that is
    where the operator looked, not what was drawn.

    A Range row keeps its point COUNT and its direction, because that is
    what the plan's form offers (from, to, points): the same effort spent
    over a smaller range, walked the way it was authored.  Its new ends are
    held to the span the scan played: past it -- the half pixel beyond an
    image's last column, a curve's margin -- the picture shows nothing
    scanned, and a range drawn there would sweep past the knob's limits
    when the scan ran up to them.  A Values row is an exact statement and
    stays one: the listed values the region holds, in their order and
    spelling.  A region holding nothing scanned on an axis it cuts is
    refused by name.
    """

    rows = plan_input_rows(draft.get("plan"))
    scanned = _scanned_axes(context)
    by_port = {row["port"]: row for row in rows}
    named = changed = False
    for chosen in getattr(selection, "ranges", ()):
        axis_id = str(getattr(chosen, "axis", ""))
        recorded = scanned.get(axis_id)
        row = None if recorded is None else by_port.get(str(recorded["port"]))
        if row is None:
            continue
        named = True
        listed = row["mode"] == "values"
        items = [item.strip() for item in row["value_text"].split(",")] if listed else []
        authored = parse_scan_values(row["value_text"]) if listed else tuple(row["values"])
        if len(authored) < 2:
            continue
        played = [float(value) for value in recorded["values"]]
        first, last = min(played), max(played)
        lower, upper = float(chosen.lower), float(chosen.upper)
        if lower <= first and upper >= last:
            continue
        source_unit = context.get("axis_units", {}).get(axis_id)
        if source_unit is None:
            raise ValueError(f"selected scan axis {axis_id!r} has no recorded unit")
        if not listed:
            lower, upper = max(lower, first), min(upper, last)
        if not lower < upper:
            raise ValueError(
                f"selected scan axis {axis_id!r}: the region lies outside the "
                f"{first:g} to {last:g} {source_unit} the scan played"
            )
        low, high = _bounds_in_row_unit(axis_id, row, (lower, upper), source_unit)
        if listed:
            kept = [item for item, value in zip(items, authored, strict=True) if low <= value <= high]
            if not kept:
                raise ValueError(
                    f"selected scan axis {axis_id!r}: the region holds none of its listed values"
                )
            row["value_text"] = ", ".join(kept)
        else:
            values = [float(value) for value in np.linspace(low, high, len(authored))]
            row["values"] = values if authored[0] <= authored[-1] else values[::-1]
        changed = True
    if not named:
        return None
    # The draft asked about is the one the region was drawn over (the core's
    # panel_selection), so a region that now narrows nothing it names gives
    # that plan back as it was written.
    return {"plan": json.dumps({"axes": rows}) if changed else draft.get("plan")}


#: What a region drawn on a scan's plot does to that scan's plan.  A cell of a
#: facet grid reports its OWN kind here -- the grid is a layout, the cell is
#: the picture -- so a box inside one cell narrows the axes that cell draws.
SCAN_PLAN_SELECTIONS = (
    SelectionMapping(
        plot_kind="image",
        selector_kind="area",
        draft_fields=("plan",),
        map_patch=_selected_plan,
    ),
    SelectionMapping(
        plot_kind="curve",
        selector_kind="x_range",
        draft_fields=("plan",),
        map_patch=_selected_plan,
    ),
    SelectionMapping(
        plot_kind="image",
        selector_kind="x_range",
        draft_fields=("plan",),
        map_patch=_selected_plan,
    ),
)


__all__ = [
    "DEVICE_PARAM_FAMILY",
    "SCAN_PLAN_SELECTIONS",
    "PULSE_PARAM_FAMILY",
    "SCAN_PULSE_CONTRACT",
    "SEAMLESS_PULSE_RESOURCE",
    "ScanAxis",
    "ScanPlan",
    "ScanPort",
    "ambiguous_api_value",
    "bind_plan",
    "device_port_parts",
    "hardware_scan_ports_for",
    "plan_from_authored",
    "DeviceKnob",
    "resolve_api_overrides",
    "plan_input_rows",
    "parse_scan_values",
    "scan_ports_for_devices",
]
