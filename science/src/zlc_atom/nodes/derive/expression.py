"""Derive's one engine: variables over existing three-domain Dataset values.

A draft is a list of VARIABLES.  Each is a formula -- its terms, one per
line: an input member or another variable, optionally picked at labels of
one of its axes (or at its first or last, whatever the axis' length), or a
number, or true/false -- joined by operators of four
kinds (arithmetic, comparison, logic, where) under one precedence, then
optionally reduced by a statistic over named axes.  A variable with
``publish`` on is a signal; one with it off is an internal helper that
other variables read.  Only the published variables and what they read
(the LIVE closure) are evaluated; a helper nothing published reads is
PARKED -- typed for the editor, never evaluated, never blocking Start.

Every rule an operation applies to a Dataset's axes, dtype, unit and
validity is a plan that reads schemas only.  Evaluation applies each plan
to arrays (:class:`Operand`) and :func:`typed_variables` applies the same
plans to the source's schemas alone, so the editor's preview and Start's
admission are the evaluation's own answer before any data exists.  The native scientific engine
does the math; nothing here runs operator-written code.

A statistic can reduce one of its axes in consecutive groups (``by`` +
``every``) instead of whole, and ``fit`` is the statistic that eats one axis
and yields a model's parameters, one published signal per parameter.  A group
over an axis of the Cell domain closes inside one evaluation and is array work
here; one over a Repeat or Point axis spans publications, and what stands
between them is :mod:`.groups`.  Its plans are these.
"""
from __future__ import annotations

import functools
import itertools
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from zlc_data import DatasetSchema, DomainSpec, SCALAR_DOMAIN, ValidityContract, ValueSchema
from zlc_data.axis import SCALAR, AxisId, AxisSpec
from zlc_data.selection import EmptySelection, take_indices
from zlc_data.snapshot_projection import _subset_axis, _subset_mapped_domain
from zlc_data.units import DEFAULT_UNITS, UnitError

from . import native_ops

from zlc_atom.authoring import AuthoringChoice, AuthoringField
from zlc_atom.nodes._framework.frames import FIRST, LAST


class DeriveError(ValueError):
    """One variable that cannot be read, typed or evaluated, at one stage:
    ``name``, ``term k`` (1-based) or ``statistic``."""

    def __init__(self, variable: str, stage: str, detail: str):
        super().__init__(f"{variable} · {stage}: {detail}")
        self.variable, self.stage, self.detail = variable, stage, detail


#: spelling -> (category, precedence, label).  Precedence 1 binds loosest;
#: every level is left-associative and there are no parentheses: a group
#: is an internal variable.
OPERATORS = {
    "where": ("where", 1, "where"), "where not": ("where", 1, "where not"),
    "or": ("logic", 2, "or"),
    "and": ("logic", 3, "and"), "and not": ("logic", 3, "and not"),
    "==": ("comparison", 4, "="), "!=": ("comparison", 4, "≠"), ">": ("comparison", 4, ">"),
    "<": ("comparison", 4, "<"), ">=": ("comparison", 4, "≥"), "<=": ("comparison", 4, "≤"),
    "+": ("arithmetic", 5, "+"), "-": ("arithmetic", 5, "−"),
    "*": ("arithmetic", 6, "×"), "/": ("arithmetic", 6, "÷"),
}
#: The four operator kinds, each a fixed set the editor shows as one strip.
CATEGORIES = (
    ("arithmetic", ("+", "-", "*", "/")),
    ("comparison", ("==", "!=", ">", "<", ">=", "<=")),
    ("logic", ("and", "or", "and not")),
    ("where", ("where", "where not")),
)
#: On booleans, as in a plot: ``sum`` is the number true, ``mean`` the
#: fraction true; ``count`` is the number of valid samples of any dtype.
#: ``fit`` reads one axis as x and yields a model's parameters.
STATISTICS = ("sum", "mean", "count", "min", "max", "std", "any", "all", "fit")
#: The statistics a dtype takes, by "is boolean": the rule a reduction is
#: refused by, and the editor greys the rest of the strip by.
STATISTICS_FOR = {True: ("sum", "mean", "count", "any", "all"),
                  False: ("sum", "mean", "count", "min", "max", "std", "fit")}
#: The one draft-level problem that belongs to no variable.
NO_PUBLISHED = "switch Publish on for at least one variable"

#: One line of a variable's formula.
TERM_COLUMNS = (
    AuthoringField("op", "choice", "Operator", "", choices=(
        AuthoringChoice("", "None"),
        *(AuthoringChoice(spelling, label) for spelling, (_kind, _level, label) in OPERATORS.items()),
    ), description="joins this term to the formula above it; the first term has none"),
    AuthoringField("of", "text", "Of", "", description="an input member or another variable (no cycles)"),
    AuthoringField("number", "float", "Number", None, description="a number instead of of"),
    AuthoringField("boolean", "choice", "Boolean", "", choices=(
        AuthoringChoice("", "None"), AuthoringChoice("true", "true"), AuthoringChoice("false", "false"),
    ), description="true or false instead of of; only after a comparison (false < true)"),
    AuthoringField("at", "text", "At", "",
                   description="what of is picked on: the name of one of its axes, in any domain, or a domain "
                               "(Repeat, Point or Cell) with two or more axes; empty picks nothing"),
    AuthoringField("labels", "text_tuple", "Labels", None,
                   description="a list of texts: labels of that axis as the catalog lists them, or rows of that "
                               "domain written 'detuning 2 · power 1', or 'first' and 'last' -- its first and "
                               "last coordinate (row) whatever its length; one drops the axis (the domain's "
                               "axes), several keep those coordinates in axis (row) order, a coordinate two of "
                               "them name once"),
)
#: One variable.
VARIABLE_COLUMNS = (
    AuthoringField("name", "text", "Name", "",
                   description="letters, digits and _, not starting with a digit; a published variable is the "
                               "signal @logic/<node>/<name>"),
    AuthoringField("publish", "bool", "Publish", True,
                   description="on: published as a signal; off: an internal helper other variables read"),
    AuthoringField("terms", "rows", "Terms", (), required=True, columns=TERM_COLUMNS,
                   description="the formula, one row per line, in order"),
    AuthoringField("statistic", "choice", "Statistic", "", choices=(
        AuthoringChoice("", "None"), *(AuthoringChoice(name, name) for name in STATISTICS),
    ), description="reduces the formula's result over the axes named in over"),
    AuthoringField("over", "text_tuple", "Over", None,
                   description="a list of axis names: the axes the statistic reduces, in any domain; "
                               "fit reads exactly one, its x"),
    AuthoringField("by", "text", "By", "",
                   description="one of the over axes, reduced in consecutive groups of every coordinates "
                               "instead of whole: it stays an axis, one coordinate per group"),
    AuthoringField("every", "int", "Every", 0, minimum=0,
                   description="how many coordinates of by make one group; 0 reduces by whole"),
    AuthoringField("model", "text", "Model", "",
                   description="with statistic fit: the model; the variable publishes <name>_<parameter>"),
)
_VARIABLE_DEFAULTS = {column.name: () if column.default is None else column.default for column in VARIABLE_COLUMNS}
#: A term with nothing written: every column at its default, as a draft projects it.
TERM_DEFAULTS = MappingProxyType({column.name: () if column.value_type == "text_tuple" else column.default
                                  for column in TERM_COLUMNS})
_BOOLEANS_MIX = ("booleans do not mix with numbers; compare booleans with booleans, "
                 "or count them with a statistic")
_DOMAINS = ("repeat_domain", "point_domain", "cell_domain")
#: The three domains as a pick names them, in domain order.
DOMAINS = ("Repeat", "Point", "Cell")
#: What a variable's name may be; the same text is what a failure's
#: ``<variable> · <stage>:`` is read back by.
_NAME = r"[A-Za-z_][A-Za-z0-9_]*"


def compact_variable(row: Mapping[str, object]) -> dict[str, object]:
    """A complete variable row with only what says something: what the
    editor emits, Layout saves and a run records.  ``name`` and ``terms``
    always stand; every other column only when it is not its default."""

    compact = {column: value for column, value in row.items()
               if column in ("name", "terms") or value != _VARIABLE_DEFAULTS[column]}
    compact["terms"] = tuple({column: value for column, value in term.items() if value != TERM_DEFAULTS[column]}
                             for term in row["terms"])
    return compact


# -- fit models --------------------------------------------------------------

def fit_models() -> Mapping[str, Mapping]:
    """The core's series fit models by id, ``{id, label, parameters: [{name,
    unit_relation}]}``: this module holds no model table.  None while the core
    cannot be asked (only a draft query or a build can ask)."""

    from zlc_plugin import PluginError, fit_models as catalogue

    try:
        return catalogue()
    except (RuntimeError, PluginError):
        return {}


def _model_parameters(model: str) -> tuple[Mapping, ...] | None:
    found = fit_models().get(model)
    return None if found is None else tuple(found["parameters"])


def variable_outputs(row: Mapping[str, object]) -> tuple[str, ...]:
    """The names a variable answers to: its own, or a fit's one per parameter
    (none while its model is not a known one)."""

    if row["statistic"] != "fit":
        return (row["name"],)
    return tuple(f"{row['name']}_{parameter['name']}" for parameter in _model_parameters(row["model"]) or ())


def _parameter_unit(relation: str, x_unit: str, y_unit: str) -> str:
    """A parameter's unit by the catalogue's relation to the x and y units: an
    inverse axis reads as a panel's fit names it (``inverse_symbol``)."""

    if relation == "inverse_axis_0":
        return DEFAULT_UNITS.inverse_symbol(x_unit)
    return {"dimensionless": "1", "value": y_unit, "value_times_axis_0": _unit_power(f"{y_unit}*{x_unit}", 1),
            "axis_0": x_unit, "radian": "rad"}[relation]


# -- schemas -----------------------------------------------------------------

def _domains(schema: DatasetSchema) -> tuple[DomainSpec, DomainSpec, DomainSpec]:
    return tuple(getattr(schema, name) for name in _DOMAINS)


def _unit(schema: DatasetSchema) -> str:
    return schema.value_schema.value_unit or "1"


def _schema(domains, dtype, unit, name=None) -> DatasetSchema:
    cell = domains[2]
    validity = (ValidityContract.value() if cell == SCALAR_DOMAIN
                else ValidityContract.components(*(a.axis_id for a in cell.axes)))
    return DatasetSchema(*domains, ValueSchema(validity, np.dtype(dtype), unit or "1", name=name))


def _normal(schema: DatasetSchema) -> DatasetSchema:
    """A member's schema as an operand carries it, while typed as while evaluated."""
    return _schema(_domains(schema), schema.value_schema.dtype, _unit(schema), schema.value_schema.name)


def _unit_power(unit, exponent):
    powers = {}
    for term in (unit or "1").split("*"):
        symbol, sep, power = term.partition("^")
        if symbol != "1":
            powers[symbol] = powers.get(symbol, 0.0) + (float(power) if sep else 1.0) * exponent
    return "*".join(symbol if power == 1 else f"{symbol}^{power:g}"
                    for symbol, power in sorted(powers.items()) if power) or "1"


def _primary(domain: DomainSpec) -> tuple:
    """A domain's primary named axes, in order; the one-value placeholder axis is never one."""
    return tuple(axis for axis in domain.axes if axis.coordinate_of is None and axis.role != SCALAR)


def _named_axes(schema: DatasetSchema) -> list:
    """(domain index, axis) of every primary named axis, in domain order."""
    return [(d, axis) for d, domain in enumerate(_domains(schema)) for axis in _primary(domain)]


def axis_groups(schema: DatasetSchema) -> tuple[tuple[str, DomainSpec, tuple], ...]:
    """``(domain word, domain, its primary named axes)`` of every domain that
    has some, in domain order: what a pick and a statistic choose from."""

    return tuple((word, domain, axes) for word, domain in zip(DOMAINS, _domains(schema))
                 if (axes := _primary(domain)))


@functools.lru_cache(maxsize=64)
def _axis_labels(axis) -> tuple[tuple[str, ...], dict[str, int] | None]:
    """An axis' labels and each one's position -- None when two read alike.
    Kept for the axes last asked about: a pick asks every publication, and
    writing out a long sampled axis' labels each time cost tens of ms."""
    labels = axis.labels()
    positions = {label: index for index, label in enumerate(labels)}
    return labels, positions if len(positions) == len(labels) else None


@functools.lru_cache(maxsize=32)
def _domain_rows(domain: DomainSpec, mapped: bool) -> tuple[tuple[str, ...], dict[str, int] | None]:
    """A domain's rows as a pick names them -- each row's primary axes,
    name and label, joined ``detuning 2 · power 1`` -- in row order, and
    each one's position (None when two read alike).  A dense domain's rows
    are every combination of its axes, the last fastest."""
    axes = _primary(domain)
    texts = [tuple(f"{axis.name} {label}" for label in _axis_labels(axis)[0]) for axis in axes]
    if mapped:
        codes = [domain.codes(axis.axis_id) for axis in axes]
        rows = tuple(" · ".join(text[int(code[row])] for text, code in zip(texts, codes))
                     for row in range(domain.size))
    else:
        rows = tuple(" · ".join(parts) for parts in itertools.product(*texts))
    positions = {row: index for index, row in enumerate(rows)}
    return rows, positions if len(positions) == len(rows) else None


def pick_values(schema: DatasetSchema, at: str) -> tuple[str, ...] | None:
    """What a pick ``at`` (an axis name, or a domain word) on a value of
    ``schema`` chooses among, in order -- its labels, or its domain's rows
    -- or None when ``at`` names nothing that can be picked there."""

    try:
        d, axes = _pick_target(schema, at, "the value")
    except ValueError:
        return None
    return _domain_rows(_domains(schema)[d], d < 2)[0] if at in DOMAINS else _axis_labels(axes[0])[0]


def _pick_target(schema, at, what):
    answer = native_ops.pick_target(schema, at, what)
    domain = _domains(schema)[answer["domain"]]
    return answer["domain"], tuple(domain.axis(AxisId(identity)) for identity in answer["axes"])


def axes_text(schema: DatasetSchema) -> str:
    """``repeat 1 × Site 35``: every named axis and its size, in domain order."""

    return " × ".join(f"{axis.name} {axis.size}" for _d, axis in _named_axes(schema)) or "one value"


def describe_schema(schema: DatasetSchema) -> str:
    """What a result is, in one line: axes, dtype and unit."""

    return f"{axes_text(schema)} · {schema.value_schema.dtype} · unit {_unit(schema)}"


def _listed(labels, limit: int = 12) -> str:
    labels = tuple(labels)
    return ", ".join(labels) if len(labels) <= limit else f"{labels[0]}, {labels[1]}, …, {labels[-1]}"


def _axis(schema, name, what, verb):
    answer = native_ops.axis(schema, name, what, verb)
    return answer["domain"], _domains(schema)[answer["domain"]].axis(AxisId(answer["axis"]))


def _isel_plan(schema, d, axis, indexes, scalar):
    return native_ops.pick_plan(schema, d, (axis,), indexes, scalar, False)


def _domain_plan(schema, d, axes, rows, scalar):
    return native_ops.pick_plan(schema, d, axes, rows, scalar, True)


def _alignment(target, other):
    """The native coordinate owner validates and plans schema alignment."""
    return native_ops.alignment(target, other)


def _differ(left, right):
    """Why two operands do not fit, in the words of what to change."""
    others = {axis.axis_id: axis for _d, axis in _named_axes(right)}
    for _d, axis in _named_axes(left):
        other = others.get(axis.axis_id)
        if other is not None and other != axis:
            return (f"{axis.name} differs between the two sides ({_listed(axis.labels())} vs "
                    f"{_listed(other.labels())}); pick the same {axis.name} values on both sides, "
                    "or one value on one side")
    return (f"operand axes differ ({axes_text(left)} vs {axes_text(right)}); "
            "one side's axes must all be on the other")


def _is_boolean(item) -> bool:
    if isinstance(item, DatasetSchema):
        return item.value_schema.dtype.kind == "b"
    return isinstance(item, bool)


_UFUNCS = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.true_divide,
           "==": np.equal, "!=": np.not_equal, ">": np.greater, "<": np.less,
           ">=": np.greater_equal, "<=": np.less_equal, "and": np.bitwise_and, "or": np.bitwise_or}


def _apply(op, left, right):
    if op == "and not":
        return np.bitwise_and(left, np.invert(right))
    return _UFUNCS[op](left, right)


def _schema_of(item):
    return item.schema if isinstance(item, Operand) else item


def _operator(op, left, right, texts):
    """``left op right`` by one plan: Operands while evaluating, schemas
    while typing; either side may be a number or true/false.

    Returns the result, or raises one sentence about this operator.
    """
    kind, _level, label = OPERATORS[op]
    left_text, right_text = texts
    if kind == "where":
        return _masked(left, right, op == "where not", texts)
    schemas = (_schema_of(left), _schema_of(right))
    data = tuple(isinstance(item, DatasetSchema) for item in schemas)
    if not any(data):
        raise ValueError("it joins two constants; write their result as one number")
    booleans = tuple(_is_boolean(item) for item in schemas)
    if kind == "logic":
        for boolean, text in zip(booleans, texts):
            if not boolean:
                raise TypeError(f"{label} needs booleans; {text} is a number")
    elif any(booleans):
        if kind == "arithmetic":
            raise TypeError(f"{label} needs numbers; {texts[booleans.index(True)]} is a boolean; count booleans "
                            "with a statistic, or compare them")
        # Two booleans take every comparison, ordered false < true as NumPy
        # orders them: a > b is a and not b.
        if not all(booleans):
            raise TypeError(_BOOLEANS_MIX)
    if all(data):
        try:
            base, steps = schemas[0], (None, _alignment(schemas[0], schemas[1]))
        except ValueError:
            try:
                base, steps = schemas[1], (_alignment(schemas[1], schemas[0]), None)
            except ValueError:
                raise ValueError(_differ(*schemas)) from None
    else:
        base, steps = schemas[data.index(True)], (None, None)
    units = tuple(_unit(item) if isinstance(item, DatasetSchema) else None for item in schemas)
    conversion = None
    if kind == "logic":
        unit = "1"
    elif op in ("*", "/"):
        # A number is a plain factor.
        left_unit, right_unit = (unit or "1" for unit in units)
        unit = _unit_power(left_unit + "*" + (right_unit if op == "*" else _unit_power(right_unit, -1)), 1)
    else:
        # + − and comparisons: the second Dataset in the first's unit, and a
        # number in the unit of the other side.
        if all(data) and units[0] != units[1]:
            conversion = (units[1], units[0])
        unit = "1" if kind == "comparison" else units[0] or units[1]
    DEFAULT_UNITS.resolve(unit)
    # The result dtype is NumPy's own answer for these operand dtypes.
    probes = [np.ones(1, item.value_schema.dtype) if isinstance(item, DatasetSchema) else item for item in schemas]
    try:
        if conversion is not None:
            probes[1] = DEFAULT_UNITS.convert(probes[1], *conversion)
    except UnitError:
        raise ValueError(f"{left_text} is in {conversion[1]} and {right_text} in {conversion[0]}; "
                         "these units do not convert") from None
    with np.errstate(all="ignore"):
        dtype = np.asarray(_apply(op, *probes)).dtype
    schema = _schema(_domains(base), dtype, unit)
    if not isinstance(left, Operand) and not isinstance(right, Operand):
        return schema
    shape = schema.physical_shape
    arrays, valid = [], np.ones(shape, bool)
    for item, step in zip((left, right), steps):
        if isinstance(item, Operand):
            aligned_values, aligned_valid = native_ops.aligned((item.values, item.valid), step, shape)
            arrays.append(aligned_values)
            valid &= aligned_valid
        else:
            arrays.append(item)
    if conversion is not None:
        arrays[1] = DEFAULT_UNITS.convert(arrays[1], *conversion)
    compute_dtype = np.result_type(*probes) if kind == "comparison" else dtype
    answer = native_ops.binary(op, arrays[0], arrays[1], valid, shape, dtype, compute_dtype)
    return Operand(schema, answer["values"], answer["valid"])


def _masked(value, mask, negate, texts):
    """``value where mask``: the value's samples, valid only where the mask
    is valid and true (false for ``where not``); geometry, values, dtype
    and unit are the value's own.  The mask's axes must all be on the value;
    it repeats along the ones it lacks."""
    left, right = _schema_of(value), _schema_of(mask)
    value_text, mask_text = texts
    if not isinstance(left, DatasetSchema):
        raise ValueError(f"where keeps samples of a value with axes; {value_text} is a number")
    if not isinstance(right, DatasetSchema) or not _is_boolean(right):
        raise TypeError(f"where needs a boolean mask; {mask_text} is a number")
    try:
        steps = _alignment(left, right)
    except ValueError:
        held = {axis.axis_id for domain in _domains(left) for axis in domain.axes}
        extra = next((axis for _d, axis in _named_axes(right) if axis.axis_id not in held), None)
        if extra is None:
            raise ValueError(_differ(left, right)) from None
        raise ValueError(f"the mask has {extra.name}, which {value_text} does not; "
                         f"pick one {extra.name} value of the mask") from None
    schema = _schema(_domains(left), left.value_schema.dtype, _unit(left), left.value_schema.name)
    if not isinstance(value, Operand):
        return schema
    keep, judged = native_ops.aligned((mask.values, mask.valid), steps, value.shape)
    kept = native_ops.call("scientific.where", valid=value.valid, mask=keep, judged=judged, negate=negate)
    return Operand(schema, value.values, kept["valid"], value.sigma, value.dof)


def changing_axes(schema: DatasetSchema, event: DatasetSchema | None) -> dict:
    """The Repeat/Point axes of ``schema`` -- what typing reads -- whose
    labels change from one publication to the next (a finite run's shots
    and scan points, a monitor's shots), each with the labels the latest
    publication ``event`` carries.  Typing refuses a pick on them.
    """

    if event is None:
        return {}
    whole = schema != event
    changing = {}
    for d in (0, 1):
        carried = {axis.axis_id: axis for axis in _domains(event)[d].axes}
        for axis in _primary(_domains(schema)[d]):
            other = carried.get(axis.axis_id)
            if other is None or (not whole and d == 0) or (
                    other != axis and _axis_labels(other)[0] != _axis_labels(axis)[0]):
                changing[axis.axis_id] = () if other is None else _axis_labels(other)[0]
    return changing


def _pick(item, term, evaluating, changing):
    """``item`` at the term's labels of its axis ``at`` -- or rows of its
    domain ``at`` -- one removing the axis (the domain's axes), several
    keeping it restricted to them, in axis (row) order.

    ``first`` and ``last`` are the axis' first and last coordinate (the
    domain's first and last row) as long as it is now, never a label
    fixed when the pick was written; a coordinate two of them name is
    kept once.  How many labels are written, not how many coordinates
    they reach, decides whether the axis stays, so a pick keeps its shape
    when the axis changes length."""
    schema = _schema_of(item)
    what, at, labels = term["of"], term["at"], term["labels"]
    d, axes = _pick_target(schema, at, what)
    moving = next((axis for axis in axes if axis.axis_id in changing), None) if not evaluating else None
    if moving is not None:
        carried = changing[moving.axis_id]
        raise ValueError(f"{moving.name} changes with each publication"
                         + (f" (the latest carries {_listed(carried)})" if carried else "")
                         + f"; reduce over {moving.name} with a statistic, or plot it in a panel")
    domain_pick = at in DOMAINS
    known, positions = _domain_rows(_domains(schema)[d], d < 2) if domain_pick else _axis_labels(axes[0])
    word = "value" if domain_pick else "label"
    if positions is None:
        raise ValueError(f"{at} {word}s repeat; it cannot be picked by {word}")
    twice = next((label for index, label in enumerate(labels) if label in labels[:index]), None)
    if twice is not None:
        raise ValueError(f"{at} {twice!r} is chosen twice")
    own = next((label for label in labels if label in (FIRST, LAST) and label in positions), None)
    if own is not None:
        raise ValueError(f"{at} has a {word} {own!r} of its own; a pick cannot tell it from its {own} {word}")
    ends = {FIRST: 0, LAST: len(known) - 1} if known else {}
    missing = next((label for label in labels if label not in ends and label not in positions), None)
    if missing is not None:
        if evaluating and d < 2:
            # Typing read the whole run's geometry; this one event of it
            # does not carry every scan point or shot.
            raise ValueError(f"this publication carries no {at} {missing!r}; "
                             f"{at} {word}s here: {_listed(known)}")
        raise ValueError(f"{what} has no {at} {missing!r}; {at} {word}s: {_listed(known)}")
    chosen = tuple(sorted({ends[label] if label in ends else positions[label] for label in labels}))
    try:
        if domain_pick:
            picked, take = _domain_plan(schema, d, axes, chosen, len(labels) == 1)
        else:
            picked, take = _isel_plan(schema, d, axes[0], chosen, len(labels) == 1)
    except EmptySelection:
        where = "this publication carries" if evaluating else f"{what} stores"
        raise ValueError(f"{where} no row at {at} {labels[0]!r}") from None
    if not isinstance(item, Operand):
        return picked
    planes = native_ops.selected((item.values, item.valid, item.sigma, item.dof), take)
    return Operand(picked, *planes)


def _reduce_plan(schema, operation, names, by="", every=0):
    return native_ops.reduction_plan(schema, operation, tuple(names), by, int(every))


def _statistic(item, operation, names, by="", every=0, errors=False):
    """Compile the named-axis request, then evaluate its complete native reduction."""
    schema, plans = _reduce_plan(_schema_of(item), operation, names, by, every)
    if not isinstance(item, Operand):
        return schema
    answer = native_ops.statistic(item, plans, operation, errors)
    return Operand(schema, answer["values"], answer["valid"], answer.get("sigma"), answer.get("dof"))


# -- fit: a statistic that eats one axis and yields parameters ---------------

@dataclass(frozen=True, eq=False)
class FitTarget:
    """What a fit over one axis of a value's schema is: that axis' domain and
    x, where each stored row lands (``rows``: output row and x position of a
    Repeat/Point axis; None for a Cell axis, by ``position``) and the schema
    with the axis gone."""

    domain: int
    position: int
    x: np.ndarray
    rows: tuple | None
    schema: DatasetSchema


def fit_target(schema, names) -> FitTarget:
    if len(names) != 1:
        raise ValueError("fit reads exactly one axis as its x")
    if _is_boolean(schema):
        raise TypeError("fit needs numbers; for booleans use sum, mean, count, any or all")
    d, axis = _axis(schema, names[0], "the value", "reduce")
    x = np.asarray(axis)
    if axis.coordinate_labels is not None or x.dtype.kind not in "iuf":
        raise ValueError(f"{axis.name} has no numeric coordinates to fit over")
    base, plans = _reduce_plan(schema, "mean", names)
    if d == 2:
        return FitTarget(d, _domains(schema)[2].axes.index(axis) + 2, x.astype(np.float64), None, base)
    codes = np.asarray(_domains(schema)[d].codes(axis.axis_id))
    inverse = next((plan[2] for plan in plans if plan[0] == "mapped"), np.zeros(codes.size, np.int64))
    if np.unique(inverse * axis.size + codes).size != codes.size:
        raise ValueError(f"{axis.name} repeats a coordinate; reduce the repeats first")
    return FitTarget(d, d, x.astype(np.float64), (inverse, codes, _domains(base)[d].size), base)


def _fit_tensor(plane, target):
    """A plane with x last, one row of it per cell; NaN where nothing stands."""
    moved = np.moveaxis(plane, target.position, -1)
    if target.rows is None:
        return moved.astype(np.float64)
    inverse, codes, size = target.rows
    tensor = np.full((*moved.shape[:-1], size, target.x.size), np.nan)
    tensor[..., inverse, codes] = moved
    return tensor


def fit_cells(item, target, model, fit, warm=None):
    """One independent fit per remaining cell of ``item``, as one batch:
    (values, errors, success), each ``target.schema``'s physical shape, values
    and errors with the parameters last.  ``warm`` is one row per cell."""
    y = _fit_tensor(np.where(item.valid, item.values, np.nan), target)
    sigma = None if item.sigma is None else _fit_tensor(np.where(item.valid, item.sigma, np.nan), target)
    dof = None if item.dof is None else _fit_tensor(np.where(item.valid, item.dof, np.inf), target)
    lead, count = y.shape[:-1], target.x.size
    answer = fit(model, target.x, y.reshape(-1, count), sigma=None if sigma is None else sigma.reshape(-1, count),
                 warm=warm, sigma_dof=None if dof is None else dof.reshape(-1, count))

    def stored(plane, tail=()):
        plane = np.asarray(plane).reshape(*lead, *tail)
        if target.rows is not None:
            plane = np.moveaxis(plane, len(lead) - 1, target.domain)
        return plane.reshape(*target.schema.physical_shape, *tail)

    width = np.asarray(answer.values).shape[-1]
    return (stored(answer.values, (width,)), stored(answer.errors, (width,)),
            stored(np.asarray(answer.success, dtype=bool)))


def fit_schemas(schema, target, row) -> dict[str, DatasetSchema]:
    """Each parameter's output name and schema, by the model's unit relations."""
    axis = _axis(schema, row["over"][0], "the value", "reduce")[1]
    x_unit, y_unit = axis.unit or "1", _unit(schema)
    answer = {}
    for parameter in _model_parameters(row["model"]):
        unit = _parameter_unit(parameter["unit_relation"], x_unit, y_unit)
        answer[f"{row['name']}_{parameter['name']}"] = _schema(_domains(target.schema), np.float64, unit)
    return answer


def fit_operands(schemas, values, errors, success) -> dict[str, "Operand"]:
    """One Operand per parameter; a failed cell is invalid and the others stand,
    and a parameter the data do not determine (no error) is invalid in that cell alone."""
    return {name: Operand(schema, values[..., k], success & np.isfinite(errors[..., k]), errors[..., k])
            for k, (name, schema) in enumerate(schemas.items())}


def _fit(item, row, fit):
    """``item`` fitted, by the row's model, in every remaining cell: by parameter name."""
    target = fit_target(_schema_of(item), row["over"])
    if _model_parameters(row["model"]) is None:
        known = ", ".join(fit_models()) or "none yet"
        raise ValueError(f"no fit model {row['model']!r}; models: {known}")
    schemas = fit_schemas(_schema_of(item), target, row)
    if not isinstance(item, Operand):
        return schemas
    return fit_operands(schemas, *fit_cells(item, target, row["model"], fit))


@dataclass(frozen=True, eq=False)
class Operand:
    """One Dataset value while evaluating: its complete schema, read-only
    values and validity.  A non-finite float is invalid.  ``sigma``, when it
    has one, is the uncertainty of these samples: a pick and a where keep it,
    an operator has no rule for it and drops it.  ``dof`` stands beside a sigma
    that was estimated here: its degrees of freedom, which a fit weighs it by."""

    schema: DatasetSchema
    values: np.ndarray
    valid: np.ndarray | None = None
    sigma: np.ndarray | None = None
    dof: np.ndarray | None = None

    def __post_init__(self):
        values = np.asarray(self.values)
        if values.shape != self.schema.physical_shape or values.dtype.kind not in "biuf":
            raise ValueError("numeric values must match the three-domain physical shape")
        valid = np.broadcast_to(np.asarray(True if self.valid is None else self.valid, dtype=bool), values.shape)
        # Only a float can hold a value that is not a number; an integer
        # frame or a boolean mask is not scanned pixel by pixel to learn so.
        if values.dtype.kind == "f":
            valid = valid & np.isfinite(values)
        values = values.view()
        values.setflags(write=False)
        valid.setflags(write=False)
        if self.sigma is not None:
            sigma = np.asarray(self.sigma, dtype=np.float64).view()
            sigma.setflags(write=False)
            object.__setattr__(self, "sigma", sigma)
        if self.dof is not None:
            dof = np.asarray(self.dof, dtype=np.float64).view()
            dof.setflags(write=False)
            object.__setattr__(self, "dof", dof)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "schema", _schema(
            _domains(self.schema), values.dtype, _unit(self.schema), self.schema.value_schema.name,
        ))

    @property
    def shape(self):
        return self.values.shape


# -- one variable: terms -> precedence tree -> fold -> statistic -------------

def _number_text(number: float) -> str:
    text = repr(float(number))
    return text[:-2] if text.endswith(".0") else text


def term_text(term: Mapping[str, object]) -> str:
    """A term as notes and messages write it: ``counts[frame 1]``,
    ``counts[frame 1, 2]``, ``counts[frame first, last]``, ``Site 1, 2, 3,
    7, … (6)`` past four labels,
    a domain's rows as they read (``counts[detuning 2 · power 1]``),
    ``30`` or ``true``."""

    if term["number"] is not None:
        return _number_text(term["number"])
    if term["boolean"]:
        return str(term["boolean"])
    if not term["at"]:
        return str(term["of"])
    labels = tuple(term["labels"])
    shown = ", ".join(labels) if len(labels) <= 4 else f"{', '.join(labels[:4])}, … ({len(labels)})"
    return f"{term['of']}[{shown}]" if term["at"] in DOMAINS else f"{term['of']}[{term['at']} {shown}]"


def _tree(terms):
    """The terms as a tree by operator precedence, left-associative at every
    level: an int is term k (0-based), ``(k, op, left, right)`` the operator
    term k introduced.  Pure; needs every later term's ``op``."""

    def build(low, high):
        split = None
        for k in range(low + 1, high):
            if split is None or OPERATORS[terms[k]["op"]][1] <= OPERATORS[terms[split]["op"]][1]:
                split = k
        if split is None:
            return low
        return (split, terms[split]["op"], build(low, split), build(split, high))

    return build(0, len(terms))


def _nodes(node):
    if isinstance(node, tuple):
        yield node
        yield from _nodes(node[2])
        yield from _nodes(node[3])


def _text(node, terms) -> str:
    """The formula under ``node`` as read with its grouping: an operator
    child is parenthesized unless it is a left child of the same level."""
    if isinstance(node, int):
        return term_text(terms[node])
    _k, op, left, right = node
    level = OPERATORS[op][1]
    left_text, right_text = _text(left, terms), _text(right, terms)
    if isinstance(left, tuple) and OPERATORS[left[1]][1] != level:
        left_text = f"({left_text})"
    if isinstance(right, tuple):
        right_text = f"({right_text})"
    return f"{left_text} {OPERATORS[op][2]} {right_text}"


def _grouping(terms) -> str | None:
    """``reads …`` when the precedence groups the formula otherwise than
    reading it left to right: exactly when an operator is a right child."""
    tree = _tree(terms)
    if any(isinstance(node[3], tuple) for node in _nodes(tree)):
        return f"reads {_text(tree, terms)}"
    return None


def where_notes(terms) -> tuple[str, ...]:
    """``valid where <mask>`` -- ``… is false`` after ``where not`` -- for
    each where of a formula that reads, in term order, the mask as the
    precedence groups it: ``counts where a and b`` masks with ``a and b``."""

    tree = _tree(terms)
    notes = []
    for _k, op, _left, right in sorted(node for node in _nodes(tree) if OPERATORS[node[1]][0] == "where"):
        mask = _text(right, terms)
        if op == "where not":
            mask = f"({mask}) is false" if isinstance(right, tuple) else f"{mask} is false"
        notes.append(f"valid where {mask}")
    return tuple(notes)


def _formula_problem(row, closing=None) -> DeriveError | None:
    """The first problem of a variable's formula and statistic that needs no
    data, or None.  ``closing`` maps a term index to the cycle it closes;
    None leaves what the variable reads unchecked -- a name it reads may be
    an input's, which only the source can say."""
    name, terms = row["name"], row["terms"]
    if not terms:
        return DeriveError(name, "term 1", "choose what it reads")
    for k, term in enumerate(terms):
        given = [column for column in ("of", "number", "boolean") if term[column] not in ("", None)]
        if not given:
            problem = "choose what it reads"
        elif len(given) > 1:
            problem = "a term reads one thing: an input or a variable, a number, or true or false"
        elif not k and term["op"]:
            problem = "the first term takes no operator"
        elif k and not term["op"]:
            problem = "choose an operator"
        elif term["boolean"] and OPERATORS.get(term["op"], ("",))[0] != "comparison":
            problem = "true and false follow a comparison only"
        elif (term["at"] or term["labels"]) and not term["of"]:
            problem = "a number or true/false has no axis to pick"
        elif term["at"] and not term["labels"]:
            problem = f"choose {term['at']} values"
        elif term["labels"] and not term["at"]:
            problem = "choose the axis of these labels"
        elif closing is not None and term["of"] == name:
            problem = f"{name} reads itself"
        elif closing is not None and k in closing:
            problem = closing[k]
        else:
            continue
        return DeriveError(name, f"term {k + 1}", problem)
    if not any(term["of"] for term in terms):
        return DeriveError(name, "term 1",
                           "a formula needs an input or a variable; numbers and true/false alone have no axes")
    chained = [k for k, op, left, _right in _nodes(_tree(terms)) if OPERATORS[op][0] == "comparison"
               and isinstance(left, tuple) and OPERATORS[left[1]][0] == "comparison"]
    if chained:
        return DeriveError(name, f"term {min(chained) + 1}",
                           "compare once between and, or and where; join two comparisons with and")
    if row["statistic"] and not row["over"]:
        return DeriveError(name, "statistic", "choose the axes it reduces over")
    if row["over"] and not row["statistic"]:
        return DeriveError(name, "statistic", "choose the statistic")
    if row["statistic"] == "fit":
        problem = ("choose a model" if not row["model"]
                   else "a fit takes no groups; group the value it reads" if row["by"] or row["every"] else None)
    elif row["model"]:
        problem = "a model belongs to statistic fit"
    elif row["every"] < 0:
        problem = "a group has at least one coordinate"
    elif row["every"] and not row["by"]:
        problem = "choose the axis it groups"
    elif row["by"] and not row["every"]:
        problem = f"choose how many {row['by']} coordinates make one group"
    elif row["every"] and row["by"] not in row["over"]:
        problem = f"{row['by']} is not among the axes it reduces"
    else:
        problem = None
    return None if problem is None else DeriveError(name, "statistic", problem)


def formula(row, read, evaluating, trace=None, changing=None):
    """One variable's terms folded through the precedence tree: what its
    statistic reduces.  ``read(name)`` gives a member's or variable's value.  A
    failure names the term of the operand or operator it came from; an
    operator's, where precedence groups the formula otherwise
    than reading it left to right, also says how the formula reads.
    ``trace``, while typing, receives what the editor shows: each term's
    operand and left side, the unit a number there takes, and the value the
    statistic reduces.  ``changing`` (typing) holds the axes no pick may
    name (:func:`changing_axes`)."""
    name, terms = row["name"], row["terms"]
    tree = _tree(terms)

    def stage_error(k, error):
        return DeriveError(name, f"term {k + 1}", str(error))

    def fold(node, grouped=False):
        if isinstance(node, int):
            term = terms[node]
            if term["number"] is not None:
                return float(term["number"])
            if term["boolean"]:
                return term["boolean"] == "true"
            try:
                value = read(term["of"])
                if trace is not None:
                    trace["terms"][node]["operand"] = value
                return _pick(value, term, evaluating, changing or {}) if term["at"] else value
            except (ValueError, TypeError) as error:
                raise stage_error(node, error) from None
        k, op, left_node, right_node = node
        left, right = fold(left_node), fold(right_node, True)
        if trace is not None:
            trace["terms"][k]["left"] = left if isinstance(left, DatasetSchema) else None
            kind = OPERATORS[op][0]
            for child, other in ((left_node, right), (right_node, left)):
                if isinstance(child, int) and terms[child]["number"] is not None:
                    trace["terms"][child]["number_unit"] = (
                        _unit(other) if kind in ("arithmetic", "comparison") and op not in ("*", "/")
                        and isinstance(other, DatasetSchema) else None)
        try:
            return _operator(op, left, right, (_text(left_node, terms), _text(right_node, terms)))
        except (ValueError, TypeError) as error:
            # An operator a looser one holds on its right, or one holding a
            # tighter one on its own right, joins other terms than the lines
            # above it suggest: say how the formula reads.
            if grouped or isinstance(right_node, tuple):
                raise stage_error(k, f"{error}; this formula reads {_text(tree, terms)}") from None
            raise stage_error(k, error) from None

    value = fold(tree)
    if trace is not None:
        trace["reducible"] = _schema_of(value)
    return value


def statistic_of(row, value, fit=None, errors=False):
    """``value`` under the row's statistic: a fit answers by parameter name."""
    if not row["statistic"]:
        return value
    try:
        if row["statistic"] == "fit":
            return _fit(value, row, fit)
        return _statistic(value, row["statistic"], row["over"], row["by"], row["every"], errors or row["every"] > 0)
    except (ValueError, TypeError) as error:
        raise DeriveError(row["name"], "statistic", str(error)) from None


# -- variables: names, references, liveness, order ---------------------------

def _name_problems(rows, members=()) -> dict[int, DeriveError]:
    """Every variable's name problem, published or not: a name is an identity."""
    problems, seen = {}, set()
    for index, row in enumerate(rows):
        name = row["name"]
        if not name:
            detail = "choose a name"
        elif not re.fullmatch(_NAME, name):
            detail = "a name uses letters, digits and _, and does not start with a digit"
        elif name in seen:
            detail = f"another variable is already named {name}"
        elif name in members:
            detail = f"{name} is an input's name; choose another"
        else:
            seen.add(name)
            outputs = variable_outputs(row) if row["statistic"] == "fit" else ()
            clash = next((other for other in outputs if other in seen or other in members), None)
            if clash is None:
                seen.update(outputs)
                continue
            detail = f"it would publish {clash}, which is already a name here"
        seen.add(name)
        problems[index] = DeriveError(name or f"variable {index + 1}", "name", detail)
    return problems


def _structure(rows, members=()):
    """What the variables' references make of them, without data.

    ``(edges, order, closing, live)``: for each variable the (term, variable)
    of every variable it reads (a name reads the first variable so named,
    never a member's namesake; an empty name is read by nothing); the
    dependency order, by depth-first search in card order; for each
    variable the cycle each term of it closes; and the LIVE set -- the
    published variables and everything they read.
    """
    index = {}
    for position, row in enumerate(rows):
        if row["name"] and row["name"] not in members:
            index.setdefault(row["name"], position)
            if row["statistic"] == "fit":
                for name in variable_outputs(row):
                    index.setdefault(name, position)
    edges = [tuple((k, index[term["of"]]) for k, term in enumerate(row["terms"])
                   if term["of"] and term["of"] in index)
             for row in rows]
    state, stack, order = [0] * len(rows), [], []
    closing = [{} for _ in rows]

    def visit(position):
        state[position] = 1
        stack.append(position)
        for k, other in edges[position]:
            if other == position:
                continue
            if state[other] == 1:
                path = stack[stack.index(other):]
                closing[position][k] = (f"{rows[position]['name']} reads "
                                        + ", which reads ".join(rows[item]["name"] for item in path))
            elif not state[other]:
                visit(other)
        stack.pop()
        state[position] = 2
        order.append(position)

    for position in range(len(rows)):
        if not state[position]:
            visit(position)
    live, frontier = set(), [position for position, row in enumerate(rows) if row["publish"]]
    while frontier:
        position = frontier.pop()
        if position not in live:
            live.add(position)
            frontier.extend(other for _k, other in edges[position])
    return edges, order, closing, live


def _unknown(name, members, rows) -> str:
    variables = ", ".join(row["name"] for row in rows if row["name"]) or "none"
    return f"no input or variable named {name!r}; inputs: {', '.join(members) or 'none'}; variables: {variables}"


def checked_variables(rows: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    """The complete rows as read without data or the source, or the first
    BLOCKING problem.

    Names are checked on every variable; at least one variable must be
    published; formulas and statistics are checked on the LIVE variables
    only, so a parked helper's problems never block Start.  What a variable
    reads -- an input, another variable, itself, a cycle -- is not checked
    here: which names are inputs is the source's to say, so typing,
    admission and the build check it (:func:`typed_variables`,
    :func:`input_members`), and a variable named like an input is never
    mistaken for one that reads itself.
    """
    rows = tuple(rows)
    for problem in _name_problems(rows).values():
        raise problem
    if not any(row["publish"] for row in rows):
        raise ValueError(NO_PUBLISHED)
    _edges, _order, _closing, live = _structure(rows)
    for position, row in enumerate(rows):
        if position in live:
            problem = _formula_problem(row)
            if problem is not None:
                raise problem
    return rows


def published(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    """The names of the published variables whose names read, in card
    order: a variable being renamed withdraws its own signal, never its
    siblings'."""
    rows = tuple(rows)
    problems = _name_problems(rows)
    return tuple(name for position, row in enumerate(rows) if row["publish"] and position not in problems
                 for name in variable_outputs(row))


def _answering(rows) -> dict[str, int]:
    """The variable each name reads: a variable's own name, and a fit's outputs."""
    first = {}
    for position, row in enumerate(rows):
        if row["name"]:
            first.setdefault(row["name"], position)
            if row["statistic"] == "fit":
                for name in variable_outputs(row):
                    first.setdefault(name, position)
    return first


def group_domain(row, schema) -> int | None:
    """The domain of the axis a row's groups span publications over, or None
    when it has no such groups (a Cell axis' close inside one evaluation)."""
    if not row["every"]:
        return None
    try:
        d = _axis(schema, row["by"], "the value", "reduce")[0]
    except ValueError as error:
        raise DeriveError(row["name"], "statistic", str(error)) from None
    return d if d < 2 else None


def _refuse_mixed(name, reads, events):
    """What reads closed groups evaluates only when they close: never beside
    groups closing at other times, or a publication's value."""
    if len(reads) > 1:
        raise DeriveError(name, "statistic", "it reads groups that close at different times; "
                                             "read groups of one size over one axis")
    if reads and events:
        raise DeriveError(name, "statistic", "it reads closed groups and a publication's value; "
                                             "group the input too, or read only one of them")


def _clock(row, value, reads, moving, changing):
    """When a variable has something to give, by the groups in what it reads:
    ``(domain, axis ids that close, the output's kept Repeat domain of a Point
    group)`` -- two variables with one clock close together -- or None for
    every evaluation.

    A variable that groups across publications starts a clock.  Its readers
    evaluate on what just closed, so they cannot reduce away what closes; a
    fit, which needs its whole axis at once, cannot take an axis a publication
    only carries part of."""
    name = row["name"]
    over = tuple(row["over"]) if row["statistic"] else ()
    named = {axis.name: axis for _d, axis in _named_axes(value)}
    d = group_domain(row, value)
    if moving is not None:
        closing = next((item for item in over if item in named and named[item].axis_id in moving
                        and not (d is not None and item == row["by"])), None)
        if closing is not None:
            raise DeriveError(name, "statistic", f"{closing} closes group by group, so it cannot be reduced whole; "
                                                 f"reduce it by {closing} every N")
    elif row["statistic"] == "fit" and over and over[0] in named and named[over[0]].axis_id in (changing or {}):
        raise DeriveError(name, "statistic", f"{over[0]} changes with each publication and a fit needs all of it; "
                                             "read a variable reduced in groups (by repeat, every N)")
    if d is None:
        return next(iter(reads), None)
    kept = _reduce_plan(value, row["statistic"], row["over"], row["by"], row["every"])[0]
    grouped = next(axis for axis in _domains(kept)[d].axes if axis.name == row["by"])
    family = tuple(sorted(axis.axis_id.value for axis in _domains(kept)[d].coordinate_axes(grouped.axis_id)))
    # A Point group closes per kept Repeat row, so what is kept there is part of when.
    return d, family, None if d == 0 else _domains(kept)[0]


def _moving(clock) -> dict:
    """What a value on this clock cannot be picked or wholly reduced on."""
    d, family, repeats = clock
    return {AxisId(value): () for value in family} | {axis.axis_id: () for axis in (repeats.axes if d == 1 else ())}


def typed_variables(rows: Sequence[Mapping[str, object]],
                    inputs: Mapping[str, DatasetSchema | None],
                    changing: Mapping[object, tuple[str, ...]] | None = None) -> tuple[dict[str, object], ...]:
    """Every variable's answer before Start, by the evaluation's own plans.

    ``inputs`` maps member names to the schemas the evaluation reads
    (``value.canonical_schema or value.snapshot.block.schema``), each None
    before the source publishes: what each variable reads is checked
    against the names alone, and values are typed once every member has a
    schema.  ``changing`` holds the axes no pick may name
    (:func:`changing_axes`).  One dict per variable, in card order:
    ``schema`` and ``error`` (a :class:`DeriveError`), ``live``,
    ``waits_for`` (the variable with a problem it reads), ``reads`` and
    ``read_by`` (variable names), ``grouping`` (``reads …`` when precedence
    groups the formula otherwise than left to right), ``reducible`` (what
    a statistic reduces), ``fit`` (a fit's schema by output name), ``clock``
    (None, or when the groups it reads or makes close: :func:`_clock`) and
    per term ``operand`` (schema before its pick),
    ``left`` (schema left of its operator) and ``number_unit`` (the unit a
    number there takes, None beside × ÷).  Typing stops at a variable's
    first problem; a variable reading one with a problem waits.
    """
    rows = tuple(rows)
    members = tuple(inputs)
    edges, order, closing, live = _structure(rows, members)
    names = _name_problems(rows, members)
    answers = []
    for position, row in enumerate(rows):
        error = names.get(position) or _formula_problem(row, closing[position])
        answers.append({
            "schema": None, "error": error, "live": position in live, "waits_for": None,
            "reads": tuple(dict.fromkeys(rows[other]["name"] for _k, other in edges[position])),
            "read_by": tuple(reader["name"] for reader, reads in zip(rows, edges)
                             if any(other == position for _k, other in reads)),
            "grouping": None if error is not None else _grouping(row["terms"]),
            "reducible": None, "fit": None, "clock": None,
            "terms": tuple({"operand": None, "left": None, "number_unit": None} for _term in row["terms"]),
        })
    if not members or any(schema is None for schema in inputs.values()):
        return tuple(answers)
    schemas = {name: _normal(schema) for name, schema in inputs.items()}
    typed = {}
    first = _answering(rows)

    def read(name):
        if name in schemas:
            return schemas[name]
        if name in first:
            answer = answers[first[name]]
            if answer["fit"] is not None:
                if name not in answer["fit"]:
                    raise ValueError(f"{name} publishes one signal per parameter; read one of "
                                     f"{', '.join(answer['fit'])}")
                return answer["fit"][name]
            return typed[first[name]]
        raise ValueError(_unknown(name, members, rows))

    for position in order:
        answer = answers[position]
        # A statistic still being chosen leaves its formula typed: the
        # statistic's choices are what that formula's value offers, and a
        # problem of the formula comes before the statistic's.
        choosing = answer["error"] is not None and position not in names and answer["error"].stage == "statistic"
        if answer["error"] is not None and not choosing:
            continue
        waits = next((rows[other]["name"] for _k, other in edges[position]
                      if answers[other]["error"] is not None or answers[other]["waits_for"] is not None), None)
        if waits is not None:
            if not choosing:
                answer["waits_for"] = waits
            continue
        try:
            if choosing:
                formula(rows[position], read, False, answer, changing)
            else:
                row = rows[position]
                reads = {answers[other]["clock"] for _k, other in edges[position]} - {None}
                events = (any(term["of"] in schemas for term in row["terms"])
                          or any(answers[other]["clock"] is None for _k, other in edges[position]))
                _refuse_mixed(row["name"], reads, events)
                moving = _moving(next(iter(reads))) if reads else None
                value = formula(row, read, False, answer, changing if moving is None else moving)
                result = statistic_of(row, value)
                answer["clock"] = _clock(row, value, reads, moving, changing)
                if row["statistic"] == "fit":
                    answer["fit"] = result
                    answer["schema"] = typed[position] = fit_target(value, row["over"]).schema
                else:
                    answer["schema"] = typed[position] = result
        except DeriveError as error:
            answer["error"] = error
    return tuple(answers)


def frames_kept(rows: Sequence[Mapping[str, object]], axis: str | None, frames: tuple[str, ...],
                members: Sequence[str] | None) -> tuple[tuple[str, ...] | None, ...]:
    """Every variable's labels of the readout-event axis named ``axis``
    (None: the source has none) when each input member carries it with
    labels ``frames`` (``()``: they carry none) -- what a source's draft
    says before any data exists.  One per variable, in card order: ``()``
    when the value carries no such axis, None when it cannot be told or
    typing would refuse it.  ``members`` are the input members' names, or
    None when only the source's axis is known (a name no variable has is
    then read as a member's).  These are typing's rules on that one axis,
    over the same precedence tree; the full list is ARCHITECTURE_DESIGN.md's
    (``describe_output_frames``).
    """
    rows = tuple(rows)
    known = () if members is None else tuple(members)
    _edges, order, closing, _live = _structure(rows, known)
    names = _name_problems(rows, known)
    first = _answering(rows)
    kept: list[tuple[str, ...] | None] = [None] * len(rows)

    def read(name):
        if name in known or (members is None and name not in first):
            return frames
        if name not in first:
            raise ValueError(f"no input or variable named {name!r}")
        if kept[first[name]] is None:
            raise ValueError(f"{name} cannot be told")
        return kept[first[name]]

    def pick(value, term):
        at, labels = term["at"], tuple(term["labels"])
        if len(set(labels)) != len(labels):
            raise ValueError(f"{at} is chosen twice")
        if at == DOMAINS[1] and value:
            if any(label in (FIRST, LAST) for label in labels):
                # Which frames the Point domain's first and last rows carry
                # is its row order, and whether it has a second axis at all
                # -- a lone frame axis is picked by its own name, never as
                # Point -- only the schemas say.
                raise ValueError(f"the frames of {at}'s ends")
            # A Point row is its axes' "name label" parts joined: two or
            # more of them, the axis named in exactly one.
            rows_picked = [label.split(" · ") for label in labels]
            named = [[part[len(axis) + 1:] for part in parts if part.startswith(f"{axis} ")] for parts in rows_picked]
            if any(len(parts) < 2 or len(own) != 1 for parts, own in zip(rows_picked, named)):
                raise ValueError(f"no such {at} row")
            chosen = [own[0] for own in named]
        elif at == axis:
            if any(label in (FIRST, LAST) and label in value for label in labels):
                raise ValueError(f"{axis} has a label of an end's name")
            ends = {FIRST: value[0], LAST: value[-1]} if value else {}
            chosen = [ends.get(label, label) for label in labels]
        else:
            return value
        if any(label not in value for label in chosen):
            raise ValueError(f"no such {axis}")
        return () if len(labels) == 1 else tuple(label for label in value if label in chosen)

    def fold(node, terms):
        if isinstance(node, int):
            term = terms[node]
            if not term["of"]:
                return None
            value = read(term["of"])
            return pick(value, term) if term["at"] else value
        _k, op, left, right = node
        left, right = fold(left, terms), fold(right, terms)
        if OPERATORS[op][0] in ("where", "logic") and (left is None or right is None):
            raise ValueError(f"{op} beside a number")
        if OPERATORS[op][0] == "where":
            if right and right != left:
                raise ValueError("the mask does not fit")
            return left
        if left is None and right is None:
            raise ValueError("two constants")
        if left and right and left != right:
            raise ValueError(f"{axis} differs")
        return left or right or ()

    for position in order:
        row = rows[position]
        if position in names or _formula_problem(row, closing[position]) is not None:
            continue
        try:
            value = fold(_tree(row["terms"]), row["terms"])
            if row["statistic"]:
                over = tuple(row["over"])
                if len(set(over)) != len(over) or (axis in over and not value):
                    raise ValueError(f"no {axis} to reduce")
                if row["by"] == axis:
                    raise ValueError(f"{axis} in groups has labels of its own")
                value = () if axis in over else value
        except ValueError:
            continue
        kept[position] = value
    return tuple(kept)


def input_members(rows: Sequence[Mapping[str, object]], available: Sequence[str]) -> tuple[str, ...]:
    """The members the LIVE variables read, in bundle order.

    Refuses, by variable and stage, a variable named like a member, and in
    a LIVE variable a name that is neither a member nor a variable, a
    variable that reads itself and a cycle.
    """
    rows, available = tuple(rows), tuple(available)
    for problem in _name_problems(rows, available).values():
        raise problem
    _edges, _order, closing, live = _structure(rows, available)
    variables = set(_answering(rows))
    requested = set()
    for position in sorted(live):
        problem = _formula_problem(rows[position], closing[position])
        if problem is not None:
            raise problem
        for k, term in enumerate(rows[position]["terms"]):
            name = term["of"]
            if not name or name in variables:
                continue
            if name not in available:
                raise DeriveError(rows[position]["name"], f"term {k + 1}", _unknown(name, available, rows))
            requested.add(name)
    return tuple(name for name in available if name in requested)


def reads_for_fit(rows, edges) -> set[int]:
    """The variables whose means a fit weighs: what fits read, through picks
    and masks, up to the statistic that made the value."""
    wanted, frontier = set(), [position for position, row in enumerate(rows) if row["statistic"] == "fit"]
    while frontier:
        for _k, other in edges[frontier.pop()]:
            if other not in wanted:
                wanted.add(other)
                if not rows[other]["statistic"]:
                    frontier.append(other)
    return wanted


def evaluate(rows: Sequence[Mapping[str, object]], members: Mapping[str, Operand], fit=None) -> dict[str, Operand]:
    """Every LIVE variable over one publication's members, in dependency order.

    The rows are a checked draft (:func:`checked_variables`) whose names
    and reads were planned against the bundle (:func:`input_members`).
    ``fit`` answers a fit statistic (``RunContext.fit``).
    """
    rows = tuple(rows)
    edges, order, _closing, live = _structure(rows, tuple(members))
    weighed = reads_for_fit(rows, edges)
    results = {}

    def read(name):
        return members[name] if name in members else results[name]

    for position in order:
        if position in live:
            row = rows[position]
            value = statistic_of(row, formula(row, read, True), fit, position in weighed)
            if row["statistic"] == "fit":
                results.update(value)
            else:
                results[row["name"]] = value
    return results


_ERROR_TEXT = re.compile(rf"(?:^|: )({_NAME}) · (name|term [1-9][0-9]*|statistic): (.*)\Z", re.S)


def error_stage(text: str) -> tuple[str, str, str] | None:
    """(variable, stage, detail) of a DeriveError's text inside a status line."""
    match = _ERROR_TEXT.search(str(text or ""))
    return None if match is None else match.groups()


__all__ = [
    "CATEGORIES", "DOMAINS", "DeriveError", "NO_PUBLISHED", "OPERATORS", "Operand", "STATISTICS",
    "STATISTICS_FOR", "TERM_COLUMNS", "TERM_DEFAULTS", "VARIABLE_COLUMNS", "axes_text", "axis_groups",
    "changing_axes", "checked_variables", "compact_variable", "describe_schema", "error_stage", "evaluate",
    "fit_models", "frames_kept", "input_members", "pick_values", "published", "term_text", "typed_variables",
    "variable_outputs", "where_notes",
]
