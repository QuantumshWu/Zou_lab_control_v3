"""Discoverable derive processor descriptor."""

from __future__ import annotations

from collections.abc import Mapping

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_plugin import DatasetInput, DatasetOutputDeclaration, NodeKind, NodeType

from .editor import derive_editor_facts
from .expression import (
    VARIABLE_COLUMNS, changing_axes, checked_variables, frames_kept, published, typed_variables,
)
from .processor import DeriveProcessor, declared_outputs


def _readable(values: Mapping[str, object]) -> None:
    """Refuse, while the draft is edited rather than at Start, variables
    that do not read without data or the source (:func:`checked_variables`)."""

    checked_variables(values["variables"])


DERIVE_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "variables",
            "rows",
            "Variables",
            (),
            required=True,
            columns=VARIABLE_COLUMNS,
            description="one row per variable; only variables with publish on become signals",
        ),
    ),
    validator=_readable,
)
#: The same rows without the validator: a raw draft's rows projected only
#: for what they declare.
_ROWS = AuthoringSchema(DERIVE_SCHEMA.fields)


def _draft_outputs(
    values: Mapping[str, object],
    devices: Mapping[str, object] | None = None,
) -> tuple[DatasetOutputDeclaration, ...]:
    """What one draft publishes: its published variables whose names read,
    in card order (:func:`published`) -- a formula being edited does not
    withdraw the signal a panel shows; nothing while the rows do not
    project."""

    del devices
    try:
        rows = _ROWS.draft_values({"variables": values.get("variables", ())})["variables"]
        return declared_outputs(rows)
    except (TypeError, ValueError):
        return ()


def _describe_output_frames(values: Mapping[str, object], output_name: str,
                            source: tuple[str | None, tuple[str, ...]] | None,
                            ) -> tuple[str | None, tuple[str, ...]] | None:
    """A published variable's readout-event axis before any data, by
    :func:`frames_kept` on the axis the source names: ``(axis, labels)``,
    ``(None, ("0",))`` (one frame) when it keeps none, None when the source
    or the draft cannot say."""

    if source is None:
        return None
    try:
        rows = _ROWS.draft_values({"variables": values.get("variables", ())})["variables"]
    except (TypeError, ValueError):
        return None
    if output_name not in published(rows):
        return None
    name, labels = source
    carried = tuple(labels) if name is not None else ()
    kept = frames_kept(rows, name, carried, None)
    frames = next(kept[position] for position, row in enumerate(rows) if row["name"] == output_name)
    return None if frames is None else (name, frames) if frames else (None, ("0",))


def _admit_source(values: Mapping[str, object], members) -> tuple[str, ...]:
    """What the variables cannot compute from what the producer last published.

    ``members`` is the selected producer's bundle, each output ``{name,
    schema, event_schema}``: the run's dataset schema and the latest event's
    (None for a whole dataset), as ``zlc_data.codec`` trees; the core asks
    once every member has published.
    Each LIVE variable's problem and any variable's name problem from
    :func:`typed_variables`, in its ``<variable> · <stage>: …`` words: the
    Edit page says them as draft issues and Start waits; a parked helper's
    problems never hold Start.
    """

    from zlc_data.codec import dataset_schema_from_tree

    inputs, changing = {}, {}
    for member in members:
        name = str(member["name"])
        schema = dataset_schema_from_tree(member["schema"])
        event_tree = member.get("event_schema")
        event = schema if event_tree is None else dataset_schema_from_tree(event_tree)
        inputs[name] = schema
        changing.update(changing_axes(schema, event))
    return tuple(
        str(answer["error"]) for answer in typed_variables(_ROWS.draft_values({"variables": values["variables"]})["variables"],
                                                           inputs, changing)
        if answer["error"] is not None and (answer["live"] or answer["error"].stage == "name")
    )


def _editor_facts(values, resources, devices, source, bundle, artifacts):
    """The variables' cards (derive.editor), typed against the source bundle."""

    del resources, devices, artifacts
    try:
        rows, problem = _ROWS.draft_values({"variables": values.get("variables", ())})["variables"], ""
    except (TypeError, ValueError) as error:
        rows, problem = (), str(error)
    return derive_editor_facts(rows, problem, tuple(bundle or ()), source)


def _build(*, source_signal: str, **values: object) -> DeriveProcessor:
    """The processor for the variables over the bound producer's bundle.

    ``source_signal`` names the selected member; every delivery carries the
    whole bundle, and the evaluation reads the members its live variables
    name (refusing, by variable and stage, a name the bundle does not have).
    """

    rows = DERIVE_SCHEMA.project_values(values)["variables"]
    selected = str(source_signal).strip()
    # An output name never contains a slash; the owner may.
    primary = selected.rpartition("/")[2]
    if not primary:
        raise ValueError("select an available input bundle")
    return DeriveProcessor(variables=rows, primary_output=primary)


LOGIC_NODE = NodeType(
    "derive",
    NodeKind.PROCESSOR,
    DERIVE_SCHEMA,
    description=(
        "Within-evaluation algebra on one producer's synchronized bundle: live, each publication; at "
        "the end of a finite run, the run's Dataset. values.variables has one row per variable: name; "
        "publish (default true; false keeps a helper internal: other variables read it, nothing "
        "publishes it); terms, the formula, one row per line; optional statistic + over. A term names "
        "of (an input member or any other variable, no cycles), or gives number, or boolean "
        "\"true\"/\"false\" (only after a comparison). at with labels (a list) picks coordinates of of: "
        "at is an axis name (any domain) and labels are that axis' labels from the catalog, or at is a domain "
        "with two or more axes (Repeat, Point, Cell) and labels are its rows written \"detuning 2 · power "
        "1\"; \"first\" and \"last\" are its first and last coordinate (row) whatever its length, resolved "
        "each evaluation, never a fixed label. One label drops the axis (the domain's axes), several keep "
        "those coordinates (one two of them name, once). A pick never "
        "names an axis each publication carries its own part of (a finite run's shots and scan points, a "
        "monitor's shots): reduce those, or plot them. Every term after the first has op: arithmetic + - "
        "* /, comparison == != > < >= <=, logic and, or, \"and not\", mask where, \"where not\". "
        "Precedence: * / then + - then one comparison then and/\"and not\" then or then where; left to "
        "right within a level; no parentheses (put a group in an internal variable). statistic sum mean "
        "count min max std any all reduces the over axis names (a list, any domains). Booleans: sum is "
        "the number true, mean the fraction true, count the valid samples; booleans never mix with "
        "numbers, and two booleans take every comparison, false < true (occupied at frame 0 > occupied "
        "at frame 2: loaded at 0 and empty at 2). An operand whose axes are a subset of the other's "
        "repeats along the rest. Only published variables become signals; an unpublished variable no "
        "published one reads is not evaluated and never blocks Start. Example: "
        "[{\"name\":\"same\",\"publish\":false,\"terms\":[{\"of\":\"occupied\",\"at\":\"frame\","
        "\"labels\":[\"0\"]},{\"op\":\"==\",\"of\":\"occupied\",\"at\":\"frame\",\"labels\":[\"2\"]}]},"
        "{\"name\":\"kept\",\"terms\":[{\"of\":\"counts\"},{\"op\":\"where\",\"of\":\"same\"}]}]. "
        "Shot averages with error bars, per-site probabilities and fits belong in a plot panel."
    ),
    input_specs=(DatasetInput("bundle", None, "exact", select_bundle=True),),
    declare_outputs=_draft_outputs,
    describe_output_frames=_describe_output_frames,
    admit_source=_admit_source,
    editor_facts=_editor_facts,
    build=_build,
)


__all__ = ["DERIVE_SCHEMA", "LOGIC_NODE"]
