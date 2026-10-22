"""What Derive's variables editor shows, as data (the node's ``editor_facts``).

v2 drew the variables as Qt cards inside this package (one line per term:
the name or the operator strip, "=", the operand, its pick, then the
statistic), typing every line against the schemas the source published.  In
v3 the cards are the UI's (apps/desktop/ui/src/views/node_editors/
derive_variables.ts) and hold no rule: what each operand may be, which axes
and labels a pick offers, which statistics fit, what a line's problem is and
what the variable comes out as are answered here by ``typed_variables``,
from the draft, the source bundle's schemas (``bundle``) and, before it
publishes, the frames its draft declares (``source``).  The cards write back
``variables``, the rows they were handed, edited.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence

import numpy as np

from zlc_atom.nodes._framework.frames import FIRST, LAST

from .expression import (
    CATEGORIES, DOMAINS, NO_PUBLISHED, OPERATORS, STATISTICS, STATISTICS_FOR, TERM_DEFAULTS, VARIABLE_COLUMNS,
    axis_groups, changing_axes, describe_schema, frames_kept, pick_values, term_text, typed_variables, where_notes,
)

#: The field the cards write; the Edit page's plain form leaves it out.
DERIVE_EDITOR_FIELDS = ("variables",)
_STRIPS = dict(CATEGORIES)
_FOOTER = "Averages over shots with error bars, per-site probabilities and fits belong in a plot panel."
_EMPTY = ("No variables yet. A variable picks values of an input, combines them line by line, can mask and "
          "reduce them; Publish makes it a signal.")
#: The operand list's constants.  A member or variable name never starts
#: with "#", so no name reads like these keys.
_NUMBER, _TRUE, _FALSE = "#number", "#true", "#false"
_CONSTANTS = {_NUMBER: "Number", _TRUE: "true", _FALSE: "false"}
#: "No pick": the axis list's first row, and a pick's first value.
_ALL = ""
#: What a pick's values start with: All, then the axis' ends, written in the
#: draft as Occupancy's frame claims write a cycle's.
_ENDS = (("All", _ALL), (FIRST, FIRST), (LAST, LAST))
#: The most labels one labels list offers; a longer axis says how many it has.
_LABEL_LIMIT = 2048
_ADD = (("arithmetic", "+ Arithmetic"), ("comparison", "+ Comparison"), ("logic", "+ Logic"),
        ("where", "+ Where"), ("statistic", "+ Statistic"))
_VARIABLE_DEFAULTS = {column.name: [] if column.default is None else column.default for column in VARIABLE_COLUMNS}


def _sentence(text: str) -> str:
    """A problem under its line reads as one sentence (v2 show_problem)."""
    text = str(text).rstrip()
    return text if not text or text.endswith((".", "?", "!", "…")) else f"{text}."


def _boolean(schema) -> bool:
    return schema.value_schema.dtype.kind == "b"


def _flavour(schema) -> str | None:
    """"b" for booleans, "n" for numbers, None while not known."""
    return None if schema is None else "b" if _boolean(schema) else "n"


def _operand_key(term: Mapping[str, object]) -> str | None:
    if term["number"] is not None:
        return _NUMBER
    if term["boolean"]:
        return f"#{term['boolean']}"
    return term["of"] or None


def _alternates(schema, at: str) -> tuple:
    """``(name, unit, labels)`` of each alternate coordinate of the axis ``at``."""
    for _word, domain, axes in axis_groups(schema):
        for axis in axes:
            if axis.name == at:
                return tuple((other.name, other.unit or "", other.labels()) for other in domain.axes
                             if other.coordinate_of == axis.axis_id)
    return ()


@functools.lru_cache(maxsize=64)
def _progression(axis) -> tuple[int, int] | None:
    """``(start, step)`` when the axis' labels are integers counting from
    start by one step -- an index axis, its coordinates implicit or written
    out -- else None (named, real or uneven coordinates spell themselves)."""

    if axis.coordinate_labels is not None or axis.coordinate_origins is not None:
        return None
    if axis.coordinates is None:
        return int(axis.index_origin), 1
    values = np.asarray(axis.coordinates)
    if values.dtype.kind not in "iu" or values.ndim != 1 or not values.size:
        return None
    step = int(values[1] - values[0]) if values.size > 1 else 1
    if step == 0 or (values.size > 1 and not np.all(np.diff(values) == step)):
        return None
    return int(values[0]), step


def _counted_axis(axis, name: str | None) -> dict | None:
    """An index axis as the window spells its labels out itself (``name``
    before each, in a domain's rows): where it starts, its step and how many
    there are."""

    counted = _progression(axis)
    return None if counted is None else {"name": name, "start": counted[0], "step": counted[1], "count": int(axis.size)}


def _label_items(sets: dict, keys: Sequence[str] | None, alternates: tuple, counted: Sequence[dict] | None = None) -> str:
    """A pick's values: All and the ends, then the axis' (domain's) own, each
    with its alternate coordinates (``1 · time 0.1 s``), at most
    ``_LABEL_LIMIT`` of them, and how many there are when that is fewer --
    kept once in ``sets`` and named by its id: every line picking one axis
    (a long one, a camera's pixels) shares the one list.  Index axes
    (``counted``: a camera's 1200 x 1920 pixels) go as where they start and
    how many, rows of them last fastest, and the window spells out the
    labels it shows (``spatial-y 0 · spatial-x 1``); ``keys`` is then unused."""

    if counted is not None and not alternates:
        total = 1
        for axis in counted:
            total *= axis["count"]
        cached = ("counted", tuple(tuple(axis.items()) for axis in counted))
        entry = {"rows": {"axes": list(counted), "shown": min(total, _LABEL_LIMIT)}}
    else:
        keys = tuple(keys or ())
        total = len(keys)
        cached = (keys, alternates)
        entry = {"values": [[key, key + "".join(f" · {name} {labels[index]}" + (f" {unit}" if unit else "")
                                                for name, unit, labels in alternates)]
                            for index, key in enumerate(keys[:_LABEL_LIMIT])]}
    found = sets.get(cached)
    if found is not None:
        return found[0]
    name = f"labels:{len(sets)}"
    sets[cached] = (name, {"ends": [[key, text] for text, key in _ENDS], **entry,
                           "more": f"{_LABEL_LIMIT} of {total} shown" if total > _LABEL_LIMIT else ""})
    return name


def _domain_counted(word: str, axes: Sequence) -> list[dict] | None:
    """A dense domain's rows (every combination of its axes, as
    ``pick_values`` names them) when each of its axes is an index axis."""

    if word in DOMAINS[:2]:
        return None  # a Repeat or Point domain's rows are its recorded points, not combinations
    counted = [_counted_axis(axis, axis.name) for axis in axes]
    return None if any(axis is None for axis in counted) else counted


def _inputs(bundle: Sequence[Mapping], inputs: Mapping, changing: Mapping) -> list:
    """The Inputs row: each member, what it is, and its axes by domain."""

    rows = []
    for member in bundle:
        name = str(member["name"])
        schema = inputs[name]
        if schema is None:
            rows.append({"name": name, "kind": "waiting for its first publication", "axes": [], "tip": ""})
            continue
        value = schema.value_schema
        axes, lines = [], []
        for word, _domain, domain_axes in axis_groups(schema):
            for axis in domain_axes:
                axes.append({"text": f"{axis.name} {axis.size}", "domain": word})
                labels = axis.labels()
                shown = ", ".join(labels) if len(labels) <= 8 else f"{labels[0]} … {labels[-1]}"
                moving = " · changes with each publication" if axis.axis_id in changing else ""
                lines.append(f"{axis.name} — {axis.axis_id.value} · {word} · {axis.role.value} · {shown}{moving}")
        rows.append({"name": name, "kind": f"{value.dtype} · unit {value.value_unit or '1'}", "axes": axes,
                     "tip": "\n".join(lines)})
    return rows


def _operands(kind, term, info, context) -> list:
    """What a line's operand may be: the inputs, the variables this one may
    read (never itself or a reader of it, so no cycle can be made), and the
    constants its operator takes -- once typed, only those of the type the
    operator wants.  Each operand says whether the line's pick stays when it
    is chosen (it has that axis, or nothing says which it has)."""

    schema_of = context["schema_of"]
    if kind is None:
        want, constants = None, (_NUMBER,)
    elif kind == "arithmetic":
        want, constants = "n", (_NUMBER,)
    elif kind == "comparison":
        left = info["left"]
        want, constants = ((None, (_NUMBER, _TRUE, _FALSE)) if left is None
                           else ("b", (_TRUE, _FALSE)) if _boolean(left) else ("n", (_NUMBER,)))
    else:
        want, constants = "b", ()
    if not context["typed"]:
        want = None

    def offered(name):
        return want is None or _flavour(schema_of(name)) in (want, None)

    inputs = [name for name in context["members"] if offered(name)]
    variables = [name for name in context["variables"] if offered(name)]
    current = _operand_key(term)
    if current is not None and current.startswith("#"):
        constants = (*constants, *((current,) if current not in constants else ()))
    elif current is not None and current not in inputs and current not in variables:
        (inputs if current in context["members"] else variables).append(current)

    def keeps(name):
        schema = schema_of(name)
        return schema is None or bool(term["at"]) and pick_values(schema, term["at"]) is not None

    items = []
    for group, keys in (("Inputs", inputs), ("Variables", variables), ("Constants", constants)):
        items.extend({"value": key, "label": _CONSTANTS.get(key, key), "group": group,
                      "keeps_pick": not key.startswith("#") and keeps(key)} for key in keys)
    return items


def _line(kind, term, info, context) -> dict:
    """One line of a card: the head (``kind`` None) or a later term."""

    schema_of = context["schema_of"]
    number = term["number"] is not None
    at = term["at"]
    schema = schema_of(term["of"]) if term["of"] else None
    groups = axis_groups(schema) if schema is not None else ()
    # Before the source publishes, the readout-event axis its draft declares
    # -- by the name it gives it -- is the one pick offered, with the labels
    # this operand keeps of it.
    frame_axis = context["frame_axis"]
    frames = context["frames_of"](term["of"]) if term["of"] and schema is None else None
    picks = bool(term["of"]) and (bool(groups) or bool(frames) or bool(at))
    axes = []
    if picks:
        changing = context["changing"]
        axes.append({"value": _ALL, "label": "All", "group": None, "labels": None})
        for word, _domain, domain_axes in groups:
            # An axis each publication carries its own part of is never a
            # pick; a domain is one when every axis of it may be.
            free = [axis for axis in domain_axes if axis.axis_id not in changing]
            if not free:
                continue
            if len(domain_axes) > 1 and len(free) == len(domain_axes):
                counted = _domain_counted(word, domain_axes)
                axes.append({"value": word, "label": word, "group": word, "labels": _label_items(
                    context["label_sets"], None if counted else pick_values(schema, word), (), counted)})
            for axis in free:
                alternates = _alternates(schema, axis.name)
                counted = _counted_axis(axis, None)
                keys = None if counted and not alternates else pick_values(schema, axis.name)
                axes.append({"value": axis.name, "label": axis.name, "group": word,
                             "labels": _label_items(context["label_sets"], keys, alternates, None if counted is None else [counted])})
        if frames:
            axes.append({"value": frame_axis, "label": frame_axis, "group": DOMAINS[1], "labels": _label_items(context["label_sets"], frames, ())})
        if at and not any(item["value"] == at for item in axes):
            axes.append({"value": at, "label": at, "group": None, "labels": _label_items(context["label_sets"], (), ())})
    return {
        "kind": kind,
        "ops": [[op, OPERATORS[op][2]] for op in _STRIPS[kind]] if kind is not None else [],
        "op": term["op"] or None,
        "operand": _operand_key(term),
        "operands": _operands(kind, term, info, context),
        "number": number,
        "number_text": term_text(term) if number else "",
        "number_unit": (info["number_unit"] or "") if number else "",
        "picks": picks,
        "at": at,
        "labels": list(term["labels"]),
        "axes": axes,
        "error": "", "muted": False,
    }


def _preview(row, answer, typed: bool) -> str:
    """What the variable comes out as, in one line under its card."""

    if answer["waits_for"]:
        return f"→ waits for {answer['waits_for']}"
    schema = answer["schema"]
    if schema is None or not typed:
        return "→ checked when the source publishes"
    notes = [describe_schema(schema)]
    if answer["grouping"]:
        notes.append(answer["grouping"])
    reducible = answer["reducible"]
    if row["statistic"] and reducible is not None and _boolean(reducible):
        notes.extend(note for note in ({"sum": "number true", "mean": "fraction true",
                                        "count": "valid samples"}.get(row["statistic"]),) if note)
    elif row["statistic"] == "count":
        notes.append("valid samples")
    notes.extend(where_notes(row["terms"]))
    if not row["publish"]:
        notes.append(f"internal · read by {', '.join(answer['read_by'])}" if answer["live"]
                     else "internal · read by nothing, so not evaluated")
    return "→ " + " · ".join(notes)


def _added(row, answer, context) -> dict:
    """The line each "+ kind" command adds after the last, written the way
    most formulas want it (v2 _VariableCard._add): beside numbers a number
    or the head's input, beside verdicts a boolean; a pick it should make
    asks for its labels at once (``pending_at``)."""

    schema_of = context["schema_of"]
    head = (row["terms"] or [dict(TERM_DEFAULTS)])[0]
    last = row["terms"][-1] if row["terms"] else head
    infos = answer.get("terms") or ()
    beside = ("n" if last["number"] is not None else "b" if last["boolean"]
              else _flavour(infos[-1]["operand"]) if row["terms"] and len(infos) == len(row["terms"]) else None)
    names = (*context["members"], *context["variables"])
    added = {}
    for kind, operators in CATEGORIES:
        term = dict(TERM_DEFAULTS, op=operators[0])
        if kind == "arithmetic" or (kind == "comparison" and beside != "b"):
            pool = [name for name in names if _flavour(schema_of(name)) != "b"]
            of = head["of"] if head["of"] in pool else pool[0] if pool else ""
            if of:
                term["of"] = of
            else:
                term["number"] = 0.0
        else:
            # A verdict: a boolean variable this one may read, else a member.
            pool = [name for name in (*context["variables"], *context["members"]) if _flavour(schema_of(name)) == "b"]
            term["of"] = head["of"] if kind == "comparison" and head["of"] in pool else pool[0] if pool else ""
        schema = schema_of(term["of"]) if term["of"] else None
        pending = None
        if schema is not None and head["at"] and pick_values(schema, head["at"]) is not None:
            if kind != "where":
                pending = head["at"]
            elif all(label in (FIRST, LAST) or label in pick_values(schema, head["at"]) for label in head["labels"]):
                # The mask at the value's own pick.
                term.update(at=head["at"], labels=tuple(head["labels"]))
        added[kind] = {"term": {name: list(value) if isinstance(value, tuple) else value for name, value in term.items()},
                       "pending_at": pending}
    return added


def _card(row, answer, context) -> dict:
    """One variable's card: its lines, its statistic, its result or problem."""

    terms = row["terms"] or [dict(TERM_DEFAULTS)]
    infos = answer["terms"] or ({"operand": None, "left": None, "number_unit": None},)
    lines = [_line(None if k == 0 else OPERATORS.get(term["op"], ("arithmetic",))[0], term, info, context)
             for k, (term, info) in enumerate(zip(terms, infos))]
    reducible = answer.get("reducible")
    over = []
    for word, _domain, axes in (() if reducible is None else axis_groups(reducible)):
        over.extend({"value": axis.name, "label": axis.name, "group": word} for axis in axes)
    known = {item["value"] for item in over}
    over.extend({"value": name, "label": name, "group": None} for name in row["over"] if name not in known)
    # With nothing to reduce over yet, a statistic picked reduces every Cell
    # axis of the value (v2 _statistic_chosen).
    cells = [] if reducible is None else [axis.name for word, _domain, axes in axis_groups(reducible)
                                         if word == "Cell" for axis in axes]
    statistic = {
        "current": row["statistic"] or None,
        "choices": [[name, reducible is None or name in STATISTICS_FOR[_boolean(reducible)]
                     or name == row["statistic"]] for name in STATISTICS],
        "over": list(row["over"]), "over_choices": over, "default_over": cells,
        "error": "", "muted": False,
    }
    error = answer["error"]
    blocks = ""
    preview = _preview(row, answer, context["typed"])
    if error is not None:
        stage, detail = error.stage, error.detail
        muted = not answer["live"] and stage != "name"
        # The line a problem at ``stage`` stands under.
        if stage == "statistic" and (row["statistic"] or row["over"]):
            said = statistic
        elif stage.startswith("term ") and int(stage.split()[1]) - 1 < len(lines):
            said = lines[int(stage.split()[1]) - 1]
        else:
            said = lines[0]
        said["error"], said["muted"] = _sentence(detail), muted
        blocks = "" if muted else str(error)
        # A parked helper's problem blocks nothing; its result says why.
        preview = "→ internal · read by nothing, so not evaluated" if muted else ""
    return {"row": dict(row), "lines": lines, "statistic": statistic, "preview": preview, "problem": blocks,
            "add_terms": _added(row, answer, context)}


def derive_editor_facts(rows: Sequence[Mapping[str, object]], problem: str, bundle: Sequence[Mapping],
                        source: Sequence | None) -> dict:
    """The cards for one draft (v2 DeriveEditor._show).

    ``rows`` are the draft's variables as projected (none when they do not
    project, ``problem`` then says why); ``bundle`` the source producer's
    members in order, each ``{name, schema, event_schema}`` (``zlc_data.codec``
    trees, None before it publishes); ``source`` the frames the source's draft
    declares, ``[axis, labels]`` or None.
    """

    from zlc_data.codec import dataset_schema_from_tree

    members = tuple(str(member["name"]) for member in bundle)
    inputs, changing = {}, {}
    for member in bundle:
        tree, event = member.get("schema"), member.get("event_schema")
        schema = None if tree is None else dataset_schema_from_tree(tree)
        inputs[str(member["name"])] = schema
        if schema is not None and event is not None:
            changing.update(changing_axes(schema, dataset_schema_from_tree(event)))
    frames = None if source is None or len(source[1]) < 2 else tuple(str(label) for label in source[1])
    frame_axis = None if frames is None else source[0]
    if not problem and rows and not any(row["publish"] for row in rows):
        problem = NO_PUBLISHED
    typed = bool(inputs) and all(schema is not None for schema in inputs.values())
    answers = typed_variables(rows, inputs, changing)
    results: dict[str, object] = {}
    reads: dict[str, set[str]] = {}
    for row, answer in zip(rows, answers):
        results.setdefault(row["name"], answer["schema"])
        reads.setdefault(row["name"], set()).update(term["of"] for term in row["terms"] if term["of"])

    def schema_of(name):
        return inputs.get(name) if name in members else results.get(name)

    # Untyped, each name's frames as the source's draft says them.
    drafted = {}
    if not typed and frames is not None:
        for row, kept in zip(rows, frames_kept(rows, frame_axis, frames, members)):
            drafted.setdefault(row["name"], kept)

    def frames_of(name):
        return frames if name in members else drafted.get(name)

    def readers(name):
        found, frontier = set(), [name]
        while frontier:
            current = frontier.pop()
            for other, read in reads.items():
                if current in read and other not in found:
                    found.add(other)
                    frontier.append(other)
        return found

    cards, label_sets = [], {}
    for row, answer in zip(rows, answers):
        closed = readers(row["name"]) | {row["name"]}
        variables = tuple(dict.fromkeys(
            other["name"] for other in rows
            if other["name"] and other["name"] not in closed and other["name"] not in members))
        cards.append(_card(row, answer, {"members": members, "variables": variables, "schema_of": schema_of,
                                         "frames_of": frames_of, "frame_axis": frame_axis, "typed": typed,
                                         "changing": changing, "label_sets": label_sets}))
    taken = {row["name"] for row in rows}
    ordinal = 1
    while f"variable_{ordinal}" in taken:
        ordinal += 1
    return {
        "editor": "derive_variables",
        "manages": list(DERIVE_EDITOR_FIELDS),
        "inputs": _inputs(bundle, inputs, changing),
        "members": list(members),
        "problem": problem,
        "empty": "" if rows else _EMPTY,
        "footer": _FOOTER,
        "cards": cards,
        # Each pick's values, once: a line's axis names its list by id.
        "label_sets": {name: items for name, items in label_sets.values()},
        "add": [[kind, text] for kind, text in _ADD],
        # Every kind's operator strip: a card line's lead is as wide as the widest (v2 DeriveEditor).
        "strips": [[[op, OPERATORS[op][2]] for op in operators] for _kind, operators in CATEGORIES],
        # What the cards' gestures write: a new variable (published, reading
        # the first input), and each column's default (a row says only what
        # differs from it, as v2 compact_variable wrote it).
        "new_variable": {"name": f"variable_{ordinal}", "terms": [{"of": members[0]} if members else {}]},
        "defaults": {"variable": {name: value for name, value in _VARIABLE_DEFAULTS.items() if name != "terms"},
                     "term": {name: list(value) if isinstance(value, tuple) else value
                              for name, value in TERM_DEFAULTS.items()}},
    }


__all__ = ["DERIVE_EDITOR_FIELDS", "derive_editor_facts"]
