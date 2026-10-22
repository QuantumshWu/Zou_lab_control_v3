"""What Frame Survival's pair matrix shows, as data (the node's ``editor_facts``).

v2 drew the pairs as a Qt start x end grid in this package; v3's matrix is the
UI's (apps/desktop/ui/src/views/node_editors/pair_matrix.ts) and holds no pair
rule: which pairs a rule selects, which a list names, how a frame is written
and what is kept but not published are answered here, from the draft and the
frames its source carries, by the functions the processor runs.  The matrix
writes back ``selection``, ``pairs`` (a rule's exceptions) and
``custom_pairs`` (Selected's own list), the rows it was handed, edited.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from zlc_atom.authoring import AuthoringSchema
from zlc_atom.nodes._framework.frames import frame_identity, resolve_frame

from .processor import PAIR_RULES, _checked_pairs, _rule_pairs, _selected_pairs, pair_label

#: The fields the matrix writes; the Edit page's plain form leaves them out.
PAIR_MATRIX_FIELDS = ("selection", "pairs", "custom_pairs")


def _sentence(text: str) -> str:
    text = str(text).rstrip()
    return text if not text or text.endswith((".", "?", "!", "…")) else f"{text}."


def pair_matrix_facts(values: Mapping[str, object], source: Sequence | None, *, schema: AuthoringSchema,
                      overrides_field: str, list_field: str, read_overrides, read_list) -> dict:
    """The matrix for one draft (v2 SurvivalForm.update_projection).

    ``read_overrides`` / ``read_list`` check a projected list as the build
    does; each list is projected on its own, as the draft projects it under
    its own choice (a list holds anything while its choice is not picked).
    """

    fields = {field.name: field for field in schema.fields}
    labels = {choice.value: choice.label for choice in fields["selection"].choices}

    def read(name, reader):
        field = fields[name]
        try:
            rows = AuthoringSchema((replace(field, enabled_when=None),)).project_values({name: values.get(name, field.default)})[name]
            return tuple(dict(row) for row in rows), reader(rows), ""
        except (TypeError, ValueError) as error:
            return None, None, _sentence(f"{field.label} cannot be marked on the grid: {error}")

    selection = str(values.get("selection", fields["selection"].default))
    rule = selection in PAIR_RULES
    override_rows, overrides, overrides_problem = read(overrides_field, read_overrides)
    list_rows, published, list_problem = read(list_field, read_list)
    problem = overrides_problem if rule else list_problem
    carried = None if source is None else tuple(str(label) for label in source[1])
    frames = None if carried is None or len(carried) < 2 else carried
    facts = {
        "editor": "pair_matrix",
        "manages": list(PAIR_MATRIX_FIELDS),
        "title": "Pairs",
        "selection": selection,
        # Under a rule the gestures write its exceptions; otherwise Selected's list.
        "rule": rule,
        # A list the grid cannot read is still a list to come back to.
        "keeps_list": list_rows is None or bool(list_rows),
        "error": problem,
        "grid": None, "selected": [], "baseline": [], "states": {}, "tips": {}, "enabled": False,
        "identity": [], "override_rows": list(override_rows or ()), "override_cells": [],
        "list_rows": list(list_rows or ()), "list_cells": [], "note": "", "parked": "",
    }
    if frames is not None:
        count = len(frames)
        facts["grid"] = {
            "rows": [f"{frames[start]} →" for start in range(count - 1)],
            "columns": [f"→ {frames[end]}" for end in range(1, count)],
            "cells": [[start, end - 1, pair_label(frames[start], frames[end])]
                      for start in range(count - 1) for end in range(start + 1, count)],
        }
        # How the matrix writes a frame: an end by identity, a middle frame by index.
        facts["identity"] = [str(frame_identity(index, count)) for index in range(count)]
    if frames is None or problem:
        # A source of one frame per cycle is known, not awaited.
        facts["summary"] = ("Waiting for the source frames" if carried is None
                            else "One frame per cycle: no pairs" if frames is None else labels.get(selection, selection))
        return facts
    count = len(frames)

    def cell(start, end):
        first, last = resolve_frame(start, count), resolve_frame(end, count)
        return None if first is None or last is None else [first, last - 1]

    rows = list_rows or ()
    selected = iter(published or ())

    def named(row):
        try:
            return _checked_pairs(((row["start"], row["end"]),))[0]
        except ValueError:
            return None

    row_pairs = tuple(next(selected) if row["selected"] else named(row) for row in rows)
    listed, listed_absent = _selected_pairs(tuple(pair for row, pair in zip(rows, row_pairs) if row["selected"]), (), count)
    wanted, absent = (_selected_pairs(selection, overrides, count) if rule
                      else (listed, listed_absent) if selection == "custom" else ((), ()))
    baseline = set(_rule_pairs(selection, count)) if rule else set()
    derived = [pair for pair in wanted if pair in baseline]
    facts["enabled"] = True
    facts["selected"] = sorted([start, end - 1] for start, end in wanted)
    # What the rule selects by itself: an edit against it is an exception.
    facts["baseline"] = sorted([start, end - 1] for start, end in baseline)
    facts["states"] = {f"{start},{end - 1}": "derived" for start, end in derived}
    facts["tips"] = {f"{start},{end - 1}": f"{pair_label(frames[start], frames[end])} follows {labels[selection]}; "
                                           "Ctrl-click to leave it out" for start, end in derived}
    facts["override_cells"] = [cell(start, end) for start, end, _selected in overrides]
    facts["list_cells"] = [None if pair is None else cell(*pair) for pair in row_pairs]
    facts["summary"] = f"{labels[selection]} · {len(wanted)} of {count * (count - 1) // 2} pairs"
    # A parked row naming a pair the list publishes is not said: the selected row wins.
    shown = set(published or ())
    parked = [row for row, pair in zip(rows, row_pairs)
              if not row["selected"] and pair not in shown] if selection == "custom" else []
    facts["note"] = "" if not absent else _sentence("kept, not in this source: " + ", ".join(
        pair_label(start, end) for start, end, _selected in absent))
    facts["parked"] = "" if not parked else _sentence("kept, not selected: " + ", ".join(
        pair_label(row["start"] or "(no start)", row["end"] or "(no end)") for row in parked))
    return facts


__all__ = ["PAIR_MATRIX_FIELDS", "pair_matrix_facts"]
