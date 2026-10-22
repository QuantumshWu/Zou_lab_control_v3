"""Discoverable frame-survival processor descriptor."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
from zlc_plugin import DatasetInput, NodeKind, NodeType

from .editor import pair_matrix_facts
from .processor import PAIR_RULES, SURVIVAL_OUTPUTS, FrameSurvivalProcessor, _checked_overrides, _checked_pairs

#: What the operator picks: a question that follows the frames the source
#: carries, no pair at all, or the pairs they marked themselves.  The labels
#: are the words on the editor's preset strip and in its summary, so they are
#: short; what each rule pairs (k→k+1, 0→k, k→last) is in the field's
#: description.  None and Selected are two choices, not one: None asks for no
#: pair, Selected for the list being marked -- empty while it is begun -- so
#: the strip marks the one the operator picked and either can follow the other.
_SELECTIONS = (
    AuthoringChoice("all", "All"),
    AuthoringChoice("consecutive", "Consecutive"),
    AuthoringChoice("from_first", "From first"),
    AuthoringChoice("to_last", "To last"),
    AuthoringChoice("none", "None"),
    AuthoringChoice("custom", "Selected"),
)
#: A rule's exception: selected adds the pair to the rule, unselected leaves
#: it out.  Every row is in use, so every row is checked.
_OVERRIDE_COLUMNS = (
    AuthoringField("start", "text", "Start frame", "0", required=True),
    AuthoringField("end", "text", "End frame", "last", required=True),
    AuthoringField("selected", "bool", "Selected", True),
)
#: A row of Selected's own list: a selected row is published and checked;
#: one kept unselected is parked -- kept as written, its ends filled in or
#: not, and never checked, against the rows in use or at all.
_LIST_COLUMNS = (
    AuthoringField("start", "text", "Start frame", "0"),
    AuthoringField("end", "text", "End frame", "last"),
    AuthoringField("selected", "bool", "Selected", True),
)
_OVERRIDES = AuthoringField(
    "pairs", "rows", "Rule overrides", (), columns=_OVERRIDE_COLUMNS, enabled_when=("selection", PAIR_RULES),
    description="The rule's exceptions: selected=true adds a pair to it, false leaves one out.")
_LIST = AuthoringField(
    "custom_pairs", "rows", "Selected pairs", (), columns=_LIST_COLUMNS, enabled_when=("selection", ("custom",)),
    description="Selected's own list: selected=true pairs are published; selected=false ones are parked -- "
                "kept as written, never checked and never published. The list is kept while a rule or None "
                "is picked, so Selected comes back to it.")


def _overrides(rows):
    """A rule's exceptions as checked ``(start, end, selected)``."""

    return _checked_overrides(tuple((row["start"], row["end"], row["selected"]) for row in rows))


def _published(rows):
    """The pairs Selected's list publishes, checked: its selected rows, and only those."""

    for index, row in enumerate(rows, 1):
        for column in _LIST_COLUMNS[:2]:
            if row["selected"] and not row[column.name].strip():
                raise ValueError(f"{_LIST.label} row {index}: {column.label} must be filled in")
    selected = tuple((row["start"], row["end"]) for row in rows if row["selected"])
    return _checked_pairs(selected) if selected else ()


def _parked(rows):
    """The pairs Selected's list keeps unselected, as written."""

    return tuple((row["start"], row["end"]) for row in rows if not row["selected"])


def _validate(values):
    # Only the list the selection reads is checked: the other one is kept as
    # it was written (enabled_when), for when its choice is picked again.  Of
    # Selected's list only the selected rows are: a parked row never stops Start.
    selection = values["selection"]
    if selection in PAIR_RULES:
        _overrides(values["pairs"])
    elif selection != "custom" or not _published(values["custom_pairs"]):
        raise ValueError("Pick a rule or select at least one pair.")


FRAME_SURVIVAL_SCHEMA = AuthoringSchema((
    AuthoringField("selection", "choice", "Published pairs", "all", required=True, choices=_SELECTIONS,
                   description="A named choice follows the frames the source carries: All forward pairs, "
                               "Consecutive (k→k+1), From first (0→k) or To last (k→last); None asks for "
                               "no pair; Selected publishes its own list. Grid edits override only the changed "
                               "pairs; first/last follow the cycle."),
    _OVERRIDES,
    _LIST,
), validator=_validate)


def _editor_facts(values, resources, devices, source, bundle, artifacts):
    """The pair matrix over the frames the source carries (frame_survival.editor)."""

    del resources, devices, bundle, artifacts
    return pair_matrix_facts(values, source, schema=FRAME_SURVIVAL_SCHEMA, overrides_field=_OVERRIDES.name,
                             list_field=_LIST.name, read_overrides=_overrides, read_list=_published)


def _build(*, source_signal: str, **values: object) -> FrameSurvivalProcessor:
    authored = FRAME_SURVIVAL_SCHEMA.project_values(values)
    selected_source = str(source_signal).strip()
    if not selected_source:
        raise ValueError("source_signal must be non-empty")
    selection = authored["selection"]
    if selection != "custom":
        return FrameSurvivalProcessor(source_signal=selected_source, pairs=selection,
                                      pair_overrides=_overrides(authored["pairs"]))
    listed = authored["custom_pairs"]
    return FrameSurvivalProcessor(source_signal=selected_source, pairs=_published(listed),
                                  parked_pairs=_parked(listed))


LOGIC_NODE = NodeType(
    "frame_survival",
    NodeKind.PROCESSOR,
    FRAME_SURVIVAL_SCHEMA,
    description=(
        "Turn Occupancy occupied data into survival for the selected earlier/later "
        "frame pairs, retaining site and scan axes. Each value is the later verdict; "
        "validity requires the earlier frame occupied and both frames judgeable, so "
        "Mean over repeats gives conditional survival. Choose the pairs with selection: "
        "all, consecutive (k->k+1), from_first (0->k) and to_last (k->last) follow the "
        "frames the source carries; none asks for no pair (Start waits for a rule or a pair). "
        "Under a rule, pairs rows override it with selected=true/false; custom publishes "
        "its own custom_pairs list, whose selected=false rows are parked: kept as written, "
        "never checked and never published. That list is kept while a rule or none is "
        "picked. Endpoints use text "
        "'0'/'last'; numbered middle frames keep their index when the cycle changes, and "
        "a selected pair whose middle frame the source lacks is kept and left out of the "
        "run, which fails only when no selected pair is left. "
        "Feed survival to Seamless Scan and use a site Facet with native Exponential "
        "fitting for per-site lifetimes."
    ),
    input_specs=(DatasetInput("occupied", None, "exact"),),
    outputs=SURVIVAL_OUTPUTS,
    build=_build,
    editor_facts=_editor_facts,
)

__all__ = ["FRAME_SURVIVAL_SCHEMA", "LOGIC_NODE"]
