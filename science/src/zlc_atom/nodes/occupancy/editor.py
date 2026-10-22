"""What Occupancy's editors show, as data (the node's ``editor_facts``).

v2 drew the calibrations as a Qt grid in this package: one row per file, the
frames it reads as columns ("First" and "Last" their own, the numbered ones
the MIDDLE frames).  In v3 the grid is the UI's (apps/desktop/ui/src/views/
node_editors/frame_claims.ts) and holds no claim rule: how the source's frames
read the files (a frame through Rest, one nobody or two files read, an absent
middle number), what each file is, and what is refused or kept are answered
here by :mod:`zlc_atom.nodes._framework.frames`, from the draft's claims, the
files the workspace resolves and the frames the source carries.  The grid
writes back the claims (``calibrations``) and the readout model.

Under it the post-selection section (post_selection.ts, v2 PostSelectionEditor):
its switch, and while it is on the Mode strip and one "Keep where" row.  Which
frames are offered (named and ordered as the grid's columns), what the
condition keeps, what is refused and what the source lacks are answered here;
the section writes back the switch and the condition's fields.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Mapping, Sequence

from zlc_atom.nodes._framework.descriptor import frame_claim_entries
from zlc_atom.nodes._framework.frames import (
    FIRST, LAST, REST, FrameToken, absent_notes, canonical_frames, claim_names, claim_problems, frame_problems,
    parked_entries, reach_frames, resolve_frame,
)

from .post_selection import (
    ANY, FRAME_A, FRAME_B, MODE, POST_SELECTION, POST_SELECTION_FIELDS, RELATION, VALUE_A, VALUE_B, condition,
    source_problem,
)

#: The artifact input this editor manages.
CALIBRATIONS = "calibrations"
_HINT = ("Click a cell to give that frame to a file. First and Last follow the cycle's ends; "
         "middle frames keep their number. A file with no frame is kept and not read.")
_EMPTY = "Add a calibration file; the first one reads every frame."


def _sentence(text: str) -> str:
    text = str(text).rstrip()
    return text if not text or text.endswith((".", "?", "!", "…")) else f"{text}."


def frame_names(frames, claimed) -> dict[FrameToken, str]:
    """The frames a claim can name, as the columns call them: First and Last
    (with the source's label once its ``frames`` are known), each middle frame
    by its label, then each ``claimed`` middle number the source does not carry
    between its ends by its number, so the claim stays visible."""

    count = None if frames is None else len(frames)
    middle = () if count is None else range(1, count - 1)
    return {
        FIRST: "First" if count is None else f"First · {frames[0]}",
        LAST: "Last" if count is None else f"Last · {frames[-1]}",
        **{number: frames[number] for number in middle},
        **{number: str(number) for number in sorted(set(claimed)) if number not in middle},
    }


@functools.lru_cache(maxsize=64)
def _described(path: str, stamp: tuple[int, int], codec) -> tuple[str, str]:
    """(what the file is, why it cannot be read): read once per version of the file."""

    del stamp
    try:
        value = codec.resolve(path).value
    except Exception as error:  # noqa: BLE001 -- said on the row, as v2's projection said it
        return "", str(error) or type(error).__name__
    return ("" if codec.describe is None else str(codec.describe(value))), ""


def _file(path: str, codec) -> dict[str, str]:
    try:
        stat = os.stat(path)
        stamp = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        stamp = (0, 0)
    summary, unread = _described(path, stamp, codec)
    return {"summary": summary, "unread": unread}


def calibrations_facts(values: Mapping[str, object], artifacts: Mapping | None, source: Sequence | None, *, codec,
                       label: str) -> dict:
    """The calibrations grid for one draft (v2 _claims_projection + CalibrationsEditor._draw)."""

    inputs = dict((artifacts or {}).get("inputs") or {})
    resolved = dict((artifacts or {}).get("resolved") or {})
    frames = None if source is None else tuple(str(label) for label in source[1])
    try:
        entries = frame_claim_entries(inputs.get(CALIBRATIONS, ()))
        problems: tuple[str, ...] = ()
    except ValueError as error:
        entries, problems = (), (str(error),)
    names = claim_names(entries)
    claimed = tuple(entry["frames"] for entry in entries)
    notes: tuple[str, ...] = ()
    reach = None
    count = None if frames is None else len(frames)
    if not problems:
        problems = claim_problems(claimed, names, what="file")
        if entries and count is not None:
            if not problems:
                problems = frame_problems(claimed, count, names, what="file")
            notes = absent_notes(claimed, count, names)
            reach = reach_frames(claimed, count)
    files = [None if not entry["path"] else _file(str(resolved.get(entry["path"], entry["path"])), codec)
             for entry in entries]
    # A file with no frame is kept and not read: muted, never red.
    parked = parked_entries(claimed)
    named = frame_names(frames, (token for entry in entries for token in entry["frames"] if isinstance(token, int)))
    # The numbered columns are the middle frames; the ends are First and Last.
    middle = () if count is None else tuple(range(1, count - 1))
    columns = (FIRST, LAST, REST, None, *(token for token in named if isinstance(token, int)))
    ends = {FIRST: 0, LAST: None if count is None else count - 1}
    labels = tuple("Rest" if token == REST else "" if token is None else named.get(token, "") for token in columns)
    cells, selected, states, tips = [], [], {}, {}
    for row, entry in enumerate(entries):
        for column, token in enumerate(columns):
            if token is None:
                continue
            name = f"{row},{column}"
            frame = ends[token] if token in ends else token if isinstance(token, int) and token in middle else None
            if isinstance(token, int) and token not in middle:
                if token not in entry["frames"]:
                    continue
                if count is not None:
                    states[name] = "kept"
                    tips[name] = f"kept: middle frame {token} is not between this source's ends; click to drop it"
            elif frame is not None and reach is not None:
                readers = reach.readers[frame]
                mine = next((how for reader, how in readers if reader == row), None)
                if len(readers) > 1 and mine is not None:
                    states[name] = "error"
                    tips[name] = "read by " + " and ".join(f"{names[reader]} ({how})" for reader, how in readers)
                elif not readers:
                    states[name] = "missing"
                    tips[name] = f"frame {frame} has no calibration; click to give it to this file"
                elif mine == REST and token not in entry["frames"]:
                    states[name] = "derived"
                    tips[name] = f"frame {frame} is read as rest"
            cells.append([row, column, labels[column]])
            if token in entry["frames"]:
                selected.append([row, column])

    def parked_tip(index):
        item = files[index]
        same = next((other for other, theirs in enumerate(entries)
                     if other != index and item is not None and theirs["path"] == entries[index]["path"]), None)
        return "kept · reads no frame" + ("" if same is None else f" · same file as {names[same]}")

    row_tips = ["\n".join(part for part in (
        entry["path"] or "no file yet",
        "" if item is None else f"cannot be read: {item['unread']}" if item["unread"] else item["summary"],
        parked_tip(index) if index in parked else "" if reach is None else
        "reads " + (", ".join(str(frame) for frame in reach.frames_of(index)) or "no frame"),
    ) if part) for index, (entry, item) in enumerate(zip(entries, files))]
    reading = [index for index in range(len(entries)) if index not in parked]
    not_read = f" · {len(parked)} file{'s' if len(parked) != 1 else ''} not read" if parked else ""
    frames_word = f"frame{'s' if count != 1 else ''}"
    summary = ("No calibration yet" if not entries else "Waiting for the source frames" if count is None
               else f"{count} {frames_word}{not_read}" if problems
               else f"{names[reading[0]]} reads every frame{not_read}" if len(reading) == 1
               else f"{count} {frames_word} · each read once{not_read}")
    return {
        "editor": "frame_claims",
        "manages": ["model_kind"],
        "manages_artifacts": [CALIBRATIONS],
        "title": label, "summary": summary,
        "grid": {"rows": list(names), "columns": list(labels), "cells": cells, "rowTips": row_tips,
                 "mutedRows": list(parked), "narrowColumns": [3]},
        "selected": selected, "states": states, "tips": tips,
        # How a column is written in a claim (None: the gap between the ends and the middle frames).
        "columns": list(columns),
        "entries": [{"path": entry["path"], "frames": list(entry["frames"])} for entry in entries],
        # The first file reads every frame; a later one is kept reading none until given frames.
        "new_frames": [] if any(REST in entry["frames"] for entry in entries) else [REST],
        "file_filter": codec.file_filter,
        "problems": [_sentence(problem) for problem in problems],
        "notes": [_sentence(note) for note in notes],
        "hint": _HINT if entries else _EMPTY,
    }


def _choices(field) -> list:
    return [[choice.value, choice.label] for choice in field.choices]


def _offer(frames, written: str) -> tuple[list, str]:
    """The frames ``frames`` carries and the one ``written`` names (its key):
    offered too when the source lacks it, and as written when it names no frame."""

    try:
        (current,) = canonical_frames((written,))
    except ValueError:
        current = written
    named = frame_names(frames, (current,) if isinstance(current, int) else ())
    if isinstance(current, int) and frames is not None and resolve_frame(current, len(frames)) is None:
        named[current] += " · not between the ends"
    offered = {str(token): label for token, label in named.items()}
    offered.setdefault(str(current), written)
    return [[value, label] for value, label in offered.items()], str(current)


def _summary(kept) -> str:
    """The condition as the title row reads it: ``kept where frame first = frame last``."""

    names = tuple(f"frame {token}" for token in kept["frames"])
    if kept["mode"] == "relation":
        return f"kept where {names[0]} {dict(_choices(RELATION))[kept['relation']]} {names[1]}"
    verdicts = dict(_choices(VALUE_A))
    return "kept where " + " and ".join(
        f"{name} is {verdicts[value]}" for name, value in zip(names, kept["values"]) if value != ANY)


def post_selection_facts(values: Mapping[str, object], source: Sequence | None) -> dict:
    """The post-selection section for one draft (v2 PostSelectionEditor.update_projection)."""

    frames = None if source is None else tuple(str(label) for label in source[1])
    try:
        kept, problem = condition(values), ""
    except ValueError as error:
        kept, problem = None, str(error)
    strips = {field.name: {"label": field.label, "choices": _choices(field), "current": values[field.name]}
              for field in (MODE, RELATION, VALUE_A, VALUE_B)}
    return {
        "editor": "post_selection",
        "manages": [field.name for field in POST_SELECTION_FIELDS],
        "title": POST_SELECTION.label,
        "on": bool(values[POST_SELECTION.name]),
        "summary": "" if kept is None else _summary(kept),
        "relation_mode": values[MODE.name] == "relation",
        "strips": strips,
        "frames": {field.name: dict(zip(("choices", "current"), _offer(frames, str(values[field.name]))), label=field.label)
                   for field in (FRAME_A, FRAME_B)},
        "problem": _sentence(problem),
        "note": "" if kept is None or frames is None else _sentence(source_problem(kept, len(frames))),
    }


__all__ = ["CALIBRATIONS", "calibrations_facts", "frame_names", "post_selection_facts"]
