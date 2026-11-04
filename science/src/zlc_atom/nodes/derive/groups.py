"""What stands between publications: statistics in groups over a Repeat or Point axis.

A group over such an axis spans events, so each open group keeps, per kept
cell, the moments of what has arrived and never a sample: Runtime alone stores
data.  An increment is whatever one delivery carries; a sealed source's whole
dataset is one increment that closes every group, a live source's events are
many.  There is no other path, and a closed group is given once and never
changes.

A group closes when every cell it expects has arrived, or -- repeats run in
order, Point scans may be shuffled -- when a later group of its axis arrived
beside the same coordinates of the other kept Repeat axes, or when the source
sealed.  A variable that groups starts a clock; what reads it
evaluates only on what just closed (:func:`.expression._clock`), each closed
run of groups as one :class:`Chunk`.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from zlc_data import DatasetSchema, DomainSpec
from zlc_data.snapshot_projection import _subset_mapped_domain

from .expression import (
    Operand, _answering, _apply_reduction_plans, _axis, _axis_labels, _domains, _primary, _reduce_plan, _schema, _structure,
    _unit, centred, evaluate, fit_cells, fit_operands, fit_schemas, fit_target, formula, group_domain, reads_for_fit,
    sigma_dof, stated_square, standard_error, statistic_of, typed_variables,
)


def restricted(schema: DatasetSchema, origin: Sequence[int], extent: Sequence[int]) -> DatasetSchema:
    """``schema`` over these Repeat and Point rows: the axes of the whole, cut."""

    def cut(domain, start, size):
        if domain.axis_codes is None:
            domain = DomainSpec(domain.shape, domain.axes, tuple(range(axis.size) for axis in domain.axes))
        return _subset_mapped_domain(domain, range(start, start + size))

    domains = _domains(schema)
    return _schema((cut(domains[0], origin[0], extent[0]), cut(domains[1], origin[1], extent[1]), domains[2]),
                   schema.value_schema.dtype, _unit(schema), schema.value_schema.name)


@dataclass(frozen=True, eq=False)
class Chunk:
    """Closed groups: an Operand of the rectangle at ``origin`` of its variable's
    whole schema; ``ordinal`` is a monitor's group index, None on a finite one."""

    operand: Operand
    origin: tuple[int, int]
    ordinal: int | None = None


# -- moments -----------------------------------------------------------------

def _measure(item: Operand, plans, operation: str) -> dict[str, np.ndarray]:
    """One increment's moments per output bucket, as :func:`_merge` adds them."""

    count = _apply_reduction_plans(item.valid, plans, "count")
    valid = count > 0
    part = {"n": count}
    if operation in ("mean", "std"):
        part["mean"], part["m2"] = centred(item, plans, count, valid)
        if operation == "mean" and item.sigma is not None:
            part["stated"] = stated_square(item, plans)
    elif operation == "sum" and item.values.dtype.kind == "b":
        part["total"] = _apply_reduction_plans(item.valid & item.values, plans, "count")
    elif operation == "sum":
        part["total"] = _apply_reduction_plans(np.where(item.valid, item.values, 0.0).astype(np.float64), plans, "sum")
    elif operation in ("min", "max"):
        fill = np.inf if operation == "min" else -np.inf
        part["extreme"] = _apply_reduction_plans(np.where(item.valid, item.values, fill).astype(np.float64), plans, operation)
    elif operation in ("any", "all"):
        part["flag"] = _apply_reduction_plans(np.where(item.valid, item.values, operation == "all"), plans, operation)
    return part


def _empty(shape, part, operation) -> dict[str, np.ndarray]:
    start = {"extreme": np.inf if operation == "min" else -np.inf, "flag": operation == "all"}
    return {key: np.full((*shape, *plane.shape), start.get(key, 0), dtype=plane.dtype)
            for key, plane in part.items()}


def _merge(tally, index, part, operation) -> None:
    """``part`` into ``tally[index]``: Welford's merge of means and squares."""

    n1, n2 = tally["n"][index], part["n"]
    total = n1 + n2
    if "mean" in part:
        delta = part["mean"] - tally["mean"][index]
        shift = np.divide(delta * n2, total, out=np.zeros(total.shape), where=total > 0)
        tally["m2"][index] += part["m2"] + delta * delta * np.divide(n1 * n2, total, out=np.zeros(total.shape), where=total > 0)
        tally["mean"][index] += shift
    if "stated" in part:
        tally["stated"][index] += part["stated"]
    if "total" in part:
        tally["total"][index] += part["total"]
    if "extreme" in part:
        tally["extreme"][index] = (np.minimum if operation == "min" else np.maximum)(tally["extreme"][index], part["extreme"])
    if "flag" in part:
        tally["flag"][index] = (np.logical_and if operation == "all" else np.logical_or)(tally["flag"][index], part["flag"])
    tally["n"][index] = total


def _result(tally, operation) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """(values, valid, sigma, its degrees of freedom) of one tally, as the statistic reads over whole data."""

    count = tally["n"]
    valid = count > 0
    if operation == "count":
        return count, valid, None, None
    if operation == "sum":
        return tally["total"], valid, None, None
    if operation in ("min", "max"):
        return tally["extreme"], valid, None, None
    if operation in ("any", "all"):
        return tally["flag"], valid, None, None
    if operation == "std":
        return np.sqrt(np.divide(tally["m2"], count, out=np.full(count.shape, np.nan), where=valid)), valid, None, None
    return (np.where(valid, tally["mean"], np.nan), valid,
            standard_error(tally["m2"], count, tally.get("stated")), sigma_dof(count))


# -- one variable's groups ---------------------------------------------------

class Accumulator:
    """One grouping variable's open groups.

    ``reducible`` is what its statistic reduces over the whole source
    (a monitor's: one event); ``spanning`` says a monitor's group index is the
    source ordinal, one shot per coordinate.  A Repeat group's unit is its
    output row, with every kept Point row; a Point group's is one output
    (Repeat, Point) cell.
    """

    def __init__(self, row: Mapping[str, object], reducible: DatasetSchema, *, spanning: bool):
        self.row, self.reducible = row, reducible
        self.operation, self.spanning = str(row["statistic"]), spanning
        self.domain = group_domain(row, reducible)
        self.schema, plans = _reduce_plan(reducible, self.operation, row["over"], row["by"], row["every"])
        mapped = {plan[1]: plan[2] for plan in plans if plan[0] == "mapped"}
        self._cell_plans = tuple(plan for plan in plans if plan[0] != "mapped")
        domains = _domains(reducible)
        self._inverse = tuple(mapped.get(d, np.arange(domains[d].size)) for d in (0, 1))
        self._counts = tuple(np.bincount(inverse) for inverse in self._inverse)
        self._tallies: dict[tuple[int, int], dict] = {}
        self._arrived: dict[tuple[int, int], int] = {}
        self._closed: set[tuple[int, int]] = set()
        #: Each kept Repeat row's (line, place along the axis in groups): a line is
        #: one set of coordinates of the other kept Repeat axes, in order on its own.
        self._line, self._step = _repeat_lines(self.schema, row["by"]) if self.domain == 0 and not spanning else (None, None)
        self._latest: dict[int, int] = {}
        #: Each kept row's identity, by its axes' codes (filled when a pick needs it).
        self._rows: list[dict | None] = [None, None]
        self._same: list[bool | None] = [None, None]

    def _canonical(self, d: int, view: DomainSpec, source: tuple[int, DomainSpec] | None) -> np.ndarray:
        """The row of the whole reducible each row of this increment is."""

        whole = _domains(self.reducible)[d]
        if view.size == whole.size:
            return np.arange(whole.size)
        if self._same[d] is None:
            self._same[d] = source is not None and source[1] == whole
        if self._same[d]:
            return source[0] + np.arange(view.size)
        # A pick dropped rows: a row is known by the codes of its axes.
        axes = tuple(axis for axis in whole.axes if axis.coordinate_of is None)
        if self._rows[d] is None:
            self._rows[d] = {key: row for row, key in enumerate(zip(*(whole.codes(a.axis_id).tolist() for a in axes)))}
        columns = []
        for axis in axes:
            positions = _axis_labels(axis)[1]
            labels = view.axis(axis.axis_id).labels()
            columns.append(np.array([positions[label] for label in labels])[view.codes(axis.axis_id)].tolist())
        return np.array([self._rows[d][key] for key in zip(*columns)], dtype=np.int64)

    def _expected(self, unit) -> int:
        if self.spanning:
            return int(self.row["every"]) * int(self._counts[0][0]) * _domains(self.reducible)[1].size
        if self.domain == 0:
            return int(self._counts[0][unit[0]]) * _domains(self.reducible)[1].size
        return int(self._counts[0][unit[0]]) * int(self._counts[1][unit[1]])

    def feed(self, item: Operand, sources: tuple[tuple[int, DomainSpec], tuple[int, DomainSpec]] | None,
             ordinal: int | None = None) -> tuple[Chunk, ...]:
        """One increment: its moments into its groups; what that closes."""

        view = _domains(item.schema)
        out = [self._inverse[d][self._canonical(d, view[d], None if sources is None else sources[d])] for d in (0, 1)]
        unique_r, local_r = np.unique(out[0], return_inverse=True)
        unique_p, local_p = np.unique(out[1], return_inverse=True)
        plans = (("mapped", 0, local_r.reshape(-1), unique_r.size), ("mapped", 1, local_p.reshape(-1), unique_p.size),
                 *self._cell_plans)
        part = _measure(item, plans, self.operation)
        hit_r, hit_p = np.bincount(local_r.reshape(-1)), np.bincount(local_p.reshape(-1))
        touched = []
        for i, row in enumerate(unique_r.tolist()):
            row = ordinal // int(self.row["every"]) if self.spanning else row
            if self.domain == 0:
                units = (((row, 0), i, slice(None), int(hit_r[i]) * view[1].size),)
            else:
                units = tuple(((row, group), i, j, int(hit_r[i]) * int(hit_p[j])) for j, group in enumerate(unique_p.tolist()))
            touched.extend(self._add(*unit, part, unique_p) for unit in units)
        if self.domain == 0:
            for unit in touched:
                line, step = self._place(unit)
                self._latest[line] = max(self._latest.get(line, step), step)
        return self._close(self._due(touched))

    def _place(self, unit) -> tuple[int, int]:
        """A monitor's rows are its groups, in one line."""

        return (0, unit[0]) if self._line is None else (int(self._line[unit[0]]), int(self._step[unit[0]]))

    def _followed(self, unit) -> bool:
        """Whether a later group stands in this unit's line: nothing more comes to it."""

        line, step = self._place(unit)
        return step < self._latest.get(line, step)

    def _add(self, unit, i, j, arrivals, part, unique_p):
        if unit in self._closed or (self.domain == 0 and self._followed(unit) and unit not in self._tallies):
            raise ValueError(f"{self.row['name']}: a {self.row['by']} sample arrived after its group closed; "
                             "repeats run in order")
        slices = {key: plane[i, j] for key, plane in part.items()}
        tally = self._tallies.get(unit)
        if tally is None:
            shape = (1, _domains(self.schema)[1].size) if self.domain == 0 else (1, 1)
            bare = {key: plane[0, 0] for key, plane in part.items()}
            tally = self._tallies[unit] = _empty(shape, bare, self.operation)
        _merge(tally, (0, unique_p) if self.domain == 0 else (0, 0), slices, self.operation)
        self._arrived[unit] = self._arrived.get(unit, 0) + arrivals
        return unit

    def _due(self, touched) -> list:
        due = [unit for unit in dict.fromkeys(touched) if self._arrived[unit] >= self._expected(unit)]
        if self.domain == 0:
            due.extend(unit for unit in self._tallies if self._followed(unit) and unit not in due)
        return due

    def close_all(self) -> tuple[Chunk, ...]:
        """The source sealed: every open group as it stands, its n smaller."""

        return self._close(list(self._tallies))

    def _close(self, units) -> tuple[Chunk, ...]:
        if not units:
            return ()
        results = {}
        for unit in units:
            results[unit] = _result(self._tallies.pop(unit), self.operation)
            del self._arrived[unit]
        self._closed.update(units)
        if self.domain == 0:
            self._closed = {unit for unit in self._closed if not self._followed(unit)}
        return tuple(self._chunk(block, results) for block in _rectangles(units, spanning=self.spanning))

    def _chunk(self, block, results) -> Chunk:
        rows, columns = block
        if self.spanning:
            schema, origin, ordinal = self.schema, (0, 0), rows[0]
        else:
            size = (len(rows), _domains(self.schema)[1].size if self.domain == 0 else len(columns))
            origin, ordinal = (rows[0], columns[0]), None
            schema = restricted(self.schema, origin, size)
        values, valid, sigma, dof = (
            None if results[(rows[0], columns[0])][k] is None else
            np.concatenate([np.concatenate([results[(row, column)][k] for column in columns], axis=1) for row in rows], axis=0)
            for k in range(4))
        dtype = schema.value_schema.dtype
        return Chunk(Operand(schema, values.astype(dtype, copy=False), valid, sigma, dof), origin, ordinal)


def _repeat_lines(schema: DatasetSchema, by: str) -> tuple[np.ndarray, np.ndarray]:
    """(line, place along ``by``) of each Repeat row of a schema grouped by it."""

    domain = _domains(schema)[0]
    axes = _primary(domain)
    others = [domain.codes(axis.axis_id) for axis in axes if axis.name != by]
    line = np.unique(np.column_stack(others), axis=0, return_inverse=True)[1].reshape(-1) if others else np.zeros(domain.size, np.int64)
    return line, np.asarray(domain.codes(next(axis for axis in axes if axis.name == by).axis_id))


def _rectangles(units, *, spanning) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Closed units as (rows, columns) rectangles of consecutive ones, in order:
    one commit holds each.  A monitor gives one event per group."""

    runs: dict[int, list[list[int]]] = {}
    for row, column in sorted(units):
        columns = runs.setdefault(row, [])
        if columns and columns[-1][-1] + 1 == column:
            columns[-1].append(column)
        else:
            columns.append([column])
    blocks: list[tuple[list[int], list[int]]] = []
    open_blocks: dict[tuple[int, ...], list[int]] = {}
    for row in sorted(runs):
        following = {}
        for columns in runs[row]:
            key = tuple(columns)
            rows = open_blocks.get(key)
            if not spanning and rows is not None and rows[-1] + 1 == row:
                rows.append(row)
            else:
                rows = [row]
                blocks.append((rows, list(columns)))
            following[key] = rows
        open_blocks = following
    return [(tuple(rows), tuple(columns)) for rows, columns in blocks]


# -- a draft's run -----------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Step:
    """One variable on a clock: it groups, or evaluates on what it reads just closed."""

    position: int
    row: Mapping[str, object]
    answer: Mapping[str, object]


class Accumulation:
    """A draft's variables over a source, one delivery after another.

    Variables on no clock evaluate over each publication as they always did
    (:func:`evaluate`); the rest over the same publication's rows as the whole
    source names them, so a chunk is a cut of what typing answered.
    """

    def __init__(self, rows: Sequence[Mapping[str, object]], inputs: Mapping[str, object], names: Sequence[str]):
        self.rows = tuple(rows)
        self.members = tuple(inputs)
        primary = inputs[names[0]]
        self.finite = primary.canonical_schema is not None
        schemas = {name: inputs[name].canonical_schema or inputs[name].snapshot.block.schema for name in names}
        answers = typed_variables(self.rows, {name: schemas.get(name) for name in names})
        for position, answer in enumerate(answers):
            if answer["error"] is not None and answer["live"]:
                raise answer["error"]
        self.edges, order, _closing, live = _structure(self.rows, tuple(names))
        self.weighed = reads_for_fit(self.rows, self.edges)
        self.steps = tuple(Step(position, self.rows[position], answers[position]) for position in order
                           if position in live and answers[position]["clock"] is not None)
        self.on_clock = {step.position for step in self.steps}
        self.first = _answering(self.rows)
        #: Only what these publish makes the plain evaluation read anything.
        self.plain_rows = tuple({**row, "publish": row["publish"] and position not in self.on_clock}
                                for position, row in enumerate(self.rows))
        self.plain = any(row["publish"] for row in self.plain_rows)
        #: The variables on no clock that a grouping one reads: over its rows, not a publication's.
        helpers = {other for step in self.steps for _k, other in self.edges[step.position]} - self.on_clock
        self.helper_rows = tuple({**row, "publish": position in helpers} for position, row in enumerate(self.rows))
        self.helpers = bool(helpers)
        self.accumulators: dict[int, Accumulator] = {}
        self.warm: dict[int, np.ndarray] = {}

    def outputs(self) -> dict[str, tuple[DatasetSchema, int]]:
        """Each published output on a clock: its schema over the whole run."""

        found = {}
        for step in self.steps:
            if step.row["publish"]:
                schemas = step.answer["fit"] or {step.row["name"]: step.answer["schema"]}
                found.update(schemas)
        return found

    def _accumulator(self, step: Step) -> Accumulator:
        """A monitor's event holds its Point rows whole, and its Repeat rows whole
        unless there is one to a coordinate: only then do its groups span events."""

        reducible = step.answer["reducible"]
        spanning = (not self.finite and group_domain(step.row, reducible) == 0
                    and _axis(reducible, step.row["by"], "the value", "reduce")[1].size == 1)
        if self.finite or spanning:
            found = self.accumulators.get(step.position)
            if found is None:
                found = self.accumulators[step.position] = Accumulator(step.row, reducible, spanning=spanning)
            return found
        return Accumulator(step.row, reducible, spanning=False)

    def step(self, members: Mapping[str, Operand] | None, sources, ordinal: int | None, fit) -> dict[str, list[Chunk]]:
        """One delivery's chunks by output name; ``members`` None: the source sealed."""

        fresh: dict[str, list[Chunk]] = {}
        values = members
        if members is not None and self.helpers:
            values = {**members, **evaluate(self.helper_rows, members, fit)}
        for step in self.steps:
            row = step.row
            if any(other in self.on_clock for _k, other in self.edges[step.position]):
                reads = [term["of"] for term in row["terms"] if term["of"] in self.first]
                feeds = tuple((
                    {name: fresh[name][k].operand for name in reads},
                    tuple(zip(fresh[reads[0]][k].origin, _domains(self._whole(reads[0])))),
                    fresh[reads[0]][k].ordinal,
                ) for k in range(len(fresh[reads[0]])))
            else:
                feeds = () if values is None else ((values, sources, ordinal),)
            is_group = group_domain(row, step.answer["reducible"]) is not None
            given = []
            for reading, where, shot in feeds:
                value = formula(row, reading.__getitem__, True)
                if is_group:
                    given.extend(self._accumulator(step).feed(value, where, shot))
                else:
                    given.append((value, where, shot))
            if is_group:
                if members is None and step.position in self.accumulators:
                    given.extend(self.accumulators[step.position].close_all())
                chunks = {row["name"]: given}
            else:
                chunks = self._reader(step, given, fit)
            fresh.update(chunks)
        return fresh

    def _whole(self, name) -> DatasetSchema:
        answer = next(step.answer for step in self.steps if step.position == self.first[name])
        return (answer["fit"] or {name: answer["schema"]})[name]

    def _reader(self, step: Step, feeds, fit) -> dict[str, list[Chunk]]:
        """A variable that reads what closed, evaluated once on each chunk of it."""

        row, answer = step.row, step.answer
        chunks: dict[str, list[Chunk]] = {name: [] for name in (answer["fit"] or (row["name"],))}
        for value, where, shot in feeds:
            if row["statistic"] == "fit":
                results = self._fit_run(step, value, fit)
            else:
                results = {row["name"]: statistic_of(row, value, None, step.position in self.weighed)}
            for name, result in results.items():
                whole = (answer["fit"] or {name: answer["schema"]})[name]
                origin = _origin(result, whole, where)
                chunks[name].append(Chunk(result, origin, shot))
        return chunks

    def _fit_run(self, step: Step, value: Operand, fit) -> dict[str, Operand]:
        """A fit of closed groups, group by group in order, each warm from the
        last good answer of its cell: so a whole dataset and its events alike."""

        row = step.row
        target = fit_target(value.schema, row["over"])
        schemas = fit_schemas(value.schema, target, row)
        d = step.answer["clock"][0]
        rows = _domains(value.schema)[d].size
        parts = []
        for k in range(rows):
            origin, extent = [0, 0], [_domains(value.schema)[0].size, _domains(value.schema)[1].size]
            origin[d], extent[d] = k, 1
            one = _take_rows(value, d, k, origin, extent)
            warm = self.warm.get(step.position)
            answer = fit_cells(one, fit_target(one.schema, row["over"]), row["model"], fit, warm)
            parts.append(answer)
            values, _errors, success = answer
            width = values.shape[-1]
            flat, good = values.reshape(-1, width), success.reshape(-1)
            if warm is None:
                warm = np.full(flat.shape, np.nan)
            self.warm[step.position] = np.where(good[:, None] & np.isfinite(flat), flat, warm)
        joined = [np.concatenate([part[k] for part in parts], axis=d) for k in range(3)]
        return fit_operands(schemas, *joined)


def _take_rows(value: Operand, d: int, k: int, origin, extent) -> Operand:
    cut = (slice(None),) * d + (slice(k, k + 1),)
    return Operand(restricted(value.schema, origin, extent), value.values[cut], value.valid[cut],
                   None if value.sigma is None else value.sigma[cut], None if value.dof is None else value.dof[cut])


def _origin(result: Operand, whole: DatasetSchema, where) -> tuple[int, int]:
    """Where a reader's chunk stands in its own whole schema: where what it read
    stood, unless the whole has fewer rows there and all of them are here."""

    view, full = _domains(result.schema), _domains(whole)
    origin = []
    for d in (0, 1):
        origin.append(0 if view[d].size == full[d].size else int(where[d][0]))
    return tuple(origin)


__all__ = ["Accumulation", "Accumulator", "Chunk", "restricted"]
