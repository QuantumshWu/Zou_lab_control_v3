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
from . import native_ops

from .expression import (
    Operand, _answering, _axis, _axis_labels, _domains, _primary, _reduce_plan, _schema, _structure,
    _unit, evaluate, fit_cells, fit_operands, fit_schemas, fit_target, formula, group_domain, reads_for_fit,
    statistic_of, typed_variables,
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


# -- one variable's groups ---------------------------------------------------

class Accumulator:
    """One grouping variable's open groups.

    ``reducible`` is what its statistic reduces over the whole source
    (a monitor's: one event); ``spanning`` says a monitor's group index is the
    source ordinal, one shot per coordinate.  A Repeat group's unit is its
    output row, with every kept Point row; a Point group's is one output
    (Repeat, Point) cell.
    """

    def __init__(self, row: Mapping[str, object], reducible: DatasetSchema, *, spanning: bool, transient: bool = False):
        self.row, self.reducible = row, reducible
        self.operation, self.spanning = str(row["statistic"]), spanning
        self.domain = group_domain(row, reducible)
        self.schema, plans = _reduce_plan(reducible, self.operation, row["over"], row["by"], row["every"])
        mapped = {plan[1]: plan[2] for plan in plans if plan[0] == "mapped"}
        self._cell_plans = tuple(plan for plan in plans if plan[0] != "mapped")
        domains = _domains(reducible)
        self._inverse = tuple(mapped.get(d, np.arange(domains[d].size)) for d in (0, 1))
        self._counts = tuple(np.bincount(inverse) for inverse in self._inverse)
        lines, steps = _repeat_lines(self.schema, row["by"]) if self.domain == 0 and not spanning else (None, None)
        self._rows: list[dict | None] = [None, None]
        self._same: list[bool | None] = [None, None]
        self._transient, self._finished = transient, False
        self._handle = native_ops.call(
            "scientific.group.create", statistic=self.operation, domain=self.domain,
            spanning=self.spanning, every=int(row["every"]), counts_r=self._counts[0].astype("<i8"),
            counts_p=self._counts[1].astype("<i8"), source_points=domains[1].size,
            output_points=_domains(self.schema)[1].size, lines=lines, steps=steps)["handle"]

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

    def feed(self, item: Operand, sources, ordinal=None) -> tuple[Chunk, ...]:
        view = _domains(item.schema)
        out = [self._inverse[d][self._canonical(d, view[d], None if sources is None else sources[d])] for d in (0, 1)]
        try:
            answer = native_ops.call("scientific.group.feed", handle=self._handle,
                                     values=item.values, valid=item.valid, sigma=item.sigma,
                                     rows=out[0].astype("<i8"), points=out[1].astype("<i8"),
                                     plans=native_ops.plans(self._cell_plans), ordinal=0 if ordinal is None else int(ordinal))
            return self._chunks(answer)
        finally:
            if self._transient:
                native_ops.call("scientific.group.drop", handle=self._handle)
                self._finished = True

    def close_all(self) -> tuple[Chunk, ...]:
        if self._finished:
            return ()
        self._finished = True
        return self._chunks(native_ops.call("scientific.group.finish", handle=self._handle))

    @property
    def held_bytes(self) -> int:
        """Numerical accumulator storage, owned by the native service."""
        return 0 if self._finished else int(native_ops.call("scientific.group.state", handle=self._handle)["held_bytes"])

    def _chunks(self, answer) -> tuple[Chunk, ...]:
        chunks = []
        for part in answer["chunks"]:
            rows, columns = part["rows"], part["columns"]
            if self.spanning:
                schema, origin, ordinal = self.schema, (0, 0), rows[0]
            else:
                size = (len(rows), _domains(self.schema)[1].size if self.domain == 0 else len(columns))
                origin, ordinal = (rows[0], columns[0]), None
                schema = restricted(self.schema, origin, size)
            chunks.append(Chunk(Operand(schema, part["values"], part["valid"], part.get("sigma"), part.get("dof")), origin, ordinal))
        return tuple(chunks)


def _repeat_lines(schema: DatasetSchema, by: str) -> tuple[np.ndarray, np.ndarray]:
    """(line, place along ``by``) of each Repeat row of a schema grouped by it."""

    domain = _domains(schema)[0]
    axes = _primary(domain)
    others = [domain.codes(axis.axis_id) for axis in axes if axis.name != by]
    line = np.unique(np.column_stack(others), axis=0, return_inverse=True)[1].reshape(-1) if others else np.zeros(domain.size, np.int64)
    return line, np.asarray(domain.codes(next(axis for axis in axes if axis.name == by).axis_id))


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
        return Accumulator(step.row, reducible, spanning=False, transient=True)

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
