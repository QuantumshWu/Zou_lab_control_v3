"""One current Dataset estimate per published variable, evaluated over Runtime-owned inputs."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from zlc_data import owned_snapshot_from_arrays
from zlc_plugin import DatasetOutputDeclaration, LiveDatasetOutput

from .expression import Operand, compact_variable, evaluate, input_members, published
from .groups import Accumulation, restricted


def declared_outputs(rows: Sequence[Mapping[str, object]]) -> tuple[DatasetOutputDeclaration, ...]:
    """One signal per published variable whose name reads, in card order; a
    fit's one per parameter."""

    return tuple(DatasetOutputDeclaration(name, f"derive.{name}", index_by_source=True)
                 for name in published(rows))


class DeriveProcessor:
    """Numerical code owns no input history, placement table or parallel schema.

    Each publication of the bound bundle is evaluated once: the LIVE
    variables over that publication's members, in dependency order, and
    each published variable is a complete current estimate.  ``inputs`` of
    one evaluation are the bundle's members by output name; the anchor is
    the member the operator selected.

    A variable with groups over a Repeat or Point axis, and what reads it, give
    only what just closed (:mod:`.groups`): placed write-once when the source
    is finite, one Latest event per group when it is a monitor.
    """

    def __init__(self, *, variables, primary_output: str):
        self._variables = tuple(variables)
        self.primary_output = str(primary_output)
        self.dataset_output_declarations = declared_outputs(self._variables)
        #: Whether anything here can give less than one result per publication.
        self._accumulating = any(row["every"] or row["statistic"] == "fit" for row in self._variables)
        self._accumulation: Accumulation | None = None
        self._context = None
        self._placed: dict[str, int] = {}
        self._last = None

    def bind_run(self, context) -> None:
        """The run this processor evaluates in: what answers a fit."""

        self._context = context

    def _fit(self, *arguments, **options):
        return self._context.fit(*arguments, **options)

    def _names(self, inputs: Mapping[str, object]) -> tuple[str, ...]:
        # The members the live variables read; a name that is neither a member
        # nor a variable is refused here, by variable and stage.
        return tuple(dict.fromkeys((self.primary_output, *input_members(self._variables, tuple(inputs)))))

    def _output(self, declaration, result, primary, **placement):
        schema = replace(result.schema, value_schema=replace(result.schema.value_schema, name=declaration.name))
        snapshot = owned_snapshot_from_arrays(
            schema, result.values, primary.snapshot.block.revision,
            validity=result.valid, sigma=result.sigma, stream_generation=primary.snapshot.ref.stream_generation,
        )
        return LiveDatasetOutput(declaration, snapshot, event_record=primary.event_record, **placement)

    def evaluate_inputs(self, inputs: Mapping[str, object]):
        primary = inputs.get(self.primary_output)
        if primary is None:
            raise TypeError("derive needs its bound input bundle")
        if self._accumulating:
            return self._increment(inputs, primary)
        results = evaluate(self._variables, {
            name: Operand(value.snapshot.block.schema, value.snapshot.materialize().block.values,
                          value.snapshot.expanded_validity())
            for name, value in ((name, inputs[name]) for name in self._names(inputs))
        })
        outputs = {}
        for declaration in self.dataset_output_declarations:
            # An arbitrary reduction is a current estimate, not an input shot
            # pasted into a guessed finite placement. Runtime alone retains it
            # and builds history only for consumers that request a lease.
            outputs[declaration.name] = self._output(declaration, results[declaration.name], primary)
        return outputs

    def _increment(self, inputs: Mapping[str, object], primary):
        """One delivery with groups in the draft: the commits it makes, none when nothing closed."""

        names = self._names(inputs)
        if self._accumulation is None:
            self._accumulation = Accumulation(self._variables, inputs, names)
        self._last = primary
        events, views = {}, {}
        for name in names:
            value = inputs[name]
            block = value.snapshot.materialize().block
            events[name] = Operand(block.schema, block.values, value.snapshot.expanded_validity(), block.sigma)
            whole = block.schema if value.canonical_schema is None else restricted(
                value.canonical_schema, value.cell_origin,
                (block.schema.repeat_domain.size, block.schema.point_domain.size))
            views[name] = Operand(whole, block.values, events[name].valid, block.sigma)
        plain = {}
        if self._accumulation.plain:
            plain = evaluate(self._accumulation.plain_rows, events, self._fit)
        sources = None
        if primary.canonical_schema is not None:
            sources = tuple(zip(primary.cell_origin, (primary.canonical_schema.repeat_domain,
                                                      primary.canonical_schema.point_domain)))
        ordinal = primary.key.get("ordinal")
        fresh = self._accumulation.step(views, sources, 0 if ordinal is None else int(ordinal), self._fit)
        return self._commits(plain, fresh, primary)

    def finish_inputs(self) -> tuple:
        """The source sealed: the groups still open, their n smaller.  A Stop never comes here."""

        if self._accumulation is None:
            return ()
        return self._commits({}, self._accumulation.step(None, None, None, self._fit), self._last)

    def run_outputs(self) -> Mapping[str, tuple]:
        """What this run declares before its first commit, ``{name: (schema, storage, index_by_source)}``:
        an output on a clock may have nothing to say in that commit.  The rest declare themselves there."""

        if self._accumulation is None:
            return {}
        storage = "finite" if self._accumulation.finite else "latest"
        return {name: (self._named(schema, name), storage, True)
                for name, schema in self._accumulation.outputs().items()}

    @staticmethod
    def _named(schema, name):
        return replace(schema, value_schema=replace(schema.value_schema, name=name))

    def _commits(self, plain: Mapping, fresh: Mapping, primary) -> tuple:
        """Everything for one commit, then each further closed run of a variable
        in a commit of its own: Runtime takes one placement per signal in a bundle."""

        declared = {declaration.name: declaration for declaration in self.dataset_output_declarations}
        finite = self._accumulation.finite
        whole = self._accumulation.outputs()
        commits: list[dict] = []
        if plain:
            commits.append({name: self._output(declared[name], plain[name], primary)
                            for name in declared if name in plain})
        for name, chunks in fresh.items():
            if name not in declared:
                continue
            for k, chunk in enumerate(chunks):
                while len(commits) <= k:
                    commits.append({})
                if finite:
                    placed = self._placed.get(name, 0)
                    self._placed[name] = placed + 1
                    commits[k][name] = self._output(
                        declared[name], chunk.operand, primary, ordinal=placed,
                        canonical_schema=self._named(whole[name], name), cell_origin=chunk.origin)
                else:
                    commits[k][name] = self._output(declared[name], chunk.operand, primary, ordinal=chunk.ordinal)
        return tuple(commit for commit in commits if commit)

    def describe_run(self, inputs: Mapping[str, object]) -> dict[str, object]:
        """Every variable as written -- internal and parked ones too -- and
        which signal fed each input name."""

        return {
            "parameters": {
                "variables": tuple(compact_variable(row) for row in self._variables),
                "source_signal": inputs[self.primary_output].name,
                "inputs": {name: value.name for name, value in inputs.items()},
            },
        }


__all__ = ["DeriveProcessor", "declared_outputs"]
