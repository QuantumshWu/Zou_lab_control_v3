"""One current Dataset estimate per published variable, evaluated over Runtime-owned inputs."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from zlc_data import owned_snapshot_from_arrays
from zlc_plugin import DatasetOutputDeclaration, LiveDatasetOutput

from .expression import Operand, compact_variable, evaluate, input_members, published


def declared_outputs(rows: Sequence[Mapping[str, object]]) -> tuple[DatasetOutputDeclaration, ...]:
    """One signal per published variable whose name reads, in card order."""

    return tuple(DatasetOutputDeclaration(name, f"derive.{name}", index_by_source=True)
                 for name in published(rows))


class DeriveProcessor:
    """Numerical code owns no input history, placement table or parallel schema.

    Each publication of the bound bundle is evaluated once: the LIVE
    variables over that publication's members, in dependency order, and
    each published variable is a complete current estimate.  ``inputs`` of
    one evaluation are the bundle's members by output name; the anchor is
    the member the operator selected.
    """

    def __init__(self, *, variables, primary_output: str):
        self._variables = tuple(variables)
        self.primary_output = str(primary_output)
        self.dataset_output_declarations = declared_outputs(self._variables)

    def evaluate_inputs(self, inputs: Mapping[str, object]):
        primary = inputs.get(self.primary_output)
        if primary is None:
            raise TypeError("derive needs its bound input bundle")
        # The members the live variables read; a name that is neither a member
        # nor a variable is refused here, by variable and stage.
        names = tuple(dict.fromkeys((self.primary_output, *input_members(self._variables, tuple(inputs)))))
        results = evaluate(self._variables, {
            name: Operand(value.snapshot.block.schema, value.snapshot.materialize().block.values,
                          value.snapshot.expanded_validity())
            for name, value in ((name, inputs[name]) for name in names)
        })
        outputs = {}
        for declaration in self.dataset_output_declarations:
            result = results[declaration.name]
            schema = replace(result.schema, value_schema=replace(
                result.schema.value_schema, name=declaration.name,
            ))
            snapshot = owned_snapshot_from_arrays(
                schema, result.values, primary.snapshot.block.revision,
                validity=result.valid, stream_generation=primary.snapshot.ref.stream_generation,
            )
            # An arbitrary reduction is a current estimate, not an input shot
            # pasted into a guessed finite placement. Runtime alone retains it
            # and builds history only for consumers that request a lease.
            outputs[declaration.name] = LiveDatasetOutput(
                declaration, snapshot, event_record=primary.event_record,
            )
        return outputs

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
