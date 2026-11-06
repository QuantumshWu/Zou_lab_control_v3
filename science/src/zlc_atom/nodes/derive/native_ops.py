"""Bulk scientific operations; both live and offline callers use the native owner."""
from __future__ import annotations

import functools
import numpy as np
from zlc_plugin.native import call
from zlc_data.codec import dataset_schema_to_tree, dataset_schema_from_tree


def plans(steps):
    return [
        {"kind": "mapped", "axis": step[1], "codes": np.asarray(step[2], dtype="<i8"), "size": step[3]}
        if step[0] == "mapped" else
        {"kind": "dense", "axes": list(step[1]), "expand": not step[3]}
        for step in steps
    ]


@functools.lru_cache(maxsize=128)
def alignment(target, other):
    answer = call("scientific.alignment", target=dataset_schema_to_tree(target), other=dataset_schema_to_tree(other))["steps"]
    return None if answer is None else tuple(answer)


def aligned(planes, steps, shape):
    if steps is None:
        return planes
    return call("scientific.align", planes=list(planes), steps=steps, shape=list(shape))["planes"]


def selected(planes, take):
    steps, expand = take
    return call("scientific.select", planes=list(planes),
                steps=[{"axis": axis, "indexes": np.asarray(indexes, dtype="<i8"), "drop": drop}
                       for axis, indexes, drop in steps], expand=expand)["planes"]


def statistic(item, steps, operation, errors):
    return call("scientific.reduce", values=item.values, valid=item.valid, sigma=item.sigma,
                plans=plans(steps), statistic=operation, errors=errors)


def binary(op, left, right, valid, shape, dtype, compute_dtype):
    return call("scientific.binary", op=op, left=left, right=right, valid=valid,
                shape=list(shape), dtype=np.dtype(dtype).str, compute_dtype=np.dtype(compute_dtype).str)


@functools.lru_cache(maxsize=256)
def axis(schema, name, what, verb):
    return call("scientific.axis", schema=dataset_schema_to_tree(schema), name=name, what=what, verb=verb)


@functools.lru_cache(maxsize=128)
def pick_target(schema, name, what):
    return call("scientific.pick_target", schema=dataset_schema_to_tree(schema), name=name, what=what)


def pick_plan(schema, domain, axes, indexes, scalar, domain_pick):
    from zlc_data.selection import EmptySelection
    try:
        answer = call("scientific.pick_plan", schema=dataset_schema_to_tree(schema), domain=domain,
                      axes=[axis.axis_id.value for axis in axes], indexes=np.asarray(indexes, dtype="<i8"),
                      scalar=scalar, domain_pick=domain_pick)
    except ValueError as error:
        if str(error).startswith("EmptySelection:"):
            raise EmptySelection(str(error)) from None
        raise
    take = answer["take"]
    return dataset_schema_from_tree(answer["schema"]), (
        tuple((step["axis"], step["indexes"], step["drop"]) for step in take["steps"]), take["expand"])


@functools.lru_cache(maxsize=128)
def reduction_plan(schema, operation, names, by, every):
    answer = call("scientific.reduce_plan", schema=dataset_schema_to_tree(schema), statistic=operation,
                  names=names, by=by, every=every)
    steps = tuple(("mapped", step["axis"], step["codes"], step["size"]) if step["kind"] == "mapped" else
                  ("dense", tuple(step["axes"]), tuple(step["shape"]), tuple(step["kept_positions"]))
                  for step in answer["plans"])
    return dataset_schema_from_tree(answer["schema"]), steps
