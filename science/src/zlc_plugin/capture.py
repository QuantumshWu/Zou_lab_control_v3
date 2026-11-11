"""Capture evidence at batch/end boundaries through the existing native service.

The driver FIFO and its worker have one Python owner. This facade carries only
batch metadata; no DLL, per-record queue RPC, or duplicate queue state exists.
"""
from zlc_plugin.services import call


def _validate(operation, **params):
    try:
        return call(operation, params)
    except ValueError as error:
        raise RuntimeError(str(error)) from error


def validate_batch(ordinals, *, expected_start: int, group_size: int = 1):
    return _validate("capture.batch", expected_start=expected_start, ordinals=list(ordinals), group_size=group_size)


def validate_group(ordinals, *, start: int, size: int) -> None:
    _validate("capture.group", expected_start=start, ordinals=list(ordinals), group_size=size)


def terminal_evidence(terminal, *, expected: int | None = None, stopped: bool = False, group_size: int = 1):
    return _validate("capture.terminal", produced=terminal.produced_count, source_stopped=terminal.source_stopped,
                joined=terminal.joined, expected=expected, stopped=stopped, group_size=group_size)


__all__ = ["validate_batch", "validate_group", "terminal_evidence"]
