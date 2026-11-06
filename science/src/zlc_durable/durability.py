"""Filesystem directory durability primitives shared by storage owners."""

from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO, Callable, Iterable


class DirectoryDurabilityError(RuntimeError):
    """The filesystem could not durably flush a directory entry.

    ``published`` names the artifact that is already complete and visible
    when the flush that would have made its directory entry crash-durable
    is what failed: the file replaced or linked, the directory made.  It
    stays on disk.  A save window that shows the error can then say where
    the work landed instead of "cannot save", and a caller that retries
    knows the name is taken by its own complete work rather than allocating
    a numbered copy beside it.  ``None`` when nothing was published by the
    call that failed.
    """

    def __init__(self, message: str, *, published: Path | None = None) -> None:
        super().__init__(message)
        self.published = published


def _context():
    from zlc_plugin.services import current_context
    return current_context()


def _native(operation: str, **params):
    from zlc_plugin.services import call
    answer = call("durable." + operation, params)
    if "error" not in answer:
        return answer["ok"]
    error = answer["error"]
    published = None if error.get("published") is None else Path(error["published"])
    if error["type"] == "DirectoryDurabilityError":
        raise DirectoryDurabilityError(error["message"], published=published)
    import builtins
    kind = getattr(builtins, error["type"], RuntimeError)
    failure = kind(error["message"])
    failure.published = published
    raise failure


def flush_directory(directory: str | os.PathLike[str]) -> None:
    """Acknowledge directory durability through the shared native implementation."""
    context = _context()
    if context is not None:
        context.artifacts.flush_directory(directory)
    else:
        _native("flush", path=str(directory))


def _flush_published(directory: Path, published: Path) -> None:
    try:
        flush_directory(directory)
    except DirectoryDurabilityError as error:
        raise DirectoryDurabilityError(
            f"{published} is published and visible, but its directory entry could not be made crash-durable: {error}",
            published=published,
        ) from error


def atomic_write_file(target: str | os.PathLike[str], writer: Callable[[BinaryIO], None]) -> Path:
    """Python writes scientific content; native code stages, syncs and publishes."""
    if not callable(writer):
        raise TypeError("writer must be callable")
    context = _context()
    if context is not None:
        return context.artifacts.write_file(target, writer)
    staged = _native("stage", path=str(target))
    try:
        with Path(staged["path"]).open("wb") as stream:
            writer(stream)
    except BaseException as error:
        try:
            _native("abort", handle=staged["handle"])
        except BaseException as cleanup:
            error.add_note(f"aborting staged artifact also failed: {cleanup}")
        raise
    return Path(_native("finalize", handle=staged["handle"])["path"])


def _atomic_write_unique_path(candidates: Iterable[Path], writer: Callable[[Path], object], *,
                              temporary_prefix: str, temporary_suffix: str) -> Path:
    if not callable(writer):
        raise TypeError("writer must be callable")
    choices = iter(candidates)
    try:
        first = next(choices)
    except StopIteration as error:
        raise ValueError("candidates must not be empty") from error
    context = _context()
    if context is not None:
        import itertools
        return context.artifacts.write_unique_path(itertools.chain((first,), choices), writer,
            temporary_prefix=temporary_prefix, temporary_suffix=temporary_suffix)
    staged = _native("stage", path=str(first), temporary_prefix=temporary_prefix, temporary_suffix=temporary_suffix)
    try:
        writer(Path(staged["path"]))
    except BaseException as error:
        try:
            _native("abort", handle=staged["handle"])
        except BaseException as cleanup:
            error.add_note(f"aborting staged artifact also failed: {cleanup}")
        raise
    destination = first
    while True:
        try:
            return Path(_native("finalize", handle=staged["handle"], candidate=str(destination))["path"])
        except FileExistsError:
            try:
                destination = next(choices)
            except StopIteration:
                _native("abort", handle=staged["handle"])
                raise


def atomic_write_bytes(target: str | os.PathLike[str], payload: bytes | bytearray | memoryview) -> Path:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload must be bytes-like")
    return atomic_write_file(target, lambda stream: stream.write(payload))


def atomic_write_text(target: str | os.PathLike[str], text: str, *, encoding: str = "utf-8") -> Path:
    if not isinstance(text, str):
        raise TypeError("text must be str")
    return atomic_write_bytes(target, text.encode(encoding))


def durable_mkdir(directory: str | os.PathLike[str]) -> Path:
    context = _context()
    return context.artifacts.mkdir(directory) if context is not None else Path(_native("mkdir", path=str(directory))["path"])


def durable_makedirs(directory: str | os.PathLike[str]) -> Path:
    context = _context()
    return context.artifacts.makedirs(directory) if context is not None else Path(_native("makedirs", path=str(directory))["path"])


def _claim_directory(directory: Path) -> Path:
    context = _context()
    return context.artifacts.claim_directory(directory) if context is not None else Path(_native("claim_directory", path=str(directory))["path"])
