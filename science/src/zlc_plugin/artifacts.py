"""Run artifact publication: Python encodes science, the core publishes and registers."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class Artifacts:
    def __init__(self, context):
        self._context = context

    def _call(self, action: str, **params):
        answer = self._context._request("run.artifacts", {"action": action, **params})
        if "error" in answer:
            error = answer["error"]
            published = None if error.get("published") is None else Path(error["published"])
            if error["type"] == "DirectoryDurabilityError":
                from zlc_durable import DirectoryDurabilityError
                raise DirectoryDurabilityError(error["message"], published=published)
            kind = {"ValueError": ValueError, "TypeError": TypeError, "FileExistsError": FileExistsError,
                    "FileNotFoundError": FileNotFoundError, "PermissionError": PermissionError,
                    "OSError": OSError}.get(error["type"], RuntimeError)
            failure = kind(error["message"])
            failure.published = published
            raise failure
        return answer["ok"]

    def flush_directory(self, path) -> None:
        self._call("flush", path=str(path))

    def mkdir(self, path) -> Path:
        return Path(self._call("mkdir", path=str(path))["path"])

    def makedirs(self, path) -> Path:
        return Path(self._call("makedirs", path=str(path))["path"])

    def claim_directory(self, path) -> Path:
        return Path(self._call("claim_directory", path=str(path))["path"])

    def write_unique_path(self, candidates, writer, *, temporary_prefix, temporary_suffix) -> Path:
        choices = iter(candidates)
        first = next(choices)
        staged = self._call("stage", path=str(first), temporary_prefix=temporary_prefix, temporary_suffix=temporary_suffix)
        try:
            writer(Path(staged["path"]))
        except BaseException as error:
            try:
                self._call("abort", token=staged["token"])
            except BaseException as cleanup:
                error.add_note(f"aborting staged artifact also failed: {cleanup}")
            raise
        candidate = first
        while True:
            try:
                return Path(self._call("finalize", token=staged["token"], candidate=str(candidate))["path"])
            except FileExistsError:
                try:
                    candidate = next(choices)
                except StopIteration:
                    self._call("abort", token=staged["token"])
                    raise

    @staticmethod
    def _options(path, name=None, role="process", contract_id="", unique=False, lineage=None):
        result = {"path": str(path), "role": role, "contract_id": contract_id, "unique": bool(unique)}
        if name is not None:
            result["name"] = str(name)
        if lineage is not None:
            result["lineage"] = lineage
        return result

    def register(self, name: str, path, *, role: str, contract_id: str = "", lineage=None):
        return self._call("register", **self._options(path, name, role, contract_id, lineage=lineage))

    def write_text(self, path, text: str, **options) -> Path:
        if not isinstance(text, str):
            raise TypeError("artifact text must be str")
        return Path(self._call("write_text", text=text, **self._options(path, **options))["path"])

    def write_json(self, path, value: Any, *, indent: int = 2, **options) -> Path:
        from zlc_durable import readable_json
        return self.write_text(path, readable_json(value, indent=indent), **options)

    def write_bytes(self, path, value: bytes | bytearray | memoryview, **options) -> Path:
        import numpy as np
        return Path(self._call("write_bytes", bytes=np.frombuffer(value, dtype=np.uint8), **self._options(path, **options))["path"])

    @contextmanager
    def stage(self, path, **options):
        staged = self._call("stage", **self._options(path, **options))
        entry = _Stage(Path(staged["path"]), Path(staged["target"]))
        try:
            yield entry
        except BaseException as error:
            try:
                self._call("abort", token=staged["token"])
            except BaseException as cleanup:
                error.add_note(f"aborting staged artifact also failed: {cleanup}")
            raise
        else:
            # The core consumes the stage even if publication's durability
            # acknowledgement fails; never delete a published result here.
            answer = self._call("finalize", token=staged["token"])
            entry.result = Path(answer["path"])

    def write_file(self, path, writer, **options) -> Path:
        if not callable(writer):
            raise TypeError("artifact writer must be callable")
        with self.stage(path, **options) as stage:
            with stage.path.open("wb") as stream:
                writer(stream)
        return stage.result


class _Stage:
    def __init__(self, path: Path, target: Path):
        self.path, self.target, self.result = path, target, target


__all__ = ["Artifacts"]
