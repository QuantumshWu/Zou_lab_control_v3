"""Frames and values on the plugin control channel (docs/PLUGIN_PROTOCOL.md §4, §6).

A frame is a little-endian u32 byte length followed by one UTF-8 JSON object.
Values are JSON plus a few tagged forms, so typed v2 contracts (dataclasses,
enums, tuples, arrays) survive a call between two plugins unchanged.
Arrays travel as shared-memory references (``$array``) or, when small, inline.
"""

from __future__ import annotations

import base64
import builtins
import dataclasses
import enum
import importlib
import json
import math
import pathlib
import struct
import threading
import traceback
import types
from typing import Any, BinaryIO, Callable

import numpy as np

PROTOCOL = (2, 0)
MAX_FRAME = 16 << 20
#: Arrays smaller than this are sent inside the frame instead of a block.
INLINE_BYTES = 16 << 10
#: What JSON decodes a scalar to: such a value decodes to itself.
_SCALARS = frozenset({int, float, str, bool, type(None)})


class Channel:
    """One framed duplex channel; sends are serialized, receives are one reader's."""

    def __init__(self, source: BinaryIO, sink: BinaryIO) -> None:
        self._source = source
        self._sink = sink
        self._lock = threading.Lock()

    def send(self, message: dict[str, Any]) -> None:
        payload = json.dumps(message, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(payload) > MAX_FRAME:
            raise ValueError(f"frame of {len(payload)} bytes exceeds {MAX_FRAME}; send bulk data as arrays")
        with self._lock:
            self._sink.write(struct.pack("<I", len(payload)) + payload)
            self._sink.flush()

    def receive(self) -> dict[str, Any] | None:
        """The next message, or None at end of stream."""

        header = self._read(4)
        if header is None:
            return None
        (length,) = struct.unpack("<I", header)
        if length > MAX_FRAME:
            raise ValueError(f"incoming frame of {length} bytes exceeds {MAX_FRAME}")
        payload = self._read(length)
        if payload is None:
            raise EOFError("channel closed inside a frame")
        message = json.loads(payload)
        if not isinstance(message, dict):
            raise ValueError("a frame must hold one JSON object")
        return message

    def _read(self, count: int) -> bytes | None:
        data = self._source.read(count)
        if not data:
            return None
        while len(data) < count:
            more = self._source.read(count - len(data))
            if not more:
                raise EOFError("channel closed inside a frame")
            data += more
        return data


def _qualified(kind: type) -> str:
    return f"{kind.__module__}:{kind.__qualname__}"


def _resolve(path: str) -> Any:
    module_name, _, qualname = path.partition(":")
    target: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        target = getattr(target, part)
    return target


class Encoder:
    """Turn one value into JSON, lending each large array once.

    ``lend(array)`` returns the ``$array`` reference for an array that may be
    sent by reference, copying it into a block first when it is not one.
    ``blocks`` collects every block lent by this message, with multiplicity.
    """

    def __init__(self, lend: Callable[[np.ndarray], dict[str, Any]] | None, *, plain: bool = False) -> None:
        self._lend = lend
        #: For the core, which reads JSON: dataclasses become objects, tuples
        #: lists, dtypes and enums their values. Arrays stay references.
        self._plain = plain
        self.blocks: list[str] = []

    def __call__(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, str)):
            return value
        if isinstance(value, enum.Enum):
            if self._plain:
                return self(value.value)
            return {"$enum": _qualified(type(value)), "value": self(value.value)}
        if isinstance(value, int):
            return value if -(1 << 63) <= value < (1 << 64) else {"$int": str(value)}
        if isinstance(value, float):
            return value if math.isfinite(value) else {"$float": repr(value)}
        if isinstance(value, np.ndarray):
            return self.array(value)
        if isinstance(value, np.generic):
            return self(value.item())
        if isinstance(value, np.dtype):
            return value.str if self._plain else {"$dtype": value.str}
        if isinstance(value, (list,)):
            return [self(item) for item in value]
        if self._plain and isinstance(value, tuple) and not hasattr(value, "_asdict"):
            return [self(item) for item in value]
        if isinstance(value, tuple) and not hasattr(value, "_fields"):
            return {"$tuple": [self(item) for item in value]}
        if self._plain and dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {item.name: self(getattr(value, item.name)) for item in dataclasses.fields(value)}
        if self._plain and isinstance(value, pathlib.PurePath):
            return str(value)
        if isinstance(value, types.MappingProxyType):
            # A read-only view of a mapping (v2 declarations freeze theirs) travels as the mapping.
            value = dict(value)
        if isinstance(value, dict):
            if all(isinstance(key, str) and not key.startswith("$") for key in value):
                return {key: self(item) for key, item in value.items()}
            return {"$dict": [[self(key), self(item)] for key, item in value.items()]}
        if isinstance(value, (bytes, bytearray, memoryview)):
            return {"$bytes": base64.b64encode(bytes(value)).decode("ascii")}
        if isinstance(value, pathlib.PurePath):
            return {"$path": str(value)}
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            fields = {item.name: self(getattr(value, item.name))
                      for item in dataclasses.fields(value) if item.init}
            return {"$dataclass": _qualified(type(value)), "fields": fields}
        if hasattr(value, "_asdict") and isinstance(value, tuple):
            return {"$namedtuple": _qualified(type(value)), "fields": self(dict(value._asdict()))}
        if isinstance(value, (set, frozenset)):
            return {"$set": [self(item) for item in value], "frozen": isinstance(value, frozenset)}
        raise TypeError(f"{type(value).__qualname__} cannot be sent to another process")

    def array(self, value: np.ndarray) -> dict[str, Any]:
        array = np.asarray(value)
        if array.dtype.hasobject or array.dtype.fields is not None:
            raise TypeError("only bool and numeric arrays can be sent")
        little = array.dtype.newbyteorder("<") if array.dtype.byteorder == ">" else array.dtype
        if array.nbytes < INLINE_BYTES or self._lend is None:
            contiguous = np.ascontiguousarray(array, dtype=little)
            return {"$inline": {"b64": base64.b64encode(contiguous.tobytes()).decode("ascii"),
                                "dtype": contiguous.dtype.str, "shape": list(array.shape)}}
        reference = self._lend(array.astype(little, copy=False))
        self.blocks.append(reference["block"])
        return {"$array": reference}


class Decoder:
    """Rebuild values; ``attach`` maps one ``$array`` reference to a read-only array."""

    def __init__(self, attach: Callable[[dict[str, Any]], np.ndarray] | None) -> None:
        self._attach = attach

    def __call__(self, value: Any) -> Any:
        if isinstance(value, list):
            # A scalar decodes to itself: only containers recurse, so a schema's
            # coordinate list costs one pass, not one call per number.
            if _SCALARS.issuperset(map(type, value)):
                return list(value)
            return [self(item) if isinstance(item, (list, dict)) else item for item in value]
        if not isinstance(value, dict):
            return value
        if value:
            tag = next(iter(value))
            if tag in _TAGS:
                return self._tagged(tag, value)
        return {key: self(item) for key, item in value.items()}

    def _tagged(self, tag: str, value: dict[str, Any]) -> Any:
        if tag == "$tuple":
            return tuple(self(item) for item in value["$tuple"])
        if tag == "$dict":
            return {self(key): self(item) for key, item in value["$dict"]}
        if tag == "$array":
            if self._attach is None:
                raise ValueError("this channel cannot receive shared arrays")
            return self._attach(value["$array"])
        if tag == "$inline":
            spec = value["$inline"]
            data = base64.b64decode(spec["b64"])
            return np.frombuffer(data, dtype=np.dtype(spec["dtype"])).reshape(spec["shape"])
        if tag == "$float":
            return float(value["$float"])
        if tag == "$int":
            return int(value["$int"])
        if tag == "$bytes":
            return base64.b64decode(value["$bytes"])
        if tag == "$path":
            return pathlib.Path(value["$path"])
        if tag == "$dtype":
            return np.dtype(value["$dtype"])
        if tag == "$enum":
            return _resolve(value["$enum"])(self(value["value"]))
        if tag == "$set":
            items = [self(item) for item in value["$set"]]
            return frozenset(items) if value.get("frozen") else set(items)
        if tag == "$dataclass":
            kind = _resolve(value["$dataclass"])
            return kind(**{key: self(item) for key, item in value["fields"].items()})
        if tag == "$namedtuple":
            kind = _resolve(value["$namedtuple"])
            return kind(**self(value["fields"]))
        raise AssertionError(tag)


#: An encoded mapping whose first key starts with "$" is always wrapped in
#: "$dict", so only these first keys are tags.
_TAGS = frozenset({"$tuple", "$dict", "$array", "$inline", "$float", "$int", "$bytes", "$path", "$dtype",
                   "$enum", "$set", "$dataclass", "$namedtuple"})


# ------------------------------------------------------------------ errors

#: Error kinds (PLUGIN_PROTOCOL.md §7); a plugin raises PluginError(kind, ...)
#: to state one explicitly.
KINDS = frozenset({"refused", "invalid", "failed", "timeout", "lost", "cancelled",
                   "stale_run", "unknown_op", "not_claimed"})


class PluginError(RuntimeError):
    """A failure with an explicit protocol kind, possibly from another process."""

    def __init__(self, kind: str, message: str, *, remote_type: str = "", remote_traceback: str = "") -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown plugin error kind {kind!r}")
        super().__init__(message)
        self.kind = kind
        self.remote_type = remote_type
        self.remote_traceback = remote_traceback


class Refused(ValueError):
    """Rejected before anything was written: the device is as it was."""


def error_payload(error: BaseException, *, kind: str | None = None) -> dict[str, Any]:
    if kind is None:
        if isinstance(error, PluginError):
            kind = error.kind
        elif isinstance(error, Refused) or type(error).__name__ == "TuneRefused":
            kind = "refused"
        else:
            kind = "failed"
    return {"kind": kind, "type": _qualified(type(error)), "message": str(error),
            "traceback": "".join(traceback.format_exception(error))}


_BUILTIN = {name: getattr(builtins, name) for name in ("TimeoutError", "ValueError", "TypeError", "RuntimeError", "KeyError", "LookupError",
                         "IndexError", "NotImplementedError", "PermissionError", "FileNotFoundError", "OSError",
                         "MemoryError", "ArithmeticError", "ZeroDivisionError", "AttributeError")}


def raise_remote(error: dict[str, Any]) -> None:
    """Raise a received error as its original type where that type is reachable."""

    kind = str(error.get("kind", "failed"))
    message = str(error.get("message", ""))
    remote_type = str(error.get("type", ""))
    remote_traceback = str(error.get("traceback", ""))
    exception: BaseException | None = None
    if kind not in {"failed", "refused", "invalid"}:
        exception = PluginError(kind if kind in KINDS else "failed", message,
                                remote_type=remote_type, remote_traceback=remote_traceback)
    elif remote_type.startswith("builtins:"):
        kind_type = _BUILTIN.get(remote_type.partition(":")[2])
        if kind_type is not None:
            exception = kind_type(message)
    elif remote_type:
        try:
            kind_type = _resolve(remote_type)
            if isinstance(kind_type, type) and issubclass(kind_type, Exception):
                exception = kind_type(message)
        except Exception:
            exception = None
    if exception is None:
        exception = PluginError(kind if kind in KINDS else "failed", message,
                                remote_type=remote_type, remote_traceback=remote_traceback)
    if remote_traceback:
        exception.add_note(f"Remote traceback:\n{remote_traceback}")
    raise exception


def plain(value: Any) -> Any:
    """Static descriptor JSON: tuples become lists, nothing else is tagged."""

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, enum.Enum):
        return plain(value.value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("descriptor numbers must be finite")
        return value
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    raise TypeError(f"{type(value).__qualname__} is not descriptor data")


__all__ = ["Channel", "Decoder", "Encoder", "INLINE_BYTES", "KINDS", "MAX_FRAME", "PROTOCOL", "PluginError",
           "Refused", "error_payload", "plain", "raise_remote"]
