"""Process-local native capture policy; payload objects and SDK handles stay in Python."""
from __future__ import annotations

import ctypes
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import threading


@lru_cache(maxsize=1)
def _library():
    named = os.environ.get("ZLC_CAPTURE_LIBRARY")
    root = Path(__file__).resolve().parents[3]
    filename = "zlc_capture.dll" if os.name == "nt" else "libzlc_capture.dylib" if sys.platform == "darwin" else "libzlc_capture.so"
    candidates = [Path(named)] if named else [root / "bin" / "engine" / filename,
        *(root / "target" / target / mode / filename
          for target in ("x86_64-pc-windows-msvc", "") for mode in ("release", "debug"))]
    path = next((path for path in candidates if path.is_file()), None)
    if path is None:
        raise RuntimeError("the native capture runtime is unavailable; build zlc-capture or set ZLC_CAPTURE_LIBRARY")
    library = ctypes.CDLL(str(path))
    library.zlc_capture_new.argtypes = []
    library.zlc_capture_new.restype = ctypes.c_void_p
    library.zlc_capture_call.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    library.zlc_capture_call.restype = ctypes.c_void_p
    library.zlc_capture_free_result.argtypes = [ctypes.c_void_p]
    library.zlc_capture_free_result.restype = None
    library.zlc_capture_free.argtypes = [ctypes.c_void_p]
    library.zlc_capture_free.restype = None
    return library


class CaptureLedger:
    """One owned native state machine. Calls never cross the plugin channel."""
    def __init__(self, what: str = "capture") -> None:
        self._library = _library()
        self._handle = self._library.zlc_capture_new()
        self._what = what
        self._lock = threading.Lock()

    def call(self, operation: str, **params):
        message = json.dumps({"op": operation, **params}, allow_nan=False).encode("utf-8")
        with self._lock:
            if not self._handle:
                raise RuntimeError("the capture ledger was released")
            pointer = self._library.zlc_capture_call(self._handle, message)
            if not pointer:
                raise RuntimeError("the native capture runtime returned no result")
            try:
                answer = json.loads(ctypes.string_at(pointer))
            finally:
                self._library.zlc_capture_free_result(pointer)
        if "error" in answer:
            error = answer["error"]
            kind = {"invalid": ValueError, "timeout": TimeoutError}.get(error["kind"], RuntimeError)
            raise kind(f"{self._what} {error['message']}")
        return answer["ok"]

    def state(self):
        return self.call("state")

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle:
            self._handle = None
            self._library.zlc_capture_free(handle)


@lru_cache(maxsize=1)
def _validator():
    return CaptureLedger()


def validate_group(ordinals, *, start: int, size: int) -> None:
    _validator().call("group", ordinals=list(ordinals), start=start, size=size)


def terminal_evidence(terminal, *, expected: int | None = None, stopped: bool = False, group_size: int = 1):
    return _validator().call("terminal", produced=terminal.produced_count, source_stopped=terminal.source_stopped,
                             joined=terminal.joined, expected=expected, stopped=stopped, group_size=group_size)


__all__ = ["CaptureLedger", "validate_group", "terminal_evidence"]
