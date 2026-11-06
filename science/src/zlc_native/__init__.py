"""Canonical native rules, in the live host or the same offline helper.

No Python algorithm fallback: an unavailable native runtime is an installation
error. Offline calls share one helper; live arrays use the existing block arena.
"""
from __future__ import annotations

import atexit
from contextlib import contextmanager
import json
import os
from pathlib import Path
import struct
import subprocess
import threading
from typing import Any, Mapping

from .wire import Decoder, Encoder, PluginError

_scope = threading.local()
_helper = None
_transport = None
_lock = threading.RLock()
_MAX_FRAME = 256 << 20


@contextmanager
def scope(context):
    previous = getattr(_scope, "context", None)
    _scope.context = context
    try:
        yield
    finally:
        _scope.context = previous


def current_context():
    """The calling plugin run, or None for catalogue/offline work."""
    return getattr(_scope, "context", None)


def _tree(value):
    # Native APIs use language-independent sequences, not Python tuple tags.
    if isinstance(value, (tuple, list)):
        return [_tree(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _tree(item) for key, item in value.items()}
    return value


def executable() -> Path:
    explicit = os.environ.get("ZLC_NATIVE_EXECUTABLE")
    root = Path(__file__).resolve().parents[3]
    candidates = ([Path(explicit)] if explicit else [
        root / "bin" / "engine" / "zlc-native.exe",
        root / "target" / "x86_64-pc-windows-msvc" / "release" / "zlc-native.exe",
        root / "target" / "release" / "zlc-native.exe",
        root / "target" / "debug" / "zlc-native.exe",
    ])
    for path in candidates:
        if path.is_file():
            return path
    raise RuntimeError("The native plugin runtime is not installed; install the current engine or build zlc-native")


def _read(stream, size):
    parts = []
    while size:
        piece = stream.read(size)
        if not piece:
            raise RuntimeError("the native helper ended during a response")
        parts.append(piece)
        size -= len(piece)
    return b"".join(parts)


def _close():
    global _helper
    with _lock:
        process, _helper = _helper, None
        if process is None:
            return
        try:
            process.stdin.close()
            process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
            process.wait()
        finally:
            process.stdout.close()


atexit.register(_close)


def _offline(operation: str, params: Mapping[str, Any], scope_id: str = ""):
    global _helper
    with _lock:
        if _helper is None or _helper.poll() is not None:
            _helper = subprocess.Popen([str(executable())], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        encoded = Encoder(None)({"operation": operation, "params": params, "scope": scope_id})
        payload = json.dumps(encoded, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(payload) > _MAX_FRAME:
            raise ValueError("native request is too large; process arrays in batches")
        try:
            _helper.stdin.write(struct.pack("<I", len(payload)) + payload)
            _helper.stdin.flush()
            length, = struct.unpack("<I", _read(_helper.stdout, 4))
            if length > _MAX_FRAME:
                raise RuntimeError("native response exceeds the protocol limit")
            reply = json.loads(_read(_helper.stdout, length))
        except (OSError, RuntimeError):
            _close()
            raise
        if "error" in reply:
            raise ValueError(reply["error"]["message"])
        return Decoder(None)(reply["ok"])


def call(operation: str, params: Mapping[str, Any] | None = None, **kwargs: Any) -> Any:
    """One coarse native operation; mutable kernel handles belong to this run."""
    arguments = _tree({**dict(params or {}), **kwargs})
    context = getattr(_scope, "context", None)
    if operation.startswith("durable."):
        if context is not None:
            raise RuntimeError("run files must be published through context.artifacts")
        # Non-run plugin caches and offline tools use a local native helper;
        # they never acquire the application's run-file publication authority.
        return _offline(operation, arguments)
    if _transport is None:
        return _offline(operation, arguments, "" if context is None else str(context.generation))
    try:
        return _transport(str(operation), arguments, context)
    except PluginError as error:
        if error.kind in ("invalid", "refused"):
            raise ValueError(str(error)) from None
        raise


def install_transport(transport):
    """Install the host's transport without coupling data packages to the SDK."""
    global _transport
    _transport = transport


__all__ = ["call", "scope", "current_context", "executable", "install_transport"]
