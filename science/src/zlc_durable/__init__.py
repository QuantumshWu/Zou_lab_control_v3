"""Durable filesystem primitives: write atomically, land where you meant to.

Saving packages use this facade and the independent native client; no plugin
SDK or application import is required. Publication/durability rules are shared
with the native archive owner, while scientific content encoders stay Python.
It deliberately does NOT carry a canonical encoder or content digests -- an
archive in this project is plain JSON and plain arrays, readable without
importing any of our packages.
"""

from __future__ import annotations

from .readable import readable_json, readable_json_bytes, strict_json_loads, write_readable_json
from .durability import (
    DirectoryDurabilityError,
    atomic_write_bytes,
    atomic_write_file,
    atomic_write_text,
    durable_makedirs,
)
from .workspace import day_folder, day_folder_path, unique_path

# The whole public surface. A caller writes bytes, creates a directory tree,
# reads JSON back strictly, or asks where durable work should land --
# implementation primitives stay owned by their submodules.
__all__ = [
    "DirectoryDurabilityError",
    "atomic_write_bytes",
    "atomic_write_file",
    "atomic_write_text",
    "readable_json",
    "readable_json_bytes",
    "strict_json_loads",
    "write_readable_json",
    "day_folder",
    "day_folder_path",
    "durable_makedirs",
    "unique_path",
]
