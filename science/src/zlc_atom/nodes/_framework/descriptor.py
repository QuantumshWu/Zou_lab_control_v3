"""Node vocabulary the plugins share beside ``zlc_plugin.NodeType``.

``NodeType``, ``DatasetInput``, ``DeviceRequirement`` and the output
declaration are the SDK's. What stays here is what only these plugins mean:
saved-artifact inputs and their codecs, workspace resources, previews and
plot-selection mappings. A run hands a node the raw value it was given (a
path, or frame-claim rows); the node turns it into its typed value with the
spec's ``resolve_value``, in its own process, because the decoder is its own.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

from zlc_plugin import DatasetOutputDeclaration


@dataclass(frozen=True)
class ResolvedArtifact:
    """One exact file and the typed value decoded from those exact bytes.

    A frame-claims row whose file did not decode is answered with the error
    decoding raised as its value: a row that reads no frame is kept whatever
    its file, and still says what is wrong with it.
    """

    path: Path
    contract_id: str
    value: object


@dataclass(frozen=True)
class ArtifactCodec:
    """The one file contract used to choose and validate a saved artifact."""

    contract_id: str
    file_filter: str
    suffixes: tuple[str, ...]
    decode: Callable[[Path], object]
    #: One line saying what a decoded file is, for whoever picks among
    #: several of them ("20 ms · 35 sites").
    describe: Callable[[object], str] | None = None

    def __post_init__(self) -> None:
        contract_id = str(self.contract_id).strip()
        file_filter = str(self.file_filter).strip()
        suffixes = tuple(str(value).strip().lower() for value in self.suffixes)
        if not contract_id or not file_filter or not suffixes:
            raise ValueError(
                "artifact codec requires contract_id, file_filter, and suffixes"
            )
        if any(not value.startswith(".") for value in suffixes):
            raise ValueError("artifact codec suffixes must begin with '.'")
        if len(set(suffixes)) != len(suffixes):
            raise ValueError("artifact codec suffixes must be unique")
        if not callable(self.decode):
            raise TypeError("artifact codec decode must be callable")
        object.__setattr__(self, "contract_id", contract_id)
        object.__setattr__(self, "file_filter", file_filter)
        object.__setattr__(self, "suffixes", suffixes)

    def resolve(self, path: str | Path) -> ResolvedArtifact:
        source = Path(path).expanduser().resolve()
        if source.suffix.lower() not in self.suffixes:
            raise ValueError(
                f"{source.name!r} must use one of {self.suffixes!r}"
            )
        if not source.is_file():
            # The sentence a refusal ends with; the path is already beside it.
            raise FileNotFoundError("file not found")
        return ResolvedArtifact(source, self.contract_id, self.decode(source))


@dataclass(frozen=True)
class ArtifactInputSpec:
    """One explicit saved-artifact path consumed by a run."""

    name: str
    label: str
    codec: ArtifactCodec
    required: bool = True
    argument_name: str = ""
    #: The run takes SEVERAL of these, each claiming frames of its Dataset
    #: input's cycle (:mod:`.frames`).  The draft value is then a list of
    #: ``{"path": ..., "frames": [...]}`` entries
    #: (:func:`frame_claim_entries`) and the build receives a tuple of
    #: :class:`~.frames.FrameClaim` whose values are the resolved artifacts
    #: (None for a row with no file, which only a row reading no frame may be).
    frame_claims: bool = False

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        label = str(self.label).strip()
        if not name or not label:
            raise ValueError("artifact input requires name and label")
        if not isinstance(self.codec, ArtifactCodec):
            raise TypeError("artifact input codec must be ArtifactCodec")
        argument_name = str(self.argument_name).strip() or str(self.name).strip()
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "argument_name", argument_name)
        object.__setattr__(self, "frame_claims", bool(self.frame_claims))

    @property
    def contract_id(self) -> str:
        return self.codec.contract_id

    def resolve_value(self, raw: object) -> object:
        """What the build receives for the raw run value: a resolved file, or
        frame claims whose rows carry their resolved files (an undecodable
        file of a row reading no frame is kept as the error it raised).
        None for an optional input left empty."""

        from .frames import FrameClaim

        if not self.frame_claims:
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                if self.required:
                    raise ValueError(f"{self.label} needs a file")
                return None
            return self.codec.resolve(str(raw))
        claims = []
        for entry in frame_claim_entries(raw):
            path = entry["path"]
            if not path:
                value: object = None
            elif entry["frames"]:
                value = self.codec.resolve(path)
            else:
                try:
                    value = self.codec.resolve(path)
                except Exception as error:  # noqa: BLE001 -- a parked row keeps what is wrong with it
                    value = ResolvedArtifact(Path(path), self.codec.contract_id, error)
            claims.append(FrameClaim(value, entry["frames"]))
        return tuple(claims)


def frame_claim_entries(raw: object) -> tuple[dict[str, object], ...]:
    """A frame-claims draft value in its one shape: ``({"path", "frames"}, ...)``.

    Paths are kept as written (an empty one is a row still to be given its
    file); frames are put in their canonical order.  Whether the claims fit
    together, and fit a cycle, is :mod:`.frames`' question, not this one.
    """

    from .frames import canonical_frames

    if isinstance(raw, (str, bytes, Mapping)) or not isinstance(raw, Sequence):
        raise ValueError("frame claims must be a list of {path, frames} entries")
    entries = []
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != {"path", "frames"}:
            raise ValueError("each frame claim needs exactly path and frames")
        if not isinstance(item["path"], str):
            raise ValueError("a frame claim's path must be text")
        entries.append({"path": item["path"].strip(), "frames": canonical_frames(item["frames"])})
    return tuple(entries)


@dataclass(frozen=True)
class ResolvedWorkspaceResource:
    """One selected workspace file and its descriptor-decoded typed value."""

    path: Path
    contract_id: str
    value: object


@dataclass(frozen=True)
class WorkspaceResourceSpec:
    """One plain file chosen from a descriptor-owned workspace collection.

    ``decode`` reads the file in the plugin; None means the core resolves it
    and the build receives what the core answers (a pulse document resolves
    to its description, :mod:`.pulse`).
    """

    field_name: str
    contract_id: str
    directory: str
    suffixes: tuple[str, ...]
    decode: Callable[[Path], object] | None
    argument_name: str = ""

    def __post_init__(self) -> None:
        field_name = str(self.field_name).strip()
        contract_id = str(self.contract_id).strip()
        directory = str(self.directory).strip()
        suffixes = tuple(str(value).strip().lower() for value in self.suffixes)
        if (
            not field_name
            or not contract_id
            or not directory
            or Path(directory).name != directory
            or not suffixes
        ):
            raise ValueError(
                "workspace resource requires field, contract, plain directory, and suffixes"
            )
        if any(not value.startswith(".") for value in suffixes):
            raise ValueError("workspace resource suffixes must begin with '.'")
        if self.decode is not None and not callable(self.decode):
            raise TypeError("workspace resource decode must be callable or None")
        object.__setattr__(self, "field_name", field_name)
        object.__setattr__(self, "contract_id", contract_id)
        object.__setattr__(self, "directory", directory)
        object.__setattr__(self, "suffixes", suffixes)
        object.__setattr__(self, "argument_name", str(self.argument_name).strip() or field_name)

    def resolve(self, path: str | Path) -> ResolvedWorkspaceResource:
        source = Path(path).expanduser().resolve()
        if source.suffix.lower() not in self.suffixes or not source.is_file():
            raise ValueError(
                f"workspace resource must be an existing {self.suffixes!r} file"
            )
        return ResolvedWorkspaceResource(
            source,
            self.contract_id,
            self.decode(source),
        )


@dataclass(frozen=True)
class ArtifactOutputSpec:
    """One saved-artifact path attribute exposed by semantic contract."""

    name: str
    contract_id: str

    def __post_init__(self) -> None:
        if not self.name or not self.contract_id:
            raise ValueError("artifact output requires name and contract_id")


@dataclass(frozen=True, slots=True)
class NodePreviewSpec:
    """UI-neutral request to preview one typed output declaration.

    A node with eight outputs knows which one an operator came to watch, and
    nothing else does.  The same is true of HOW it is watched: the plotting
    package can see the shape of a dataset but not its physics.  Three frames
    of a calibration cycle are three point rows of an image to it, and
    averaging them is a perfectly sensible thing to do with three point rows
    -- but they are a long reference, a short readout and a long reference,
    and the only reason to look at them is side by side.  The node states the
    kind because the node is what knows that.  ``producer`` is empty for the
    node's own output; a plain suffix names a stable companion producer owned
    by the same run.  ``semantic`` is the plot projection assignment, shared
    unchanged with any estimator that consumes the same publication.
    ``overlay`` optionally names a distinct output from that same producer.
    """

    output: DatasetOutputDeclaration
    plot_kind: str
    semantic: Mapping[str, object] = field(default_factory=dict)
    producer: str = ""
    overlay: DatasetOutputDeclaration | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.output, DatasetOutputDeclaration):
            raise TypeError("node preview output must be DatasetOutputDeclaration")
        plot_kind = str(self.plot_kind).strip()
        if not plot_kind:
            raise ValueError("node preview requires a plot kind")
        semantic = dict(self.semantic)
        if any(not isinstance(name, str) or not name.strip() for name in semantic):
            raise TypeError("node preview semantic names must be non-empty text")
        producer = str(self.producer).strip()
        if producer and ("/" in producer or "\\" in producer):
            raise ValueError("node preview producer must be a plain owner suffix")
        if self.overlay is not None and self.overlay.name == self.output.name:
            raise ValueError("node preview overlay must differ from its primary output")
        object.__setattr__(self, "plot_kind", plot_kind)
        object.__setattr__(self, "semantic", MappingProxyType(semantic))
        object.__setattr__(self, "producer", producer)


@dataclass(frozen=True)
class SelectionMapping:
    """Data-only translation from one committed plot selection to a draft patch.

    ``map_patch(selection, draft, context)`` reads the selection's
    ``plot_kind``, ``selector_kind`` and ``ranges`` (each with ``axis``,
    ``lower``, ``upper``). It answers None when the region names nothing this
    producer can be set from -- a box drawn off the sensor is a place, not a
    crop: "nothing to set" is an answer, not an error.
    """

    plot_kind: str
    selector_kind: str
    draft_fields: tuple[str, ...]
    map_patch: Callable[[Any, Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any] | None]

    def __post_init__(self) -> None:
        fields = tuple(self.draft_fields)
        if not self.plot_kind or not self.selector_kind:
            raise ValueError("selection mapping requires plot_kind and selector_kind")
        if not fields or any(not isinstance(value, str) or not value for value in fields):
            raise ValueError("selection mapping requires non-empty draft fields")
        if not callable(self.map_patch):
            raise TypeError("selection mapping map_patch must be callable")
        object.__setattr__(self, "draft_fields", fields)


__all__ = [
    "ArtifactCodec",
    "ArtifactInputSpec",
    "ArtifactOutputSpec",
    "NodePreviewSpec",
    "ResolvedArtifact",
    "ResolvedWorkspaceResource",
    "SelectionMapping",
    "WorkspaceResourceSpec",
    "frame_claim_entries",
]
