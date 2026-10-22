"""What a node run sees: v2's NodeExecutionContext over the protocol (§5, §6).

A node's ``execute(context)`` ports unchanged: the same method names publish
through the core's Runtime instead of an in-process signal plane. Runtime
declares every output of a generation at once, so ``set_run_record`` and the
first ``commit_live`` together become one ``run.begin``; an output first
committed later must be declared before that with ``declare_output``.
"""

from __future__ import annotations

import collections.abc
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from .declare import NODE_INSTANCE, DatasetOutputDeclaration
from .wire import PluginError

if TYPE_CHECKING:
    from .host import Run

STORAGES = ("latest", "finite")


@dataclass(frozen=True)
class LiveDatasetOutput:
    """One immutable event and, for a finite run, its placement in the run's dataset.

    ``snapshot`` is a ``zlc_data.OwnedSnapshot``. A finite output states the
    complete run schema and the event's ``(repeat, point)`` origin; a monitor
    states neither and Runtime keeps only its latest event (plus any history a
    consumer leases). Coverage is Runtime's fact, not the producer's.
    ``ordinal`` is the acquisition's own sample index; omitted, it is the
    consumed source event's ordinal, or else events count 0, 1, 2 ... per
    output in this run.
    """

    declaration: DatasetOutputDeclaration
    snapshot: Any
    canonical_schema: Any = None
    cell_origin: tuple[int, int] | None = None
    event_record: Mapping[str, Any] | None = None
    shot_time_seconds: float | None = None
    ordinal: int | None = None

    def __post_init__(self) -> None:
        if (self.canonical_schema is None) != (self.cell_origin is None):
            raise ValueError("canonical_schema and cell_origin must be supplied together")


@dataclass(frozen=True)
class Read:
    """One leased Runtime read: its publication key and a zlc_data snapshot.

    The lease pins that publication in the core (for ``source_publication``)
    until this object is garbage; its arrays hold their blocks independently.
    ``lineage`` is its ancestry, nearest first, each publication once:
    ``{key, run_record, event_record}`` (v2's direct parents walked to the root).
    """

    lease: int
    key: Mapping[str, Any]
    snapshot: Any
    metadata: Mapping[str, Any]
    lineage: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class SignalValue:
    """One delivered input of a processor (v2's SignalValue without coverage).

    ``snapshot`` is the event; a finite source also states its run's
    ``canonical_schema`` and this event's ``cell_origin``. ``run_record`` is
    the source run's declared record, ``event_record`` this event's.
    """

    name: str
    snapshot: Any
    canonical_schema: Any
    cell_origin: tuple[int, int] | None
    run_record: Mapping[str, Any]
    event_record: Mapping[str, Any] | None
    lease: int
    key: Mapping[str, Any]


def signal_value(read: Mapping[str, Any], release: Any, refs: "RunRefs | None" = None) -> SignalValue:
    """One read as its processor sees it; ``refs`` is its run's table of what
    the run's deliveries name by number (:class:`RunRefs`)."""

    placement = read.get("placement") or {}
    origin = placement.get("cell_origin")
    metadata = read.get("metadata") or {}
    output = str(read["signal"]).rsplit("/", 1)[-1]
    # A plugin's commit keeps each output's record under "records"; another producer's metadata is its record.
    event_record = metadata.get("records", {}).get(output) if "records" in metadata else metadata
    record = read.get("run_record_ref")
    value = SignalValue(str(read["signal"]), snapshot_from_wire(read, refs),
                        _schema(placement, "canonical_schema", refs),
                        None if origin is None else (int(origin[0]), int(origin[1])),
                        (read.get("run_record") or {}) if record is None else refs.run_records[int(record)],
                        event_record, int(read["lease"]), read["key"])
    weakref.finalize(value, release, value.lease)
    return value


class RunRefs:
    """What a run's deliveries name by number (§5): schemas and the source runs'
    records, each sent whole once. Deliveries reach the run in the order the
    core sent them, so a number is defined by the time a read names it. A run
    record is shared by every delivery that names it: read it, do not change it."""

    def __init__(self) -> None:
        self.schemas: dict[int, Any] = {}
        self.run_records: dict[int, Mapping[str, Any]] = {}

    def adopt(self, params: Mapping[str, Any]) -> None:
        """One delivery's definitions; ``refs_reset`` (a new source) empties the table first."""

        from zlc_data.codec import dataset_schema_from_tree

        if params.get("refs_reset"):
            self.schemas.clear()
            self.run_records.clear()
        for number, tree in (params.get("schemas") or {}).items():
            self.schemas[int(number)] = dataset_schema_from_tree(tree)
        for number, record in (params.get("run_records") or {}).items():
            self.run_records[int(number)] = record


class DeliveredInputs(collections.abc.Mapping):
    """One delivery's inputs by member, each made a :class:`SignalValue` when
    first read: a bundle member the node never reads costs nothing.
    ``release_unread`` lets go of the leases of those members."""

    def __init__(self, reads: Mapping[str, Mapping[str, Any]], refs: RunRefs, release: Any) -> None:
        self._reads, self._refs, self._release = reads, refs, release
        self._values: dict[str, SignalValue] = {}

    def __getitem__(self, name: str) -> SignalValue:
        value = self._values.get(name)
        if value is None:
            value = self._values[name] = signal_value(self._reads[name], self._release, self._refs)
        return value

    def __iter__(self):
        return iter(self._reads)

    def __len__(self) -> int:
        return len(self._reads)

    def publication(self, name: str) -> Any:
        """What a commit names as its parent: the member as read, else its key and lease."""

        value = self._values.get(name)
        if value is not None:
            return value
        read = self._reads[name]
        return _Publication(read["key"], int(read["lease"]))

    def release_unread(self) -> None:
        for name, read in self._reads.items():
            if name not in self._values:
                self._release(int(read["lease"]))


@dataclass(frozen=True)
class _Publication:
    key: Mapping[str, Any]
    lease: int


def _schema(holder: Mapping[str, Any], field: str, refs: RunRefs | None) -> Any:
    """The schema ``field`` of a read: sent whole, or named by number
    (``<field>_ref``) in its run's table; None when the read has none."""

    reference = holder.get(f"{field}_ref")
    if reference is not None:
        return refs.schemas[int(reference)]
    tree = holder.get(field)
    if tree is None:
        return None
    from zlc_data.codec import dataset_schema_from_tree

    return dataset_schema_from_tree(tree)


def _tree(schema: Any) -> dict[str, Any]:
    from zlc_data.codec import dataset_schema_to_tree

    return dataset_schema_to_tree(schema)


def _planes(snapshot: Any) -> tuple[Any, Any, Any, Any]:
    block = snapshot.materialize().block if hasattr(snapshot, "materialize") else snapshot.block
    values, mask, sigma = block.as_segment()
    return block.schema, values, mask, sigma


def snapshot_from_wire(read: Mapping[str, Any], refs: RunRefs | None = None) -> Any:
    """The zlc_data snapshot of one read, sharing the received planes."""

    import zlc_data

    schema = _schema(read, "schema", refs)
    segments, origins, shapes = [], [], []
    for segment in read["segments"]:
        data = segment["data"]
        segments.append((data["values"], data["validity"], data["sigma"]))
        origins.append(segment["origin"])
        shapes.append(segment["extent"])
    key = read["key"]
    window = read.get("window")
    block = zlc_data.DataBlock(
        zlc_data.BlockId(str(read.get("block_id") or key["signal"])), zlc_data.DatasetRevision(int(key["revision"])),
        None, zlc_data.INVALID, schema, segments=tuple(segments),
        segment_origins=np.asarray(origins, dtype="<i8").reshape(len(origins), 2),
        segment_shapes=np.asarray(shapes, dtype="<i8").reshape(len(shapes), 2),
        window=None if window is None else zlc_data.IndexedWindow(int(window[0]), int(window[1])))
    return zlc_data.OwnedSnapshot(block.ref(zlc_data.StreamGenerationId(str(key["generation"]))), block)


class RunContext:
    """The capabilities and identity one node run publishes with."""

    def __init__(self, run: "Run") -> None:
        self._run = run
        self._record: dict[str, Any] = {}
        self._declared: dict[str, tuple[Any, str, bool]] = {}
        self._begun = False
        self._ordinals: dict[str, int] = {}
        self._sent_schema: dict[str, str] = {}

    # ---------------------------------------------------------- identity

    @property
    def instance_id(self) -> str:
        """The node's id (its plugin instance is named under the core's node prefix)."""

        name = self._run.instance.name
        return name[len(NODE_INSTANCE):] if name.startswith(NODE_INSTANCE) else name

    @property
    def generation(self) -> str:
        """This run's identity; a restart is a new one."""

        return self._run.run_id

    def cancel_requested(self) -> bool:
        return self._run.cancel.is_set()

    # ---------------------------------------------------------- lifecycle facts

    def _event(self, op: str, params: dict[str, Any]) -> None:
        self._run.instance.process.event(op, {"instance": self._run.instance.name, "run": self.generation, **params})

    def _request(self, op: str, params: dict[str, Any], **options: Any) -> Any:
        return self._run.instance.process.request(op, {"instance": self._run.instance.name, "run": self.generation, **params},
                                                  **options)

    def report_ready(self) -> None:
        self._event("run.ready", {})

    def report_progress(self, message: str, *, current: int | None = None, total: int | None = None) -> None:
        self._event("run.progress", {"message": str(message), "current": current, "total": total})

    # ---------------------------------------------------------- an exact input

    def next_input(self, timeout: float) -> SignalValue | None:
        """The next event of this run's exact input, in order, or None after
        ``timeout`` seconds without one.

        For a node that declares a dataset input and is started with a
        source signal but is not a processor (a scan that fires the pulse
        driving its own source): the core delivers that signal's events as
        it does a processor's, and the run takes them here. Each is
        acknowledged as it is taken, so the core sends the next. The end of
        the source (sealed, retired, failed) ends the wait with an error.
        """

        import queue

        process = self._run.instance.process
        try:
            identity, params = self._run.deliveries.get(timeout=max(0.0, float(timeout)))
        except queue.Empty:
            return None
        kind = params.get("kind")
        process.reply(identity, {})
        if kind != "data":
            reason = {"sealed": "the source signal ended", "retired": "the source run was replaced"}.get(
                str(kind), str(params.get("error") or "the input failed"))
            raise PluginError("failed", f"{reason} during the run")
        self._run.refs.adopt(params)
        inputs = DeliveredInputs(params["inputs"], self._run.refs, process.release_lease)
        value = inputs[params["anchor"]]
        inputs.release_unread()
        return value

    def discard_inputs(self, quiet_seconds: float = 0.05) -> int:
        """Drop every event of the exact input delivered so far: what was
        published before this point belongs to the world before it. Returns
        how many; stops once none has come for ``quiet_seconds``."""

        dropped = 0
        while self.next_input(quiet_seconds) is not None:
            dropped += 1
        return dropped

    def request_operator_input(self, kind: str, *, title: str, message: str,
                               payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Wait for the operator's answer. A cancel, or the run's Stop, is the run
        stopping (v2: ``InterruptedError("operator input was cancelled")`` with
        the stop set), so the run ends stopped, not failed. Asked after the Stop it is that
        Stop at once: nothing is shown and nothing waits."""

        if self._run.cancel.is_set():
            raise InterruptedError("operator input was cancelled")
        try:
            return self._request("run.operator_input", {"kind": kind, "title": title, "message": message,
                                                        "payload": dict(payload)})
        except PluginError as error:
            if error.kind != "cancelled":
                raise
            self._run.cancel.set()
            raise InterruptedError("operator input was cancelled") from None

    @property
    def run_directory(self) -> Any:
        directory = self._run.params.get("run_directory")
        if directory is None:
            raise RuntimeError("only a hosted Task has a run directory")
        import pathlib

        return pathlib.Path(directory)

    def register_artifact(self, name: str, path: Any, *, role: str, contract_id: str = "") -> Any:
        return self._request("run.artifact", {"name": name, "path": str(path), "role": role,
                                              "contract_id": contract_id})

    def save_figure(self, path: Any, *, plot_input: Any, spec: Any, parameters: Mapping[str, Any] | None = None,
                    size: str = "4x4", source: Mapping[str, Any] | None = None,
                    lineage: Mapping[str, Any] | None = None, image: bool = True,
                    **configuration: Any) -> tuple[Any, Any]:
        """Have the core write ``path``'s figure: the archive (``<path>.npz``) of
        this dataset (an ``ImageFrame`` for one with points over it) and the
        plot ``spec`` (zlc_plugin.figures) with these ``parameters``/``size``
        and recipe parts, then -- unless ``image`` is False -- its picture.
        Answers ``(image path or None, archive path)``; registering them as
        run artifacts stays the caller's."""

        import pathlib

        from .figures import figure_request

        answer = self._request("run.figure", figure_request(
            path, plot_input=plot_input, spec=spec, parameters=parameters, size=size, source=source,
            lineage=lineage, image=image, configuration=configuration))
        written = answer.get("image")
        return (None if written is None else pathlib.Path(written)), pathlib.Path(answer["archive"])

    def save_animation(self, path: Any, images: Any, *, frame_ms: int = 200) -> Any:
        """Have the core write ``path``: a looping GIF of these images (the
        run's figure previews), ``frame_ms`` each. Answers the written path;
        registering it as a run artifact stays the caller's."""

        import pathlib

        answer = self._request("run.animation", {"path": str(path), "images": [str(image) for image in images],
                                                 "frame_ms": int(frame_ms)})
        return pathlib.Path(answer["path"])

    def seal_terminal(self, *, accept_stop: bool = False, partial: bool = False) -> None:
        """Enter the run's non-cancellable terminal commit (v2's NodeExecutionContext rule).

        A Stop that came first refuses the seal (the run ends stopped) unless
        ``accept_stop``, which keeps the work and records the run as stopped;
        once sealed, a later Stop no longer interrupts or relabels it.
        ``partial`` says the committed data is a cut-short prefix.
        """

        run = self._run
        with run.lock:
            stopped = run.cancel.is_set()
            if stopped and not accept_stop:
                raise PluginError("cancelled", "the run was stopped before its terminal commit")
            run.sealed, run.stop_accepted, run.partial = True, stopped and bool(accept_stop), bool(partial)

    def register_partial_exit_writer(self, writer: Any) -> None:
        """The Task's one writer of what a stopped or failed run leaves: called
        once, on the run thread, with the ending (``"stopped"``/``"failed"``) and
        the error, before the run reports its end."""

        if not callable(writer):
            raise TypeError("partial exit writer must be callable")
        if self._run.partial_exit_writer is not None:
            raise RuntimeError("the run already registered its partial exit writer")
        self._run.partial_exit_writer = writer

    def acquire(self, node: str, *, completion: str = "ready") -> Any:
        """Start the acquisition node this run drives (v2's restart_logic) and wait
        until it is ``ready`` or ``terminal``."""

        return self._request("run.acquire", {"node": node, "completion": completion})

    def core(self, command: str, **params: Any) -> Any:
        """Run one catalogue command under this run's origin (its claims apply)."""

        return self._request("core", {"command": command, "params": params})

    # ---------------------------------------------------------- publication

    def set_run_record(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        """Declare this run's prepared facts before its first event."""

        if self._begun:
            raise RuntimeError("the run record is declared before the first commit")
        self._record = dict(record)
        return self._record

    def declare_output(self, name: str, schema: Any, *, storage: str = "latest", index_by_source: bool = False) -> None:
        """Declare an output before the first commit (needed only if that commit omits it)."""

        if self._begun:
            raise RuntimeError("outputs are declared before the first commit")
        if storage not in STORAGES:
            raise ValueError(f"storage must be one of {STORAGES}")
        self._declared[name] = (schema, storage, bool(index_by_source))

    def allocate(self, shape: tuple[int, ...] | int, dtype: Any) -> np.ndarray:
        from .host import allocate

        return allocate(shape, dtype)

    def commit_live(self, outputs: Mapping[str, LiveDatasetOutput], *,
                    source_publication: Read | None = None) -> Mapping[str, Mapping[str, Any]]:
        """Commit one atomic event bundle; returns each output's publication key."""

        if not outputs:
            raise ValueError("a commit needs at least one output")
        # A processor's event is the same sample as the source event it consumed.
        source_ordinal = None if source_publication is None else source_publication.key.get("ordinal")
        entries, records, times = [], {}, {}
        for name, output in outputs.items():
            if not isinstance(output, LiveDatasetOutput):
                raise TypeError("commit_live takes LiveDatasetOutput values")
            if output.declaration.name != name:
                raise ValueError(f"output {name!r} carries the declaration of {output.declaration.name!r}")
            schema, values, mask, sigma = _planes(output.snapshot)
            finite = output.canonical_schema is not None
            if not self._begun and name not in self._declared:
                self._declared[name] = (output.canonical_schema if finite else schema,
                                        "finite" if finite else "latest", output.declaration.index_by_source)
            ordinal = (int(output.ordinal) if output.ordinal is not None else int(source_ordinal)
                       if source_ordinal is not None else self._ordinals.get(name, 0))
            entry: dict[str, Any] = {"output": name, "ordinal": ordinal,
                                     "origin": list(output.cell_origin) if finite else None,
                                     "data": {"values": values, "validity": mask, "sigma": sigma}}
            fingerprint = schema.fingerprint
            if self._sent_schema.get(name) != fingerprint:
                declared = self._declared.get(name)
                if finite or declared is None or declared[0].fingerprint != fingerprint:
                    entry["schema"] = _tree(schema)
                self._sent_schema[name] = fingerprint
            entries.append(entry)
            if output.event_record is not None:
                records[name] = dict(output.event_record)
            if output.shot_time_seconds is not None:
                times[name] = float(output.shot_time_seconds)
        if not self._begun:
            self._request("run.begin", {"record": self._record, "outputs": [
                {"name": name, "schema": _tree(schema), "storage": storage, "index_by_source": index}
                for name, (schema, storage, index) in self._declared.items()]})
            self._begun = True
        metadata: dict[str, Any] = {}
        if records:
            metadata["records"] = records
        if times:
            metadata["shot_time_seconds"] = times
        parents = [] if source_publication is None else [source_publication.lease]
        keys = self._request("run.commit", {"inputs": entries, "metadata": metadata, "parents": parents})
        for entry in entries:
            self._ordinals[entry["output"]] = entry["ordinal"] + 1
        return keys

    # ---------------------------------------------------------- reading

    def read(self, signal: str, *, mode: str = "event", width: int | None = None) -> Read:
        """Lease one Runtime read: ``event``, ``dataset`` or ``window`` (with ``width``)."""

        answer = self._request("runtime.read", {"signal": signal, "mode": mode, "width": width})
        read = Read(int(answer["lease"]), answer["key"], snapshot_from_wire(answer), answer.get("metadata", {}),
                    tuple(answer.get("lineage") or ()))
        weakref.finalize(read, self._run.instance.process.release_lease, read.lease)
        return read

    def current_dataset(self, output_name: str, publication: Read | None = None) -> Any:
        """The current complete dataset of one of this run's own outputs."""

        if publication is not None:
            return publication.snapshot
        return self.read(f"{self.instance_id}/{output_name}", mode="dataset").snapshot


__all__ = ["DeliveredInputs", "LiveDatasetOutput", "Read", "RunContext", "RunRefs", "SignalValue", "signal_value",
           "snapshot_from_wire"]
