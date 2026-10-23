"""The plugin host process: ``python -m zlc_plugin host``, every device and node
instance in one process (PLUGIN_PROTOCOL.md §1, §3-§5).

The main thread reads frames. ``ping``, ``cancel``, ``release``, ``abandon`` and
``run.stop`` are answered there; every other request for an instance runs on
that instance's lane thread, in arrival order. A node run executes on its own
thread so its lane stays free for queries and its device calls' replies. The
catalogue (``describe``) and hardware scans (``discover``) run on threads of their own.
"""

from __future__ import annotations

import itertools
import logging
import os
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from . import capabilities, tunable
from .blocks import Arena, Attachments
from .declare import ADMIT, QUERY_PARAMETERS, DeviceType, NodeKind, NodeType, Opened, field_json
from .describe import describe, find
from .wire import (PROTOCOL, Channel, Decoder, Encoder, PluginError, error_payload, plain, raise_remote)


#: Where the bench's device code narrates (v2 _ServerLogBuffer.SOURCES).
_LOG_SOURCES = ("zlc_pulse.remote", "zlc_atom.devices")


class _DeviceLog(logging.Handler):
    """v2's device log: every INFO line of device code, as ``[HH:MM:SS] message``, sent
    to the core for each open device it is about -- under the device type's
    ``log_channels``, or ending in the device's ``device=<identity>``."""

    def __init__(self, process: "Process") -> None:
        super().__init__(level=logging.INFO)
        self.process = process
        self.loggers: set[str] = set()

    def listen(self, names: tuple[str, ...]) -> None:
        for name in names:
            # A logger under one already heard reaches it by propagation.
            if any(name == heard or name.startswith(heard + ".") for heard in self.loggers):
                continue
            self.loggers.add(name)
            logger = logging.getLogger(name)
            logger.addHandler(self)
            if logger.getEffectiveLevel() > logging.INFO:
                logger.setLevel(logging.INFO)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = f"[{time.strftime('%H:%M:%S', time.localtime(record.created))}] {record.getMessage()}"
            for name, instance in list(self.process.instances.items()):
                declared = instance.declared
                if not isinstance(declared, DeviceType):
                    continue
                if record.name.startswith(declared.log_channels) or (instance.log_token and line.endswith(instance.log_token)):
                    self.process.event("device.log", {"instance": name, "line": line})
        except Exception:  # noqa: BLE001 -- a log line never breaks the code that wrote it
            self.handleError(record)


class _Pending:
    __slots__ = ("event", "message")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.message: dict[str, Any] | None = None


class Process:
    """The channel, the block arena and every instance this process hosts."""

    def __init__(self, channel: Channel) -> None:
        self.channel = channel
        self.session = "unset"
        self.arena: Arena | None = None
        self._releases: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()
        self.attachments = Attachments(lambda name: self._releases.put(("blocks", name)))
        self.instances: dict[str, Instance] = {}
        self.shared: dict[str, Any] = {}
        self._pending: dict[int, _Pending] = {}
        self._pending_lock = threading.Lock()
        self._ids = itertools.count(1)
        self.device_log = _DeviceLog(self)
        threading.Thread(target=self._send_releases, name="zlc-plugin-release", daemon=True).start()

    # ---------------------------------------------------------------- values

    def encode(self, value: Any, *, plain: bool = False) -> tuple[Any, list[str]]:
        if self.arena is None:
            encoder = Encoder(None, plain=plain)
        else:
            arena, holder = self.arena, self.attachments
            encoder = Encoder(lambda array: arena.lend(array, holder), plain=plain)
        return encoder(value), encoder.blocks

    def decode(self, value: Any) -> Any:
        return Decoder(self.attachments.attach)(value)

    # --------------------------------------------------------------- sending

    def send(self, message: dict[str, Any]) -> None:
        self.channel.send(message)

    def reply(self, identity: int, result: Any = None, *, error: BaseException | None = None,
              kind: str | None = None, plain: bool = False) -> None:
        if error is not None:
            self.send({"t": "res", "id": identity, "error": error_payload(error, kind=kind)})
            return
        value, blocks = self.encode(result, plain=plain)
        self.send({"t": "res", "id": identity, "ok": value, "blocks": blocks})

    def event(self, op: str, params: dict[str, Any], *, plain: bool = False) -> None:
        value, blocks = self.encode(params, plain=plain)
        self.send({"t": "evt", "op": op, "p": value, "blocks": blocks})

    def request(self, op: str, params: dict[str, Any], *, deadline_ms: int | None = None) -> Any:
        """Ask the core and wait for its answer; remote errors are raised here."""

        identity = next(self._ids)
        pending = _Pending()
        with self._pending_lock:
            self._pending[identity] = pending
        value, blocks = self.encode(params)
        message: dict[str, Any] = {"t": "req", "id": identity, "op": op, "p": value, "blocks": blocks}
        if deadline_ms is not None:
            message["deadline_ms"] = int(deadline_ms)
        self.send(message)
        pending.event.wait()
        answer = pending.message
        assert answer is not None
        if "error" in answer:
            raise_remote(answer["error"])
        return self.decode(answer.get("ok"))

    def release_lease(self, lease: int) -> None:
        """Called by a read's finalizer: only enqueues."""

        self._releases.put(("leases", lease))

    def _send_releases(self) -> None:
        while True:
            items = [self._releases.get()]
            time.sleep(0.002)  # let one GC pass's releases share a frame
            while True:
                try:
                    items.append(self._releases.get_nowait())
                except queue.Empty:
                    break
            released: dict[str, list[Any]] = {"blocks": [], "leases": []}
            for kind, value in items:
                released[kind].append(value)
            try:
                self.send({"t": "evt", "op": "release", "p": released})
            except (OSError, ValueError):
                return

    # --------------------------------------------------------------- serving

    def serve(self) -> int:
        while True:
            try:
                message = self.channel.receive()
            except EOFError:
                message = None
            if message is None:
                break
            kind = message.get("t")
            if kind == "res":
                with self._pending_lock:
                    pending = self._pending.pop(int(message["id"]), None)
                if pending is not None:
                    pending.message = message
                    pending.event.set()
            elif kind == "evt":
                self._event(message)
            elif kind == "req":
                if self._request(message) == "exit":
                    break
        for instance in list(self.instances.values()):
            instance.shutdown()
        return 0

    def _event(self, message: dict[str, Any]) -> None:
        op = message.get("op")
        params = message.get("p") or {}
        if op == "release" and self.arena is not None:
            for name in params.get("blocks", ()):
                self.arena.released(str(name))
        elif op == "run.stop":
            instance = self.instances.get(str(params.get("instance")))
            if instance is not None:
                instance.stop_run(str(params.get("run")))
        elif op == "abandon":
            # The core gave up on an instance whose call overran its deadline (§7): it
            # rejects new opens until its lane returns and releases its device.
            instance = self.instances.get(str(params.get("instance")))
            if instance is not None:
                instance.abandon()

    def _discover(self, identity: int, params: dict[str, Any]) -> None:
        try:
            declared = find(str(params["type_id"]), str(params["module"]))
            if not isinstance(declared, DeviceType) or declared.discover is None:
                raise PluginError("unknown_op", f"{params['type_id']} has no hardware scan")
            found, notes = declared.discover()
            self.reply(identity, {"found": [dict(entry) for entry in found], "notes": [str(note) for note in notes]}, plain=True)
        except BaseException as error:  # noqa: BLE001 -- the scan's own refusal, reported per family
            self.reply(identity, error=error)

    def _describe(self, identity: int, params: dict[str, Any]) -> None:
        try:
            self.reply(identity, describe([str(package) for package in params.get("packages") or ()]), plain=True)
        except BaseException as error:  # noqa: BLE001 -- the catalogue's own failure, said to the core
            self.reply(identity, error=error)

    def _request(self, message: dict[str, Any]) -> str | None:
        identity = int(message["id"])
        op = str(message.get("op"))
        params = message.get("p") or {}
        if op == "hello":
            major = int(params["protocol"][0])
            if major != PROTOCOL[0]:
                self.reply(identity, error=PluginError("invalid", f"protocol {major} is not {PROTOCOL[0]}"))
                return "exit"
            self.session = str(params["session"])
            self.arena = Arena(self.session)
            self.reply(identity, {"protocol": list(PROTOCOL), "pid": os.getpid(),
                                  "python": sys.version.split()[0], "sdk": "zlc_plugin 1"})
            return None
        if op == "ping":
            self.reply(identity, {})
            return None
        if op == "exit":
            self.reply(identity, {})
            return "exit"
        if op in ("discover", "describe"):
            # A hardware scan of one type, no instance opened (v2 Scan hardware), or the
            # catalogue: on a thread of its own, so a vendor call that hangs or a slow
            # import holds up neither this process's instances nor its channel; the core
            # stops waiting at its deadline and a late answer is dropped.
            target = self._discover if op == "discover" else self._describe
            threading.Thread(target=target, args=(identity, params), name=f"zlc-{op}", daemon=True).start()
            return None
        name = str(params.get("instance", ""))
        instance = self.instances.get(name)
        if instance is None:
            if op == "close":
                self.reply(identity, {"state": "closed"})
                return None
            if op != "open":
                self.reply(identity, error=PluginError("invalid", f"no open instance {name!r}"))
                return None
            instance = Instance(self, name)
            self.instances[name] = instance
        if instance.abandoned:
            self.reply(identity, error=PluginError("refused", f"{name} is still releasing its previous operation; retry after cleanup completes"))
            return None
        if op == "spi" and instance.concurrent(str(params.get("method", ""))):
            # A call the device declared it answers beside its lane (a cancel
            # while a playback holds the lane): on a thread of its own.
            threading.Thread(target=instance._handle, args=(message,), name=f"zlc-call-{name}",
                             daemon=True).start()
            return None
        instance.lane.put(message)
        return None


# One process; plugin code reaches it through these module functions.
_PROCESS: Process | None = None


def process() -> Process:
    if _PROCESS is None:
        raise RuntimeError("zlc_plugin is not running inside a plugin host process")
    return _PROCESS


class Instance:
    """One device or node instance and its lane."""

    def __init__(self, owner: Process, name: str) -> None:
        self.process = owner
        self.name = name
        self.abandoned = False
        self.declared: DeviceType | NodeType | None = None
        self.adapter: object | None = None
        self.closer: Callable[[], None] | None = None
        self.admit_peers: Callable[[bool], None] | None = None
        self.values: dict[str, Any] = {}
        #: ``device=<identity>``: how this device's own log lines end (v2 identity_token).
        self.log_token: str | None = None
        self.runs: dict[str, Run] = {}
        self.prepared: dict[str, tuple[object, dict[str, Any]]] = {}
        self.lane: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._serve, name=f"zlc-lane-{name}", daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while True:
            message = self.lane.get()
            if message is not None:
                if self.abandoned:
                    self.process.reply(int(message["id"]), error=PluginError("cancelled", "the instance is releasing an overdue operation"))
                    continue
                if self._handle(message):
                    continue
            if self.abandoned:
                # An overrun owns its handle until it actually returns. If
                # cleanup fails, keep this lane available for a cleanup retry.
                try:
                    self.close_device()
                except BaseException as error:
                    print(f"{self.name}: cleanup after timeout failed: {error}", file=sys.stderr)
                    self.abandoned = False
                    continue
                self._detach()
            return

    def abandon(self) -> None:
        self.abandoned = True
        for run in list(self.runs.values()):
            run.cancel.set()
        self.lane.put(None)

    def _handle(self, message: dict[str, Any]) -> bool:
        """Answer one request; False when the instance is gone.

        Its own frame owns the arguments and the result, so a lent array is
        released as soon as the answer is sent, not at the next request.
        """

        identity = int(message["id"])
        op = str(message.get("op"))
        try:
            params = self.process.decode(message.get("p") or {})
            if op == "run.deliver":
                # Answered by the run thread once taken, or refused once the run has ended.
                run = self.runs.get(str(params.get("run")))
                if run is None:
                    _refuse_delivery(self.process, identity, params)
                else:
                    run.deliver(identity, params)
                return True
            handler = _HANDLERS.get(op)
            if handler is None:
                self.process.reply(identity, error=PluginError("unknown_op", f"unknown operation {op!r}"))
                return True
            result = handler(self, params)
        except BaseException as error:  # noqa: BLE001 -- report the failure and keep cleanup ownership
            cause = error.cause if isinstance(error, _Invalid) else error
            keep = True
            if op == "open":
                try:
                    self.close_device()
                except BaseException as cleanup:
                    cause.add_note(f"cleanup after failed Init also failed: {cleanup}")
                else:
                    self._detach()
                    keep = False
            # A failed Close keeps its adapter, closer and lane. A subsequent
            # Close or Init can finish cleanup instead of opening it twice.
            self.process.reply(identity, error=cause, kind="invalid" if isinstance(error, _Invalid) else None)
            return keep
        if op == "close":
            # An acknowledgement is a lifecycle barrier: its recipient may
            # immediately Open this name on the channel's other thread.
            self._detach()
        self.process.reply(identity, result, plain=bool((message.get("p") or {}).get("plain")))
        return op != "close"

    def _detach(self) -> None:
        # A late abandoned lane must never remove a replacement with this name.
        if self.process.instances.get(self.name) is self:
            self.process.instances.pop(self.name)

    def concurrent(self, method: str) -> bool:
        declared = self.declared
        return isinstance(declared, DeviceType) and self.adapter is not None and method in declared.concurrent

    def close_device(self) -> None:
        if callable(self.closer):
            self.closer()
        self.closer, self.adapter = None, None
        self.admit_peers, self.log_token = None, None

    def shutdown(self) -> None:
        for run in list(self.runs.values()):
            run.cancel.set()
        try:
            self.close_device()
        except Exception as error:  # noqa: BLE001 -- the process is ending; say what failed
            print(f"{self.name}: close at shutdown failed: {error}", file=sys.stderr)
        self.lane.put(None)

    def stop_run(self, run: str) -> None:
        current = self.runs.get(run)
        if current is not None:
            with current.lock:
                if not current.sealed:
                    current.cancel.set()

    def device(self) -> DeviceType:
        if not isinstance(self.declared, DeviceType) or self.adapter is None:
            raise PluginError("invalid", f"{self.name} is not an open device")
        return self.declared

    def node(self) -> NodeType:
        if not isinstance(self.declared, NodeType):
            raise PluginError("invalid", f"{self.name} is not a node")
        return self.declared


class _Invalid(Exception):
    def __init__(self, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.cause = cause


def _projected(project: Callable[[Any], dict[str, Any]], values: Any) -> dict[str, Any]:
    try:
        return project(values)
    except (TypeError, ValueError) as error:
        raise _Invalid(error) from error


def _tunable(adapter: object) -> bool:
    return all(callable(getattr(adapter, name, None)) for name in capabilities.TUNABLE)


def _open(instance: Instance, params: dict[str, Any]) -> dict[str, Any]:
    # A prior Close/Init may have failed with its handle still owned here.
    # Release that exact handle before constructing another device.
    instance.close_device()
    declared = find(str(params["type_id"]), str(params["module"]))
    instance.declared = declared
    if isinstance(declared, DeviceType):
        values = _projected(declared.project, params.get("parameters"))
        instance.values = values
        # This open's simulation parameters (``simulation_parameters``), whatever else
        # the process opens beside it.
        _OPENING.simulation = dict(params.get("simulation") or {})
        try:
            opened = declared.factory(instance.name, values)
        finally:
            _OPENING.simulation = None
        if isinstance(opened, Opened):
            instance.adapter, instance.closer, instance.admit_peers = opened.adapter, opened.close, opened.admit_peers
        else:
            instance.adapter, instance.closer = opened, getattr(opened, "close", None)
        identity = getattr(instance.adapter, "identity", None)
        instance.log_token = f"device={identity}" if identity else None
        instance.process.device_log.listen(_LOG_SOURCES + declared.log_channels)
        # What a draft query may call on it: the core refuses every other method then.
        # `narrates`: whether anything is its device log (v2 says so when not).
        return {"state": "open", "tunable": _tunable(instance.adapter), "reads": list(capabilities.reads(instance.adapter)),
                "narrates": bool(declared.log_channels or instance.log_token)}
    instance.values = dict(params.get("parameters") or {})
    return {"state": "open"}


def _configure(instance: Instance, params: dict[str, Any]) -> dict[str, Any]:
    declared = instance.device()
    values = _projected(declared.project, params.get("parameters"))
    reconfigure = getattr(instance.adapter, "reconfigure", None)
    if not callable(reconfigure):
        raise PluginError("refused", f"{declared.type_id} cannot change its parameters while open; close and reopen it")
    reconfigure(values)
    instance.values = values
    return {"state": "open"}


def _close(instance: Instance, params: dict[str, Any]) -> dict[str, Any]:
    for run in list(instance.runs.values()):
        run.cancel.set()
        run.thread.join(timeout=10.0)
    instance.close_device()
    return {"state": "closed"}


def _spi(instance: Instance, params: dict[str, Any]) -> Any:
    instance.device()
    method = str(params["method"])
    if method.startswith("get:"):
        return getattr(instance.adapter, method[4:])
    if method.startswith("_"):
        raise PluginError("invalid", f"{method} is not part of a device contract")
    if method == "tunable_fields":
        # As data -- the rows Device Control shows -- so a run's proxy can read the knobs it moves.
        return _fields(instance, params)
    if method == "read_tunable_in_unit":
        # One knob read in a unit, by the device's own conversion: a row as well.
        return _row(instance.adapter.read_tunable_in_unit(*params.get("args", ()), **params.get("kwargs", {})))
    if method == "tune_in_unit":
        return tunable.tune_in_unit(instance.adapter, *params.get("args", ()), **params.get("kwargs", {}))
    target = getattr(instance.adapter, method)
    return target(*params.get("args", ()), **params.get("kwargs", {}))


def _command(instance: Instance, params: dict[str, Any]) -> Any:
    declared = instance.device()
    name = str(params["name"])
    command = next((item for item in declared.commands if item.name == name), None)
    if command is None:
        raise PluginError("unknown_op", f"{declared.type_id} has no command {name!r}")
    given = dict(params.get("params") or {})
    if command.file is None:
        return _handled(instance, command, given)
    # The operator's file, read or written by the core (FileParam): the handler
    # works on a private copy with its name, and a write's bytes go back.
    file = given.get(command.file.param)
    if not isinstance(file, dict) or not str(file.get("name") or "").strip():
        raise PluginError("invalid", f"{name}'s file comes from the core (its {command.file.param} names it)")
    with tempfile.TemporaryDirectory(prefix="zlc-file-") as folder:
        copy = Path(folder) / Path(str(file["name"])).name
        if command.file.direction == "read":
            copy.write_bytes(np.asarray(file.get("bytes"), dtype=np.uint8).tobytes())
        given[command.file.param] = str(copy)
        value = _handled(instance, command, given)
        if command.file.direction == "read":
            return value
        if not copy.is_file():
            raise PluginError("failed", f"{name} wrote no file")
        return {"file": np.frombuffer(copy.read_bytes(), dtype=np.uint8), "value": value}


def _handled(instance: Instance, command: Any, given: dict[str, Any]) -> Any:
    project = getattr(command.params, "project_values", None)
    values = _projected(project, given) if project else given
    handler = command.handler
    _COMMANDING.instance = instance
    try:
        if callable(handler):
            return handler(instance.adapter, **values)
        return getattr(instance.adapter, handler)(**values)
    finally:
        _COMMANDING.instance = None


#: The device instance whose command runs on this thread (its lane).
_COMMANDING = threading.local()


class Notifier:
    """Tells the core a fact of one open device outside any request: event
    ``device.notice {instance, name, value}``, its arrays lent (PLUGIN_PROTOCOL
    section 5). Callable from any thread; ``alive`` while the device it was
    made for is still the instance's open device."""

    def __init__(self, instance: Instance) -> None:
        self.instance = instance.name
        self._owner = instance
        self._adapter = instance.adapter

    @property
    def alive(self) -> bool:
        current = self._owner.process.instances.get(self.instance)
        return current is self._owner and self._owner.adapter is self._adapter and self._adapter is not None

    def __call__(self, name: str, value: Any) -> bool:
        if not self.alive:
            return False
        # As plain JSON, as the core reads every answer (arrays stay references).
        self._owner.process.event("device.notice", {"instance": self.instance, "name": str(name), "value": value}, plain=True)
        return True


def notifier() -> Notifier:
    """Inside a device command: the device's Notifier (an SLM Editor tells the
    core its solved draft when the solve ends, long after the command)."""

    instance = getattr(_COMMANDING, "instance", None)
    if instance is None:
        raise RuntimeError("zlc_plugin.notifier is for device commands")
    return Notifier(instance)


def _fields(instance: Instance, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Every tunable field, read fresh; `units` names the spelling a field is shown in."""
    instance.device()
    adapter: Any = instance.adapter
    units = dict(params.get("units") or {})
    rows = []
    for item in tunable.refresh_tunable_fields(adapter):
        unit = str(units.get(item.metadata.name) or "")
        if unit and unit != (item.metadata.unit or ""):
            item = tunable.read_tunable_in_unit(adapter, item.metadata.name, unit)
        rows.append(_row(item))
    return rows


def _row(item: Any) -> dict[str, Any]:
    """A TunableField as data (Device Control's row; a proxy reads it back)."""
    return {"field": field_json(item.metadata), "current": plain(item.current), "live_write": item.live_write,
            "dependency_group": list(item.dependency_group),
            "device_limits": None if item.device_limits is None else list(item.device_limits)}


def _tune(instance: Instance, params: dict[str, Any]) -> Any:
    instance.device()
    adapter: Any = instance.adapter
    # The value is in `unit` (the spelling the field is shown in), and so is the readback.
    return plain(tunable.tune_in_unit(adapter, str(params["field"]), params["value"], str(params.get("unit") or "")))


def _values(instance: Instance, params: dict[str, Any]) -> Any:
    instance.device()
    adapter: Any = instance.adapter
    return {"values": plain(dict(adapter.tunable_values())), "provenance": plain(dict(adapter.settings_provenance()))}


def _publish(instance: Instance, params: dict[str, Any]) -> dict[str, Any]:
    """Remote on or off for a device with a server of its own (DeviceType.announce):
    its server admits other machines or drops them; the answer is what a peer opens."""

    declared = instance.device()
    if declared.announce is None or instance.admit_peers is None:
        raise PluginError("invalid", f"{instance.name} has no server of its own to announce")
    instance.admit_peers(bool(params.get("published")))
    type_id, endpoint = declared.announce
    return {"type_id": str(type_id), "parameters": dict(endpoint(dict(instance.values)))}


def _make_safe(instance: Instance, params: dict[str, Any]) -> dict[str, Any]:
    instance.device()
    make_safe = getattr(instance.adapter, "make_safe", None)
    if not callable(make_safe):
        raise PluginError("failed", f"{instance.name} declares physical outputs but has no make_safe")
    make_safe(str(params["output"]))
    return {}


def _query(instance: Instance, params: dict[str, Any]) -> Any:
    name = str(params["name"])
    if name == "prepare_run":
        return _prepare_run(instance, params)
    declared = instance.declared
    if isinstance(declared, NodeType):
        if name == "selection_patch":
            return declared_selection_patch(declared, params)
        if name == ADMIT:
            return _admit(declared, params)
        function = getattr(declared, name, None) if name in QUERY_PARAMETERS else None
        if function is not None:
            # The core names the facts; the v2 callable takes them positionally.
            facts = dict(params)
            facts["devices"] = Run.proxies(instance, dict(params.get("devices") or {}), run=None)
            _QUERYING.instance = instance
            try:
                return function(*(facts.get(parameter) for parameter in QUERY_PARAMETERS[name]))
            finally:
                _QUERYING.instance = None
    raise PluginError("unknown_op", f"{instance.name} answers no query {name!r}")


#: The node instance whose draft query runs on this thread (its lane).
_QUERYING = threading.local()


def _admit(declared: NodeType, params: dict[str, Any]) -> list[str]:
    """The draft's admission only its own code can answer (v2 finalize_logic_draft): its
    values projected as its schema projects them, the cross-field validator included;
    what its type says of the resources it was given (``admit_resources``); and each
    artifact file it names, decoded. The issues, in that order; none is a draft Start
    may take."""

    values = dict(params.get("values") or {})
    issues: list[str] = []
    try:
        projected = declared.project(values)
    except (TypeError, ValueError) as error:
        projected = None
        # Only incomplete (the core says which field): a draft the schema refuses otherwise is an issue.
        draft = getattr(declared.authoring_schema, "draft_values", None)
        try:
            if callable(draft):
                draft(values)
        except (TypeError, ValueError) as refused:
            issues.append(str(refused))
        else:
            if not callable(draft):
                issues.append(str(error))
    if projected is not None and declared.admit_resources is not None:
        issues.extend(str(issue) for issue in declared.admit_resources(projected, dict(params.get("resources") or {}), None))
    issues.extend(_artifact_issues(declared, dict(params.get("artifacts") or {})))
    return list(dict.fromkeys(issue for issue in issues if issue))


#: Artifact files already decoded: (contract, path, size, modified) -> why it cannot be used ("" when it can).
_DECODED: dict[tuple[str, str, int, int], str] = {}


def _artifact_issues(declared: NodeType, artifacts: dict[str, Any]) -> list[str]:
    """Why a file the draft names for an artifact input cannot be used (v2 ``_invalid_artifact``).
    A file is decoded once while it stays the same; one that does not exist is the core's to say."""

    inputs, resolved = dict(artifacts.get("inputs") or {}), dict(artifacts.get("resolved") or {})
    issues = []
    for spec in declared.input_specs:
        codec = getattr(spec, "codec", None)
        if codec is None or spec.name not in inputs:
            continue
        value = inputs[spec.name]
        texts = [value] if isinstance(value, str) else [str(claim.get("path") or "") for claim in value if isinstance(claim, dict)] \
            if isinstance(value, list) else []
        for text in (text.strip() for text in texts):
            path = Path(str(resolved.get(text) or text))
            if not text or not path.is_file():
                continue
            stat = path.stat()
            key = (codec.contract_id, str(path), stat.st_size, stat.st_mtime_ns)
            if key not in _DECODED:
                try:
                    codec.resolve(path)
                    _DECODED[key] = ""
                except Exception as error:  # noqa: BLE001 -- the file's refusal is the draft's issue
                    _DECODED[key] = f"{codec.contract_id} artifact “{text}” is invalid: {error}"
            if _DECODED[key]:
                issues.append(_DECODED[key])
    return issues


def core(command: str, **params: Any) -> Any:
    """One read-only catalogue command asked from a draft query (an estimate's
    ``pulse_estimate``); the core refuses anything that is not a read. A run
    asks through ``ctx.core``, under its own origin."""

    instance = getattr(_QUERYING, "instance", None)
    if instance is None:
        raise RuntimeError("zlc_plugin.core is for draft queries; a run calls ctx.core")
    return instance.process.request("core", {"instance": instance.name, "run": None, "command": command, "params": params})


def declared_selection_patch(declared: NodeType, params: dict[str, Any]) -> Any:
    # The core sends v2's panel_selection_document; a mapping reads it as v2's
    # SelectionState (attributes, ranges and facets as tuples of records).
    from types import SimpleNamespace

    document = params["selection"]
    selection = SimpleNamespace(
        **{key: value for key, value in document.items() if key not in ("ranges", "facets", "drawn")},
        ranges=tuple(SimpleNamespace(**item) for item in document.get("ranges", ())),
        facets=tuple(SimpleNamespace(**item) for item in document.get("facets", ())),
        drawn=None if document.get("drawn") is None else SimpleNamespace(**document["drawn"]),
    )
    for mapping in declared.selection_mappings:
        if (selection.plot_kind, selection.selector_kind) == (mapping.plot_kind, mapping.selector_kind):
            mapped = mapping.map_patch(selection, params.get("draft", {}), params.get("context", {}))
            return None if mapped is None else dict(mapped)
    return None


def _prepare_run(instance: Instance, params: dict[str, Any]) -> dict[str, Any]:
    import uuid
    declared = instance.node()
    devices = Run.proxies(instance, dict(params.get("devices") or {}), run=None)
    arguments = dict(_projected(declared.project, params.get("values")))
    arguments.update(devices)
    arguments.update({f"{key}_key": info.get("instance") for key, info in (params.get("devices") or {}).items()})
    arguments.update(params.get("resources") or {})
    if params.get("source_signal") is not None:
        arguments["source_signal"] = params["source_signal"]
    _QUERYING.instance = instance
    try:
        node = declared if declared.build is None else declared.build(**arguments)
    finally:
        _QUERYING.instance = None
    token = uuid.uuid4().hex
    instance.prepared[token] = (node, devices)
    return {"prepared": token}


def _discard_run(instance: Instance, params: dict[str, Any]) -> dict[str, Any]:
    instance.prepared.pop(str(params["prepared"]), None)
    return {}


def _run_start(instance: Instance, params: dict[str, Any]) -> dict[str, Any]:
    declared = instance.node()
    run_id = str(params["run"])
    if any(run.thread.is_alive() for run in instance.runs.values()):
        raise PluginError("refused", f"{instance.name} is already running")
    prepared = None
    if params.get("prepared"):
        prepared = instance.prepared.pop(str(params["prepared"]), None)
        if prepared is None:
            raise PluginError("invalid", "the prepared run was discarded or already started")
        for device in prepared[1].values():
            device._run = run_id
    run = Run(instance, declared, run_id, params)
    run.prepared_node = None if prepared is None else prepared[0]
    instance.runs[run_id] = run
    run.thread.start()
    return {"run": run_id}


_HANDLERS: dict[str, Callable[[Instance, dict[str, Any]], Any]] = {
    "open": _open, "configure": _configure, "close": _close, "spi": _spi, "command": _command,
    "fields": _fields, "tune": _tune, "values": _values, "make_safe": _make_safe, "query": _query, "publish": _publish,
    "run.start": _run_start, "run.discard": _discard_run,
}


class _DeviceProxy:
    """A claimed device of a run, reached through the core."""

    def __init__(self, instance: Instance, run: str | None, argument: str, info: dict[str, Any]) -> None:
        self._instance = instance
        self._run = run
        self._argument = argument
        self._info = info
        self.role = str(info.get("role") or info.get("instance") or "")

    def _call(self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        timeout = kwargs.get("timeout")
        deadline = None if not isinstance(timeout, (int, float)) else int((float(timeout) + 60.0) * 1000)
        # Outside a run (a draft query) the core routes by the device it names; reads only.
        return self._instance.process.request("device", {
            "instance": self._instance.name, "run": self._run, "argument": self._argument,
            "device": self._info.get("instance"), "method": method, "args": list(args), "kwargs": kwargs}, deadline_ms=deadline)

    def __repr__(self) -> str:
        return f"<device {self._info.get('instance')} ({self._info.get('type_id')}) as {self._argument}>"


_PROXY_CLASSES: dict[tuple[tuple[str, ...], bool], type] = {}


def _proxy_class(tokens: tuple[str, ...], tunable: bool, reads: tuple[str, ...] = ()) -> type:
    key = (tokens, tunable, reads)
    if key not in _PROXY_CLASSES:
        namespace: dict[str, Any] = {}
        for token in tokens:
            properties, methods = capabilities.members(token)
            for name in properties:
                namespace[name] = property(lambda self, name=name: self._call(f"get:{name}", (), {}))
            for name in methods:
                namespace[name] = lambda self, *args, name=name, **kwargs: self._call(name, args, kwargs)
        # The quartet when the device is tunable, and every read it implements.
        for name in (*(capabilities.TUNABLE if tunable else ()), *reads):
            namespace.setdefault(name, lambda self, *args, name=name, **kwargs: self._call(name, args, kwargs))
        if tunable:
            namespace["tune_in_unit"] = lambda self, *args, **kwargs: self._call("tune_in_unit", args, kwargs)
        _PROXY_CLASSES[key] = type("DeviceProxy", (_DeviceProxy,), namespace)
    return _PROXY_CLASSES[key]


class Run:
    """One Start..terminal of a node instance, on its own thread."""

    def __init__(self, instance: Instance, declared: NodeType, run_id: str, params: dict[str, Any]) -> None:
        from .context import RunContext, RunRefs

        self.instance = instance
        self.declared = declared
        self.run_id = run_id
        self.prepared_node = None
        self.params = params
        self.cancel = threading.Event()
        #: seal_terminal's facts: sealed runs ignore a later Stop.
        self.lock = threading.Lock()
        self.sealed = self.stop_accepted = self.partial = False
        #: Set when the run has ended: a delivery is then refused, not queued.
        self.ended = False
        self.partial_exit_writer: Callable[[str, BaseException], None] | None = None
        self.context = RunContext(self)
        #: A processor's deliveries (request id, params), answered once committed.
        self.deliveries: queue.Queue[tuple[int, dict[str, Any]]] = queue.Queue()
        #: What this run's deliveries name by number: schemas, source run records.
        self.refs = RunRefs()
        self.thread = threading.Thread(target=self._main, name=f"zlc-run-{instance.name}", daemon=True)

    def deliver(self, identity: int, params: dict[str, Any]) -> None:
        """Queue one delivery for the run thread, which answers it; refuse one that comes after the end."""

        with self.lock:
            if not self.ended:
                self.deliveries.put((identity, params))
                return
        _refuse_delivery(self.instance.process, identity, params)

    def _end_deliveries(self) -> None:
        """The run has ended: each delivery it did not take is refused, so the core's wait on it ends."""

        with self.lock:
            self.ended = True
        while not self.deliveries.empty():
            _refuse_delivery(self.instance.process, *self.deliveries.get_nowait())

    def _process(self, node: Any) -> dict[str, Any]:
        """v2's follow processor: evaluate each delivery in order and commit it
        with the delivery as parent; the source's seal completes the run."""

        from .context import DeliveredInputs

        process = self.instance.process
        bundle = any(getattr(item, "select_bundle", False) for item in self.declared.input_specs)
        evaluated = 0
        while not self.cancel.is_set():
            try:
                identity, params = self.deliveries.get(timeout=0.2)
            except queue.Empty:
                continue
            kind = params.get("kind")
            if kind != "data":
                process.reply(identity, {})
                if kind == "sealed":
                    if evaluated == 0:
                        raise PluginError("failed", "the source ended before it published")
                    return {"evaluated": evaluated}
                if kind == "retired":
                    # A source that moved on is the source's lifecycle, not a fault (v2: cancelled).
                    self.cancel.set()
                    raise PluginError("cancelled", "the source run was replaced")
                raise PluginError("failed", str(params.get("error") or "the processor input failed"))
            # A bundle member is decoded when the node reads it; the rest are let go after the commit.
            inputs = DeliveredInputs(params["inputs"], self.refs, process.release_lease)
            try:
                self.refs.adopt(params)
                outputs = node.evaluate_inputs(inputs) if bundle else node.evaluate(inputs[params["anchor"]])
                if not isinstance(outputs, Mapping) or not outputs:
                    raise TypeError("processor evaluate() must return a non-empty mapping")
                if evaluated == 0 and callable(getattr(node, "describe_run", None)):
                    self.context.set_run_record(node.describe_run(inputs))
                self.context.commit_live(outputs, source_publication=inputs.publication(params["anchor"]))
            except BaseException as error:
                process.reply(identity, error=error)
                raise
            finally:
                inputs.release_unread()
            evaluated += 1
            del inputs, outputs
            process.reply(identity, {})
        raise PluginError("cancelled", "the run was stopped")

    exit_failure: BaseException | None = None

    def _exit_writer(self, status: str, error: BaseException) -> None:
        """Call the Task's partial exit writer once; what it could not save is part of how the run ends."""

        writer, self.partial_exit_writer = self.partial_exit_writer, None
        if writer is None:
            return
        try:
            writer(status, error)
        except BaseException as failure:  # noqa: BLE001 -- reported with the run's end
            self.exit_failure = failure

    @staticmethod
    def proxies(instance: Instance, devices: dict[str, Any], *, run: str | None) -> dict[str, Any]:
        built = {}
        for argument, info in devices.items():
            kind = _proxy_class(tuple(info.get("capabilities", ())), bool(info.get("tunable")), tuple(info.get("reads") or ()))
            built[argument] = kind(instance, run, argument, info)
        return built

    def _main(self) -> None:
        process = self.instance.process
        outcome, result, error = "completed", None, None
        try:
            node = self.prepared_node
            if node is None:
                values = _projected(self.declared.project, self.params.get("values"))
                devices = self.params.get("devices", {})
                arguments: dict[str, Any] = dict(values)
                arguments.update(self.proxies(self.instance, devices, run=self.run_id))
                arguments.update({f"{argument}_key": info.get("instance") for argument, info in devices.items()})
                arguments.update(self.params.get("resources", {}))
                if self.params.get("source_signal") is not None:
                    arguments["source_signal"] = self.params["source_signal"]
                node = self.declared if self.declared.build is None else self.declared.build(**arguments)
            result = self._process(node) if self.declared.kind is NodeKind.PROCESSOR else node.execute(self.context)
            if self.stop_accepted or self.cancel.is_set():
                outcome = "stopped"
                if not self.stop_accepted:
                    self._exit_writer("stopped", PluginError("cancelled", "the run was stopped"))
        except _Invalid as invalid:
            outcome, error = "failed", error_payload(invalid.cause, kind="invalid")
        except BaseException as failure:  # noqa: BLE001 -- the terminal reports it
            # An error out of a run the operator stopped is the stop itself
            # (v2's NodeHost): it was interrupted mid-step, which is what Stop does.
            stopped = self.cancel.is_set() and not self.stop_accepted
            outcome = "stopped" if stopped else "failed"
            error = None if stopped else error_payload(failure)
            self._exit_writer(outcome, failure)
        if self.exit_failure is not None:
            error = error_payload(self.exit_failure)
        try:
            encoded, blocks = process.encode({"instance": self.instance.name, "run": self.run_id,
                                              "outcome": outcome, "result": plain_result(result), "error": error})
            process.send({"t": "evt", "op": "run.finished", "p": encoded, "blocks": blocks})
        finally:
            self.instance.runs.pop(self.run_id, None)
            self._end_deliveries()


def _refuse_delivery(process: Any, identity: int, params: dict[str, Any]) -> None:
    """Answer a delivery its run will not take (`stale_run`), letting go of its reads."""

    for read in (params.get("inputs") or {}).values():
        process.release_lease(int(read["lease"]))
    process.reply(identity, error=PluginError("stale_run", f"{params.get('run')} is not running here"))


def plain_result(value: Any) -> Any:
    try:
        return plain(value)
    except (TypeError, ValueError):
        return repr(value)


_SHARED_LOCK = threading.RLock()


def share(name: str, build: Callable[[], Any]) -> tuple[Any, Callable[[], None]]:
    """The one object of this name in this process, and a release for this user.

    State its instances share (the simulation world every virtual device stands
    in) is built by its first user and dropped when the last one releases it, so
    reopening them all starts from a fresh object.
    """

    owner = process()
    # Instances open on their own lanes at the same time (Init opens them in parallel):
    # finding or building the object is one step, or each builds its own world.
    with _SHARED_LOCK:
        entry = owner.shared.get(name)
        if entry is None:
            entry = owner.shared[name] = [build(), 0]
        entry[1] += 1
    released = threading.Event()

    def release() -> None:
        with _SHARED_LOCK:
            if released.is_set():
                return
            released.set()
            entry[1] -= 1
            if entry[1] == 0 and owner.shared.get(name) is entry:
                del owner.shared[name]

    return entry[0], release


#: The device being opened on this thread: its simulation parameters.
_OPENING = threading.local()


def simulation_parameters() -> dict[str, Any]:
    """The installation's simulation parameters, as given to the device being opened
    (its type declares ``simulation=True``); {} for any other."""

    return dict(getattr(_OPENING, "simulation", None) or {})


def allocate(shape: tuple[int, ...] | int, dtype: Any) -> Any:
    """A writable array in a shared block: sent by reference, never copied."""

    arena = process().arena
    if arena is None:
        raise RuntimeError("the plugin host has not completed its handshake")
    return arena.allocate(shape, dtype)


def main() -> int:
    global _PROCESS
    source = os.fdopen(os.dup(0), "rb")
    sink = os.fdopen(os.dup(1), "wb")
    # Plugin output must not reach the channel: fd 1 and sys.stdout become the log.
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    _PROCESS = Process(Channel(source, sink))
    return _PROCESS.serve()
