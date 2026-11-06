"""Headless hardware and virtual-device contracts."""

from collections import deque
from typing import Callable
import threading
import time


class RecordQueue:
    """Owned Python records over the process-local Rust capture ledger.

    Native policy decides admission, ordinals, capacity, readiness and terminal
    evidence. Python retains object lifetime, SDK workers and condition wakeups.
    """

    def __init__(self, what: str, *, join_timeout_seconds: float) -> None:
        from zlc_plugin.capture import CaptureLedger
        self._what = str(what)
        self._join_timeout = float(join_timeout_seconds)
        self._condition = threading.Condition()
        self._queue: deque[object] = deque()
        self._ledger = CaptureLedger(self._what)
        self._cause: BaseException | None = None
        self._worker: threading.Thread | None = None
        self._stop: threading.Event | None = None
        self._ready = threading.Event()

    def _state(self, key):
        with self._condition:
            return self._ledger.state()[key]

    @property
    def armed(self) -> bool:
        return self._state("armed")

    @property
    def accepting(self) -> bool:
        return self._state("accepting")

    @property
    def produced_count(self) -> int:
        return self._state("produced")

    @property
    def pending_count(self) -> int:
        return self._state("pending")

    @property
    def capacity(self) -> int:
        return self._state("capacity")

    @property
    def failure(self) -> BaseException | None:
        message = self._state("failure")
        return None if message is None else self._cause or RuntimeError(f"{self._what} {message}")

    def arm(self, records: int | None, *, buffer_record_count: int,
            worker: Callable[[threading.Event], threading.Thread] | None = None) -> None:
        with self._condition:
            self._ledger.call("arm", expected=None if records is None else int(records), capacity=int(buffer_record_count),
                              worker_alive=self._worker is not None and self._worker.is_alive())
            self._queue.clear()
            self._cause = None
            self._ready.clear()
            if worker is None:
                return
            stop = threading.Event()
            try:
                thread = worker(stop)
                self._worker, self._stop = thread, stop
                thread.start()
            except BaseException:
                self._worker = self._stop = None
                self._ledger.call("close", worker_alive=False)
                raise

    def mark_ready(self) -> None:
        with self._condition:
            self._ledger.call("ready")
            self._ready.set()

    def wait_ready(self, timeout: float) -> None:
        if not self._ready.wait(timeout):
            raise TimeoutError(f"{self._what} did not become ready")
        with self._condition:
            self._ledger.call("check_ready")

    def push(self, record: object) -> bool:
        with self._condition:
            ordinal = getattr(record, "source_ordinal", None)
            answer = self._ledger.call("push", ordinal=ordinal if type(ordinal) is int else None,
                                       observed_at_ns=time.time_ns())
            if answer["accepted"]:
                self._queue.append(record)
            else:
                self._ready.set()
            self._condition.notify_all()
            return bool(answer["accepted"])

    def fail(self, error: BaseException) -> None:
        with self._condition:
            if self._ledger.state()["failure"] is None:
                self._cause = error
            self._ledger.call("fail", message=str(error))
            self._ready.set()
            self._condition.notify_all()

    def read(self, n: int, *, timeout: float, exact: bool) -> list[object]:
        requested = int(n)
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while True:
                remaining = deadline - time.monotonic()
                answer = self._ledger.call("read", count=requested, exact=bool(exact), timed_out=remaining <= 0.0)
                if not answer["wait"]:
                    return [self._queue.popleft() for _ in range(answer["count"])]
                self._condition.wait(remaining)

    def finish(self) -> int:
        with self._condition:
            self._ledger.call("begin_finish")
            self._ready.set()
            worker, stop = self._worker, self._stop
            if stop is not None:
                stop.set()
            self._condition.notify_all()
        if worker is not None:
            worker.join(timeout=self._join_timeout)
        with self._condition:
            joined = worker is None or not worker.is_alive()
            if joined:
                self._worker = self._stop = None
            self._condition.notify_all()
            return int(self._ledger.call("finish", joined=joined)["produced"])

    def close(self) -> None:
        with self._condition:
            self._ledger.call("close", worker_alive=self._worker is not None and self._worker.is_alive())
            self._queue.clear()
            self._cause = None
            self._worker = self._stop = None
            self._ready.set()
            self._condition.notify_all()


def discovered(instance: str, parameters: dict) -> dict:
    """One hardware-scan find: the id (and name) it suggests, as v2's scan named it, and its parameters."""

    return {"instance_id": instance, "role": instance, "parameters": dict(parameters)}


def underscored(text: object) -> str:
    """``text`` with every character that is not a letter or digit as ``_`` (v2's scan names: ``COM3``, ``ASRL3::INSTR``)."""

    return "".join(c if c.isalnum() else "_" for c in str(text))


def serial_name(port: object) -> str:
    """A COM port as the core names one in ``serial:COM3``: upper-case, without ``\\\\.\\``."""

    text = str(port).strip()
    return (text[4:] if text.startswith("\\\\.\\") else text).upper()


def free_serial_ports(excluded: frozenset[str] = frozenset()) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """This machine's serial ports in order: (those free to open, those bound and passed over).

    ``excluded`` is the core's bound resources (``serial:COM3``); a port an
    open device or another row of the apparatus binds is never opened -- the
    pulse board's UART while its server starts, an IMU that is opening.
    """

    from serial.tools import list_ports

    bound = {serial_name(item.split(":", 1)[1]) for item in excluded if item.lower().startswith("serial:")}
    ports = [info.device for info in sorted(list_ports.comports(), key=lambda item: item.device)]
    return tuple(port for port in ports if serial_name(port) not in bound), tuple(port for port in ports if serial_name(port) in bound)


__all__ = ["RecordQueue", "discovered", "free_serial_ports", "serial_name", "underscored"]
