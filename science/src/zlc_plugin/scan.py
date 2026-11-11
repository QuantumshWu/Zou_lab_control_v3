"""Run-scoped native scan service. Source data never crosses this facade."""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

_ROW_BATCH = 1_024


class NativeScan:
    """A factored plan and its core-owned placement/collection session."""

    def __init__(self, context: Any, spec: Mapping[str, Any]) -> None:
        self._context = context
        self._spec = dict(spec)
        prepared = context._request("run.scan.prepare", {"spec": self._spec})
        self.handle = prepared["handle"]
        self.ports = tuple(prepared["ports"])
        self.outer_count = int(prepared["outer_count"])
        self.inner_count = int(prepared["inner_count"])
        self.point_count = int(prepared["point_count"])
        self.total_shots = int(prepared["total_shots"])
        self.written = 0
        self._widths = {"outer": int(prepared["outer_width"]), "inner": int(prepared["inner_width"])}
        self._max_row_values = int(prepared["max_row_values"])

    def _request(self, operation: str, **params: Any) -> Any:
        return self._context._request("run.scan." + operation, {"handle": self.handle, **params})

    def rows(self, scope: str, start: int, count: int) -> tuple[tuple[float, ...], ...]:
        if scope not in self._widths:
            raise ValueError("scan row scope must be outer or inner")
        answer = self._request("rows", scope=scope, start=start, count=count)
        return tuple(tuple(row) for row in answer["rows"])

    def iter_rows(self, scope: str) -> Iterator[tuple[float, ...]]:
        if scope not in self._widths:
            raise ValueError("scan row scope must be outer or inner")
        count = self.outer_count if scope == "outer" else self.inner_count
        batch = min(_ROW_BATCH, max(1, self._max_row_values // max(1, self._widths[scope])))
        for start in range(0, count, batch):
            rows = self.rows(scope, start, min(batch, count - start))
            if len(rows) != min(batch, count - start):
                raise RuntimeError("native scan returned an incomplete row page")
            yield from rows

    def configure(self, *, record: Mapping[str, Any], output: str = "scan", effective_inner=None) -> Any:
        params = {"record": dict(record), "output": output}
        if effective_inner is not None:
            params["effective_inner"] = [[float(value) for value in row] for row in effective_inner]
        return self._request("configure", **params)

    def pulse_segment(self, outer_index: int = 0) -> dict[str, Any]:
        """A native plan reference, never an expanded board table."""
        if not 0 <= outer_index < self.outer_count:
            raise ValueError("the scan outer point is outside its plan")
        return {"handle": self.handle, "outer_index": int(outer_index)}

    def arm(self, *, row_offset: int, scan_repeat_base: int, sweeps: int,
            quiet_seconds: float, actual_devices: Mapping[str, Any], new_source: bool = False) -> Any:
        return self._request("arm", row_offset=row_offset, scan_repeat_base=scan_repeat_base,
                             sweeps=sweeps, quiet_seconds=quiet_seconds, actual_devices=dict(actual_devices), new_source=bool(new_source))

    def collect(self, *, timeout_seconds: float = 0.1, max_events: int = 1_024) -> Mapping[str, Any]:
        answer = self._request("collect", timeout_seconds=timeout_seconds, max_events=max_events)
        self.written = int(answer["written"])
        return answer


__all__ = ["NativeScan"]
