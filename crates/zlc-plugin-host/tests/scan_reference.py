"""Test-only scan RPC oracle. Production scan execution has no Python fallback."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

from zlc_atom.nodes.scan.dataset import ScanDatasetWriter
from zlc_atom.nodes.scan.plan import ScanAxis, ScanPlan, scan_rows, split_outer_axes
from zlc_plugin.scan import NativeScan


class ScanServiceOracle:
    def __init__(self, source, publish, record):
        self.source, self.publish, self.record = source, publish, record
        self.calls = []
        self.written = 0

    def request(self, op, params):
        self.calls.append((op, deepcopy(params)))
        if op == "run.scan.prepare":
            self.spec = params["spec"]
            self.plan = ScanPlan(tuple(ScanAxis(a["port"], tuple(a["values"]), a["unit"], a.get("coordinate_of")) for a in self.spec["axes"]))
            outer, inner = split_outer_axes(self.plan)
            self.outer, self.inner = scan_rows(outer), scan_rows(inner)
            return {"handle": "test-scan", "ports": [a.port for a in self.plan.axes],
                    "outer_width": len(outer), "inner_width": len(inner), "max_row_values": 65_536,
                    "outer_count": len(self.outer), "inner_count": len(self.inner), "point_count": self.plan.point_count,
                    "total_shots": self.plan.point_count * self.spec["scan_repeats"] * self.spec["run_repeats"]}
        if op == "run.scan.rows":
            rows = self.outer if params["scope"] == "outer" else self.inner
            return {"rows": rows[params["start"]:params["start"] + params["count"]]}
        if op == "run.scan.arm":
            self.arm = deepcopy(params)
            self.segment_written = 0
            self.segment_total = len(self.inner) * params["sweeps"] * self.spec["run_repeats"]
            return {}
        if op == "run.scan.configure":
            inner = params.get("effective_inner", getattr(self, "effective_inner", self.inner))
            rows = tuple(tuple(a) + tuple(b) for a in self.outer for b in inner)
            names = [a["identity"] for a in self.spec["axes"]]
            ports = [a.port for a in self.plan.axes]
            self.writer = ScanDatasetWriter(rows, [(a["identity"], a["unit"]) for a in self.spec["axes"]],
                scan_repeats=self.spec["scan_repeats"], run_repeats=self.spec["run_repeats"],
                axis_names=[a["label"] for a in self.spec["axes"]],
                coordinate_of=[ports.index(a.coordinate_of) if a.coordinate_of else None for a in self.plan.axes])
            record = deepcopy(params["record"])
            if hasattr(self, "applied"):
                record.setdefault("device_snapshots", {})["sequencer"] = deepcopy(self.applied)
            for index, axis in enumerate(record["plan"]["axes"]):
                mapping = {wanted[index]: actual[index] for wanted, actual in zip(self.plan.rows(), rows, strict=True)}
                axis["values"] = [mapping[value] for value in axis["values"]]
            self.record(record)
            return {"axes": record["plan"]["axes"]}
        if op == "run.scan.collect":
            count = min(params["max_events"], self.segment_total - self.segment_written)
            shots = self.spec["run_repeats"]
            for _ in range(count):
                sweep, rest = divmod(self.segment_written, len(self.inner) * shots)
                row, shot = divmod(rest, shots)
                value = self.source()
                output = self.writer.write(value, row=self.arm["row_offset"] + row,
                    scan_repeat=self.arm["scan_repeat_base"] + sweep, run_repeat=shot)
                if self.arm["actual_devices"]:
                    event = dict(output.event_record or {})
                    event["actual_devices"] = {**event.get("actual_devices", {}), **self.arm["actual_devices"]}
                    output = replace(output, event_record=event)
                self.publish(output)
                self.segment_written += 1
                self.written += 1
            return {"collected": count, "segment_written": self.segment_written, "segment_total": self.segment_total,
                    "written": self.written, "total": self.plan.point_count * self.spec["scan_repeats"] * shots,
                    "complete": self.segment_written == self.segment_total}
        raise AssertionError(op)


def attach_scan_oracle(context, source, publish, record):
    services = []
    def prepare(spec):
        service = ScanServiceOracle(source, publish, record)
        services.append(service)
        return NativeScan(SimpleNamespace(_request=service.request), spec)
    core = context.core
    def pulse(command, **params):
        reference = params.pop("scan_plan", None)
        if command != "pulse_run" or reference is None:
            return core(command, **params)
        service = services[-1]
        outer, inner = split_outer_axes(service.plan)
        carried = [(axis, value) for axis, value in zip(outer, service.outer[reference["outer_index"]])
                   if axis.port.startswith("pulse:param:")]
        fields = [axis.port[len("pulse:param:"):] for axis in inner] + [axis.port[len("pulse:param:"):] for axis, _ in carried]
        if fields:
            params["scan"] = {"fields": fields, "rows": [list(row) + [value for _, value in carried] for row in service.inner]}
        receipt = core(command, **params)
        service.applied = receipt["applied"]
        if inner:
            table = receipt.get("scan")
            if not table:
                raise RuntimeError("FIRE receipt is missing its loaded scan table")
            service.effective_inner = tuple(tuple(row[table["fields"].index(axis.port[len("pulse:param:"):])] for axis in inner)
                                            for row in table["rows"])
        return {**receipt, "scan": {"handle": reference["handle"], "row_count": len(service.inner), "fields": fields}}
    context.scan = prepare
    context.core = pulse
    return services
