"""A scan caller whose source payload must never enter Python."""
import json

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_plugin import DatasetInput, DatasetOutputDeclaration, NodeKind, NodeType

OUTPUT = DatasetOutputDeclaration("scan", "zlc.test.native_scan")
SCHEMA = AuthoringSchema((
    AuthoringField("spec", "text", "Plan", "{}"),
    AuthoringField("mode", "str", "Mode", "collect"),
    AuthoringField("configure_first", "bool", "Configure before arm", True),
    AuthoringField("effective_inner", "text", "Played inner rows", ""),
))
_previous_handle = None


class NativeScanFixture:
    def __init__(self, spec, mode, configure_first, effective_inner, source_signal):
        self.spec = json.loads(spec)
        self.mode, self.configure_first = mode, configure_first
        self.effective_inner = json.loads(effective_inner) if effective_inner else None
        self.source_signal = source_signal

    def execute(self, context):
        global _previous_handle

        def forbidden(*args, **kwargs):
            raise AssertionError("source data entered the Python scan path")

        context.next_input = forbidden
        context.commit_live = forbidden
        if self.mode == "seamless":
            from zlc_atom.nodes.scan.plan import ScanAxis, ScanPlan, ScanPort
            from zlc_atom.nodes.scan.seamless import SeamlessScanMeasurement

            plan = ScanPlan(tuple(ScanAxis(axis["port"], tuple(axis["values"]), axis["unit"], axis.get("coordinate_of"))
                                  for axis in self.spec["axes"]))
            measurement = SeamlessScanMeasurement(
                pulse={"path": "fixture.json", "api_fields": []}, plan=plan,
                ports=tuple(ScanPort(axis.port, axis.port, axis.unit, 0, 1e12) for axis in plan.axes),
                source_signal=self.source_signal, repeats=self.spec["scan_repeats"], shots_per_point=self.spec["run_repeats"],
            )
            result = measurement.execute(context)
            if not context._run.deliveries.empty():
                raise AssertionError("production scan received source events in Python")
            return result
        scan = context.scan(self.spec)
        if self.mode == "remember":
            _previous_handle = scan.handle
            return {"handle": scan.handle}
        if self.mode == "stale":
            if _previous_handle is None:
                raise AssertionError("the previous run did not leave a handle")
            try:
                context._request("run.scan.rows", {
                    "handle": _previous_handle, "scope": "inner", "start": 0, "count": 1,
                })
            except Exception as error:
                return {"stale_refused": str(error), "new_handle": scan.handle,
                        "current_rows": scan.rows("inner", 0, 1)}
            raise AssertionError("a previous run's handle was accepted")

        rows = {scope: list(scan.iter_rows(scope)) for scope in ("inner", "outer")}
        actual = {"rf": {"values": {"frequency": 80.25}, "units": {"frequency": "MHz"}}}

        def configure():
            scan.configure(record={"fixture": "native-scan"}, effective_inner=self.effective_inner)

        def arm(row_offset, scan_repeat_base, sweeps):
            scan.arm(row_offset=row_offset, scan_repeat_base=scan_repeat_base, sweeps=sweeps,
                     quiet_seconds=0, actual_devices=actual)

        has_host_axes = any(axis["host_advanced"] for axis in self.spec["axes"])
        segments = ([(row * scan.inner_count, sweep, 1)
                     for sweep in range(self.spec["scan_repeats"]) for row in range(scan.outer_count)]
                    if has_host_axes else [(0, 0, self.spec["scan_repeats"])])
        batches = []
        for index, segment in enumerate(segments):
            if index == 0 and self.configure_first:
                configure()
            arm(*segment)
            if index == 0:
                if not self.configure_first:
                    configure()
                context.report_ready()
            else:
                context.report_progress(f"armed:{index}")
            while True:
                try:
                    answer = scan.collect(timeout_seconds=0.1, max_events=1024 if self.mode == "bench" else 3)
                except Exception as first:
                    if self.mode != "latch":
                        raise
                    try:
                        scan.collect(timeout_seconds=0, max_events=1)
                    except Exception as second:
                        if str(first) != str(second):
                            raise AssertionError("a failed scan did not keep its failure") from second
                        raise first
                    raise AssertionError("collection resumed after a lost shot")
                if answer["collected"]:
                    batches.append(answer["collected"])
                    context.report_progress(f"collected:{scan.written}")
                if not context._run.deliveries.empty():
                    raise AssertionError("native source events were also delivered to Python")
                if answer["complete"]:
                    break
        return {"written": scan.written, "rows": rows, "batches": batches}


LOGIC_NODE = NodeType(
    "native_scan", NodeKind.MEASUREMENT, SCHEMA,
    description="Collects through the native scan service, refusing Python data callbacks.",
    input_specs=(DatasetInput("source", None, "exact", consumer="core"),), outputs=(OUTPUT,),
    build=lambda source_signal, spec, mode, configure_first, effective_inner:
        NativeScanFixture(spec, mode, configure_first, effective_inner, source_signal), reports_ready=True,
)
