"""Native scan SDK/control boundary tests: no hardware or production Python collector."""
import itertools
import math
import pathlib
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "science" / "src"))
from zlc_atom.nodes.scan.plan import ScanAxis, ScanPlan, scan_row_count
from zlc_atom.nodes.scan.seamless import SeamlessScanMeasurement, estimate_scan_time
from zlc_plugin import DatasetInput
from zlc_plugin.context import Watched
from zlc_plugin.scan import NativeScan


class Service:
    """Only control replies: an input relay would fail these tests."""
    def __init__(self, *, silent=False):
        self.calls = []
        self.written = 0
        self.silent = silent

    def request(self, op, params):
        self.calls.append((op, params))
        if op == "run.scan.prepare":
            self.spec = params["spec"]
            axes = self.spec["axes"]
            self.count = math.prod(len(a["values"]) for a in axes if not a.get("coordinate_of"))
            self.total = self.count * self.spec["scan_repeats"] * self.spec["run_repeats"]
            return {"handle": 1, "ports": [a["port"] for a in axes], "outer_count": self.count,
                    "outer_width": len(axes), "inner_width": 0, "max_row_values": 65_536,
                    "inner_count": 1, "point_count": self.count, "total_shots": self.total}
        if op == "run.scan.rows":
            axes = self.spec["axes"]
            rows = []
            for index in range(params["start"], params["start"] + params["count"]):
                row = []
                for axis in reversed(axes):
                    index, at = divmod(index, len(axis["values"]))
                    row.append(axis["values"][at])
                rows.append(row[::-1])
            return {"rows": rows}
        if op == "run.scan.arm":
            self.segment_written = 0
            self.segment_total = params["sweeps"] * self.spec["run_repeats"]
            return {}
        if op == "run.scan.configure":
            self.configured = True
            return {"axes": self.spec["axes"]}
        if op == "run.scan.collect":
            assert self.configured
            count = 0 if self.silent else min(params["max_events"], self.segment_total - self.segment_written)
            self.segment_written += count
            self.written += count
            return {"collected": count, "segment_written": self.segment_written,
                    "segment_total": self.segment_total, "written": self.written,
                    "total": self.total, "complete": self.segment_written == self.segment_total}
        raise AssertionError(op)


def forbidden(*args, **kwargs):
    raise AssertionError("scan source values and publications must remain in the core")


class NativeScanTests(unittest.TestCase):
    def test_input_descriptor_selects_native_consumer_and_rejects_unsupported_modes(self):
        self.assertEqual(DatasetInput("source", None, "exact").consumer, "plugin")
        for options in ({"consumer": "other"}, {"consumer": "core", "select_bundle": True}):
            with self.assertRaises(ValueError): DatasetInput("source", None, "exact", **options)
        with self.assertRaises(ValueError): DatasetInput("source", None, "latest", consumer="core")
        from zlc_atom.nodes.seamless_scan.logic_node import LOGIC_NODE
        self.assertEqual(LOGIC_NODE.descriptor("test")["node"]["inputs"][0]["consumer"], "core")

    def test_factored_plan_pages_outer_rows_without_expanding_the_product(self):
        service = Service()
        axes = [{"port": f"manual:{n}", "identity": str(n), "label": str(n), "unit": "", "values": list(range(10)),
                 "coordinate_of": None, "host_advanced": True} for n in range(6)]
        native = NativeScan(types.SimpleNamespace(_request=service.request), {"axes": axes, "scan_repeats": 2, "run_repeats": 3})
        rows = list(itertools.islice(native.iter_rows("outer"), 1025))
        self.assertEqual(native.point_count, 1_000_000)
        self.assertEqual(native.total_shots, 6_000_000)
        self.assertEqual(rows[0], (0,) * 6)
        self.assertEqual(rows[-1], (0, 0, 1, 0, 2, 4))
        pages = [p for op, p in service.calls if op == "run.scan.rows"]
        self.assertEqual([p["count"] for p in pages], [1024, 1024])
        self.assertTrue(all(p["count"] * len(axes) <= 65_536 for p in pages))

    def test_configure_encodes_effective_rows_as_wire_arrays_and_arm_forwards_new_source(self):
        from zlc_plugin.wire import Encoder
        service = Service()
        spec = {"axes": [{"port": "manual:x", "identity": "x", "label": "x", "unit": "", "values": [1],
                          "coordinate_of": None, "host_advanced": True}], "scan_repeats": 1, "run_repeats": 1}
        native = NativeScan(types.SimpleNamespace(_request=service.request), spec)
        native.configure(record={}, effective_inner=((1.5, 2.5), (3.5, 4.5)))
        op, params = service.calls[-1]
        self.assertEqual(op, "run.scan.configure")
        # Tuples use a tagged wire object; Rust expects ordinary JSON arrays.
        self.assertIsInstance(params["effective_inner"], list)
        self.assertTrue(all(isinstance(row, list) for row in params["effective_inner"]))
        self.assertEqual(Encoder(None)(params)["effective_inner"], [[1.5, 2.5], [3.5, 4.5]])
        native.arm(row_offset=0, scan_repeat_base=0, sweeps=1, quiet_seconds=.05, actual_devices={}, new_source=True)
        self.assertIs(service.calls[-1][1]["new_source"], True)
        native.arm(row_offset=0, scan_repeat_base=0, sweeps=1, quiet_seconds=0, actual_devices={})
        self.assertIs(service.calls[-1][1]["new_source"], False)

    def test_count_and_record_do_not_request_cartesian_rows(self):
        plan = ScanPlan(tuple(ScanAxis(f"manual:{n}", tuple(range(100))) for n in range(5)))
        scan = SeamlessScanMeasurement(pulse={"path": "test.json", "api_fields": []}, plan=plan,
            ports=(), source_signal="source", repeats=2, shots_per_point=3)
        with mock.patch.object(ScanPlan, "rows", side_effect=AssertionError("expanded product")):
            self.assertEqual(scan_row_count(plan.axes), 10_000_000_000)
            estimate = estimate_scan_time(scan.pulse, plan, repeats=2, shots_per_point=3)
            self.assertEqual(estimate["total_shots"], 60_000_000_000)
            record = scan.run_record(sequencer={}, tunables={})
        self.assertEqual(len(record["plan"]["axes"]), 5)
        self.assertTrue(all(len(a["values"]) == 100 for a in record["plan"]["axes"]))

    def context(self, service):
        calls = []
        def core(command, **params):
            calls.append((command, params))
            if command == "pulse_run":
                self.assertNotIn("scan", params, "the production caller must not expand an inner table")
                reference = params["scan_plan"]
                return {"command_id": 1, "applied": {"config": {}},
                        "scan": {**reference, "fields": [], "row_count": 1}}
            return {"done": True, "status": "done", "elapsed_seconds": 0.0}
        def watch(command, **params):
            # As the core answers a watched read: at once, and nothing more is told (the fake's report is in).
            watched = Watched()
            watched.answered(core(command, _watch="0", **params))
            return watched
        context = types.SimpleNamespace(_request=service.request, core=core, watch=watch, cancel_requested=lambda: False,
            report_progress=lambda *a, **k: None, request_operator_input=lambda *a, **k: None,
            record_devices=lambda: None, next_input=forbidden, discard_inputs=forbidden,
            commit_live=forbidden, set_run_record=forbidden)
        context.scan = lambda spec: NativeScan(context, spec)
        return context, calls

    def test_native_batches_replace_the_per_shot_python_relay(self):
        service = Service()
        context, board_calls = self.context(service)
        scan = SeamlessScanMeasurement(pulse={"path": "test.json", "api_fields": []},
            plan=ScanPlan((ScanAxis("manual:knob", (1.0, 2.0)),)), ports=(), source_signal="source",
            repeats=1, shots_per_point=1025)
        with mock.patch.object(ScanPlan, "rows", side_effect=AssertionError("expanded product")):
            self.assertEqual(scan.execute(context), {"points": 2, "shots": 2050})
        arms = [p for op, p in service.calls if op == "run.scan.arm"]
        self.assertEqual([p["quiet_seconds"] for p in arms], [0.05, 0.0])
        self.assertEqual([p["row_offset"] for p in arms], [0, 1])
        self.assertEqual(len([op for op, _ in service.calls if op == "run.scan.collect"]), 4)
        self.assertEqual(len([op for op, _ in service.calls if op == "run.scan.configure"]), 1)
        self.assertEqual(len([op for op, _ in board_calls if op == "pulse_wait_done"]), 2)

    def test_missing_source_after_board_done_keeps_bounded_grace_and_stops_board(self):
        service = Service(silent=True)
        context, calls = self.context(service)
        scan = SeamlessScanMeasurement(pulse={"path": "test.json", "api_fields": []},
            plan=ScanPlan((ScanAxis("manual:knob", (1.0,)),)), ports=(), source_signal="source",
            repeats=1, shots_per_point=1)
        with mock.patch("zlc_atom.nodes.scan.seamless._SOURCE_GRACE_SECONDS", 0.0):
            with self.assertRaisesRegex(RuntimeError, "source delivered 0"):
                scan.execute(context)
        self.assertEqual(len([op for op, _ in calls if op == "pulse_stop"]), 2)


if __name__ == "__main__": unittest.main()
