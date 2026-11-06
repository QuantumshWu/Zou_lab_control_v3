"""Local driver queues and coarse native capture evidence; no capture DLL."""
import pathlib
import sys
import types
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "science" / "src"))
from zlc_atom.devices import RecordQueue
from zlc_plugin.capture import terminal_evidence, validate_batch, validate_group
from unittest import mock

class Capture(unittest.TestCase):
    def test_driver_queue_needs_neither_native_process_nor_dll(self):
        with mock.patch.dict("os.environ", {"ZLC_CAPTURE_LIBRARY": "missing.dll", "ZLC_NATIVE_EXECUTABLE": "missing.exe"}), \
             mock.patch("zlc_plugin.services.call", side_effect=AssertionError("queue must not call native services")):
            queue = RecordQueue("local", join_timeout_seconds=.1)
            queue.arm(2, buffer_record_count=2)
            records = [types.SimpleNamespace(source_ordinal=index) for index in range(2)]
            for record in records: self.assertTrue(queue.push(record))
            self.assertEqual((queue.produced_count, queue.pending_count), (2, 2))
            self.assertEqual(queue.read(2, timeout=0, exact=True), records)
            self.assertEqual(queue.finish(), 2)
            queue.close()

    def test_failed_worker_construction_rolls_back_the_owned_queue(self):
        queue = RecordQueue("local", join_timeout_seconds=.1)
        def failed(_stop): raise RuntimeError("worker construction failed")
        with self.assertRaisesRegex(RuntimeError, "worker construction"):
            queue.arm(None, buffer_record_count=2, worker=failed)
        self.assertFalse(queue.armed)
        queue.arm(1, buffer_record_count=1)
        self.assertTrue(queue.push(types.SimpleNamespace(source_ordinal=0)))
        self.assertEqual(queue.finish(), 1)

    def test_overflow_preserves_owned_records_and_reports_first_failure(self):
        queue = RecordQueue("test", join_timeout_seconds=.1)
        queue.arm(None, buffer_record_count=1)
        record = types.SimpleNamespace(source_ordinal=0)
        self.assertTrue(queue.push(record))
        self.assertFalse(queue.push(types.SimpleNamespace(source_ordinal=1)))
        queue.fail(RuntimeError("later"))
        self.assertIn("overflow", str(queue.failure))
        self.assertEqual((queue.pending_count, queue.produced_count), (1, 1))
        self.assertIs(queue._queue[0], record)
        with self.assertRaisesRegex(RuntimeError, "overflow"): queue.read(1, timeout=0, exact=True)
        with self.assertRaisesRegex(RuntimeError, "overflow"): queue.finish()
        queue.close()
        queue.arm(1, buffer_record_count=1)
        self.assertTrue(queue.push(types.SimpleNamespace(source_ordinal=0)))
        self.assertFalse(queue.accepting)
        self.assertEqual(queue.finish(), 1)
        self.assertEqual(len(queue.read(1, timeout=0, exact=True)), 1)

    def test_failed_join_does_not_invent_a_terminal_or_release_its_worker(self):
        queue = RecordQueue("test", join_timeout_seconds=.01)
        queue.arm(None, buffer_record_count=1)
        worker = types.SimpleNamespace(join=lambda timeout: None, is_alive=lambda: True)
        queue._worker = worker
        with self.assertRaisesRegex(RuntimeError, "did not stop producing"): queue.finish()
        self.assertTrue(queue.armed)
        self.assertIs(queue._worker, worker)
        with self.assertRaises(RuntimeError): queue.close()
        worker.is_alive = lambda: False
        self.assertEqual(queue.finish(), 0)
        queue.close()

    def test_camera_monitor_validates_one_batch_and_uses_returned_group_boundaries(self):
        from zlc_atom.nodes.camera import measurement
        records = [types.SimpleNamespace(source_ordinal=i) for i in range(5)]
        node = object.__new__(measurement.CameraMeasurementNode)
        node._request = types.SimpleNamespace(frames_per_cycle=2, photoelectrons=False)
        node._actual_working_point = types.SimpleNamespace(electrons_per_count=None)
        node._record_received = None
        node._next_record_ordinal = 0
        node._capture_group_ends = frozenset()
        reads = []
        batches = iter([records, ()])
        def read(count, **options): reads.append(count); return next(batches)
        node.camera = types.SimpleNamespace(read_frame_records=read)
        output = []
        capture = measurement.MonitorCapture(node.camera, node=node, commit_live=lambda data: output.append(data))
        with mock.patch.object(measurement, "validate_batch", return_value={"next": 5, "group_ends": [1, 3]}) as validate, \
             mock.patch.object(measurement, "_monitor_cycle_output", side_effect=lambda node, cycle, revision: tuple(r.source_ordinal for r in cycle)):
            self.assertIs(capture.poll(), records[-1])
            self.assertIsNone(capture.poll())
        self.assertEqual(reads, [16, 16])
        self.assertEqual(validate.call_count, 1, "neither per-record grouping nor empty polls issue a service call")
        self.assertEqual([data["frames"] for data in output], [(0, 1), (2, 3)])
        self.assertEqual([r.source_ordinal for r in capture._pending_records], [4])

    def test_native_groups_and_terminal_evidence_distinguish_complete_and_partial_cycles(self):
        validate_group([4, 5], start=4, size=2)
        with self.assertRaises(RuntimeError): validate_group([4, 6], start=4, size=2)
        terminal = types.SimpleNamespace(produced_count=5, source_stopped=True, joined=True)
        with self.assertRaises(RuntimeError): terminal_evidence(terminal, expected=4)
        proof = terminal_evidence(terminal, expected=4, stopped=True, group_size=2)
        self.assertEqual((proof["complete_groups"], proof["partial_records"]), (2, 1))
        first = validate_batch([0], expected_start=0, group_size=2)
        self.assertEqual(first, {"next": 1, "group_ends": []})
        second = validate_batch([1, 2, 3], expected_start=first["next"], group_size=2)
        self.assertEqual(second, {"next": 4, "group_ends": [1, 3]})

if __name__ == "__main__": unittest.main()
