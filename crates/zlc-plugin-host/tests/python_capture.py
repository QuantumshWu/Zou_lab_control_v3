"""Native capture ABI and Python record ownership; requires the built capture DLL."""
import pathlib
import sys
import types
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "science" / "src"))
from zlc_atom.devices import RecordQueue
from zlc_plugin.capture import CaptureLedger, terminal_evidence, validate_group

class Capture(unittest.TestCase):
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
        with self.assertRaisesRegex(RuntimeError, "join deadline"): queue.finish()
        self.assertTrue(queue.armed)
        self.assertIs(queue._worker, worker)
        with self.assertRaises(RuntimeError): queue.close()
        worker.is_alive = lambda: False
        self.assertEqual(queue.finish(), 0)
        queue.close()

    def test_native_groups_and_terminal_evidence_distinguish_complete_and_partial_cycles(self):
        validate_group([4, 5], start=4, size=2)
        with self.assertRaises(RuntimeError): validate_group([4, 6], start=4, size=2)
        terminal = types.SimpleNamespace(produced_count=5, source_stopped=True, joined=True)
        with self.assertRaises(RuntimeError): terminal_evidence(terminal, expected=4)
        proof = terminal_evidence(terminal, expected=4, stopped=True, group_size=2)
        self.assertEqual((proof["complete_groups"], proof["partial_records"]), (2, 1))
        ledger = CaptureLedger()
        self.assertFalse(ledger.call("append_group", size=2)["complete"])
        self.assertTrue(ledger.call("append_group", size=2)["complete"])

if __name__ == "__main__": unittest.main()
