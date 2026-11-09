"""Durable artifact publication through the native helper and hosted SDK facade."""
import json
import pathlib
import sys
import tempfile
import types
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "science" / "src"))
from zlc_durable import atomic_write_text, atomic_write_file, durable_makedirs, DirectoryDurabilityError
from zlc_durable.durability import _native
from zlc_durable.workspace import unique_path
from zlc_plugin.artifacts import Artifacts


class ArtifactsTests(unittest.TestCase):
    def test_offline_atomic_writer_and_unique_names_use_native_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            nested = durable_makedirs(root / "nested")
            path = atomic_write_text(nested / "report.txt", "old")
            def failed(stream): stream.write(b"partial"); raise RuntimeError("writer failed")
            with self.assertRaisesRegex(RuntimeError, "writer failed"): atomic_write_file(path, failed)
            self.assertEqual(path.read_text(), "old")
            self.assertFalse(list(nested.glob(".*.tmp")))
            first = unique_path(nested, "sample", ".bin", writer=lambda target: target.write_bytes(b"first"))
            second = unique_path(nested, "sample", ".bin", writer=lambda target: target.write_bytes(b"second"))
            self.assertEqual((first.name, second.name), ("sample.bin", "sample-2.bin"))
            self.assertEqual((first.read_bytes(), second.read_bytes()), (b"first", b"second"))

    def test_hosted_scientific_writer_uses_stage_finalize_and_cleans_a_failed_writer(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            calls = []
            def request(op, params):
                self.assertEqual(op, "run.artifacts")
                calls.append(params["action"])
                action = params["action"]
                if action == "stage":
                    result = _native("stage", path=params["path"])
                    return {"ok": {"token": result["handle"], "path": result["path"], "target": result["target"]}}
                if action == "abort": return {"ok": _native("abort", handle=params["token"])}
                if action == "finalize": return {"ok": _native("finalize", handle=params["token"])}
                raise AssertionError(action)
            artifacts = Artifacts(types.SimpleNamespace(_request=request))
            path = artifacts.write_file(root / "science.bin", lambda stream: stream.write(b"science"))
            self.assertEqual(path.read_bytes(), b"science")
            self.assertEqual(calls, ["stage", "finalize"])
            def failed(stream): stream.write(b"partial"); raise ValueError("bad payload")
            with self.assertRaisesRegex(ValueError, "bad payload"):
                artifacts.write_file(root / "science.bin", failed)
            self.assertEqual(calls[-2:], ["stage", "abort"])
            self.assertEqual(path.read_bytes(), b"science")
            self.assertFalse(list(root.glob(".*.tmp")))

    def test_hosted_json_is_written_as_readable_text(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "summary.json"
            def request(op, params):
                self.assertEqual((op, params["action"]), ("run.artifacts", "write_text"))
                return {"ok": {"path": str(atomic_write_text(params["path"], params["text"]))}}
            written = Artifacts(types.SimpleNamespace(_request=request)).write_json(path, {"a": [1, 2]}, role="summary")
            self.assertEqual(written, path)
            self.assertEqual(json.loads(path.read_text()), {"a": [1, 2]})

    def test_published_but_not_durable_is_not_reported_as_no_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "visible.json"
            path.write_text("complete")
            context = types.SimpleNamespace(_request=lambda *args: {"error": {
                "type": "DirectoryDurabilityError", "message": "directory acknowledgement failed", "published": str(path)}})
            with self.assertRaises(DirectoryDurabilityError) as failure:
                Artifacts(context).write_text(path, "complete")
            self.assertEqual(failure.exception.published, path)
            self.assertEqual(path.read_text(), "complete")


if __name__ == "__main__": unittest.main()
