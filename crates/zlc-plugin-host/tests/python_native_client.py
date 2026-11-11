"""Offline native calls use the existing desktop executable in headless mode."""
import io
import json
import pathlib
import struct
import sys
import types
import unittest
from unittest import mock
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "science" / "src"))
import zlc_plugin.services as zlc_native


class NativeClient(unittest.TestCase):
    def test_interrupted_reply_retires_helper_before_the_next_call(self):
        interruption = KeyboardInterrupt("stop this request")

        def worker(value, *, interrupt=False):
            payload = json.dumps({"ok": {"value": value}}).encode()

            class Replies(io.BytesIO):
                interrupted = False

                def read(self, size=-1):
                    if interrupt and not self.interrupted:
                        self.interrupted = True
                        raise interruption
                    return super().read(size)

            return types.SimpleNamespace(
                stdin=io.BytesIO(), stdout=Replies(struct.pack("<I", len(payload)) + payload),
                poll=lambda: None, wait=mock.Mock(return_value=0), kill=mock.Mock())

        first, second = worker(1, interrupt=True), worker(2)
        with mock.patch.object(zlc_native, "_helper", None), \
             mock.patch.object(zlc_native, "executable", return_value=pathlib.Path("zlc-desktop.exe")), \
             mock.patch.object(zlc_native.subprocess, "Popen", side_effect=[first, second]) as launched:
            with self.assertRaises(KeyboardInterrupt) as stopped:
                zlc_native._offline("example.first", {})
            self.assertIs(stopped.exception, interruption)
            self.assertIsNone(zlc_native._helper)
            self.assertTrue(first.stdin.closed)
            self.assertTrue(first.stdout.closed)
            self.assertEqual(zlc_native._offline("example.second", {}), {"value": 2})
            self.assertEqual(launched.call_count, 2)
            zlc_native._close()

    def test_offline_transport_launches_desktop_services_mode(self):
        response = json.dumps({"ok": {"next": 3, "group_ends": [2]}}).encode()
        worker = types.SimpleNamespace(stdin=io.BytesIO(), stdout=io.BytesIO(struct.pack("<I", len(response)) + response), poll=lambda: None)
        with mock.patch.object(zlc_native, "_helper", None), \
             mock.patch.object(zlc_native, "executable", return_value=pathlib.Path("zlc-desktop.exe")), \
             mock.patch.object(zlc_native.subprocess, "Popen", return_value=worker) as launched:
            answer = zlc_native._offline("capture.batch", {"expected_start": 0, "ordinals": [0, 1, 2], "group_size": 3})
            self.assertEqual(answer["next"], 3)
            self.assertEqual(launched.call_args.args[0], ["zlc-desktop.exe", "--plugin-services"])
            payload = worker.stdin.getvalue()
            size, = struct.unpack("<I", payload[:4])
            self.assertEqual(size, len(payload) - 4)
            self.assertEqual(json.loads(payload[4:])["operation"], "capture.batch")

    def test_an_old_standalone_helper_is_not_a_runtime_fallback(self):
        with mock.patch.dict("os.environ", {"ZLC_NATIVE_EXECUTABLE": ""}), \
             mock.patch.object(pathlib.Path, "is_file", lambda path: path.name == "zlc-native.exe"):
            with self.assertRaisesRegex(RuntimeError, "build zlc-desktop"):
                zlc_native.executable()

    def test_neutral_transport_import_does_not_load_the_sdk(self):
        import subprocess
        root = pathlib.Path(__file__).resolve().parents[3] / "science" / "src"
        process = subprocess.run([sys.executable, "-c",
            "import sys;sys.path.insert(0,sys.argv[1]);import zlc_plugin.services as zlc_native;assert 'zlc_plugin.host' not in sys.modules;assert zlc_native._helper is None", str(root)],
            capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.assertEqual(process.returncode, 0, process.stderr)


if __name__ == "__main__": unittest.main()
