"""Pylon lifetime regressions with exclusive SDK handles, no camera required."""
import pathlib
import sys
import threading
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'science' / 'src'))
from zlc_atom.devices.camera.pylon import PylonCameraAdapter, PylonCameraConfig


class Node:
    def SetValue(self, value): self.value = value
    def GetValue(self): return self.value


class Camera:
    def __init__(self, *, fail_close=False, fail_destroy=False, fail_open=False):
        self.calls = []; self.attached = True; self.grabbing = False
        self.fail_close = fail_close; self.fail_destroy = fail_destroy; self.fail_open = fail_open
        for name in ['TriggerSelector', 'TriggerMode', 'TriggerSource', 'TriggerActivation']:
            setattr(self, name, Node())
    def IsGrabbing(self): return self.grabbing
    def StopGrabbing(self): self.calls.append('stop'); self.grabbing = False
    def Open(self):
        self.calls.append('open')
        if self.fail_open: raise RuntimeError('SDK open failed')
    def Close(self):
        self.calls.append('close')
        if self.fail_close: raise RuntimeError('SDK close failed')
    def DestroyDevice(self):
        self.calls.append('destroy')
        if self.fail_destroy: raise RuntimeError('SDK destroy failed')
        self.attached = False


class PylonLifecycle(unittest.TestCase):
    def adapter(self, camera): return PylonCameraAdapter(PylonCameraConfig('123'), camera=camera)

    def test_close_destroys_handle_even_while_wrapper_is_retained_and_is_idempotent(self):
        camera = Camera(); camera.grabbing = True; adapter = self.adapter(camera)
        adapter.close(); adapter.close()
        self.assertEqual(camera.calls, ['stop', 'close', 'destroy'])
        self.assertFalse(camera.attached)
        self.assertIsNone(adapter._camera)

    def test_close_failure_still_destroys_and_reports_the_original_failure(self):
        camera = Camera(fail_close=True); adapter = self.adapter(camera)
        with self.assertRaisesRegex(RuntimeError, 'SDK close failed'): adapter.close()
        self.assertFalse(camera.attached)
        self.assertIsNone(adapter._camera)
        adapter.close()
        self.assertEqual(camera.calls, ['close', 'destroy'])

    def test_failed_destroy_retains_ownership_for_retry(self):
        camera = Camera(fail_destroy=True); adapter = self.adapter(camera)
        with self.assertRaisesRegex(RuntimeError, 'SDK destroy failed'): adapter.close()
        self.assertIs(adapter._camera, camera)
        camera.fail_destroy = False; adapter.close()
        self.assertFalse(camera.attached)

    def test_sdk_open_failure_destroys_the_candidate_before_returning(self):
        camera = Camera(fail_open=True)
        factory = types.SimpleNamespace(EnumerateDevices=lambda: [types.SimpleNamespace(GetSerialNumber=lambda: '123')],
                                        CreateDevice=lambda info: object())
        pylon = types.SimpleNamespace(TlFactory=types.SimpleNamespace(GetInstance=lambda: factory),
                                     InstantCamera=lambda device: camera)
        adapter = PylonCameraAdapter(PylonCameraConfig('123'))
        with patch.dict(sys.modules, {'pypylon': types.SimpleNamespace(pylon=pylon)}):
            with self.assertRaisesRegex(RuntimeError, 'SDK open failed'): adapter.open()
        self.assertEqual(camera.calls, ['open', 'close', 'destroy'])
        self.assertFalse(camera.attached)
        self.assertIsNone(adapter._camera)

    def test_receiver_is_joined_before_sdk_destruction(self):
        camera = Camera(); adapter = self.adapter(camera)
        def receive():
            adapter._worker_stop.wait(2)
            camera.calls.append('receiver finished')
        adapter._worker = threading.Thread(target=receive); adapter._worker.start()
        adapter.close()
        self.assertEqual(camera.calls, ['receiver finished', 'close', 'destroy'])

    def test_a_stuck_receiver_retains_the_sdk_handle_until_a_retry(self):
        camera = Camera(); adapter = self.adapter(camera)
        adapter._worker = types.SimpleNamespace(join=lambda **kwargs: None, is_alive=lambda: True)
        with self.assertRaisesRegex(RuntimeError, 'camera retained'): adapter.close()
        self.assertIs(adapter._camera, camera)
        self.assertEqual(camera.calls, [])
        adapter._worker.is_alive = lambda: False
        adapter.close()
        self.assertFalse(camera.attached)

    def test_failed_wrapper_construction_releases_the_factory_handle(self):
        device = object(); destroyed = []
        factory = types.SimpleNamespace(EnumerateDevices=lambda: [types.SimpleNamespace(GetSerialNumber=lambda: '123')],
                                        CreateDevice=lambda info: device, DestroyDevice=destroyed.append)
        def attach(device): raise RuntimeError('SDK attach failed')
        pylon = types.SimpleNamespace(TlFactory=types.SimpleNamespace(GetInstance=lambda: factory), InstantCamera=attach)
        adapter = PylonCameraAdapter(PylonCameraConfig('123'))
        with patch.dict(sys.modules, {'pypylon': types.SimpleNamespace(pylon=pylon)}):
            with self.assertRaisesRegex(RuntimeError, 'SDK attach failed'): adapter.open()
        self.assertEqual(destroyed, [device])
        self.assertIsNone(adapter._camera)


if __name__ == '__main__': unittest.main()
