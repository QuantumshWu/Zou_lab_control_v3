"""Lifecycle races in the real Python SDK, with exclusive stand-in device handles."""
import pathlib
import sys
import threading
import unittest
from unittest.mock import patch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'science' / 'src'))
from zlc_plugin import host

class Owner:
    _request = host.Process._request
    _event = host.Process._event
    def __init__(self):
        self.instances = {}
        self.answers = {}
        self.events = {}
        self.hook = lambda identity: None
    def decode(self, value): return value
    def reply(self, identity, result=None, **kwargs):
        self.answers[identity] = (result, kwargs)
        self.hook(identity)
        self.events.setdefault(identity, threading.Event()).set()
    def send(self, identity, op, name='n100_com3'):
        self.events.setdefault(identity, threading.Event())
        self._request({'id':identity,'op':op,'p':{'instance':name}})
    def wait(self, identity):
        return self.events.setdefault(identity, threading.Event()).wait(2)
    def cleanup(self):
        for instance in list(self.instances.values()):
            instance.lane.put(None)
            instance._thread.join(2)
            instance.shutdown()

class Lifecycle(unittest.TestCase):
    def test_close_reply_is_a_barrier_before_immediate_reinit(self):
        for name in ['n100_com3', 'pylon_camera']:
            with self.subTest(name=name):
                owner=Owner(); held=set()
                def opened(instance, params):
                    if name in held: raise RuntimeError('device is already in use')
                    held.add(name)
                    instance.adapter=object()
                    instance.closer=lambda: held.remove(name)
                    return {}
                with patch.dict(host._HANDLERS, {'open':opened}):
                    try:
                        owner.send(1,'open',name);self.assertTrue(owner.wait(1))
                        owner.hook=lambda identity: owner.send(3,'open',name) if identity==2 else None
                        owner.send(2,'close',name);self.assertTrue(owner.wait(2))
                        self.assertTrue(owner.wait(3),'re-Init was queued on a dead instance lane')
                        self.assertNotIn('error',owner.answers[3][1])
                        self.assertIn(name, owner.instances)
                        owner.hook=lambda identity: None
                        owner.send(4,'close',name);self.assertTrue(owner.wait(4))
                        self.assertFalse(held)
                    finally: owner.cleanup()

    def test_failed_open_releases_handle_before_error_and_allows_retry(self):
        owner=Owner(); held=set(); count=0
        def opened(instance, params):
            nonlocal count
            count+=1
            if held: raise RuntimeError('device is already in use')
            held.add('port');instance.adapter=object();instance.closer=lambda:held.clear()
            if count==1: raise RuntimeError('initialization verification failed')
            return {}
        with patch.dict(host._HANDLERS, {'open':opened}):
            try:
                owner.hook=lambda identity: owner.send(2,'open') if identity==1 else None
                owner.send(1,'open');self.assertTrue(owner.wait(1))
                self.assertTrue(owner.wait(2),'failed Init poisoned the next Init')
                self.assertNotIn('error',owner.answers[2][1])
            finally: owner.cleanup()

    def test_failed_close_keeps_the_handle_for_cleanup_retry(self):
        owner=Owner(); instance=host.Instance(owner,'n100_com3');owner.instances[instance.name]=instance
        device=object();instance.adapter=device; calls=[]
        def close():
            calls.append(1)
            if len(calls)==1: raise RuntimeError('reader has not stopped yet')
        instance.closer=close
        try:
            owner.send(1,'close');self.assertTrue(owner.wait(1))
            self.assertIs(instance.adapter,device)
            self.assertIs(owner.instances.get(instance.name),instance)
            owner.send(2,'close');self.assertTrue(owner.wait(2))
            self.assertEqual(len(calls),2)
            self.assertNotIn(instance.name,owner.instances)
        finally: owner.cleanup()


    def test_abandoned_operation_keeps_exclusive_ownership_until_cleanup(self):
        owner=Owner(); entered=threading.Event(); release=threading.Event(); held=set()
        def opened(instance, params):
            held.add('port');instance.adapter=object();instance.closer=lambda: held.clear()
            entered.set();release.wait(2)
            return {}
        with patch.dict(host._HANDLERS, {'open':opened}):
            try:
                owner.send(1,'open');self.assertTrue(entered.wait(2))
                old=owner.instances['n100_com3']
                owner._event({'op':'abandon','p':{'instance':'n100_com3'}})
                owner.send(2,'open');self.assertTrue(owner.wait(2))
                self.assertIn('still releasing',str(owner.answers[2][1]['error']))
                self.assertEqual(held,{'port'})
                release.set();old._thread.join(2)
                self.assertFalse(held)
                self.assertNotIn('n100_com3',owner.instances)
                owner.send(3,'open');self.assertTrue(owner.wait(3))
                self.assertNotIn('error',owner.answers[3][1])
            finally: release.set();owner.cleanup()

    def test_invalid_open_is_retired_before_its_error_reply(self):
        owner=Owner(); attempts=[]
        def opened(instance, params):
            attempts.append(instance)
            if len(attempts)==1: raise host._Invalid(ValueError('bad settings'))
            return {}
        with patch.dict(host._HANDLERS, {'open':opened}):
            try:
                owner.hook=lambda identity: owner.send(2,'open') if identity==1 else None
                owner.send(1,'open');self.assertTrue(owner.wait(2))
                self.assertEqual(owner.answers[1][1]['kind'],'invalid')
                self.assertIsNot(attempts[0],attempts[1])
            finally: owner.cleanup()

if __name__=='__main__': unittest.main()
