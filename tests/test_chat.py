import os
import hashlib
from unittest.mock import patch
import queue
import threading
import unittest
from types import SimpleNamespace
import torch
from fastapi.testclient import TestClient
import chat
from runtime import Runtime, Session
from chamber_worker import SteeringWorker

class MockDecoderLayer(torch.nn.Module):
    def forward(self, hidden):
        return hidden, None

class SteeringTest(unittest.TestCase):
    def test_batched_requests_have_independent_live_doses(self):
        model=torch.nn.Module()
        model.layers=torch.nn.ModuleList([MockDecoderLayer() for _ in range(19)])
        worker=SteeringWorker()
        worker.device='cpu'
        worker.model_runner=SimpleNamespace(model=model,input_batch=SimpleNamespace(
            req_ids=['a','b'],num_reqs=2),query_start_loc=SimpleNamespace(cpu=torch.tensor([0,3,4])))
        worker.chamber_setup(18,[1,1,1])
        worker.chamber_controls({'a':2,'b':5})
        hidden=model.layers[18](torch.zeros(4,3))[0]
        self.assertEqual(hidden.tolist(),[[0,0,0],[0,0,0],[2,2,2],[5,5,5]])
        worker.chamber_controls({'a':0,'b':1})
        hidden=model.layers[18](torch.zeros(4,3))[0]
        self.assertEqual(hidden.tolist(),[[0,0,0],[0,0,0],[0,0,0],[1,1,1]])

    def test_session_controls_are_isolated(self):
        a,b=Session(),Session()
        a.set_dose(4)
        self.assertEqual(b.dose,0)
        a.stop.set()
        self.assertFalse(b.stop.is_set())
        for dose in (-1,11,float('nan'),float('inf')):
            with self.assertRaises(ValueError):a.set_dose(dose)

class ChatTest(unittest.TestCase):
    def test_model_change_requires_password_on_server(self):
        digest=hashlib.sha256(b'test-secret').hexdigest()
        with patch.dict(os.environ, {'CHAMBER_MODEL_PASSWORD_SHA256':digest}), TestClient(chat.app) as client, patch.object(chat.runtime, 'load') as load:
            for headers in ({}, {'X-Model-Password':'wrong'}):
                self.assertEqual(client.post('/api/load/qwen3-4b', headers=headers).status_code,403)
                self.assertEqual(client.post('/api/model-unlock', headers=headers).status_code,403)
            load.assert_not_called()
            headers={'X-Model-Password':'test-secret'}
            self.assertEqual(client.post('/api/model-unlock', headers=headers).status_code,200)
            self.assertEqual(client.post('/api/load/qwen3-4b', headers=headers).status_code,200)
            load.assert_called_once_with('qwen3-4b')
        with patch.dict(os.environ, {'CHAMBER_MODEL_PASSWORD_SHA256':''}), TestClient(chat.app) as client:
            self.assertEqual(client.post('/api/model-unlock', headers=headers).status_code,403)

    def test_concurrent_sessions_stop_history_and_reset(self):
        original=chat.runtime
        runtime=Runtime();runtime.state='ready';runtime.key='qwen3-4b'
        chat.runtime=runtime
        observed=[]
        def generate(session,messages,events):
            observed.append(messages)
            events.put({'type':'token','text':'Ответ'})
            session.stop.wait(5)
            with runtime.lock:runtime.active-=1
            events.put({'type':'done','stopped':True})
        runtime.generate=generate
        try:
            with TestClient(chat.app) as client:
                self.assertEqual(client.get('/').status_code,200)
                with client.websocket_connect('/ws') as a,client.websocket_connect('/ws') as b:
                    for ws in (a,b):
                        ws.send_json({'type':'message','text':'Привет','model':'qwen3-4b','dose':2})
                        self.assertEqual(ws.receive_json()['type'],'started')
                        self.assertEqual(ws.receive_json()['type'],'token')
                    self.assertEqual(runtime.active,2)
                    a.send_json({'type':'dose','value':4})
                    self.assertEqual(a.receive_json()['value'],4)
                    a.send_json({'type':'stop'})
                    self.assertTrue(a.receive_json()['stopped'])
                    self.assertEqual(runtime.active,1)
                    a.send_json({'type':'message','text':'Дальше','model':'qwen3-4b'})
                    self.assertEqual(a.receive_json()['type'],'started')
                    self.assertEqual(a.receive_json()['type'],'token')
                    self.assertEqual(len(observed[-1]),4)
                    a.send_json({'type':'stop'});a.receive_json()
                    b.send_json({'type':'stop'});b.receive_json()
                    a.send_json({'type':'reset'})
                    self.assertEqual(a.receive_json()['type'],'reset')
                    self.assertEqual(runtime.active,0)
        finally:chat.runtime=original

if __name__=='__main__':unittest.main()
