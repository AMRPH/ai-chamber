import unittest
from types import SimpleNamespace
import torch
from fastapi.testclient import TestClient
import chat
from runtime import Runtime, Session
from chamber_worker import SteeringWorker
from research.analysis import direction, auc, layer_cv
import numpy as np

class MockDecoderLayer(torch.nn.Module):
    def forward(self,hidden):return hidden,None

class MultiTest(unittest.TestCase):
    def test_batched_mixture_covers_prompt_tokens_and_stays_per_request(self):
        model=torch.nn.Module();model.layers=torch.nn.ModuleList([MockDecoderLayer() for _ in range(2)])
        worker=SteeringWorker();worker.device='cpu'
        worker.model_runner=SimpleNamespace(model=model,input_batch=SimpleNamespace(req_ids=['a','b'],num_reqs=2),query_start_loc=SimpleNamespace(cpu=torch.tensor([0,2,3])))
        worker.chamber_setup_multi({'joy':{'layer':0,'vector':[1,0]},'hate':{'layer':0,'vector':[0,1]},'alignment':{'layer':1,'vector':[2,2]}})
        worker.chamber_controls({'a':{'joy':1,'hate':2},'b':{'alignment':.5}})
        hidden=model.layers[0](torch.zeros(3,2))[0];hidden=model.layers[1](hidden)[0]
        self.assertEqual(hidden.tolist(),[[1,2],[1,2],[1,1]])
        worker.chamber_controls({'a':{'hate':3},'b':{'joy':2}})
        hidden=model.layers[0](torch.zeros(3,2))[0];hidden=model.layers[1](hidden)[0]
        self.assertEqual(hidden.tolist(),[[0,3],[0,3],[2,0]])

    def test_level_updates_are_atomic_and_isolated(self):
        a,b=Session(),Session();a.set_levels({'joy':1,'alignment':2,'antialignment':3})
        self.assertEqual(b.levels['joy'],0)
        before=dict(a.levels)
        for values in ({'joy':2,'hate':4},{'unknown':1},{'hate':float('nan')},[]):
            with self.assertRaises((ValueError,TypeError)):a.set_levels(values)
            self.assertEqual(a.levels,before)

    def test_fifo_queue_and_cancel(self):
        original=chat.runtime;runtime=Runtime();runtime.state='ready';runtime.key='gemma4-nvfp4';runtime.parallel=1;chat.runtime=runtime
        seen=[]
        def generate(session,messages,events):
            seen.append(messages[-1]['content']);events.put({'type':'token','text':'OK'});session.stop.wait(5)
            with runtime.lock:runtime.active-=1
            events.put({'type':'done','stopped':True})
        runtime.generate=generate
        try:
            with TestClient(chat.app) as client,client.websocket_connect('/ws') as a,client.websocket_connect('/ws') as b,client.websocket_connect('/ws') as c:
                a.send_json({'type':'message','model':'gemma4-nvfp4','text':'A','levels':{'joy':1}})
                self.assertEqual(a.receive_json()['type'],'started');a.receive_json()
                for ws,text in [(b,'B'),(c,'C')]:
                    ws.send_json({'type':'message','model':'gemma4-nvfp4','text':text,'levels':{'hate':1}})
                    self.assertEqual(ws.receive_json()['type'],'queued')
                self.assertEqual(client.get('/api/status').json()['queued'],2)
                b.send_json({'type':'levels','values':{'hate':2}});self.assertEqual(b.receive_json()['values']['hate'],2)
                c.send_json({'type':'stop'});self.assertTrue(c.receive_json()['stopped'])
                a.send_json({'type':'stop'});a.receive_json()
                while b.receive_json()['type']!='started':pass
                self.assertEqual(b.receive_json()['type'],'token');b.send_json({'type':'stop'});b.receive_json()
                self.assertEqual(seen,['A','B']);self.assertEqual(runtime.active,0);self.assertFalse(chat.waiting)
        finally:chat.runtime=original

    def test_control_pca_denoising_and_group_cv(self):
        rng=np.random.default_rng(42);groups=np.repeat(np.arange(20),10);labels=np.tile([0]*5+[1]*5,20).astype(bool)
        noise=rng.normal(size=(200,3));noise[:,0]*=30
        noise[:,2]=labels*4+rng.normal(0,.1,200)
        v,k=direction(noise,labels)
        self.assertLess(abs(v[0]),.1);self.assertEqual(k,1);self.assertGreater(auc(labels,noise@v),.99)
        best,curves=layer_cv(np.stack([rng.normal(size=(200,3)),noise],axis=1),labels,groups)
        self.assertEqual(best,1);self.assertGreater(curves[1]['auc'],.99)

if __name__=='__main__':unittest.main()
