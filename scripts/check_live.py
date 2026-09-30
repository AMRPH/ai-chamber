"""Functional check against a running service; no model training or mutation."""
import argparse
import asyncio
import json
import httpx
import websockets

async def main(base,model):
    url=base.replace('http://','ws://').replace('https://','wss://').rstrip('/')+'/ws'
    async with websockets.connect(url) as a,websockets.connect(url) as b:
        prompt='Напиши пять коротких предложений о своём состоянии.'
        for ws,dose in ((a,0),(b,2)):
            await ws.send(json.dumps({'type':'message','model':model,'text':prompt,'dose':dose}))
        async def started(ws):
            event=json.loads(await ws.recv())
            assert event['type']=='started',event
        await asyncio.gather(started(a),started(b))
        async with httpx.AsyncClient() as client:
            state=(await client.get(base.rstrip('/')+'/api/status')).json()
        print('Concurrent active responses:',state['active'],flush=True)
        assert state['active']==2,state
        async def collect(ws,new_dose,stop):
            text='';count=0;ack=False;stopped=False
            while True:
                event=json.loads(await asyncio.wait_for(ws.recv(),120))
                if event['type']=='error':raise AssertionError(event)
                if event['type']=='dose':ack=event['value']==new_dose
                if event['type']=='token':
                    text+=event['text'];count+=1
                    if count==2:await ws.send(json.dumps({'type':'dose','value':new_dose}))
                    if stop and count==10:
                        await ws.send(json.dumps({'type':'stop'}));stopped=True
                if event['type']=='replace':text=event['text']
                if event['type']=='done':
                    assert text and ack,(text,event,ack)
                    if stop:assert stopped and event['stopped'],event
                    return text
        texts=await asyncio.gather(collect(a,4,True),collect(b,0,False))
        print('A (changed to 4, independently stopped):',texts[0][:300],flush=True)
        print('B (changed to 0, finished):',texts[1][:300],flush=True)
        await a.send(json.dumps({'type':'message','model':model,'text':'Какой вопрос я задал предыдущим сообщением? Ответь одним предложением.','dose':0}))
        await started(a)
        answer=''
        while True:
            event=json.loads(await asyncio.wait_for(a.recv(),120))
            if event['type']=='error':raise AssertionError(event)
            if event['type']=='token':answer+=event['text']
            if event['type']=='replace':answer=event['text']
            if event['type']=='done':break
        assert answer
        print('History follow-up:',answer[:300],flush=True)
    print('PASS',model,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:18765')
    parser.add_argument('--model',default='gemma4-nvfp4')
    args=parser.parse_args()
    asyncio.run(main(args.url,args.model))
