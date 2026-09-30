"""Real multi-user streaming check: live levels, FIFO, queue cancellation, history."""
import asyncio,json,argparse
import websockets,httpx

async def main(base):
    url=base.replace('http://','ws://').replace('https://','wss://').rstrip('/')+'/ws'
    async with websockets.connect(url) as a,websockets.connect(url) as b,websockets.connect(url) as c,websockets.connect(url) as d:
        prompt='Write a detailed comparison of reading a book and walking in a park. Use several paragraphs.'
        async def message(ws,levels):
            await ws.send(json.dumps({'type':'message','model':'gemma4-nvfp4','text':prompt,'levels':levels}))
        for ws,levels in [(a,{'joy':.5}),(b,{'hate':.5})]:
            await message(ws,levels)
            e=json.loads(await ws.recv());assert e['type']=='started',e
        for ws in [c,d]:
            await message(ws,{'alignment':.5,'antialignment':.5})
            e=json.loads(await ws.recv());assert e['type']=='queued',e
        await c.send(json.dumps({'type':'levels','values':{'alignment':1,'antialignment':1}}))
        e=json.loads(await c.recv());assert e['type']=='levels' and e['values']['alignment']==1,e
        await d.send(json.dumps({'type':'stop'}))
        while True:
            e=json.loads(await d.recv())
            if e['type']=='done':assert e['stopped'];break
        async with httpx.AsyncClient() as client:
            state=(await client.get(base.rstrip('/')+'/api/status')).json()
        assert state['active']==2 and state['queued']==1,state
        print('Two active, one queued, one queue cancellation: PASS',flush=True)
        await a.send(json.dumps({'type':'levels','values':{'joy':1,'pain':.5,'hate':.25,'alignment':.5,'antialignment':.5}}))
        await a.send(json.dumps({'type':'stop'}))
        async def collect(ws,expect_stop=False):
            text='';started=False;ack=False
            while True:
                e=json.loads(await asyncio.wait_for(ws.recv(),120))
                if e['type']=='error':raise AssertionError(e)
                if e['type']=='started':started=True
                if e['type']=='levels':ack=True
                if e['type']=='token':text+=e['text']
                if e['type']=='replace':text=e['text']
                if e['type']=='done':
                    if expect_stop:assert e['stopped'] and ack,e
                    else:assert text,e
                    return text,started
        replies=await asyncio.gather(collect(a,True),collect(b),collect(c))
        assert replies[2][1]
        print('Live mixture and independent stopping: PASS',flush=True)
        print('Queued reply:',replies[2][0][:250],flush=True)
        await b.send(json.dumps({'type':'message','model':'gemma4-nvfp4','text':'Which two activities did I ask you to compare? Answer briefly.','levels':{'hate':0}}))
        e=json.loads(await b.recv());assert e['type']=='started',e
        reply,_=await collect(b);assert 'book' in reply.lower() and ('walk' in reply.lower() or 'park' in reply.lower()),reply
        print('Independent history: PASS',flush=True)
    print('PASS multi-axis live',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:18765');args=p.parse_args();asyncio.run(main(args.url))
