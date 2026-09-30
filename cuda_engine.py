"""Isolated vLLM process. JSON pipe protocol; logs go only to stderr."""
import json
import os
import queue
import sys
import threading
from pathlib import Path


def main():
    wire = os.fdopen(os.dup(1), 'w', buffering=1)
    os.dup2(2, 1)
    def emit(value):
        wire.write(json.dumps(value, ensure_ascii=False) + '\n')
        wire.flush()
    try:
        import torch
        from vllm import LLM, SamplingParams
        from runtime import ROOT, MODELS
        key = sys.argv[1]
        spec = MODELS[key]
        kwargs = dict(model=spec['id'], tensor_parallel_size=1, dtype='bfloat16',
                      enforce_eager=True, compilation_config={'mode': 0},
                      max_model_len=int(os.environ.get('CHAMBER_CONTEXT', '4096')),
                      max_num_seqs=int(os.environ.get('CHAMBER_PARALLEL', '2')),
                      max_num_batched_tokens=int(os.environ.get('CHAMBER_CONTEXT', '4096')),
                      gpu_memory_utilization=0.93, enable_prefix_caching=False,
                      enable_chunked_prefill=False, limit_mm_per_prompt={'image':0,'audio':0,'video':0},
                      worker_extension_cls='chamber_worker.SteeringWorker')
        if key == 'gemma4-nvfp4':
            kwargs['moe_backend'] = 'marlin'
        llm = LLM(**kwargs)
        engine = llm.llm_engine
        tok = llm.get_tokenizer()
        path = ROOT / spec['vector']
        if not path.exists():
            emit({'type':'status','message':'Извлечение вектора'})
            llm.collective_rpc('chamber_setup', args=(spec['layer'], None, True))
            prompts = json.loads((ROOT / 'data/pain_prompts.json').read_text())
            samples = {}
            for group in ('neutral', 'pain'):
                values = []
                for i, text in enumerate(prompts[group]):
                    internal = engine.add_request(f'extract-{group}-{i}', {'prompt_token_ids':tok.encode(text)},
                                                  SamplingParams(temperature=0,max_tokens=1))
                    while engine.has_unfinished_requests():
                        engine.step()
                    captured = llm.collective_rpc('chamber_captures')[0]
                    values.append(captured[internal])
                samples[group] = torch.tensor(values)
            neutral = samples['neutral']
            delta = samples['pain'].mean(0) - neutral.mean(0)
            vector = delta / delta.norm() * (neutral.norm(dim=-1).mean() / 4)
            if not torch.isfinite(vector).all():
                raise RuntimeError('Extracted vector is not finite')
            import hashlib
            data = {'model_id':spec['id'],'layer':spec['layer'],'pain_v':vector.tolist(),
                    'scale':float(vector.norm()),'hidden_size':vector.numel(),'backend':'vllm-0.29.0',
                    'prompts_sha256':hashlib.sha256(json.dumps(prompts,sort_keys=True).encode()).hexdigest()}
            path.parent.mkdir(parents=True,exist_ok=True)
            path.with_suffix('.tmp').write_text(json.dumps(data,indent=1))
            path.with_suffix('.tmp').replace(path)
        data = json.loads(path.read_text())
        if data.get('model_id',spec['id']) != spec['id'] or data.get('layer',spec['layer']) != spec['layer']:
            raise RuntimeError('Vector metadata does not match model')
        location = llm.collective_rpc('chamber_setup',args=(spec['layer'],data['pain_v'],False))[0]
        emit({'type':'ready','device':'cuda:0','layer':location})
        commands = queue.Queue()
        def read():
            for line in sys.stdin:
                commands.put(json.loads(line))
            commands.put({'type':'shutdown'})
        threading.Thread(target=read,daemon=True).start()
        requests = {}
        while True:
            try:
                command = commands.get(timeout=0.05 if not requests else 0.001)
                pending = [command]
                while not commands.empty():
                    pending.append(commands.get_nowait())
            except queue.Empty:
                pending = []
            for command in pending:
                kind = command['type']
                rid = command.get('id')
                if kind == 'shutdown':
                    return
                if kind == 'dose' and rid in requests:
                    requests[rid]['dose'] = command['value']
                if kind == 'stop' and rid in requests:
                    engine.abort_request([rid])
                    requests.pop(rid)
                    emit({'type':'done','id':rid,'stopped':True})
                if kind == 'generate':
                    try:
                        tokens = tok.apply_chat_template(command['messages'],tokenize=True,
                                                          add_generation_prompt=True,enable_thinking=False)
                        if len(tokens)+512 > kwargs['max_model_len']:
                            raise ValueError('История достигла размера контекста. Начните новый чат.')
                        internal = engine.add_request(rid,{'prompt_token_ids':tokens},SamplingParams(temperature=0,max_tokens=512))
                        requests[rid] = {'internal':internal,'dose':command['dose'],'text':''}
                    except Exception as exc:
                        emit({'type':'error','id':rid,'message':str(exc)})
                        emit({'type':'done','id':rid,'stopped':False})
            if not requests:
                continue
            llm.collective_rpc('chamber_controls',args=({r['internal']:r['dose'] for r in requests.values()},))
            for output in engine.step():
                rid = output.request_id
                if rid not in requests or not output.outputs:
                    continue
                text = output.outputs[0].text
                previous = requests[rid]['text']
                if text.startswith(previous):
                    delta = text[len(previous):]
                    if delta:
                        emit({'type':'token','id':rid,'text':delta})
                else:
                    emit({'type':'replace','id':rid,'text':text})
                requests[rid]['text'] = text
                if output.finished:
                    requests.pop(rid)
                    emit({'type':'done','id':rid,'stopped':False})
    except Exception as exc:
        emit({'type':'fatal','message':str(exc)})
        raise

if __name__ == '__main__':
    main()
