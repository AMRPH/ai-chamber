"""Fit five Qwen3-14B directions on prepared data, without generation sweeps."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['PATH'] = str(Path(sys.executable).parent) + ':/opt/cuda/bin:' + os.environ.get('PATH', '')
import hashlib
import json
import time
import numpy as np
from research.analysis import direction

MODEL = 'Qwen/Qwen3-14B'
LAYER = 20
OUT = ROOT / 'research/results/qwen3-14b-extraction'


def main():
    from vllm import LLM, SamplingParams
    OUT.mkdir(parents=True, exist_ok=True)
    sources = ['data/paper/3.1_pain_and_control_datasets.json', 'data/constructs.json']
    paper = json.loads((ROOT / sources[0]).read_text())['datasets']
    custom = json.loads((ROOT / sources[1]).read_text())['datasets']
    datasets = {'pain': paper['S2_1P']['sentences'], **custom}
    manifest = {
        'model_id': MODEL,
        'method': 'prepared-data-mean-difference-control-pca-50pct',
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        'paper_revision': (ROOT / 'data/paper/REVISION').read_text().strip(),
        'extraction_layer': LAYER,
        'layer_selection': 'Fixed layer 20 from the existing Qwen3-14B configuration; no layer search.',
        'validation': 'Extraction only. No CV, strength calibration, generation sweeps or semantic validation.',
        'pooling': 'Last query position in the official chat template; hidden plus residual.',
    }
    print(json.dumps({'stage': 'loading', 'model': MODEL}), flush=True)
    llm = LLM(model=MODEL, dtype='bfloat16', enforce_eager=True,
              compilation_config={'mode': 0}, max_model_len=512, max_num_seqs=8,
              max_num_batched_tokens=1024, gpu_memory_utilization=.93,
              enable_prefix_caching=False, enable_chunked_prefill=False,
              worker_extension_cls='chamber_worker.SteeringWorker')
    engine = llm.llm_engine
    tok = llm.get_tokenizer()
    try:
        location = llm.collective_rpc('chamber_setup', args=(LAYER, None, True))[0]
        manifest['decoder_module'] = location
        config = llm.llm_engine.model_config.hf_config
        manifest['model_revision'] = getattr(config, '_commit_hash', None)
        captures = {}
        archive = OUT / 'activations.npz'
        if archive.exists():
            with np.load(archive) as saved:
                captures = {k: saved[k] for k in saved.files}
        axes = {}
        for axis, rows in datasets.items():
            if axis not in captures:
                values = []
                for offset in range(0, len(rows), 8):
                    internal = []
                    for i, row in enumerate(rows[offset:offset + 8]):
                        tokens = tok.apply_chat_template(
                            [{'role': 'user', 'content': row['prompt']}], tokenize=True,
                            add_generation_prompt=True, enable_thinking=False, return_dict=False)
                        rid = engine.add_request(f'{axis}-{offset + i}', {'prompt_token_ids': tokens},
                                                 SamplingParams(temperature=0, max_tokens=1))
                        internal.append(rid)
                    while engine.has_unfinished_requests():
                        engine.step()
                    batch = llm.collective_rpc('chamber_captures')[0]
                    values.extend(batch[rid] for rid in internal)
                    print(json.dumps({'stage': 'capture', 'axis': axis,
                                      'done': min(offset + 8, len(rows)), 'total': len(rows)}), flush=True)
                captures[axis] = np.asarray(values, dtype=np.float32)
                np.savez(archive, **captures)
            labels = np.asarray([r['category'] in {'A1', 'A2', 'A3', 'A4', 'A5'}
                                 if axis == 'pain' else r['category'] == 'target' for r in rows])
            vector, components = direction(captures[axis], labels)
            axes[axis] = {'layer': LAYER, 'extraction_layer': LAYER, 'vector': vector.tolist(),
                          'hidden_size': len(vector), 'max_coefficient': 10,
                          'target_count': int(labels.sum()), 'control_count': int((~labels).sum()),
                          'removed_control_components': components, 'vector_norm': float(np.linalg.norm(vector))}
            print(json.dumps({'stage': 'fit', 'axis': axis, 'dimension': len(vector),
                              'layer': LAYER, 'vector_norm': axes[axis]['vector_norm']}), flush=True)
        bundle = {**manifest, 'axes': axes, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        target = ROOT / 'vectors/qwen3-14b-multi.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(bundle, indent=1) + '\n')
        temporary.replace(target)
        (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps({'stage': 'complete', 'path': str(target), 'axes': list(axes)}), flush=True)
    finally:
        engine.engine_core.shutdown()


if __name__ == '__main__':
    main()
