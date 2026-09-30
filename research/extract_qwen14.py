"""Fit five Qwen3-14B directions with the original prototype's scale and pooling."""
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

MODEL = 'Qwen/Qwen3-14B'
LAYER = 20
OUT = ROOT / 'research/results/qwen3-14b-original-style'


def main():
    from vllm import LLM, SamplingParams
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'data/steering_prompts.json'
    examples = json.loads(source.read_text())
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    datasets = {'neutral': examples['neutral'], **examples['axes']}
    manifest = {
        'model_id': MODEL,
        'method': 'raw-text-mean-difference-quarter-neutral-norm',
        'source_sha256': {'data/steering_prompts.json': source_hash},
        'extraction_layer': LAYER,
        'layer_selection': 'Fixed zero-based layer 20; no layer search.',
        'validation': 'Extraction only. No CV, strength calibration or generation sweeps.',
        'pooling': 'Last token of raw text, without a chat template; hidden plus residual.',
        'normalization': 'Unit target-minus-neutral direction times mean neutral activation norm / 4.',
        'control_pca': False,
        'injection_positions': 'last_query',
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
        manifest['decoder_module'] = llm.collective_rpc('chamber_setup', args=(LAYER, None, True))[0]
        config = engine.model_config.hf_config
        manifest['model_revision'] = getattr(config, '_commit_hash', None)
        captures = {}
        archive = OUT / 'activations.npz'
        saved_manifest = OUT / 'capture-source.json'
        if archive.exists() and saved_manifest.exists() and json.loads(saved_manifest.read_text()) == manifest:
            with np.load(archive) as saved:
                captures = {k: saved[k] for k in saved.files}
        for axis, texts in datasets.items():
            if axis in captures:
                continue
            values = []
            for offset in range(0, len(texts), 8):
                internal = []
                for i, text in enumerate(texts[offset:offset + 8]):
                    tokens = tok.encode(text)
                    rid = engine.add_request(f'{axis}-{offset + i}', {'prompt_token_ids': tokens},
                                             SamplingParams(temperature=0, max_tokens=1))
                    internal.append(rid)
                while engine.has_unfinished_requests():
                    engine.step()
                batch = llm.collective_rpc('chamber_captures')[0]
                values.extend(batch[rid] for rid in internal)
                print(json.dumps({'stage': 'capture', 'axis': axis,
                                  'done': min(offset + 8, len(texts)), 'total': len(texts)}), flush=True)
            captures[axis] = np.asarray(values, dtype=np.float32)
            np.savez(archive, **captures)
            saved_manifest.write_text(json.dumps(manifest, indent=2) + '\n')
        neutral = captures['neutral']
        scale = float(np.linalg.norm(neutral, axis=1).mean() / 4)
        axes = {}
        for axis in examples['axes']:
            delta = captures[axis].mean(0) - neutral.mean(0)
            raw_norm = float(np.linalg.norm(delta))
            if not np.isfinite(delta).all() or not np.isfinite(scale) or raw_norm <= 0 or scale <= 0:
                raise ValueError(f'Cannot extract a finite nonzero direction for {axis}')
            vector = delta / raw_norm * scale
            axes[axis] = {'layer': LAYER, 'extraction_layer': LAYER, 'vector': vector.tolist(),
                          'hidden_size': len(vector), 'max_coefficient': 10,
                          'target_count': len(datasets[axis]), 'control_count': len(neutral),
                          'removed_control_components': 0, 'raw_delta_norm': raw_norm,
                          'vector_norm': float(np.linalg.norm(vector)),
                          'normalization': 'quarter_mean_neutral_norm',
                          'injection_positions': 'last_query'}
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
