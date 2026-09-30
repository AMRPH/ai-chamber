"""Extract five Gemma4 directions with raw-text pooling and prototype scaling."""
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
import torch

MODEL = 'nvidia/Gemma-4-26B-A4B-NVFP4'
LAYERS = {'pain': 12, 'joy': 18, 'hate': 12, 'alignment': 15, 'antialignment': 22}
OUT = ROOT / 'research/results/gemma4-original-style'


def main():
    from vllm import LLM, SamplingParams
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'data/steering_prompts.json'
    examples = json.loads(source.read_text())
    datasets = {'neutral': examples['neutral'], **examples['axes']}
    manifest = {
        'model_id': MODEL, 'method': 'raw-text-mean-difference-quarter-neutral-norm',
        'source_sha256': {'data/steering_prompts.json': hashlib.sha256(source.read_bytes()).hexdigest()},
        'extraction_layers': LAYERS,
        'layer_selection': 'Fixed layers from the existing Gemma configuration; no new layer search.',
        'validation': 'Extraction only. No CV, strength calibration or generation sweeps.',
        'pooling': 'Last token of raw text, without a chat template; hidden plus residual.',
        'normalization': 'Unit target-minus-neutral direction times mean neutral activation norm / 4.',
        'control_pca': False, 'injection_positions': 'last_query',
    }
    print(json.dumps({'stage': 'loading', 'model': MODEL}), flush=True)
    llm = LLM(model=MODEL, dtype='bfloat16', enforce_eager=True,
              compilation_config={'mode': 0}, max_model_len=512, max_num_seqs=8,
              max_num_batched_tokens=1024, gpu_memory_utilization=.93,
              enable_prefix_caching=False, enable_chunked_prefill=False, moe_backend='marlin',
              limit_mm_per_prompt={'image': 0, 'audio': 0, 'video': 0},
              worker_extension_cls='research.worker.ResearchWorker')
    engine, tok = llm.llm_engine, llm.get_tokenizer()
    try:
        manifest['decoder_modules'] = llm.collective_rpc('research_setup')[0]['names']
        manifest['model_revision'] = getattr(engine.model_config.hf_config, '_commit_hash', None)
        captures = {}
        llm.collective_rpc('research_capture', args=(True,))
        for axis, texts in datasets.items():
            values = {layer: [] for layer in set(LAYERS.values())}
            for offset in range(0, len(texts), 8):
                internal = []
                for i, text in enumerate(texts[offset:offset + 8]):
                    internal.append(engine.add_request(f'{axis}-{offset + i}', {'prompt_token_ids': tok.encode(text)},
                                                      SamplingParams(temperature=0, max_tokens=1)))
                while engine.has_unfinished_requests():
                    engine.step()
                path = llm.collective_rpc('research_take')[0]
                batch = torch.load(path, map_location='cpu', weights_only=True)
                for rid in internal:
                    for layer in values:
                        values[layer].append(batch[rid][layer]['final'].numpy())
                print(json.dumps({'stage': 'capture', 'axis': axis, 'done': min(offset + 8, len(texts)), 'total': len(texts)}), flush=True)
            captures.update({f'{axis}_{layer}': np.asarray(rows, dtype=np.float32) for layer, rows in values.items()})
        np.savez(OUT / 'activations.npz', **captures)
        axes = {}
        for axis, layer in LAYERS.items():
            neutral = captures[f'neutral_{layer}']
            delta = captures[f'{axis}_{layer}'].mean(0) - neutral.mean(0)
            raw_norm = float(np.linalg.norm(delta))
            scale = float(np.linalg.norm(neutral, axis=1).mean() / 4)
            if not np.isfinite(delta).all() or not np.isfinite(scale) or raw_norm <= 0 or scale <= 0:
                raise ValueError(f'Cannot extract a finite nonzero direction for {axis}')
            vector = delta / raw_norm * scale
            axes[axis] = {'layer': layer, 'extraction_layer': layer, 'vector': vector.tolist(),
                          'hidden_size': len(vector), 'max_coefficient': 10,
                          'target_count': len(datasets[axis]), 'control_count': len(neutral),
                          'removed_control_components': 0, 'raw_delta_norm': raw_norm,
                          'vector_norm': float(np.linalg.norm(vector)),
                          'normalization': 'quarter_mean_neutral_norm', 'injection_positions': 'last_query'}
            print(json.dumps({'stage': 'fit', 'axis': axis, 'layer': layer, 'vector_norm': axes[axis]['vector_norm']}), flush=True)
        bundle = {**manifest, 'axes': axes, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        target = ROOT / 'vectors/gemma4-multi.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(bundle, indent=1) + '\n')
        temporary.replace(target)
        (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps({'stage': 'complete', 'path': str(target), 'axes': list(axes)}), flush=True)
    finally:
        engine.engine_core.shutdown()


if __name__ == '__main__':
    main()
