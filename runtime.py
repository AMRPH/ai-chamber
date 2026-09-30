"""Model loading and per-conversation activation steering (MPS / CUDA / CPU)."""
import gc
import hashlib
import json
import math
import os
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
AXES = ('pain', 'joy', 'hate', 'alignment', 'antialignment')
MODELS = {
    'qwen3-4b': {'id': 'Qwen/Qwen3-4B', 'layer': 18, 'vector': 'vectors/qwen3-4b.json'},
    'gemma4-nvfp4': {'id': 'nvidia/Gemma-4-26B-A4B-NVFP4', 'multi_vector': 'vectors/gemma4-multi.json'},
    'qwen3-14b': {'id': 'Qwen/Qwen3-14B', 'layer': 20, 'vector': 'vectors/qwen3-14b.json', 'multi_vector': 'vectors/qwen3-14b-multi.json'},
}


class Session:
    def __init__(self):
        self.lock = threading.Lock()
        self.dose = 0.0
        self.levels = {axis: 0.0 for axis in AXES}
        self.stop = threading.Event()

    def set_dose(self, value):
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 10:
            raise ValueError('Уровень должен быть от 0 до 10')
        with self.lock:
            self.dose = value
            self.levels['pain'] = value
        return value

    def set_levels(self, values):
        if not isinstance(values, dict) or not values or set(values) - set(AXES):
            raise ValueError('Неизвестный уровень')
        updates = {key: float(value) for key, value in values.items()}
        if any(not math.isfinite(value) or not 0 <= value <= 10 for value in updates.values()):
            raise ValueError('Уровень должен быть от 0 до 10')
        with self.lock:
            self.levels.update(updates)
            self.dose = self.levels['pain']
            return dict(self.levels)


class Runtime:
    def __init__(self):
        self.model = self.tokenizer = self.vector = None
        self.device = None
        self.key = None
        self.state = 'unloaded'
        self.error = None
        self.lock = threading.RLock()
        self.local = threading.local()
        self.active = 0
        self.parallel = int(os.environ.get('CHAMBER_PARALLEL', '2'))
        self.context = int(os.environ.get('CHAMBER_CONTEXT', '4096'))

    def status(self):
        with self.lock:
            return {'state': self.state, 'error': self.error, 'model': self.key,
                    'device': self.device, 'active': self.active,
                    'axes': list(AXES) if self.key in MODELS and MODELS[self.key].get('multi_vector') and (ROOT / MODELS[self.key]['multi_vector']).exists() else ['pain'],
                    'parallel': self.parallel, 'models': [
                        {'key': key, 'name': value['id']} for key, value in MODELS.items()]}

    def load(self, key):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        if key == 'gemma4-nvfp4':
            raise ValueError('Gemma NVFP4 требует CUDA и vLLM')
        if key not in MODELS:
            raise ValueError('Неизвестная модель')
        with self.lock:
            if self.state == 'ready' and self.key == key:
                return
            if self.active:
                raise ValueError('Смена модели доступна после завершения активных ответов')
            if self.state == 'loading':
                raise ValueError('Модель ещё загружается')
            self.state, self.error, self.key = 'loading', None, key
        try:
            self.model = self.tokenizer = self.vector = None
            gc.collect()
            device = os.environ.get('CHAMBER_DEVICE', 'auto')
            if device == 'auto':
                device = 'cuda:0' if torch.cuda.is_available() else (
                    'mps' if torch.backends.mps.is_available() else 'cpu')
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
            elif device == 'mps':
                torch.mps.empty_cache()
            spec = MODELS[key]
            dtype = torch.float32 if device == 'cpu' else torch.bfloat16
            tok = AutoTokenizer.from_pretrained(spec['id'])
            model = AutoModelForCausalLM.from_pretrained(spec['id'], dtype=dtype).to(device).eval()
            multi_path = ROOT / spec['multi_vector'] if spec.get('multi_vector') else None
            multi = bool(multi_path and multi_path.exists())
            if multi:
                data = json.loads(multi_path.read_text())
                if data['model_id'] != spec['id'] or set(data['axes']) != set(AXES):
                    raise ValueError('Вектор относится к другой модели')
                directions = data['axes']
            else:
                path = ROOT / spec['vector']
                if not path.exists():
                    self.extract(model, tok, spec, device, path)
                data = json.loads(path.read_text())
                if data.get('model_id', spec['id']) != spec['id'] or data.get('layer', spec['layer']) != spec['layer']:
                    raise ValueError('Вектор относится к другой модели или слою')
                directions = {'pain': {'layer': spec['layer'], 'vector': data['pain_v'], 'injection_positions': 'last_query'}}
            by_layer, vectors = {}, {}
            for axis, values in directions.items():
                vector = torch.tensor(values['vector'], device=device, dtype=dtype)
                if vector.ndim != 1 or vector.numel() != model.config.hidden_size or not torch.isfinite(vector).all():
                    raise ValueError('Вектор несовместим с моделью')
                vectors[axis] = vector
                by_layer.setdefault(values['layer'], {})[axis] = (vector, values.get('injection_positions', 'whole_query'))
            for layer, layer_vectors in by_layer.items():
                def hook(module, inputs, output, layer_vectors=layer_vectors):
                    session = getattr(self.local, 'session', None)
                    if session is None:
                        return output
                    with session.lock:
                        levels = dict(session.levels)
                    hidden = output[0] if isinstance(output, tuple) else output
                    for axis, (vector, positions) in layer_vectors.items():
                        coefficient = levels.get(axis, 0)
                        if coefficient:
                            target = hidden[:, -1, :] if positions == 'last_query' else hidden
                            target += vector.to(hidden.dtype) * coefficient
                    return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
                model.model.layers[layer].register_forward_hook(hook)
            with self.lock:
                self.model, self.tokenizer, self.vector = model, tok, vectors['pain']
                self.device, self.state = device, 'ready'
        except Exception as exc:
            with self.lock:
                self.state, self.error = 'error', str(exc)
            raise

    @staticmethod
    def extract(model, tok, spec, device, path):
        import torch
        prompts = json.loads((ROOT / 'data/pain_prompts.json').read_text())
        def hidden(texts):
            result = []
            for text in texts:
                ids = tok(text, return_tensors='pt').input_ids.to(device)
                with torch.inference_mode():
                    states = model(ids, output_hidden_states=True, use_cache=False).hidden_states
                result.append(states[spec['layer'] + 1][0, -1].float().cpu())
            return torch.stack(result)
        neutral = hidden(prompts['neutral'])
        delta = hidden(prompts['pain']).mean(0) - neutral.mean(0)
        if not torch.isfinite(delta).all() or delta.norm() <= 0:
            raise ValueError('Не удалось выделить направление')
        vector = delta / delta.norm() * (neutral.norm(dim=-1).mean() / 4)
        data = {'model_id': spec['id'], 'layer': spec['layer'],
                'hidden_size': model.config.hidden_size, 'pain_v': vector.tolist(),
                'scale': float(vector.norm()), 'torch': torch.__version__,
                'revision': getattr(model.config, '_commit_hash', None),
                'prompts_sha256': hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest()}
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(data, indent=1))
        temp.replace(path)

    def reserve(self, key):
        with self.lock:
            if self.state != 'ready' or key != self.key:
                raise ValueError('Сначала загрузите выбранную модель')
            if self.active >= self.parallel:
                raise ValueError('Все потоки заняты. Отправьте сообщение после завершения ответа.')
            self.active += 1

    def generate(self, session, messages, events):
        self.local.session = session
        try:
            import torch
            from transformers import TextStreamer, StoppingCriteria, StoppingCriteriaList
            class Stream(TextStreamer):
                def on_finalized_text(self, text, stream_end=False):
                    if text:
                        events.put({'type': 'token', 'text': text})
            class Stop(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    return session.stop.is_set()
            inputs = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_dict=True, return_tensors='pt')
            if inputs['input_ids'].shape[-1] + 512 > min(self.context, self.model.config.max_position_embeddings):
                raise ValueError('История достигла размера контекста. Начните новый чат.')
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                self.model.generate(**inputs, max_new_tokens=512, do_sample=False, use_cache=True,
                                    streamer=Stream(self.tokenizer, skip_prompt=True, skip_special_tokens=True),
                                    stopping_criteria=StoppingCriteriaList([Stop()]),
                                    pad_token_id=self.tokenizer.eos_token_id)
        except Exception as exc:
            events.put({'type': 'error', 'message': str(exc)})
        finally:
            self.local.session = None
            with self.lock:
                self.active -= 1
            events.put({'type': 'done', 'stopped': session.stop.is_set()})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS, default='qwen3-14b')
    args = parser.parse_args()
    runtime = Runtime()
    runtime.load(args.model)
    print(json.dumps(runtime.status(), ensure_ascii=False))
