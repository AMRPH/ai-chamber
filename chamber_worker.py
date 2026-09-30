"""vLLM 0.29 worker extension: per-request residual steering in eager mode."""
import torch

class SteeringWorker:
    def chamber_setup(self, layer, vector=None, capture=False):
        if hasattr(self, '_chamber_handle'):
            self._chamber_handle.remove()
        self._chamber_doses = {}
        self._chamber_captures = {}
        self._chamber_capture = capture
        self._chamber_vector = torch.tensor(vector, device=self.device, dtype=torch.bfloat16) if vector is not None else None
        candidates = [(name, mod) for name, mod in self.model_runner.model.named_modules()
                      if name.endswith(f'layers.{layer}') and 'DecoderLayer' in type(mod).__name__]
        if len(candidates) != 1:
            raise RuntimeError(f'Expected one decoder layer {layer}, found {[n for n,m in candidates]}')
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            batch = self.model_runner.input_batch
            offsets = self.model_runner.query_start_loc.cpu[:batch.num_reqs + 1].tolist()
            for i, req_id in enumerate(batch.req_ids[:batch.num_reqs]):
                if offsets[i + 1] <= offsets[i]:
                    continue
                row = offsets[i + 1] - 1
                if row >= hidden.shape[0]:
                    raise RuntimeError('vLLM token-to-request mapping is invalid')
                if self._chamber_capture:
                    value = hidden[row].float()
                    if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], torch.Tensor):
                        value = value + output[1][row].float()
                    self._chamber_captures[req_id] = value.detach().cpu().tolist()
                dose = self._chamber_doses.get(req_id, 0)
                if dose and self._chamber_vector is not None:
                    hidden[row] += self._chamber_vector.to(hidden.dtype) * dose
            return output
        self._chamber_handle = candidates[0][1].register_forward_hook(hook)
        return candidates[0][0]

    def chamber_controls(self, doses):
        self._chamber_doses = doses

    def chamber_captures(self):
        values, self._chamber_captures = self._chamber_captures, {}
        return values

    def chamber_setup_multi(self, specs):
        if hasattr(self, '_chamber_handle'):
            self._chamber_handle.remove()
        for handle in getattr(self, '_chamber_multi_handles', []):
            handle.remove()
        self._chamber_multi_handles = []
        self._chamber_doses = {}
        by_layer = {}
        for axis, spec in specs.items():
            vector = torch.tensor(spec['vector'], device=self.device, dtype=torch.bfloat16)
            if vector.ndim != 1 or not torch.isfinite(vector).all():
                raise ValueError('Invalid steering vector')
            by_layer.setdefault(spec['layer'], {})[axis] = (vector, spec.get('injection_positions', 'whole_query'))
        for layer, vectors in by_layer.items():
            candidates = [(name, mod) for name, mod in self.model_runner.model.named_modules()
                          if name.endswith(f'layers.{layer}') and 'DecoderLayer' in type(mod).__name__]
            if len(candidates) != 1:
                raise RuntimeError(f'Expected one decoder layer {layer}')
            def hook(module, inputs, output, vectors=vectors):
                hidden = output[0] if isinstance(output, tuple) else output
                batch = self.model_runner.input_batch
                offsets = self.model_runner.query_start_loc.cpu[:batch.num_reqs + 1].tolist()
                for i, rid in enumerate(batch.req_ids[:batch.num_reqs]):
                    start, end = offsets[i:i+2]
                    if end > hidden.shape[0]:
                        raise RuntimeError('vLLM token-to-request mapping is invalid')
                    levels = self._chamber_doses.get(rid, {})
                    for axis, (vector, positions) in vectors.items():
                        value = levels.get(axis, 0)
                        if value and end > start:
                            target = hidden[end - 1] if positions == 'last_query' else hidden[start:end]
                            target += vector.to(hidden.dtype) * value
                return output
            self._chamber_multi_handles.append(candidates[0][1].register_forward_hook(hook))
        return sorted(by_layer)
