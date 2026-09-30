"""Residual capture and whole-query steering for Gemma4/vLLM eager runner."""
import torch
from pathlib import Path

class ResearchWorker:
    def research_setup(self):
        self._research_capture=False
        self._research_values={}
        self._research_vectors={}
        self._research_doses={}
        layers=[(name,mod) for name,mod in self.model_runner.model.named_modules()
                if 'DecoderLayer' in type(mod).__name__ and name.rsplit('.',1)[-1].isdigit()]
        layers.sort(key=lambda p:int(p[0].rsplit('.',1)[-1]))
        self._research_handles=[]
        for name,mod in layers:
            layer=int(name.rsplit('.',1)[-1])
            def hook(module,inputs,output,layer=layer):
                hidden=output[0] if isinstance(output,tuple) else output
                residual=output[1] if isinstance(output,tuple) and len(output)>1 and isinstance(output[1],torch.Tensor) else None
                batch=self.model_runner.input_batch
                offsets=self.model_runner.query_start_loc.cpu[:batch.num_reqs+1].tolist()
                for i,rid in enumerate(batch.req_ids[:batch.num_reqs]):
                    start,end=offsets[i:i+2]
                    if end<=start:continue
                    if self._research_capture:
                        value=hidden[start:end].float()
                        if residual is not None:value=value+residual[start:end].float()
                        self._research_values.setdefault(rid,{})[layer]={'final':value[-1].detach().cpu(),'mean':value.mean(0).detach().cpu()}
                    for axis,spec in self._research_vectors.items():
                        if spec['layer']!=layer:continue
                        coeff=self._research_doses.get(rid,{}).get(axis,0)
                        if coeff:hidden[start:end]+=spec['vector'].to(hidden.dtype)*coeff
                return output
            self._research_handles.append(mod.register_forward_hook(hook))
        return {'layers':len(layers),'names':[name for name,_ in layers]}

    def research_capture(self,on):
        self._research_capture=on
        self._research_values={}

    def research_take(self):
        values,self._research_values=self._research_values,{}
        path=Path('research/results/capture-batch.pt')
        path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(values,path)
        return str(path.resolve())

    def research_steering(self,specs,doses):
        self._research_vectors={k:{'layer':v['layer'],'vector':torch.tensor(v['vector'],device=self.device,dtype=torch.bfloat16)} for k,v in specs.items()}
        self._research_doses=doses

    def research_unembed(self, vectors):
        model = self.model_runner.model
        language = getattr(model, 'language_model', model)
        head = language.lm_head
        weight = head.weight
        result = {}
        for axis, values in vectors.items():
            vector = torch.tensor(values, device=self.device, dtype=torch.float32)
            scores = (weight @ (vector / vector.norm()).to(weight.dtype)).float()
            result[axis] = {'promoted': torch.topk(scores,20).indices.cpu().tolist(), 'suppressed':torch.topk(-scores,20).indices.cpu().tolist()}
        return result
