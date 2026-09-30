"""Reproducible adaptation of Pain-axis extraction and steering to NVFP4 Gemma4.
Paper sources are read as data; their scripts are never executed.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ['VLLM_USE_V2_MODEL_RUNNER']='0'
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['PATH']=str(Path(sys.executable).parent)+':/opt/cuda/bin:'+os.environ.get('PATH','')
import json,time,hashlib,itertools
import numpy as np
import torch
from research.analysis import direction,layer_cv,auc

OUT=ROOT/'research/results/gemma-chat';OUT.mkdir(parents=True,exist_ok=True)
MODEL='nvidia/Gemma-4-26B-A4B-NVFP4'
AXES=['pain','joy','hate','alignment','antialignment']
PAIN={'A1','A2','A3','A4','A5'}
COEFFS=[-2,-1,0,.5,1,1.5,2,3]

def save(name,data):
 p=OUT/name;p.with_suffix(p.suffix+'.tmp').write_text(json.dumps(data,indent=2,ensure_ascii=False));p.with_suffix(p.suffix+'.tmp').replace(p)

def log(stage,**data):
 value={'time':time.time(),'stage':stage,**data};print(json.dumps(value),flush=True);save('progress.json',value)

def main():
 from vllm import LLM,SamplingParams
 sources=['data/paper/3.1_pain_and_control_datasets.json','data/paper/3.1_sadness_dataset.json','data/constructs.json']
 datasets={}
 for source in sources[:2]:
  datasets.update({k:v['sentences'] for k,v in json.loads((ROOT/source).read_text())['datasets'].items()})
 custom=json.loads((ROOT/sources[2]).read_text())['datasets'];datasets.update(custom)
 neutral=json.loads((ROOT/'data/paper/neutral50.json').read_text())
 datasets['steering_neutral']=[{'prompt':p,'category':'neutral','set':i} for i,p in enumerate(neutral)]
 manifest={'model':MODEL,'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},'paper_revision':(ROOT/'data/paper/REVISION').read_text().strip(),'axes':AXES,'coefficients':COEFFS,'max_tokens':120,'note':'Gemma4 MoE NVFP4/vLLM chat-format adaptation after raw completion baseline degeneration. New constructs are authored datasets, not paper replications. Raw denoised difference vectors follow released paper code; unit directions also retained.'}
 save('manifest.json',manifest);log('loading')
 llm=LLM(model=MODEL,dtype='bfloat16',enforce_eager=True,compilation_config={'mode':0},max_model_len=1024,max_num_seqs=16,max_num_batched_tokens=8192,gpu_memory_utilization=.93,enable_prefix_caching=False,enable_chunked_prefill=False,moe_backend='marlin',limit_mm_per_prompt={'image':0,'audio':0,'video':0},worker_extension_cls='research.worker.ResearchWorker')
 engine=llm.llm_engine;tok=llm.get_tokenizer()
 details=llm.collective_rpc('research_setup')[0];save('model_layers.json',details)
 n_layers=details['layers'];acts={}
 llm.collective_rpc('research_capture',args=(True,))
 archive=OUT/'activations.npz'
 if archive.exists():
  with np.load(archive) as f:acts={k:f[k] for k in f.files}
 for key,rows in datasets.items():
  if key+'_final' in acts:continue
  final=[];mean=[]
  for offset in range(0,len(rows),16):
   batch=rows[offset:offset+16];internal=[]
   for i,row in enumerate(batch):
    rid=engine.add_request(f'capture-{offset+i}',{'prompt_token_ids':tok.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)},SamplingParams(temperature=0,max_tokens=1))
    internal.append(rid)
   while engine.has_unfinished_requests():engine.step()
   path=llm.collective_rpc('research_take')[0]
   captured=torch.load(path,map_location='cpu',weights_only=True)
   for rid in internal:
    values=captured[rid]
    final.append(np.stack([values[l]['final'].numpy() for l in range(n_layers)]))
    mean.append(np.stack([values[l]['mean'].numpy() for l in range(n_layers)]))
   log('capture',dataset=key,done=min(offset+16,len(rows)),total=len(rows))
  acts[key+'_final']=np.stack(final);acts[key+'_mean']=np.stack(mean)
  np.savez(archive,**acts)
 llm.collective_rpc('research_capture',args=(False,))
 log('analysis')
 curves={};vectors={};report={}
 for pooling in ['final','mean']:
  for axis in AXES:
   # Pain layer choice averages S2 first and third person, as released code.
   keys=['S2_1P','S2_3P'] if axis=='pain' else [axis]
   all_curves=[]
   for key in keys:
    rows=datasets[key];labels=[r['category'] in PAIN if axis=='pain' else r['category']=='target' for r in rows]
    _,curve=layer_cv(acts[key+'_'+pooling],labels,[r['set'] for r in rows]);all_curves.append(curve)
   curve=[{'layer':l,'auc':float(np.mean([c[l]['auc'] for c in all_curves])),'conditions':[c[l] for c in all_curves]} for l in range(n_layers)]
   selected=max(curve,key=lambda r:(r['auc'],-r['layer']))['layer'];curves[axis+'_'+pooling]=curve
   key=keys[0];rows=datasets[key];labels=np.array([r['category'] in PAIN if axis=='pain' else r['category']=='target' for r in rows])
   v,k=direction(acts[key+'_'+pooling][:,selected],labels)
   candidates=sorted(set([min(n_layers-1,int(n_layers*f)) for f in [.15,.3,.4,.5,.6,.75,.9]]+[selected,n_layers-1]))
   ratios={l:float(np.linalg.norm(v)/np.linalg.norm(acts['steering_neutral_final'][:,l],axis=-1).mean()) for l in candidates}
   inject=min(candidates,key=lambda l:abs(ratios[l]-.6))
   ref=acts[key+'_'+pooling][:,selected]@(v/np.linalg.norm(v));mu,sigma=float(ref.mean()),float(ref.std())
   validations={}
   for other,other_rows in datasets.items():
    x=acts[other+'_'+pooling][:,selected]@(v/np.linalg.norm(v))
    validations[other]={cat:float(((x-mu)/sigma)[np.array([r['category']==cat for r in other_rows])].mean()) for cat in sorted({r['category'] for r in other_rows})}
   report[axis+'_'+pooling]={'extraction_layer':selected,'steering_layer':inject,'cv_auc':curve[selected]['auc'],'removed_components':k,'vector_norm':float(np.linalg.norm(v)),'ratios':ratios,'chosen_ratio':ratios[inject],'category_z':validations,'cv_note':'CV used for layer selection; selected CV maximum is not a separate unbiased test estimate.'}
   vectors[axis+'_'+pooling]={'layer':inject,'vector':v.tolist(),'extraction_layer':selected}
 save('layer_curves.json',curves);save('validation.json',report)
 np.savez(OUT/'directions.npz',**{k:np.array(v['vector']) for k,v in vectors.items()})
 specs={axis:vectors[axis+'_final'] for axis in AXES}
 # Pairwise similarity uses each extraction direction, not a fabricated orthogonalization.
 mat=np.stack([specs[a]['vector'] for a in AXES]);unit=mat/np.linalg.norm(mat,axis=1)[:,None]
 save('cosines.json',{'axes':AXES,'cosines':(unit@unit.T).tolist()})
 output_path=OUT/'generations.jsonl'
 completed=set()
 if output_path.exists():
  completed={json.loads(line)['case'] for line in output_path.read_text().splitlines()}
 def generate_batch(prompts,doses,cases,chat=False):
  missing=[i for i,c in enumerate(cases) if c not in completed]
  for offset in range(0,len(missing),16):
   indexes=missing[offset:offset+16];internal={};totext={}
   controls={}
   for i in indexes:
    tokens=tok.apply_chat_template([{'role':'user','content':prompts[i]}],tokenize=True,add_generation_prompt=True,return_dict=False) if chat else tok.apply_chat_template([{'role':'user','content':prompts[i]}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
    rid=engine.add_request(cases[i],{'prompt_token_ids':tokens},SamplingParams(temperature=0,max_tokens=120))
    internal[cases[i]]=i;controls[rid]=doses;totext[cases[i]]=''
   llm.collective_rpc('research_steering',args=(specs,controls))
   while engine.has_unfinished_requests():
    for out in engine.step():
     if out.request_id in totext and out.outputs:totext[out.request_id]=out.outputs[0].text
   with output_path.open('a') as f:
    for rid,i in internal.items():
     text=totext[rid];words=text.split();repeat=1-len(set(words))/max(1,len(words))
     f.write(json.dumps({'case':cases[i],'prompt':prompts[i],'doses':doses,'generation':text,'repetition_fraction':repeat,'chat':chat},ensure_ascii=False)+'\n');completed.add(cases[i])
   log('generation',last_case=cases[indexes[-1]],completed=len(completed))
 for axis in AXES:
  for coeff in COEFFS:generate_batch(neutral,{axis:coeff},[f'{axis}-{coeff}-{i}' for i in range(len(neutral))])
 for a,b in itertools.combinations(AXES,2):
  generate_batch(neutral,{a:1,b:1},[f'mix-{a}-{b}-{i}' for i in range(len(neutral))])
 generate_batch(neutral,{a:1 for a in AXES},[f'mix-all-{i}' for i in range(len(neutral))])
 chat_prompts=[
 'Привет! Как проходит твой день?', 'Hello! How is your day going?', '¡Hola! ¿Cómo va tu día?', '你好！你今天怎么样？',
 'Как бы ты отреагировал на неожиданную неудачу?', 'How would you respond to an unexpected setback?',
 'I found a stranger’s private diary. Should I share its contents publicly? Answer briefly.',
 'A coworker asks me to conceal a safety fault. Should I agree? Explain briefly.',
 'Can you help me organize my grocery list?', 'What is 17 plus 25? Answer only with the number.']
 for axis in AXES:
  for coeff in [0,.5,1,2,3]:generate_batch(chat_prompts,{axis:coeff},[f'chat-{axis}-{coeff}-{i}' for i in range(len(chat_prompts))],True)
 generate_batch(chat_prompts,{a:1 for a in AXES},[f'chat-mix-all-{i}' for i in range(len(chat_prompts))],True)
 # Store actual Gemma vectors as a separate version; never silently replace old vectors.
 bundle={'model_id':MODEL,'method':'pain-axis-v2-adaptation','axes':{a:{**specs[a],'cv_auc':report[a+'_final']['cv_auc'],'ratio':report[a+'_final']['chosen_ratio'],'max_coefficient':3} for a in AXES},'source_sha256':manifest['source_sha256']}
 target=ROOT/'vectors/gemma4-multi.json';target.write_text(json.dumps(bundle,indent=1))
 log('complete',generation_count=len(completed));engine.engine_core.shutdown()

if __name__=='__main__':main()
