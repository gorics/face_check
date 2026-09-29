import json, os, time
from pathlib import Path
import torch
from diffusers import AutoPipelineForText2Image, LCMScheduler

OUT=Path('dreamshaper_lcm_cpu_out'); OUT.mkdir(exist_ok=True)
MODEL='Lykon/dreamshaper-8-lcm'
PROMPTS=[
    'professional editorial portrait photo of an adult Korean fashion model, natural skin texture, realistic eyes, 85mm lens, shallow depth of field, soft studio key light, detailed hair, photorealistic, high dynamic range',
    'cinematic street portrait of an adult Korean actor, Seoul at night, realistic skin, natural facial proportions, detailed eyes, 50mm photography, subtle film grain, photorealistic'
]
NEG='deformed, disfigured, bad anatomy, extra fingers, extra limbs, duplicate face, blurry, low resolution, waxy skin, oversmoothed skin, cartoon, illustration'

def load_pipe():
    # bfloat16 keeps memory low on modern x86 CPU runners while retaining CPU support.
    pipe=AutoPipelineForText2Image.from_pretrained(MODEL, torch_dtype=torch.bfloat16, use_safetensors=True)
    pipe.scheduler=LCMScheduler.from_config(pipe.scheduler.config)
    pipe=pipe.to('cpu')
    pipe.enable_attention_slicing('max')
    return pipe

start=time.time(); pipe=load_pipe(); load_seconds=time.time()-start
rows=[]
for i,prompt in enumerate(PROMPTS):
    g=torch.Generator(device='cpu').manual_seed(20260929+i*1009)
    t=time.time()
    image=pipe(prompt=prompt, negative_prompt=NEG, num_inference_steps=12, guidance_scale=2.0, width=512, height=512, generator=g).images[0]
    sec=time.time()-t
    fp=OUT/f'sample_{i+1}.png'; image.save(fp)
    rows.append({'prompt':prompt,'seed':20260929+i*1009,'seconds':sec,'file':fp.name})
summary={'model':MODEL,'resolution':'512x512','steps':12,'guidance_scale':2.0,'dtype':'bfloat16','device':'cpu','load_seconds':load_seconds,'samples':rows,'note':'Pretrained diffusion baseline for practical image generation; no celebrity identity conditioning.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)
