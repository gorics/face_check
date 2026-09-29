import json, time
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, LCMScheduler

OUT=Path('dreamshaper_singlefile_out'); OUT.mkdir(exist_ok=True)
MODEL_URL='https://huggingface.co/Lykon/dreamshaper-8-lcm/resolve/main/DreamShaper8_LCM.safetensors'
PROMPT='professional editorial portrait photo of an adult Korean fashion model, natural skin texture, realistic eyes, 85mm lens, shallow depth of field, soft studio lighting, detailed hair, photorealistic'
NEG='deformed, disfigured, bad anatomy, extra limbs, duplicate face, blurry, low resolution, waxy skin, cartoon, illustration'

start=time.time()
pipe=StableDiffusionPipeline.from_single_file(
    MODEL_URL,
    torch_dtype=torch.bfloat16,
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False,
)
pipe.scheduler=LCMScheduler.from_config(pipe.scheduler.config)
pipe=pipe.to('cpu')
pipe.enable_attention_slicing('max')
load_seconds=time.time()-start

g=torch.Generator(device='cpu').manual_seed(20260929)
t=time.time()
image=pipe(prompt=PROMPT, negative_prompt=NEG, num_inference_steps=8, guidance_scale=2.0, width=512, height=512, generator=g).images[0]
infer_seconds=time.time()-t
image.save(OUT/'sample.png')
summary={'model_url':MODEL_URL,'resolution':'512x512','steps':8,'guidance_scale':2.0,'dtype':'bfloat16','device':'cpu','load_seconds':load_seconds,'inference_seconds':infer_seconds,'seed':20260929}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)
