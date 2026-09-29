import json, random, time, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.datasets import fetch_lfw_people
from torch.utils.data import Dataset, DataLoader

SEED=20260929
SIZE=128
LATENT=160
MAX_IMAGES=6000
VAL_N=600
BATCH=48
MAX_GENERATIONS=12
CANDIDATES=4
SUBSET_N=3000
TIME_BUDGET_SEC=2050
PATIENCE=3
OUT=Path('lfw_generic_max_out'); OUT.mkdir(exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1,min(4,torch.get_num_threads())))

class FaceDS(Dataset):
    def __init__(self, imgs, idx, aug=False): self.imgs=imgs; self.idx=np.asarray(idx); self.aug=aug
    def __len__(self): return len(self.idx)
    def __getitem__(self,i):
        a=self.imgs[int(self.idx[i])]; h,w,_=a.shape; s=min(h,w); y=(h-s)//2; x=(w-s)//2; a=a[y:y+s,x:x+s]
        t=torch.from_numpy(a.copy()).permute(2,0,1).float(); t=F.interpolate(t[None],(SIZE,SIZE),mode='bilinear',align_corners=False)[0]; t=t*2-1
        if self.aug and torch.rand(())<.5: t=torch.flip(t,[2])
        if self.aug and torch.rand(())<.25: t=(t*(.94+torch.rand(())*.12)).clamp(-1,1)
        if self.aug and torch.rand(())<.15: t=(t+torch.randn_like(t)*.01).clamp(-1,1)
        return t

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.Sequential(nn.Conv2d(3,32,4,2,1),nn.GroupNorm(4,32),nn.SiLU(),nn.Conv2d(32,64,4,2,1),nn.GroupNorm(8,64),nn.SiLU(),nn.Conv2d(64,128,4,2,1),nn.GroupNorm(8,128),nn.SiLU(),nn.Conv2d(128,192,4,2,1),nn.GroupNorm(12,192),nn.SiLU(),nn.Conv2d(192,256,4,2,1),nn.GroupNorm(16,256),nn.SiLU())
        flat=256*4*4; self.mu=nn.Linear(flat,LATENT); self.lv=nn.Linear(flat,LATENT); self.fc=nn.Linear(LATENT,flat)
        self.dec=nn.Sequential(nn.ConvTranspose2d(256,192,4,2,1),nn.GroupNorm(12,192),nn.SiLU(),nn.ConvTranspose2d(192,128,4,2,1),nn.GroupNorm(8,128),nn.SiLU(),nn.ConvTranspose2d(128,64,4,2,1),nn.GroupNorm(8,64),nn.SiLU(),nn.ConvTranspose2d(64,32,4,2,1),nn.GroupNorm(4,32),nn.SiLU(),nn.ConvTranspose2d(32,3,4,2,1),nn.Tanh())
    def encode(self,x): h=self.enc(x).flatten(1); return self.mu(h),self.lv(h)
    def decode(self,z): return self.dec(self.fc(z).view(-1,256,4,4))
    def forward(self,x):
        mu,lv=self.encode(x); z=mu+torch.randn_like(mu)*torch.exp(.5*lv); return self.decode(z),mu,lv

def edge_loss(a,b):
    dx1=a[:,:,:,1:]-a[:,:,:,:-1]; dx2=b[:,:,:,1:]-b[:,:,:,:-1]
    dy1=a[:,:,1:,:]-a[:,:,:-1,:]; dy2=b[:,:,1:,:]-b[:,:,:-1,:]
    return .5*(F.l1_loss(dx1,dx2)+F.l1_loss(dy1,dy2))

def metrics(m,dl):
    m.eval(); n=0; rec=ed=kl=0.0
    with torch.no_grad():
        for x in dl:
            r,mu,lv=m(x); b=x.size(0); rr=F.l1_loss(r,x); ee=edge_loss(r,x); kk=-.5*torch.mean(1+lv-mu.pow(2)-lv.exp())
            n+=b; rec+=float(rr)*b; ed+=float(ee)*b; kl+=float(kk)*b
    return {'reconstruction_l1':rec/n,'edge_l1':ed/n,'kl':kl/n}

def score(v): return v['reconstruction_l1']+0.16*v['edge_l1']

def train_epoch(m,dl,lr,edge_w,kl_w):
    m.train(); opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4)
    for x in dl:
        r,mu,lv=m(x); rec=F.l1_loss(r,x); ed=edge_loss(r,x); kl=-.5*torch.mean(1+lv-mu.pow(2)-lv.exp())
        loss=rec+edge_w*ed+kl_w*kl; opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()

print('Loading LFW...',flush=True)
lfw=fetch_lfw_people(color=True,resize=1.0,download_if_missing=True); imgs=np.asarray(lfw.images,dtype=np.float32)
rng=np.random.default_rng(SEED); idx=rng.permutation(len(imgs))[:min(MAX_IMAGES,len(imgs))]; va_idx=idx[:VAL_N]; pool=idx[VAL_N:]
va=DataLoader(FaceDS(imgs,va_idx,False),batch_size=BATCH,shuffle=False,num_workers=0)

parent_path=next(Path('parent_artifact').rglob('*.pt'))
ck=torch.load(parent_path,map_location='cpu',weights_only=False); parent=VAE(); sd={k:(v.float() if torch.is_floating_point(v) else v) for k,v in ck['state_dict'].items()}; parent.load_state_dict(sd)
parent_v=metrics(parent,va); parent_s=score(parent_v); baseline_s=parent_s
print('PARENT',json.dumps({'path':str(parent_path),'metrics':parent_v,'score':parent_s}),flush=True)

configs=[(2.0e-4,.12,.00020),(3.0e-4,.16,.00025),(4.0e-4,.20,.00030),(5.0e-4,.24,.00035)]
history=[]; no_improve=0; start=time.time(); generations=0
for gen in range(1,MAX_GENERATIONS+1):
    if time.time()-start>TIME_BUDGET_SEC: print('TIME_BUDGET_STOP',gen,flush=True); break
    perm=np.random.default_rng(SEED+gen*101).permutation(pool); sub=perm[:min(SUBSET_N,len(perm))]
    best_m=None; best_v=None; best_s=1e9; best_cfg=None; cand_rows=[]
    for ci,(lr,ew,kw) in enumerate(configs):
        if time.time()-start>TIME_BUDGET_SEC: break
        torch.manual_seed(SEED+gen*1000+ci)
        m=copy.deepcopy(parent)
        dl=DataLoader(FaceDS(imgs,sub,True),batch_size=BATCH,shuffle=True,num_workers=0)
        train_epoch(m,dl,lr,ew,kw); v=metrics(m,va); s=score(v)
        row={'generation':gen,'candidate':ci,'lr':lr,'edge_w':ew,'kl_w':kw,'metrics':v,'score':s}; cand_rows.append(row); print('CAND',json.dumps(row),flush=True)
        if s<best_s: best_m,best_v,best_s,best_cfg=m,v,s,(lr,ew,kw)
    improved=best_m is not None and best_s < parent_s-1e-5
    if improved:
        parent=best_m; parent_v=best_v; parent_s=best_s; no_improve=0
    else:
        no_improve+=1
    history.append({'generation':gen,'improved':improved,'winner_config':best_cfg,'winner_score':best_s if best_m else None,'parent_score_after':parent_s,'candidates':cand_rows})
    generations=gen
    print('GEN',json.dumps(history[-1]),flush=True)
    if no_improve>=PATIENCE: print('PATIENCE_STOP',gen,flush=True); break

final_v=metrics(parent,va); final_s=score(final_v)
sd16={k:(v.half() if torch.is_floating_point(v) else v) for k,v in parent.state_dict().items()}
torch.save({'state_dict':sd16,'architecture':'unlabeled_conv_vae','resolution':SIZE,'latent':LATENT,'identity_labels_used':False,'evolution_generations':generations,'baseline_score':baseline_s,'final_score':final_s},OUT/'lfw_generic_face_vae_max.pt')

parent.eval(); g=torch.Generator().manual_seed(SEED+777)
with torch.no_grad(): sm=parent.decode(torch.randn(36,LATENT,generator=g)).clamp(-1,1)
can=Image.new('RGB',(SIZE*6,SIZE*6),'white')
for i in range(36):
    a=((sm[i]+1)*127.5).byte().permute(1,2,0).numpy(); can.paste(Image.fromarray(a),(i%6*SIZE,i//6*SIZE))
can.save(OUT/'generated_samples_max.png')

summary={'source':'LFW via scikit-learn','identity_labels_used':False,'images_pool':int(len(idx)),'validation_images':VAL_N,'resolution':SIZE,'latent':LATENT,'parameters':sum(p.numel() for p in parent.parameters()),'max_generations':MAX_GENERATIONS,'generations_completed':generations,'candidates_per_generation':CANDIDATES,'subset_per_candidate':SUBSET_N,'time_budget_sec':TIME_BUDGET_SEC,'baseline_validation':parent_v if generations==0 else None,'baseline_score':baseline_s,'final_validation':final_v,'final_score':final_s,'relative_score_improvement_pct':(baseline_s-final_s)/baseline_s*100,'elapsed_seconds':round(time.time()-start,2),'raw_photos_in_artifact':False}
(OUT/'summary_max.json').write_text(json.dumps(summary,indent=2)); (OUT/'evolution_history.json').write_text(json.dumps(history,indent=2))
print('FINAL',json.dumps(summary),flush=True)
