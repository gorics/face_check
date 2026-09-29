import json, random, time, copy, os
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
BATCH=48
GENERATIONS=3
OUT=Path('lfw_generic_recursive_out'); OUT.mkdir(exist_ok=True)
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
        return t

class VAE(nn.Module):
    def __init__(self):
        super().__init__(); self.enc=nn.Sequential(
            nn.Conv2d(3,32,4,2,1),nn.GroupNorm(4,32),nn.SiLU(),
            nn.Conv2d(32,64,4,2,1),nn.GroupNorm(8,64),nn.SiLU(),
            nn.Conv2d(64,128,4,2,1),nn.GroupNorm(8,128),nn.SiLU(),
            nn.Conv2d(128,192,4,2,1),nn.GroupNorm(12,192),nn.SiLU(),
            nn.Conv2d(192,256,4,2,1),nn.GroupNorm(16,256),nn.SiLU())
        flat=256*4*4; self.mu=nn.Linear(flat,LATENT); self.lv=nn.Linear(flat,LATENT); self.fc=nn.Linear(LATENT,flat)
        self.dec=nn.Sequential(
            nn.ConvTranspose2d(256,192,4,2,1),nn.GroupNorm(12,192),nn.SiLU(),
            nn.ConvTranspose2d(192,128,4,2,1),nn.GroupNorm(8,128),nn.SiLU(),
            nn.ConvTranspose2d(128,64,4,2,1),nn.GroupNorm(8,64),nn.SiLU(),
            nn.ConvTranspose2d(64,32,4,2,1),nn.GroupNorm(4,32),nn.SiLU(),
            nn.ConvTranspose2d(32,3,4,2,1),nn.Tanh())
    def encode(self,x): h=self.enc(x).flatten(1); return self.mu(h),self.lv(h)
    def decode(self,z): return self.dec(self.fc(z).view(-1,256,4,4))
    def forward(self,x):
        mu,lv=self.encode(x); z=mu+torch.randn_like(mu)*torch.exp(.5*lv); return self.decode(z),mu,lv

def edge(a,b):
    dx1=a[:,:,:,1:]-a[:,:,:,:-1]; dx2=b[:,:,:,1:]-b[:,:,:,:-1]; dy1=a[:,:,1:,:]-a[:,:,:-1,:]; dy2=b[:,:,1:,:]-b[:,:,:-1,:]
    return .5*(F.l1_loss(dx1,dx2)+F.l1_loss(dy1,dy2))

def evaluate(m,dl):
    m.eval(); n=0; rec=ed=kl=0.
    with torch.no_grad():
        for x in dl:
            r,mu,lv=m(x); b=x.size(0); rr=F.l1_loss(r,x); ee=edge(r,x); kk=-.5*torch.mean(1+lv-mu.pow(2)-lv.exp())
            n+=b; rec+=float(rr)*b; ed+=float(ee)*b; kl+=float(kk)*b
    return {'reconstruction_l1':rec/n,'edge_l1':ed/n,'kl':kl/n}

def one_epoch(base_state, tr, va, lr, edge_w, kl_w, seed):
    torch.manual_seed(seed); m=VAE(); m.load_state_dict(base_state)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4)
    m.train(); n=0; loss_sum=0.
    for x in tr:
        r,mu,lv=m(x); rec=F.l1_loss(r,x); ed=edge(r,x); kl=-.5*torch.mean(1+lv-mu.pow(2)-lv.exp()); loss=rec+edge_w*ed+kl_w*kl
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); n+=x.size(0); loss_sum+=float(loss.detach())*x.size(0)
    v=evaluate(m,va); score=v['reconstruction_l1']+edge_w*v['edge_l1']
    return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}, {'lr':lr,'edge_w':edge_w,'kl_w':kl_w,'train_loss':loss_sum/n,'val':v,'score':score}

print('Loading LFW...',flush=True)
lfw=fetch_lfw_people(color=True,resize=1.0,download_if_missing=True); imgs=np.asarray(lfw.images,dtype=np.float32)
rng=np.random.default_rng(SEED); idx=rng.permutation(len(imgs))[:MAX_IMAGES]; va_idx=idx[:600]; tr_idx=idx[600:]
tr=DataLoader(FaceDS(imgs,tr_idx,True),batch_size=BATCH,shuffle=True,num_workers=0); va=DataLoader(FaceDS(imgs,va_idx),batch_size=BATCH,num_workers=0)

hits=list(Path('parent_artifact').rglob('lfw_generic_face_vae_v5.pt'))
if not hits: raise SystemExit('parent checkpoint missing')
ck=torch.load(hits[0],map_location='cpu'); m0=VAE(); state={k:(v.float() if torch.is_floating_point(v) else v) for k,v in ck['state_dict'].items()}; m0.load_state_dict(state)
parent={k:v.detach().cpu().clone() for k,v in m0.state_dict().items()}; base_eval=evaluate(m0,va)
lineage=[{'generation':0,'source':'v5 parent','val':base_eval,'score':base_eval['reconstruction_l1']+.16*base_eval['edge_l1']}]
print('PARENT',json.dumps(lineage[-1]),flush=True)

lr=.00045; ew=.16; kw=.00035; t0=time.time()
for gen in range(1,GENERATIONS+1):
    configs=[
      (lr*.72,min(.24,ew*1.10),max(.00008,kw*.80)),
      (lr*1.00,max(.08,ew*.92),min(.0008,kw*1.12)),
      (lr*.86,ew,max(.00008,kw*.62)),
    ]
    cand=[]
    for ci,(clr,cew,ckw) in enumerate(configs):
        st,res=one_epoch(parent,tr,va,clr,cew,ckw,SEED+gen*100+ci); res.update({'generation':gen,'candidate':ci}); cand.append((res['score'],st,res)); print('CAND',json.dumps(res),flush=True)
    cand.sort(key=lambda x:x[0]); _,parent,best=cand[0]; lr,ew,kw=best['lr'],best['edge_w'],best['kl_w']; lineage.append(best); print('WINNER',json.dumps(best),flush=True)

mf=VAE(); mf.load_state_dict(parent); final=evaluate(mf,va); params=sum(p.numel() for p in mf.parameters())
sd16={k:(v.half() if torch.is_floating_point(v) else v) for k,v in parent.items()}
torch.save({'state_dict':sd16,'architecture':'unlabeled_conv_vae_recursive','resolution':SIZE,'latent':LATENT,'parameters':params,'identity_labels_used':False,'generations':GENERATIONS,'final_hparams':{'lr':lr,'edge_w':ew,'kl_w':kw}},OUT/'lfw_generic_face_vae_recursive_g3.pt')

mf.eval(); g=torch.Generator().manual_seed(SEED+555)
with torch.no_grad(): s=mf.decode(torch.randn(36,LATENT,generator=g)).clamp(-1,1)
can=Image.new('RGB',(SIZE*6,SIZE*6),'white')
for i in range(36):
    a=((s[i]+1)*127.5).byte().permute(1,2,0).numpy(); can.paste(Image.fromarray(a),(i%6*SIZE,i//6*SIZE))
can.save(OUT/'generated_samples_recursive.png')

batch=next(iter(va))[:12]
with torch.no_grad(): recon,_,_=mf(batch)
can=Image.new('RGB',(SIZE*12,SIZE*2),'white')
for i in range(12):
    a=((batch[i]+1)*127.5).clamp(0,255).byte().permute(1,2,0).numpy(); b=((recon[i]+1)*127.5).clamp(0,255).byte().permute(1,2,0).numpy(); can.paste(Image.fromarray(a),(i*SIZE,0)); can.paste(Image.fromarray(b),(i*SIZE,SIZE))
can.save(OUT/'reconstruction_recursive.png')
summary={'source':'LFW unlabeled','identity_labels_used':False,'images_used':MAX_IMAGES,'resolution':SIZE,'parameters':params,'recursive_generations':GENERATIONS,'parent_validation':base_eval,'final_validation':final,'lineage':lineage,'training_seconds':round(time.time()-t0,2),'raw_photos_in_artifact':False}
(OUT/'summary_recursive.json').write_text(json.dumps(summary,indent=2)); print('FINAL',json.dumps(summary),flush=True)
