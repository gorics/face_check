import json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.datasets import fetch_lfw_people
from torch.utils.data import Dataset, DataLoader

SEED=20260929
SIZE=128
LATENT=160
MAX_IMAGES=6000
EPOCHS=6
BATCH=48
OUT=Path('lfw_generic_v5_out')
OUT.mkdir(exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1,min(4,torch.get_num_threads())))

class FaceDS(Dataset):
    def __init__(self, imgs, idx, aug=False):
        self.imgs=imgs; self.idx=np.asarray(idx); self.aug=aug
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        a=self.imgs[int(self.idx[i])]
        h,w,_=a.shape
        side=min(h,w); y=(h-side)//2; x=(w-side)//2
        a=a[y:y+side,x:x+side]
        t=torch.from_numpy(a.copy()).permute(2,0,1).float()
        t=F.interpolate(t.unsqueeze(0),size=(SIZE,SIZE),mode='bilinear',align_corners=False).squeeze(0)
        t=t*2-1
        if self.aug and torch.rand(())<0.5: t=torch.flip(t,[2])
        if self.aug and torch.rand(())<0.25:
            gain=0.94+torch.rand(())*0.12
            t=(t*gain).clamp(-1,1)
        return t

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.Sequential(
            nn.Conv2d(3,32,4,2,1),nn.GroupNorm(4,32),nn.SiLU(),
            nn.Conv2d(32,64,4,2,1),nn.GroupNorm(8,64),nn.SiLU(),
            nn.Conv2d(64,128,4,2,1),nn.GroupNorm(8,128),nn.SiLU(),
            nn.Conv2d(128,192,4,2,1),nn.GroupNorm(12,192),nn.SiLU(),
            nn.Conv2d(192,256,4,2,1),nn.GroupNorm(16,256),nn.SiLU())
        flat=256*4*4
        self.mu=nn.Linear(flat,LATENT); self.lv=nn.Linear(flat,LATENT)
        self.fc=nn.Linear(LATENT,flat)
        self.dec=nn.Sequential(
            nn.ConvTranspose2d(256,192,4,2,1),nn.GroupNorm(12,192),nn.SiLU(),
            nn.ConvTranspose2d(192,128,4,2,1),nn.GroupNorm(8,128),nn.SiLU(),
            nn.ConvTranspose2d(128,64,4,2,1),nn.GroupNorm(8,64),nn.SiLU(),
            nn.ConvTranspose2d(64,32,4,2,1),nn.GroupNorm(4,32),nn.SiLU(),
            nn.ConvTranspose2d(32,3,4,2,1),nn.Tanh())
    def encode(self,x):
        h=self.enc(x).flatten(1); return self.mu(h),self.lv(h)
    def decode(self,z): return self.dec(self.fc(z).view(-1,256,4,4))
    def forward(self,x):
        mu,lv=self.encode(x); z=mu+torch.randn_like(mu)*torch.exp(.5*lv); return self.decode(z),mu,lv

def edge_loss(a,b):
    dx1=a[:,:,:,1:]-a[:,:,:,:-1]; dx2=b[:,:,:,1:]-b[:,:,:,:-1]
    dy1=a[:,:,1:,:]-a[:,:,:-1,:]; dy2=b[:,:,1:,:]-b[:,:,:-1,:]
    return .5*(F.l1_loss(dx1,dx2)+F.l1_loss(dy1,dy2))

def evaluate(m,dl):
    m.eval(); n=0; rec=ed=kl=0.0
    with torch.no_grad():
        for x in dl:
            r,mu,lv=m(x); b=x.size(0)
            rr=F.l1_loss(r,x); ee=edge_loss(r,x); kk=-.5*torch.mean(1+lv-mu.pow(2)-lv.exp())
            n+=b; rec+=float(rr)*b; ed+=float(ee)*b; kl+=float(kk)*b
    return {'reconstruction_l1':rec/n,'edge_l1':ed/n,'kl':kl/n}

print('Downloading/loading LFW...',flush=True)
lfw=fetch_lfw_people(color=True, resize=1.0, download_if_missing=True)
imgs=np.asarray(lfw.images,dtype=np.float32)
print(json.dumps({'raw_shape':list(imgs.shape),'raw_min':float(imgs.min()),'raw_max':float(imgs.max())}),flush=True)
# Deliberately discard names/identity labels: this is an unlabeled generic face generator.
rng=np.random.default_rng(SEED)
all_idx=rng.permutation(len(imgs))[:min(MAX_IMAGES,len(imgs))]
nv=max(400,int(len(all_idx)*0.1)); va_idx=all_idx[:nv]; tr_idx=all_idx[nv:]
tr=DataLoader(FaceDS(imgs,tr_idx,True),batch_size=BATCH,shuffle=True,num_workers=0)
va=DataLoader(FaceDS(imgs,va_idx,False),batch_size=BATCH,shuffle=False,num_workers=0)

m=VAE(); params=sum(p.numel() for p in m.parameters())
opt=torch.optim.AdamW(m.parameters(),lr=9e-4,weight_decay=1e-4)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=1e-4)
best=None; best_score=1e9; log=[]; t0=time.time()
for ep in range(1,EPOCHS+1):
    m.train(); n=0; sl=sr=se=sk=0.0
    for x in tr:
        r,mu,lv=m(x)
        rec=F.l1_loss(r,x); ed=edge_loss(r,x); kl=-.5*torch.mean(1+lv-mu.pow(2)-lv.exp())
        loss=rec+0.16*ed+0.00035*kl
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        b=x.size(0); n+=b; sl+=float(loss.detach())*b; sr+=float(rec.detach())*b; se+=float(ed.detach())*b; sk+=float(kl.detach())*b
    sch.step(); v=evaluate(m,va); score=v['reconstruction_l1']+0.16*v['edge_l1']
    row={'epoch':ep,'train_loss':sl/n,'train_reconstruction_l1':sr/n,'train_edge_l1':se/n,'train_kl':sk/n,'val':v,'score':score}
    log.append(row); print(json.dumps(row),flush=True)
    if score<best_score:
        best_score=score; best={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}

m.load_state_dict(best); final=evaluate(m,va)
# fp16 storage only; load back into float model for inference.
sd16={k:(v.half() if torch.is_floating_point(v) else v) for k,v in m.state_dict().items()}
torch.save({'state_dict':sd16,'architecture':'unlabeled_conv_vae','resolution':SIZE,'latent':LATENT,'parameters':params,'identity_labels_used':False},OUT/'lfw_generic_face_vae_v5.pt')

# random samples
m.eval(); g=torch.Generator().manual_seed(SEED+99)
with torch.no_grad(): samples=m.decode(torch.randn(36,LATENT,generator=g)).clamp(-1,1)
cell=SIZE; canvas=Image.new('RGB',(cell*6,cell*6),'white')
for i in range(36):
    a=((samples[i]+1)*127.5).byte().permute(1,2,0).numpy(); canvas.paste(Image.fromarray(a),(i%6*cell,i//6*cell))
canvas.save(OUT/'generated_samples.png')

# reconstruction grid
batch=next(iter(va))[:12]
with torch.no_grad(): recon,_,_=m(batch)
canvas=Image.new('RGB',(cell*12,cell*2),'white')
for i in range(12):
    a=((batch[i]+1)*127.5).clamp(0,255).byte().permute(1,2,0).numpy(); b=((recon[i]+1)*127.5).clamp(0,255).byte().permute(1,2,0).numpy()
    canvas.paste(Image.fromarray(a),(i*cell,0)); canvas.paste(Image.fromarray(b),(i*cell,cell))
canvas.save(OUT/'reconstruction_grid.png')

summary={'source':'LFW via scikit-learn','identity_labels_used':False,'images_used':int(len(all_idx)),'train_images':int(len(tr_idx)),'validation_images':int(len(va_idx)),'resolution':SIZE,'latent':LATENT,'parameters':params,'epochs':EPOCHS,'training_seconds':round(time.time()-t0,2),'final_validation':final,'best_score':best_score,'raw_photos_in_artifact':False}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)); (OUT/'training_log.json').write_text(json.dumps(log,indent=2))
print('FINAL',json.dumps(summary),flush=True)
