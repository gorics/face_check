import json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.datasets import fetch_lfw_people
from torch.utils.data import Dataset, DataLoader

SEED=20260929; SIZE=128; LATENT=160; MAX_IMAGES=6000; TRAIN_SUBSET=1200; VAL_N=600; BATCH=48
MAX_GENERATIONS=12; PATIENCE=3; MIN_IMPROVEMENT=0.00012
OUT=Path('lfw_generic_once_turbo_out'); OUT.mkdir(exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(max(1,min(4,torch.get_num_threads())))

class FaceDS(Dataset):
    def __init__(self, imgs, idx, aug=False): self.imgs=imgs; self.idx=np.asarray(idx); self.aug=aug
    def __len__(self): return len(self.idx)
    def __getitem__(self,i):
        a=self.imgs[int(self.idx[i])]; h,w,_=a.shape; s=min(h,w); y=(h-s)//2; x=(w-s)//2; a=a[y:y+s,x:x+s]
        t=torch.from_numpy(a.copy()).permute(2,0,1).float(); t=F.interpolate(t[None],(SIZE,SIZE),mode='bilinear',align_corners=False)[0]*2-1
        if self.aug and torch.rand(())<.5: t=torch.flip(t,[2])
        if self.aug and torch.rand(())<.25: t=(t*(.94+torch.rand(())*.12)).clamp(-1,1)
        return t

class VAE(nn.Module):
    def __init__(self):
        super().__init__(); self.enc=nn.Sequential(nn.Conv2d(3,32,4,2,1),nn.GroupNorm(4,32),nn.SiLU(),nn.Conv2d(32,64,4,2,1),nn.GroupNorm(8,64),nn.SiLU(),nn.Conv2d(64,128,4,2,1),nn.GroupNorm(8,128),nn.SiLU(),nn.Conv2d(128,192,4,2,1),nn.GroupNorm(12,192),nn.SiLU(),nn.Conv2d(192,256,4,2,1),nn.GroupNorm(16,256),nn.SiLU())
        flat=256*4*4; self.mu=nn.Linear(flat,LATENT); self.lv=nn.Linear(flat,LATENT); self.fc=nn.Linear(LATENT,flat)
        self.dec=nn.Sequential(nn.ConvTranspose2d(256,192,4,2,1),nn.GroupNorm(12,192),nn.SiLU(),nn.ConvTranspose2d(192,128,4,2,1),nn.GroupNorm(8,128),nn.SiLU(),nn.ConvTranspose2d(128,64,4,2,1),nn.GroupNorm(8,64),nn.SiLU(),nn.ConvTranspose2d(64,32,4,2,1),nn.GroupNorm(4,32),nn.SiLU(),nn.ConvTranspose2d(32,3,4,2,1),nn.Tanh())
    def encode(self,x): h=self.enc(x).flatten(1); return self.mu(h),self.lv(h)
    def decode(self,z): return self.dec(self.fc(z).view(-1,256,4,4))
    def forward(self,x): mu,lv=self.encode(x); z=mu+torch.randn_like(mu)*torch.exp(.5*lv); return self.decode(z),mu,lv

def edge(a,b):
    return .5*(F.l1_loss(a[:,:,:,1:]-a[:,:,:,:-1],b[:,:,:,1:]-b[:,:,:,:-1])+F.l1_loss(a[:,:,1:,:]-a[:,:,:-1,:],b[:,:,1:,:]-b[:,:,:-1,:]))

def evaluate(m,dl):
    m.eval(); n=0; rec=ed=kl=0.
    with torch.no_grad():
        for x in dl:
            r,mu,lv=m(x); b=x.size(0); rr=F.l1_loss(r,x); ee=edge(r,x); kk=-.5*torch.mean(1+lv-mu.pow(2)-lv.exp()); n+=b; rec+=float(rr)*b; ed+=float(ee)*b; kl+=float(kk)*b
    return {'reconstruction_l1':rec/n,'edge_l1':ed/n,'kl':kl/n}
def metric(v): return v['reconstruction_l1']+.16*v['edge_l1']

def train_candidate(base_state,imgs,pool_idx,va,lr,ew,kw,seed):
    rng=np.random.default_rng(seed); sub=rng.choice(pool_idx,size=min(TRAIN_SUBSET,len(pool_idx)),replace=False)
    tr=DataLoader(FaceDS(imgs,sub,True),batch_size=BATCH,shuffle=True,num_workers=0)
    torch.manual_seed(seed); m=VAE(); m.load_state_dict(base_state); opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4); n=0; sl=0.
    m.train()
    for x in tr:
        r,mu,lv=m(x); rec=F.l1_loss(r,x); ed=edge(r,x); kl=-.5*torch.mean(1+lv-mu.pow(2)-lv.exp()); loss=rec+ew*ed+kw*kl
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); n+=x.size(0); sl+=float(loss.detach())*x.size(0)
    v=evaluate(m,va); return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()},{'lr':lr,'edge_w':ew,'kl_w':kw,'train_subset':len(sub),'train_loss':sl/n,'val':v,'score':metric(v)}

print('Loading LFW...',flush=True); lfw=fetch_lfw_people(color=True,resize=1.0,download_if_missing=True); imgs=np.asarray(lfw.images,dtype=np.float32)
rng=np.random.default_rng(SEED); idx=rng.permutation(len(imgs))[:MAX_IMAGES]; va_idx=idx[:VAL_N]; pool_idx=idx[VAL_N:]
va=DataLoader(FaceDS(imgs,va_idx),batch_size=BATCH,num_workers=0)
hits=list(Path('parent_artifact').rglob('lfw_generic_face_vae_v5.pt'))
if not hits: raise SystemExit('parent checkpoint missing')
ck=torch.load(hits[0],map_location='cpu'); m0=VAE(); state={k:(v.float() if torch.is_floating_point(v) else v) for k,v in ck['state_dict'].items()}; m0.load_state_dict(state)
parent={k:v.detach().cpu().clone() for k,v in m0.state_dict().items()}; base_eval=evaluate(m0,va); parent_score=metric(base_eval)
lineage=[{'generation':0,'source':'v5 parent','val':base_eval,'score':parent_score,'accepted':True}]; print('PARENT',json.dumps(lineage[-1]),flush=True)
lr=.00045; ew=.16; kw=.00035; stale=0; attempted=0; t0=time.time()
for gen in range(1,MAX_GENERATIONS+1):
    configs=[(lr*.72,min(.24,ew*1.10),max(.00008,kw*.80)),(lr,max(.08,ew*.92),min(.0008,kw*1.12)),(lr*.86,ew,max(.00008,kw*.62))]
    cand=[]
    for ci,(clr,cew,ckw) in enumerate(configs):
        attempted+=1; st,res=train_candidate(parent,imgs,pool_idx,va,clr,cew,ckw,SEED+gen*100+ci); res.update({'generation':gen,'candidate':ci}); cand.append((res['score'],st,res)); print('CAND',json.dumps(res),flush=True)
    cand.sort(key=lambda x:x[0]); score,st,best=cand[0]; imp=parent_score-score; accepted=imp>=MIN_IMPROVEMENT; best['improvement']=imp; best['accepted']=accepted
    if accepted: parent=st; parent_score=score; lr,ew,kw=best['lr'],best['edge_w'],best['kl_w']; stale=0
    else: stale+=1; lr=max(8e-5,lr*.72); kw=max(.00008,kw*.8)
    lineage.append(best); print('GEN_RESULT',json.dumps(best),flush=True)
    if stale>=PATIENCE: print('CONVERGED',json.dumps({'generation':gen,'stale':stale}),flush=True); break

mf=VAE(); mf.load_state_dict(parent); final=evaluate(mf,va); params=sum(p.numel() for p in mf.parameters()); accepted=sum(1 for x in lineage[1:] if x.get('accepted'))
sd16={k:(v.half() if torch.is_floating_point(v) else v) for k,v in parent.items()}; torch.save({'state_dict':sd16,'architecture':'unlabeled_conv_vae_recursive_turbo','resolution':SIZE,'latent':LATENT,'parameters':params,'identity_labels_used':False,'accepted_generations':accepted},OUT/'lfw_generic_face_vae_recursive_turbo.pt')
mf.eval(); g=torch.Generator().manual_seed(SEED+888)
with torch.no_grad(): s=mf.decode(torch.randn(36,LATENT,generator=g)).clamp(-1,1)
can=Image.new('RGB',(SIZE*6,SIZE*6),'white')
for i in range(36): a=((s[i]+1)*127.5).byte().permute(1,2,0).numpy(); can.paste(Image.fromarray(a),(i%6*SIZE,i//6*SIZE))
can.save(OUT/'generated_samples_recursive_turbo.png')
summary={'source':'LFW unlabeled','identity_labels_used':False,'images_pool':MAX_IMAGES,'train_subset_per_candidate':TRAIN_SUBSET,'validation_images':VAL_N,'resolution':SIZE,'parameters':params,'max_generations':MAX_GENERATIONS,'generations_executed':len(lineage)-1,'accepted_generations':accepted,'candidates_trained':attempted,'parent_validation':base_eval,'parent_score':metric(base_eval),'final_validation':final,'final_score':metric(final),'absolute_score_improvement':metric(base_eval)-metric(final),'relative_score_improvement_pct':100*(metric(base_eval)-metric(final))/metric(base_eval),'lineage':lineage,'training_seconds':round(time.time()-t0,2),'raw_photos_in_artifact':False}
(OUT/'summary_recursive_turbo.json').write_text(json.dumps(summary,indent=2)); print('FINAL',json.dumps(summary),flush=True)
