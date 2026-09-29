import argparse
import json
import math
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2
import gdown
import numpy as np
import requests
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from gdown.download_folder import _GoogleDriveFile, _parse_embedded_folder_view

ROOT_ID = "14V2QCmqjrMXgasbnuZ0NpnLzWU2621fC"  # official KoIn50 Drive root
TOTAL_CLASSES = 50
CLASSES_PER_SHARD = 5
TARGET_PER_CLASS = 8
MAX_CANDIDATES_PER_CLASS = 16
MIN_PER_CLASS = 5
SIZE = 64
LATENT = 96
EMBED = 32
EPOCHS = 24
BATCH = 20
BASE_SEED = 20260929
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"})
    return s


def list_folder(sess, fid):
    r = _parse_embedded_folder_view(sess=sess, folder_id=fid, verify=True, timeout=40)
    if r is None:
        raise RuntimeError(f"Cannot list Drive folder {fid}")
    return r


def discover_all_train_classes():
    s = make_session()
    _, root = list_folder(s, ROOT_ID)
    train_id = None
    for child in root:
        cid, name, typ = child[:3]
        if typ == _GoogleDriveFile.TYPE_FOLDER and name.strip().lower() == "train":
            train_id = cid
            break
    if not train_id:
        raise RuntimeError("KoIn50/train not found")
    _, children = list_folder(s, train_id)
    rows = []
    for child in children:
        cid, name, typ = child[:3]
        if typ == _GoogleDriveFile.TYPE_FOLDER and name.isdigit():
            rows.append((name.zfill(4), cid))
    rows.sort()
    if len(rows) < TOTAL_CLASSES:
        raise RuntimeError(f"Expected {TOTAL_CLASSES} classes, found {len(rows)}")
    return rows[:TOTAL_CLASSES]


def ahash(rgb, side=12):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sm = cv2.resize(g, (side, side), interpolation=cv2.INTER_AREA)
    return (sm > sm.mean()).reshape(-1)


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def face_crop(rgb):
    H, W = rgb.shape[:2]
    scale = min(1.0, 1000.0 / max(H, W))
    det = cv2.resize(rgb, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    gray = cv2.cvtColor(det, cv2.COLOR_RGB2GRAY)
    faces = CASCADE.detectMultiScale(gray, scaleFactor=1.06, minNeighbors=3, minSize=(18,18))
    if len(faces) == 0:
        return None
    x,y,w,h = max(faces, key=lambda b: int(b[2])*int(b[3]))
    if scale < 1:
        x,y,w,h = [int(round(v/scale)) for v in (x,y,w,h)]
    side = int(max(w,h)*2.0)
    cx,cy=x+w//2,y+h//2
    x1,y1=cx-side//2,cy-side//2
    x2,y2=x1+side,y1+side
    pl,pt=max(0,-x1),max(0,-y1)
    pr,pb=max(0,x2-W),max(0,y2-H)
    if any((pl,pt,pr,pb)):
        rgb=cv2.copyMakeBorder(rgb,pt,pb,pl,pr,cv2.BORDER_REFLECT_101)
        x1+=pl;x2+=pl;y1+=pt;y2+=pt
    crop=rgb[y1:y2,x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop,(SIZE,SIZE),interpolation=cv2.INTER_AREA)


class DS(Dataset):
    def __init__(self, items, aug=False):
        self.items=items; self.aug=aug
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        p,y=self.items[i]
        a=np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/127.5-1.0
        x=torch.from_numpy(a).permute(2,0,1)
        if self.aug and torch.rand(())<0.5: x=torch.flip(x,dims=[2])
        return x,torch.tensor(y,dtype=torch.long)


class CVAE(nn.Module):
    def __init__(self,nc):
        super().__init__(); self.nc=nc
        self.enc=nn.Sequential(
            nn.Conv2d(3,32,4,2,1),nn.GroupNorm(4,32),nn.SiLU(),
            nn.Conv2d(32,64,4,2,1),nn.GroupNorm(8,64),nn.SiLU(),
            nn.Conv2d(64,128,4,2,1),nn.GroupNorm(8,128),nn.SiLU(),
            nn.Conv2d(128,192,4,2,1),nn.GroupNorm(12,192),nn.SiLU())
        flat=192*4*4
        self.mu=nn.Linear(flat,LATENT); self.lv=nn.Linear(flat,LATENT)
        self.id_head=nn.Sequential(nn.LayerNorm(LATENT),nn.Linear(LATENT,nc))
        self.emb=nn.Embedding(nc,EMBED); self.fc=nn.Linear(LATENT+EMBED,flat)
        self.dec=nn.Sequential(
            nn.ConvTranspose2d(192,128,4,2,1),nn.GroupNorm(8,128),nn.SiLU(),
            nn.ConvTranspose2d(128,64,4,2,1),nn.GroupNorm(8,64),nn.SiLU(),
            nn.ConvTranspose2d(64,32,4,2,1),nn.GroupNorm(4,32),nn.SiLU(),
            nn.ConvTranspose2d(32,3,4,2,1),nn.Tanh())
    def encode(self,x):
        h=self.enc(x).flatten(1); return self.mu(h),self.lv(h)
    def decode(self,z,y):
        h=torch.cat([z,self.emb(y)],1)
        return self.dec(self.fc(h).view(-1,192,4,4))
    def forward(self,x,y):
        mu,lv=self.encode(x); z=mu+torch.randn_like(mu)*torch.exp(.5*lv)
        return self.decode(z,y),mu,lv,self.id_head(mu)


def evaluate(model,dl):
    model.eval(); s=defaultdict(float); n=0
    with torch.no_grad():
        for x,y in dl:
            r,mu,lv,lg=model(x,y)
            rec=F.l1_loss(r,x); kl=-.5*torch.mean(1+lv-mu.pow(2)-lv.exp()); ce=F.cross_entropy(lg,y)
            b=x.size(0); n+=b
            s['rec']+=float(rec)*b;s['kl']+=float(kl)*b;s['ce']+=float(ce)*b
            s['acc']+=float((lg.argmax(1)==y).float().sum())
    return {k:s[k]/n for k in ('rec','kl','ce','acc')}


def collect(shard, selected, raw, tmp):
    sess=make_session(); counts={}; failures=defaultdict(int)
    for local_y,(cls,fid_folder) in enumerate(selected):
        outd=raw/cls;outd.mkdir(parents=True,exist_ok=True)
        _,children=list_folder(sess,fid_folder)
        files=[]
        for ch in children:
            fid,name,typ=ch[:3]
            if typ!=_GoogleDriveFile.TYPE_FOLDER and Path(name).suffix.lower() in IMG_EXT:
                files.append((fid,name))
        rng=random.Random(BASE_SEED+shard*100003+local_y*1009);rng.shuffle(files)
        accepted=0; seen=[]
        for j,(fid,name) in enumerate(files[:MAX_CANDIDATES_PER_CLASS]):
            if accepted>=TARGET_PER_CLASS: break
            ext=Path(name).suffix.lower() or '.jpg'; fp=tmp/f"{cls}_{j}{ext}"
            try:
                r=gdown.download(id=fid,output=str(fp),quiet=True,use_cookies=False,timeout=35,retries=1)
                if not r or not fp.exists(): failures['download']+=1;continue
                with Image.open(fp) as im: rgb=np.asarray(im.convert('RGB'))
                crop=face_crop(rgb)
                if crop is None: failures['no_face']+=1;continue
                ph=ahash(crop)
                if any(hamming(ph,h)<=7 for h in seen): failures['duplicate']+=1;continue
                seen.append(ph); Image.fromarray(crop).save(outd/f"{accepted:02d}.jpg",quality=92);accepted+=1
            except Exception as e:
                failures['exception']+=1
                print(f"shard {shard} skip {cls} {name}: {type(e).__name__}",flush=True)
            finally:
                fp.unlink(missing_ok=True)
        counts[cls]=accepted
        print(f"SHARD {shard} class {cls}: {accepted}/{TARGET_PER_CLASS} accepted from {len(files)}",flush=True)
        if accepted<MIN_PER_CLASS: raise RuntimeError(f"shard {shard} class {cls}: only {accepted} usable")
    return counts,dict(failures)


def split(selected,raw,seed):
    tr=[];va=[]
    for y,(cls,_) in enumerate(selected):
        fs=sorted((raw/cls).glob('*.jpg'));random.Random(seed+y*71).shuffle(fs)
        nv=max(1,round(len(fs)*.25));va += [(p,y) for p in fs[:nv]];tr += [(p,y) for p in fs[nv:]]
    random.Random(seed).shuffle(tr);return tr,va


def train(shard,selected,raw,out,counts,failures):
    seed=BASE_SEED+shard;torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    tr,va=split(selected,raw,seed);td=DS(tr,True);vd=DS(va,False)
    tl=DataLoader(td,batch_size=min(BATCH,len(td)),shuffle=True,num_workers=0);vl=DataLoader(vd,batch_size=min(BATCH,len(vd)),shuffle=False,num_workers=0)
    model=CVAE(len(selected));opt=torch.optim.AdamW(model.parameters(),lr=1.3e-3,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=1.5e-4)
    beta=.0005;iw=.45;log=[];best=None;bs=-1e9;t0=time.time()
    for ep in range(1,EPOCHS+1):
        model.train(); s=defaultdict(float);n=0
        for x,y in tl:
            r,mu,lv,lg=model(x,y);rec=F.l1_loss(r,x);kl=-.5*torch.mean(1+lv-mu.pow(2)-lv.exp());ce=F.cross_entropy(lg,y)
            loss=rec+beta*kl+iw*ce;opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            b=x.size(0);n+=b;s['loss']+=float(loss.detach())*b;s['rec']+=float(rec.detach())*b;s['acc']+=float((lg.argmax(1)==y).float().sum())
        sch.step();vm=evaluate(model,vl);score=vm['acc']-.2*vm['rec']
        row={'epoch':ep,'train_loss':s['loss']/n,'train_rec':s['rec']/n,'train_acc':s['acc']/n,'val':vm,'score':score};log.append(row);print(json.dumps(row),flush=True)
        if score>bs:bs=score;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best);vm=evaluate(model,vl);elapsed=time.time()-t0
    ids=[c for c,_ in selected]
    torch.save({'state_dict':model.state_dict(),'global_class_ids':ids,'shard':shard,'resolution':SIZE,'latent':LATENT,'embedding':EMBED,'val':vm},out/f"koin50_shard_{shard:02d}.pt")
    (out/f"training_log_{shard:02d}.json").write_text(json.dumps(log,indent=2),encoding='utf-8')
    model.eval();
    with torch.no_grad():
        y=torch.arange(len(ids));g=torch.Generator().manual_seed(seed+999);z=torch.randn(len(ids),LATENT,generator=g);samples=model.decode(z,y).clamp(-1,1)
    canvas=Image.new('RGB',(SIZE*len(ids),SIZE+18),'white');dr=ImageDraw.Draw(canvas)
    for i,cls in enumerate(ids):
        a=((samples[i].cpu()+1)*127.5).byte().permute(1,2,0).numpy();canvas.paste(Image.fromarray(a),(i*SIZE,0));dr.text((i*SIZE+2,SIZE+2),cls,fill='black')
    canvas.save(out/f"samples_shard_{shard:02d}.png")
    summary={'shard':shard,'global_class_ids':ids,'accepted_images':sum(counts.values()),'per_class_counts':counts,'failures':failures,'train_images':len(td),'val_images':len(vd),'parameters':sum(p.numel() for p in model.parameters()),'epochs':EPOCHS,'training_seconds':round(elapsed,3),'val_identity_accuracy':vm['acc'],'chance_identity_accuracy':1/len(ids),'val_reconstruction_l1':vm['rec'],'raw_photos_uploaded':False,'usage_scope':'KoIn official README: academic purposes'}
    (out/f"summary_shard_{shard:02d}.json").write_text(json.dumps(summary,indent=2),encoding='utf-8');print('FINAL',json.dumps(summary),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);a=ap.parse_args();shard=a.shard
    if not 0<=shard<10: raise SystemExit('shard must be 0..9')
    allc=discover_all_train_classes();selected=allc[shard*CLASSES_PER_SHARD:(shard+1)*CLASSES_PER_SHARD]
    out=Path(f'artifacts_koin50_shard_{shard:02d}');raw=Path(f'_crops_{shard:02d}');tmp=Path(f'_tmp_{shard:02d}');out.mkdir(exist_ok=True);raw.mkdir(exist_ok=True);tmp.mkdir(exist_ok=True)
    try:
        counts,failures=collect(shard,selected,raw,tmp);train(shard,selected,raw,out,counts,failures)
    finally:
        shutil.rmtree(raw,ignore_errors=True);shutil.rmtree(tmp,ignore_errors=True)

if __name__=='__main__': main()
